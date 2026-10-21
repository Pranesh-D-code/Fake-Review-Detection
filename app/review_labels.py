"""Human labels for scraped reviews, and the automatic retraining they feed.

The review list shows each review with the verdict the detector gave it. A person
presses **Real** or **Fake** on a review; that choice is stored in ``review_labels``
(the table the project reserved for this) and becomes training data for the text
model (``scripts/train_review_classifier.py`` reads it alongside the dataset).

Why a person and not the detector's own verdict: a model that is trained on its own
guesses learns nothing new and slowly locks in its mistakes. The label has to come
from outside the model.

After ``AUTO_RETRAIN_MIN_NEW_LABELS`` (default 50) new labels, ``scripts/auto_retrain.py``
runs in the background. It trains a candidate model, checks it has not got worse on the
held-out dataset rows, and only then replaces the live one (keeping the previous model).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STATUS_PATH = MODELS_DIR / "retrain_status.json"
MAX_TEXT_CHARS = 5000
MIN_LABELS_TO_TRAIN = 20            # total labels before any live-label retraining
MIN_PER_CLASS = 5                   # and at least this many real and this many fake
RUNNING_STALE_MINUTES = 90          # a "running" state older than this is treated as dead

_lock = threading.Lock()
_thread: threading.Thread | None = None

LABELS = {"real": "real", "fake": "fake"}


def review_hash(text: str) -> str:
    """Same hash the review store uses (engines._review_hash), so labels join to stored reviews."""
    normalised = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    from . import engines                # imported late: engines imports modules that import this one
    return engines._db()


# ----------------------------------------------------------------------------- saving

def save_label(text: str, label: str, rating: int | None = None, product_url: str | None = None,
               platform: str | None = None, annotator: str = "user") -> dict[str, Any]:
    """Store (or change, or with label='clear' remove) the human label for one review."""
    text = (text or "").strip()
    if not text:
        raise ValueError("The review text is empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"The review is longer than {MAX_TEXT_CHARS} characters.")
    label = (label or "").strip().casefold()
    if label != "clear" and label not in LABELS:
        raise ValueError("label must be 'real', 'fake' or 'clear'.")
    digest = review_hash(text)
    now = _now()

    db = _connect()
    try:
        if label == "clear":
            db.execute("DELETE FROM review_labels WHERE review_hash = ?", (digest,))
        else:
            # The training script reads the text from review_observations; make sure it is there
            # even when review storage is switched off or the review came from somewhere else.
            known = db.execute("SELECT 1 FROM review_observations WHERE review_hash = ? LIMIT 1", (digest,)).fetchone()
            if not known:
                db.execute("""
                    INSERT OR IGNORE INTO review_observations (
                        product_url, platform, sku, review_hash, review_text, rating, review_timestamp,
                        verified_purchase, collection_source, first_seen_at, last_seen_at,
                        baseline_label, baseline_confidence, baseline_signals)
                    VALUES (?, ?, NULL, ?, ?, ?, NULL, 0, 'user_label', ?, ?, 'needs-review', 0.0, '[]')
                """, (product_url or "user://labelled", platform or "User labelled", digest, text,
                      rating if rating in (1, 2, 3, 4, 5) else None, now, now))
            db.execute("""
                INSERT INTO review_labels (review_hash, human_label, annotator, note, labeled_at)
                VALUES (?, ?, ?, NULL, ?)
                ON CONFLICT(review_hash) DO UPDATE SET
                    human_label = excluded.human_label, annotator = excluded.annotator, labeled_at = excluded.labeled_at
            """, (digest, LABELS[label], annotator, now))
        db.commit()
    finally:
        db.close()
    return {"review_hash": digest, "label": None if label == "clear" else LABELS[label]}


def lookup(hashes: list[str]) -> dict[str, str]:
    """{review_hash: 'real'|'fake'} for the hashes that already carry a human label."""
    wanted = [h for h in dict.fromkeys(hashes) if h]
    if not wanted:
        return {}
    db = _connect()
    try:
        found: dict[str, str] = {}
        for start in range(0, len(wanted), 500):          # SQLite variable limit
            chunk = wanted[start:start + 500]
            marks = ",".join("?" * len(chunk))
            found.update(db.execute(
                f"SELECT review_hash, human_label FROM review_labels WHERE review_hash IN ({marks})", chunk).fetchall())
        return found
    finally:
        db.close()


# ----------------------------------------------------------------------------- status

def label_counts() -> dict[str, int]:
    db = _connect()
    try:
        rows = db.execute("SELECT human_label, COUNT(*) FROM review_labels GROUP BY 1").fetchall()
    finally:
        db.close()
    counts = {"real": 0, "fake": 0}
    for name, number in rows:
        if name in counts:
            counts[name] = number
    counts["total"] = counts["real"] + counts["fake"]
    return counts


def _read_status() -> dict[str, Any]:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_status(**fields: Any) -> dict[str, Any]:
    status = {**_read_status(), **fields}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def _is_running(status: dict[str, Any]) -> bool:
    if status.get("state") != "running":
        return False
    try:
        started = datetime.fromisoformat(status["started_at"])
    except (KeyError, ValueError):
        return False
    return (datetime.now(timezone.utc) - started).total_seconds() < RUNNING_STALE_MINUTES * 60


def threshold() -> int:
    try:
        return max(5, int(os.getenv("AUTO_RETRAIN_MIN_NEW_LABELS", "50")))
    except ValueError:
        return 50


def auto_enabled() -> bool:
    return os.getenv("AUTO_RETRAIN", "true").strip().casefold() not in {"0", "false", "no", "off"}


def status() -> dict[str, Any]:
    """Everything the UI shows about labelling and retraining."""
    from . import ml_detector
    counts = label_counts()
    meta = ml_detector.metadata() or {}
    used = int(meta.get("live_labels_total", 0) or 0)
    saved = _read_status()
    running = _is_running(saved)
    return {
        "labels": counts,
        "labels_used_by_current_model": used,
        "new_labels_since_training": max(0, counts["total"] - used),
        "retrain_at_new_labels": threshold(),
        "auto_retrain": auto_enabled(),
        "can_train": _enough_labels(counts),
        "running": running,
        "last_run": {k: saved.get(k) for k in ("state", "started_at", "finished_at", "message", "dataset_test_f1",
                                               "previous_dataset_test_f1", "live_labels_total", "promoted")} if saved else None,
        "model_trained_at": meta.get("created_at"),
    }


def _enough_labels(counts: dict[str, int]) -> bool:
    return counts["total"] >= MIN_LABELS_TO_TRAIN and counts["real"] >= MIN_PER_CLASS and counts["fake"] >= MIN_PER_CLASS


# ----------------------------------------------------------------------------- retraining

def _spawn_trainer() -> int:
    """Run scripts/auto_retrain.py to completion (blocking); returns its exit code."""
    from .proc import hidden_kwargs
    kwargs: dict[str, Any] = {"cwd": str(BASE_DIR), "capture_output": True, "encoding": "utf-8", "errors": "replace"}
    kwargs.update(hidden_kwargs())
    proc = subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "auto_retrain.py")], **kwargs)
    return proc.returncode


def _run_in_background() -> None:
    from . import ml_detector
    try:
        code = _spawn_trainer()
        if code != 0:
            _write_status(state="failed", finished_at=_now(), message=f"The trainer exited with code {code}.")
    except Exception as exc:                                  # never lose the app over a training problem
        _write_status(state="failed", finished_at=_now(), message=f"Could not run the trainer ({type(exc).__name__}).")
    finally:
        ml_detector.reset()                                   # pick up a promoted model without a restart


def maybe_start_retrain(force: bool = False) -> dict[str, Any]:
    """Start a background retrain when enough new labels exist. Returns the current status."""
    global _thread
    current = status()
    if current["running"] or not current["can_train"]:
        return current
    if not force and not (current["auto_retrain"] and current["new_labels_since_training"] >= current["retrain_at_new_labels"]):
        return current
    with _lock:
        if _thread is not None and _thread.is_alive():
            return current
        _write_status(state="running", started_at=_now(), finished_at=None, message="Training a candidate model on the dataset plus your labels.")
        _thread = threading.Thread(target=_run_in_background, name="auto-retrain", daemon=True)
        _thread.start()
    return status()
