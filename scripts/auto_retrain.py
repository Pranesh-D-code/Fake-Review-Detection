"""Retrain the fake-review model with the labels people gave in the app, safely.

Run by the app in the background (see ``app/review_labels.py``) or by hand:

    python scripts/auto_retrain.py

It trains a *candidate* model into ``models/_candidate`` with
``scripts/train_review_classifier.py`` (dataset + person-labelled reviews), then compares
it with the model in use. The candidate replaces the live model only if its F1 on the
held-out *dataset* rows has not dropped by more than ``--tolerance`` (default 0.015),
so a few noisy labels cannot quietly make the detector worse. The previous model is
kept as ``review_detector.prev.joblib``. Progress and the outcome go to
``models/retrain_status.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CANDIDATE = MODELS / "_candidate"
STATUS = MODELS / "retrain_status.json"
FILES = ("review_detector.joblib", "review_detector.metrics.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(**fields) -> None:
    try:
        current = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {}
    current.update(fields)
    MODELS.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(current, indent=2), encoding="utf-8")


def dataset_f1(metrics: dict) -> float | None:
    """F1 on held-out dataset rows only (falls back to the overall test F1 for models trained before labels existed)."""
    block = metrics.get("metrics", {})
    return (block.get("test_dataset_only") or block.get("test_at_0.5") or {}).get("f1_fake")


def decide(candidate: dict, current: dict | None, tolerance: float) -> tuple[bool, str]:
    """(promote?, why) - pure so it can be tested without training anything."""
    new_f1 = dataset_f1(candidate)
    if new_f1 is None:
        return False, "The candidate has no held-out F1, so it was not used."
    if current is None:
        return True, f"No model was in use; the candidate scored F1 {new_f1:.4f} on held-out dataset rows."
    old_f1 = dataset_f1(current)
    if old_f1 is None:
        return True, f"The model in use has no held-out F1 to compare; the candidate scored {new_f1:.4f}."
    if new_f1 + tolerance < old_f1:
        return False, (f"Kept the current model: the candidate's F1 on held-out dataset rows fell from "
                       f"{old_f1:.4f} to {new_f1:.4f} (allowed drop {tolerance}).")
    return True, f"The candidate's F1 on held-out dataset rows is {new_f1:.4f} (was {old_f1:.4f})."


def promote() -> None:
    """Move the candidate files into place, keeping the previous model as *.prev.*"""
    for name in FILES:
        live, new = MODELS / name, CANDIDATE / name
        if live.exists():
            backup = MODELS / name.replace("review_detector.", "review_detector.prev.", 1)
            shutil.copy2(live, backup)
        os.replace(new, live)               # atomic on the same drive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance", type=float, default=0.015)
    parser.add_argument("--trainer-args", nargs=argparse.REMAINDER, default=[],
                        help="extra arguments passed to train_review_classifier.py (put last)")
    args = parser.parse_args()

    write_status(state="running", started_at=now(), finished_at=None, promoted=False,
                 message="Training a candidate model on the dataset plus your labels.")
    if CANDIDATE.exists():
        shutil.rmtree(CANDIDATE)

    kwargs = {"cwd": str(ROOT), "capture_output": True, "encoding": "utf-8", "errors": "replace"}
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_review_classifier.py"),
                           "--out-dir", str(CANDIDATE), *args.trainer_args], **kwargs)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        write_status(state="failed", finished_at=now(), message="Training failed: " + " | ".join(tail)[:400])
        return 1

    try:
        candidate = json.loads((CANDIDATE / FILES[1]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        write_status(state="failed", finished_at=now(), message="Training finished but wrote no metrics.")
        return 1
    try:
        current = json.loads((MODELS / FILES[1]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = None

    ok, why = decide(candidate, current, args.tolerance)
    if ok:
        promote()
    live_test = (candidate.get("metrics") or {}).get("live_test") or {}
    write_status(
        state="done" if ok else "rejected", finished_at=now(), promoted=ok, message=why,
        dataset_test_f1=dataset_f1(candidate), previous_dataset_test_f1=dataset_f1(current) if current else None,
        live_labels_total=candidate.get("live_labels_total"), live_labels_in_training=candidate.get("live_labels_in_training"),
        live_test=live_test,
    )
    shutil.rmtree(CANDIDATE, ignore_errors=True)
    print(why)
    return 0


if __name__ == "__main__":
    sys.exit(main())
