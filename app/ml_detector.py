"""Inference side of the trained fake-review text classifier.

``scripts/train_review_classifier.py`` writes ``models/review_detector.joblib`` (TF-IDF +
logistic regression) and the optional ``scripts/train_transformer_classifier.py`` writes
``models/review_transformer/`` (a fine-tuned transformer). ``REVIEW_MODEL_BACKEND`` picks
which one answers (``auto`` | ``transformer`` | ``tfidf``); a missing or unloadable
transformer falls back to TF-IDF, and a missing TF-IDF model falls back to the rules.
This module loads the chosen model lazily and exposes:

* ``is_available()``             - whether a model file could be loaded
* ``predict(texts)``             - per-review likelihood of being computer-generated
* ``explain(text)``              - the words/phrases that pushed the score
* ``assess(...)``                - merge the model with the rule-based signals

The model estimates how much a review's *wording* resembles the computer-generated
reviews it was trained on. It is not proof of fakery, it was not trained on live
marketplace data, and it is ignored for very short or non-English text, where it
has no basis for a judgement. When no model file exists everything degrades to
the existing rule-based classifier.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "review_detector.joblib"
DEFAULT_TRANSFORMER_PATH = BASE_DIR / "models" / "review_transformer"
BACKENDS = ("auto", "transformer", "tfidf")
TRANSFORMER_BATCH = 32

log = logging.getLogger(__name__)
_lock = threading.Lock()
_infer_lock = threading.Lock()
_bundle: dict[str, Any] | None = None
_load_failed = False


def _model_path() -> Path:
    return Path(os.getenv("REVIEW_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def _transformer_path() -> Path:
    return Path(os.getenv("REVIEW_TRANSFORMER_PATH", str(DEFAULT_TRANSFORMER_PATH)))


def _enabled() -> bool:
    return os.getenv("ENABLE_ML_DETECTOR", "true").strip().casefold() not in {"0", "false", "no", "off"}


def _backend_choice() -> str:
    choice = os.getenv("REVIEW_MODEL_BACKEND", "auto").strip().casefold()
    return choice if choice in BACKENDS else "auto"


def available_backends() -> dict[str, bool]:
    """Which trained models exist on disk (cheap check; nothing is loaded)."""
    return {
        "tfidf": _model_path().is_file(),
        "transformer": (_transformer_path() / "meta.json").is_file() and (_transformer_path() / "model.safetensors").is_file(),
    }


def _load_tfidf() -> dict[str, Any] | None:
    path = _model_path()
    if not path.is_file():
        return None
    try:
        import joblib
        loaded = joblib.load(path)
        if not isinstance(loaded, dict) or "pipeline" not in loaded:
            raise ValueError("unexpected model bundle")
        return {"backend": "tfidf", **loaded}
    except Exception:
        return None


def _load_transformer() -> dict[str, Any] | None:
    if not available_backends()["transformer"]:
        return None
    path = _transformer_path()
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.eval()
        return {"backend": "transformer", "model": model, "tokenizer": tokenizer, "torch": torch, "meta": meta}
    except Exception as exc:  # torch/transformers not installed, or a damaged model folder
        log.warning("Transformer review model could not be loaded (%s); falling back.", exc)
        return None


def _load() -> dict[str, Any] | None:
    global _bundle, _load_failed
    if _bundle is not None or _load_failed or not _enabled():
        return _bundle
    with _lock:
        if _bundle is not None or _load_failed:
            return _bundle
        choice = _backend_choice()
        order = (_load_tfidf,) if choice == "tfidf" else (_load_transformer, _load_tfidf)
        for loader in order:
            _bundle = loader()
            if _bundle is not None:
                break
        if _bundle is None:
            _load_failed = True
        elif choice == "transformer" and _bundle["backend"] != "transformer":
            log.warning("REVIEW_MODEL_BACKEND=transformer but no usable transformer model; using TF-IDF.")
    return _bundle


def reset() -> None:
    """Forget the cached model (used by tests and after retraining)."""
    global _bundle, _load_failed
    with _lock:
        _bundle, _load_failed = None, False


def is_available() -> bool:
    return _load() is not None


def metadata() -> dict[str, Any] | None:
    bundle = _load()
    return {**bundle["meta"], "backend": bundle["backend"]} if bundle else None


def _latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isascii()) / len(letters)


def applicable(text: str) -> tuple[bool, str | None]:
    """The model only judges English-looking text with enough words."""
    bundle = _load()
    if not bundle:
        return False, "no trained model available"
    minimum = int(bundle["meta"].get("min_words_for_ml", 6))
    if len(text.split()) < minimum:
        return False, f"too short for the text model (needs {minimum}+ words)"
    if _latin_share(text) < 0.9:
        return False, "text is not in the Latin script the model was trained on"
    return True, None


def predict(texts: list[str]) -> list[float | None]:
    """Probability (0..1) that each text resembles the computer-generated class.

    ``None`` where the model should not judge the text (see ``applicable``).
    """
    bundle = _load()
    if not bundle or not texts:
        return [None] * len(texts)
    usable = [i for i, text in enumerate(texts) if applicable(text)[0]]
    out: list[float | None] = [None] * len(texts)
    if usable:
        selected = [texts[i] for i in usable]
        if bundle["backend"] == "transformer":
            probabilities = _predict_transformer(bundle, selected)
        else:
            probabilities = bundle["pipeline"].predict_proba(selected)[:, 1]
        for index, probability in zip(usable, probabilities):
            out[index] = round(float(probability), 4)
    return out


def _predict_transformer(bundle: dict[str, Any], texts: list[str]) -> list[float]:
    """Calibrated P(computer-generated); same tokenisation (strip, truncate) as in training."""
    torch, tokenizer, model = bundle["torch"], bundle["tokenizer"], bundle["model"]
    meta = bundle["meta"]
    max_length, temperature = int(meta["max_length"]), float(meta.get("temperature", 1.0))
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [0.0] * len(texts)
    with _infer_lock, torch.inference_mode():
        for start in range(0, len(order), TRANSFORMER_BATCH):
            chunk = order[start:start + TRANSFORMER_BATCH]
            encoded = tokenizer([texts[i].strip() for i in chunk], truncation=True, max_length=max_length,
                                padding=True, return_tensors="pt")
            logits = model(**encoded).logits
            margin = (logits[:, 1] - logits[:, 0]).double() / temperature
            for index, probability in zip(chunk, torch.sigmoid(margin).tolist()):
                out[index] = probability
    return out


def explain(text: str, top: int = 4) -> list[str]:
    """Words/phrases whose weight pushed the score toward computer-generated, strongest first."""
    bundle = _load()
    if not bundle or bundle["backend"] != "tfidf":
        return []        # the transformer has no per-word weights to report
    try:
        features = bundle["pipeline"].named_steps["features"]
        word_vectoriser = features.transformer_list[0][1]
        classifier = bundle["pipeline"].named_steps["clf"]
        vector = word_vectoriser.transform([text])
        names = word_vectoriser.get_feature_names_out()
        coefficients = classifier.coef_[0][: len(names)]
        contributions = vector.multiply(coefficients).tocoo()
        ranked = sorted(zip(contributions.col, contributions.data), key=lambda pair: pair[1], reverse=True)
        return [names[i] for i, weight in ranked[:top] if weight > 0]
    except Exception:
        return []


@dataclass
class Assessment:
    classification: str
    confidence: float
    risk: str
    signals: list[str]
    probability: float | None
    detector: str                      # "ml+rules" | "rules"


def assess(
    heuristic_classification: str,
    heuristic_confidence: float,
    heuristic_risk: str,
    heuristic_signals: list[str],
    text: str,
    probability: float | None,
) -> Assessment:
    """Merge the model probability with the rule-based result.

    * Rule-based "suspicious" is never downgraded: concrete signals (duplicate text,
      rating/sentiment mismatch, promotional phrasing) stand on their own.
    * The model alone can raise a review to "suspicious" only above the
      precision-calibrated threshold learned at training time, and can mark
      "genuine-looking" only below the matching low threshold.
    * Anything in between stays "needs-review".
    """
    bundle = _load()
    if probability is None or not bundle:
        return Assessment(heuristic_classification, heuristic_confidence, heuristic_risk,
                          heuristic_signals, None, "rules")

    thresholds = bundle["meta"]["thresholds"]
    suspicious_at, genuine_below = float(thresholds["suspicious_at"]), float(thresholds["genuine_below"])
    # Live reviews are shorter and written in other English than the training data, so the flag never goes below
    # REVIEW_FLAG_MIN_PROBABILITY (default 0.9) even if training chose a lower cut-off.
    try:
        suspicious_at = max(suspicious_at, float(os.getenv("REVIEW_FLAG_MIN_PROBABILITY", "0.9")))
    except ValueError:
        pass
    signals = list(heuristic_signals)
    percent = round(probability * 100)

    if probability >= suspicious_at:
        terms = explain(text)
        hint = f" (driven by: {', '.join(terms)})" if terms else ""
        signals.append(f"Text model: {percent}% resemblance to computer-generated reviews{hint}")
        confidence = max(heuristic_confidence if heuristic_classification == "suspicious" else 0.0,
                         min(0.95, round(probability, 2)))
        return Assessment("suspicious", confidence, "high" if probability >= 0.95 else "medium", signals, probability, "ml+rules")

    if heuristic_classification == "suspicious":
        signals.append(f"Text model: {percent}% resemblance to computer-generated reviews (below its flag threshold)")
        return Assessment("suspicious", heuristic_confidence, heuristic_risk, signals, probability, "ml+rules")

    if probability <= genuine_below and not [s for s in heuristic_signals if "No promotional" not in s and "Too little" not in s]:
        signals = [s for s in signals if "Too little evidence" not in s]
        signals.append(f"Text model: only {percent}% resemblance to computer-generated reviews")
        return Assessment("genuine-looking", min(0.9, round(1 - probability, 2)), "low", signals, probability, "ml+rules")

    signals = [s for s in signals if "Too little evidence" not in s]
    signals.append(f"Text model: {percent}% resemblance to computer-generated reviews (inconclusive)")
    return Assessment("needs-review", 0.45, "medium", signals, probability, "ml+rules")
