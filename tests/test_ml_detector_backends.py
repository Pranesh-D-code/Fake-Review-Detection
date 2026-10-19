"""Backend selection for the text model (REVIEW_MODEL_BACKEND).

The fallback tests need no transformer. The transformer tests are skipped unless
models/review_transformer/ has been trained (scripts/train_transformer_classifier.py)
and torch + transformers are installed.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from app import ml_detector

ROOT = Path(__file__).resolve().parent.parent
GENERIC = ("I love this product. It is very good and works very well. The quality is great and the price is also "
           "very good. I would recommend this product to anyone who is looking for a good product.")
SPECIFIC = ("Delivery was late by 3 days, packaging was torn, but the item itself is fine. Sound is a bit tinny at "
            "high volume though bass is okay for the price. Returned the first piece because left side was dead.")
NEEDS_TEXT = ["Too little evidence to classify this review reliably"]

has_tfidf = pytest.mark.skipif(not ml_detector.available_backends()["tfidf"], reason="train the TF-IDF model first")
has_transformer = pytest.mark.skipif(
    not ml_detector.available_backends()["transformer"]
    or importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="train the transformer first (scripts/train_transformer_classifier.py; pip install -r requirements-transformer.txt)",
)


@pytest.fixture
def select_backend(monkeypatch):
    def select(name: str, **env: object) -> None:
        monkeypatch.setenv("REVIEW_MODEL_BACKEND", name)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        ml_detector.reset()
    yield select
    ml_detector.reset()


# ----------------------------------------------------------- fallbacks (no transformer needed)

@has_tfidf
def test_missing_transformer_falls_back_to_tfidf(select_backend, tmp_path):
    select_backend("transformer", REVIEW_TRANSFORMER_PATH=tmp_path / "absent")
    assert ml_detector.is_available() and ml_detector.metadata()["backend"] == "tfidf"


@has_tfidf
def test_tfidf_backend_can_be_forced(select_backend):
    select_backend("tfidf")
    assert ml_detector.metadata()["backend"] == "tfidf"


def test_without_any_model_everything_falls_back_to_rules(select_backend, tmp_path):
    select_backend("auto", REVIEW_TRANSFORMER_PATH=tmp_path / "absent", REVIEW_MODEL_PATH=tmp_path / "absent.joblib")
    assert not ml_detector.is_available()
    assert ml_detector.predict([GENERIC]) == [None]
    result = ml_detector.assess("needs-review", 0.35, "medium", NEEDS_TEXT, GENERIC, None)
    assert result.detector == "rules" and result.classification == "needs-review"


def test_unknown_backend_value_behaves_like_auto(select_backend, tmp_path):
    select_backend("not-a-backend", REVIEW_TRANSFORMER_PATH=tmp_path / "absent", REVIEW_MODEL_PATH=tmp_path / "absent.joblib")
    assert not ml_detector.is_available()


# ------------------------------------------------------------------ transformer backend

@has_transformer
def test_transformer_is_selected_explicitly_and_by_auto(select_backend):
    for name in ("transformer", "auto"):
        select_backend(name)
        meta = ml_detector.metadata()
        assert meta["backend"] == "transformer" and meta["thresholds"]["suspicious_at"] > meta["thresholds"]["genuine_below"]


@has_transformer
def test_transformer_scores_generic_text_above_specific_text(select_backend):
    select_backend("transformer")
    generic, specific = ml_detector.predict([GENERIC, SPECIFIC])
    assert 0.0 <= specific < generic <= 1.0


@has_transformer
def test_transformer_keeps_the_short_text_and_script_guards(select_backend):
    select_backend("transformer")
    hindi = "बहुत अच्छा प्रोडक्ट है मुझे बहुत पसंद आया और डिलीवरी भी समय पर हुई थी धन्यवाद"
    assert ml_detector.predict(["Nice product", hindi]) == [None, None]


@has_transformer
def test_transformer_assess_uses_its_own_thresholds(select_backend):
    select_backend("transformer")
    thresholds = ml_detector.metadata()["thresholds"]
    above = ml_detector.assess("needs-review", 0.35, "medium", NEEDS_TEXT, GENERIC, min(0.999, thresholds["suspicious_at"] + 0.001))
    assert above.classification == "suspicious" and above.detector == "ml+rules"
    below = ml_detector.assess("needs-review", 0.35, "medium", NEEDS_TEXT, SPECIFIC, max(0.0, thresholds["genuine_below"] - 0.001))
    assert below.classification == "genuine-looking"
    flagged_by_rules = ml_detector.assess("suspicious", 0.8, "high", ["Near-duplicate wording"], SPECIFIC, 0.0)
    assert flagged_by_rules.classification == "suspicious"


@has_transformer
def test_transformer_has_no_word_level_explanation(select_backend):
    select_backend("transformer")
    assert ml_detector.explain(GENERIC) == []


@has_transformer
def test_served_probabilities_match_the_ones_recorded_at_training_time(select_backend):
    """Guards against tokenisation/truncation/calibration drift between training and serving."""
    db_path = ROOT / "trustengine.db"
    saved_path = ml_detector._transformer_path() / "test_predictions.npz"
    if not db_path.is_file() or not saved_path.is_file():
        pytest.skip("needs trustengine.db and the training-time test predictions")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import train_review_classifier as tfidf
    finally:
        sys.path.pop(0)
    select_backend("transformer")
    meta = ml_detector.metadata()
    _, _, texts, _, _ = tfidf.load_dataset(db_path, meta["dataset"])
    saved = np.load(saved_path)
    temperature = float(meta["temperature"])
    rows = [(i, z) for i, z in zip(saved["test_idx"], saved["z"]) if ml_detector.applicable(texts[i])[0]][:24]
    assert len(rows) >= 12
    served = ml_detector.predict([texts[i] for i, _ in rows])
    expected = [float(expit(z / temperature)) for _, z in rows]
    assert np.allclose(served, expected, atol=5e-3)
