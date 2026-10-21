"""Text-model behaviour. Skipped when models/review_detector.joblib has not been trained."""
import pytest

from app import ml_detector

pytestmark = pytest.mark.skipif(not ml_detector.is_available(), reason="train the model first (scripts/train_review_classifier.py)")

GENERIC = ("I love this product. It is very good and works very well. The quality is great and the price is also "
           "very good. I would recommend this product to anyone who is looking for a good product.")
SPECIFIC = ("got the green line after 2 days.replaced at service at free of cost but that is not the problem,it may come in future")      # a real Flipkart review


def test_generic_repetitive_review_scores_high_and_specific_review_low():
    generic, specific = ml_detector.predict([GENERIC, SPECIFIC])
    assert generic > 0.83 > 0.12 > specific


def test_model_declines_short_and_non_latin_text():
    assert ml_detector.predict(["Nice product", "बहुत अच्छा प्रोडक्ट है मुझे बहुत पसंद आया और डिलीवरी भी समय पर हुई थी धन्यवाद"]) == [None, None]


def test_assess_raises_to_suspicious_only_above_threshold():
    result = ml_detector.assess("needs-review", 0.35, "medium", ["Too little evidence to classify this review reliably"], GENERIC, 0.99)
    assert result.classification == "suspicious" and result.detector == "ml+rules"
    assert any("Text model" in s for s in result.signals)


def test_assess_never_downgrades_a_rule_based_flag():
    result = ml_detector.assess("suspicious", 0.8, "high", ["Near-duplicate wording found in another collected review"], SPECIFIC, 0.02)
    assert result.classification == "suspicious"


def test_assess_marks_genuine_looking_only_for_low_probability_without_signals():
    clean = ml_detector.assess("needs-review", 0.35, "medium", ["Too little evidence to classify this review reliably"], SPECIFIC, 0.01)
    assert clean.classification == "genuine-looking"
    middle = ml_detector.assess("needs-review", 0.35, "medium", ["Too little evidence to classify this review reliably"], SPECIFIC, 0.5)
    assert middle.classification == "needs-review"


def test_assess_falls_back_to_rules_without_a_probability():
    result = ml_detector.assess("needs-review", 0.35, "medium", ["x"], "short", None)
    assert result.detector == "rules" and result.classification == "needs-review"


def test_explanation_names_driving_phrases():
    assert ml_detector.explain(GENERIC)


def test_the_flag_threshold_never_drops_below_the_configured_floor(monkeypatch):
    base = ["Too little evidence to classify this review reliably"]
    monkeypatch.setenv("REVIEW_FLAG_MIN_PROBABILITY", "0.9")
    assert ml_detector.assess("needs-review", 0.35, "medium", base, GENERIC, 0.86).classification != "suspicious"
    assert ml_detector.assess("needs-review", 0.35, "medium", base, GENERIC, 0.93).classification == "suspicious"
    monkeypatch.setenv("REVIEW_FLAG_MIN_PROBABILITY", "0.5")
    assert ml_detector.assess("needs-review", 0.35, "medium", base, GENERIC, 0.86).classification == "suspicious"
