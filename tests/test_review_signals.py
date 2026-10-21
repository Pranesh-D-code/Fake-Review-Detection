from datetime import datetime, timedelta, timezone

from app.models import Review
from app.nlp_engine import FAST_DUPLICATE_THRESHOLD, _near_duplicate_indices, _near_duplicate_indices_fast
from app.review_signals import parse_review_date, review_burst

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def test_marketplace_date_phrases_are_parsed():
    assert parse_review_date("Reviewed in India on 5 March 2024").date().isoformat() == "2024-03-05"
    assert parse_review_date("March 5, 2024").date().isoformat() == "2024-03-05"
    assert parse_review_date("2024-03-05T10:00:00Z").date().isoformat() == "2024-03-05"
    assert parse_review_date("sometime last year") is None and parse_review_date(None) is None


def _reviews(spec):
    out = []
    for day, count in spec:
        out += [Review(text=f"review {day} {i}", rating=5, timestamp=NOW - timedelta(days=day)) for i in range(count)]
    return out


def test_single_day_spike_is_reported_as_a_burst():
    spec = [(d, 1) for d in range(10, 70, 6)] + [(30, 25)]
    summary = review_burst(_reviews(spec), now=NOW)
    assert summary and summary.peak_count >= 25 and summary.max_z_score >= 2
    assert any(p["count"] >= 25 for p in summary.burst_points)


def test_steady_posting_is_not_a_burst():
    summary = review_burst(_reviews([(d, 1) for d in range(1, 41)]), now=NOW)
    assert summary and summary.burst_points == []


def test_too_few_dated_reviews_gives_no_summary():
    assert review_burst(_reviews([(d, 1) for d in range(1, 8)]), now=NOW) is None


def test_fast_duplicate_scan_finds_what_the_exact_scan_finds():
    base = "this charger stopped working after two weeks and the seller refused to replace it at all"
    texts = [base, base + " really", "completely different review about the colour and the size of the shoes I bought",
             "another unrelated sentence talking about battery life and the camera quality of the phone"]
    reviews = [Review(text=t, rating=3) for t in texts]
    assert _near_duplicate_indices_fast([t.lower() for t in texts]) == _near_duplicate_indices(reviews) == {0, 1}
    assert FAST_DUPLICATE_THRESHOLD >= 100


def test_burst_is_not_computed_when_only_the_newest_reviews_carry_an_exact_date():
    """Flipkart gives exact days only for recent reviews ("3 days ago") and months for older ones; z-scores on that
    biased slice would invent a burst. Below 80% dated coverage there is no burst summary at all."""
    dated = _reviews([(d, 1) for d in range(1, 21)])
    undated = [Review(text=f"older review {i}", rating=4) for i in range(60)]
    assert review_burst(dated + undated, now=NOW) is None
    assert review_burst(dated + undated[:3], now=NOW) is not None


def test_every_review_gets_a_two_way_verdict():
    from app.nlp_engine import review_verdict
    assert review_verdict("suspicious", 0.9, 0.99) == ("not_genuine", "clear", None)
    assert review_verdict("suspicious", 0.5, None) == ("not_genuine", "limited", None)
    assert review_verdict("genuine-looking", 0.9, 0.01) == ("genuine", "clear", None)
    assert review_verdict("genuine-looking", 0.58, None) == ("genuine", "limited", None)
    assert review_verdict("needs-review", 0.45, 0.7) == ("not_genuine", "limited", "close call")      # leans AI-written, below the flag
    assert review_verdict("needs-review", 0.45, 0.2) == ("genuine", "limited", "close call")
    assert review_verdict("needs-review", 0.35, None) == ("genuine", "limited", "too short to judge")  # too short for the model, no warning sign
