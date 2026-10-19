from datetime import datetime, timedelta, timezone

from app.models import Review
from app.nlp_engine import FAST_DUPLICATE_THRESHOLD, _near_duplicate_indices, _near_duplicate_indices_fast
from app.review_signals import parse_review_date, review_burst

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def test_marketplace_date_phrases_are_parsed():
    assert parse_review_date("Reviewed in India on 5 March 2024").date().isoformat() == "2024-03-05"
    assert parse_review_date("March 5, 2024").date().isoformat() == "2024-03-05"
    assert parse_review_date("2024-03-05T10:00:00Z").date().isoformat() == "2024-03-05"
    assert parse_review_date("yesterday") is None and parse_review_date(None) is None


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
