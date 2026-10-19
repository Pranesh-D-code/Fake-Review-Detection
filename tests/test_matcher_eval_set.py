"""Regression guard over the hand-written pairs in tests/fixtures/match_pairs.json (offline).

The pairs are invented, not measured on live listings: passing here means the matcher still behaves as
the rules intend on these examples, not that it reaches any accuracy on real store data.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("evaluate_matcher", ROOT / "scripts" / "evaluate_matcher.py")
evaluate_matcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_matcher)

PAIRS = evaluate_matcher.load_pairs()
REQUIRED_CATEGORIES = {"phones", "laptops", "tvs-acs", "footwear", "grocery", "beauty", "accessories"}


def test_fixture_is_large_and_covers_the_requested_categories():
    assert len(PAIRS) >= 60
    assert REQUIRED_CATEGORIES <= {p["category"] for p in PAIRS}
    assert len({p["id"] for p in PAIRS}) == len(PAIRS)
    assert {p["expected"] for p in PAIRS} == {"verified", "candidate", "rejected"}
    cases = " ".join(p["case"] for p in PAIRS)
    for needle in ("sibling model", "other storage size", "other pack size", "refurbished listing", "accessory listing"):
        assert needle in cases


@pytest.mark.parametrize("pair", [p for p in PAIRS if not p.get("known_gap")], ids=lambda p: p["id"])
def test_pair_gets_the_expected_status(pair):
    result = evaluate_matcher.judge(pair)
    assert result.status == pair["expected"], (pair["source_title"], pair["candidate_title"], result.score, result.flags)


def test_known_gaps_are_few_and_explained():
    gaps = [p for p in PAIRS if p.get("known_gap")]
    assert len(gaps) <= 10 and all(len(p["known_gap"]) > 20 for p in gaps)


def test_no_pair_outside_the_known_gaps_is_wrongly_verified():
    summary = evaluate_matcher.evaluate(PAIRS)
    unexpected = [r["pair"]["id"] for r in summary["false_verified"] if not r["pair"].get("known_gap")]
    assert unexpected == []
    assert evaluate_matcher.unexpected_misses(summary) == []


def test_report_says_the_pairs_are_hand_written_not_live(capsys):
    assert evaluate_matcher.main([]) == 0
    out = capsys.readouterr().out
    assert out.count("hand-written, not measured on live listings") == 2          # printed before and after the figures
    assert "verified precision" in out and "verified recall" in out
