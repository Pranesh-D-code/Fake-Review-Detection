"""GTIN/EAN and model/MPN comparison in score_candidate and the multistore pass-through (offline, fixtures only)."""
import pytest

from app.matcher import normalise_gtin, normalise_mpn, score_candidate
from app.product_identity import parse_identity
from tests.test_matcher_stock import CROMA, SOURCE_TITLE, VIJAY, _deep, _run

M34 = ("Samsung Galaxy M34 5G (Waterfall Blue,6GB,128GB)|120Hz sAMOLED Display|50MP Triple No Shake Cam|"
       "6000 mAh Battery|12GB RAM with RAM Plus|Android 13")
FULL = "SAMSUNG Galaxy M34 5G (6GB RAM, 128GB, Midnight Blue)"
TRUNCATED = "SAMSUNG Galaxy ..."


def with_check_digit(body: str) -> str:
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return body + str((10 - total % 10) % 10)


EAN_A = with_check_digit("890123456789")          # 13 digits
EAN_B = with_check_digit("890123456780")


def score(candidate_title=FULL, source_title=M34, brand="Samsung", **kwargs):
    return score_candidate(parse_identity(source_title, brand), candidate_title, **kwargs)


# --- normalisation -----------------------------------------------------------------------------------

def test_gtin_normalisation_accepts_only_real_gtins():
    assert normalise_gtin(EAN_A) == "0" + EAN_A
    assert normalise_gtin("0" + EAN_A) == "0" + EAN_A                       # GTIN-14 form of the same EAN
    upc = with_check_digit("03600029145")                                    # 12-digit UPC
    assert normalise_gtin(upc) == normalise_gtin("0" + upc)                  # UPC-A == EAN-13 with a leading zero
    assert normalise_gtin(f" {EAN_A[:3]}-{EAN_A[3:]} ") == "0" + EAN_A       # separators are ignored
    assert normalise_gtin(int(EAN_A)) == "0" + EAN_A
    for bad in (None, "", "abc", "12345", "0000000000000", "1111111111111", EAN_A[:-1] + str((int(EAN_A[-1]) + 1) % 10), True):
        assert normalise_gtin(bad) is None, bad


def test_mpn_normalisation():
    assert normalise_mpn("SM-M346B/DS") == "SMM346BDS"
    assert normalise_mpn("MTP03HN/A") == "MTP03HNA"
    for bad in (None, "", "N/A", "abc", "12", "NONE", False):
        assert normalise_mpn(bad) is None, bad


# --- GTIN ------------------------------------------------------------------------------------------

def test_matching_gtin_makes_a_sparse_title_verifiable():
    assert score(TRUNCATED).status != "verified"
    result = score(TRUNCATED, source_gtin=EAN_A, candidate_gtin=EAN_A)
    assert result.status == "verified", result
    assert any("GTIN/EAN" in r and "matches" in r for r in result.reasons)


def test_gtin_forms_of_the_same_product_agree():
    result = score(TRUNCATED, source_gtin=EAN_A, candidate_gtin="0" + EAN_A)
    assert result.status == "verified"


def test_conflicting_gtin_is_a_veto_even_when_the_titles_agree():
    assert score(FULL).status == "verified"
    result = score(FULL, source_gtin=EAN_A, candidate_gtin=EAN_B)
    assert result.status == "rejected", result
    assert any("GTIN/EAN differs" in f for f in result.flags) and "different GTIN/EAN" in result.flags


def test_matching_gtin_does_not_override_an_explicit_contradiction():
    # same GTIN reported, but the listing is plainly an accessory: stay safe
    result = score("Tempered Glass Screen Protector for Samsung Galaxy M34 5G", source_gtin=EAN_A, candidate_gtin=EAN_A)
    assert result.status == "rejected"
    # and a bare "match" on a listing with a different brand and no shared name words is not trusted
    other = score("Mobile Phone 5G Blue", source_gtin=EAN_A, candidate_gtin=EAN_A)
    assert other.status == "rejected" and any("does not support" in f for f in other.flags)


@pytest.mark.parametrize("kwargs", [
    {},
    {"source_gtin": EAN_A},                                     # candidate page exposes none
    {"candidate_gtin": EAN_B},                                  # source scraper exposes none
    {"source_gtin": EAN_A, "candidate_gtin": "not-a-gtin"},     # unusable value counts as absent
    {"source_gtin": "1234567890123", "candidate_gtin": EAN_B},  # bad check digit counts as absent
    {"source_mpn": "SM-M346B"},
    {"candidate_mpn": "SM-M356B"},
])
def test_absent_or_unusable_identifiers_change_nothing(kwargs):
    for title in (FULL, TRUNCATED, "SAMSUNG Galaxy M34 5G (8GB RAM, 256GB, Midnight Blue)"):
        baseline = score(title)
        result = score(title, **kwargs)
        assert (result.status, result.score, result.reasons, result.flags) == (baseline.status, baseline.score, baseline.reasons, baseline.flags)


# --- model / MPN -----------------------------------------------------------------------------------

def test_equal_mpn_is_positive_evidence():
    base = score(TRUNCATED)
    result = score(TRUNCATED, source_mpn="SM-M346B", candidate_mpn="SM-M346B")
    assert result.score > base.score
    assert any("model/MPN SMM346B matches" in r for r in result.reasons)


def test_mpn_with_a_longer_colour_suffix_still_agrees():
    result = score(FULL, source_mpn="SM-M346B", candidate_mpn="SM-M346BZBGINU")
    assert result.status == "verified" and not any("differs" in f for f in result.flags)


def test_conflicting_mpn_is_a_veto():
    result = score(FULL, source_mpn="SM-M346B", candidate_mpn="SM-M356B")
    assert result.status == "rejected", result
    assert "different model/MPN" in result.flags
    short = score(FULL, source_mpn="SM-A52", candidate_mpn="SM-A525F")      # too short to be a "prefix" of a longer model
    assert short.status == "rejected"


def test_source_mpn_printed_in_the_candidate_title_is_positive_evidence():
    base = score("Samsung Galaxy 5G mobile SM-M346B/DS")
    result = score("Samsung Galaxy 5G mobile SM-M346B/DS", source_mpn="SM-M346B")
    assert result.score > base.score
    assert any("appears in candidate title" in r for r in result.reasons)
    # an MPN that is simply not in the title is no evidence either way
    assert score(FULL, source_mpn="SM-M346B").score == score(FULL).score


def test_gtin_match_does_not_hide_a_conflicting_mpn():
    result = score(FULL, source_gtin=EAN_A, candidate_gtin=EAN_A, source_mpn="SM-M346B", candidate_mpn="SM-M356B")
    assert result.status == "rejected"


# --- pass-through in app/multistore.py (stubbed node helper) ---------------------------------------------

def test_multistore_applies_source_identifiers_to_opened_pages(monkeypatch):
    details = [_deep(CROMA, price=21999.0, gtin=EAN_B, mpn="SM-M366B"), _deep(VIJAY, price=23499.0, gtin=EAN_A, mpn="SM-M366B")]
    matches, attempts, _ = _run(monkeypatch, [CROMA, VIJAY], details, source_gtin=EAN_A, source_mpn="SM-M366B")
    assert [m["platform"] for m in matches] == ["Vijay Sales"]              # Croma's GTIN conflicts: dropped
    vijay = matches[0]
    assert vijay["match_status"] == "verified" and vijay["gtin"] == EAN_A
    assert any("GTIN/EAN" in r for r in vijay["match_reasons"])
    croma_attempt = next(a for a in attempts if a["platform"] == "Croma")
    assert croma_attempt["result_count"] == 0 and croma_attempt["filtered_out"] == 1


def test_multistore_without_source_identifiers_behaves_as_before(monkeypatch):
    details = [_deep(CROMA, gtin=EAN_B), _deep(VIJAY, gtin=EAN_A)]
    matches, _, _ = _run(monkeypatch, [CROMA, VIJAY], details)             # no source ids given
    assert sorted(m["platform"] for m in matches) == ["Croma", "Vijay Sales"]
    assert all(not any("GTIN" in r for r in m["match_reasons"]) for m in matches)
