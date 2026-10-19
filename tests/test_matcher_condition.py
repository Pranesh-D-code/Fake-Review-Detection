"""Product-condition guard: refurbished / pre-owned / open-box never verify against a new source (offline)."""
import pytest

from app.matcher import score_candidate
from app.product_identity import detect_condition, parse_identity, strip_condition_words

IPHONE_NEW = "Apple iPhone 15 (128 GB) - Blue"


def verdict(source_title, candidate_title, brand=None, url=None, source_price=None, candidate_price=None):
    return score_candidate(parse_identity(source_title, brand), candidate_title, url, source_price, candidate_price)


@pytest.mark.parametrize("title, expected", [
    ("Apple iPhone 15 (Renewed)", "refurbished"),
    ("Renewed Apple iPhone 15 128GB", "refurbished"),
    ("Apple iPhone 15 128GB | Amazon Renewed", "refurbished"),
    ("Samsung Galaxy M34 5G Refurbished (6GB, 128GB)", "refurbished"),
    ("LG 1.5 Ton 5 Star Split AC (Certified Refurbished)", "refurbished"),
    ("Dell Inspiron 15 Laptop - Reconditioned", "refurbished"),
    ("Apple iPhone 12 128GB Pre-Owned", "used"),
    ("OnePlus 9 Pro 256GB Preowned", "used"),
    ("Sony PlayStation 5 Console - Used - Like New", "used"),
    ("iPhone 13 Mini 128GB second hand", "used"),
    ("Samsung Galaxy S23 Like New 256GB", "used"),
    ("Apple iPhone 15 128GB Open Box", "open_box"),
    ("Sony WH-1000XM5 Headphones (Open-Box)", "open_box"),
    # new listings, including wording that only looks like a condition
    ("Apple iPhone 15 (128 GB) - Blue", "new"),
    ("Samsung Galaxy M34 5G Brand New Sealed", "new"),
    ("Samsung Galaxy M34 5G - not refurbished, genuine", "new"),
    ("Cetaphil Face Wash used for oily skin 250 ml", "new"),
    ("Olay Renewed Formula Night Serum 50 g", "new"),
    ("Unused sealed pack toothbrush", "new"),
])
def test_condition_detection(title, expected):
    assert parse_identity(title).condition == expected == detect_condition(title)


def test_condition_words_are_not_part_of_the_product_name():
    renewed = parse_identity("Apple iPhone 15 (Renewed)", "Apple")
    assert renewed.head_tokens == ["apple", "iphone", "15"] and renewed.bare_numbers == {"15"}
    assert "renewed" not in strip_condition_words("Apple iPhone 15 (Renewed)")
    assert renewed.title == "Apple iPhone 15 (Renewed)"             # the stored title is unchanged


@pytest.mark.parametrize("candidate", [
    "Apple iPhone 15 (128 GB) - Blue (Renewed)",
    "Apple iPhone 15 128GB Blue Refurbished - Grade A",
    "Apple iPhone 15 128GB Blue Pre-Owned",
    "Apple iPhone 15 128GB Blue Open Box",
    "Apple iPhone 15 128GB Blue Used - Like New",
])
def test_non_new_candidate_never_verifies_against_a_new_source(candidate):
    result = verdict(IPHONE_NEW, candidate, "Apple", source_price=64999, candidate_price=52000)
    assert result.status == "rejected", result
    assert any("condition differs" in f for f in result.flags)
    assert any("different condition" in f for f in result.flags)


def test_new_candidate_never_verifies_against_a_renewed_source():
    source = "Apple iPhone 15 (128 GB) - Blue (Renewed)"
    result = verdict(source, "Apple iPhone 15 (128 GB) - Blue", "Apple")
    assert result.status == "rejected"
    assert any("source is refurbished, candidate is new" in f for f in result.flags)


def test_same_condition_still_verifies():
    source = "Apple iPhone 15 (128 GB) - Blue (Renewed)"
    result = verdict(source, "Apple iPhone 15 128GB Blue Refurbished", "Apple")
    assert result.status == "verified", result
    assert "condition agrees (refurbished)" in result.reasons
    assert verdict(IPHONE_NEW, "Apple iPhone 15 128GB Blue", "Apple").status == "verified"


def test_other_second_hand_grade_is_not_verified_but_stays_listed():
    source = "Apple iPhone 15 (128 GB) - Blue (Renewed)"
    result = verdict(source, "Apple iPhone 15 128GB Blue Pre-Owned", "Apple")
    assert result.status == "candidate", result
    assert any("different condition" in f for f in result.flags)


def test_condition_in_the_url_slug_is_seen():
    url = "https://www.cashify.in/buy-refurbished-apple-iphone-15-128gb-blue/p/1"
    result = verdict(IPHONE_NEW, "Apple iPhone 15 128GB Blue", "Apple", url=url)
    assert result.status == "rejected"


def test_condition_check_leaves_unrelated_titles_alone():
    # same inputs as the pre-existing matcher tests, with "used"/"renewed" wording only in feature text
    source = "Cetaphil Gentle Skin Cleanser 250 ml"
    result = verdict(source, "Cetaphil Gentle Skin Cleanser 250ml - used for dry sensitive skin", "Cetaphil")
    assert result.status == "verified", result


def test_result_reports_the_candidates_condition():
    refurbished_source = "Apple iPhone 15 (128 GB) - Blue (Renewed)"
    same = verdict(refurbished_source, "Apple iPhone 15 128GB Blue Refurbished", "Apple")
    assert same.status == "verified" and same.condition == "refurbished" and "refurbished/used listing" in same.flags
    new = verdict(IPHONE_NEW, "Apple iPhone 15 128GB Blue", "Apple")
    assert new.condition == "new" and "refurbished/used listing" not in new.flags
    assert verdict(IPHONE_NEW, "Apple iPhone 15 128GB Blue Open Box", "Apple").condition == "open_box"
