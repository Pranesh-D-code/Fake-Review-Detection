"""Exact product vs similar product: what each listing is called and how a similar one is described (offline)."""
import pytest

from app.matcher import UNCONFIRMED_NOTE, score_candidate
from app.product_identity import parse_identity

PHONE = "Samsung Galaxy M36 5G (Orange Haze, 8GB RAM, 128GB Storage)"
SHOES = "Nike Revolution 7 Men Running Shoes"


def judge(source, candidate, **kwargs):
    return score_candidate(parse_identity(source, None), candidate, "https://store.example/p/1", **kwargs)


def test_the_same_product_is_exact_with_nothing_to_explain():
    result = judge(PHONE, "Samsung Galaxy M36 5G 8GB 128GB Velvet Black")
    assert result.relation == "exact" and result.status == "verified" and result.differences == []


@pytest.mark.parametrize("candidate, expected", [
    ("Samsung Galaxy M36 5G 8GB 256GB", "storage 256 GB instead of 128 GB"),
    ("Samsung Galaxy M35 5G 8GB 128GB", "different model code (m35)"),
])
def test_a_sibling_phone_is_similar_and_says_how(candidate, expected):
    result = judge(PHONE, candidate)
    assert result.relation == "similar" and result.status != "verified"
    assert expected in result.differences


def test_pro_plus_is_similar_to_pro_and_not_exact():
    result = judge("Xiaomi Redmi Note 14 Pro 5G 8GB 256GB", "Xiaomi Redmi Note 14 Pro+ 5G 8GB 256GB")
    assert result.relation == "similar" and result.status == "rejected"
    assert any("Plus" in d for d in result.differences)


def test_the_neighbouring_shoe_model_is_similar():
    result = judge(SHOES, "Nike Revolution 6 Men Running Shoes")
    assert result.relation == "similar"
    assert "model 6 instead of 7" in result.differences
    assert result.closeness > 0.8


def test_another_model_in_the_same_range_names_what_changed():
    result = judge(SHOES, "Nike Downshifter 13 Men Running Shoes")
    assert result.relation == "similar"
    assert any("downshifter" in d for d in result.differences)


@pytest.mark.parametrize("source, candidate", [
    (PHONE, "Samsung Galaxy M36 Back Cover Case"),                       # an accessory is not a similar phone
    (PHONE, "Apple iPhone 15 128GB"),                                    # another make
    (PHONE, "Samsung 43 inch Smart TV"),                                 # another kind of product
    (SHOES, "Nike Revolution 7 Women Running Shoes"),                    # another gender
    (SHOES, "Adidas Duramo Men Running Shoes"),                          # another brand: not "similar text"
])
def test_things_that_are_something_else_are_not_listed_at_all(source, candidate):
    assert judge(source, candidate).relation == "none"


def test_an_unconfirmed_listing_is_similar_with_the_plain_reason():
    # same words but too little to be sure (no model code, thin title): possible, not verified
    result = judge("Dove Intense Repair Shampoo", "Dove Intense Repair Shampoo for dry hair")
    assert result.relation in {"exact", "similar"}
    if result.relation == "similar":
        assert result.differences and (UNCONFIRMED_NOTE in result.differences or result.differences)


def test_a_different_looking_photo_demotes_an_otherwise_exact_match_to_similar():
    result = judge(PHONE, "Samsung Galaxy M36 5G 8GB 128GB Velvet Black", image_similarity=0.2, visual_verdict="different")
    assert result.relation == "similar" and "the photo looks different" in result.differences


ULTRABOOST = "ADIDAS ULTRABOOST 5 Running Shoes For Men"


@pytest.mark.parametrize("candidate", [
    "Adidas Men's ULTRABOOST 5 Blue Running Shoes",              # another colour: still the same shoe
    "Adidas Ultra Boost 5 Men's Grey Running Shoes",             # "Ultra Boost" is only a spaced spelling, not an "Ultra" model
])
def test_the_same_shoe_in_other_wording_or_colour_is_exact(candidate):
    result = judge(ULTRABOOST, candidate)
    assert result.relation == "exact", (result.flags, result.differences)


@pytest.mark.parametrize("candidate, expected", [
    ("ADIDAS Men Ultraboost 5X Running Shoes", "5x instead of 5"),
    ("ADIDAS Men ULTRABOOST 5 AUDI REVOLUT F1 TEAM SHOES", "audi"),         # a special edition is its own product
    ("ADIDAS Ultraboost Light Running Shoes", "light"),
    ("ADIDAS Men Ultraboost 20 Running Shoes", "model 20 instead of 5"),
])
def test_a_neighbouring_or_special_edition_shoe_is_similar_not_exact(candidate, expected):
    result = judge(ULTRABOOST, candidate)
    assert result.relation == "similar" and result.status != "verified"
    assert any(expected in d for d in result.differences), result.differences


def test_description_words_after_the_name_are_not_a_different_edition():
    result = judge("Cetaphil Gentle Skin Cleanser 250 ml", "Cetaphil Gentle Skin Cleanser 250ml - used for dry sensitive skin")
    assert result.relation == "exact"


def test_a_plain_source_is_not_matched_to_its_5x_sibling_either_way():
    assert judge("ADIDAS Men Ultraboost 5X Running Shoes", ULTRABOOST).relation == "similar"


def test_a_store_that_keeps_the_brand_only_in_the_address_is_still_read_correctly():
    result = score_candidate(parse_identity(ULTRABOOST, None), "Men Running Shoes Ultraboost 5 - Brown",
                             "https://www.decathlon.in/p/1000005129/adidas-ultraboost-5-men-running-shoes-brown")
    assert result.relation == "exact"
