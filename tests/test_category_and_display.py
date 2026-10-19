import pytest

from app.matcher import score_candidate, text_similarity
from app.multistore import select_stores
from app.product_identity import parse_identity


@pytest.mark.parametrize("title,brand,expected", [
    ("adidas Unisex-Adult Anavarin Court Sneaker", "adidas", "fashion"),          # singular "sneaker"
    ("PUMA Catch Suede Sneakers", "PUMA", "fashion"),
    ("Samsung Galaxy M36 5G Mobile (Velvet Black, 6GB RAM, 128GB Storage)", "Samsung", "electronics"),
    ("Onida 139 cm (55 inches) Ultra HD (4K) Smart LED Google TV 55UFD", "Onida", "electronics"),
    ("Maybelline New York Fit Me Matte Liquid Foundation 128", "Maybelline", "beauty"),
    ("Nike Revolution 7", "Nike", "fashion"),                                      # only the brand says what it is
])
def test_category_is_found_from_singular_words_and_brands(title, brand, expected):
    assert parse_identity(title, brand).category == expected


def test_fashion_products_are_searched_on_fashion_stores_not_electronics_stores():
    stores = select_stores("Amazon", "https://www.amazon.in/dp/B0X", parse_identity("adidas Anavarin Court Sneaker", "adidas").category)
    assert {"Myntra", "Ajio"} <= set(stores) and not {"Croma", "Reliance Digital", "Vijay Sales"} & set(stores)


def test_text_match_shown_for_a_verified_listing_is_not_diluted_by_marketing_text():
    source = parse_identity("Samsung Galaxy M36 5G Mobile (Velvet Black, 6GB RAM, 128GB Storage) | Google Gemini | Gorilla Glass Victus+ | "
                            "7.7mm | AI Enhanced 50MP OIS Triple Camera | Nightography | Lag-free Gaming | Without Charger", "Samsung")
    result = score_candidate(source, "Samsung M36 5G 128 GB, 6 GB RAM, Black, Mobile Phone", None, 22999, 21999)
    assert result.status == "verified"
    assert result.title_similarity >= 0.7                      # was ~0.4 when it compared against the headline only
    assert text_similarity("Samsung Galaxy M36", "Samsung Galaxy M36") == 1.0


def test_a_phone_title_that_says_without_charger_is_not_an_accessory():
    for title in (
        "Samsung Galaxy M36 5G Mobile (Velvet Black, 6GB RAM, 128GB Storage) | Google Gemini | Without Charger",
        "Samsung Galaxy M36 5G (6GB, 128GB) Without Charger",
        "Samsung Galaxy M36 5G (6GB, 128GB) with Charger",
    ):
        assert not parse_identity(title, "Samsung").is_accessory, title
    for title in ("Samsung 25W Super Fast Charger Type-C", "QRIOH Printed Samsung Galaxy M36 5G Back Case"):
        assert parse_identity(title, "Samsung").is_accessory, title
