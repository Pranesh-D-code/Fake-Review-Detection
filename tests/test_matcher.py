"""Matcher behaviour on realistic cross-store titles (offline, no network)."""
import pytest

from app.matcher import score_candidate, url_slug_text
from app.product_identity import build_queries, extract_specs, parse_identity

M34 = ("Samsung Galaxy M34 5G (Waterfall Blue,6GB,128GB)|120Hz sAMOLED Display|50MP Triple No Shake Cam|"
       "6000 mAh Battery|12GB RAM with RAM Plus|Android 13")


def verdict(source_title, candidate_title, brand=None, url=None, source_price=None, candidate_price=None, image=None):
    return score_candidate(parse_identity(source_title, brand), candidate_title, url, source_price, candidate_price, image)


def test_same_phone_other_store_is_verified():
    for title in (
        "Samsung Galaxy M34 5G 128 GB, 6 GB RAM, Waterfall Blue, Mobile Phone",
        "SAMSUNG Galaxy M34 5G (6GB RAM, 128GB, Midnight Blue)",
    ):
        result = verdict(M34, title, "Samsung", source_price=16999, candidate_price=15999)
        assert result.status == "verified", (title, result)


def test_other_storage_size_is_not_verified():
    result = verdict(M34, "Samsung Galaxy M34 5G (8GB RAM, 256GB, Midnight Blue)", "Samsung")
    assert result.status != "verified"
    assert any("specification" in f for f in result.flags)


@pytest.mark.parametrize("title", [
    "QRIOH Visionary Cycle Printed Samsung Galaxy M34 5G Back Case Mobile Accessories",
    "Tempered Glass Screen Protector for Samsung Galaxy M34 5G",
    "80W Power Adapter SUPERVOOC Ultra Fast Charger for OnePlus 12",
])
def test_accessories_are_rejected(title):
    assert verdict(M34, title, "Samsung").status == "rejected"


@pytest.mark.parametrize("title", [
    "SAMSUNG Galaxy A37 5G (8GB RAM, 128GB, Awesome Charcoal)",
    "Samsung M36 5G 128 GB, 8 GB RAM, Black, Mobile Phone",
    "Samsung Galaxy Watch8 46mm BT (2025) 64GB, 445mAh, 1.3 Inch Display (Black)",
])
def test_sibling_models_are_not_verified(title):
    assert verdict(M34, title, "Samsung").status != "verified"


def test_variant_words_must_agree():
    source = "Apple iPhone 15 Pro Max (256 GB) - Natural Titanium"
    assert verdict(source, "Apple iPhone 15 Pro Max 256GB Natural Titanium", "Apple").status == "verified"
    assert verdict(source, "Apple iPhone 15 256GB Blue", "Apple").status != "verified"
    assert verdict(source, "Apple iPhone 14 Pro Max 256GB", "Apple").status != "verified"


def test_pack_size_and_weight_matter():
    source = "Aashirvaad Whole Wheat Atta 10 kg"
    assert verdict(source, "Aashirvaad Select Whole Wheat Atta 10kg Pack", "Aashirvaad").status == "verified"
    assert verdict(source, "Aashirvaad Whole Wheat Atta 5 kg", "Aashirvaad").status != "verified"


def test_gender_and_model_number_for_footwear():
    source = "Nike Revolution 6 Men's Road Running Shoes"
    assert verdict(source, "Nike Men Revolution 6 Next Nature Road Running Shoes", "Nike").status != "rejected"
    assert verdict(source, "Nike Revolution 7 Men's Road Running Shoes", "Nike").status != "verified"
    assert verdict(source, "Nike Women's Revolution 6 Road Running Shoes", "Nike").status != "verified"


def test_implausible_price_blocks_verification():
    result = verdict(M34, "Samsung Galaxy M34 5G 128 GB, 6 GB RAM", "Samsung", source_price=16999, candidate_price=499)
    assert result.status == "rejected"


def test_url_slug_supplies_full_name_for_truncated_titles():
    url = "https://www.croma.com/samsung-galaxy-m34-5g-6gb-ram-128gb-waterfall-blue-/p/300001"
    assert "m34" in url_slug_text(url)
    result = verdict(M34, "SAMSUNG Galaxy ...", "Samsung", url=url)
    assert result.status in {"verified", "candidate"}


def test_spec_extraction_distinguishes_network_from_grams():
    specs = extract_specs("Galaxy M34 5G 128GB 6.7 inch Whey 500 g 1.5 kg")
    assert 128.0 in specs["storage_gb"] and 6.7 in specs["inch"]
    assert 500.0 in specs["g"] and 1500.0 in specs["g"]
    assert 5.0 not in specs.get("g", set())


def test_queries_go_from_precise_to_broad():
    queries = build_queries(parse_identity(M34, "Samsung"))
    assert queries[0].startswith("samsung galaxy m34")
    assert "128gb" in queries[0]
    assert len(queries[-1].split()) <= len(queries[0].split())


def test_ram_and_storage_are_compared_separately():
    ident = parse_identity("Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB Storage)", "Samsung")
    assert ident.specs["ram_gb"] == {8.0} and ident.specs["storage_gb"] == {128.0}
    unlabeled = extract_specs("Waterfall Blue,6GB,128GB")
    assert unlabeled["ram_gb"] == {6.0} and unlabeled["storage_gb"] == {128.0}
    # same RAM, different storage must not look like an agreement
    source = "Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB Storage) | 6.7 inch Super AMOLED | 50MP OIS Triple Camera"
    assert verdict(source, "Samsung Galaxy M36 5G Smartphone 8 GB/256 GB", "Samsung").status != "verified"
    assert verdict(source, "Samsung M36 5G 128 GB, 8 GB RAM, Black, Mobile Phone", "Samsung",
                   source_price=23999, candidate_price=23999).status == "verified"


def test_a_different_model_code_is_rejected_not_listed():
    result = verdict(M34, "SAMSUNG Galaxy A07 5G (6GB RAM, 128GB, Light Violet)", "Samsung")
    assert result.status == "rejected"
    assert any("different model" in f for f in result.flags)


def test_marketing_words_in_the_tail_do_not_create_variants_or_types():
    source = parse_identity("Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB) | 6.7 inch Super AMOLED | Triple Camera | RAM Plus")
    assert source.variants == set() and source.types == set()


def test_url_slug_keeps_decimal_sizes():
    assert "6.7 inch" in url_slug_text("https://www.vijaysales.com/p/P1/2/samsung-galaxy-m36-6-7-inch-display")
    assert "1.5 ton" in url_slug_text("https://www.croma.com/lg-1-5-ton-3-star-split-ac-white-/p/1234")


def test_queries_drop_colours_and_screen_size_for_phones():
    queries = build_queries(parse_identity("Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB Storage) | 6.7 inch Super AMOLED", "Samsung"))
    assert queries[0] == "samsung galaxy m36 5g 128gb"
    assert "velvet" not in " ".join(queries) and "inch" not in queries[0]
