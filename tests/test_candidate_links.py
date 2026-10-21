"""candidate_links keeps legacy behaviour and accepts extended matches (offline)."""
from app import engines
from app.models import Product


def _product(matches, platform="Amazon"):
    product = Product(platform=platform, url="https://www.amazon.in/dp/B0TEST", title="Samsung Galaxy M36 5G 8GB 128GB",
                      image_url="https://img/src.jpg", category="electronics")
    product._p_data = {"competitor_matches": matches}
    return product


def _legacy(platform, title, price, url):
    return {"platform": platform, "title": title, "url": url, "price": price, "image_url": None, "reviews": []}


def _extended(platform, title, price, url, status, score):
    return {"platform": platform, "title": title, "url": url, "price": price, "image_url": None, "reviews": [],
            "match_source": "enhanced", "match_status": status, "match_score": score,
            "match_reasons": ["model code m36 matches"] if status == "verified" else [],
            "match_flags": [] if status == "verified" else ["different specification"], "title_similarity": 0.6,
            "differences": [] if status == "verified" else ["storage 256 GB instead of 128 GB"], "closeness": 0.9}


def test_legacy_rules_still_apply_to_legacy_matches(monkeypatch):
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: None)
    product = _product([
        _legacy("Croma", "Samsung Galaxy M36 5G", 20000, "https://www.croma.com/a/p/1"),            # legacy rule: skipped
        _legacy("Myntra", "Samsung Galaxy M36 5G", 20000, "https://www.myntra.com/a/1/buy"),       # not fashion: skipped
        _legacy("Flipkart", "Samsung Galaxy M36 5G 8GB 128GB", 21000, "https://www.flipkart.com/a/p/itm1"),
    ])
    candidates, lowest = engines.candidate_links(product)
    assert [c.platform for c in candidates] == ["Flipkart"]
    assert candidates[0].match_source is None and lowest is None        # no image evidence -> not verified


def test_extended_matches_pass_through_and_verified_comes_first(monkeypatch):
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: None)
    product = _product([
        _extended("Croma", "SAMSUNG Galaxy M36 5G (8GB RAM, 128GB)", 24999, "https://www.croma.com/m36/p/1", "verified", 0.88),
        _extended("Vijay Sales", "Samsung Galaxy M36 5G 8GB/256GB", 18999, "https://www.vijaysales.com/p/2", "candidate", 0.6),
    ])
    candidates, lowest = engines.candidate_links(product)
    assert [c.platform for c in candidates] == ["Croma", "Vijay Sales"]      # verified first even though pricier
    assert candidates[0].status == "verified" and candidates[0].match_score == 0.88
    assert candidates[1].status == "candidate" and candidates[1].relation == "similar"
    assert candidates[1].differences == ["storage 256 GB instead of 128 GB"] and "256 GB" in candidates[1].reason
    assert candidates[0].relation == "exact" and candidates[0].differences == []
    assert lowest.platform == "Croma"                       # a cheaper similar product never becomes the lowest verified price


def test_same_store_duplicates_keep_the_stronger_evidence(monkeypatch):
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: None)
    product = _product([
        _legacy("Snapdeal", "Samsung Galaxy M36 5G 8GB 128GB", 19000, "https://www.snapdeal.com/product/x/1"),
        _extended("Snapdeal", "Samsung Galaxy M36 5G 8GB 128GB", 19500, "https://www.snapdeal.com/product/x/2", "verified", 0.9),
    ], platform="Croma")
    candidates, _ = engines.candidate_links(product)
    assert len(candidates) == 1 and candidates[0].match_source == "enhanced"


def test_legacy_page_text_and_accessories_are_hidden_but_real_matches_stay(monkeypatch):
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: None)
    product = _product([
        _legacy("Flipkart", "Samsung Galaxy M36 5G 8GB 128GB", 21000, "https://www.flipkart.com/m36/p/itm1"),
        _legacy("Flipkart", "AceKing SAMSUNG M36 BACK COVER BACK DOOR PANEL Back Panel", 430, "https://www.flipkart.com/cover/p/itm2"),
        _legacy("Flipkart", "Need help?", None, "https://www.flipkart.com/help/p/itm3"),
    ])
    candidates, _ = engines.candidate_links(product)
    assert [c.title for c in candidates] == ["Samsung Galaxy M36 5G 8GB 128GB"]
    monkeypatch.setenv("LEGACY_RESULT_FILTER", "false")
    assert len(engines.candidate_links(product)[0]) == 3
