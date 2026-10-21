"""Any source store -> other stores -> price comparison, with every scraper stubbed (no network, no browser)."""
import pytest
from fastapi.testclient import TestClient

from app import engines, main, multistore
from app.matcher import OUT_OF_STOCK_FLAG
from app.models import Product
from app.source_extractor import SourceReadError

CROMA_URL = "https://www.croma.com/samsung-galaxy-m36-5g-8gb-ram-128gb/p/315001"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "DATA_PATH", tmp_path / "observations.db")
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: None)
    main.REPORTS_CACHE.clear()


def extended(platform, price, status="verified", url=None, flags=(), title="Samsung Galaxy M36 5G (8GB RAM, 128GB)", score=0.9):
    return {"platform": platform, "title": title, "url": url or f"https://{platform.lower().replace(' ', '')}.example/m36/p/1",
            "price": price, "rating": 4.2, "review_count": 300, "image_url": None, "reviews": [],
            "match_source": "enhanced", "match_status": status, "match_score": score,
            "match_reasons": ["model code m36 matches", "storage agrees"] if status == "verified" else [],
            "match_flags": list(flags) if flags else ([] if status == "verified" else ["different specification"]),
            "title_similarity": 0.7}


def croma_product(matches, price=21999.0):
    product = Product(platform="Croma", url=CROMA_URL, title="Samsung Galaxy M36 5G (8GB RAM, 128GB, Velvet Black)",
                      price=price, brand="Samsung", rating=4.3, review_count=1204, category="electronics",
                      image_url="https://img/m36.jpg")
    product._p_data = {"competitor_matches": matches, "in_stock": True, "gtin": "8806095000000",
                       "competitor_collection": {"attempts": [
                           {"platform": "Amazon", "status": "results collected", "result_count": 1},
                           {"platform": "Snapdeal", "status": "blocked", "result_count": 0}]}}
    return product


def test_analyse_from_a_croma_link_lists_every_store_and_marks_the_lowest(monkeypatch):
    matches = [
        extended("Amazon", 20999.0), extended("Flipkart", 19499.0), extended("Vijay Sales", 20250.0),
        extended("Tata CLiQ", 17000.0, flags=[OUT_OF_STOCK_FLAG]),
        extended("Reliance Digital", 12000.0, status="candidate", title="Samsung Galaxy M36 5G 8GB 256GB"),
    ]
    monkeypatch.setattr(main, "extract_product", lambda url, mode="extract": (croma_product(matches), []))
    client = TestClient(main.app)

    response = client.post("/api/analyse", json={"url": CROMA_URL})
    assert response.status_code == 200
    body = client.get(f"/api/analyse/{response.json()['analysis_id']}").json()     # after the background pass

    comparison = body["price_comparison"]
    assert comparison["source_store"] == "Croma"
    stores = {entry["store"]: entry for entry in comparison["series"]}
    assert set(stores) == {"Croma", "Amazon", "Flipkart", "Vijay Sales", "Tata CLiQ"}
    assert all(entry["url"].startswith("http") for entry in comparison["series"])
    assert comparison["lowest"]["store"] == "Flipkart" and comparison["lowest"]["price"] == 19499.0
    assert comparison["lowest_summary"].startswith("Lowest: Rs 19,499 at Flipkart")
    assert stores["Tata CLiQ"]["excluded_reason"] == "out of stock" and stores["Tata CLiQ"]["is_lowest"] is False
    assert [e["store"] for e in comparison["possible"]] == ["Reliance Digital"]          # listed, but kept apart
    assert [(s["store"], s["status"]) for s in comparison["stores_searched"]] == [("Amazon", "results collected"), ("Snapdeal", "blocked")]
    # the legacy fields are still delivered unchanged
    assert body["lowest_verified_match"]["platform"] == "Flipkart"
    assert body["price_history"] and body["price_summary"]["obs_count"] >= 1


def test_candidate_prices_are_stored_once_and_become_real_history(monkeypatch):
    matches = [extended("Amazon", 20999.0), extended("Tata CLiQ", 17000.0, flags=[OUT_OF_STOCK_FLAG]),
               extended("Flipkart", 19499.0, status="candidate", title="Samsung Galaxy M36 5G 8GB 256GB")]
    monkeypatch.setattr(main, "extract_product", lambda url, mode="extract": (croma_product(matches), []))
    monkeypatch.setattr(main, "CACHE_SECONDS", 0)            # this test is about repeat runs reaching the store again
    client = TestClient(main.app)
    client.post("/api/analyse", json={"url": CROMA_URL})

    amazon = engines.observed_price_history("https://amazon.example/m36/p/1")
    assert [price for _, price in amazon] == [20999.0]                       # real: stored by the enrichment pass
    assert engines.observed_price_history("https://tatacliq.example/m36/p/1") == []     # out of stock: nothing stored
    assert len(engines.observed_price_history(CROMA_URL)) == 1                # source stored by the first pass only

    # a repeat run within 30 minutes at the same price adds nothing; a changed price adds a point
    client.post("/api/analyse", json={"url": CROMA_URL})
    assert len(engines.observed_price_history("https://amazon.example/m36/p/1")) == 1
    matches[0]["price"] = 19999.0
    client.post("/api/analyse", json={"url": CROMA_URL})
    assert [price for _, price in engines.observed_price_history("https://amazon.example/m36/p/1")] == [20999.0, 19999.0]


def test_observed_history_never_creates_or_pads_the_database(tmp_path, monkeypatch):
    path = tmp_path / "missing.db"
    monkeypatch.setattr(engines, "DATA_PATH", path)
    assert engines.observed_price_history("https://x.example/p") == []
    assert not path.exists()


def test_unreadable_page_returns_a_plain_error(monkeypatch):
    def refuse(url, mode="extract"):
        raise SourceReadError("Could not read this page: the site showed a CAPTCHA / access-denied page to automated requests.", "blocked")

    monkeypatch.setattr(main, "extract_product", refuse)
    response = TestClient(main.app).post("/api/analyse", json={"url": "https://www.vijaysales.com/p/123"})
    assert response.status_code == 400
    assert response.json()["reason"] == "blocked" and response.json()["detail"].startswith("Could not read this page")


def test_links_that_cannot_be_a_product_page_are_rejected_early():
    client = TestClient(main.app)
    assert client.post("/api/analyse", json={"url": "http://localhost:8000/admin"}).status_code == 400
    assert client.post("/api/analyse", json={"url": "http://192.168.0.1/p"}).status_code == 400


# --- extract_product: a store scraper.js has no selectors for ----------------------------------------------

def _stub_scraper(monkeypatch, scraped, generic=None, error=None):
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": scraped)

    def read(url):
        return generic, error

    monkeypatch.setattr(engines, "_read_generic_source", read)
    monkeypatch.setattr(multistore, "enhanced_competitor_matches", lambda *a, **k: ([], [], None))


GENERIC = {"title": "Samsung Galaxy M36 5G (8GB RAM, 128GB, Velvet Black)", "price": 19999.0, "currency": "INR", "brand": "Samsung",
           "image_url": "https://www.croma.com/m36.jpg", "gtin": "8806095000000", "mpn": "SM-M366BZKGINU", "sku": "315001",
           "rating": 4.3, "review_count": 1204, "availability": "https://schema.org/InStock", "in_stock": True,
           "sources": ["http:json-ld"], "attempts": ["http: HTTP 200"]}


def test_extract_product_reads_a_store_scraper_js_does_not_know(monkeypatch):
    junk = {"platform": "croma.com", "title": "h1 guess", "price": 5.0, "sku": None, "reviews": [], "competitor_matches": []}
    _stub_scraper(monkeypatch, junk, generic=GENERIC)
    product, notices = engines.extract_product("https://www.croma.com/m36/p/315001", mode="source")
    assert product.platform == "Croma"                                     # registry name, not "croma.com"
    assert product.title.startswith("Samsung Galaxy M36") and product.price == 19999.0 and product.brand == "Samsung"
    assert product._p_data["gtin"] == "8806095000000" and product._p_data["in_stock"] is True


def test_extract_product_works_when_the_headless_scraper_fails_but_the_page_declares_the_product(monkeypatch):
    _stub_scraper(monkeypatch, None, generic=GENERIC)
    product, _ = engines.extract_product("https://www.vijaysales.com/p/315001", mode="source")
    assert product.platform == "Vijay Sales" and product.price == 19999.0


def test_extract_product_reports_a_wall_even_if_scraper_js_returned_a_title(monkeypatch):
    _stub_scraper(monkeypatch, {"platform": "croma.com", "title": "Access Denied"}, error=SourceReadError("Could not read this page: blocked.", "blocked"))
    with pytest.raises(SourceReadError):
        engines.extract_product("https://www.croma.com/m36/p/315001", mode="source")


def test_extract_product_does_not_trust_scraper_js_on_an_http_error_page(monkeypatch):
    _stub_scraper(monkeypatch, {"platform": "vijaysales.com", "title": "Air Conditioners"},
                  error=SourceReadError("Could not read this page: the site answered HTTP 404.", "http_404"))
    with pytest.raises(SourceReadError) as caught:
        engines.extract_product("https://www.vijaysales.com/p/gone", mode="source")
    assert caught.value.reason == "http_404"


def test_extract_product_rejects_scraper_js_junk_that_has_no_price(monkeypatch):
    _stub_scraper(monkeypatch, {"platform": "reliancedigital.in", "title": "Buy undefined - | Online Electronic Shopping Store in India"},
                  error=SourceReadError("Could not read this page: no product data.", "no_product_data"))
    with pytest.raises(SourceReadError):
        engines.extract_product("https://www.reliancedigital.in/product/x", mode="source")


def test_extract_product_keeps_scraper_js_data_when_the_page_just_has_no_markup(monkeypatch):
    _stub_scraper(monkeypatch, {"platform": "Nykaa", "title": "Maybelline Fit Me Foundation", "price": 499.0, "reviews": []},
                  error=SourceReadError("Could not read this page: no product data.", "no_product_data"))
    product, _ = engines.extract_product("https://www.nykaa.com/maybelline/p/1", mode="source")
    assert product.title == "Maybelline Fit Me Foundation" and product.price == 499.0


def test_amazon_links_never_touch_the_generic_reader(monkeypatch):
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": {"platform": "Amazon", "title": "Samsung Galaxy M36 5G", "price": 21000.0, "reviews": []})
    monkeypatch.setattr(engines, "_read_generic_source", lambda url: pytest.fail("generic reader must not run for Amazon"))
    product, _ = engines.extract_product("https://www.amazon.in/dp/B0FDBB2VRC", mode="source")
    assert product.platform == "Amazon" and "gtin" not in product._p_data


FLIPKART = "https://www.flipkart.com/samsung-galaxy-m36/p/itm123"


def test_flipkart_first_pass_reads_the_page_markup_and_skips_the_slow_browser_scraper(monkeypatch):
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": pytest.fail("quick Flipkart pass must not start Chrome"))
    monkeypatch.setattr(engines, "_read_generic_source", lambda url: (GENERIC, None))
    product, _ = engines.extract_product(FLIPKART, mode="source")
    assert product.platform == "Flipkart" and product.price == 19999.0 and product._p_data["gtin"] == "8806095000000"


def test_flipkart_first_pass_falls_back_to_the_scraper_when_the_page_declares_nothing(monkeypatch):
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": {"platform": "Flipkart", "title": "Samsung Galaxy M36 5G", "price": 21000.0, "reviews": []})
    monkeypatch.setattr(engines, "_read_generic_source", lambda url: (None, SourceReadError("no product data", "no_product_data")))
    product, _ = engines.extract_product(FLIPKART, mode="source")
    assert product.platform == "Flipkart" and product.price == 21000.0


def test_flipkart_background_pass_still_runs_the_scraper_for_its_legacy_amazon_search(monkeypatch):
    ran = []
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": ran.append(mode) or {
        "platform": "Flipkart", "title": "Samsung Galaxy M36 5G", "price": 21000.0, "reviews": [],
        "competitor_matches": [{"platform": "Amazon", "title": "Samsung Galaxy M36 5G", "url": "https://www.amazon.in/dp/B0X", "price": 20999.0}]})
    monkeypatch.setattr(engines, "_read_generic_source", lambda url: (GENERIC, None))
    monkeypatch.setattr(multistore, "enhanced_competitor_matches", lambda *a, **k: ([], [], None))
    product, _ = engines.extract_product(FLIPKART, mode="extract")
    assert ran == ["extract"] and product._p_data["competitor_matches"][0]["platform"] == "Amazon"


def test_reviews_collected_from_the_store_are_added_without_duplicates(monkeypatch):
    from app.review_collectors import CollectedReviews
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": {
        "platform": "Flipkart", "title": "Samsung Galaxy M36 5G", "price": 21000.0,
        "reviews": [{"text": "Battery easily lasts a full day", "rating": 4}]})
    monkeypatch.setattr(engines, "_read_generic_source", lambda url: (GENERIC, None))
    monkeypatch.setattr(engines, "_collect_store_reviews", lambda url, mode: CollectedReviews(
        reviews=[{"text": "Battery easily lasts a full day", "rating": 4},
                 {"text": "Camera is poor in low light", "rating": 2, "date": "3 days ago", "timestamp": "2026-10-17T00:00:00+00:00", "verified_purchase": True}],
        source="Flipkart review pages", reported_total=461))
    product, notices = engines.extract_product(FLIPKART, mode="source")
    assert [r.text for r in product.reviews] == ["Battery easily lasts a full day", "Camera is poor in low light"]
    assert product.reviews[1].verified_purchase is True and product.reviews[1].timestamp is not None
    assert any("Flipkart review pages" in n for n in notices)


def test_the_second_pass_reuses_what_the_first_pass_read(monkeypatch):
    reads = []
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": None)
    monkeypatch.setattr(engines, "_read_generic_source", lambda url: reads.append(url) or (GENERIC, None))
    monkeypatch.setattr(multistore, "enhanced_competitor_matches", lambda *a, **k: ([], [], None))
    engines.extract_product("https://www.croma.com/m36/p/315001", mode="source")
    engines.extract_product("https://www.croma.com/m36/p/315001", mode="extract")
    assert len(reads) == 1                                                  # the background pass did not read the page again


def test_progress_hook_hears_about_matches_and_reviews_as_they_finish(monkeypatch):
    from app.review_collectors import CollectedReviews
    monkeypatch.setattr(engines, "run_puppeteer_scraper", lambda url, mode="extract": None)
    monkeypatch.setattr(engines, "_read_generic_source", lambda url: (GENERIC, None))
    monkeypatch.setattr(engines, "_collect_store_reviews", lambda url, mode: CollectedReviews(
        reviews=[{"text": "Works great for the price", "rating": 5}], source="page markup"))
    monkeypatch.setattr(multistore, "enhanced_competitor_matches", lambda *a, **k: ([extended("Amazon", 19000.0)], [], None))
    heard = []
    token = engines.set_progress_hook(lambda kind, value: heard.append(kind))
    try:
        engines.extract_product("https://www.croma.com/m36/p/315001", mode="extract")
    finally:
        engines.reset_progress_hook(token)
    assert {"reviews", "matches"} <= set(heard)


# --- searching the other stores ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform,url,category,must_have,must_not_have", [
    ("Croma", "https://www.croma.com/m36/p/1", "electronics", {"Amazon", "Flipkart", "Reliance Digital", "Vijay Sales", "Tata CLiQ", "Snapdeal"}, {"Croma"}),
    ("croma.com", "https://www.croma.com/m36/p/1", "electronics", {"Amazon", "Flipkart"}, {"Croma"}),
    ("Reliancedigital", "https://www.reliancedigital.in/x/p/1", "electronics", {"Amazon", "Flipkart", "Croma"}, {"Reliance Digital"}),
    ("vijaysales.com", "https://www.vijaysales.com/p/1", "electronics", {"Amazon", "Flipkart", "Croma"}, {"Vijay Sales"}),
    ("Snapdeal", "https://www.snapdeal.com/product/x/1", "general", {"Amazon", "Flipkart", "Croma"}, {"Snapdeal"}),
    ("Myntra", "https://www.myntra.com/shoes/nike/1/buy", "fashion", {"Amazon", "Flipkart", "Ajio", "Tata CLiQ"}, {"Myntra"}),
    ("Nykaa", "https://www.nykaa.com/x/p/1", "beauty", {"Amazon", "Flipkart", "Myntra"}, {"Nykaa"}),
    ("Amazon", "https://www.amazon.in/dp/B0X", "electronics", {"Croma", "Reliance Digital", "Vijay Sales"}, {"Amazon", "Flipkart"}),
    ("Flipkart", "https://www.flipkart.com/x/p/itm1", "electronics", {"Croma", "Snapdeal"}, {"Amazon", "Flipkart"}),
    ("pepperfry.com", "https://www.pepperfry.com/product/x-1", "general", {"Amazon", "Flipkart", "Snapdeal"}, {"Pepperfry"}),
])
def test_cross_store_search_skips_the_source_store_and_covers_the_rest(platform, url, category, must_have, must_not_have):
    stores = set(multistore.select_stores(platform, url, category))
    assert must_have <= stores, stores
    assert not (must_not_have & stores), stores
    assert not {"JioMart", "Meesho", "BigBasket"} & stores               # disabled stores are never searched


def test_amazon_and_flipkart_come_first_when_the_source_is_elsewhere():
    assert multistore.select_stores("Croma", "https://www.croma.com/m36/p/1", "electronics")[:2] == ["Amazon", "Flipkart"]


def test_extended_match_from_the_source_store_is_dropped_by_candidate_links():
    product = croma_product([extended("Croma", 15000.0, url="https://www.croma.com/m36-other/p/9"), extended("Amazon", 20999.0)])
    candidates, _ = engines.candidate_links(product)
    assert [c.platform for c in candidates] == ["Amazon"]


def test_a_link_analysed_a_moment_ago_answers_instantly_without_fetching_again(monkeypatch):
    calls = []

    def fake(url, mode="extract"):
        calls.append(mode)
        return croma_product([extended("Amazon", 20999.0)]), []

    monkeypatch.setattr(main, "extract_product", fake)
    client = TestClient(main.app)
    first = client.post("/api/analyse", json={"url": CROMA_URL}).json()
    assert calls == ["source", "extract"]                                   # the quick pass, then the background pass
    again = client.post("/api/analyse", json={"url": CROMA_URL + "?utm_source=mail&ref=abc"}).json()
    assert calls == ["source", "extract"]                                   # nothing fetched the second time
    assert again["enrichment_status"] == "complete" and again["analysis_id"] != first["analysis_id"]
    assert again["notices"][0].startswith("Showing the finished analysis of this link")
    assert again["candidates"] and client.get(f"/api/analyse/{again['analysis_id']}").status_code == 200


def test_partial_results_appear_while_the_slow_work_is_still_running(monkeypatch):
    """Stores and reviews answer at different times; the page must be able to show each as it arrives."""
    from app.review_collectors import CollectedReviews
    seen = {}

    def slow_extract(url, mode="extract"):
        hook = engines._progress_hook.get()
        hook("matches", ([extended("Vijay Sales", 19999.0)], [{"platform": "Vijay Sales", "status": "results collected", "result_count": 1}], None))
        seen["after_stores"] = dict(main.REPORTS_CACHE[seen["id"]])
        hook("reviews", CollectedReviews(reviews=[{"text": "Battery easily lasts a full day and the screen is bright", "rating": 5}], source="Flipkart review pages"))
        seen["after_reviews"] = dict(main.REPORTS_CACHE[seen["id"]])
        return croma_product([extended("Vijay Sales", 19999.0)]), []

    def quick(url, mode="extract"):
        return (croma_product([]), []) if mode == "source" else slow_extract(url, mode)

    monkeypatch.setattr(main, "extract_product", quick)
    original = main._enrich_analysis
    monkeypatch.setattr(main, "_enrich_analysis", lambda analysis_id, url, first=None: (seen.__setitem__("id", analysis_id), original(analysis_id, url, first))[1])
    client = TestClient(main.app)
    client.post("/api/analyse", json={"url": CROMA_URL})

    assert seen["after_stores"]["enrichment_status"] == "pending" and [c["platform"] for c in seen["after_stores"]["candidates"]] == ["Vijay Sales"]
    assert seen["after_reviews"]["enrichment_status"] == "pending" and len(seen["after_reviews"]["findings"]) == 1
    final = client.get(f"/api/analyse/{seen['id']}").json()
    assert final["enrichment_status"] == "complete"
