"""Review collection for any store (offline: every page is a small fixture)."""
import json

import pytest

from app import review_collectors as rc
from app.review_signals import parse_review_date


def flipkart_page(reviews):
    nodes = [{"value": {"type": "ProductReviewValue", "id": r[0], "author": "A", "certifiedBuyer": "True", "created": r[3],
                        "helpfulCount": "2", "rating": str(r[1]), "text": r[2], "title": "Nice"}} for r in reviews]
    state = {"multiWidgetState": {"widgetsData": {"slots": [{"slotData": {"widget": {"data": {"renderableComponents": [n]}}}} for n in nodes]}}}
    return f"<html><script>window.__INITIAL_STATE__ = {json.dumps(state)};</script></html>"


PRODUCT_URL = "https://www.flipkart.com/samsung-m36/p/itm123"
PRODUCT_HTML = '<script type="application/ld+json">{"@type":"Product","sku":"MOBHEYY2FCCGHYEM","aggregateRating":{"reviewCount":461}}</script>'


def fake_get(pages):
    calls = []

    def get(url, params=None, timeout=15.0):
        calls.append((url, dict(params or {})))
        if "/product-reviews/" not in url:
            return 200, PRODUCT_HTML
        key = (params["sortOrder"], params["page"])
        return (200, flipkart_page(pages.get(key, []))) if key in pages else (200, flipkart_page([]))
    get.calls = calls
    return get


def test_flipkart_review_url_uses_the_listings_own_product_id():
    base = rc._flipkart_review_base(PRODUCT_HTML, PRODUCT_URL)
    assert base == ("https://www.flipkart.com/samsung-m36/product-reviews/itm123", {"pid": "MOBHEYY2FCCGHYEM"})
    # a ?pid= on the pasted link wins; pids of recommended products on the page are never used
    assert rc._flipkart_review_base("<a>MOBOTHER0000000</a> pid=MOBOTHER0000000", PRODUCT_URL + "?pid=MOBMINE1234567")[1] == {"pid": "MOBMINE1234567"}
    # the listing id in the path is enough on its own: the pid is only a hint
    assert rc._flipkart_review_base("no product data", PRODUCT_URL) == ("https://www.flipkart.com/samsung-m36/product-reviews/itm123", {})
    assert rc._flipkart_review_base("x", "https://www.flipkart.com/some-page") is None


def test_flipkart_pages_are_parsed_and_deduplicated_across_sorts():
    pages = {
        ("MOST_RECENT", 1): [("r1", 5, "Great phone, battery lasts two days easily", "Today"), ("r2", 1, "Stopped working after a week", "3 months ago")],
        ("MOST_RECENT", 2): [("r3", 4, "Good value for money overall", "Oct, 2025")],
        ("NEGATIVE_FIRST", 1): [("r2", 1, "Stopped working after a week", "3 months ago"), ("r4", 1, "Terrible camera in low light", "1 year ago")],
    }
    get = fake_get(pages)
    got = rc.collect_reviews(PRODUCT_URL, limit=100, get=get)
    assert [r["text"] for r in got.reviews] == [
        "Great phone, battery lasts two days easily", "Stopped working after a week", "Good value for money overall", "Terrible camera in low light"]
    first = got.reviews[0]
    assert first["rating"] == 5 and first["verified_purchase"] is True and first["helpful"] == 2
    assert first["timestamp"] is not None                                   # "Today" is an exact day
    assert got.reviews[1]["timestamp"] is None                              # "3 months ago" is not, so no invented date
    assert "Flipkart" in got.source
    # the review pages are requested without first downloading the product page
    assert all("product-reviews" in call[0] for call in get.calls)
    with_page = rc.collect_reviews(PRODUCT_URL, html=PRODUCT_HTML, limit=100, get=fake_get(pages))
    assert with_page.reported_total == 461


def test_flipkart_limit_is_respected_and_small_limits_stay_small():
    pages = {("MOST_RECENT", p): [(f"id{p}{i}", 4, f"Review number {p}-{i} with enough words in it", "Today") for i in range(10)] for p in range(1, 13)}
    get = fake_get(pages)
    assert len(rc.collect_reviews(PRODUCT_URL, limit=25, get=get).reviews) == 25
    requested = {c[1]["sortOrder"] for c in get.calls if "product-reviews" in c[0]}
    assert requested == {"MOST_RECENT"}                                      # a quick look does not hit every sort order


def test_page_markup_reviews_are_read_for_any_store():
    html = """<script type="application/ld+json">{"@type":"Product","name":"X","aggregateRating":{"ratingValue":4.2,"reviewCount":"1,204"},
      "review":[{"@type":"Review","reviewBody":"Works as described and arrived on time.","reviewRating":{"ratingValue":"5"},"datePublished":"2026-03-04","author":{"name":"Asha"}},
                {"@type":"Review","reviewBody":"Strap broke within a month, disappointed.","reviewRating":{"ratingValue":2}}]}</script>
      <div itemprop="review"><meta itemprop="ratingValue" content="4"><p itemprop="reviewBody">Microdata review text goes here.</p></div>"""
    got = rc.collect_reviews("https://www.example-store.com/p/1", html=html)
    assert [r["rating"] for r in got.reviews] == [5, 2, 4]
    assert got.reviews[0]["author"] == "Asha" and got.reported_total == 1204


def test_a_page_with_no_reviews_gives_an_empty_result_not_an_error():
    got = rc.collect_reviews("https://www.example-store.com/p/1", html="<html><h1>Product</h1></html>")
    assert got.reviews == [] and got.issues == []


def test_browser_scan_runs_only_when_asked_and_only_when_the_fast_layers_found_little(monkeypatch):
    calls = []
    monkeypatch.setattr(rc, "_run_browser_collector", lambda url, limit, timeout=50: calls.append(url) or
                        {"reviews": [{"text": "Fits well and the colour is true to the photos.", "rating": 5, "date": "2 weeks ago"}], "via": ["reviews drawn on the page"], "issues": []})
    page = "<html></html>"
    assert rc.collect_reviews("https://www.example-store.com/p/1", html=page).reviews == [] and calls == []        # not asked
    got = rc.collect_reviews("https://www.example-store.com/p/1", html=page, use_browser=True)
    assert len(got.reviews) == 1 and "Chrome" in got.source and len(calls) == 1
    assert got.reviews[0]["timestamp"] is not None
    # plenty found by the fast layers: the browser is not started
    rich = "".join(f'<div itemprop="review"><meta itemprop="ratingValue" content="5"><p itemprop="reviewBody">Review number {i} says it is good</p></div>' for i in range(4))
    rc.collect_reviews("https://www.example-store.com/p/1", html=rich, use_browser=True)
    assert len(calls) == 1


def test_a_failing_store_never_raises(monkeypatch):
    def broken(url, params=None, timeout=15.0):
        raise RuntimeError("boom")
    got = rc.collect_reviews(PRODUCT_URL, get=broken)
    assert got.reviews == [] and got.issues


@pytest.mark.parametrize("text,exact", [("Today", True), ("yesterday", True), ("3 days ago", True), ("2 weeks ago", True),
                                       ("2 months ago", False), ("1 year ago", False), ("Jan, 2025", False)])
def test_relative_dates_are_only_dated_when_they_are_exact(text, exact):
    assert (parse_review_date(text) is not None) is exact


def test_flipkart_falls_back_to_the_product_page_when_the_link_alone_returns_nothing():
    pages = {("MOST_RECENT", 1): [("r1", 5, "Works well and the screen is bright and sharp", "Today")]}
    calls = []

    def get(url, params=None, timeout=15.0):
        calls.append((url, dict(params or {})))
        if "/product-reviews/" not in url:
            return 200, PRODUCT_HTML
        # without the page's own pid Flipkart answers with nothing; with it, reviews
        return (200, flipkart_page(pages.get((params["sortOrder"], params["page"]), []))) if params.get("pid") else (200, flipkart_page([]))

    got = rc.collect_reviews(PRODUCT_URL, limit=10, get=get)
    assert [r["text"] for r in got.reviews] == ["Works well and the screen is bright and sharp"]
    assert any(c[0] == PRODUCT_URL for c in calls)
