"""A listed link must open on the product: dead ones are dropped, unreadable ones are shown unchecked (offline)."""
import httpx
import pytest

from app import link_check

PRODUCT = "Samsung Galaxy M36 5G (Orange Haze, 8GB RAM, 128GB Storage)"
URL = "https://www.croma.com/samsung-galaxy-m36-5g-8gb-128gb/p/315970"

PRODUCT_PAGE = """<html><head><title>Samsung Galaxy M36 5G 8GB 128GB | Croma</title>
<script type="application/ld+json">{"@type": "Product", "name": "Samsung Galaxy M36 5G (8GB RAM, 128GB)",
 "offers": {"@type": "Offer", "price": "21999"}}</script></head><body><h1>Samsung Galaxy M36 5G</h1></body></html>"""


@pytest.fixture(autouse=True)
def switched_on(monkeypatch):
    monkeypatch.setenv("LINK_CHECK", "true")
    link_check.clear()


def serve(monkeypatch, handler):
    calls = []

    def recording(request):
        calls.append(str(request.url))
        return handler(request)

    monkeypatch.setattr(link_check, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(recording), follow_redirects=True, max_redirects=6, event_hooks={"request": [link_check._guard]}))
    return calls


def page(status=200, html="", headers=None):
    return lambda request: httpx.Response(status, text=html, headers=headers or {"content-type": "text/html"})


def test_a_product_page_is_ok(monkeypatch):
    serve(monkeypatch, page(html=PRODUCT_PAGE))
    result = link_check.check(URL, PRODUCT)
    assert result.state == "ok"


@pytest.mark.parametrize("status", [404, 410])
def test_a_missing_page_is_dead(monkeypatch, status):
    serve(monkeypatch, page(status, "<title>gone</title>"))
    assert link_check.check(URL, PRODUCT).state == "dead"


def test_a_redirect_to_the_home_page_is_dead(monkeypatch):
    def handler(request):
        if request.url.path == "/":
            return httpx.Response(200, text="<html><title>Croma - Online Electronics</title></html>")
        return httpx.Response(301, headers={"location": "https://www.croma.com/"})
    serve(monkeypatch, handler)
    result = link_check.check(URL, PRODUCT)
    assert result.state == "dead" and "home page" in result.reason


def test_a_redirect_to_a_search_page_is_dead(monkeypatch):
    def handler(request):
        if request.url.path == "/search":
            return httpx.Response(200, text="<html><title>Search results</title></html>")
        return httpx.Response(302, headers={"location": "https://www.croma.com/search?q=samsung"})
    serve(monkeypatch, handler)
    assert link_check.check(URL, PRODUCT).state == "dead"


def test_a_not_found_page_served_with_status_200_is_dead(monkeypatch):
    serve(monkeypatch, page(html="<html><head><title>Page Not Found | Croma</title></head><body><h1>Oops!</h1></body></html>"))
    assert link_check.check(URL, PRODUCT).state == "dead"


def test_a_page_that_shows_another_product_is_dead(monkeypatch):
    other = PRODUCT_PAGE.replace("Samsung Galaxy M36 5G (8GB RAM, 128GB)", "Apple iPhone 15 Pro Max").replace("Samsung Galaxy M36 5G", "Apple iPhone 15 Pro Max")
    serve(monkeypatch, page(html=other))
    result = link_check.check("https://www.croma.com/item/p/77", PRODUCT)
    assert result.state == "dead" and "different product" in result.reason


@pytest.mark.parametrize("status", [403, 429, 503])
def test_a_store_that_refuses_automated_requests_is_unchecked_not_dead(monkeypatch, status):
    serve(monkeypatch, page(status, "blocked"))
    assert link_check.check(URL, PRODUCT).state == "unchecked"


def test_a_verification_page_is_unchecked(monkeypatch):
    serve(monkeypatch, page(html="<html><head><title>Robot Check</title></head><body>captcha</body></html>"))
    assert link_check.check(URL, PRODUCT).state == "unchecked"


def test_a_page_built_by_script_is_unchecked(monkeypatch):
    serve(monkeypatch, page(html="<html><head><title>Croma</title></head><body><div id='root'></div></body></html>"))
    assert link_check.check("https://www.croma.com/p/1", PRODUCT).state == "unchecked"


def test_a_slow_store_is_unchecked_and_asked_again_next_time(monkeypatch):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)
    serve(monkeypatch, slow)
    assert link_check.check(URL, PRODUCT).state == "unchecked"
    calls = serve(monkeypatch, page(html=PRODUCT_PAGE))
    assert link_check.check(URL, PRODUCT).state == "ok" and len(calls) == 1             # the failure was not remembered


def test_a_redirect_loop_is_dead(monkeypatch):
    serve(monkeypatch, lambda request: httpx.Response(302, headers={"location": str(request.url)}))
    assert link_check.check(URL, PRODUCT).state == "dead"


def test_answers_are_kept_so_a_listing_is_opened_once(monkeypatch):
    calls = serve(monkeypatch, page(html=PRODUCT_PAGE))
    assert link_check.check(URL, PRODUCT).state == "ok"
    assert link_check.check(URL, PRODUCT).state == "ok"
    assert len(calls) == 1


def test_private_and_malformed_addresses_are_dead_without_a_request(monkeypatch):
    calls = serve(monkeypatch, page(html=PRODUCT_PAGE))
    assert link_check.check("http://127.0.0.1/admin", PRODUCT).state == "dead"
    assert link_check.check("not a url", PRODUCT).state == "dead"
    assert link_check.check(None, PRODUCT).state == "dead"
    assert calls == []


def test_a_redirect_into_a_private_network_is_dead(monkeypatch):
    serve(monkeypatch, lambda request: httpx.Response(302, headers={"location": "http://192.168.1.5/router"}))
    assert link_check.check(URL, PRODUCT).state == "dead"


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("LINK_CHECK", "false")
    calls = serve(monkeypatch, page(404))
    assert link_check.check(URL, PRODUCT).state == "unchecked" and calls == []


def test_the_browsers_view_of_a_page_is_judged_the_same_way():
    seen = {"status": "ok", "final_url": URL, "title": "SAMSUNG Galaxy M36 5G (8GB RAM, 128GB, Black)", "price": 22499}
    assert link_check.judge_browser(URL, seen, PRODUCT).state == "ok"
    assert link_check.judge_browser(URL, {"status": "http_404", "final_url": URL}, PRODUCT).state == "dead"
    assert link_check.judge_browser(URL, {"status": "http_403"}, PRODUCT).state == "unchecked"
    assert link_check.judge_browser(URL, {"status": "blocked"}, PRODUCT).state == "unchecked"
    other = {"status": "ok", "final_url": URL, "title": "Apple iPhone 15 Pro Max 256GB", "price": 150000}
    assert link_check.judge_browser(URL, other, PRODUCT).state == "dead"
    home = {"status": "ok", "final_url": "https://www.croma.com/", "title": "Croma - Online Electronics Shopping"}
    assert link_check.judge_browser(URL, home, PRODUCT).state == "dead"
