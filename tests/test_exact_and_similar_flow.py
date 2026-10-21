"""The search lists exact and similar products apart, and never lists a link that does not open (offline)."""
import json

import pytest

from app import engines, link_check, multistore
from app.link_check import LinkCheck
from app.models import Product
from app.price_graph import build_price_comparison

TITLE = "Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB Storage)"


class FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def kill(self):
        pass


def item(platform, title, price, url):
    return {"platform": platform, "title": title, "url": url, "price": price, "image_url": None}


EXACT = item("Croma", "SAMSUNG Galaxy M36 5G (8GB RAM, 128GB, Black)", 22499, "https://www.croma.com/m36/p/9")
BIGGER = item("Croma", "Samsung Galaxy M36 5G 8GB 256GB", 24999, "https://www.croma.com/m36-256/p/10")
NEXT_MODEL = item("Croma", "Samsung Galaxy M35 5G 8GB 128GB", 17999, "https://www.croma.com/m35/p/11")
CASE = item("Croma", "Back Case for Samsung Galaxy M36 5G", 399, "https://www.croma.com/case/p/12")
DEAD = item("Croma", "Samsung Galaxy M36 5G 8GB 512GB", 27999, "https://www.croma.com/m36-512/p/13")

STORE = {"store_done": {"platform": "Croma", "status": "ok", "result_count": 5, "query": "q"}, "items": [EXACT, BIGGER, NEXT_MODEL, CASE, DEAD]}
FINAL = {"stores": [STORE["store_done"]], "matches": STORE["items"]}


@pytest.fixture
def run(monkeypatch):
    def go(links=None, progress=None, details=None):
        monkeypatch.setattr(multistore.subprocess, "Popen", lambda *a, **k: FakeProc([json.dumps(x) + "\n" for x in [STORE, FINAL]]))
        monkeypatch.setattr(multistore.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no node")))
        monkeypatch.setattr(multistore, "select_stores", lambda *a, **k: ["Croma"])
        monkeypatch.setattr(multistore, "_run_node", lambda mode, payload, timeout: {"stores": [STORE["store_done"]], "matches": STORE["items"]} if mode == "search" else {"details": details or []})
        monkeypatch.setattr(multistore, "compute_image_similarity", lambda a, b: None)
        monkeypatch.setattr(multistore, "_page_image", lambda url: None)
        monkeypatch.setenv("LINK_CHECK", "true")
        answers = links or {}
        monkeypatch.setattr(link_check, "check", lambda url, title=None: answers.get(url, LinkCheck("ok", "the page opens and shows this product")))
        return multistore.enhanced_competitor_matches(TITLE, "Samsung", 21999.0, None, "Amazon", "https://www.amazon.in/dp/B0X", progress=progress)
    return go


def by_url(matches):
    return {m["url"]: m for m in matches}


def test_exact_and_similar_products_are_listed_apart(run):
    matches, attempts, _ = run()
    found = by_url(matches)
    assert found[EXACT["url"]]["relation"] == "exact" and found[EXACT["url"]]["differences"] == []
    assert found[BIGGER["url"]]["relation"] == "similar" and "storage 256 GB instead of 128 GB" in found[BIGGER["url"]]["differences"]
    assert found[NEXT_MODEL["url"]]["relation"] == "similar"
    assert CASE["url"] not in found                                              # an accessory is neither
    assert attempts[0]["similar_count"] >= 2


def test_a_link_the_store_says_is_dead_is_never_listed(run):
    matches, attempts, _ = run(links={DEAD["url"]: LinkCheck("dead", "the store says this page does not exist (HTTP 404)", None, 404)})
    assert DEAD["url"] not in by_url(matches)
    assert attempts[0]["dead_links"] == 1


def test_link_state_travels_with_each_match(run):
    matches, _, _ = run(links={BIGGER["url"]: LinkCheck("unchecked", "the store did not allow an automated check (HTTP 403)", None, 403)})
    found = by_url(matches)
    assert found[EXACT["url"]]["link_status"] == "ok"
    assert found[BIGGER["url"]]["link_status"] == "unchecked" and "403" in found[BIGGER["url"]]["link_note"]


def test_the_early_view_has_the_same_rules(run):
    heard = []
    run(links={DEAD["url"]: LinkCheck("dead", "gone", None, 404)}, progress=lambda m, a, i: heard.append(m))
    first = by_url(heard[0])
    assert EXACT["url"] in first and DEAD["url"] not in first and CASE["url"] not in first
    assert first[BIGGER["url"]]["relation"] == "similar"


def test_a_similar_product_is_never_the_lowest_verified_price(run):
    matches, _, _ = run()
    product = Product(platform="Amazon", url="https://www.amazon.in/dp/B0X", title=TITLE, price=21999.0, brand="Samsung", category="electronics")
    product._p_data = {"competitor_matches": matches}
    candidates, lowest = engines.candidate_links(product)
    assert [c.relation for c in candidates][:1] == ["exact"]                         # exact first, similar after
    assert lowest.url == EXACT["url"]                                                # the cheaper M35 is only "similar"
    comparison = build_price_comparison(product, candidates, legacy_summary_present=False)
    shown = {entry["url"] for entry in comparison["series"] + comparison["possible"]}
    assert BIGGER["url"] not in shown and NEXT_MODEL["url"] not in shown            # similar products stay out of the price chart
    assert EXACT["url"] in shown


def test_the_same_store_keeps_an_exact_listing_and_its_other_storage_sibling(run):
    matches, _, _ = run()
    product = Product(platform="Amazon", url="https://www.amazon.in/dp/B0X", title=TITLE, price=21999.0, brand="Samsung", category="electronics")
    product._p_data = {"competitor_matches": matches}
    candidates, _ = engines.candidate_links(product)
    urls = {c.url for c in candidates}
    assert {EXACT["url"], BIGGER["url"]} <= urls                                     # 98% similar titles, still two products


def test_a_store_that_refuses_plain_requests_gets_a_second_look_in_the_browser(run):
    refused = LinkCheck("unchecked", "the store did not allow an automated check (HTTP 403)", None, 403)
    gone = {"platform": "Croma", "url": BIGGER["url"], "status": "http_404", "final_url": BIGGER["url"]}
    matches, attempts, _ = run(links={BIGGER["url"]: refused}, details=[gone])
    assert BIGGER["url"] not in by_url(matches) and attempts[0]["dead_links"] == 1          # the browser saw a 404

    shown = {"platform": "Croma", "url": BIGGER["url"], "status": "ok", "final_url": BIGGER["url"], "title": "Samsung Galaxy M36 5G 8GB 256GB", "price": 24999}
    matches, _, _ = run(links={BIGGER["url"]: refused}, details=[shown])
    assert by_url(matches)[BIGGER["url"]]["link_status"] == "ok"                              # the browser saw the product

    matches, _, _ = run(links={BIGGER["url"]: refused}, details=[])
    assert by_url(matches)[BIGGER["url"]]["link_status"] == "unchecked"                       # nobody could look: shown, but not ticked
