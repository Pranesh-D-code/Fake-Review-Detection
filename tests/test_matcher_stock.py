"""Out-of-stock handling for extended matches (offline: the node search helper is stubbed)."""
import pytest

from app import engines, multistore
from app.matcher import OUT_OF_STOCK_FLAG
from app.models import Product

SOURCE_TITLE = "Samsung Galaxy M36 5G (Velvet Black, 8GB RAM, 128GB Storage)"
SOURCE_URL = "https://www.amazon.in/dp/B0TEST"

CROMA = {"platform": "Croma", "title": "SAMSUNG Galaxy M36 5G (8GB RAM, 128GB, Velvet Black)",
         "url": "https://www.croma.com/samsung-galaxy-m36-5g/p/301", "price": 21999.0, "out_of_stock": True}
VIJAY = {"platform": "Vijay Sales", "title": "Samsung Galaxy M36 5G 8GB RAM 128GB Storage Velvet Black",
         "url": "https://www.vijaysales.com/p/m36-5g/302", "price": 23499.0, "out_of_stock": False}


def _stub_node(items, deep_details=None):
    """Replacement for multistore._run_node: never starts node, never touches a network."""
    calls = []

    def run(mode, payload, timeout):
        calls.append(mode)
        if mode == "search":
            platforms = sorted({i["platform"] for i in items})
            return {"stores": [{"platform": p, "status": "ok", "result_count": 1, "query": "q"} for p in platforms],
                    "matches": [dict(i) for i in items]}
        if deep_details is None:
            return None
        return {"details": [dict(d) for d in deep_details]}

    run.calls = calls
    return run


def _run(monkeypatch, items, deep_details=None, **kwargs):
    stub = _stub_node(items, deep_details)
    monkeypatch.setattr(multistore, "_run_node", stub)
    monkeypatch.setattr(multistore, "select_stores", lambda *a, **k: sorted({i["platform"] for i in items}))
    matches, attempts, identity = multistore.enhanced_competitor_matches(
        SOURCE_TITLE, "Samsung", 23999.0, None, "Amazon", SOURCE_URL, **kwargs)
    return matches, attempts, stub


def _deep(item, **extra):
    return {"platform": item["platform"], "url": item["url"], "status": "ok", **extra}


@pytest.mark.parametrize("deep_runs", [True, False])
def test_out_of_stock_flag_reaches_the_returned_match(monkeypatch, deep_runs):
    # With the page opened the match is re-scored; the flag must survive that second scoring.
    details = [_deep(CROMA, price=21999.0), _deep(VIJAY, price=23499.0)] if deep_runs else None
    matches, _, stub = _run(monkeypatch, [CROMA, VIJAY], details)
    by_store = {m["platform"]: m for m in matches}
    assert ("deep" in stub.calls) is True                     # deep is attempted either way (its result may be None)
    assert OUT_OF_STOCK_FLAG in by_store["Croma"]["match_flags"]
    assert OUT_OF_STOCK_FLAG not in by_store["Vijay Sales"]["match_flags"]
    assert by_store["Croma"]["match_status"] == "verified"    # still the right product
    assert all(m["match_source"] == "enhanced" for m in matches)


def _candidate_links(matches):
    product = Product(platform="Amazon", url=SOURCE_URL, title=SOURCE_TITLE, image_url="https://img/src.jpg", category="electronics")
    product._p_data = {"competitor_matches": matches}
    return engines.candidate_links(product)


def test_out_of_stock_match_is_never_the_lowest_verified_match(monkeypatch):
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: None)
    matches, _, _ = _run(monkeypatch, [CROMA, VIJAY], [_deep(CROMA), _deep(VIJAY)])
    candidates, lowest = _candidate_links(matches)
    assert lowest is not None and lowest.platform == "Vijay Sales"      # Croma is cheaper but cannot be bought
    assert [c.platform for c in candidates] == ["Vijay Sales", "Croma"]  # in stock first, still listed
    croma = candidates[1]
    assert croma.status == "verified" and OUT_OF_STOCK_FLAG in croma.match_flags


def test_lowest_verified_is_none_when_every_verified_match_is_out_of_stock(monkeypatch):
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: None)
    matches, _, _ = _run(monkeypatch, [CROMA], [_deep(CROMA)])
    candidates, lowest = _candidate_links(matches)
    assert [c.platform for c in candidates] == ["Croma"] and lowest is None


def test_legacy_matches_are_not_affected_by_the_stock_rule(monkeypatch):
    monkeypatch.setattr(engines, "compute_image_similarity", lambda a, b: 0.9)
    legacy = {"platform": "Flipkart", "title": SOURCE_TITLE, "url": "https://www.flipkart.com/a/p/itm1", "price": 20000.0,
              "image_url": "https://img/f.jpg", "reviews": [],
              "match_flags": [OUT_OF_STOCK_FLAG]}                        # legacy dicts never carry this; it must be ignored
    product = Product(platform="Croma", url="https://www.croma.com/p/1", title=SOURCE_TITLE, image_url="https://img/src.jpg", category="electronics")
    product._p_data = {"competitor_matches": [legacy]}
    candidates, lowest = engines.candidate_links(product)
    assert candidates[0].match_source is None and lowest is not None and lowest.platform == "Flipkart"


def test_matches_carry_in_stock_and_condition_keys(monkeypatch):
    unknown = {**VIJAY, "platform": "Reliance Digital", "url": "https://www.reliancedigital.in/m36/302"}
    unknown.pop("out_of_stock")                                     # the store did not say
    matches, _, _ = _run(monkeypatch, [CROMA, VIJAY, unknown], [_deep(CROMA), _deep(VIJAY), _deep(unknown)])
    by_store = {m["platform"]: m for m in matches}
    assert by_store["Croma"]["in_stock"] is False
    assert by_store["Vijay Sales"]["in_stock"] is True
    assert by_store["Reliance Digital"]["in_stock"] is None
    assert all(m["condition"] == "new" for m in matches)
