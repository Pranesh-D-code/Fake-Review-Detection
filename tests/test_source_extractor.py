"""Source-site-agnostic product extraction, offline (no network, no browser)."""
import json

import pytest

from app import source_extractor as sx
from app.source_extractor import SourceReadError, extract_source_product, parse_product_html


@pytest.fixture(autouse=True)
def _fresh_cache():
    sx.clear_cache()
    yield
    sx.clear_cache()


JSON_LD_PAGE = """
<html><head>
<title>Buy Samsung Galaxy M36 5G (8GB RAM, 128GB, Velvet Black) Online - Croma</title>
<meta property="og:site_name" content="Croma">
<script type="application/ld+json">
{"@context":"https://schema.org","@graph":[
  {"@type":"BreadcrumbList","itemListElement":[]},
  {"@type":"Product","name":"Samsung Galaxy M36 5G (8GB RAM, 128GB, Velvet Black)",
   "brand":{"@type":"Brand","name":"Samsung"},"image":["/img/m36.jpg"],
   "sku":"315001","mpn":"SM-M366BZKGINU","gtin13":"8806095000000",
   "aggregateRating":{"@type":"AggregateRating","ratingValue":"4.3","reviewCount":"1,204"},
   "offers":{"@type":"Offer","price":"19,999.00","priceCurrency":"INR","availability":"https://schema.org/InStock"}}
]}
</script></head><body><h1>Samsung Galaxy M36</h1></body></html>
"""

MICRODATA_PAGE = """
<html><head><title>Mixer Grinder</title></head><body>
<div itemscope itemtype="https://schema.org/Product">
  <h1 itemprop="name">Prestige Iris 750 W Mixer Grinder (3 Jars, White)</h1>
  <span itemprop="brand" itemscope itemtype="https://schema.org/Brand"><meta itemprop="name" content="Prestige"></span>
  <img itemprop="image" src="/m.jpg">
  <span itemprop="sku">PR-4411</span>
  <div itemprop="offers" itemscope itemtype="https://schema.org/Offer">
    <meta itemprop="priceCurrency" content="INR"><span itemprop="price" content="3499">Rs. 3,499</span>
    <link itemprop="availability" href="https://schema.org/OutOfStock">
  </div>
  <div itemprop="aggregateRating" itemscope><span itemprop="ratingValue">4.1</span><span itemprop="reviewCount">88</span></div>
</div></body></html>
"""

OPENGRAPH_PAGE = """
<html><head>
<meta property="og:type" content="product">
<meta property="og:site_name" content="Vijay Sales">
<meta property="og:title" content="Sony WH-CH520 Wireless Headphones | Vijay Sales">
<meta property="og:image" content="https://cdn.example.com/ch520.jpg">
<meta property="product:price:amount" content="3,990">
<meta property="product:price:currency" content="INR">
<meta property="product:availability" content="out of stock">
<meta property="product:brand" content="Sony">
</head><body></body></html>
"""


def test_json_ld_product_inside_a_graph_is_read_fully():
    product = parse_product_html(JSON_LD_PAGE, "https://www.croma.com/samsung-galaxy-m36/p/315001")
    assert product.title == "Samsung Galaxy M36 5G (8GB RAM, 128GB, Velvet Black)"
    assert product.price == 19999.0 and product.currency == "INR"
    assert product.brand == "Samsung"
    assert product.image_url == "https://www.croma.com/img/m36.jpg"
    assert product.gtin == "8806095000000" and product.mpn == "SM-M366BZKGINU" and product.sku == "315001"
    assert product.rating == 4.3 and product.review_count == 1204
    assert product.in_stock is True and product.structured is True
    assert "json-ld" in product.sources


def test_microdata_product_with_out_of_stock_offer():
    product = parse_product_html(MICRODATA_PAGE, "https://shop.example.in/p/iris")
    assert product.title.startswith("Prestige Iris 750 W")
    assert product.brand == "Prestige" and product.price == 3499.0 and product.sku == "PR-4411"
    assert product.in_stock is False
    assert product.rating == 4.1 and product.review_count == 88
    assert "microdata" in product.sources


def test_opengraph_only_page_strips_the_site_suffix_and_reads_stock():
    product = parse_product_html(OPENGRAPH_PAGE, "https://www.vijaysales.com/p/1")
    assert product.title == "Sony WH-CH520 Wireless Headphones"
    assert product.price == 3990.0 and product.brand == "Sony" and product.in_stock is False
    assert product.sources == ["opengraph"]


def test_page_without_product_data_yields_no_price_or_structure():
    product = parse_product_html("<html><head><title>Contact us</title></head><body><h1>Contact us</h1></body></html>")
    assert product.price is None and product.structured is False


def test_broken_json_ld_is_skipped_not_fatal():
    html = '<script type="application/ld+json">{"@type": "Product", "name": </script><meta property="og:title" content="Fallback name product">'
    assert parse_product_html(html).title == "Fallback name product"


@pytest.mark.parametrize("value,expected", [
    ("₹24,999.00", 24999.0), ("Rs. 1,299", 1299.0), (499, 499.0), ("0", None), (None, None), ("free", None), (-5, None),
])
def test_price_parsing_never_invents_a_value(value, expected):
    assert sx.parse_price(value) == expected


@pytest.mark.parametrize("raw,expected", [
    ("https://schema.org/InStock", True), ("InStock", True), ("http://schema.org/OutOfStock", False),
    ("SoldOut", False), ("https://schema.org/PreOrder", None), ("", None), (None, None),
])
def test_availability_mapping(raw, expected):
    assert sx.normalise_availability(raw) is expected


def test_url_gate_accepts_public_stores_and_rejects_local_targets():
    assert sx.is_acceptable_source_url("https://www.vijaysales.com/p/123")
    assert sx.is_acceptable_source_url("http://shop.example.co.in/item/9")
    for bad in ("ftp://example.com/x", "http://localhost:8000/x", "http://127.0.0.1/x", "http://192.168.1.4/p",
                "http://10.0.0.7/p", "http://printer.local/p", "http://intranet/p", "not a url", ""):
        assert not sx.is_acceptable_source_url(bad), bad


def test_http_fetch_is_enough_when_the_page_is_complete():
    calls = []

    def fetcher(url):
        calls.append("http")
        return 200, JSON_LD_PAGE, url

    def renderer(url):
        calls.append("browser")
        raise AssertionError("browser must not be used when the plain fetch had everything")

    data = extract_source_product("https://www.croma.com/m36/p/1", fetcher=fetcher, renderer=renderer)
    assert calls == ["http"] and data["price"] == 19999.0 and data["gtin"] == "8806095000000"


def test_browser_fallback_fills_what_a_plain_fetch_missed():
    shell = "<html><head><title>Loading</title></head><body><div id=app></div></body></html>"
    data = extract_source_product(
        "https://www.reliancedigital.in/x/p/1",
        fetcher=lambda url: (200, shell, url), renderer=lambda url: (200, JSON_LD_PAGE, url))
    assert data["title"].startswith("Samsung Galaxy M36") and data["price"] == 19999.0
    assert any(s.startswith("browser:") for s in data["sources"])


def test_blocked_page_is_reported_not_bypassed():
    wall = "<html><head><title>Access Denied</title></head><body>Access Denied. You don't have permission to access this server.</body></html>"
    with pytest.raises(SourceReadError) as caught:
        extract_source_product("https://www.croma.com/x/p/2", fetcher=lambda u: (403, wall, u), renderer=lambda u: (200, wall, u))
    assert caught.value.reason == "blocked"
    assert "Could not read this page" in str(caught.value)


def test_unreachable_site_gives_a_plain_reason():
    def down(url):
        raise SourceReadError("the site could not be reached (ConnectError)", "network")

    with pytest.raises(SourceReadError) as caught:
        extract_source_product("https://nowhere.example/p/1", fetcher=down, renderer=down)
    assert caught.value.reason == "network" and "could not be reached" in str(caught.value)


def test_non_product_page_is_refused_with_a_clear_message():
    page = "<html><head><title>Help centre and returns policy</title></head><body><h1>Help centre</h1></body></html>"
    with pytest.raises(SourceReadError) as caught:
        extract_source_product("https://www.example.in/help", fetcher=lambda u: (200, page, u), renderer=lambda u: (200, page, u))
    assert caught.value.reason == "no_product_data" and "does not look like a single product page" in str(caught.value)


def test_results_are_cached_per_url():
    count = {"n": 0}

    def fetcher(url):
        count["n"] += 1
        return 200, JSON_LD_PAGE, url

    extract_source_product("https://www.croma.com/m36/p/1", fetcher=fetcher, renderer=fetcher)
    extract_source_product("https://www.croma.com/m36/p/1", fetcher=fetcher, renderer=fetcher)
    assert count["n"] == 1


def test_parsed_result_is_json_serialisable():
    data = extract_source_product("https://www.croma.com/m36/p/1", fetcher=lambda u: (200, JSON_LD_PAGE, u), renderer=lambda u: (200, "", u))
    json.dumps(data)


def test_empty_app_shell_is_not_mistaken_for_a_product():
    shell = ('<html><head><title>Buy undefined - | Online Electronic Shopping Store in India</title>'
             '<meta property="og:type" content="product"><meta property="og:title" content="Buy undefined - | Online Electronic Shopping Store in India">'
             '</head><body><div id="root"></div></body></html>')
    product = parse_product_html(shell, "https://www.reliancedigital.in/product/x")
    assert product.title is None and product.structured is False
    with pytest.raises(SourceReadError) as caught:
        extract_source_product("https://www.reliancedigital.in/product/x", fetcher=lambda u: (200, shell, u), renderer=lambda u: (200, shell, u))
    assert caught.value.reason == "no_product_data"


def test_declared_product_page_without_a_price_in_markup_is_still_read_by_name():
    page = ('<html><head><meta property="og:type" content="product">'
            '<meta property="og:title" content="Maybelline New York Fit Me Foundation 125 Nude Beige - 30 ml"></head><body></body></html>')
    data = extract_source_product("https://www.tatacliq.com/x/p-mp000000020418957", fetcher=lambda u: (200, page, u), renderer=lambda u: (200, "", u))
    assert data["title"].startswith("Maybelline New York Fit Me") and data["price"] is None and data["declared_product"] is True


def test_title_only_page_is_accepted_on_a_known_stores_product_url_but_not_elsewhere():
    page = '<html><head><meta property="og:title" content="Maybelline New York Fit Me Foundation 125 Nude Beige - 30 ml"></head><body></body></html>'
    ok = extract_source_product("https://www.tatacliq.com/maybelline-fit-me/p-mp000000020418957", fetcher=lambda u: (200, page, u), renderer=lambda u: (200, "", u))
    assert ok["title"].startswith("Maybelline New York") and ok["price"] is None
    with pytest.raises(SourceReadError):
        extract_source_product("https://www.tatacliq.com/women-clothing", fetcher=lambda u: (200, page, u), renderer=lambda u: (200, "", u))
    with pytest.raises(SourceReadError):
        extract_source_product("https://blog.example.com/best-foundations", fetcher=lambda u: (200, page, u), renderer=lambda u: (200, "", u))


def test_rendered_page_title_beats_the_generic_site_title_of_an_app_shell():
    shell = '<html><head><meta property="og:title" content="Online Fashion &amp; Lifestyle Shopping for Women, Men &amp; Kids in India"></head><body></body></html>'
    rendered = '<html><head><meta property="og:title" content="Maybelline New York Fit Me Foundation 125 Nude Beige - 30 ml"></head><body></body></html>'
    data = extract_source_product("https://www.tatacliq.com/maybelline-fit-me/p-mp000000020418957",
                                  fetcher=lambda u: (200, shell, u), renderer=lambda u: (200, rendered, u))
    assert data["title"].startswith("Maybelline New York Fit Me")
