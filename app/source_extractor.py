"""Read a product from *any* store link, not only the ones scraper.js knows by name.

scraper.js has CSS selectors for Amazon and Flipkart (and a few more). For every
other store this module reads what the page itself declares about the product:

1. schema.org JSON-LD (``Product`` / ``ProductGroup`` with ``offers``);
2. schema.org microdata (``itemscope itemtype=".../Product"``);
3. OpenGraph / ``product:*`` meta tags;
4. ``<h1>`` and ``<title>`` as a last resort for the name.

It returns title, price, currency, brand, image, GTIN/MPN/SKU, rating and
availability - only the values the page states. Nothing is guessed or filled in.

Fetching: a plain HTTP request first (fast); if that is refused or carries no
product data, the page is rendered once in headless Chrome (``scraper_search.js
page``). A page that shows a CAPTCHA / "access denied" wall, needs a login, or has
no product on it raises ``SourceReadError`` with a plain reason - no attempt is
made to get around any of that.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .proc import hidden_kwargs
from .store_registry import looks_like_product_url

BASE_DIR = Path(__file__).resolve().parent.parent
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
CACHE_TTL_SECONDS = 600
_CACHE: dict[str, tuple[float, dict]] = {}

BLOCK_RE = re.compile(
    r"captcha|access denied|are you a robot|unusual traffic|verify you are human|robot check|"
    r"enter the characters you see|request blocked|pardon our interruption|checking your browser",
    re.I,
)
LOGIN_RE = re.compile(r"\b(?:sign in|log in|login)\b.{0,60}\b(?:to continue|to view|to see|required)\b", re.I)


class SourceReadError(RuntimeError):
    """The page could not be turned into a product. ``reason`` is a stable short code."""

    def __init__(self, message: str, reason: str = "unreadable"):
        super().__init__(message)
        self.reason = reason


@dataclass
class PageProduct:
    title: str | None = None
    price: float | None = None
    currency: str | None = None
    brand: str | None = None
    image_url: str | None = None
    gtin: str | None = None
    mpn: str | None = None
    sku: str | None = None
    rating: float | None = None
    review_count: int | None = None
    availability: str | None = None          # raw text as the page states it
    in_stock: bool | None = None             # True / False when the page says so, None when it does not
    sources: list[str] = field(default_factory=list)   # which markup the values came from
    structured: bool = False                 # a JSON-LD / microdata Product (or og:type=product with a price) was present
    declared_product: bool = False           # og:type=product: the page says it is a product even if it shows no price in markup
    final_url: str | None = None

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in (
            "title", "price", "currency", "brand", "image_url", "gtin", "mpn", "sku", "rating",
            "review_count", "availability", "in_stock", "sources", "structured", "declared_product", "final_url")}


# --------------------------------------------------------------------------- URL gate

def is_acceptable_source_url(url: str | None) -> bool:
    """Any public http(s) link may be analysed; local and private addresses may not.

    Known stores pass through ``engines.is_valid_ecommerce_url`` first (unchanged);
    this is the wider gate for every other store.
    """
    try:
        parsed = urlparse(re.sub(r"\s+", "", url or ""))
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not host or "." not in host and ":" not in host:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home", ".corp")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast or address.is_unspecified)


# --------------------------------------------------------------------------- value parsing

_PRICE_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_price(value: object) -> float | None:
    """'24,999', '₹24,999.00', 'Rs. 24999', 24999 -> 24999.0; zero, negatives and junk -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip()
    match = _PRICE_NUMBER.search(text)
    if not match:
        return None
    number = match.group(0).replace(",", "")
    try:
        parsed = float(number)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def parse_int(value: object) -> int | None:
    parsed = parse_price(value)
    return int(parsed) if parsed else None


def parse_rating(value: object) -> float | None:
    parsed = parse_price(value)
    return parsed if parsed and 0 < parsed <= 5 else None


def normalise_availability(value: object) -> bool | None:
    """schema.org availability (URL or words) -> True / False / None (unknown or pre-order)."""
    if value is None:
        return None
    text = re.sub(r"[^a-z]", "", str(value).lower().rsplit("/", 1)[-1])
    if not text:
        return None
    if any(word in text for word in ("outofstock", "soldout", "discontinued", "unavailable", "currentlyunavailable", "notavailable")):
        return False
    if any(word in text for word in ("preorder", "presale", "backorder")):
        return None
    if any(word in text for word in ("instock", "limitedavailability", "onlineonly", "instoreonly", "available")):
        return True
    return None


def _clean(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if v), None)
    if isinstance(value, dict):
        value = value.get("name") or value.get("@value") or value.get("url")
    if not isinstance(value, (str, int, float)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def clean_brand(value: object) -> str | None:
    text = _clean(value)
    if not text:
        return None
    text = re.sub(r"^(?:visit the|brand:?)\s*", "", text, flags=re.I)
    return re.sub(r"\s+store$", "", text, flags=re.I).strip(" -|:") or None


def _image(value: object, base_url: str | None) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("url") or value.get("contentUrl")
    text = _clean(value)
    if not text or text.startswith("data:"):
        return None
    return urljoin(base_url or "", text) if base_url else text


def _gtin(node: dict) -> str | None:
    for key in ("gtin13", "gtin", "gtin12", "gtin14", "gtin8", "isbn"):
        text = _clean(node.get(key))
        digits = re.sub(r"\D", "", text or "")
        if len(digits) in (8, 12, 13, 14):
            return digits
    return None


_PLACEHOLDER_TITLE = re.compile(r"\b(?:undefined|null|nan)\b|^\W*(?:home|loading|just a moment|404|page not found)\W*$", re.I)


def clean_title(title: str | None, site_name: str | None = None) -> str | None:
    """Drop "| Croma" / "- Buy online" style tails that stores append to names."""
    text = _clean(title)
    if not text or _PLACEHOLDER_TITLE.search(text):    # an unrendered template, not a product name
        return None
    if site_name:
        text = re.sub(rf"\s*[\|\-–—:]\s*{re.escape(site_name)}\s*$", "", text, flags=re.I)
        text = re.sub(rf"^\s*{re.escape(site_name)}\s*[\|\-–—:]\s*", "", text, flags=re.I)
    text = re.sub(r"\s*[\|\-–—]\s*(?:buy|shop)\b.*$", "", text, flags=re.I)
    tail = re.search(r"\s+online\b(.{0,50})$", text, re.I)
    if tail and tail.start() >= 8:                       # "Buy <name> Online - Store" -> "<name>"
        text = re.sub(r"^\s*buy\s+(?=\S)", "", text[: tail.start()], flags=re.I)
    return text.strip(" -|:") or None


# --------------------------------------------------------------------------- structured data

def _json_nodes(soup: BeautifulSoup) -> list[dict]:
    nodes: list[dict] = []

    def visit(item: object, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, dict):
            nodes.append(item)
            for key, child in item.items():
                if key != "offers" and isinstance(child, (list, dict)):
                    visit(child, depth + 1)

    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text() or ""
        for candidate in (raw, re.sub(r"[\x00-\x1f]+", " ", raw), re.sub(r"<!--.*?-->|/\*.*?\*/", "", raw, flags=re.S)):
            try:
                visit(json.loads(candidate))
                break
            except ValueError:
                continue
    return nodes


def _is_product(node: dict) -> bool:
    kinds = node.get("@type")
    kinds = kinds if isinstance(kinds, list) else [kinds]
    return any(isinstance(kind, str) and kind.rsplit("/", 1)[-1] in {"Product", "ProductGroup", "IndividualProduct", "ProductModel", "Vehicle"} for kind in kinds)


def _offer_list(node: dict) -> list[dict]:
    offers = node.get("offers")
    if offers is None and node.get("hasVariant"):
        for variant in node["hasVariant"] if isinstance(node["hasVariant"], list) else [node["hasVariant"]]:
            if isinstance(variant, dict) and variant.get("offers"):
                offers = variant["offers"]
                break
    out: list[dict] = []
    for offer in offers if isinstance(offers, list) else [offers]:
        if isinstance(offer, dict):
            out.append(offer)
            nested = offer.get("offers")
            for inner in nested if isinstance(nested, list) else [nested]:
                if isinstance(inner, dict):
                    out.append(inner)
    return out


def _offer_price(offer: dict) -> tuple[float | None, str | None]:
    spec = offer.get("priceSpecification")
    spec = spec[0] if isinstance(spec, list) and spec else spec
    for source in (offer, spec if isinstance(spec, dict) else {}):
        for key in ("price", "lowPrice") if source is offer else ("price", "minPrice"):
            price = parse_price(source.get(key))
            if price:
                return price, _clean(source.get("priceCurrency") or offer.get("priceCurrency"))
    return None, None


def _from_jsonld(soup: BeautifulSoup, result: PageProduct, base_url: str | None) -> None:
    products = [node for node in _json_nodes(soup) if _is_product(node) and _clean(node.get("name"))]
    if not products:
        return
    # the node that carries an offer is the one describing the page's own product
    products.sort(key=lambda node: 0 if _offer_list(node) else 1)
    node = products[0]
    result.structured = True
    result.sources.append("json-ld")
    result.title = result.title or clean_title(node.get("name"))
    result.brand = result.brand or clean_brand(node.get("brand") if not isinstance(node.get("brand"), list) else node["brand"][0])
    result.image_url = result.image_url or _image(node.get("image"), base_url)
    result.gtin = result.gtin or _gtin(node)
    result.mpn = result.mpn or _clean(node.get("mpn"))
    result.sku = result.sku or _clean(node.get("sku"))
    aggregate = node.get("aggregateRating")
    if isinstance(aggregate, dict):
        result.rating = result.rating or parse_rating(aggregate.get("ratingValue"))
        result.review_count = result.review_count or parse_int(aggregate.get("reviewCount") or aggregate.get("ratingCount"))
    offers = _offer_list(node)
    for offer in offers:
        price, currency = _offer_price(offer)
        if price and not result.price:
            result.price, result.currency = price, currency or result.currency
            result.availability = _clean(offer.get("availability"))
            result.in_stock = normalise_availability(offer.get("availability"))
        if not result.gtin:
            result.gtin = _gtin(offer)
    if len(offers) > 1 and result.in_stock is False:
        # several sellers: the product is in stock if any offer is
        flags = [normalise_availability(offer.get("availability")) for offer in offers]
        if any(flag is True for flag in flags):
            result.in_stock = True


def _prop(scope, name: str):
    element = scope.find(attrs={"itemprop": name})
    if element is None:
        return None
    for attr in ("content", "href", "src", "value", "datetime"):
        if element.get(attr):
            return element.get(attr)
    return element.get_text(" ", strip=True) or None


def _from_microdata(soup: BeautifulSoup, result: PageProduct, base_url: str | None) -> None:
    scopes = [el for el in soup.find_all(attrs={"itemtype": re.compile(r"schema\.org/(?:Product|ProductGroup|IndividualProduct)\b", re.I)})]
    if not scopes:
        return
    scope = scopes[0]
    name = _prop(scope, "name")
    if not name and not _prop(scope, "price"):
        return
    result.structured = True
    result.sources.append("microdata")
    result.title = result.title or clean_title(name)
    brand = scope.find(attrs={"itemprop": "brand"})
    if brand is not None:
        result.brand = result.brand or clean_brand(_prop(brand, "name") or brand.get("content") or brand.get_text(" ", strip=True))
    result.image_url = result.image_url or _image(_prop(scope, "image"), base_url)
    for key in ("gtin13", "gtin", "gtin12", "gtin14", "gtin8"):
        digits = re.sub(r"\D", "", str(_prop(scope, key) or ""))
        if len(digits) in (8, 12, 13, 14):
            result.gtin = result.gtin or digits
    result.mpn = result.mpn or _clean(_prop(scope, "mpn"))
    result.sku = result.sku or _clean(_prop(scope, "sku"))
    result.price = result.price or parse_price(_prop(scope, "price") or _prop(scope, "lowPrice"))
    result.currency = result.currency or _clean(_prop(scope, "priceCurrency"))
    availability = _prop(scope, "availability")
    if availability and result.in_stock is None:
        result.availability = result.availability or _clean(availability)
        result.in_stock = normalise_availability(availability)
    result.rating = result.rating or parse_rating(_prop(scope, "ratingValue"))
    result.review_count = result.review_count or parse_int(_prop(scope, "reviewCount") or _prop(scope, "ratingCount"))


def _meta(soup: BeautifulSoup, *names: str) -> str | None:
    for name in names:
        element = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if element is not None and _clean(element.get("content")):
            return _clean(element.get("content"))
    return None


def _from_meta(soup: BeautifulSoup, result: PageProduct, base_url: str | None) -> None:
    og_type = (_meta(soup, "og:type") or "").lower()
    site_name = _meta(soup, "og:site_name")
    title = _meta(soup, "og:title", "twitter:title")
    price = parse_price(_meta(soup, "product:price:amount", "og:price:amount", "product:sale_price:amount"))
    if "product" in og_type:
        result.declared_product = True
        if price:
            result.structured = True      # og:type=product alone is also stamped on empty app shells
    used = False
    if title and not result.title:
        result.title, used = clean_title(title, site_name), True
    if price and not result.price:
        result.price, used = price, True
        result.currency = result.currency or _meta(soup, "product:price:currency", "og:price:currency")
    image = _meta(soup, "og:image", "og:image:url", "twitter:image")
    if image and not result.image_url:
        result.image_url, used = _image(image, base_url), True
    brand = _meta(soup, "product:brand", "og:brand")
    if brand and not result.brand:
        result.brand, used = clean_brand(brand), True
    availability = _meta(soup, "product:availability", "og:availability")
    if availability and result.in_stock is None:
        result.availability = result.availability or availability
        result.in_stock = normalise_availability(availability)
        used = True
    if used:
        result.sources.append("opengraph")


def _from_html_text(soup: BeautifulSoup, result: PageProduct) -> None:
    if not result.title:
        heading = soup.find("h1")
        text = _clean(heading.get_text(" ", strip=True)) if heading else None
        if text and len(text) >= 6:
            result.title = text
            result.sources.append("h1")
    if not result.title and soup.title:
        site_name = _meta(soup, "og:site_name")
        text = clean_title(soup.title.get_text(" ", strip=True), site_name)
        if text and len(text) >= 6:
            result.title = text
            result.sources.append("title")


def parse_product_html(html: str, url: str | None = None) -> PageProduct:
    """Pure function: page markup in, the product the page declares out."""
    soup = BeautifulSoup(html or "", "html.parser")
    result = PageProduct(final_url=url)
    _from_jsonld(soup, result, url)
    _from_microdata(soup, result, url)
    _from_meta(soup, result, url)
    _from_html_text(soup, result)           # a bare <h1>/<title> alone is not evidence of a product page
    if result.title:
        result.title = clean_title(result.title, _meta(soup, "og:site_name")) or result.title
    return result


def page_problem(html: str, status: int | None) -> tuple[str, str] | None:
    """(reason code, readable text) when the page is a wall rather than a product page."""
    head = BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)[:2500] if html else ""
    if BLOCK_RE.search(head) and len(head) < 2500:
        return "blocked", "the site showed a CAPTCHA / access-denied page to automated requests"
    if status in (401, 403, 429):
        return "blocked", f"the site refused the request (HTTP {status})"
    if status and status >= 400:
        return f"http_{status}", f"the site answered HTTP {status}"
    if LOGIN_RE.search(head[:600]) and len(head) < 900:
        return "login", "the page asks for a sign-in before it shows the product"
    return None


# --------------------------------------------------------------------------- fetching

def fetch_html(url: str, timeout: float = 12.0) -> tuple[int | None, str, str]:
    """Plain HTTP GET like an ordinary browser visit: (status, html, final_url). Shared with other readers of the same page."""
    from .http_cache import fetch_once
    return fetch_once(("page", url), lambda: _fetch_html_uncached(url, timeout))


def _fetch_html_uncached(url: str, timeout: float = 12.0) -> tuple[int | None, str, str]:
    import httpx

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={
            "User-Agent": BROWSER_USER_AGENT, "Accept-Language": "en-IN,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }) as client:
            response = client.get(url)
            return response.status_code, response.text[:3_000_000], str(response.url)
    except httpx.TimeoutException:
        raise SourceReadError("the site did not answer in time", "timeout")
    except httpx.HTTPError as exc:
        raise SourceReadError(f"the site could not be reached ({type(exc).__name__})", "network")


def render_html(url: str, timeout: int = 50) -> tuple[int | None, str, str]:
    """Render once in headless Chrome and return the final DOM: (status, html, final_url)."""
    kwargs = {"capture_output": True, "encoding": "utf-8", "errors": "replace", "timeout": timeout, "cwd": str(BASE_DIR)}
    kwargs.update(hidden_kwargs())
    try:
        proc = subprocess.run(["node", "scraper_search.js", "page", json.dumps({"url": url})], **kwargs)
    except subprocess.TimeoutExpired:
        raise SourceReadError("the page did not finish loading in time", "timeout")
    except (OSError, subprocess.SubprocessError):
        raise SourceReadError("the headless browser could not be started", "no_browser")
    lines = [line for line in (proc.stdout or "").strip().splitlines() if line.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        raise SourceReadError("the page could not be opened in the headless browser", "browser_error")
    try:
        data = json.loads(lines[-1])
    except ValueError:
        raise SourceReadError("the headless browser returned unreadable output", "browser_error")
    return data.get("status"), data.get("html") or "", data.get("final_url") or url


Fetcher = Callable[[str], "tuple[int | None, str, str]"]


def _complete(product: PageProduct) -> bool:
    return bool(product.title and product.price and product.structured)


def extract_source_product(url: str, fetcher: Fetcher | None = None, renderer: Fetcher | None = None,
                           use_cache: bool = True) -> dict:
    """Read the product behind ``url``. Raises ``SourceReadError`` with a plain reason.

    ``fetcher`` / ``renderer`` are injectable so tests never touch the network.
    """
    now = time.time()
    cached = _CACHE.get(url) if use_cache else None
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return dict(cached[1])

    fetcher = fetcher or fetch_html
    renderer = renderer or render_html
    best = PageProduct(final_url=url)
    problem: tuple[str, str] | None = None
    attempts: list[str] = []

    for label, load in (("http", fetcher), ("browser", renderer)):
        if label == "browser" and os.getenv("SOURCE_BROWSER_FALLBACK", "true").strip().casefold() in {"0", "false", "no", "off"}:
            break
        try:
            status, html, final_url = load(url)
        except SourceReadError as exc:
            problem = problem or (exc.reason, str(exc))
            attempts.append(f"{label}: {exc}")
            continue
        wall = page_problem(html, status)
        parsed = parse_product_html(html, final_url or url) if html else PageProduct(final_url=final_url or url)
        attempts.append(f"{label}: HTTP {status}" if status else f"{label}: no status")
        if wall and not (parsed.title and (parsed.price or parsed.structured)):
            problem = wall
            continue
        problem = None
        # the better-evidenced read (declared product data, then price) leads; the other only fills gaps
        quality = lambda item: (item.structured, bool(item.price), bool(item.title))
        lead, other = (parsed, best) if quality(parsed) >= quality(best) else (best, parsed)   # on a tie the later (rendered) read leads
        merged = PageProduct(final_url=best.final_url)
        for name in ("title", "price", "currency", "brand", "image_url", "gtin", "mpn", "sku", "rating",
                     "review_count", "availability", "in_stock"):
            value = getattr(lead, name)
            setattr(merged, name, value if value not in (None, "") else getattr(other, name))
        merged.sources = sorted(set(best.sources) | set(f"{label}:{s}" for s in parsed.sources))
        merged.structured = best.structured or parsed.structured
        merged.declared_product = best.declared_product or parsed.declared_product
        best = merged
        if _complete(best):
            break

    if not best.title:
        reason, text = problem or ("no_product_data", "no product name could be found on the page")
        raise SourceReadError(f"Could not read this page: {text}.", reason)
    if not (best.price or best.structured or best.declared_product or looks_like_product_url(url)):
        raise SourceReadError(
            "Could not read this page: it does not look like a single product page (no price or product data found).",
            "no_product_data")

    data = best.as_dict()
    data["attempts"] = attempts
    if use_cache:
        _CACHE[url] = (now, dict(data))
    return data


def clear_cache() -> None:
    _CACHE.clear()
