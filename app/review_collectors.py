"""Collect review text for a product page on any store, as a logged-out visitor would see it.

Layers, fastest first:

1. a store adapter that reads the store's own public review pages
   (Flipkart: its review listing, 10 reviews per request, several sort orders);
2. the reviews a product page declares in schema.org JSON-LD / microdata
   (many stores publish a few; plain HTTP, no browser);
3. optionally, a headless-Chrome pass (``scraper_search.js reviews``) for pages that
   build their reviews with JavaScript.

Only what the store shows without signing in is collected: no login, no CAPTCHA
handling, no hidden endpoints. Amazon is the exception that proves the rule: it shows
the full list only to signed-in users, so Amazon keeps using the product page plus the
configured provider (see ``review_scraper.py``).
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .review_signals import parse_review_date
from .proc import hidden_kwargs

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/122.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9", "Accept": "text/html,application/xhtml+xml"}
FLIPKART_SORTS = ("MOST_RECENT", "NEGATIVE_FIRST", "POSITIVE_FIRST", "MOST_HELPFUL")
BASE_DIR = Path(__file__).resolve().parent.parent
FLIPKART_MAX_PAGES = 12            # beyond this the listing is empty for a logged-out visitor


def default_limit() -> int:
    try:
        return max(5, min(int(os.getenv("REVIEW_FETCH_LIMIT", "150")), 1000))
    except ValueError:
        return 150


@dataclass
class CollectedReviews:
    reviews: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""
    reported_total: int | None = None          # what the store says exists (not what could be read)
    issues: list[str] = field(default_factory=list)
    seconds: float = 0.0


Fetcher = Callable[[str], "tuple[int | None, str]"]


def _get(url: str, params: dict | None = None, timeout: float = 15.0) -> tuple[int | None, str]:
    """GET shared with other readers of the same URL for a couple of minutes (see http_cache)."""
    from .http_cache import fetch_once

    def load() -> tuple[int | None, str]:
        try:
            with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=timeout) as client:
                response = client.get(url, params=params)
            return response.status_code, response.text
        except httpx.HTTPError:
            return None, ""

    return fetch_once(("get", url, tuple(sorted((params or {}).items()))), load)


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(str(value or ""))).strip()


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _rating(value: object) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if 1 <= number <= 5 and number == int(number) else (int(round(number)) if 1 <= number <= 5 else None)


# ----------------------------------------------------------------------------- generic: schema.org

def _walk_reviews(node: Any, out: list[dict]) -> None:
    if isinstance(node, dict):
        if node.get("reviewBody") or (str(node.get("@type", "")).endswith("Review") and node.get("description")):
            out.append(node)
        for value in node.values():
            _walk_reviews(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_reviews(value, out)


def reviews_from_markup(html: str) -> tuple[list[dict], int | None]:
    """Reviews declared in JSON-LD or microdata, plus the store's stated review count."""
    soup = BeautifulSoup(html, "html.parser")
    found: list[dict] = []
    total: int | None = None
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text() or "{}")
        except ValueError:
            continue
        nodes: list[dict] = []
        _walk_reviews(data, nodes)
        for node in nodes:
            rating = node.get("reviewRating")
            value = rating.get("ratingValue") if isinstance(rating, dict) else rating
            author = node.get("author")
            found.append({
                "text": _clean_text(node.get("reviewBody") or node.get("description")),
                "title": _clean_text(node.get("name")),
                "rating": _rating(value),
                "date": _clean_text(node.get("datePublished")) or None,
                "author": _clean_text(author.get("name") if isinstance(author, dict) else author) or None,
                "verified_purchase": False,
            })
        aggregate = data.get("aggregateRating") if isinstance(data, dict) else None
        if isinstance(aggregate, dict):
            try:
                total = int(float(str(aggregate.get("reviewCount") or aggregate.get("ratingCount")).replace(",", "")))
            except (TypeError, ValueError):
                pass
    for scope in soup.select('[itemprop="review"]'):
        body = scope.select_one('[itemprop="reviewBody"]')
        rating = scope.select_one('[itemprop="ratingValue"]')
        date = scope.select_one('[itemprop="datePublished"]')
        if body and _clean_text(body.get_text(" ")):
            found.append({
                "text": _clean_text(body.get_text(" ")), "title": "",
                "rating": _rating(rating.get("content") or rating.get_text()) if rating else None,
                "date": (date.get("content") or date.get_text()).strip() if date else None,
                "author": None, "verified_purchase": False,
            })
    return [r for r in found if len(r["text"]) >= 4], total


# ----------------------------------------------------------------------------- Flipkart

def _flipkart_review_base(product_html: str, product_url: str) -> tuple[str, dict[str, str]] | None:
    """The product's own review listing: (path-url, query params such as pid)."""
    match = re.search(r'href="([^"]*product-reviews/[^"]+)"', product_html)
    if match:                                                   # the page links to it: use that exactly
        link = urljoin(product_url, html_lib.unescape(match.group(1)))
        parsed = urlparse(link)
        params = dict(parse_qsl(parsed.query))
        params.pop("page", None)
        return urlunparse(parsed._replace(query="", fragment="")), params
    parsed = urlparse(product_url)
    if "/p/" not in parsed.path:
        return None
    query = dict(parse_qsl(parsed.query))
    # The listing's own product id: the pasted link's ?pid=, else the page's schema.org sku. Other pids on the
    # page belong to recommended products, whose reviews would be the wrong product's.
    pid = query.get("pid") or (re.search(r'"sku"\s*:\s*"([A-Z0-9]{12,20})"', product_html or "") or [None, None])[1]
    path = parsed.path.replace("/p/", "/product-reviews/", 1)
    # The /itm... id in the path identifies the listing by itself; the pid is only a hint, so the review pages
    # can be requested at the same moment as the product page instead of after it.
    return urlunparse(parsed._replace(path=path, query="", fragment="")), ({"pid": pid} if pid else {})


def _flipkart_state(page_html: str) -> dict | None:
    match = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});\s*</script>", page_html, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except ValueError:
        return None


def _flipkart_nodes(node: Any, out: list[dict]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "ProductReviewValue" and (node.get("text") or node.get("title")):
            out.append(node)
        for value in node.values():
            _flipkart_nodes(value, out)
    elif isinstance(node, list):
        for value in node:
            _flipkart_nodes(value, out)


def parse_flipkart_page(page_html: str) -> list[dict]:
    state = _flipkart_state(page_html)
    if not state:
        return []
    nodes: list[dict] = []
    _flipkart_nodes(state, nodes)
    out = []
    for node in nodes:
        text = _clean_text(node.get("text")) or _clean_text(node.get("title"))
        if len(text) < 2:
            continue
        created = _clean_text(node.get("created"))
        parsed = parse_review_date(created)
        out.append({
            "id": node.get("id"), "text": text, "title": _clean_text(node.get("title")),
            "rating": _rating(node.get("rating")), "date": created or None,
            "timestamp": parsed.isoformat() if parsed else None,
            "author": _clean_text(node.get("author")) or None,
            "verified_purchase": str(node.get("certifiedBuyer")).casefold() == "true",
            "helpful": int(node["helpfulCount"]) if str(node.get("helpfulCount", "")).isdigit() else 0,
        })
    return out


def collect_flipkart(product_url: str, product_html: str | None, limit: int, get: Callable = _get) -> CollectedReviews:
    started = time.time()
    result = CollectedReviews(source="Flipkart review pages")
    html = product_html or ""

    def attempt(page_html: str) -> list[dict]:
        base = _flipkart_review_base(page_html, product_url)
        if not base:
            return []
        review_url, params = base
        pages_recent = min(FLIPKART_MAX_PAGES, max(1, -(-limit // 10)))
        jobs = [("MOST_RECENT", page) for page in range(1, pages_recent + 1)]
        if limit > 30:      # the extremes (worst / best / most helpful) are where fake and genuine reviews cluster
            jobs += [(sort, page) for sort in FLIPKART_SORTS[1:] for page in (1, 2, 3)]

        def fetch(job: tuple[str, int]) -> list[dict]:
            sort, page = job
            status, body = get(review_url, {**params, "page": page, "sortOrder": sort, "marketplace": "FLIPKART"})
            return parse_flipkart_page(body) if status == 200 else []

        with ThreadPoolExecutor(max_workers=8) as pool:
            return [review for batch in pool.map(fetch, jobs) for review in batch]

    collected = attempt(html)
    if not collected and not html:               # the link alone was not enough: read the product page for its own ids and retry once
        _, html = get(product_url)
        collected = attempt(html)
    if not html and not collected:
        result.issues.append("Flipkart: this page has no review-listing link.")
    seen: set[str] = set()
    for review in collected:
        ident = review.get("id") or _key(review["text"])
        if ident in seen:
            continue
        seen.add(ident)
        result.reviews.append(review)
    result.reviews = result.reviews[:limit]
    if not result.reviews and not result.issues:
        result.issues.append("Flipkart returned no review text for this product.")
    markup_total = re.search(r'"reviewCount"\s*:\s*(\d+)', html)
    result.reported_total = int(markup_total.group(1)) if markup_total else None
    result.seconds = round(time.time() - started, 1)
    return result


# ----------------------------------------------------------------------------- headless Chrome

def _run_browser_collector(url: str, limit: int, timeout: int = 50) -> dict | None:
    """scraper_reviews.js: scroll the page, press its review controls, read what the visitor sees."""
    kwargs: dict[str, Any] = {"capture_output": True, "encoding": "utf-8", "errors": "replace", "timeout": timeout, "cwd": str(BASE_DIR)}
    kwargs.update(hidden_kwargs())
    try:
        proc = subprocess.run(["node", "scraper_reviews.js", json.dumps({"url": url, "limit": limit, "budget_ms": (timeout - 10) * 1000})], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip().startswith("{")]
    try:
        return json.loads(lines[-1]) if proc.returncode == 0 and lines else None
    except ValueError:
        return None


def _from_browser(payload: dict) -> list[dict]:
    out = []
    for item in payload.get("reviews", []):
        text = _clean_text(item.get("text"))
        if len(text) < 4:
            continue
        parsed = parse_review_date(item.get("date"))
        out.append({"text": text, "title": "", "rating": _rating(item.get("rating")), "date": item.get("date"),
                    "timestamp": parsed.isoformat() if parsed else None, "author": item.get("author"),
                    "verified_purchase": bool(item.get("verified_purchase"))})
    return out


# ----------------------------------------------------------------------------- entry point

def collect_reviews(url: str, html: str | None = None, limit: int | None = None, get: Callable = _get,
                    use_browser: bool = False, browser_timeout: int = 50) -> CollectedReviews:
    """Reviews a logged-out visitor can read on ``url``. Never raises for a store problem.

    ``use_browser`` adds the slower headless-Chrome pass when the fast layers found fewer than 3 reviews.
    """
    started = time.time()
    limit = limit or default_limit()
    host = (urlparse(url).hostname or "").lower()
    out = CollectedReviews()

    def merge(extra: list[dict], label: str) -> None:
        known = {_key(r["text"]) for r in out.reviews}
        fresh = [r for r in extra if _key(r["text"]) not in known]
        if fresh:
            out.reviews += fresh
            out.source = (out.source + " + " if out.source else "") + label

    try:
        if "flipkart." in host:
            out = collect_flipkart(url, html, limit, get)
        if len(out.reviews) < 3:                                   # adapter gave (almost) nothing: try the page's own markup
            if html is None:
                _, html = get(url)
            markup, total = reviews_from_markup(html or "")
            merge(markup, "reviews declared in the product page markup")
            out.reported_total = out.reported_total or total
        if use_browser and len(out.reviews) < 3:
            scanned = _run_browser_collector(url, limit, browser_timeout)
            if scanned is None:
                out.issues.append("The browser review scan did not finish.")
            else:
                merge(_from_browser(scanned), "page scan in Chrome (" + ", ".join(scanned.get("via") or ["nothing found"]) + ")")
                out.issues += [i for i in scanned.get("issues", []) if i not in out.issues]
                out.reported_total = out.reported_total or scanned.get("reported_total")
    except Exception as exc:                                       # a store quirk must not lose the analysis
        out.issues.append(f"Review collection hit a problem ({type(exc).__name__}).")
    out.reviews = out.reviews[:limit]
    out.seconds = round(time.time() - started, 1)
    return out
