"""Is this product link one a person can click and land on the product?

Every match shown in the app is opened once with an ordinary page request, as a visitor's browser would, before it
is listed. The answer is one of:

* ``ok``        the page opened and shows this product
* ``dead``      the store says it does not exist (404/410), the link redirects to the store's home page or to a
                search / category page, the page is a "not found" page, or it shows a different product.
                Dead links are never shown.
* ``unchecked`` the store did not let an automated request through (403/429/CAPTCHA/5xx), did not answer in time,
                or builds the page with script so nothing could be read. These are shown (they are normally fine
                in a browser) but without the "link checked" tick.

Nothing here tries to get past a CAPTCHA or an access wall: such a page is reported as ``unchecked`` and left alone.
Set ``LINK_CHECK=false`` to switch the check off. Answers are kept for ten minutes.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .source_extractor import is_acceptable_source_url

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/122.0.0.0 Safari/537.36")
MAX_BYTES = 400_000
TTL_SECONDS = 600
MAX_ENTRIES = 500

_lock = threading.Lock()
_cache: dict[str, tuple[float, "LinkCheck"]] = {}

_SOFT_404 = re.compile(
    r"\b(?:404|page not found|not found|page (?:does ?n[o']t|does not) exist|no longer (?:exists?|available on)|"
    r"(?:can[' ]?t|could ?n[o']t|cannot) (?:find|be found)|looking for (?:is )?(?:not|no longer)|link (?:is )?(?:broken|expired)|"
    r"oops|something went wrong|product (?:is )?(?:removed|discontinued|not found))\b", re.I)
_BLOCK_PAGE = re.compile(
    r"captcha|robot check|are you a (?:human|robot)|access denied|just a moment|attention required|security check|unusual traffic|"
    r"verify you are|request blocked|forbidden", re.I)
_LISTING_PATH = re.compile(r"/(?:search|s|c|category|categories|collections?|shop|browse|list|listing|catalog|results)(?:/|$)", re.I)
_SPEC_LIKE = re.compile(r"^\d+(?:\.\d+)?[a-z]{0,3}$")
_SHOP_WORDS = {"buy", "online", "price", "best", "india", "offers", "with", "and", "the", "for", "new", "from", "free", "delivery"}


@dataclass
class LinkCheck:
    state: str                  # "ok" | "dead" | "unchecked"
    reason: str
    final_url: str | None = None
    status_code: int | None = None


def enabled() -> bool:
    return os.getenv("LINK_CHECK", "true").strip().casefold() not in {"0", "false", "no", "off"}


def clear() -> None:
    with _lock:
        _cache.clear()


class _Refused(httpx.RequestError):
    """A redirect led somewhere that is not a public web address."""


def _guard(request: httpx.Request) -> None:
    if not is_acceptable_source_url(str(request.url)):
        raise _Refused("not a public web address", request=request)


def _client() -> httpx.Client:
    """One short-lived client per check; tests replace this with a mocked transport."""
    return httpx.Client(
        timeout=httpx.Timeout(7.0, connect=4.0), follow_redirects=True, max_redirects=6,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5", "Accept-Language": "en-IN,en;q=0.9"},
        event_hooks={"request": [_guard]},
    )


# ------------------------------------------------------------------ page reading

def _product_names(soup: BeautifulSoup) -> tuple[list[str], bool]:
    """(names declared by schema.org Product data, whether the page declares a product at all)."""
    names: list[str] = []
    declared = False

    def walk(node):
        nonlocal declared
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kind = node.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if any(isinstance(k, str) and k.lower() in {"product", "productgroup", "individualproduct"} for k in kinds):
                declared = True
                if isinstance(node.get("name"), str):
                    names.append(node["name"])
            for value in node.values():
                if isinstance(value, (list, dict)):
                    walk(value)

    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        try:
            walk(json.loads(script.string or script.get_text() or ""))
        except ValueError:
            continue
    og_type = soup.find("meta", attrs={"property": "og:type"})
    if og_type and "product" in (og_type.get("content") or "").lower():
        declared = True
    if soup.find(attrs={"itemprop": "price"}) or soup.find("meta", attrs={"property": re.compile(r"^(?:product|og):price", re.I)}):
        declared = True
    return names, declared


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3 and w not in _SHOP_WORDS]


def judge(requested_url: str, final_url: str, status: int, html: str, title: str | None = None) -> LinkCheck:
    """The decision, separate from the network so it can be tested on saved pages."""
    if status in (404, 410):
        return LinkCheck("dead", f"the store says this page does not exist (HTTP {status})", final_url, status)
    if status in (401, 403, 407, 429, 202) or status >= 500:
        return LinkCheck("unchecked", f"the store did not allow an automated check (HTTP {status})", final_url, status)
    if status >= 400:
        return LinkCheck("dead", f"the store rejects this address (HTTP {status})", final_url, status)
    if status >= 300 or status < 200:
        return LinkCheck("unchecked", f"unexpected answer (HTTP {status})", final_url, status)

    original, landed = urlparse(requested_url), urlparse(final_url)
    moved = (original.path.rstrip("/"), original.hostname) != (landed.path.rstrip("/"), landed.hostname)
    if moved and landed.path.strip("/") == "" and original.path.strip("/") != "":
        return LinkCheck("dead", "the link redirects to the store's home page", final_url, status)
    if moved and (_LISTING_PATH.search(landed.path) or "q=" in (landed.query or "")) and not _LISTING_PATH.search(original.path):
        return LinkCheck("dead", "the link redirects to a search or category page", final_url, status)

    soup = BeautifulSoup(html or "", "html.parser")
    page_title = (soup.title.get_text(" ", strip=True) if soup.title else "")
    og = soup.find("meta", attrs={"property": "og:title"})
    og_title = (og.get("content") or "").strip() if og else ""
    h1 = soup.find("h1")
    heading = h1.get_text(" ", strip=True) if h1 else ""
    product_names, declares_product = _product_names(soup)
    headline = " ".join([page_title, og_title, heading])

    if _BLOCK_PAGE.search(page_title) or _BLOCK_PAGE.search(heading):
        return LinkCheck("unchecked", "the store showed a verification page instead of the product", final_url, status)
    if _SOFT_404.search(page_title) or _SOFT_404.search(heading):
        return LinkCheck("dead", "the page says the product was not found", final_url, status)

    if title:
        wanted = list(dict.fromkeys(_words(title)))
        # the leading words name the product ("samsung galaxy m36"); storage / RAM figures alone must not make a page "the same"
        names = [w for w in wanted if not _SPEC_LIKE.match(w)][:3] or wanted
        # what the page itself says decides; the address is not evidence (a store can keep a slug and show something else)
        seen = set(_words(" ".join([headline, " ".join(product_names)])))
        shared = [w for w in names if w in seen]
        if names and len(shared) >= min(2, len(names)):
            return LinkCheck("ok", "the page opens and shows this product", final_url, status)
        if (product_names or declares_product) and headline.strip() and len(_words(headline)) >= 2:
            return LinkCheck("dead", "the page shows a different product", final_url, status)
        return LinkCheck("unchecked", "the page is built by script, so its product could not be read", final_url, status)

    if declares_product or page_title:
        return LinkCheck("ok", "the page opens", final_url, status)
    return LinkCheck("unchecked", "the page is built by script, so its product could not be read", final_url, status)


def judge_browser(requested_url: str, detail: dict, listed_title: str | None = None) -> LinkCheck:
    """Second opinion for a store that refuses plain requests: what a headless browser saw when it opened the page.

    ``detail`` is one entry of ``scraper_search.js deep`` (status, final_url, title, price, gtin ...).
    """
    from html import escape
    status = str(detail.get("status") or "")
    final_url = detail.get("final_url") or requested_url
    if status in ("http_404", "http_410"):
        return LinkCheck("dead", f"the store says this page does not exist (HTTP {status[5:]})", final_url, int(status[5:]))
    if status != "ok":
        return LinkCheck("unchecked", "the store would not show the page to an automated browser either")
    title = str(detail.get("title") or "")
    declared = any(detail.get(key) for key in ("price", "gtin", "sku", "mpn"))
    product = ('<script type="application/ld+json">' + json.dumps({"@type": "Product", "name": title}) + "</script>") if declared and title else ""
    html = f"<html><head><title>{escape(title)}</title>{product}</head><body></body></html>"
    return judge(requested_url, final_url, 200, html, listed_title)


# ------------------------------------------------------------------ the check

def check(url: str | None, title: str | None = None) -> LinkCheck:
    """Open ``url`` once and say whether it is a working product page for ``title``."""
    if not enabled():
        return LinkCheck("unchecked", "link checking is switched off")
    if not url or not is_acceptable_source_url(url):
        return LinkCheck("dead", "not a valid public web address")
    key = f"{url}\n{title or ''}"
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < TTL_SECONDS:
            return hit[1]
    result = _fetch_and_judge(url, title)
    if result.state != "unchecked" or result.status_code is not None:       # network hiccups are not remembered
        with _lock:
            if len(_cache) >= MAX_ENTRIES:
                _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
            _cache[key] = (time.time(), result)
    return result


def _fetch_and_judge(url: str, title: str | None) -> LinkCheck:
    try:
        with _client() as client:
            with client.stream("GET", url) as response:
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= MAX_BYTES:
                        break
                html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                return judge(url, str(response.url), response.status_code, html, title)
    except httpx.TooManyRedirects:
        return LinkCheck("dead", "the link keeps redirecting and never reaches a page")
    except _Refused:
        return LinkCheck("dead", "the link leads away from the public web")
    except httpx.TimeoutException:
        return LinkCheck("unchecked", "the store did not answer in time")
    except httpx.HTTPError:
        return LinkCheck("unchecked", "could not connect to the store to check")
    except Exception:
        return LinkCheck("unchecked", "could not check this link")
