"""Read-only view of config/stores.json: which store a URL or platform label belongs to.

``app/multistore.py`` and ``scraper_search.js`` read the same file for searching;
this module only answers "which known store is this link / label?" so the rest of
the app can name a store the same way everywhere ("Croma", not "croma.com").
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "stores.json"
LEGACY_HOSTS = {"amazon": "Amazon", "flipkart": "Flipkart"}


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["stores"]
    except (OSError, ValueError, KeyError):
        return {}


def host_of(url: str | None) -> str:
    return (urlparse(url or "").hostname or "").lower().rstrip(".").removeprefix("www.")


def store_for_url(url: str | None) -> str | None:
    """Registry name for a link ("https://www.croma.com/..." -> "Croma"), else None."""
    host = host_of(url)
    if not host:
        return None
    for name, config in registry().items():
        domain = str(config.get("host", "")).lower()
        if domain and (host == domain or host.endswith("." + domain)):
            return name
    return None


def canonical_store(platform: str | None, url: str | None = None) -> str:
    """One display name per store. Amazon/Flipkart keep the legacy labels untouched."""
    host = host_of(url)
    for key, label in LEGACY_HOSTS.items():
        if key in host:
            return label
    by_url = store_for_url(url)
    if by_url:
        return by_url
    label = (platform or "").strip()
    if label:
        lowered = label.casefold().replace(" ", "")
        for name in registry():
            if name.casefold().replace(" ", "") == lowered:
                return name
        return label
    return host or "Unknown store"


def same_store(platform_a: str | None, url_a: str | None, platform_b: str | None, url_b: str | None) -> bool:
    """True when two listings belong to one store (by canonical name or by host)."""
    if canonical_store(platform_a, url_a).casefold() == canonical_store(platform_b, url_b).casefold():
        return True
    host_a, host_b = host_of(url_a), host_of(url_b)
    return bool(host_a and host_a == host_b)


def looks_like_product_url(url: str | None) -> bool:
    """True when the link is on a known store and has that store's product-page path pattern ("/p-", "/dp/" ...)."""
    name = store_for_url(url)
    if not name:
        return False
    parsed = urlparse(url or "")
    target = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    return any(pattern in target for pattern in registry()[name].get("link_contains", []) if pattern != "myntra.com/")
