"""Enhanced cross-store product search.

Adds precise queries, more stores and attribute-aware verification *on top of*
the legacy Amazon <-> Flipkart search, which is left exactly as it was:

* If the product is on Amazon or Flipkart, those two stores are skipped here
  (the legacy scraper already covers them well) and the other stores are added.
* For any other source store, Amazon and Flipkart are searched again with a
  precise query and every other applicable store is added.

Pipeline: build queries (product_identity) -> search stores (scraper_search.js)
-> score and drop mismatches (matcher) -> open the best few pages for rating,
image and identifiers -> add image similarity -> return legacy-shaped dicts.

Nothing is invented: a store that blocks or returns nothing is reported as such.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .image_matcher import compute_image_similarity
from .matcher import OUT_OF_STOCK_FLAG, score_candidate
from .product_identity import ProductIdentity, build_queries, parse_identity
from .store_registry import store_for_url

BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = BASE_DIR / "config" / "stores.json"
LEGACY_PAIR = {"Amazon", "Flipkart"}


def enabled() -> bool:
    return os.getenv("ENHANCED_MATCHING", "true").strip().casefold() not in {"0", "false", "no", "off"}


def _registry() -> dict[str, dict]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["stores"]
    except (OSError, ValueError, KeyError):
        return {}


def _host(url: str | None) -> str:
    return (urlparse(url or "").hostname or "").lower().removeprefix("www.")


def select_stores(source_platform: str, source_url: str, category: str, limit: int | None = None) -> list[str]:
    """Stores worth searching for this product, best category fits first."""
    limit = limit or int(os.getenv("ENHANCED_MAX_STORES", "8") or 8)
    source_host = _host(source_url)
    # "croma.com" / "Reliancedigital" / "Reliance Digital" all name the same store: never search the source store itself.
    source_names = {(source_platform or "").casefold().replace(" ", "")}
    source_store = store_for_url(source_url)
    if source_store:
        source_names.add(source_store.casefold().replace(" ", ""))
    ranked: list[tuple[int, int, str]] = []
    for name, config in _registry().items():
        if config.get("enabled") is False:
            continue
        if config["host"] in source_host or name.casefold().replace(" ", "") in source_names:
            continue
        if source_platform in LEGACY_PAIR and name in LEGACY_PAIR:
            continue                                   # legacy search already handles this pair
        categories = config.get("categories", [])
        if category in categories:
            rank = 0
        elif "all" in categories:
            rank = 1
        elif category == "general" and "general" in categories:
            rank = 2
        else:
            continue
        # the two biggest marketplaces are searched whenever the source is somewhere else
        ranked.append((0 if name in LEGACY_PAIR else 1, rank, name))
    ranked.sort(key=lambda item: item[:2])
    return [name for _, _, name in ranked][:limit]


def _run_node(mode: str, payload: dict, timeout: int) -> dict | None:
    kwargs = {"capture_output": True, "encoding": "utf-8", "errors": "replace", "timeout": timeout, "cwd": str(BASE_DIR)}
    if sys.platform == "win32":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        kwargs["startupinfo"] = startup
        kwargs["creationflags"] = 0x08000000           # CREATE_NO_WINDOW
    try:
        proc = subprocess.run(["node", "scraper_search.js", mode, json.dumps(payload)], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip().startswith("{")]
    try:
        return json.loads(lines[-1]) if lines else None
    except ValueError:
        return None


def _page_image(url: str) -> str | None:
    """The page's own og:image / twitter:image, read with one plain GET (search cards often lack a photo)."""
    try:
        import httpx
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
        with httpx.Client(timeout=6.0, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text[:400_000], "html.parser")
        for attrs in ({"property": "og:image"}, {"name": "twitter:image"}, {"property": "og:image:secure_url"}):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content", "").strip():
                from urllib.parse import urljoin
                return urljoin(url, tag["content"].strip())
    except Exception:
        return None
    return None


def _canonical(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def enhanced_competitor_matches(
    title: str | None,
    brand: str | None,
    price: float | None,
    image_url: str | None,
    source_platform: str,
    source_url: str,
    source_gtin: str | None = None,
    source_mpn: str | None = None,
) -> tuple[list[dict], list[dict], ProductIdentity | None]:
    """Return (matches in legacy competitor_matches shape, store attempts, identity).

    ``source_gtin`` / ``source_mpn`` (EAN/UPC and manufacturer part or model number of the source listing) are
    optional: when given they are compared with the identifiers read from each opened candidate page.
    """
    if not enabled() or not title:
        return [], [], None

    identity = parse_identity(title, brand)
    stores = select_stores(source_platform, source_url, identity.category)
    queries = build_queries(identity)
    if not stores or not queries:
        return [], [], identity

    searched = _run_node(
        "search", {"queries": queries, "stores": stores, "per_store": 8},
        timeout=int(os.getenv("ENHANCED_SEARCH_TIMEOUT", "100")),
    )
    if not searched:
        return [], [{"platform": "Multi-store search", "status": "unavailable (search helper did not run)", "result_count": 0}], identity

    scored: list[tuple[dict, object]] = []
    seen: set[str] = set()
    rejected_by_store: dict[str, int] = {}
    for item in searched.get("matches", []):
        if not item.get("url") or not item.get("title"):
            continue
        key = _canonical(item["url"])
        if key in seen:
            continue
        seen.add(key)
        result = score_candidate(identity, item["title"], item["url"], price, item.get("price"))
        if result.status == "rejected":
            rejected_by_store[item["platform"]] = rejected_by_store.get(item["platform"], 0) + 1
            continue
        scored.append((item, result))

    # best three per store, then best twelve overall
    scored.sort(key=lambda pair: pair[1].score, reverse=True)
    per_store: dict[str, int] = {}
    kept: list[tuple[dict, object]] = []
    for item, result in scored:
        if per_store.get(item["platform"], 0) >= 3:
            continue
        per_store[item["platform"]] = per_store.get(item["platform"], 0) + 1
        kept.append((item, result))
    kept = kept[:12]

    # open the strongest pages for rating / image / identifiers
    deep: dict[str, dict] = {}
    to_open = [(i, r) for i, r in kept if r.score >= 0.6][:4]
    if to_open:
        opened = _run_node("deep", {"urls": [{"platform": i["platform"], "url": i["url"]} for i, _ in to_open]}, timeout=60)
        for detail in (opened or {}).get("details", []):
            if detail.get("status") == "ok":
                deep[_canonical(detail["url"])] = detail

    def finish(pair):
        item, result = pair
        detail = deep.get(_canonical(item["url"]), {})
        cand_image = item.get("image_url") or detail.get("image_url") or (_page_image(item["url"]) if image_url else None)
        image_similarity = compute_image_similarity(image_url, cand_image) if (image_url and cand_image) else None
        final = score_candidate(
            identity, item["title"], item["url"], price, item.get("price") or detail.get("price"), image_similarity,
            source_gtin=source_gtin, candidate_gtin=detail.get("gtin"),
            source_mpn=source_mpn, candidate_mpn=detail.get("mpn"),
        )
        # The re-score above builds fresh flags, so the stock note must be added to *this* result.
        if item.get("out_of_stock") and OUT_OF_STOCK_FLAG not in final.flags:
            final.flags.append(OUT_OF_STOCK_FLAG)
        return item, detail, cand_image, image_similarity, final

    with ThreadPoolExecutor(max_workers=6) as pool:
        finished = list(pool.map(finish, kept))

    matches: list[dict] = []
    for item, detail, cand_image, image_similarity, final in finished:
        if final.status == "rejected":
            rejected_by_store[item["platform"]] = rejected_by_store.get(item["platform"], 0) + 1   # e.g. identifier veto after opening the page
            continue
        matches.append({
            "platform": item["platform"],
            "title": item["title"],
            "url": item["url"],
            "price": item.get("price") or detail.get("price"),
            "rating": detail.get("rating") or item.get("rating"),
            "review_count": detail.get("review_count") or item.get("review_count"),
            "review_sample_count": 0,
            "reviews": [],
            "image_url": cand_image,
            "image_similarity": image_similarity,
            "match_source": "enhanced",
            "in_stock": None if item.get("out_of_stock") is None else not item["out_of_stock"],   # None = the store did not say
            "condition": final.condition,                     # "new" | "refurbished" | "open_box" | "used" (from the listing title)
            "match_score": final.score,
            "match_status": final.status,
            "match_reasons": final.reasons,
            "match_flags": final.flags,
            "title_similarity": final.title_similarity,
            "gtin": detail.get("gtin"),
            "mpn": detail.get("mpn"),
        })

    attempts = []
    for report in searched.get("stores", []):
        name = report["platform"]
        attempts.append({
            "platform": name,
            "status": ("results collected" if report["status"] == "ok" else report["status"]),
            "result_count": sum(1 for m in matches if m["platform"] == name),
            "raw_results": report.get("result_count", 0),
            "filtered_out": rejected_by_store.get(name, 0),
            "query": report.get("query"),
        })
    return matches, attempts, identity
