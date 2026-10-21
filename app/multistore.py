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
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .image_matcher import compute_image_similarity
from . import visual_match
from . import link_check
from .matcher import OUT_OF_STOCK_FLAG, score_candidate
from .product_identity import ProductIdentity, build_queries, parse_identity
from .store_registry import store_for_url
from .proc import hidden_kwargs

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


def select_stores(source_platform: str, source_url: str, category: str, limit: int | None = None, brand: str | None = None) -> list[str]:
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
        if config.get("brand") and config["brand"] not in (brand or "").casefold():
            continue                                   # a brand's own shop only sells that brand
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
    kwargs.update(hidden_kwargs())
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


def _run_node_stream(mode: str, payload: dict, timeout: int, on_store) -> dict | None:
    """Like _run_node, but hands every {"store_done": ...} line to ``on_store`` as soon as the helper prints it.

    If the helper is cut off by the timeout, what it had already printed is still returned.
    """
    kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "text": True, "encoding": "utf-8", "errors": "replace", "cwd": str(BASE_DIR)}
    kwargs.update(hidden_kwargs())
    try:
        proc = subprocess.Popen(["node", "scraper_search.js", mode, json.dumps(payload)], **kwargs)
    except OSError:
        return None
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    final: dict | None = None
    stores: list[dict] = []
    matches: list[dict] = []
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if "store_done" in obj:
                stores.append(obj["store_done"])
                matches.extend(obj.get("items", []))
                try:
                    on_store(obj)
                except Exception:                      # showing a partial result must never break the search
                    pass
            else:
                final = obj
        proc.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
    if final is not None:
        return final
    return {"stores": stores, "matches": matches} if stores else None        # cut off: keep what arrived


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
    progress=None,
) -> tuple[list[dict], list[dict], ProductIdentity | None]:
    """Return (matches in legacy competitor_matches shape, store attempts, identity).

    ``source_gtin`` / ``source_mpn`` (EAN/UPC and manufacturer part or model number of the source listing) are
    optional: when given they are compared with the identifiers read from each opened candidate page.
    """
    if not enabled() or not title:
        return [], [], None

    identity = parse_identity(title, brand)
    stores = select_stores(source_platform, source_url, identity.category, brand=identity.brand)
    queries = build_queries(identity)
    if not stores or not queries:
        return [], [], identity

    def shape(item: dict, detail: dict, cand_image, image_similarity, final, link=None) -> dict:
        return {
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
            "relation": final.relation,                       # "exact" | "similar"
            "differences": final.differences,                 # how a similar product differs
            "closeness": final.closeness,
            "link_status": link.state if (link is not None and link_check.enabled()) else None,
            "link_note": link.reason if (link is not None and link_check.enabled()) else None,
        }

    def open_links(pairs: list[tuple[dict, object]]) -> list:
        """One page request per listing, in parallel (answers are cached, so repeats are free)."""
        if not pairs or not link_check.enabled():
            return [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lambda pair: link_check.check(pair[0]["url"], pair[0]["title"]), pairs))

    dead_by_store: dict[str, int] = {}

    # Show each store's matches the moment that store answers (a first look: no page visits, no image check yet).
    shown_matches: list[dict] = []
    shown_attempts: list[dict] = []

    def on_store(obj: dict) -> None:
        if progress is None:
            return
        report = obj["store_done"]
        rejected = 0
        exact_like: list[tuple[dict, object]] = []
        similar: list[tuple[dict, object]] = []
        for item in obj.get("items", []):
            if not item.get("url") or not item.get("title"):
                continue
            result = score_candidate(identity, item["title"], item["url"], price, item.get("price"))
            if result.relation == "none":
                rejected += 1
                continue
            if item.get("out_of_stock") and OUT_OF_STOCK_FLAG not in result.flags:
                result.flags.append(OUT_OF_STOCK_FLAG)
            (exact_like if result.status != "rejected" else similar).append((item, result))
        exact_like.sort(key=lambda pair: pair[1].score, reverse=True)
        similar.sort(key=lambda pair: pair[1].closeness, reverse=True)
        chosen = exact_like[:3] + similar[:2]
        links = open_links(chosen)
        listed = 0
        for pair, link in zip(chosen, links):
            if link is not None and link.state == "dead":
                dead_by_store[report["platform"]] = dead_by_store.get(report["platform"], 0) + 1
                continue
            shown_matches.append(shape(pair[0], {}, pair[0].get("image_url"), None, pair[1], link))
            listed += 1
        shown_attempts.append({"platform": report["platform"], "status": "results collected" if report["status"] == "ok" else report["status"],
                               "result_count": listed, "raw_results": report.get("result_count", 0),
                               "filtered_out": rejected, "query": report.get("query")})
        progress(list(shown_matches), list(shown_attempts), identity)

    search_payload = {"queries": queries, "stores": stores, "per_store": 8}
    search_timeout = int(os.getenv("ENHANCED_SEARCH_TIMEOUT", "100"))
    if progress is not None:          # the app wants each store's answer as it arrives
        searched = _run_node_stream("search", search_payload, timeout=search_timeout, on_store=on_store)
    else:
        searched = _run_node("search", search_payload, timeout=search_timeout)
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
        if result.relation == "none":
            rejected_by_store[item["platform"]] = rejected_by_store.get(item["platform"], 0) + 1
            continue
        if item.get("out_of_stock") and OUT_OF_STOCK_FLAG not in result.flags:
            result.flags.append(OUT_OF_STOCK_FLAG)
        scored.append((item, result))

    # best three per store, then best twelve overall; similar products (other storage / variant / model number ...)
    # are kept apart: two per store, ten overall, nearest first
    main_pool = sorted((pair for pair in scored if pair[1].status != "rejected"), key=lambda pair: pair[1].score, reverse=True)
    similar_pool = sorted((pair for pair in scored if pair[1].status == "rejected"), key=lambda pair: pair[1].closeness, reverse=True)
    per_store: dict[str, int] = {}
    kept: list[tuple[dict, object]] = []
    for item, result in main_pool:
        if per_store.get(item["platform"], 0) >= 3:
            continue
        per_store[item["platform"]] = per_store.get(item["platform"], 0) + 1
        kept.append((item, result))
    kept = kept[:12]
    per_store_similar: dict[str, int] = {}
    kept_similar: list[tuple[dict, object]] = []
    for item, result in similar_pool:
        if per_store_similar.get(item["platform"], 0) >= 2:
            continue
        per_store_similar[item["platform"]] = per_store_similar.get(item["platform"], 0) + 1
        kept_similar.append((item, result))
    kept_similar = kept_similar[:10]

    # Every listing's link is opened once with a plain request. Pages a store refuses to show to a plain request
    # (403, CAPTCHA, built by script) get a second look through the same headless browser visit that reads
    # rating / image / identifiers, so one browser launch serves both.
    link_of: dict[str, object] = {}
    if link_check.enabled():
        everything = kept + kept_similar
        with ThreadPoolExecutor(max_workers=8) as pool:
            for (item, _), link in zip(everything, pool.map(lambda pair: link_check.check(pair[0]["url"], pair[0]["title"]), everything)):
                link_of[_canonical(item["url"])] = link

    deep: dict[str, dict] = {}
    browser_saw: dict[str, dict] = {}
    to_open = [(i, r) for i, r in kept if r.score >= 0.6][:4]
    opened_keys = {_canonical(i["url"]) for i, _ in to_open}
    unverified = [(i, r) for i, r in kept + kept_similar
                  if link_of.get(_canonical(i["url"])) is not None and link_of[_canonical(i["url"])].state == "unchecked" and _canonical(i["url"]) not in opened_keys]
    visit = to_open + unverified[: max(0, 8 - len(to_open))]
    if visit:
        opened = _run_node("deep", {"urls": [{"platform": i["platform"], "url": i["url"]} for i, _ in visit]}, timeout=60)
        for detail in (opened or {}).get("details", []):
            browser_saw[_canonical(detail["url"])] = detail
            if detail.get("status") == "ok":
                deep[_canonical(detail["url"])] = detail

    def link_for(item: dict):
        link = link_of.get(_canonical(item["url"]))
        if link is not None and link.state == "unchecked" and _canonical(item["url"]) in browser_saw:
            second = link_check.judge_browser(item["url"], browser_saw[_canonical(item["url"])], item["title"])
            if second.state != "unchecked":
                return second
        return link

    def finish(pair):
        item, result = pair
        detail = deep.get(_canonical(item["url"]), {})
        cand_image = item.get("image_url") or detail.get("image_url") or (_page_image(item["url"]) if image_url else None)
        # Photos: an image-embedding comparison (perceptual hash if the model is not installed). Either way the number
        # shown is "how likely the two photos show the same product", and the verdict feeds the match decision.
        visual = visual_match.compare(image_url, cand_image) if (image_url and cand_image) else None
        if visual is None and image_url and cand_image:
            image_similarity, visual_verdict = compute_image_similarity(image_url, cand_image), None
        else:
            image_similarity, visual_verdict = (visual.score, visual.verdict) if visual else (None, None)
        final = score_candidate(
            identity, item["title"], item["url"], price, item.get("price") or detail.get("price"), image_similarity,
            source_gtin=source_gtin, candidate_gtin=detail.get("gtin"),
            source_mpn=source_mpn, candidate_mpn=detail.get("mpn"),
            visual_verdict=visual_verdict,
        )
        # The re-score above builds fresh flags, so the stock note must be added to *this* result.
        if item.get("out_of_stock") and OUT_OF_STOCK_FLAG not in final.flags:
            final.flags.append(OUT_OF_STOCK_FLAG)
        return item, detail, cand_image, image_similarity, final, link_for(item)

    def finish_similar(pair):
        # a similar product needs no photo comparison: it is listed with how it differs and whether its link opens
        item, result = pair
        return item, {}, item.get("image_url"), None, result, link_for(item)

    with ThreadPoolExecutor(max_workers=8) as pool:
        finished = list(pool.map(finish, kept)) + list(pool.map(finish_similar, kept_similar))

    matches: list[dict] = []
    for item, detail, cand_image, image_similarity, final, link in finished:
        if final.relation == "none":
            rejected_by_store[item["platform"]] = rejected_by_store.get(item["platform"], 0) + 1   # e.g. identifier veto after opening the page
            continue
        if link is not None and link.state == "dead":
            dead_by_store[item["platform"]] = dead_by_store.get(item["platform"], 0) + 1          # a link that does not open is never listed
            continue
        matches.append(shape(item, detail, cand_image, image_similarity, final, link))

    attempts = []
    for report in searched.get("stores", []):
        name = report["platform"]
        attempts.append({
            "platform": name,
            "status": ("results collected" if report["status"] == "ok" else report["status"]),
            "result_count": sum(1 for m in matches if m["platform"] == name),
            "similar_count": sum(1 for m in matches if m["platform"] == name and m["relation"] == "similar"),
            "dead_links": dead_by_store.get(name, 0),
            "raw_results": report.get("result_count", 0),
            "filtered_out": rejected_by_store.get(name, 0),
            "query": report.get("query"),
        })
    return matches, attempts, identity
