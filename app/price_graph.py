"""Cross-store price comparison for the price graph (``price_comparison`` in /api/analyse).

One entry per store: the source listing plus every store that has the same product.
Only values the stores actually showed are used - nothing is estimated:

* ``points`` are the current listing price (observed now) plus whatever the
  ``observations`` table genuinely recorded for that exact listing URL. A listing seen
  once has one point; there is no back-filled history.
* The headline "lowest" is chosen from the source listing and *verified* matches only,
  and never from a listing that is out of stock, refurbished/used, in another currency
  or without a price. Possible (unverified) matches are returned separately and can
  never be the lowest.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

from . import engines
from .matcher import OUT_OF_STOCK_FLAG
from .store_registry import canonical_store

BASE_CURRENCY = "INR"
MAX_POINTS = 60
LIVE_DUPLICATE_WINDOW = timedelta(minutes=15)
CONDITION_RE = re.compile(r"\b(?:refurbished|renewed|reconditioned|pre-?owned|second[- ]hand|open[- ]box|used)\b", re.I)


def format_inr(value: float | None) -> str:
    """24999 -> 'Rs 24,999' (Indian digit grouping)."""
    if value is None:
        return "price not read"
    whole = f"{abs(value):.0f}" if float(value).is_integer() else f"{abs(value):.2f}"
    head, _, tail = whole.partition(".")
    if len(head) > 3:
        head = re.sub(r"(?<=\d)(?=(?:\d\d)+\d$)", ",", head)
    return f"Rs {head}{'.' + tail if tail else ''}"


def _stock_from_flags(flags: Iterable[str]) -> bool | None:
    """False when the matcher noted the listing as out of stock; None (unknown) otherwise."""
    return False if any(flag == OUT_OF_STOCK_FLAG or "out of stock" in flag.lower() for flag in flags) else None


def _clean_points(points: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    return sorted({(when, price) for when, price in points if price and price > 0}, key=lambda item: item[0])[-MAX_POINTS:]


def _entry(*, store: str, kind: str, title: str, url: str, price: float | None, currency: str,
           rating: float | None, review_count: int | None, image_url: str | None, stock: bool | None,
           match_score: float | None = None, reasons: Iterable[str] = (), flags: Iterable[str] = (),
           source_title: str = "") -> dict[str, Any]:
    excluded: str | None = None
    if not price or price <= 0:
        excluded = "no price shown"
    elif stock is False:
        excluded = "out of stock"
    elif currency != BASE_CURRENCY:
        excluded = f"price in {currency}, not comparable"
    elif ((CONDITION_RE.search(title or "") or "refurbished/used listing" in list(flags))
          and not CONDITION_RE.search(source_title or "")):
        excluded = "refurbished / used listing"
    return {
        "store": store, "kind": kind, "title": title, "url": url, "price": price, "currency": currency,
        "rating": rating, "review_count": review_count, "image_url": image_url,
        "in_stock": stock,                                 # True / False only when the store said so, else None
        "match_score": match_score, "reasons": list(reasons), "flags": list(flags),
        "eligible": kind in ("source", "verified") and excluded is None,
        "excluded_reason": excluded,
        "is_lowest": False,
        "points": [], "observed_points": 0, "recorded_points": 0,
    }


def _attach_history(entry: dict, loader: Callable[[str], list[tuple[datetime, float]]], now: datetime) -> None:
    try:
        recorded = loader(entry["url"]) if entry["url"] else []
    except Exception:
        recorded = []
    points = _clean_points(list(recorded))
    entry["recorded_points"] = len(points)             # rows really stored for this URL, before today's live point
    entry["observed_points"] = len(points)
    price = entry["price"]
    live = [p for p in points if p[0] >= now - LIVE_DUPLICATE_WINDOW and p[1] == price]
    if price and price > 0 and not entry["excluded_reason"] == "out of stock" and not live:
        points.append((now, float(price)))
        entry["observed_points"] = len(points)
    entry["points"] = [{"observed_at": when.isoformat(), "price": value} for when, value in points[-MAX_POINTS:]]


def _best_per_store(entries: list[dict], key: Callable[[dict], Any]) -> list[dict]:
    best: dict[str, dict] = {}
    for entry in entries:
        current = best.get(entry["store"].casefold())
        if current is None or key(entry) < key(current):
            best[entry["store"].casefold()] = entry
    return list(best.values())


def build_price_comparison(
    product: Any,
    candidates: list[Any],
    *,
    attempts: list[dict] | None = None,
    history_loader: Callable[[str], list[tuple[datetime, float]]] | None = None,
    legacy_summary_present: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    loader = history_loader or engines.observed_price_history
    p_data = getattr(product, "_p_data", {}) or {}
    source_store = canonical_store(product.platform, product.url)
    source_currency = (product.currency or BASE_CURRENCY).upper()

    source = _entry(
        store=source_store, kind="source", title=product.title or "", url=product.url, price=product.price,
        currency=source_currency, rating=product.rating, review_count=product.review_count,
        image_url=product.image_url, stock=p_data.get("in_stock"),
        reasons=["the listing you pasted"], source_title=product.title or "")

    verified: list[dict] = []
    possible: list[dict] = []
    for candidate in candidates:
        store = canonical_store(candidate.platform, candidate.url)
        if store.casefold() == source_store.casefold():
            continue
        entry = _entry(
            store=store, kind="verified" if candidate.status == "verified" else "possible",
            title=candidate.title, url=candidate.url, price=candidate.price, currency=BASE_CURRENCY,
            rating=candidate.rating, review_count=candidate.review_count, image_url=candidate.image_url,
            stock=_stock_from_flags(candidate.match_flags),
            match_score=candidate.match_score,
            reasons=candidate.match_reasons or ([candidate.reason] if candidate.reason else []),
            flags=candidate.match_flags, source_title=product.title or "")
        (verified if entry["kind"] == "verified" else possible).append(entry)

    # one listing per store: the cheapest one that can actually be bought, else the cheapest shown
    def cheapest(entry: dict) -> tuple:
        return (not entry["eligible"], entry["price"] if entry["price"] else float("inf"))

    series = [source] + _best_per_store(verified, cheapest)
    shown_stores = {entry["store"].casefold() for entry in series}
    # a "possible" listing is only worth a marker where the store has no verified one
    possible_best = _best_per_store(
        [entry for entry in possible if entry["store"].casefold() not in shown_stores],
        lambda entry: (-(entry["match_score"] or 0), entry["price"] if entry["price"] else float("inf")))

    for entry in series + possible_best:
        _attach_history(entry, loader, now)

    eligible = [entry for entry in series if entry["eligible"]]
    lowest = min(eligible, key=lambda entry: (entry["price"], entry["kind"] != "source", entry["store"])) if eligible else None
    notes: list[str] = []
    if lowest:
        lowest["is_lowest"] = True
    for entry in series:
        if entry["excluded_reason"] and entry["kind"] != "possible":
            notes.append(f"{entry['store']}: {entry['excluded_reason']} - not counted for the lowest price.")
    hidden_verified = len(verified) - (len(series) - 1)
    if hidden_verified > 0:
        notes.append(f"{hidden_verified} more verified listing(s) from stores already shown; see the match list.")
    if possible:
        notes.append(f"{len(possible)} possible match(es) are not verified and are never used as the lowest price.")
    if source_currency != BASE_CURRENCY:
        notes.append(f"The source page lists prices in {source_currency}; it is left out of the comparison.")

    # The legacy and the extended search can both report on Amazon / Flipkart: one line per store, the fuller report wins.
    by_store: dict[str, dict] = {}
    for attempt in attempts or []:
        report = {
            "store": canonical_store(attempt.get("platform"), None), "status": attempt.get("status"),
            "listings": attempt.get("result_count", 0), "raw_results": attempt.get("raw_results"),
        }
        key = report["store"].casefold()
        if key not in by_store or (report["listings"] or 0) > (by_store[key]["listings"] or 0):
            by_store[key] = report
    searched = list(by_store.values())

    summary = None
    link = None
    if lowest:
        summary = f"Lowest: {format_inr(lowest['price'])} at {lowest['store']}"
        link = lowest["url"]
        if lowest["kind"] == "source":
            summary += " (the listing you pasted)"
        elif source["eligible"]:
            saving = source["price"] - lowest["price"]
            if saving > 0:
                summary += f" - Rs {saving:,.0f} below {source_store}"

    return {
        "currency": BASE_CURRENCY,
        "generated_at": now.isoformat(),
        "source_store": source_store,
        "series": series,
        "possible": possible_best,
        "lowest": ({key: lowest[key] for key in ("store", "kind", "price", "currency", "url", "title", "rating", "review_count")}
                   if lowest else None),
        "lowest_summary": summary,
        "lowest_link": link,
        "lowest_rule": "lowest price among the pasted listing and verified matches that are priced and not out of stock; possible matches never qualify",
        "stores_searched": searched,
        "notes": notes,
        # record_price() in engines.py invents a 180-day history when a URL has fewer than 2 stored rows;
        # that padded series is never used here, and the UI labels the legacy summary bar accordingly.
        "legacy_history_synthetic": bool(legacy_summary_present and source["recorded_points"] < 2),
        "observed_history_points": sum(entry["observed_points"] for entry in series),
    }
