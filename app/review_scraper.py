"""Public-page review collection fallbacks.

These collectors do not bypass authentication or access controls.  They only
parse review text that a marketplace exposes to an ordinary public request.
"""
from __future__ import annotations

import re
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
APIFY_DEFAULT_ACTOR = "pro100chok/amazon-scraper"


def _configured_limit(default: int) -> int:
    try:
        return max(1, min(int(os.getenv("APIFY_REVIEW_LIMIT", str(default))), 100))
    except ValueError:
        return default


def _rating(value: str | None) -> int | None:
    match = re.search(r"([1-5])(?:\.0)?", value or "")
    return int(match.group(1)) if match else None


def _amazon_review_url(product_url: str, asin: str) -> str:
    parsed = urlparse(product_url)
    origin = f"{parsed.scheme or 'https'}://{parsed.netloc or 'www.amazon.in'}"
    return f"{origin}/product-reviews/{asin}/?ie=UTF8&reviewerType=all_reviews&sortBy=recent"


def _parse_amazon_html(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict[str, Any]] = []
    for card in soup.select('[data-hook="review"]'):
        body = card.select_one('[data-hook="review-body"]')
        if not body:
            continue
        text = body.get_text(" ", strip=True)
        if len(text) < 4:
            continue
        stars = card.select_one('[data-hook="review-star-rating"] span, [data-hook="cmps-review-star-rating"] span')
        date = card.select_one('[data-hook="review-date"]')
        reviews.append({
            "text": text,
            "rating": _rating(stars.get_text(" ", strip=True) if stars else None),
            "date": date.get_text(" ", strip=True) if date else None,
            "verified_purchase": bool(card.select_one('[data-hook="avp-badge"]')),
        })
    return reviews


def _marketplace(product_url: str) -> str:
    """Return the Amazon marketplace suffix expected by the selected Actor."""
    host = (urlparse(product_url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host.removeprefix("amazon.") or "in"


def _first_text(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"([\d,]+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _integer(value: Any) -> int | None:
    parsed = _number(value)
    return int(parsed) if parsed is not None else None


def _run_apify_actor(actor: str, actor_input: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    token = os.getenv("APIFY_API_TOKEN", "").strip()
    if not token:
        return [], "Apify is not configured (add APIFY_API_TOKEN to .env)."
    if not actor:
        return [], "Apify Actor is not configured."

    actor_id = actor.replace("/", "~", 1)
    endpoint = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    try:
        with httpx.Client(timeout=80.0) as client:
            response = client.post(endpoint, params={"token": token}, json=actor_input)
            response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)], ""
        return [], "Apify returned an unexpected response."
    except httpx.HTTPStatusError as exc:
        return [], f"Apify request failed ({exc.response.status_code}). Check the token, Actor, and available credits."
    except (httpx.HTTPError, ValueError) as exc:
        return [], f"Apify request could not complete ({type(exc).__name__})."


def _normalise_apify_reviews(payload: Any, limit: int) -> list[dict[str, Any]]:
    """Map Actor output variants into the app's provider-neutral review shape."""
    records = payload if isinstance(payload, list) else []
    reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        text = _first_text(
            record,
            "reviewText", "review_text", "reviewBody", "review_body",
            "text", "body", "content",
        )
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        reviews.append({
            "text": text,
            "rating": _rating(str(record.get("rating") or record.get("stars") or record.get("reviewRating") or "")),
            "date": _first_text(record, "date", "reviewDate", "review_date", "createdAt"),
            "verified_purchase": bool(record.get("verifiedPurchase") or record.get("verified_purchase") or record.get("isVerifiedPurchase")),
        })
        if len(reviews) >= limit:
            break
    return reviews


def collect_apify_amazon_reviews(product_url: str, limit: int = 20) -> tuple[list[dict[str, Any]], str]:
    """Collect reviews through a configured Apify Actor without user credentials.

    The Actor is configurable because its output schema and commercial terms are
    controlled by its publisher.  This adapter only sends the public product URL
    and never uploads a shopper's Amazon cookies or account credentials.
    """
    limit = _configured_limit(limit)
    actor = os.getenv("APIFY_AMAZON_REVIEWS_ACTOR", APIFY_DEFAULT_ACTOR).strip()
    actor_input = {
        "scrapeType": "reviews",
        "reviewProductUrls": [product_url],
        "marketplace": _marketplace(product_url),
        "maxItemsPerInput": limit,
    }
    records, error = _run_apify_actor(actor, actor_input)
    if error:
        return [], f"Apify review request: {error}"
    reviews = _normalise_apify_reviews(records, limit)
    if reviews:
        return reviews, f"Apify Actor {actor}"
    return [], f"Apify Actor {actor} returned no accessible individual review text."


def collect_apify_amazon_product(product_url: str) -> tuple[dict[str, Any], str]:
    """Collect structured Amazon product metadata from the same configured Actor.

    Only values actually returned by Apify are exposed.  The caller decides how
    to merge these values with page-derived fields, preserving a transparent
    fallback when the provider has no record for a listing.
    """
    if os.getenv("REVIEW_PROVIDER", "public").strip().casefold() != "apify":
        return {}, "Apify product provider is disabled."

    actor = os.getenv("APIFY_AMAZON_PRODUCT_ACTOR", APIFY_DEFAULT_ACTOR).strip()
    records, error = _run_apify_actor(actor, {
        "scrapeType": "product",
        "productUrls": [product_url],
        "marketplace": _marketplace(product_url),
        "maxItemsPerInput": 1,
    })
    if error:
        return {}, f"Apify product request: {error}"
    if not records:
        return {}, f"Apify Actor {actor} returned no product record."

    record = records[0]
    price = _number(record.get("price") or record.get("priceText") or record.get("currentPrice"))
    rating = _number(record.get("rating") or record.get("ratingStars") or record.get("stars"))
    mapped = {
        "title": _first_text(record, "title", "productTitle", "name"),
        "brand": _first_text(record, "brand", "brandName"),
        "sku": _first_text(record, "asin", "ASIN", "sku", "productId"),
        "price": price,
        "currency": _first_text(record, "currency", "priceCurrency") or ("INR" if ".in" in product_url else None),
        "image_url": _first_text(record, "imageUrl", "image_url", "mainImage", "image", "thumbnail"),
        "rating": rating,
        "review_count": _integer(record.get("reviewCount") or record.get("reviewsCount") or record.get("totalReviews") or record.get("ratingsCount")),
        "availability": _first_text(record, "availability", "availabilityText", "stockStatus"),
    }
    return {key: value for key, value in mapped.items() if value is not None}, f"Apify Actor {actor}"


def collect_amazon_reviews(product_url: str, asin: str, limit: int = 20) -> tuple[list[dict[str, Any]], str]:
    """Try the selected managed provider, then public HTML and Playwright."""
    provider = os.getenv("REVIEW_PROVIDER", "public").strip().casefold()
    if provider == "apify":
        reviews, source = collect_apify_amazon_reviews(product_url, limit)
        if reviews:
            return reviews, source

    review_url = _amazon_review_url(product_url, asin)
    html_error = ""
    try:
        with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=12.0) as client:
            response = client.get(review_url, headers={"Referer": product_url})
        reviews = _parse_amazon_html(response.text)
        if reviews:
            return reviews[:limit], "BeautifulSoup public review page"
        lower = response.text.lower()
        if "sign in or create account" in lower or "enter mobile number or email" in lower:
            return [], f"{source if provider == 'apify' else 'Amazon'} Amazon requires sign-in for its public review page"
    except Exception as exc:
        # Playwright is the next independent collection method.
        html_error = type(exc).__name__

    playwright_error = ""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT, viewport={"width": 1280, "height": 800})
            page.goto(review_url, wait_until="domcontentloaded", timeout=15000, referer=product_url)
            page.wait_for_selector('[data-hook="review"]', timeout=5000)
            reviews = page.eval_on_selector_all('[data-hook="review"]', """cards => cards.map(card => {
                const body = card.querySelector('[data-hook="review-body"]');
                const stars = card.querySelector('[data-hook="review-star-rating"] span, [data-hook="cmps-review-star-rating"] span');
                const date = card.querySelector('[data-hook="review-date"]');
                return body ? {text: body.innerText.trim(), rating: stars ? stars.innerText : null,
                  date: date ? date.innerText.trim() : null,
                  verified_purchase: Boolean(card.querySelector('[data-hook="avp-badge"]'))} : null;
            }).filter(Boolean)""")
            browser.close()
        for review in reviews:
            review["rating"] = _rating(str(review.get("rating") or ""))
        return reviews[:limit], "Playwright public review page"
    except Exception as exc:
        playwright_error = type(exc).__name__
        details = ", ".join(part for part in (html_error, playwright_error) if part)
        prefix = f"{source}; " if provider == "apify" else ""
        return [], f"{prefix}Amazon did not expose reviews to BeautifulSoup or Playwright{f' ({details})' if details else ''}"
