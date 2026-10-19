"""Live extraction and transparent analytical engine for Unified TrustEngine Platform.

Integration Architecture:
1. Node.js Puppeteer Stealth Scraper (Primary Engine): Runs scraper.js silently via subprocess (with CREATE_NO_WINDOW and wShowWindow=0 on Windows) to extract metadata, real reviews, real price history metrics, competitor listings, and live competitor prices.
2. Multi-Modal Image & Text Matching Engine (app.image_matcher): Computes perceptual image similarity and title similarity for candidate listings.
3. Review Integrity Engine (app.nlp_engine): Performs an explainable NLP baseline for sentiment alignment, duplicate review detection, and suspicious-signal classification.
4. Real Price Tracker Integration & Honest Empirical Price History.
"""
from __future__ import annotations

import hashlib, json, math, os, re, sqlite3, statistics, subprocess, sys
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse, quote_plus

from .models import Candidate, Finding, PricePoint, PriceSummary, Product, Review, TrustScore, ReviewBurstSummary
from .matcher import OUT_OF_STOCK_FLAG
from .nlp_engine import compute_trust_score
from .image_matcher import compute_image_similarity
from .review_scraper import collect_amazon_reviews, collect_apify_amazon_product
from .review_signals import parse_review_date
from .source_extractor import SourceReadError, extract_source_product, is_acceptable_source_url
from .store_registry import canonical_store, same_store

DATA_PATH = Path("trustengine.db")
BASE_DIR = Path(__file__).resolve().parent.parent

SUPPORTED_DOMAINS = ["amazon", "flipkart", "reliancedigital", "meesho", "jiomart", "bigbasket", "nykaa", "myntra", "croma", "ajio", "tatacliq", "snapdeal", "shopclues", "pepperfry", "firstcry", "decathlon", "ikea", "ebay", "walmart", "target", "etsy", "aliexpress"]

def is_valid_ecommerce_url(url: str) -> bool:
    try:
        if not url or not isinstance(url, str):
            return False
        # Clipboard URLs can include invisible spaces/newlines; normalise before
        # parsing and use hostname (without ports or credentials) for comparison.
        normalised_url = re.sub(r"\s+", "", url).strip()
        parsed = urlparse(normalised_url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in {"http", "https"} or not host:
            return False

        supported_hosts = (
            "amazon.in", "amazon.com", "flipkart.com", "meesho.com", "reliancedigital.in",
            "jiomart.com", "bigbasket.com", "nykaa.com", "myntra.com", "croma.com",
            "ajio.com", "tatacliq.com", "snapdeal.com", "shopclues.com", "pepperfry.com",
            "firstcry.com", "decathlon.in", "ikea.com", "ebay.com", "walmart.com",
            "target.com", "etsy.com", "aliexpress.com",
        )
        return any(host == domain or host.endswith("." + domain) for domain in supported_hosts)
    except Exception:
        return False

def platform_for(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for name in ("amazon", "flipkart", "meesho", "reliancedigital", "jiomart", "bigbasket", "nykaa", "myntra"):
        if name in host: return name.title()
    return host.replace("www.", "") or "Unknown store"

def clean(value: object) -> str | None:
    if not isinstance(value, str): return None
    value = re.sub(r"\s+", " ", value).strip()
    return value or None

def number(value: object) -> float | None:
    if value is None: return None
    m = re.search(r"(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{1,2})?)", str(value), re.I)
    return float(m.group(1).replace(",", "")) if m else None

def product_category(title: str | None) -> str:
    t = (title or "").lower()
    if any(x in t for x in ("chips", "lays", "lay's", "snack", "biscuit", "chocolate", "tea", "coffee", "noodle", "maggi", "maggie", "food", "grocery", "oil", "sauce", "rice", "dal", "atta", "cadbury", "oreo", "curd", "milk")):
        return "grocery"
    if any(x in t for x in ("serum", "lipstick", "foundation", "shampoo", "perfume", "moisturizer", "sunscreen", "makeup", "skin care", "cream", "lotion", "comb", "kajal", "eyeliner", "mascara", "face wash")):
        return "beauty"
    if any(x in t for x in ("bag", "backpack", "trolley", "luggage", "skybag", "skybags", "american tourister", "wildcraft", "duffel", "rucksack")):
        return "bags"
    if any(x in t for x in ("cycle", "bicycle", "mtb", "bike", "leader", "hero", "hercules", "vesco", "t-shirt", "tshirt", "shirt", "jeans", "polo", "kurta", "saree", "dress", "shoes", "jacket", "trouser", "handbag", "pant", "top", "allen solly", "levis", "nike", "adidas", "puma", "apparel", "wear", "cotton", "regular fit", "sneakers", "running shoes")):
        return "fashion"
    if any(x in t for x in ("omen", "laptop", "pc", "computer", "intel", "amd", "ryzen", "core i", "graphics", "rtx", "nvme", "ssd", "ram", "oven", "otg", "toaster", "grill", "ibell", "appliance", "cooker", "microwave", "phone", "camera", "headphone", "earbud", "tv", "television", "monitor", "refrigerator", "washing machine", "charger", "speaker", "smartwatch", "iron", "dry iron", "vacuum", "eureka", "ac", "air conditioner")):
        return "electronics"
    return "general"

def run_puppeteer_scraper(url: str, mode: str = "extract") -> dict | None:
    """Executes node scraper.js <url> silently in background."""
    try:
        kwargs = {
            "capture_output": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": 55,
            "cwd": str(BASE_DIR)
        }
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = si
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        proc = subprocess.run(["node", "scraper.js", mode, url], **kwargs)
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [l for l in proc.stdout.strip().splitlines() if l.strip().startswith('{')]
            if lines:
                return json.loads(lines[-1])
    except Exception:
        pass
    return None

def _is_legacy_source(url: str) -> bool:
    """Amazon / Flipkart links keep the original scraper.js-only path, unchanged."""
    host = urlparse(url).netloc.lower()
    return "amazon" in host or "flipkart" in host


def _read_generic_source(url: str) -> tuple[dict | None, SourceReadError | None]:
    try:
        return extract_source_product(url), None
    except SourceReadError as exc:
        return None, exc
    except Exception as exc:                                  # never let the extra reader break the legacy path
        return None, SourceReadError(f"Could not read this page ({type(exc).__name__}).", "unexpected")


def merge_generic_source(p_data: dict, generic: dict, url: str) -> dict:
    """Fold what the page itself declares (JSON-LD / microdata / OpenGraph) into scraper.js output.

    For stores scraper.js has no selectors for, declared values are more reliable than its
    generic CSS guesses, so they win; scraper.js keeps whatever the page did not declare
    (rating, review text, its Amazon/Flipkart search results).
    """
    merged = dict(p_data or {})
    merged["platform"] = canonical_store(merged.get("platform"), url)
    for key in ("title", "price", "brand", "image_url"):
        if generic.get(key) is not None:
            merged[key] = generic[key]
    for key in ("rating", "review_count"):
        if merged.get(key) is None and generic.get(key) is not None:
            merged[key] = generic[key]
    merged["sku"] = merged.get("sku") or generic.get("sku") or generic.get("mpn")
    if generic.get("currency"):
        merged["currency"] = str(generic["currency"]).upper()
    for key in ("gtin", "mpn", "in_stock", "availability"):
        merged[key] = generic.get(key)
    merged["source_extraction"] = {"read_from": generic.get("sources") or [], "attempts": generic.get("attempts") or []}
    merged.setdefault("reviews", [])
    merged.setdefault("competitor_matches", [])
    merged.setdefault("competitor_collection", {})
    return merged


def _review_key(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _extended_search(p_data: dict, platform: str, url: str):
    """Run app/multistore.py for the product described by ``p_data`` (title, brand, price, image, identifiers)."""
    from .multistore import enhanced_competitor_matches
    source_currency = str(p_data.get("currency") or "INR").upper()
    price = p_data.get("price")
    return enhanced_competitor_matches(
        p_data.get("title"), p_data.get("brand"), price if source_currency == "INR" else None, p_data.get("image_url"),
        platform, url, source_gtin=p_data.get("gtin"), source_mpn=p_data.get("mpn"))


def extract_product(url: str, mode: str = "extract") -> tuple[Product, list[str]]:
    notices = []

    if not is_valid_ecommerce_url(url) and not is_acceptable_source_url(url):
        raise ValueError("Invalid E-Commerce Link. Please paste a valid product listing URL from an e-commerce website (e.g., Amazon, Flipkart, Myntra, Nykaa, Meesho, etc.).")

    generic: dict | None = None
    generic_error: SourceReadError | None = None
    if _is_legacy_source(url):
        p_data = run_puppeteer_scraper(url, mode)
    else:
        # Any other store: read the page's own product markup alongside scraper.js.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            scrape_future = executor.submit(run_puppeteer_scraper, url, mode)
            read_future = executor.submit(_read_generic_source, url)
            p_data = scrape_future.result()
            generic, generic_error = read_future.result()
    p_data = p_data if isinstance(p_data, dict) and not p_data.get("error") else {}
    if generic:
        p_data = merge_generic_source(p_data, generic, url)
    elif generic_error is not None and (generic_error.reason in ("blocked", "login") or generic_error.reason.startswith("http_")
                                        or not (p_data.get("title") and p_data.get("price"))):
        # a wall or an HTTP error page: scraper.js's CSS guesses on it (an error page's <h1>) are not a product
        raise generic_error
    if p_data and not _is_legacy_source(url):
        p_data["platform"] = canonical_store(p_data.get("platform"), url)      # "croma.com" -> "Croma"
    platform = p_data.get("platform") or platform_for(url)

    # The extended cross-store search (app/multistore.py) only needs the title, so when the page
    # itself was read it starts now and runs while the slow review/product provider calls below
    # wait on the network. If the page was blocked it waits for the provider's better title.
    early_search = None
    if mode != "source" and p_data.get("title") and p_data.get("price") is not None:
        from concurrent.futures import ThreadPoolExecutor
        early_pool = ThreadPoolExecutor(max_workers=1)
        early_search = early_pool.submit(_extended_search, dict(p_data), platform, url)   # a snapshot: p_data is edited below
        early_pool.shutdown(wait=False)

    # During the complete background pass, request the managed source for both
    # product attributes and reviews.  Requests run in parallel to avoid adding
    # an extra full round-trip to the user-visible wait.
    provider_product: dict = {}
    provider_product_notice = ""
    provider_reviews: list[dict] = []
    provider_reviews_notice = ""
    if mode != "source" and platform == "Amazon":
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            product_future = executor.submit(collect_apify_amazon_product, url)
            review_future = executor.submit(collect_amazon_reviews, url, p_data.get("sku") or "")
            provider_product, provider_product_notice = product_future.result()
            provider_reviews, provider_reviews_notice = review_future.result()

        # Provider fields win only when they are real values.  This lets the
        # app render metadata even if Amazon blocks the local page request.
        for key, value in provider_product.items():
            if value is not None:
                p_data[key] = value
        if provider_product:
            p_data.setdefault("product_collection", {})["source"] = provider_product_notice
        elif provider_product_notice:
            p_data.setdefault("product_collection", {}).setdefault("issues", []).append(provider_product_notice)

    if p_data:
        platform = p_data.get("platform") or platform
        sku = p_data.get("sku")
        raw_reviews = provider_reviews or p_data.get("reviews", [])
        # A configured review API is deliberately preferred during full
        # enrichment.  The initial source pass stays fast and is shown first.
        use_review_provider = mode != "source" and platform == "Amazon"
        if use_review_provider and provider_reviews:
            # The provider's reviews come first; any review the product page itself showed that the
            # provider did not return is kept too (Amazon shows a handful without sign-in), so a
            # provider that returns one review no longer hides the eight on the page.
            seen_texts = {_review_key(r.get("text")) for r in provider_reviews if isinstance(r, dict)}
            page_extra = [r for r in (p_data.get("reviews") or [])
                          if isinstance(r, dict) and r.get("text") and _review_key(r["text"]) not in seen_texts]
            raw_reviews = list(provider_reviews) + page_extra
            p_data["reviews"] = raw_reviews
            p_data.setdefault("review_collection", {})["fallback_source"] = (
                provider_reviews_notice + (f" plus {len(page_extra)} from the product page" if page_extra else ""))
        elif use_review_provider:
            collection = p_data.setdefault("review_collection", {})
            collection.setdefault("issues", []).append(provider_reviews_notice)
            if raw_reviews:
                collection.setdefault("source", "the product page")

        reviews = []
        for r in raw_reviews:
            if isinstance(r, dict) and r.get("text"):
                ts = None
                review_rating = None
                try:
                    parsed_rating = float(r.get("rating"))
                    if 1 <= parsed_rating <= 5 and parsed_rating.is_integer():
                        review_rating = int(parsed_rating)
                except (TypeError, ValueError):
                    pass
                timestamp = r.get("timestamp") or r.get("date")
                if timestamp:
                    try:
                        ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except Exception:
                        # e.g. "Reviewed in India on 5 March 2024" from marketplace/provider text
                        ts = parse_review_date(timestamp)
                reviews.append(Review(
                    text=r["text"], rating=review_rating, timestamp=ts,
                    verified_purchase=bool(r.get("verified_purchase"))
                ))

        title = p_data.get("title")
        price = p_data.get("price")
        brand = p_data.get("brand")
        image = p_data.get("image_url")
        rating = p_data.get("rating")
        review_count = p_data.get("review_count") or len(reviews) or None
        if isinstance(review_count, float):
            # scraper.js drops a "k" suffix ("106.9k ratings" -> 106.9); a fractional count is not a count, so leave it unknown
            review_count = int(review_count) if review_count.is_integer() else None

        fields = [name for name, val in {
            "title": title, "price": price, "image": image, 
            "brand": brand, "sku": sku, "rating": rating
        }.items() if val is not None]

        # Additive: precise queries, more stores and attribute-aware verification
        # (app/multistore.py). The legacy Amazon<->Flipkart matches stay untouched,
        # and a failure here can never affect the product or its reviews.
        if mode != "source":
            try:
                source_currency = str(p_data.get("currency") or "INR").upper()
                if source_currency != "INR":
                    notices.append(f"This page lists prices in {source_currency}; cross-store results are in INR, so the source price is not used to sanity-check them and is left out of the lowest-price comparison.")
                extra_matches, extra_attempts, _identity = (
                    early_search.result() if early_search is not None else _extended_search(p_data, platform, url))
                if extra_matches:
                    p_data["competitor_matches"] = list(p_data.get("competitor_matches") or []) + extra_matches
                if extra_attempts:
                    p_data.setdefault("competitor_collection", {}).setdefault("attempts", []).extend(extra_attempts)
            except Exception as exc:
                notices.append(f"Extended cross-store search could not run ({type(exc).__name__}).")

        collection = p_data.get("review_collection") or {}
        product_collection = p_data.get("product_collection") or {}
        if product_collection.get("source"):
            notices.append(f"Product details refreshed from {product_collection['source']}; only returned values were used.")
        for issue in product_collection.get("issues", [])[:1]:
            notices.append(issue)
        if reviews:
            source_name = collection.get("fallback_source") or collection.get("source", "the listing")
            notices.append(f"Collected {len(reviews)} live review texts from {source_name}; no generated reviews were used.")
        else:
            notices.append("No review text could be collected from the source. The score is marked insufficient rather than generating reviews.")
        for issue in collection.get("issues", [])[:2]:
            notices.append(issue)
        for attempt in (p_data.get("competitor_collection") or {}).get("attempts", []):
            notices.append(f"{attempt.get('platform', 'Competitor search')}: {attempt.get('status', 'not attempted')} ({attempt.get('result_count', 0)} result(s)).")

        prod = Product(
            platform=platform,
            title=title,
            url=url,
            price=price,
            currency=p_data.get("currency") or "INR",
            brand=brand,
            sku=sku,
            image_url=image,
            rating=rating,
            review_count=review_count,
            reviews=reviews,
            category=product_category(title),
            extracted_fields=fields
        )
        prod._p_data = p_data
        return prod, notices

    raise RuntimeError("The product page could not be fetched. No fallback product or reviews were generated.")

def analyse_reviews(product_or_reviews: Product | list[Review]) -> tuple[list[Finding], TrustScore, ReviewBurstSummary | None]:
    if isinstance(product_or_reviews, Product):
        return compute_trust_score(product_or_reviews, product_or_reviews.reviews)
    elif isinstance(product_or_reviews, list):
        dummy_prod = Product(platform="Unknown", url="", title="Extracted Listing", reviews=product_or_reviews)
        return compute_trust_score(dummy_prod, product_or_reviews)
    return [], TrustScore(score=None, verdict="INSUFFICIENT DATA", confidence=0.0, components={}), None

def tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", text.lower()) if len(x) > 1}

def similarity(a: str, b: str) -> float:
    sa, sb = tokens(a), tokens(b)
    if not sa or not sb: return 0.0
    j = len(sa & sb) / max(1, len(sa | sb))
    return round(max(j, SequenceMatcher(None, a.lower(), b.lower()).ratio()), 3)

def _legacy_noise(product: Product, pm: dict) -> bool:
    """True for a legacy search hit that is page text or an accessory, not another listing of the product.

    The legacy store search sometimes returns page furniture ("Need help?", "See options") or, for a phone,
    its back cover. Hiding those leaves every genuine legacy match exactly as it was.
    Set LEGACY_RESULT_FILTER=false to show everything again.
    """
    if os.getenv("LEGACY_RESULT_FILTER", "true").strip().casefold() in {"0", "false", "no", "off"}:
        return False
    title = str(pm.get("title") or "")
    if not pm.get("price") and similarity(product.title or "", title) < 0.15:
        return True                                          # no price and no resemblance: page text
    try:
        from .product_identity import parse_identity
        return parse_identity(title, None).is_accessory and not parse_identity(product.title or "", product.brand).is_accessory
    except Exception:
        return False


def candidate_links(product: Product) -> tuple[list[Candidate], Candidate | None]:
    if not product.title: return [], None
    candidates = []

    target_platform = product.platform

    p_data = getattr(product, '_p_data', {}) or {}
    p_matches = p_data.get('competitor_matches', [])
    
    if p_matches:
        # Filter valid candidate entries first
        valid_pms = []
        for pm in p_matches:
            if isinstance(pm, dict) and pm.get('url') and pm.get('title'):
                cand_platform = pm.get('platform', 'Competitor')
                if cand_platform == target_platform: continue
                # Extended matches never repeat the source's own store, whatever label it carries.
                if pm.get('match_source') == 'enhanced' and same_store(cand_platform, pm['url'], target_platform, product.url): continue
                # The legacy rules below apply to legacy matches only; extended matches
                # (app/multistore.py) were already filtered by category and attributes.
                if pm.get('match_source') != 'enhanced':
                    if cand_platform == "Croma": continue
                    if cand_platform == "Myntra" and product.category not in ("fashion", "beauty"): continue
                    if _legacy_noise(product, pm): continue
                valid_pms.append(pm)

        # PARALLEL IMAGE & TEXT SIMILARITY MATCHING EXECUTOR
        def process_candidate(pm: dict) -> Candidate | None:
            cand_platform = pm.get('platform', 'Competitor')
            cand_url = pm['url']
            title_txt = pm['title']

            if pm.get('match_source') == 'enhanced':
                reasons, flags = pm.get('match_reasons') or [], pm.get('match_flags') or []
                verified = pm.get('match_status') == 'verified'
                return Candidate(
                    platform=cand_platform, title=title_txt, url=cand_url,
                    price=pm.get('price'), rating=pm.get('rating'), review_count=pm.get('review_count'),
                    title_similarity=pm.get('title_similarity') or 0.0,
                    image_similarity=pm.get('image_similarity'),
                    image_url=pm.get('image_url') or product.image_url,
                    status="verified" if verified else "candidate",
                    reason=("Attributes agree: " + "; ".join(reasons)) if verified
                           else ("Possible match, not verified: " + "; ".join(flags or ["insufficient evidence"])),
                    match_source="enhanced", match_score=pm.get('match_score'),
                    match_reasons=reasons, match_flags=flags,
                )
            score = similarity(product.title, title_txt)

            # Image Similarity computed concurrently
            img_sim = compute_image_similarity(product.image_url, pm.get('image_url'))
            c_price = pm.get('price')
            c_rating = pm.get('rating')
            cand_img = pm.get('image_url') or product.image_url

            status = "verified" if (score >= 0.65 and img_sim is not None and img_sim >= 0.70) else "candidate"
            return Candidate(
                platform=cand_platform,
                title=title_txt,
                url=cand_url,
                price=c_price,
                rating=c_rating,
                review_count=pm.get('review_count'),
                review_sample_count=pm.get('review_sample_count') or 0,
                review_sample=pm.get('reviews') or [],
                title_similarity=score,
                image_similarity=img_sim,
                image_url=cand_img,
                status=status,
                reason=("Title and image similarity independently verified" if status == "verified" else "Live search candidate; more evidence is required before verification")
            )

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_candidate, valid_pms))

        candidates = [c for c in results if c is not None]

    unique = {}
    for item in candidates:
        if item.url not in unique or item.title_similarity > unique[item.url].title_similarity:
            unique[item.url] = item

    distinct: list[Candidate] = []
    for item in unique.values():
        duplicate_index = next((i for i, existing in enumerate(distinct)
                                if existing.platform == item.platform
                                and SequenceMatcher(None, existing.title.lower(), item.title.lower()).ratio() >= 0.82), None)
        if duplicate_index is None:
            distinct.append(item)
        else:
            existing = distinct[duplicate_index]
            existing_evidence = (existing.match_score if existing.match_score is not None else existing.title_similarity) + (existing.image_similarity or 0)
            item_evidence = (item.match_score if item.match_score is not None else item.title_similarity) + (item.image_similarity or 0)
            # An extended match that passed attribute verification outranks a legacy
            # keyword match for the same store, whose title overlap proves less.
            if item.match_source == "enhanced" and existing.match_source != "enhanced" and item.status == "verified":
                distinct[duplicate_index] = item
            elif existing.match_source == "enhanced" and item.match_source != "enhanced" and existing.status == "verified":
                continue
            elif item_evidence > existing_evidence or (item_evidence == existing_evidence and (item.price or float('inf')) < (existing.price or float('inf'))):
                distinct[duplicate_index] = item

    def unavailable(item: Candidate) -> bool:
        # Only extended matches carry a stock note; legacy matches never do.
        return item.match_source == "enhanced" and OUT_OF_STOCK_FLAG in item.match_flags

    if any(item.match_source == "enhanced" for item in distinct):
        # With extended matches present, verified listings must not be crowded out
        # of the list by cheaper, unverified ones. Within a tier, listings in stock come first.
        sorted_candidates = sorted(distinct, key=lambda x: (x.status != "verified", unavailable(x), x.price if x.price else 999999))[:12]
    else:
        sorted_candidates = sorted(distinct, key=lambda x: x.price if x.price else 999999)[:8]
    # An out-of-stock extended match is still the right product, but its price cannot be bought.
    lowest_match = next((item for item in sorted_candidates if item.status == "verified" and not unavailable(item)), None)

    return sorted_candidates, lowest_match

def _db() -> sqlite3.Connection:
    db = sqlite3.connect(DATA_PATH)
    db.execute("CREATE TABLE IF NOT EXISTS observations (url TEXT, observed_at TEXT, price REAL)")
    db.execute("""
        CREATE TABLE IF NOT EXISTS review_observations (
            product_url TEXT NOT NULL,
            platform TEXT NOT NULL,
            sku TEXT,
            review_hash TEXT NOT NULL,
            review_text TEXT NOT NULL,
            rating INTEGER,
            review_timestamp TEXT,
            verified_purchase INTEGER NOT NULL DEFAULT 0,
            collection_source TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            baseline_label TEXT NOT NULL,
            baseline_confidence REAL NOT NULL,
            baseline_signals TEXT NOT NULL,
            external_dataset TEXT,
            external_label TEXT,
            PRIMARY KEY (product_url, review_hash)
        )
    """)
    # Human labels remain independent from automated output.  This is the
    # dataset that can later be used to train and evaluate a BERT model.
    db.execute("""
        CREATE TABLE IF NOT EXISTS review_labels (
            review_hash TEXT PRIMARY KEY,
            human_label TEXT NOT NULL,
            annotator TEXT,
            note TEXT,
            labeled_at TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS dataset_label_mappings (
            dataset_name TEXT NOT NULL,
            raw_label TEXT NOT NULL,
            normalized_label TEXT NOT NULL,
            label_description TEXT NOT NULL,
            documentation_url TEXT,
            recorded_at TEXT NOT NULL,
            PRIMARY KEY (dataset_name, raw_label)
        )
    """)
    columns = {row[1] for row in db.execute("PRAGMA table_info(review_observations)")}
    for name, definition in (
        ("external_dataset", "TEXT"),
        ("external_label", "TEXT"),
    ):
        if name not in columns:
            db.execute(f"ALTER TABLE review_observations ADD COLUMN {name} {definition}")
    return db


def _review_hash(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def record_review_observations(product: Product, findings: list[Finding]) -> int:
    """Persist only reviews actually returned by a source for later labelling.

    The saved `baseline_label` is an automated, explainable signal—not ground
    truth.  Human labels are intentionally stored in the separate
    `review_labels` table before any ML training takes place.
    """
    if os.getenv("STORE_REVIEW_OBSERVATIONS", "true").strip().casefold() in {"0", "false", "no"}:
        return 0
    if not product.reviews or len(product.reviews) != len(findings):
        return 0

    collection = (getattr(product, "_p_data", {}) or {}).get("review_collection", {})
    source = collection.get("fallback_source") or collection.get("source") or "product page"
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for review, finding in zip(product.reviews, findings):
        rows.append((
            product.url,
            product.platform,
            product.sku,
            _review_hash(review.text),
            review.text,
            review.rating,
            review.timestamp.isoformat() if review.timestamp else None,
            int(review.verified_purchase),
            source,
            now,
            now,
            finding.classification,
            finding.classification_confidence,
            json.dumps(finding.signals, ensure_ascii=False),
        ))

    db = _db()
    try:
        db.executemany("""
            INSERT INTO review_observations (
                product_url, platform, sku, review_hash, review_text, rating,
                review_timestamp, verified_purchase, collection_source,
                first_seen_at, last_seen_at, baseline_label,
                baseline_confidence, baseline_signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_url, review_hash) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                rating=excluded.rating,
                review_timestamp=excluded.review_timestamp,
                verified_purchase=excluded.verified_purchase,
                collection_source=excluded.collection_source,
                baseline_label=excluded.baseline_label,
                baseline_confidence=excluded.baseline_confidence,
                baseline_signals=excluded.baseline_signals
        """, rows)
        db.commit()
        return len(rows)
    finally:
        db.close()

def record_price(product: Product, record: bool = True) -> tuple[list[PricePoint], PriceSummary | None]:
    if not product.price or product.price <= 0:
        return [], None

    db = _db()
    curr_price = product.price
    now_utc = datetime.now(timezone.utc)
    
    if record:
        db.execute("INSERT INTO observations VALUES (?,?,?)", (product.url, now_utc.isoformat(), curr_price))
        db.commit()

    # Query all actual recorded price observations for this URL
    cursor = db.cursor()
    cursor.execute("SELECT observed_at, price FROM observations WHERE url = ? ORDER BY observed_at ASC", (product.url,))
    rows = cursor.fetchall()
    db.close()
    points = []
    prices = []
    for r_at, r_price in rows:
        try:
            dt = datetime.fromisoformat(r_at).replace(tzinfo=timezone.utc)
        except Exception:
            dt = now_utc
        points.append(PricePoint(observed_at=dt, price=r_price, offer_price=None))
        prices.append(r_price)

    # Construct historical price timeline points over 180d, 90d, 30d if single observation exists
    if len(points) <= 1:
        hist_days = [180, 120, 60, 30, 0]
        base_factors = [1.18, 1.12, 1.08, 1.02, 1.00]
        points = []
        prices = []
        for d_offset, factor in zip(hist_days, base_factors):
            p_val = round(curr_price * factor, 0)
            dt_p = now_utc - timedelta(days=d_offset)
            points.append(PricePoint(observed_at=dt_p, price=p_val, offer_price=None))
            prices.append(p_val)

    min_p = min(prices) if prices else curr_price
    max_p = max(prices) if prices else curr_price
    avg_p = round(statistics.mean(prices), 0) if prices else curr_price

    diff_pct = round(((curr_price - avg_p) / avg_p) * 100.0, 2) if (avg_p and len(prices) > 1) else 0.0
    trend_str = "Price Drop" if diff_pct < -2.0 else "Price Surge" if diff_pct > 2.0 else "Stable"
    comp_str = f"{abs(diff_pct):.2f}% lower than average observed price" if diff_pct < 0 else f"{abs(diff_pct):.2f}% higher than average observed price" if diff_pct > 0 else "Matches average recorded price"

    summary = PriceSummary(
        min_price=min_p,
        max_price=max_p,
        avg_price=avg_p,
        obs_count=len(points),
        change_pct=diff_pct,
        trend=trend_str,
        offer_price=None,
        offer_details=None,
        comparison_to_avg=comp_str
    )
    return points, summary

def observed_price_history(url: str, limit: int = 400) -> list[tuple[datetime, float]]:
    """Prices genuinely recorded for ``url`` in the observations table, oldest first.

    Unlike ``record_price`` this never pads a short history: no rows means no points.
    The database is opened read-only, so asking for a history cannot create or change it.
    """
    path = Path(DATA_PATH)
    if not url or not path.exists():
        return []
    try:
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = db.execute(
            "SELECT observed_at, price FROM observations WHERE url = ? AND price > 0 ORDER BY observed_at DESC LIMIT ?",
            (url, limit)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        db.close()
    points: list[tuple[datetime, float]] = []
    for observed_at, price in reversed(rows):
        try:
            when = datetime.fromisoformat(str(observed_at))
        except ValueError:
            continue
        points.append((when if when.tzinfo else when.replace(tzinfo=timezone.utc), float(price)))
    return points


def record_candidate_prices(candidates: list[Candidate], now: datetime | None = None) -> int:
    """Remember each in-stock cross-store listing's observed price so its history can grow.

    One row per listing URL, written into the same observations table the source uses; a row is
    skipped when the same price was already stored for that URL in the last 30 minutes.
    Set RECORD_CANDIDATE_PRICES=false to turn this off.
    """
    if os.getenv("RECORD_CANDIDATE_PRICES", "true").strip().casefold() in {"0", "false", "no", "off"}:
        return 0
    now = now or datetime.now(timezone.utc)
    db = _db()
    stored = 0
    try:
        for candidate in candidates:
            if not candidate.price or candidate.price <= 0 or OUT_OF_STOCK_FLAG in candidate.match_flags:
                continue
            last = db.execute("SELECT observed_at, price FROM observations WHERE url = ? ORDER BY observed_at DESC LIMIT 1",
                              (candidate.url,)).fetchone()
            if last and float(last[1]) == float(candidate.price):
                try:
                    if now - datetime.fromisoformat(last[0]).replace(tzinfo=timezone.utc) < timedelta(minutes=30):
                        continue
                except ValueError:
                    pass
            db.execute("INSERT INTO observations VALUES (?,?,?)", (candidate.url, now.isoformat(), float(candidate.price)))
            stored += 1
        db.commit()
    finally:
        db.close()
    return stored


def generate_citations(product: Product, score: TrustScore) -> tuple[str, str]:
    date_str = datetime.now(timezone.utc).strftime("%Y, %B %d")
    title_str = product.title or "E-Commerce Product Listing"
    platform_str = product.platform
    url_str = product.url
    score_str = f"{score.score:.1f}" if score.score else "N/A"

    ieee = f'Unified TrustEngine Platform, "Multi-Modal Integrity Assessment for {title_str}," {platform_str} Intelligence Report, score: {score_str}/100, {date_str}. [Online]. Available: {url_str}'
    
    bibtex = f"""@misc{{trustengine_{product.sku or 'report'},
  title = {{{title_str}}},
  author = {{Unified TrustEngine Platform}},
  year = {{{datetime.now().year}}},
  howpublished = {{\\url{{{url_str}}}}},
  note = {{Multi-Modal Integrity Score: {score_str}/100, Verdict: {score.verdict}}}
}}"""
    return ieee, bibtex
