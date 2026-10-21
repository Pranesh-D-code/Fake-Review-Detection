"""FastAPI application for live product analysis."""
from __future__ import annotations

import re
import threading
import time
import traceback
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlparse

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .engines import (candidate_links, extract_product, generate_citations, is_valid_ecommerce_url, parse_reviews, record_candidate_prices,
                      record_price, record_review_observations, reset_progress_hook, set_progress_hook)
from .matcher import UNCONFIRMED_NOTE
from .models import Product, Review
from . import review_labels
from .nlp_engine import compute_trust_score
from .price_graph import build_price_comparison
from .review_signals import parse_review_date, summarise_findings
from .source_extractor import SourceReadError, is_acceptable_source_url

app = FastAPI(title="Unified TrustEngine Platform", version="2.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


def _warm_up() -> None:
    """Load the text model at startup (importing scikit-learn and the vocabulary takes seconds), not in the first request."""
    try:
        from . import ml_detector
        ml_detector.predict(["warming up the review model so the first real analysis does not have to load it"])
        from . import visual_match
        visual_match.warm_up()                       # image model: importing PyTorch takes seconds
    except Exception:
        pass


app.add_event_handler("startup", lambda: threading.Thread(target=_warm_up, name="model-warm-up", daemon=True).start())
REPORTS_CACHE: dict[str, dict[str, Any]] = {}
# A link analysed a moment ago answers instantly: finished results are kept for ten minutes.
ANALYSIS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_SECONDS = 600


def _cache_key(url: str) -> str:
    """Same product, same key: Amazon by ASIN, other stores by path (tracking parameters ignored)."""
    parsed = urlparse(url)
    asin = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", parsed.path)
    host = parsed.netloc.lower().removeprefix("www.")
    if asin:
        return f"{host}|{asin.group(1)}"
    keep = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(parsed.query)) if k in ("pid", "variant", "sku"))
    return f"{host}|{parsed.path.rstrip('/').lower()}|{keep}"


class AnalyseRequest(BaseModel):
    url: str


def _build_result(analysis_id: str, product: Any, notices: list[str], enrichment_status: str, record_observation: bool,
                  record_candidates: bool = False, persist_reviews: bool = True) -> dict[str, Any]:
    findings, trust_score, burst_summary = compute_trust_score(product, product.reviews)
    stored_review_count = record_review_observations(product, findings) if persist_reviews else 0
    if stored_review_count:
        notices.append(
            f"Saved {stored_review_count} source-collected review observation(s) for later human labelling and model evaluation."
        )
    try:                                          # show labels a person already gave these reviews
        already = review_labels.lookup([f.review_hash for f in findings if f.review_hash])
        for finding in findings:
            finding.human_label = already.get(finding.review_hash)
    except Exception:
        pass
    history_points, price_summary = record_price(product, record=record_observation)
    candidates, lowest_verified = candidate_links(product)
    if record_candidates and candidates:
        try:
            record_candidate_prices(candidates)      # real observed prices only; lets each store's history grow
        except Exception as exc:
            notices.append(f"Cross-store prices could not be saved for later history ({type(exc).__name__}).")
    p_data = getattr(product, "_p_data", {}) or {}
    try:
        price_comparison = build_price_comparison(
            product, candidates, attempts=(p_data.get("competitor_collection") or {}).get("attempts", []),
            legacy_summary_present=price_summary is not None)
    except Exception as exc:                     # the comparison is additive: never lose the analysis over it
        price_comparison = None
        notices.append(f"The cross-store price comparison could not be built ({type(exc).__name__}).")
    citation_ieee, citation_bibtex = generate_citations(product, trust_score)
    history = [{
        "observed_at": point.observed_at.isoformat() if hasattr(point.observed_at, "isoformat") else str(point.observed_at),
        "price": point.price, "offer_price": point.offer_price,
    } for point in history_points]
    return {
        "analysis_id": analysis_id,
        "enrichment_status": enrichment_status,
        "source": {name: getattr(product, name) for name in (
            "platform", "url", "title", "price", "currency", "brand", "sku", "image_url",
            "rating", "review_count", "category", "extracted_fields",
        )},
        "source_details": {key: p_data.get(key) for key in ("gtin", "mpn", "in_stock", "availability", "source_extraction")},
        "score": trust_score.model_dump(),
        "findings": [finding.model_dump() for finding in findings],
        "stored_review_count": stored_review_count,
        "price_history": history,
        "price_summary": price_summary.model_dump() if price_summary else None,
        "review_burst_summary": burst_summary.model_dump() if burst_summary else None,
        "review_analysis": summarise_findings(findings),
        "candidates": [candidate.model_dump() for candidate in candidates],
        "price_comparison": price_comparison,
        "lowest_verified_match": lowest_verified.model_dump() if lowest_verified else None,
        # the best listing that may be this very product; a similar product's price must not stand in for it
        "best_collected_match": next((c.model_dump() for c in candidates if c.relation == "exact" or c.differences == [UNCONFIRMED_NOTE]), None),
        "exact_count": sum(1 for c in candidates if c.relation == "exact"),
        "similar_count": sum(1 for c in candidates if c.relation == "similar"),
        "candidate_diagnostics": (getattr(product, "_p_data", {}) or {}).get("competitor_collection", {}),
        "citation_ieee": citation_ieee,
        "citation_bibtex": citation_bibtex,
        "notices": notices,
    }


def _partial_result(analysis_id: str, first: Product, notices: list[str], state: dict[str, Any]) -> dict[str, Any]:
    """What is known so far: the first-pass product plus whichever of the stores / reviews have come back."""
    product = first.model_copy(deep=True)
    p_data = dict(getattr(first, "_p_data", {}) or {})
    if state.get("matches"):
        extra, attempts, _identity = state["matches"]
        p_data["competitor_matches"] = list(p_data.get("competitor_matches") or []) + list(extra)
        p_data["competitor_collection"] = {"attempts": list((p_data.get("competitor_collection") or {}).get("attempts", [])) + list(attempts)}
    collected = state.get("reviews")
    if collected is not None and collected.reviews:
        known = {r.text.casefold() for r in product.reviews}
        product.reviews = list(product.reviews) + [r for r in parse_reviews(collected.reviews) if r.text.casefold() not in known]
        product.review_count = product.review_count or collected.reported_total or len(product.reviews)
    product._p_data = p_data
    working = [n for n in notices if not n.startswith("Still ")] + ["Still collecting more reviews and searching the remaining stores - this list fills in as they answer."]
    return _build_result(analysis_id, product, working, "pending", record_observation=False, persist_reviews=False)


def _enrich_analysis(analysis_id: str, url: str, first: Product | None = None) -> None:
    """Slow review pagination and cross-store search, run after the first response.

    Reviews and each group of stores are published as they arrive (the page polls), instead of everything
    appearing at the end.
    """
    state: dict[str, Any] = {}
    lock = threading.Lock()
    first_notices = list((REPORTS_CACHE.get(analysis_id) or {}).get("notices", []))

    def on_progress(kind: str, value: Any) -> None:
        if first is None:
            return
        with lock:
            state[kind] = value
            try:
                partial = _partial_result(analysis_id, first, first_notices, state)
            except Exception:
                return
            if (REPORTS_CACHE.get(analysis_id) or {}).get("enrichment_status") == "pending":
                REPORTS_CACHE[analysis_id] = partial

    token = set_progress_hook(on_progress)
    try:
        product, notices = extract_product(url, mode="extract")
        notices.insert(0, "Full review and cross-store enrichment completed.")
        with lock:
            result = _build_result(analysis_id, product, notices, "complete", record_observation=False, record_candidates=True)
            REPORTS_CACHE[analysis_id] = result
        ANALYSIS_CACHE[_cache_key(url)] = (time.time(), result)
    except Exception as exc:
        current = REPORTS_CACHE.get(analysis_id, {})
        current["enrichment_status"] = "failed"
        current.setdefault("notices", []).append(f"Background enrichment could not finish: {exc}")
        REPORTS_CACHE[analysis_id] = current
    finally:
        reset_progress_hook(token)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyse")
def analyse(req: AnalyseRequest, background_tasks: BackgroundTasks):
    url = re.sub(r"\s+", "", req.url).strip()
    if not url.startswith("http"):
        url = "https://" + url
    if not is_valid_ecommerce_url(url) and not is_acceptable_source_url(url):
        return JSONResponse(status_code=400, content={"detail": "Invalid link. Paste the web address of a product page (http or https) from an online store."})

    cached = ANALYSIS_CACHE.get(_cache_key(url))
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        age = int(time.time() - cached[0])
        result = dict(cached[1])
        result["analysis_id"] = str(uuid.uuid4())
        result["notices"] = [f"Showing the finished analysis of this link from {age // 60} min {age % 60} s ago (kept for {CACHE_SECONDS // 60} minutes); nothing was fetched again."] + list(cached[1].get("notices", []))
        REPORTS_CACHE[result["analysis_id"]] = result
        return JSONResponse(result)

    try:
        analysis_id = str(uuid.uuid4())
        product, notices = extract_product(url, mode="source")
        notices.insert(0, "Showing live product details now; review collection and cross-store matching continue in the background.")
        result = _build_result(analysis_id, product, notices, "pending", record_observation=True)
        REPORTS_CACHE[analysis_id] = result
        background_tasks.add_task(_enrich_analysis, analysis_id, url, product)
        return JSONResponse(result)
    except SourceReadError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc), "reason": exc.reason})
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"detail": f"Analysis failed: {exc}"})


class ReviewIn(BaseModel):
    text: str
    rating: int | None = None
    date: str | None = None
    verified_purchase: bool = False


class ClassifyReviewsRequest(BaseModel):
    reviews: list[ReviewIn]
    title: str | None = None


@app.post("/api/reviews/classify")
def classify_reviews(req: ClassifyReviewsRequest):
    """Run the fake-review analysis on reviews you already hold.

    Use this for a full review export you are licensed to use (a provider's
    bulk download, a seller-centre export, a research dataset). Only the text,
    optional 1-5 rating and optional date are read.
    """
    if not req.reviews:
        raise HTTPException(status_code=400, detail="Send at least one review.")
    if len(req.reviews) > 5000:
        raise HTTPException(status_code=400, detail="Send at most 5000 reviews per request.")
    reviews = []
    for item in req.reviews:
        text = item.text.strip()
        if not text:
            continue
        rating = item.rating if item.rating and 1 <= item.rating <= 5 else None
        reviews.append(Review(text=text, rating=rating, timestamp=parse_review_date(item.date),
                              verified_purchase=item.verified_purchase))
    if not reviews:
        raise HTTPException(status_code=400, detail="All reviews were empty.")
    product = Product(platform="Supplied reviews", url="", title=req.title or "Supplied review set", reviews=reviews)
    findings, trust_score, burst = compute_trust_score(product, reviews)
    return JSONResponse({
        "review_analysis": summarise_findings(findings),
        "findings": [finding.model_dump() for finding in findings],
        "review_burst_summary": burst.model_dump() if burst else None,
        "score_components": trust_score.components,
    })


class LabelRequest(BaseModel):
    text: str
    label: str                       # "real" | "fake" | "clear"
    rating: int | None = None
    product_url: str | None = None
    platform: str | None = None


@app.post("/api/reviews/label")
def label_review(req: LabelRequest):
    """Save a person's Real/Fake verdict on one review; may start a background retrain."""
    try:
        saved = review_labels.save_label(req.text, req.label, rating=req.rating, product_url=req.product_url, platform=req.platform)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        state = review_labels.maybe_start_retrain()
    except Exception:                             # a training problem must never lose the label that was just saved
        state = review_labels.status()
    return JSONResponse({**saved, "status": state})


@app.get("/api/model/status")
def model_status():
    return JSONResponse(review_labels.status())


@app.post("/api/model/retrain")
def model_retrain():
    """Start a retrain now (needs 20+ labels with at least 5 real and 5 fake)."""
    state = review_labels.status()
    if not state["can_train"]:
        raise HTTPException(status_code=400, detail="Label at least 20 reviews first, with at least 5 real and 5 fake.")
    return JSONResponse(review_labels.maybe_start_retrain(force=True))


@app.get("/api/analyse/{analysis_id}")
async def analysis_status(analysis_id: str):
    data = REPORTS_CACHE.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis expired or not found")
    return JSONResponse(data)


@app.get("/api/report/{analysis_id}", response_class=HTMLResponse)
async def view_report(request: Request, analysis_id: str):
    data = REPORTS_CACHE.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analytical report expired or not found")
    return templates.TemplateResponse("report.html", {"request": request, "report": data})


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "Unified TrustEngine Platform 2.1"}
