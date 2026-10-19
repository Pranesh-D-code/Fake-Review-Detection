"""FastAPI application for live product analysis."""
from __future__ import annotations

import re
import traceback
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .engines import candidate_links, extract_product, generate_citations, is_valid_ecommerce_url, record_candidate_prices, record_price, record_review_observations
from .models import Product, Review
from . import review_labels
from .nlp_engine import compute_trust_score
from .price_graph import build_price_comparison
from .review_signals import parse_review_date, summarise_findings
from .source_extractor import SourceReadError, is_acceptable_source_url

app = FastAPI(title="Unified TrustEngine Platform", version="2.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
REPORTS_CACHE: dict[str, dict[str, Any]] = {}


class AnalyseRequest(BaseModel):
    url: str


def _build_result(analysis_id: str, product: Any, notices: list[str], enrichment_status: str, record_observation: bool,
                  record_candidates: bool = False) -> dict[str, Any]:
    findings, trust_score, burst_summary = compute_trust_score(product, product.reviews)
    stored_review_count = record_review_observations(product, findings)
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
        "best_collected_match": candidates[0].model_dump() if candidates else None,
        "candidate_diagnostics": (getattr(product, "_p_data", {}) or {}).get("competitor_collection", {}),
        "citation_ieee": citation_ieee,
        "citation_bibtex": citation_bibtex,
        "notices": notices,
    }


def _enrich_analysis(analysis_id: str, url: str) -> None:
    """Slow review pagination and cross-store search, run after the first response."""
    try:
        product, notices = extract_product(url, mode="extract")
        notices.insert(0, "Full review and cross-store enrichment completed.")
        REPORTS_CACHE[analysis_id] = _build_result(analysis_id, product, notices, "complete", record_observation=False, record_candidates=True)
    except Exception as exc:
        current = REPORTS_CACHE.get(analysis_id, {})
        current["enrichment_status"] = "failed"
        current.setdefault("notices", []).append(f"Background enrichment could not finish: {exc}")
        REPORTS_CACHE[analysis_id] = current


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyse")
async def analyse(req: AnalyseRequest, background_tasks: BackgroundTasks):
    url = re.sub(r"\s+", "", req.url).strip()
    if not url.startswith("http"):
        url = "https://" + url
    if not is_valid_ecommerce_url(url) and not is_acceptable_source_url(url):
        return JSONResponse(status_code=400, content={"detail": "Invalid link. Paste the web address of a product page (http or https) from an online store."})
    try:
        analysis_id = str(uuid.uuid4())
        product, notices = extract_product(url, mode="source")
        notices.insert(0, "Showing live product details now; review collection and cross-store matching continue in the background.")
        result = _build_result(analysis_id, product, notices, "pending", record_observation=True)
        REPORTS_CACHE[analysis_id] = result
        background_tasks.add_task(_enrich_analysis, analysis_id, url)
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
