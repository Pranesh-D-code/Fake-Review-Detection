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

from .engines import candidate_links, extract_product, generate_citations, is_valid_ecommerce_url, record_price, record_review_observations
from .nlp_engine import compute_trust_score

app = FastAPI(title="Unified TrustEngine Platform", version="2.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
REPORTS_CACHE: dict[str, dict[str, Any]] = {}


class AnalyseRequest(BaseModel):
    url: str


def _build_result(analysis_id: str, product: Any, notices: list[str], enrichment_status: str, record_observation: bool) -> dict[str, Any]:
    findings, trust_score, burst_summary = compute_trust_score(product, product.reviews)
    stored_review_count = record_review_observations(product, findings)
    if stored_review_count:
        notices.append(
            f"Saved {stored_review_count} source-collected review observation(s) for later human labelling and model evaluation."
        )
    history_points, price_summary = record_price(product, record=record_observation)
    candidates, lowest_verified = candidate_links(product)
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
        "score": trust_score.model_dump(),
        "findings": [finding.model_dump() for finding in findings],
        "stored_review_count": stored_review_count,
        "price_history": history,
        "price_summary": price_summary.model_dump() if price_summary else None,
        "review_burst_summary": burst_summary.model_dump() if burst_summary else None,
        "candidates": [candidate.model_dump() for candidate in candidates],
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
        REPORTS_CACHE[analysis_id] = _build_result(analysis_id, product, notices, "complete", record_observation=False)
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
    if not is_valid_ecommerce_url(url):
        return JSONResponse(status_code=400, content={"detail": "Invalid e-commerce link. Paste a product URL from a supported marketplace."})
    try:
        analysis_id = str(uuid.uuid4())
        product, notices = extract_product(url, mode="source")
        notices.insert(0, "Showing live product details now; review collection and cross-store matching continue in the background.")
        result = _build_result(analysis_id, product, notices, "pending", record_observation=True)
        REPORTS_CACHE[analysis_id] = result
        background_tasks.add_task(_enrich_analysis, analysis_id, url)
        return JSONResponse(result)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"detail": f"Analysis failed: {exc}"})


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
