"""Unified TrustEngine Platform FastAPI application.

Endpoints:
- GET /: Serves index.html UI
- POST /api/analyse: Accepts {"url": "..."} and returns complete multi-modal analysis JSON
- GET /api/report/{id}: Generates printable analytical evidence report
- GET /health: Status diagnostic check
"""
from __future__ import annotations

import traceback, uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .engines import extract_product, record_price, candidate_links, generate_citations, is_valid_ecommerce_url
from .nlp_engine import compute_trust_score

app = FastAPI(title="Unified TrustEngine Platform", version="2.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

REPORTS_CACHE: dict[str, dict[str, Any]] = {}

class AnalyseRequest(BaseModel):
    url: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyse")
async def analyse(req: AnalyseRequest):
    url = req.url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    if not is_valid_ecommerce_url(url):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid E-Commerce Link. Please paste a valid product listing URL from an e-commerce website (e.g., Amazon, Flipkart, Myntra, Nykaa, Meesho, etc.)."}
        )

    try:
        product, notices = extract_product(url)
        findings, trust_score, burst_summary = compute_trust_score(product, product.reviews)
        history_points, price_summary = record_price(product)
        candidates, lowest_verified = candidate_links(product)
        citation_ieee, citation_bibtex = generate_citations(product, trust_score)

        analysis_id = str(uuid.uuid4())
        
        # Serialize datetime objects to ISO strings safely for JSONResponse
        serialized_price_history = []
        for p in history_points:
            dt_str = p.observed_at.isoformat() if hasattr(p.observed_at, "isoformat") else str(p.observed_at)
            serialized_price_history.append({
                "observed_at": dt_str,
                "price": p.price,
                "offer_price": p.offer_price
            })

        result_data = {
            "analysis_id": analysis_id,
            "source": {
                "platform": product.platform,
                "url": product.url,
                "title": product.title,
                "price": product.price,
                "currency": product.currency,
                "brand": product.brand,
                "sku": product.sku,
                "image_url": product.image_url,
                "rating": product.rating,
                "review_count": product.review_count,
                "category": product.category,
                "extracted_fields": product.extracted_fields
            },
            "score": {
                "score": trust_score.score,
                "verdict": trust_score.verdict,
                "confidence": trust_score.confidence,
                "components": trust_score.components,
                "raw_rating": trust_score.raw_rating,
                "adjusted_rating": trust_score.adjusted_rating,
                "spam_filtered_count": trust_score.spam_filtered_count
            },
            "findings": [f.dict() for f in findings],
            "price_history": serialized_price_history,
            "price_summary": price_summary.dict(),
            "review_burst_summary": burst_summary.dict() if burst_summary else None,
            "candidates": [c.dict() for c in candidates],
            "lowest_verified_match": lowest_verified.dict() if lowest_verified else None,
            "citation_ieee": citation_ieee,
            "citation_bibtex": citation_bibtex,
            "notices": notices
        }

        REPORTS_CACHE[analysis_id] = result_data
        return JSONResponse(result_data)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"detail": f"Analysis failed: {str(e)}"}
        )

@app.get("/api/report/{analysis_id}", response_class=HTMLResponse)
async def view_report(request: Request, analysis_id: str):
    data = REPORTS_CACHE.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analytical report expired or not found")
    return templates.TemplateResponse("report.html", {"request": request, "report": data})

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "Unified TrustEngine Platform 2.0"}
