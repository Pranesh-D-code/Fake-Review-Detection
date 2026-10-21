from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

class AnalyseRequest(BaseModel):
    url: str = Field(min_length=3)

    @field_validator('url', mode='before')
    @classmethod
    def normalize_url(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip()
            if v and not v.startswith(('http://', 'https://')):
                return f"https://{v}"
        return v

class Review(BaseModel):
    text: str
    rating: int | None = Field(default=None, ge=1, le=5)
    timestamp: datetime | None = None
    verified_purchase: bool = False

class Product(BaseModel):
    platform: str
    title: str | None = None
    url: str
    price: float | None = None
    currency: str = "INR"
    brand: str | None = None
    sku: str | None = None
    image_url: str | None = None
    rating: float | None = None
    review_count: int | None = None
    category: str = "general"
    reviews: list[Review] = []
    extracted_fields: list[str] = []

class Candidate(BaseModel):
    platform: str
    title: str
    url: str
    price: float | None = None
    rating: float | None = None
    review_count: int | None = None
    review_sample_count: int = 0
    review_sample: list[dict[str, Any]] = []
    title_similarity: float
    image_similarity: float | None = None
    image_url: str | None = None
    status: Literal["verified", "candidate", "unverified"]
    reason: str
    # Set only by the extended cross-store matcher (app/matcher.py); None for legacy matches.
    match_source: str | None = None
    match_score: float | None = None
    match_reasons: list[str] = []
    match_flags: list[str] = []
    # "exact" = verified to be the same product; "similar" = same make and kind of product that differs in a way
    # listed in `differences` (shown apart, never used for the lowest price).
    relation: Literal["exact", "similar"] = "exact"
    differences: list[str] = []
    closeness: float | None = None
    # app/link_check.py: "ok" = the page opened and shows the product; "unchecked" = the store would not let a
    # check through (normally fine in a browser). Links the store says are dead are never listed.
    link_status: Literal["ok", "unchecked"] | None = None
    link_note: str | None = None

class Finding(BaseModel):
    text: str
    translation: str | None = None
    rating: int | None = None
    sentiment: float | None = None
    mismatch: float | None = None
    risk: Literal["low", "medium", "high"] = "low"
    classification: Literal["genuine-looking", "suspicious", "needs-review"] = "needs-review"
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: list[str] = []
    # Filled when the trained text model judged this review (app/ml_detector.py).
    ml_fake_probability: float | None = None
    detector: str = "rules"
    # Lets the UI save a person's Real/Fake label for this review (app/review_labels.py).
    review_hash: str | None = None
    human_label: str | None = None
    # The two-way answer shown to the user. 'limited' = thin or mixed evidence (short text, close call).
    verdict: Literal["genuine", "not_genuine"] = "genuine"
    verdict_certainty: Literal["clear", "limited"] = "limited"
    verdict_note: str | None = None

class TrustScore(BaseModel):
    score: float | None = None
    verdict: Literal["VERIFIED", "CAUTION", "INSUFFICIENT DATA", "BLOCKED"]
    confidence: float
    components: dict[str, float | None]
    raw_rating: float | None = None
    adjusted_rating: float | None = None
    spam_filtered_count: int = 0

class PricePoint(BaseModel):
    observed_at: datetime
    price: float
    offer_price: float | None = None

class PriceSummary(BaseModel):
    min_price: float | None = None
    max_price: float | None = None
    avg_price: float | None = None
    obs_count: int = 0
    change_pct: float | None = None
    trend: str = "Stable"
    offer_price: float | None = None
    offer_details: str | None = None
    comparison_to_avg: str | None = None

class ReviewBurstSummary(BaseModel):
    peak_date: str | None = None
    peak_count: int = 0
    max_z_score: float | None = None
    vol_30d: int = 0
    vol_90d: int = 0
    vol_365d: int = 0
    burst_points: list[dict] = []

class AnalysisResult(BaseModel):
    analysis_id: str
    generated_at: datetime
    source: Product
    score: TrustScore
    findings: list[Finding]
    candidates: list[Candidate]
    lowest_verified_match: Candidate | None = None
    price_history: list[PricePoint] = []
    price_summary: PriceSummary | None = None
    review_burst_summary: ReviewBurstSummary | None = None
    notices: list[str] = []
    citation_ieee: str | None = None
    citation_bibtex: str | None = None
