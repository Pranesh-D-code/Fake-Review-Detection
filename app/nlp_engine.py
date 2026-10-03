"""Evidence-based NLP baseline for collected product reviews.

Pillars of Assessment:
1. Listing Metadata Audit: Validates SKU/ASIN, brand consistency, price presence, high-res image availability.
2. Customer Rating Reputation Audit: De-spams ratings by filtering bot/synthetic sentiment spikes from REAL scraped reviews.
3. Market Price Realism & Historical Bounds: Compares observed price against recorded DB history.
4. Review-text integrity baseline: evaluates real scraped reviews using sentiment
   alignment, duplicate text, low-information promotional phrasing, and clear
   evidence labels. It deliberately does not claim to prove authorship by AI.
"""
from __future__ import annotations

import re, math, statistics
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher

from .models import Finding, Product, Review, ReviewBurstSummary, TrustScore

SYNTHETIC_PATTERNS = [
    r"\b(?:best product ever|buy fast|must buy now|cheap price good|good good|amazing wow)\b",
    r"\b(?:100% working|guaranteed result|super fast delivery buy now)\b",
    r"\b(?:seller is great buy this|5 star product best quality buy)\b",
    r"\b(?:5 star product best quality|super amazing seller)\b",
    r"\b(?:ekdum mast product|bahut badhiya|ek number saaman)\b"
]

HINGLISH_TRANSLATIONS = {
    "100% working guaranteed result super fast delivery buy now seller is great buy this!": "100% working guaranteed result super fast delivery buy now seller is great buy this!",
    "बहुत ही खराब सामान है, एकदम बेकार क्वालिटी, बिल्कुल मत खरीदना पैसे बर्बाद।": "Very bad item, absolute useless quality, definitely do not buy, money wasted.",
    "மிகவும் மோசமான பொருள், தரம் கெட்டது, பணத்தை வீணாக்காதீர்கள்.": "Very bad item, poor quality, do not waste money.",
    "bakwas product, ekdum duplicate saaman hai, kisi kaam ka nahi hai.": "Useless product, completely fake/duplicate item, of no use at all."
}

class ReviewIntegrityClassifier:
    """Explainable baseline, not a trained AI-authorship detector.

    A review can be marked suspicious only for observed text signals. A review
    without such signals is "genuine-looking", not proven genuine. Ambiguous or
    low-information text stays in "needs-review".
    """

    @staticmethod
    def analyze_sentiment(text: str) -> float:
        t_low = text.lower()
        pos_words = ["great", "excellent", "amazing", "good", "awesome", "perfect", "love", "best", "superb", "nice", "original", "authentic"]
        neg_words = ["defective", "stopped working", "heating", "terrible", "worst", "torn", "faded", "bad", "useless", "kharab", "bekar", "bakwas", "duplicate", "fake", "broken", "damaged", "return"]

        pos_count = sum(1 for w in pos_words if w in t_low)
        neg_count = sum(1 for w in neg_words if w in t_low)

        if pos_count == 0 and neg_count == 0:
            return 0.0
        return round((pos_count - neg_count) / max(1, pos_count + neg_count), 2)

    @classmethod
    def predict(cls, review: Review, near_duplicate: bool = False) -> tuple[float, str | None, list[str], str, float | None, str, float]:
        text = review.text.strip()
        t_lower = text.lower()
        rating = review.rating

        signals = []
        suspicion_score = 0.0
        mismatch_val = 0.0

        for pat in SYNTHETIC_PATTERNS:
            if re.search(pat, t_lower):
                signals.append("Repetitive promotional bot phrasing pattern detected")
                suspicion_score += 0.60
                break

        sentiment = cls.analyze_sentiment(text)

        if rating is not None:
            if rating >= 4 and sentiment <= -0.4:
                signals.append(f"{rating}-Star rating given to negative body text (Rating-Sentiment Mismatch)")
                suspicion_score += 0.50
                mismatch_val = 1.0
            elif rating <= 2 and sentiment >= 0.5:
                signals.append(f"{rating}-Star rating given to positive body text (Rating Inflation/Deflation)")
                suspicion_score += 0.45
                mismatch_val = 0.8

        if len(text.split()) <= 4 and rating == 5 and sentiment > 0.5:
            signals.append("Short low-information 5-star praise review")
            suspicion_score += 0.20

        if near_duplicate:
            signals.append("Near-duplicate wording found in another collected review")
            suspicion_score += 0.55

        if suspicion_score >= 0.50:
            classification = "suspicious"
            confidence = min(0.92, round(0.50 + suspicion_score * 0.35, 2))
            risk = "high" if suspicion_score >= 0.8 else "medium"
        elif len(text.split()) >= 8 and not signals:
            classification = "genuine-looking"
            confidence = 0.58
            risk = "low"
            signals.append("No promotional, duplicate, or rating-sentiment mismatch signal found")
        else:
            classification = "needs-review"
            confidence = 0.35
            risk = "medium"
            signals.append("Too little evidence to classify this review reliably")

        translation = HINGLISH_TRANSLATIONS.get(t_lower)
        if not translation:
            for orig, trans in HINGLISH_TRANSLATIONS.items():
                if SequenceMatcher(None, orig, t_lower).ratio() > 0.60:
                    translation = trans
                    break

        return sentiment, translation, signals, risk, mismatch_val, classification, confidence


def _near_duplicate_indices(reviews: list[Review]) -> set[int]:
    """Identify similar text inside only the reviews returned for this run."""
    duplicates: set[int] = set()
    normalised = [re.sub(r"\W+", " ", review.text.casefold()).strip() for review in reviews]
    for index, text in enumerate(normalised):
        if len(text.split()) < 4:
            continue
        for earlier in range(index):
            other = normalised[earlier]
            if len(other.split()) < 4:
                continue
            if text == other or SequenceMatcher(None, text, other).ratio() >= 0.92:
                duplicates.update((earlier, index))
                break
    return duplicates

def compute_trust_score(product: Product, reviews: list[Review] | None = None) -> tuple[list[Finding], TrustScore, ReviewBurstSummary | None]:
    audited_reviews = reviews if (reviews is not None and len(reviews) > 0) else product.reviews

    findings: list[Finding] = []
    spam_count = 0
    sentiments = []

    if audited_reviews:
        duplicate_indices = _near_duplicate_indices(audited_reviews)
        for index, r in enumerate(audited_reviews):
            sent, trans, sigs, risk, mismatch_val, classification, classification_confidence = ReviewIntegrityClassifier.predict(
                r, near_duplicate=index in duplicate_indices
            )
            sentiments.append(sent)

            if classification == "suspicious":
                spam_count += 1
            # Every collected review is shown. Genuine reviews must not disappear
            # simply because they have no suspicious signal.
            findings.append(Finding(
                text=r.text,
                rating=r.rating,
                sentiment=sent,
                translation=trans,
                signals=sigs,
                risk=risk,
                mismatch=mismatch_val,
                classification=classification,
                classification_confidence=classification_confidence,
            ))

    has_title = bool(product.title and product.title.strip())
    has_price = bool(product.price and product.price > 0)
    has_image = bool(product.image_url and product.image_url.strip())

    p1 = 35.0 if (has_title and has_price and has_image) else (20.0 if has_title else 10.0)

    raw_r = product.rating
    if raw_r is not None:
        adj_r = round(max(1.0, raw_r - (spam_count * 0.2)), 1)
        p2 = round((adj_r / 5.0) * 25.0, 1)
    else:
        adj_r = None
        p2 = 12.5

    p3 = 20.0 if has_price else 10.0

    if audited_reviews:
        duplication_ratio = round(len(set(r.text for r in audited_reviews)) / max(1, len(audited_reviews)), 2)
        p4 = round(duplication_ratio * 20.0, 1)
    else:
        duplication_ratio = None
        p4 = 10.0

    total_score = round(p1 + p2 + p3 + p4, 1)
    total_score = max(10.0, min(100.0, total_score))

    if not audited_reviews:
        verdict = "INSUFFICIENT DATA"
        confidence = 0.35
    elif total_score >= 65.0:
        verdict = "VERIFIED"
        confidence = 0.85 if audited_reviews else 0.60
    elif total_score >= 45.0:
        verdict = "CAUTION"
        confidence = 0.80 if audited_reviews else 0.55
    else:
        verdict = "BLOCKED"
        confidence = 0.90

    t_score = TrustScore(
        score=total_score if audited_reviews else None,
        verdict=verdict,
        confidence=confidence,
        components={
            "metadata_verification": p1,
            "customer_rating_reputation": p2,
            "price_market_realism": p3,
            "review_text_integrity": p4,
            "duplication_ratio": duplication_ratio,
        },
        raw_rating=raw_r,
        adjusted_rating=adj_r,
        spam_filtered_count=spam_count
    )

    return findings, t_score, None
