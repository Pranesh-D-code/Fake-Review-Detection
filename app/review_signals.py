"""Product-level review signals that need no model: dates and posting bursts.

A sudden spike of reviews on one day is a classic sign of a paid or coordinated
campaign. ``review_burst`` measures it with a z-score over daily review counts.
It needs enough *dated* reviews to mean anything; below that it returns None
rather than guessing.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import date, datetime, timezone

from .models import Review, ReviewBurstSummary

MIN_DATED_REVIEWS = 15
BURST_Z = 2.0

_MONTHS = {name: number for number, names in enumerate((
    ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
    ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"),
    ("november", "nov"), ("december", "dec")), start=1) for name in names}
_DAY_MONTH_YEAR = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})")
_MONTH_DAY_YEAR = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})")


def parse_review_date(value: object) -> datetime | None:
    """ISO strings and marketplace phrases like 'Reviewed in India on 5 March 2024'."""
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    match = _DAY_MONTH_YEAR.search(text)
    if match and match.group(2).lower() in _MONTHS:
        day, month, year = int(match.group(1)), _MONTHS[match.group(2).lower()], int(match.group(3))
    else:
        match = _MONTH_DAY_YEAR.search(text)
        if not match or match.group(1).lower() not in _MONTHS:
            return None
        month, day, year = _MONTHS[match.group(1).lower()], int(match.group(2)), int(match.group(3))
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def review_burst(reviews: list[Review], now: datetime | None = None) -> ReviewBurstSummary | None:
    dated = [r.timestamp for r in reviews if r.timestamp]
    if len(dated) < MIN_DATED_REVIEWS:
        return None
    now = now or datetime.now(timezone.utc)
    days = [d.astimezone(timezone.utc).date() for d in dated]
    first, last = min(days), max(days)
    span = (last - first).days + 1
    if span < 3:
        return None

    counts = Counter(days)
    series = [counts.get(date.fromordinal(first.toordinal() + offset), 0) for offset in range(span)]
    mean = statistics.fmean(series)
    spread = statistics.pstdev(series)
    peak_day, peak_count = max(counts.items(), key=lambda item: item[1])
    z_of = lambda count: round((count - mean) / spread, 2) if spread > 0 else 0.0

    points = [
        {"date": day.isoformat(), "count": count, "z_score": z_of(count)}
        for day, count in sorted(counts.items()) if count >= 3 and z_of(count) >= BURST_Z
    ]
    within = lambda window: sum(1 for d in dated if (now - d.astimezone(timezone.utc)).days <= window)
    return ReviewBurstSummary(
        peak_date=peak_day.isoformat(), peak_count=peak_count, max_z_score=z_of(peak_count),
        vol_30d=within(30), vol_90d=within(90), vol_365d=within(365), burst_points=points,
    )


def summarise_findings(findings: list) -> dict:
    """Counts shown next to the review list, plus what the text model is and is not."""
    from . import ml_detector

    total = len(findings)
    counts = Counter(f.classification for f in findings)
    scored = sum(1 for f in findings if getattr(f, "ml_fake_probability", None) is not None)
    summary: dict = {
        "total_reviews": total,
        "suspicious": counts.get("suspicious", 0),
        "genuine_looking": counts.get("genuine-looking", 0),
        "needs_review": counts.get("needs-review", 0),
        "suspicious_share": round(counts.get("suspicious", 0) / total, 3) if total else None,
        "scored_by_text_model": scored,
        "text_model": None,
    }
    meta = ml_detector.metadata()
    if meta:
        summary["text_model"] = {
            "name": meta.get("model"),
            "trained_on": meta.get("dataset"),
            "trained_at": meta.get("created_at"),
            "flag_threshold": meta["thresholds"]["suspicious_at"],
            "caveat": meta.get("caveat"),
        }
    return summary
