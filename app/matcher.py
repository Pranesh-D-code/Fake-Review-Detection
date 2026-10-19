"""Attribute-aware scoring of a cross-store candidate against the source product.

The legacy matcher (``engines.candidate_links``) compares title words and a
perceptual image hash. That is enough between Amazon and Flipkart, but across
other stores it lets through accessories ("... Samsung Galaxy M34 Back Case"),
other storage sizes, other pack sizes and sibling models. This module scores:

* name coverage   - how many of the source's leading words the candidate repeats
* brand           - same brand present in the candidate text
* model codes     - alphanumeric codes such as ``m34`` / ``sm-m346b`` agree
* hard specs      - storage, weight, volume, screen size, tonnage, pack count
* variants        - ``Pro`` / ``Ultra`` / ``FE`` words agree on both sides
* gender, price sanity and accessory detection
* condition       - refurbished / pre-owned / open-box never verifies against a new source
* identifiers     - equal GTIN/EAN or MPN is strong evidence, conflicting ones veto; absent ones change nothing
* image similarity as supporting evidence only

A candidate is "verified" only when the evidence is strong *and* nothing
contradicts it. The reasons are returned so the UI can show why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from urllib.parse import unquote, urlparse

from .product_identity import GENERIC_NAME_WORDS, STOPWORDS, STRONG_UNITS, ProductIdentity, normalise, parse_identity

VERIFIED_SCORE = 0.72
CANDIDATE_SCORE = 0.45

# Added to a match's flags when the store lists the product as unavailable (set by app/multistore.py;
# app/engines.py reads it so such a match is never reported as the lowest verified price).
OUT_OF_STOCK_FLAG = "listed as out of stock"

_CONDITION_LABELS = {"new": "new", "refurbished": "refurbished", "open_box": "open-box", "used": "pre-owned"}


@dataclass
class MatchResult:
    score: float
    status: str                       # "verified" | "candidate" | "rejected"
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    title_similarity: float = 0.0
    condition: str = "new"            # the candidate's own condition: "new" | "refurbished" | "open_box" | "used"


def text_similarity(a: str, b: str) -> float:
    """Same measure the legacy matcher shows as "Text Match": best of word overlap and sequence ratio."""
    words = lambda t: {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) > 1}
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    jaccard = len(wa & wb) / max(1, len(wa | wb))
    return round(max(jaccard, SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()), 3)


def url_slug_text(url: str | None) -> str:
    """Readable words from the product URL path (stores often keep the full name there)."""
    if not url:
        return ""
    segments = [s for s in unquote(urlparse(url).path).split("/") if re.search(r"[a-z]+-[a-z0-9]+-", s, re.I)]
    # "6-7-inch" in a URL means 6.7 inch, "1-5-ton" means 1.5 ton
    fixed = [re.sub(r"(?<!\d)(\d)-(\d)(?=-(?:inch|ton|tons|kg|l|ltr|litre|cm|mm)\b)", r"\1.\2", s) for s in segments]
    return " ".join(re.sub(r"[-_]+", " ", s) for s in fixed)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalise(text))


def _code_present(code: str, haystack_words: list[str], compact_text: str) -> bool:
    """Exact code, or the source code as the prefix of a longer code (m34 -> m346b)."""
    if code in compact_text:
        return True
    for word in haystack_words:
        flat = word.replace("-", "")
        if len(code) >= 3 and flat.startswith(code) and len(flat) - len(code) <= 2 and re.search(r"\d", flat):
            return True
    return False


def _name_coverage(source: ProductIdentity, cand_words: set[str], cand_compact: str) -> float:
    head = [w for w in source.head_tokens if len(w) > 1]
    if not head:
        return 0.0
    hits = 0
    for word in head:
        flat = word.replace("-", "")
        if word in cand_words or flat in cand_compact:
            hits += 1
    return hits / len(head)


def _spec_check(source: ProductIdentity, cand: ProductIdentity) -> tuple[float, list[str], list[str]]:
    """Return (agreement 0..1, reasons, mismatch descriptions) over units both titles state."""
    reasons: list[str] = []
    mismatches: list[str] = []
    strong = set(STRONG_UNITS)
    if (source.types | cand.types) & {"tv", "laptop", "tablet"}:
        strong.add("inch")                                   # screen size only identifies screens
    compared = agreed = 0
    for unit, wanted in source.specs.items():
        found = cand.specs.get(unit)
        if not found or unit not in strong:
            continue
        compared += 1
        if unit in ("ram_gb", "storage_gb"):
            overlap = bool(wanted & found)                   # "RAM Plus"/expandable storage add extra figures
        else:
            overlap = any(abs(w - f) <= max(0.03 * w, 0.11) for w in wanted for f in found)
        label = {"ram_gb": "RAM", "storage_gb": "storage"}.get(unit, unit)
        if overlap:
            agreed += 1
            reasons.append(f"{label} agrees")
        else:
            mismatches.append(f"{label} differs ({sorted(wanted)} vs {sorted(found)})")
    if not compared:
        return 0.7, reasons, mismatches
    return agreed / compared, reasons, mismatches


def normalise_gtin(value: object) -> str | None:
    """Digits of a valid GTIN-8/12/13/14 (EAN/UPC), zero-padded to 14; None when absent or not a real GTIN.

    Stores sometimes put an internal SKU or a placeholder in the GTIN field, so a value without a valid
    GS1 check digit is treated as "no identifier" rather than as a conflicting one.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    digits = re.sub(r"\D", "", str(value))
    if len(digits) not in (8, 12, 13, 14) or len(set(digits)) == 1:
        return None
    padded = digits.zfill(14)
    total = sum(int(d) * (3 if index % 2 == 0 else 1) for index, d in enumerate(reversed(padded[:-1])))
    return padded if (10 - total % 10) % 10 == int(padded[-1]) else None


def normalise_mpn(value: object) -> str | None:
    """Upper-case letters and digits of a manufacturer part / model number; None when it is too short to mean anything."""
    if value is None or isinstance(value, bool):
        return None
    flat = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return flat if len(flat) >= 4 and re.search(r"\d", flat) else None


def _mpns_agree(a: str, b: str) -> bool:
    """Equal, or one is the other plus a suffix (stores often cut or extend the colour/region suffix)."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 6 and longer.startswith(shorter)


def score_candidate(
    source: ProductIdentity,
    candidate_title: str,
    candidate_url: str | None = None,
    source_price: float | None = None,
    candidate_price: float | None = None,
    image_similarity: float | None = None,
    *,
    source_gtin: object = None,
    candidate_gtin: object = None,
    source_mpn: object = None,
    candidate_mpn: object = None,
) -> MatchResult:
    slug = url_slug_text(candidate_url)
    full_text = f"{candidate_title} {slug}".strip()
    cand = parse_identity(full_text, None)
    cand_words = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", normalise(full_text)))
    cand_word_list = list(cand_words)
    cand_compact = _compact(full_text)

    reasons: list[str] = []
    flags: list[str] = []
    veto: str | None = None            # hard: cannot be the same product
    spec_conflict: str | None = None   # soft: same family, different storage/size/pack

    coverage = _name_coverage(source, cand_words | {w.replace("-", "") for w in cand_words}, cand_compact)
    title_sim = round(SequenceMatcher(None, normalise(" ".join(source.head_tokens)), normalise(candidate_title)).ratio(), 3)

    # brand
    brand_score = 0.5
    if source.brand:
        brand_flat = _compact(source.brand)
        if brand_flat and brand_flat in cand_compact:
            brand_score = 1.0
            reasons.append(f"brand '{source.brand}' present")
        else:
            brand_score = 0.0
            flags.append("brand not found in candidate title")

    # model codes
    model_score = 0.6
    if source.model_codes:
        present = [c for c in source.model_codes if _code_present(c, cand_word_list, cand_compact)]
        model_score = len(present) / len(source.model_codes)
        if present:
            reasons.append("model code " + "/".join(sorted(present)) + " matches")
        if model_score == 0:
            flags.append("model code " + "/".join(sorted(source.model_codes)) + " not found")
            if cand.model_codes:
                veto = "different model (" + "/".join(sorted(cand.model_codes)) + ")"

    # without a model code, the distinctive words ("revolution", "fit me") must carry the match
    if not source.model_codes:
        brand_words = set((source.brand or "").split())
        distinctive = [w for w in source.head_tokens
                       if w not in brand_words and w not in GENERIC_NAME_WORDS and w not in STOPWORDS and (len(w) > 1 or w.isdigit())]
        if len(distinctive) >= 2:
            seen = sum(1 for w in distinctive if w in cand_words or w.replace("-", "") in cand_compact)
            if seen / len(distinctive) < 0.67:
                flags.append("distinctive name words missing: " + ", ".join(w for w in distinctive if w not in cand_words))
                veto = veto or "different product name"

    # hard specs
    spec_score, spec_reasons, spec_mismatches = _spec_check(source, cand)
    reasons.extend(spec_reasons)
    if spec_mismatches:
        spec_conflict = "different specification: " + "; ".join(spec_mismatches)

    score = 0.28 * coverage + 0.17 * brand_score + 0.25 * model_score + 0.15 * spec_score + 0.15 * title_sim

    # variant words (Pro / Ultra / FE ...) must agree
    if source.variants != cand.variants:
        extra = sorted(source.variants ^ cand.variants)
        score -= 0.20
        flags.append("variant differs (" + ", ".join(extra) + ")")
        if {"pro", "max", "ultra", "plus", "fe", "mini", "lite"} & set(extra):
            veto = veto or "different model variant: " + ", ".join(extra)

    # bare model numbers ("iphone 15" vs "iphone 14")
    if source.bare_numbers and cand.bare_numbers and not (source.bare_numbers & cand.bare_numbers):
        score -= 0.20
        flags.append(f"model number differs ({sorted(source.bare_numbers)} vs {sorted(cand.bare_numbers)})")
        veto = veto or "different model number"

    if source.genders and cand.genders and not (source.genders & cand.genders):
        score -= 0.25
        veto = veto or "different target gender"

    if source.types and cand.types and not (source.types & cand.types):
        score -= 0.20
        flags.append("product type differs")
        veto = veto or "different product type"

    # accessories are never the same product
    if cand.is_accessory and not source.is_accessory:
        score = min(score, 0.20)
        veto = "accessory listing, not the product itself"
    elif source.is_accessory and not cand.is_accessory:
        score = min(score, 0.30)
        veto = veto or "source is an accessory; candidate is a main product"

    # condition: a refurbished / pre-owned / open-box unit is another offer from a new one (either direction)
    condition_conflict: str | None = None
    if source.condition != cand.condition:
        pair = f"source is {_CONDITION_LABELS[source.condition]}, candidate is {_CONDITION_LABELS[cand.condition]}"
        flags.append(f"condition differs ({_CONDITION_LABELS[source.condition]} vs {_CONDITION_LABELS[cand.condition]})")
        if "new" in (source.condition, cand.condition):
            score = min(score, 0.30)
            veto = veto or "different condition: " + pair
        else:
            condition_conflict = "different condition: " + pair      # e.g. refurbished vs pre-owned: same unit family, other grade
    elif source.condition != "new":
        reasons.append(f"condition agrees ({_CONDITION_LABELS[source.condition]})")
    if cand.condition != "new":
        flags.append("refurbished/used listing")                 # lets the UI and price rules tell second-hand offers apart

    # identifiers: only compared when both sides state one
    gtin_s, gtin_c = normalise_gtin(source_gtin), normalise_gtin(candidate_gtin)
    mpn_s, mpn_c = normalise_mpn(source_mpn), normalise_mpn(candidate_mpn)
    identifier_match = False
    if gtin_s and gtin_c:
        if gtin_s == gtin_c:
            reasons.append(f"GTIN/EAN {gtin_s.lstrip('0')} matches")
            if brand_score > 0 or coverage >= 0.5:
                identifier_match = True
                score = max(score + 0.30, VERIFIED_SCORE + 0.05)
            else:
                # stores sometimes reuse one GTIN for unrelated listings: with a different brand and
                # almost no shared name words the identifier alone is not enough
                score += 0.10
                flags.append("GTIN/EAN matches but the title does not support it")
        else:
            flags.append(f"GTIN/EAN differs ({gtin_s.lstrip('0')} vs {gtin_c.lstrip('0')})")
            veto = veto or "different GTIN/EAN"
    if mpn_s and mpn_c:
        if _mpns_agree(mpn_s, mpn_c):
            identifier_match = True
            score += 0.10
            reasons.append(f"model/MPN {mpn_s} matches")
        else:
            flags.append(f"model/MPN differs ({mpn_s} vs {mpn_c})")
            veto = veto or "different model/MPN"
    elif mpn_s and not mpn_c and len(mpn_s) >= 6 and mpn_s in cand_compact.upper():
        identifier_match = True                                       # the source's part number is printed in the candidate title/URL
        score += 0.10
        reasons.append(f"model/MPN {mpn_s} appears in candidate title")

    # price sanity
    if source_price and candidate_price:
        ratio = candidate_price / source_price
        if ratio < 0.25 or ratio > 4:
            score -= 0.25
            flags.append(f"price far from source (x{ratio:.2f})")
            veto = veto or "price is implausible for the same product"
        elif ratio < 0.5 or ratio > 2:
            score -= 0.08
            flags.append(f"price differs a lot (x{ratio:.2f})")

    # image similarity only supports; different shots of one product often hash differently
    if image_similarity is not None:
        if image_similarity >= 0.80:
            score += 0.10
            reasons.append(f"image similarity {image_similarity:.0%}")
        elif image_similarity >= 0.70:
            score += 0.05
            reasons.append(f"image similarity {image_similarity:.0%}")

    score = round(max(0.0, min(1.0, score)), 3)
    if veto:
        score = min(score, 0.44)
        flags.append(veto)
    if spec_conflict:
        score = min(score, VERIFIED_SCORE - 0.05)
        flags.append(spec_conflict)
    if condition_conflict:
        score = min(score, VERIFIED_SCORE - 0.05)
        flags.append(condition_conflict)

    identified = identifier_match or bool(source.model_codes and model_score >= 1.0) or (
        not source.model_codes and coverage >= 0.8 and brand_score >= 0.5
    )
    if veto or score < CANDIDATE_SCORE:
        status = "rejected"
    elif score >= VERIFIED_SCORE and identified and not spec_conflict and not condition_conflict:
        status = "verified"
    else:
        status = "candidate"
    # What the card shows as "Text Match": the legacy full-title measure, or the share of the product's
    # key words the listing repeats if that is higher (long marketing titles dilute word overlap).
    shown = round(max(text_similarity(source.title, candidate_title), coverage if source.head_tokens else 0.0), 3)
    return MatchResult(score=score, status=status, reasons=reasons, flags=flags, title_similarity=shown, condition=cand.condition)
