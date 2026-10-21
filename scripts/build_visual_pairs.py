"""Collect image pairs from live stores to measure visual matching.

For each seed product the other stores are searched, and every listing is judged by the *text* matcher alone:

* positive - the text matcher verified it (same brand, model code and specs)
* negative - the text matcher found a clear difference (another model, an accessory, another product type/name)
* skipped  - the same family in another storage / size / variant, which looks alike but is not the same listing

Photos are then compared by independent methods in ``scripts/evaluate_visual_matching.py``. Agreement between two
independent signals is evidence both are right; it is not hand-checked ground truth, and the report says so.

    python scripts/build_visual_pairs.py            # writes data/visual_eval/pairs.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.matcher import score_candidate                      # noqa: E402
from app.multistore import _run_node, select_stores          # noqa: E402
from app.product_identity import build_queries, parse_identity  # noqa: E402
from app.source_extractor import SourceReadError, extract_source_product  # noqa: E402

SEEDS = [
    "samsung galaxy m36 5g", "apple iphone 15 128gb", "boat airdopes 141", "sony wh-1000xm5 headphones",
    "nike revolution 7 running shoes", "adidas ultraboost running shoes", "puma smash sneakers", "wildcraft backpack 35l",
    "maybelline fit me foundation", "lakme 9to5 lipstick", "lg 1.5 ton split ac", "samsung 43 inch smart tv",
    "hp pavilion 15 laptop", "prestige induction cooktop", "philips air fryer", "titan analog watch men",
]
NEGATIVE_REASONS = ("different model", "accessory listing", "different product name", "different product type", "different model number")


def first_flipkart_listing(query: str) -> str | None:
    found = _run_node("search", {"queries": [query], "stores": ["Flipkart"], "per_store": 4}, timeout=70)
    for item in (found or {}).get("matches", []):
        if item.get("price") and item.get("url"):
            return item["url"].split("?")[0]
    return None


def main() -> int:
    out_dir = ROOT / "data" / "visual_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []
    for query in SEEDS:
        url = first_flipkart_listing(query)
        if not url:
            print(f"- {query}: no Flipkart listing")
            continue
        try:
            source = extract_source_product(url)
        except SourceReadError as exc:
            print(f"- {query}: cannot read {url} ({exc})")
            continue
        if not source.get("image_url") or not source.get("title"):
            print(f"- {query}: source has no title/image")
            continue
        identity = parse_identity(source["title"], source.get("brand"))
        stores = [s for s in select_stores("Flipkart", url, identity.category, limit=10) if s != "Flipkart"]
        searched = _run_node("search", {"queries": build_queries(identity), "stores": stores, "per_store": 8}, timeout=120)
        if not searched:
            print(f"- {query}: store search failed")
            continue
        counts = {"positive": 0, "negative": 0, "skipped": 0}
        for item in searched.get("matches", []):
            if not item.get("image_url") or not item.get("title"):
                continue
            result = score_candidate(identity, item["title"], item["url"], source.get("price"), item.get("price"))
            flags = " | ".join(result.flags)
            if result.status == "verified":
                label = 1
            elif result.status == "rejected" and any(reason in flags for reason in NEGATIVE_REASONS):
                label = 0
            else:
                counts["skipped"] += 1
                continue
            counts["positive" if label else "negative"] += 1
            pairs.append({"seed": query, "source_title": source["title"], "source_url": url, "source_image": source["image_url"],
                          "store": item["platform"], "candidate_title": item["title"], "candidate_url": item["url"],
                          "candidate_image": item["image_url"], "label": label, "text_score": result.score, "why": flags[:160]})
        print(f"- {query}: {len(stores)} stores -> {counts}")
        (out_dir / "pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    print(f"{len(pairs)} pairs ({sum(p['label'] for p in pairs)} positive) -> {out_dir / 'pairs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
