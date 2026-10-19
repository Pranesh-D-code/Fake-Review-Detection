"""Score the hand-written product pairs in tests/fixtures/match_pairs.json with app/matcher.py.

The pairs are invented titles written to look like marketplace listings (phones, laptops, TVs/ACs,
footwear, grocery, beauty, accessories, refurbished and pre-owned offers, a few identifier cases).
They were NOT measured on live listings, so the precision and recall printed here are a regression
guard for the matcher's rules, not an accuracy figure for real stores.

Metrics (for the status "verified", the only one that feeds the "lowest verified match"):
    precision = pairs correctly verified / pairs the matcher verified
    recall    = pairs correctly verified / pairs labelled verified
and the exact-status agreement over all pairs (verified / candidate / rejected).

Usage:
    python scripts/evaluate_matcher.py
    python scripts/evaluate_matcher.py --fixtures other.json --strict   # exit 1 on an unexpected miss
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.matcher import score_candidate          # noqa: E402
from app.product_identity import parse_identity  # noqa: E402

DEFAULT_FIXTURES = ROOT / "tests" / "fixtures" / "match_pairs.json"
STATUSES = ("verified", "candidate", "rejected")
DISCLAIMER = ("NOTE: these pairs are hand-written, not measured on live listings. The figures below are a "
              "regression guard for the matcher's rules, not a live accuracy number.")


def load_pairs(path: Path = DEFAULT_FIXTURES) -> list[dict]:
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    for pair in pairs:
        if pair["expected"] not in STATUSES:
            raise ValueError(f"{pair['id']}: expected must be one of {STATUSES}")
    return pairs


def judge(pair: dict):
    """Run the matcher on one fixture pair."""
    return score_candidate(
        parse_identity(pair["source_title"], pair.get("source_brand")),
        pair["candidate_title"], pair.get("candidate_url"),
        pair.get("source_price"), pair.get("candidate_price"),
        source_gtin=pair.get("source_gtin"), candidate_gtin=pair.get("candidate_gtin"),
        source_mpn=pair.get("source_mpn"), candidate_mpn=pair.get("candidate_mpn"),
    )


def evaluate(pairs: list[dict]) -> dict:
    rows = []
    for pair in pairs:
        result = judge(pair)
        rows.append({"pair": pair, "got": result.status, "score": result.score, "flags": result.flags, "reasons": result.reasons})

    expected_verified = [r for r in rows if r["pair"]["expected"] == "verified"]
    predicted_verified = [r for r in rows if r["got"] == "verified"]
    true_verified = [r for r in predicted_verified if r["pair"]["expected"] == "verified"]
    precision = len(true_verified) / len(predicted_verified) if predicted_verified else 1.0
    recall = len(true_verified) / len(expected_verified) if expected_verified else 1.0
    exact = sum(1 for r in rows if r["got"] == r["pair"]["expected"])

    misses = [r for r in rows if r["got"] != r["pair"]["expected"]]
    # a wrong "verified" is the harmful kind: it can become the "lowest verified match"
    false_verified = [r for r in misses if r["got"] == "verified"]
    missed_verified = [r for r in misses if r["pair"]["expected"] == "verified"]

    by_category: dict[str, dict] = defaultdict(lambda: {"pairs": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        stats = by_category[r["pair"]["category"]]
        stats["pairs"] += 1
        stats["exact"] += r["got"] == r["pair"]["expected"]
        stats["tp"] += r["got"] == "verified" and r["pair"]["expected"] == "verified"
        stats["fp"] += r["got"] == "verified" and r["pair"]["expected"] != "verified"
        stats["fn"] += r["got"] != "verified" and r["pair"]["expected"] == "verified"

    return {
        "rows": rows, "misses": misses, "false_verified": false_verified, "missed_verified": missed_verified,
        "precision": precision, "recall": recall, "exact": exact / len(rows) if rows else 1.0,
        "confusion": Counter((r["pair"]["expected"], r["got"]) for r in rows),
        "by_category": dict(by_category),
        "counts": {"pairs": len(rows), "expected_verified": len(expected_verified), "predicted_verified": len(predicted_verified),
                   "true_verified": len(true_verified)},
    }


def _fmt(value: float) -> str:
    return f"{value:.1%}"


def report(summary: dict) -> str:
    counts = summary["counts"]
    lines = [DISCLAIMER, "",
             f"pairs: {counts['pairs']}   labelled verified: {counts['expected_verified']}   "
             f"matcher said verified: {counts['predicted_verified']}   correctly verified: {counts['true_verified']}",
             f"verified precision: {_fmt(summary['precision'])}   verified recall: {_fmt(summary['recall'])}   "
             f"exact-status agreement: {_fmt(summary['exact'])}", "",
             "confusion (rows = label, columns = matcher):",
             "  " + " " * 10 + "".join(f"{s:>11}" for s in STATUSES)]
    for expected in STATUSES:
        lines.append("  " + f"{expected:<10}" + "".join(f"{summary['confusion'].get((expected, got), 0):>11}" for got in STATUSES))
    lines += ["", "by category:"]
    for category, stats in sorted(summary["by_category"].items()):
        lines.append(f"  {category:<12} pairs {stats['pairs']:>3}  exact {stats['exact']:>3}  "
                     f"verified: ok {stats['tp']:>2}  wrongly verified {stats['fp']:>2}  missed {stats['fn']:>2}")

    def listing(title, rows):
        out = ["", f"{title} ({len(rows)}):"]
        if not rows:
            out.append("  none")
        for r in rows:
            pair = r["pair"]
            gap = f"  [known gap: {pair['known_gap']}]" if pair.get("known_gap") else ""
            out.append(f"  - {pair['id']}: expected {pair['expected']}, got {r['got']} (score {r['score']}){gap}")
            out.append(f"      source:    {pair['source_title'][:110]}")
            out.append(f"      candidate: {pair['candidate_title'][:110]}")
            if r["flags"]:
                out.append(f"      flags:     {'; '.join(r['flags'])[:200]}")
        return out

    lines += listing("WRONGLY VERIFIED (harmful: could become the lowest verified match)", summary["false_verified"])
    lines += listing("labelled verified but not verified (lost recall)", summary["missed_verified"])
    others = [r for r in summary["misses"] if r not in summary["false_verified"] and r not in summary["missed_verified"]]
    lines += listing("other status disagreements (candidate vs rejected)", others)
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def unexpected_misses(summary: dict) -> list[dict]:
    return [r for r in summary["misses"] if not r["pair"].get("known_gap")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--strict", action="store_true", help="exit 1 when a pair without a known_gap is missed")
    args = parser.parse_args(argv)
    summary = evaluate(load_pairs(args.fixtures))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report(summary))
    return 1 if args.strict and unexpected_misses(summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
