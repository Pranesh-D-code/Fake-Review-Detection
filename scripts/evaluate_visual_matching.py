"""Measure visual matching on the pairs from scripts/build_visual_pairs.py and write the calibration the app uses.

    python scripts/evaluate_visual_matching.py

Compared on identical pairs (photos that the *text* matcher judged the same product, or clearly a different one):

* legacy      - what the app used before: 64x64 resize + perceptual hash
* phash_trim  - perceptual hash after trimming the plain background
* dino_cls / dino_mean / dino_concat - DINOv2-small embeddings (class token, patch average, both) cosine
* combined    - cosine + colour + hash, one small logistic model (5-fold, grouped by product so a product is never
                in both the training and the test fold)

Scores are out-of-fold. The chosen method's mapping to a probability and its "same" / "different" thresholds go to
``models/visual/calibration.json``; all numbers go to ``models/visual/evaluation.json``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import visual_match as vm  # noqa: E402

DATA = ROOT / "data" / "visual_eval"
OUT = ROOT / "models" / "visual"


def cached_image(url: str):
    path = DATA / "images" / (hashlib.sha1(url.encode()).hexdigest() + ".png")
    if path.exists():
        image = Image.open(path).convert("RGB")
        image.thumbnail((vm.MAX_SIDE, vm.MAX_SIDE))
        return image
    image = vm.load_image(url)
    if image is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    return image


_parts: dict = {}


def parts_for(url: str, image: Image.Image):
    if url not in _parts:
        _parts[url] = vm.embed_parts(image)
    return _parts[url]


def legacy_score(a: Image.Image, b: Image.Image) -> float:
    import imagehash
    h1 = imagehash.phash(a.resize((64, 64)))
    h2 = imagehash.phash(b.resize((64, 64)))
    return max(0.0, 1.0 - (h1 - h2) / 64.0)


def summarise(name: str, y: np.ndarray, score: np.ndarray, hard: np.ndarray) -> dict:
    out = {"auc": round(float(roc_auc_score(y, score)), 4)}
    hard_rows = (y == 1) | hard
    out["auc_vs_hard_negatives"] = round(float(roc_auc_score(y[hard_rows], score[hard_rows])), 4) if len(set(y[hard_rows])) == 2 else None
    best = max(((float(((score >= t) == (y == 1)).mean()), float(t)) for t in np.unique(score)), default=(0.0, 0.0))
    out["best_accuracy"], out["at_threshold"] = round(best[0], 4), round(best[1], 4)
    print(f"  {name:14} AUC {out['auc']:.4f} | vs hard negatives {out['auc_vs_hard_negatives']} | best accuracy {out['best_accuracy']:.3f} @ {out['at_threshold']:.3f}")
    return out


def main() -> int:
    pairs = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    urls = sorted({u for pair in pairs for u in (pair["source_image"], pair["candidate_image"])})
    print(f"downloading {len(urls)} photos (cached ones are skipped)...", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(cached_image, urls))
    rows = []
    for index, pair in enumerate(pairs):
        a, b = cached_image(pair["source_image"]), cached_image(pair["candidate_image"])
        if a is None or b is None:
            continue
        pa, pb = parts_for(pair["source_image"], a), parts_for(pair["candidate_image"], b)
        if pa is None or pb is None:
            print("model missing: run scripts/fetch_visual_model.py")
            return 1
        ta, tb = vm.trim_to_product(a), vm.trim_to_product(b)
        why = pair.get("why", "")
        rows.append({
            "seed": pair["seed"], "label": pair["label"], "hard": pair["label"] == 0 and ("different model" in why or "different product name" in why or "model number" in why),
            "legacy": legacy_score(a, b), "phash_trim": vm._phash_score(a, b), "colour": vm.colour_similarity(ta, tb),
            "cls": float(np.dot(pa[0], pb[0])), "mean": float(np.dot(pa[1], pb[1])),
        })
        rows[-1]["concat"] = float((rows[-1]["cls"] + rows[-1]["mean"]) / 2)
    (DATA / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    y = np.array([r["label"] for r in rows])
    hard = np.array([r["hard"] for r in rows])
    groups = np.array([r["seed"] for r in rows])
    print(f"{len(rows)} usable pairs: {int(y.sum())} same product, {int((1 - y).sum())} different ({int(hard.sum())} are same-category look-alikes)")
    if y.sum() < 10 or (1 - y).sum() < 10:
        print("not enough pairs")
        return 1

    col = lambda key: np.array([r[key] for r in rows])
    report = {"pairs": len(rows), "same": int(y.sum()), "different": int((1 - y).sum()), "hard_negatives": int(hard.sum()), "methods": {}}
    for name, key in (("legacy", "legacy"), ("phash_trim", "phash_trim"), ("colour", "colour"), ("dino_cls", "cls"), ("dino_mean", "mean"), ("dino_concat", "concat")):
        report["methods"][name] = summarise(name, y, col(key), hard)

    # out-of-fold combined model: cosine + colour + hash
    features = np.column_stack([col("concat"), col("colour"), col("phash_trim")])
    oof = np.zeros(len(rows))
    folds = min(5, len(set(groups)))
    for train, test in GroupKFold(n_splits=folds).split(features, y, groups):
        model = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced").fit(features[train], y[train])
        oof[test] = model.predict_proba(features[test])[:, 1]
    report["methods"]["combined_oof"] = summarise("combined_oof", y, oof, hard)

    # calibration of the single cosine score (also out-of-fold, for the thresholds)
    cos = col("concat").reshape(-1, 1)
    oof_cos = np.zeros(len(rows))
    for train, test in GroupKFold(n_splits=folds).split(cos, y, groups):
        oof_cos[test] = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced").fit(cos[train], y[train]).predict_proba(cos[test])[:, 1]
    final_cos = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced").fit(cos, y)
    combined_model = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced").fit(features, y)

    use_combined = report["methods"]["combined_oof"]["auc"] >= report["methods"]["dino_concat"]["auc"] + 0.01
    chosen_scores = oof if use_combined else oof_cos
    positives, negatives = np.sort(chosen_scores[y == 1]), chosen_scores[y == 0]
    # "different": at most 5% of real same-product pairs may fall below it. "same": at most 3% of different-product pairs may reach it.
    different_below = float(np.quantile(positives, 0.05))
    same_at = float(np.quantile(negatives, 0.97))
    hard_scores = chosen_scores[hard]
    report["chosen"] = "dino combined (cosine+colour+hash)" if use_combined else "dino cosine"
    report["same_at"], report["different_below"] = round(same_at, 4), round(different_below, 4)
    report["at_those_thresholds"] = {
        "same_pairs_called_same": round(float((positives >= same_at).mean()), 4),
        "different_pairs_called_same": round(float((negatives >= same_at).mean()), 4),
        "same_pairs_called_different": round(float((positives <= different_below).mean()), 4),
        "different_pairs_called_different": round(float((negatives <= different_below).mean()), 4),
        "look_alike_pairs_called_same": round(float((hard_scores >= same_at).mean()), 4) if len(hard_scores) else None,
        "look_alike_pairs_called_different": round(float((hard_scores <= different_below).mean()), 4) if len(hard_scores) else None,
    }
    print("chosen:", report["chosen"], "| same_at", report["same_at"], "| different_below", report["different_below"], "\n", json.dumps(report["at_those_thresholds"]))

    calibration = {"dinov2": {"slope": round(float(final_cos.coef_[0][0]), 4), "intercept": round(float(final_cos.intercept_[0]), 4),
                              "same_at": report["same_at"], "different_below": report["different_below"]},
                   "phash": {"same_at": 0.9, "different_below": 0.6}}
    if use_combined:
        calibration["dinov2"]["combiner"] = {"cosine": round(float(combined_model.coef_[0][0]), 4), "colour": round(float(combined_model.coef_[0][1]), 4),
                                             "phash": round(float(combined_model.coef_[0][2]), 4), "intercept": round(float(combined_model.intercept_[0]), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    report["note"] = ("Labels come from the text matcher (brand / model code / specs), not from a person. The two signals are independent, "
                      "so agreement is evidence both are right, but this is not hand-checked ground truth.")
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", OUT / "calibration.json", "and evaluation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
