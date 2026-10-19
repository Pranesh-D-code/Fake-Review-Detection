"""Train and evaluate the fake-review text classifier used by TrustEngine.

The model is trained on the externally labelled reviews that
``scripts/import_external_reviews.py`` stored in ``trustengine.db``. Which raw
label means "fake" is read from the ``dataset_label_mappings`` table, never
guessed. Only text in the labelled dataset is used; nothing is fabricated.

Outputs (default ``models/``):
    review_detector.joblib       model bundle loaded by ``app/ml_detector.py``
    review_detector.metrics.json held-out metrics and the training provenance

Usage:
    python scripts/train_review_classifier.py
    python scripts/train_review_classifier.py --dataset archive.zip --seed 7

What the metrics mean: they measure how well the model separates *this
dataset's* original reviews from its computer-generated reviews on unseen rows.
They are not an accuracy guarantee for live marketplace reviews. The
leave-one-category-out numbers are included as a rough domain-shift probe.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAKE_NORMALIZED_LABEL = "computer-generated-fake"
ORIGINAL_NORMALIZED_LABEL = "original"


def load_dataset(db_path: Path, dataset: str | None):
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        if dataset is None:
            row = db.execute(
                "SELECT external_dataset FROM review_observations "
                "WHERE external_dataset IS NOT NULL GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchone()
            if not row:
                raise SystemExit("No external dataset found. Run scripts/import_external_reviews.py first.")
            dataset = row[0]
        mapping = {
            raw: normalized for raw, normalized in db.execute(
                "SELECT raw_label, normalized_label FROM dataset_label_mappings WHERE dataset_name = ?",
                (dataset,),
            )
        }
        if set(mapping.values()) != {FAKE_NORMALIZED_LABEL, ORIGINAL_NORMALIZED_LABEL}:
            raise SystemExit(
                f"dataset_label_mappings for '{dataset}' must map to exactly "
                f"'{ORIGINAL_NORMALIZED_LABEL}' and '{FAKE_NORMALIZED_LABEL}'; found {mapping}."
            )
        rows = db.execute(
            "SELECT review_hash, review_text, sku, external_label FROM review_observations "
            "WHERE external_dataset = ?", (dataset,),
        ).fetchall()
    finally:
        db.close()

    seen, texts, labels, categories = set(), [], [], []
    for review_hash, text, category, raw_label in rows:
        if review_hash in seen or raw_label not in mapping or not text or not text.strip():
            continue
        seen.add(review_hash)
        texts.append(text.strip())
        labels.append(1 if mapping[raw_label] == FAKE_NORMALIZED_LABEL else 0)
        categories.append(category or "unclassified")
    return dataset, mapping, texts, np.array(labels), np.array(categories)


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def load_live_labels(db_path: Path):
    """Reviews a person marked 'real' (0) or 'fake' (1) in the app, with their text."""
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT l.review_hash, l.human_label, "
            "(SELECT o.review_text FROM review_observations o WHERE o.review_hash = l.review_hash LIMIT 1) "
            "FROM review_labels l"
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        db.close()
    texts, labels = [], []
    for _, human, text in rows:
        if human in ("real", "fake") and text and text.strip():
            texts.append(text.strip())
            labels.append(1 if human == "fake" else 0)
    return texts, np.array(labels, dtype=int)


def build_pipeline(c_value: float, max_features: int) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            max_features=max_features, strip_accents=None, lowercase=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
            max_features=max_features, lowercase=True,
        )),
    ])
    classifier = LogisticRegression(C=c_value, solver="liblinear", max_iter=400)
    return Pipeline([("features", features), ("clf", classifier)])


def metrics_at(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision_fake": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall_fake": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1_fake": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def full_metrics(y_true, prob) -> dict:
    out = metrics_at(y_true, prob, 0.5)
    out["roc_auc"] = round(float(roc_auc_score(y_true, prob)), 4)
    out["pr_auc"] = round(float(average_precision_score(y_true, prob)), 4)
    out["brier"] = round(float(brier_score_loss(y_true, prob)), 4)
    return out


def threshold_for_precision(y_true, prob, target: float) -> float:
    """Lowest threshold whose precision on the validation set reaches ``target``."""
    for threshold in np.round(np.arange(0.50, 0.995, 0.005), 3):
        flagged = prob >= threshold
        if flagged.sum() >= 30 and precision_score(y_true, flagged.astype(int), zero_division=0) >= target:
            return float(threshold)
    return 0.99


def threshold_for_genuine(y_true, prob, target: float) -> float:
    """Highest threshold below which at least ``target`` of the rows are original."""
    for threshold in np.round(np.arange(0.50, 0.01, -0.005), 3):
        kept = prob <= threshold
        if kept.sum() >= 30 and (1 - y_true[kept].mean()) >= target:
            return float(threshold)
    return 0.05


def heuristic_baseline(texts, labels) -> dict:
    """Score the existing rule-based classifier on the same held-out rows."""
    from app.models import Review
    from app.nlp_engine import ReviewIntegrityClassifier

    pred = []
    for text in texts:
        review = Review(text=text, rating=None)
        classification = ReviewIntegrityClassifier.predict(review)[5]
        pred.append(1 if classification == "suspicious" else 0)
    pred = np.array(pred)
    return {
        "accuracy": round(float(accuracy_score(labels, pred)), 4),
        "precision_fake": round(float(precision_score(labels, pred, zero_division=0)), 4),
        "recall_fake": round(float(recall_score(labels, pred, zero_division=0)), 4),
        "f1_fake": round(float(f1_score(labels, pred, zero_division=0)), 4),
        "flagged_share": round(float(pred.mean()), 4),
    }


def top_terms(pipeline: Pipeline, count: int = 15) -> dict:
    word = pipeline.named_steps["features"].transformer_list[0][1]
    names = np.array(word.get_feature_names_out())
    coefficients = pipeline.named_steps["clf"].coef_[0][: len(names)]
    order = np.argsort(coefficients)
    return {
        "toward_original": [names[i] for i in order[:count]],
        "toward_computer_generated": [names[i] for i in order[::-1][:count]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=ROOT / "trustengine.db")
    parser.add_argument("--dataset", default=None, help="external_dataset name (default: the largest one)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=150_000, help="per vectoriser")
    parser.add_argument("--suspicious-precision", type=float, default=0.98,
                        help="validation precision required before a review is called 'suspicious'")
    parser.add_argument("--genuine-purity", type=float, default=0.98,
                        help="validation share of original reviews required below the 'genuine-looking' cut-off")
    parser.add_argument("--jobs", type=int, default=4, help="parallel model fits")
    parser.add_argument("--no-live-labels", action="store_true",
                        help="train on the dataset only, ignoring reviews a person labelled in the app")
    parser.add_argument("--live-weight", type=float, default=5.0,
                        help="how much more each person-labelled review counts than a dataset row")
    parser.add_argument("--min-live-labels", type=int, default=10,
                        help="ignore person labels until at least this many exist (a handful only adds noise)")
    parser.add_argument("--holdout-categories", type=int, default=3,
                        help="categories to hold out in the domain-shift probe (0 disables it)")
    args = parser.parse_args()

    started = time.time()
    dataset, mapping, texts, labels, categories = load_dataset(args.db, args.dataset)
    print(f"Dataset '{dataset}': {len(texts)} unique reviews, {int(labels.sum())} fake, "
          f"{int((1 - labels).sum())} original. Label mapping: {mapping}")

    sources = np.array(["dataset"] * len(texts))
    weights = np.ones(len(texts))
    live_available = live_used = 0
    if not args.no_live_labels:
        live_texts, live_labels = load_live_labels(args.db)
        known = {normalise_text(t) for t in texts}            # a dataset text must not leak into the live set
        keep = [i for i, t in enumerate(live_texts) if normalise_text(t) not in known]
        live_available = len(keep)
        picked = live_labels[keep] if keep else np.array([], dtype=int)
        enough = live_available >= args.min_live_labels and int(picked.sum()) >= 2 and int((1 - picked).sum()) >= 2
        if enough:
            texts = texts + [live_texts[i] for i in keep]
            labels = np.concatenate([labels, picked])
            categories = np.concatenate([categories, np.array(["live"] * live_available)])
            sources = np.concatenate([sources, np.array(["live"] * live_available)])
            weights = np.concatenate([weights, np.full(live_available, args.live_weight)])
            live_used = live_available
            print(f"Added {live_used} person-labelled reviews ({int(picked.sum())} fake, {int((1 - picked).sum())} real), weight x{args.live_weight}")
        else:
            print(f"Person-labelled reviews not used yet: {live_available} available (need {args.min_live_labels}+ with 2+ of each class).")

    indices = np.arange(len(texts))
    train_idx, rest_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=args.seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=labels[rest_idx], random_state=args.seed)
    take = lambda idx: ([texts[i] for i in idx], labels[idx])
    x_train, y_train = take(train_idx)
    x_val, y_val = take(val_idx)
    x_test, y_test = take(test_idx)
    w_train, w_val = weights[train_idx], weights[val_idx]
    print(f"Split: train {len(x_train)}, validation {len(x_val)}, test {len(x_test)}")

    # Choose C on the validation set only (fits run in parallel).
    def fit_candidate(c_value: float):
        fitted = build_pipeline(c_value, args.max_features).fit(x_train, y_train, clf__sample_weight=w_train)
        prob = fitted.predict_proba(x_val)[:, 1]
        return f1_score(y_val, (prob >= 0.5).astype(int)), c_value, fitted, prob

    candidates = Parallel(n_jobs=args.jobs)(delayed(fit_candidate)(c) for c in (2.0, 8.0, 32.0))
    for score, c_value, _, _ in candidates:
        print(f"  C={c_value:<5} validation F1={score:.4f}")
    best = max(candidates, key=lambda item: item[0])
    _, chosen_c, pipeline, val_prob = best
    print(f"Chosen C={chosen_c}")

    # Decision thresholds come from validation data only (never the test set):
    # flagging a real review as fake is the costlier error, so "suspicious"
    # requires high validation precision, and "genuine-looking" requires a high
    # share of original rows below the cut-off; everything between is "needs-review".
    thresholds = {
        "suspicious_at": threshold_for_precision(y_val, val_prob, args.suspicious_precision),
        "genuine_below": threshold_for_genuine(y_val, val_prob, args.genuine_purity),
    }
    print(f"Thresholds from validation: {thresholds}")

    test_prob = pipeline.predict_proba(x_test)[:, 1]
    results = {
        "test_at_0.5": full_metrics(y_test, test_prob),
        "test_at_suspicious_threshold": metrics_at(y_test, test_prob, thresholds["suspicious_at"]),
        "validation_at_0.5": full_metrics(y_val, val_prob),
    }
    genuine_mask = test_prob <= thresholds["genuine_below"]
    results["test_genuine_band"] = {
        "share_of_rows": round(float(genuine_mask.mean()), 4),
        "original_share_within_band": round(float(1 - y_test[genuine_mask].mean()), 4) if genuine_mask.any() else None,
    }
    mid = ~genuine_mask & (test_prob < thresholds["suspicious_at"])
    results["test_uncertain_band_share"] = round(float(mid.mean()), 4)

    dataset_rows = sources[test_idx] == "dataset"
    results["test_dataset_only"] = full_metrics(y_test[dataset_rows], test_prob[dataset_rows])
    live_rows = ~dataset_rows
    if live_rows.sum() >= 5 and len(set(y_test[live_rows])) == 2:
        results["live_test"] = {"rows": int(live_rows.sum()), **metrics_at(y_test[live_rows], test_prob[live_rows], 0.5)}
    else:
        results["live_test"] = {"rows": int(live_rows.sum()),
                                "note": "too few person-labelled reviews in the test split to measure live accuracy"}

    print("Scoring the existing rule-based baseline on the same test rows...")
    results["rule_based_baseline_on_test"] = heuristic_baseline(x_test, y_test)

    shift = {}
    if args.holdout_categories > 0:
        sizes = sorted(set(categories), key=lambda c: -int((categories == c).sum()))
        held_out = [c for c in sizes if (categories == c).sum() >= 1000][: args.holdout_categories]
        def probe(category: str):
            mask = categories == category
            sub_train = np.where(~mask)[0]
            sub_test = np.where(mask)[0]
            shifted = build_pipeline(chosen_c, args.max_features).fit(
                [texts[i] for i in sub_train], labels[sub_train], clf__sample_weight=weights[sub_train])
            prob = shifted.predict_proba([texts[i] for i in sub_test])[:, 1]
            return category, {"rows": int(mask.sum()), **full_metrics(labels[sub_test], prob)}

        for category, outcome in Parallel(n_jobs=args.jobs)(delayed(probe)(c) for c in held_out):
            shift[category] = outcome
            print(f"  hold out {category:<30} accuracy={outcome['accuracy']:.4f} f1={outcome['f1_fake']:.4f}")
    results["leave_one_category_out"] = shift

    # Ship the model trained on train + validation (the test rows stay unseen).
    final = build_pipeline(chosen_c, args.max_features).fit(
        x_train + x_val, np.concatenate([y_train, y_val]), clf__sample_weight=np.concatenate([w_train, w_val]))
    words = np.array([len(t.split()) for t in texts])

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "dataset": dataset,
        "label_mapping": mapping,
        "positive_class": FAKE_NORMALIZED_LABEL,
        "rows": {"total": len(texts), "train": len(x_train), "validation": len(x_val), "test": len(x_test), "live": live_used},
        "live_labels_total": live_available,             # person labels that existed when this model was trained
        "live_labels_in_training": live_used,
        "live_weight": args.live_weight if live_used else None,
        "model": f"TF-IDF(word 1-2 + char_wb 3-5) + LogisticRegression(C={chosen_c})",
        "min_words_for_ml": int(max(6, np.percentile(words, 2))),
        "thresholds": thresholds,
        "seed": args.seed,
        "caveat": (
            "Trained to separate original from computer-generated reviews in one labelled English "
            "dataset" + (f" plus {live_used} reviews a person labelled real or fake in this app" if live_used else "") + ". "
            "Output is a text-style likelihood, not proof that a review is fake, and it "
            + ("was checked on only a handful of person-labelled live reviews." if live_used
               else "has not been validated on live marketplace reviews.")
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out_dir / "review_detector.joblib"
    joblib.dump({"pipeline": final, "meta": meta}, bundle_path, compress=3)

    report = {**meta, "metrics": results, "top_terms": top_terms(final),
              "training_seconds": round(time.time() - started, 1)}
    metrics_path = args.out_dir / "review_detector.metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    headline = results["test_at_0.5"]
    print("\n=== Held-out test (never used for fitting or threshold choice) ===")
    print(f"accuracy {headline['accuracy']}  precision {headline['precision_fake']}  recall {headline['recall_fake']}  "
          f"F1 {headline['f1_fake']}  ROC-AUC {headline['roc_auc']}  Brier {headline['brier']}")
    print(f"rule-based baseline on same rows: {results['rule_based_baseline_on_test']}")
    print(f"Saved {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB) and {metrics_path.name}")


if __name__ == "__main__":
    main()
