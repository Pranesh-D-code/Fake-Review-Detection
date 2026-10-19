"""Compare the TF-IDF and transformer fake-review classifiers on identical held-out rows.

Both models were trained on the stratified 70/15/15 split (seed 42, one row per
``review_hash``) built by ``scripts/train_review_classifier.py`` and
``scripts/train_transformer_classifier.py``. This script rebuilds that split, checks
the test rows are the ones the transformer was scored on, refits the TF-IDF model on
the training rows only (the shipped TF-IDF model also saw the validation rows, which
would not be a like-for-like comparison) and prints:

* test metrics for both models at 0.5 and at each model's own validation-chosen flag threshold
* paired bootstrap 95% confidence intervals for the difference (transformer - TF-IDF)
* an exact McNemar test on the 0.5 decisions
* the leave-one-category-out numbers stored in both metrics files

Output is also written to ``models/model_comparison.json``.

Usage:
    python scripts/compare_review_models.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_review_classifier as tfidf  # noqa: E402
from train_transformer_classifier import fingerprint, make_split  # noqa: E402

COLUMNS = ("accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc", "pr_auc", "brier")


def paired_bootstrap(y, prob_a, prob_b, rounds: int, seed: int) -> dict:
    """95% interval of (b - a) for accuracy, F1 and ROC-AUC at 0.5, resampling the same test rows for both models."""
    rng = np.random.default_rng(seed)
    pred_a, pred_b = (prob_a >= 0.5).astype(int), (prob_b >= 0.5).astype(int)
    diffs = {"accuracy": [], "f1_fake": [], "roc_auc": []}
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        diffs["accuracy"].append(accuracy_score(y[idx], pred_b[idx]) - accuracy_score(y[idx], pred_a[idx]))
        diffs["f1_fake"].append(f1_score(y[idx], pred_b[idx]) - f1_score(y[idx], pred_a[idx]))
        diffs["roc_auc"].append(roc_auc_score(y[idx], prob_b[idx]) - roc_auc_score(y[idx], prob_a[idx]))
    return {name: {"mean_diff": round(float(np.mean(v)), 4),
                   "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
            for name, v in diffs.items()}


def mcnemar(y, prob_a, prob_b) -> dict:
    ok_a, ok_b = (prob_a >= 0.5).astype(int) == y, (prob_b >= 0.5).astype(int) == y
    only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_tfidf_correct": only_a, "only_transformer_correct": only_b, "p_value": round(float(p), 6)}


def table(rows: list[tuple[str, dict]], keys=COLUMNS) -> str:
    head = "| model | " + " | ".join(keys) + " |"
    line = "|---|" + "---|" * len(keys)
    body = ["| " + name + " | " + " | ".join(str(m.get(k, "")) for k in keys) + " |" for name, m in rows]
    return "\n".join([head, line, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=ROOT / "trustengine.db")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    tfidf_report = json.loads((args.models_dir / "review_detector.metrics.json").read_text(encoding="utf-8"))
    trans_report = json.loads((args.models_dir / "review_transformer.metrics.json").read_text(encoding="utf-8"))
    saved = np.load(args.models_dir / "review_transformer" / "test_predictions.npz")

    dataset, _, texts, labels, categories = tfidf.load_dataset(args.db, args.dataset or tfidf_report["dataset"])
    train_idx, val_idx, test_idx = make_split(labels, args.seed)
    if fingerprint(texts, test_idx) != trans_report["split_fingerprint"] or not np.array_equal(test_idx, saved["test_idx"]):
        raise SystemExit("The test rows rebuilt from the database differ from the ones the transformer was scored on "
                         "(database or seed changed); retrain before comparing.")
    y_test = labels[test_idx]
    temperature = float(trans_report["temperature"])
    trans_prob = expit(saved["z"] / temperature)

    c_value = float(re.search(r"C=([0-9.]+)", tfidf_report["model"]).group(1))
    print(f"Refitting TF-IDF (C={c_value}) on the training rows only...", flush=True)
    pipeline = tfidf.build_pipeline(c_value, 150_000).fit([texts[i] for i in train_idx], labels[train_idx])
    tfidf_prob = pipeline.predict_proba([texts[i] for i in test_idx])[:, 1]
    refit = tfidf.full_metrics(y_test, tfidf_prob)
    recorded = tfidf_report["metrics"]["test_at_0.5"]
    if abs(refit["accuracy"] - recorded["accuracy"]) > 0.002 or abs(refit["roc_auc"] - recorded["roc_auc"]) > 0.002:
        print(f"WARNING: refit TF-IDF test metrics {refit} differ from the recorded {recorded}")

    flag = {"TF-IDF": float(tfidf_report["thresholds"]["suspicious_at"]),
            "transformer": float(trans_report["thresholds"]["suspicious_at"])}
    at_half = [("TF-IDF + LogReg", refit), ("DistilRoBERTa", tfidf.full_metrics(y_test, trans_prob))]
    at_flag = [(f"TF-IDF @ {flag['TF-IDF']}", tfidf.metrics_at(y_test, tfidf_prob, flag["TF-IDF"])),
               (f"DistilRoBERTa @ {flag['transformer']}", tfidf.metrics_at(y_test, trans_prob, flag["transformer"]))]
    boot = paired_bootstrap(y_test, tfidf_prob, trans_prob, args.bootstrap, args.seed)
    test = mcnemar(y_test, tfidf_prob, trans_prob)

    loco = {}
    for category, theirs in (trans_report["metrics"].get("leave_one_category_out") or {}).items():
        ours = (tfidf_report["metrics"].get("leave_one_category_out") or {}).get(category)
        if ours:
            loco[category] = {"tfidf": ours, "transformer": theirs}

    print(f"\nHeld-out test: {len(y_test)} rows (identical for both models), dataset '{dataset}'\n")
    print("Test at probability 0.5\n" + table(at_half))
    print("\nTest at each model's own flag threshold (chosen on validation)\n"
          + table(at_flag, ("threshold", "accuracy", "precision_fake", "recall_fake", "f1_fake")))
    print(f"\nPaired bootstrap, transformer minus TF-IDF ({args.bootstrap} resamples): {json.dumps(boot)}")
    print(f"McNemar on 0.5 decisions: {json.dumps(test)}")
    if loco:
        print("\nLeave-one-category-out (model retrained without the category, scored on all of its rows)")
        rows = []
        for category, pair in loco.items():
            rows += [(f"{category} - TF-IDF", pair["tfidf"]), (f"{category} - DistilRoBERTa", pair["transformer"])]
        print(table(rows, ("rows", "accuracy", "f1_fake", "roc_auc", "brier")))
        print(f"transformer protocol: {trans_report.get('leave_one_category_out_protocol')}")

    out = {"dataset": dataset, "test_rows": int(len(y_test)), "test_at_0.5": dict(at_half),
           "test_at_flag_threshold": dict(at_flag), "paired_bootstrap_transformer_minus_tfidf": boot,
           "mcnemar": test, "leave_one_category_out": loco,
           "note": "TF-IDF refit on training rows only for a like-for-like comparison; shipped TF-IDF also saw validation rows."}
    (args.models_dir / "model_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved models/model_comparison.json")


if __name__ == "__main__":
    main()
