"""The training script learns from downloaded datasets and reports how well that carries over (tiny synthetic data)."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

from app import engines


def _load_trainer():
    spec = importlib.util.spec_from_file_location("trainer_extra", Path(__file__).resolve().parent.parent / "scripts" / "train_review_classifier.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trainer_extra"] = module
    spec.loader.exec_module(module)
    return module


def _seed_kaggle_like(db_path):
    db = engines._db()
    try:
        db.executemany("INSERT INTO dataset_label_mappings VALUES (?,?,?,?,?,?)", [
            ("tiny.zip", "0", "original", "orig", None, "now"), ("tiny.zip", "1", "computer-generated-fake", "fake", None, "now")])
        rows = []
        for i in range(120):
            rows.append((f"external://tiny/c{i % 2}", "External dataset", f"c{i % 2}", f"k-real-{i}",
                         f"Honest review {i}: the battery lasted two days, delivery was on time and the screen is sharp", 4, None, 0,
                         "external_dataset", "now", "now", "needs-review", 0.0, "[]", "tiny.zip", "0"))
            rows.append((f"external://tiny/c{i % 2}", "External dataset", f"c{i % 2}", f"k-fake-{i}",
                         f"Great product {i} great quality great price great seller I love it so much buy buy buy", 5, None, 0,
                         "external_dataset", "now", "now", "needs-review", 0.0, "[]", "tiny.zip", "1"))
        db.executemany("INSERT INTO review_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        db.commit()
    finally:
        db.close()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text_", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_downloaded_datasets_are_trained_on_deduplicated_and_evaluated_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "DATA_PATH", tmp_path / "t.db")
    _seed_kaggle_like(tmp_path / "t.db")
    base = tmp_path / "external"
    modern = [{"text_": f"Sleek design, snappy performance and a battery that outlasts my commute {i}; truly delighted with it", "label": "CG"} for i in range(80)]
    already_known = [{"text_": "Honest review 3: the battery lasted two days, delivery was on time and the screen is sharp", "label": "OR"}]
    new_real = [{"text_": f"Meh, arrived late and the box was crushed but it works {i}", "label": "OR"} for i in range(10)]
    _write_csv(base / "modern" / "train.csv", modern + already_known + new_real)
    gp = [{"text_": f"Reception was quick and the doctor listened carefully {i}", "label": "0"} for i in range(20)] + \
         [{"text_": f"I would like to express my profound gratitude to the entire team {i} for their exceptional care", "label": "1"} for i in range(20)]
    _write_csv(base / "gp" / "all.csv", gp)
    manifest = {"datasets": [
        {"name": "modern-x", "license": "CC BY 4.0", "role": "train", "source_url": "https://example.test/x", "files": [{"file": "modern/train.csv"}],
         "text_col": "text_", "label_col": "label", "label_map": {"OR": 0, "CG": 1}, "category_col": None},
        {"name": "gp-x", "license": "MIT", "role": "eval", "files": [{"file": "gp/all.csv"}],
         "text_col": "text_", "label_col": "label", "label_map": {"0": 0, "1": 1}, "category_col": None}]}
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    trainer = _load_trainer()
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--db", str(tmp_path / "t.db"), "--out-dir", str(out), "--jobs", "1", "--holdout-categories", "0",
                                      "--max-features", "2000", "--no-live-labels", "--extra-manifest", str(base / "manifest.json")])
    trainer.main()
    report = json.loads((out / "review_detector.metrics.json").read_text(encoding="utf-8"))
    extra = {e["name"]: e for e in report["extra_datasets"]}
    assert extra["modern-x"]["rows_used"] == 90 and extra["modern-x"]["fake"] == 80 and extra["modern-x"]["real"] == 10     # the known review was not added twice
    assert "modern-x" in report["metrics"]["per_source_test"] and "dataset" in report["metrics"]["per_source_test"]
    assert set(report["metrics"]["evaluation_only_sets"]) == {"gp-x"} and report["metrics"]["evaluation_only_sets"]["gp-x"]["rows"] == 40
    probes = report["metrics"]["generalisation_probes"]
    assert any(name.startswith("kaggle_only") for name in probes) and "never_saw_modern-x" in probes
    assert "modern-x" in report["caveat"]


def test_flag_threshold_keeps_wrongly_flagged_real_reviews_under_the_cap():
    import numpy as np
    trainer = _load_trainer()
    rng = np.random.default_rng(0)
    y = np.array([0] * 1000 + [1] * 1000)
    prob = np.concatenate([rng.beta(1, 8, 1000), rng.beta(8, 1, 1000)])          # real mostly low, fake mostly high, some overlap
    cut = trainer.threshold_for_fpr(y, prob, 0.02)
    assert (prob[y == 0] >= cut).mean() <= 0.021
    assert cut > trainer.threshold_for_fpr(y, prob, 0.10)                       # a stricter cap means a higher cut-off
    assert trainer.balance_weights(np.ones(6), np.array([1, 1, 1, 1, 0, 0])).sum() == 4.0   # fake weight scaled down to match real
