"""Person-labelled reviews: saving, status, retrain trigger and the safety check. Uses a temp database only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import engines, ml_detector, review_labels
from app.main import app


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "DATA_PATH", tmp_path / "labels.db")
    monkeypatch.setattr(review_labels, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(review_labels, "STATUS_PATH", tmp_path / "models" / "retrain_status.json")
    return tmp_path / "labels.db"


def test_hash_matches_the_review_store():
    text = "  Great   phone, battery lasts ALL day  "
    assert review_labels.review_hash(text) == engines._review_hash(text)


def test_save_change_and_clear_a_label(tmp_db):
    text = "The charger stopped working after two weeks and support never answered my emails."
    saved = review_labels.save_label(text, "fake", rating=1, product_url="https://example.com/p/1", platform="Example")
    assert saved["label"] == "fake"
    assert review_labels.lookup([saved["review_hash"]]) == {saved["review_hash"]: "fake"}
    review_labels.save_label(text, "real")                       # changing your mind overwrites
    assert review_labels.lookup([saved["review_hash"]])[saved["review_hash"]] == "real"
    assert review_labels.label_counts() == {"real": 1, "fake": 0, "total": 1}
    review_labels.save_label(text, "clear")
    assert review_labels.lookup([saved["review_hash"]]) == {} and review_labels.label_counts()["total"] == 0


def test_the_review_text_is_kept_so_training_can_read_it(tmp_db):
    saved = review_labels.save_label("Not stored by the scraper but labelled by hand in the app.", "real")
    db = engines._db()
    try:
        row = db.execute("SELECT review_text, collection_source FROM review_observations WHERE review_hash = ?", (saved["review_hash"],)).fetchone()
    finally:
        db.close()
    assert row and row[0].startswith("Not stored") and row[1] == "user_label"


def test_bad_input_is_rejected(tmp_db):
    for text, label in (("", "real"), ("   ", "fake"), ("fine review text here", "maybe"), ("x" * 6000, "real")):
        with pytest.raises(ValueError):
            review_labels.save_label(text, label)


def _label_many(real: int, fake: int):
    for i in range(real):
        review_labels.save_label(f"honest review number {i} about the battery, screen and the delivery being on time", "real")
    for i in range(fake):
        review_labels.save_label(f"amazing product number {i} best product ever buy now great great great", "fake")


def test_no_retrain_until_enough_new_labels(tmp_db, monkeypatch):
    started = []
    monkeypatch.setattr(review_labels, "_spawn_trainer", lambda: started.append(1) or 0)
    monkeypatch.setattr(ml_detector, "metadata", lambda: {"live_labels_total": 0})
    _label_many(10, 10)                                           # 20 labels: trainable, but below the 50 trigger
    state = review_labels.maybe_start_retrain()
    assert state["can_train"] and state["new_labels_since_training"] == 20 and not started
    _label_many(40, 0)                                            # now 60 labels
    review_labels.maybe_start_retrain()
    review_labels._thread.join(timeout=5)
    assert started == [1]


def test_one_sided_labels_never_start_training(tmp_db, monkeypatch):
    monkeypatch.setattr(review_labels, "_spawn_trainer", lambda: pytest.fail("must not train on one class"))
    monkeypatch.setattr(ml_detector, "metadata", lambda: {})
    _label_many(60, 0)
    assert review_labels.maybe_start_retrain()["can_train"] is False


def test_a_running_retrain_blocks_a_second_one(tmp_db, monkeypatch):
    monkeypatch.setattr(ml_detector, "metadata", lambda: {})
    _label_many(30, 30)
    review_labels._write_status(state="running", started_at=review_labels._now())
    monkeypatch.setattr(review_labels, "_spawn_trainer", lambda: pytest.fail("already running"))
    assert review_labels.maybe_start_retrain()["running"] is True


def test_api_label_flow(tmp_db, monkeypatch):
    monkeypatch.setattr(review_labels, "maybe_start_retrain", lambda force=False: review_labels.status())
    client = TestClient(app)
    body = {"text": "Delivered late and the box was dented but the speaker works fine.", "label": "real", "rating": 3,
            "product_url": "https://www.amazon.in/dp/B0TEST", "platform": "Amazon"}
    response = client.post("/api/reviews/label", json=body)
    assert response.status_code == 200 and response.json()["label"] == "real"
    assert response.json()["status"]["labels"]["real"] == 1
    assert client.get("/api/model/status").json()["labels"]["total"] == 1
    assert client.post("/api/reviews/label", json={**body, "label": "nonsense"}).status_code == 400
    assert client.post("/api/model/retrain").status_code == 400            # too few labels to train


def test_findings_carry_hash_and_existing_label(tmp_db):
    from app.models import Product, Review
    from app.nlp_engine import compute_trust_score
    text = "Delivered late and the box was dented but the speaker works fine and sounds good."
    review_labels.save_label(text, "real")
    reviews = [Review(text=text, rating=4)]
    findings, _, _ = compute_trust_score(Product(platform="X", url="", title="t", reviews=reviews), reviews)
    assert findings[0].review_hash == review_labels.review_hash(text)
    assert review_labels.lookup([findings[0].review_hash]) == {findings[0].review_hash: "real"}


# ---- the safety check used by scripts/auto_retrain.py ----

def _load_auto_retrain():
    spec = importlib.util.spec_from_file_location("auto_retrain", Path(__file__).resolve().parent.parent / "scripts" / "auto_retrain.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_retrain"] = module
    spec.loader.exec_module(module)
    return module


def _metrics(f1):
    return {"metrics": {"test_dataset_only": {"f1_fake": f1}}}


def test_candidate_replaces_the_model_only_if_it_is_not_worse():
    auto = _load_auto_retrain()
    assert auto.decide(_metrics(0.951), _metrics(0.950), 0.015)[0] is True
    assert auto.decide(_metrics(0.940), _metrics(0.950), 0.015)[0] is True          # within tolerance
    ok, why = auto.decide(_metrics(0.900), _metrics(0.950), 0.015)
    assert ok is False and "Kept the current model" in why
    assert auto.decide({"metrics": {}}, _metrics(0.95), 0.015)[0] is False           # nothing to judge it by
    assert auto.decide(_metrics(0.9), None, 0.015)[0] is True                        # no model in use yet
    # an older model that only has the overall test number can still be compared
    assert auto.dataset_f1({"metrics": {"test_at_0.5": {"f1_fake": 0.95}}}) == 0.95


# ---- the training script really reads the labels (tiny synthetic data, seconds) ----

def test_training_script_uses_person_labels(tmp_db, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("trainer", Path(__file__).resolve().parent.parent / "scripts" / "train_review_classifier.py")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["trainer"] = trainer
    spec.loader.exec_module(trainer)

    db = engines._db()
    try:
        db.executemany("INSERT INTO dataset_label_mappings VALUES (?,?,?,?,?,?)", [
            ("tiny.zip", "0", "original", "orig", None, "now"), ("tiny.zip", "1", "computer-generated-fake", "fake", None, "now")])
        rows = []
        for i in range(150):
            rows.append((f"external://tiny/c{i % 3}", "External dataset", f"c{i % 3}", f"h-real-{i}",
                         f"Honest review {i}: the battery lasted two days, delivery was on time and the screen is sharp", 4, None, 0,
                         "external_dataset", "now", "now", "needs-review", 0.0, "[]", "tiny.zip", "0"))
            rows.append((f"external://tiny/c{i % 3}", "External dataset", f"c{i % 3}", f"h-fake-{i}",
                         f"Great product {i} great quality great price great seller I love it so much buy buy buy", 5, None, 0,
                         "external_dataset", "now", "now", "needs-review", 0.0, "[]", "tiny.zip", "1"))
        db.executemany("INSERT INTO review_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        db.commit()
    finally:
        db.close()
    _label_many(12, 12)

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--db", str(tmp_db), "--out-dir", str(out), "--jobs", "1",
                                      "--holdout-categories", "0", "--max-features", "2000"])
    trainer.main()
    report = json.loads((out / "review_detector.metrics.json").read_text(encoding="utf-8"))
    assert report["live_labels_total"] == 24 and report["live_labels_in_training"] == 24
    assert report["rows"]["live"] == 24 and "labelled real or fake in this app" in report["caveat"]
    assert "test_dataset_only" in report["metrics"] and "live_test" in report["metrics"]

    # with labels switched off the same data trains on the dataset alone
    monkeypatch.setattr(sys, "argv", ["train", "--db", str(tmp_db), "--out-dir", str(tmp_path / "out2"), "--jobs", "1",
                                      "--holdout-categories", "0", "--max-features", "2000", "--no-live-labels"])
    trainer.main()
    plain = json.loads((tmp_path / "out2" / "review_detector.metrics.json").read_text(encoding="utf-8"))
    assert plain["live_labels_in_training"] == 0
