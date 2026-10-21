"""Pin the text-model backend to TF-IDF so the suite behaves the same whether or not a
transformer model has been trained. Tests that exercise the transformer select it
themselves (see tests/test_ml_detector_backends.py)."""
import os

os.environ["REVIEW_MODEL_BACKEND"] = "tfidf"
os.environ["LINK_CHECK"] = "false"          # tests never open store pages; tests/test_link_check.py switches it on with a fake transport


import pytest


@pytest.fixture(autouse=True)
def no_live_review_collection(monkeypatch):
    """Tests never open a store: review collection returns nothing unless a test stubs it itself."""
    from app import engines
    from app.review_collectors import CollectedReviews

    monkeypatch.setattr(engines, "_collect_store_reviews", lambda url, mode: CollectedReviews())
    engines._SNAPSHOTS.clear()
    from app import main
    main.ANALYSIS_CACHE.clear()
    from app import http_cache, link_check
    http_cache.clear()
    link_check.clear()
