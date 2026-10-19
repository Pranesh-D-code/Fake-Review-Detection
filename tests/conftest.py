"""Pin the text-model backend to TF-IDF so the suite behaves the same whether or not a
transformer model has been trained. Tests that exercise the transformer select it
themselves (see tests/test_ml_detector_backends.py)."""
import os

os.environ["REVIEW_MODEL_BACKEND"] = "tfidf"
