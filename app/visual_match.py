"""Visual product matching: are two store photos pictures of the same product?

Method (in order of preference):

1. **Image embedding** - both photos are trimmed to the product (plain backgrounds removed), padded to a square and
   passed through DINOv2-small (a vision transformer trained without labels, Apache-2.0, ~88 MB, fetched by
   ``scripts/fetch_visual_model.py``). Photos of the same item land close together in that space; the score is the
   cosine similarity of the two embeddings, mapped to a probability by ``models/visual/calibration.json``.
2. **Perceptual hash fallback** - if the model or PyTorch is not available, the trimmed photos are compared with a
   perceptual hash. Weaker (the evaluation file shows by how much) but dependency-free.

How good it is on real store photos is measured by ``scripts/evaluate_visual_matching.py`` and stored next to the
calibration, together with the score thresholds this module uses for "same product" and "different product".
"""
from __future__ import annotations

import io
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "visual" / "dinov2-small"
CALIBRATION_PATH = BASE_DIR / "models" / "visual" / "calibration.json"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/122.0.0.0 Safari/537.36")
MAX_IMAGE_BYTES = 8_000_000
MAX_SIDE = 512                    # the model looks at 224 px; larger photos only slow trimming and hashing down

_lock = threading.Lock()
_model = None
_model_failed = False
_infer = threading.Semaphore(2)
_images: "OrderedDict[str, Image.Image | None]" = OrderedDict()
_embeddings: "OrderedDict[str, np.ndarray]" = OrderedDict()
_calibration: dict | None = None


@dataclass
class VisualScore:
    score: float                 # 0..1, how likely the two photos show the same product
    cosine: float | None         # raw embedding similarity (None for the hash fallback)
    colour: float                # colour-histogram agreement of the products themselves
    method: str                  # "dinov2" | "phash"
    verdict: str                 # "same" | "different" | "unclear"


# ------------------------------------------------------------------ images

def load_image(url: str | None) -> Image.Image | None:
    if not url or not url.startswith(("http://", "https://")):
        return None
    with _lock:
        if url in _images:
            _images.move_to_end(url)
            return _images[url]
    image = None
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True, headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.5"}) as client:
            response = client.get(url)
        if response.status_code == 200 and 100 < len(response.content) <= MAX_IMAGE_BYTES:
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
    except Exception:
        image = None
    with _lock:
        _images[url] = image
        while len(_images) > 160:
            _images.popitem(last=False)
    return image


def trim_to_product(image: Image.Image, size: int = 224) -> Image.Image:
    """Cut away the plain background around the product and centre it on a square canvas."""
    array = np.asarray(image, dtype=np.int16)
    border = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])
    background = np.median(border, axis=0)
    differs = np.abs(array - background).max(axis=2) > 24
    box = None
    if differs.any():
        ys, xs = np.where(differs)
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    height, width = array.shape[:2]
    if box is None or (box[2] - box[0]) * (box[3] - box[1]) < 0.02 * width * height:
        cropped = image                                   # no clear object on a plain background: keep the whole photo
    else:
        margin_x, margin_y = int(0.05 * (box[2] - box[0])) + 2, int(0.05 * (box[3] - box[1])) + 2
        cropped = image.crop((max(0, box[0] - margin_x), max(0, box[1] - margin_y), min(width, box[2] + margin_x), min(height, box[3] + margin_y)))
    side = max(cropped.size)
    canvas = Image.new("RGB", (side, side), tuple(int(v) for v in background))
    canvas.paste(cropped, ((side - cropped.size[0]) // 2, (side - cropped.size[1]) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def colour_signature(image: Image.Image) -> np.ndarray:
    """Hue / saturation / value histogram of the pixels that are not plain background."""
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    array = np.asarray(image, dtype=np.int16)
    border = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])
    foreground = np.abs(array - np.median(border, axis=0)).max(axis=2) > 24
    pixels = hsv[foreground] if foreground.sum() > 200 else hsv.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=(12, 3, 3), range=((0, 256), (0, 256), (0, 256)))
    total = hist.sum()
    return (hist / total).ravel() if total else hist.ravel()


def colour_similarity(a: Image.Image, b: Image.Image) -> float:
    return float(np.minimum(colour_signature(a), colour_signature(b)).sum())


# ------------------------------------------------------------------ embeddings

def available() -> bool:
    return _load_model() is not None


def _load_model():
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    with _lock:
        if _model is not None or _model_failed:
            return _model
        try:
            if not (MODEL_DIR / "model.safetensors").is_file():
                raise FileNotFoundError("run scripts/fetch_visual_model.py")
            import torch
            from transformers import AutoModel
            torch.set_num_threads(2)
            model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
            model.eval()
            _model = model
        except Exception:
            _model_failed = True
            _model = None
    return _model


def embed_parts(image: Image.Image) -> tuple[np.ndarray, np.ndarray] | None:
    """(class-token vector, mean-patch vector), each L2-normalised."""
    model = _load_model()
    if model is None:
        return None
    import torch
    square = trim_to_product(image)
    array = (np.asarray(square, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    with _infer, torch.no_grad():
        hidden = model(pixel_values=tensor).last_hidden_state[0]
    unit = lambda v: v / (np.linalg.norm(v) or 1.0)
    return unit(hidden[0].numpy()), unit(hidden[1:].mean(dim=0).numpy())


def embed(image: Image.Image) -> np.ndarray | None:
    parts = embed_parts(image)
    if parts is None:
        return None
    vector = np.concatenate(parts)                         # class token + average of the patch tokens
    return vector / (np.linalg.norm(vector) or 1.0)


def _embedding_for(url: str, image: Image.Image) -> np.ndarray | None:
    with _lock:
        if url in _embeddings:
            _embeddings.move_to_end(url)
            return _embeddings[url]
    vector = embed(image)
    if vector is not None:
        with _lock:
            _embeddings[url] = vector
            while len(_embeddings) > 400:
                _embeddings.popitem(last=False)
    return vector


# ------------------------------------------------------------------ scoring

def calibration() -> dict:
    """Thresholds and mapping measured by scripts/evaluate_visual_matching.py (with safe defaults)."""
    global _calibration
    if _calibration is None:
        try:
            _calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _calibration = {}
    return _calibration


def reset_calibration() -> None:
    global _calibration
    _calibration = None


def _phash_score(a: Image.Image, b: Image.Image) -> float:
    import imagehash
    ha, hb = imagehash.phash(trim_to_product(a, 64), hash_size=16), imagehash.phash(trim_to_product(b, 64), hash_size=16)
    return max(0.0, 1.0 - (ha - hb) / 256.0)


def probability_from_features(cosine: float, colour: float, phash: float) -> float:
    cal = calibration().get("dinov2") or {}
    combiner = cal.get("combiner")
    if combiner:
        z = combiner["intercept"] + combiner["cosine"] * cosine + combiner["colour"] * colour + combiner["phash"] * phash
    else:
        z = cal.get("slope", 12.0) * cosine + cal.get("intercept", -8.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def verdict_for(score: float, method: str) -> str:
    cal = (calibration().get(method) or {})
    same_at, different_below = cal.get("same_at", 0.85 if method == "dinov2" else 0.9), cal.get("different_below", 0.25 if method == "dinov2" else 0.6)
    return "same" if score >= same_at else "different" if score <= different_below else "unclear"


def compare_images(a: Image.Image, b: Image.Image, key_a: str = "", key_b: str = "") -> VisualScore:
    ta, tb = trim_to_product(a), trim_to_product(b)
    colour = colour_similarity(ta, tb)
    va = _embedding_for(key_a or str(id(a)), a)
    vb = _embedding_for(key_b or str(id(b)), b)
    if va is not None and vb is not None:
        cosine = float(np.dot(va, vb))
        score = probability_from_features(cosine, colour, _phash_score(a, b))
        return VisualScore(round(score, 4), round(cosine, 4), round(colour, 4), "dinov2", verdict_for(score, "dinov2"))
    score = _phash_score(a, b)
    return VisualScore(round(score, 4), None, round(colour, 4), "phash", verdict_for(score, "phash"))


def compare(url_a: str | None, url_b: str | None) -> VisualScore | None:
    """Score two photos by URL; None when either cannot be downloaded."""
    a, b = load_image(url_a), load_image(url_b)
    if a is None or b is None:
        return None
    return compare_images(a, b, url_a or "", url_b or "")


def warm_up() -> None:
    _load_model()
