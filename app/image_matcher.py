"""Multi-Modal Image Similarity Engine for Unified TrustEngine Platform.

Performs perceptual hash & visual feature comparison between source product image
and cross-platform candidate images.
"""
from __future__ import annotations

import io
import httpx
from PIL import Image

def compute_image_similarity(source_url: str | None, target_url: str | None) -> float | None:
    if not source_url or not target_url:
        return None
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        with httpx.Client(timeout=4.0, follow_redirects=True, headers=headers) as client:
            r1 = client.get(source_url)
            r2 = client.get(target_url)
            if r1.status_code != 200 or r2.status_code != 200:
                return None
            
            img1 = Image.open(io.BytesIO(r1.content)).convert("RGB").resize((64, 64))
            img2 = Image.open(io.BytesIO(r2.content)).convert("RGB").resize((64, 64))

            try:
                import imagehash
                h1 = imagehash.phash(img1)
                h2 = imagehash.phash(img2)
                diff = h1 - h2
                sim = max(0.0, 1.0 - (diff / 64.0))
                return round(sim, 3)
            except Exception:
                # Fast mean pixel difference fallback
                pixels1 = list(img1.getdata())
                pixels2 = list(img2.getdata())
                total_diff = sum(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]) for p1, p2 in zip(pixels1, pixels2))
                max_diff = 64 * 64 * 255 * 3
                sim = max(0.0, 1.0 - (total_diff / max_diff))
                return round(sim, 3)
    except Exception:
        return None
