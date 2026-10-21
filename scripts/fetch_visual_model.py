"""Download the small image model used for visual product matching (about 88 MB, Apache-2.0).

    python scripts/fetch_visual_model.py

Model: facebook/dinov2-small (Meta AI), a self-supervised vision transformer whose embeddings place two photos of the
same product close together. Only the weights file ``model.safetensors`` (plain tensors, no code) and two JSON config
files are fetched into ``models/visual/dinov2-small/``. Without it the app falls back to a perceptual-hash comparison.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "visual" / "dinov2-small"
REPO = "facebook/dinov2-small"
FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


def main() -> int:
    sizes = {item["path"]: int(item.get("size") or 0) for item in
             httpx.get(f"https://huggingface.co/api/models/{REPO}/tree/main", timeout=30).json() if item.get("type") == "file"}
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        destination = TARGET / name
        if destination.exists() and destination.stat().st_size == sizes.get(name, -1):
            print(f"already have {name}")
            continue
        print(f"downloading {name} ({sizes.get(name, 0) / 1e6:.1f} MB) from huggingface.co/{REPO}")
        with httpx.stream("GET", f"https://huggingface.co/{REPO}/resolve/main/{name}", timeout=300, follow_redirects=True) as response:
            response.raise_for_status()
            with destination.open("wb") as out:
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)
        if sizes.get(name) and destination.stat().st_size != sizes[name]:
            print(f"size mismatch for {name}", file=sys.stderr)
            return 1
    digest = hashlib.sha256((TARGET / "model.safetensors").read_bytes()).hexdigest()
    (TARGET / "SOURCE.txt").write_text(f"https://huggingface.co/{REPO} (Apache-2.0)\nmodel.safetensors sha256 {digest}\n", encoding="utf-8")
    print("done:", TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
