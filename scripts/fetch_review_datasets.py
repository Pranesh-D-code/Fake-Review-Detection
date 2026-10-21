"""Download the open review datasets used to train and check the fake-review model.

Everything comes from Hugging Face dataset repositories with an open licence, is stored as plain data files
(Parquet / CSV - never executed) under ``data/external/<name>/`` and is described in
``data/external/manifest.json``, which ``scripts/train_review_classifier.py`` reads.

    python scripts/fetch_review_datasets.py            # download what is missing
    python scripts/fetch_review_datasets.py --list     # show what would be fetched

Datasets (role "train" is learned from, "eval" is only used to measure how well the model carries over):

* Flowerly/modern-fake-reviews (+ -glm, -lfm)  CC BY 4.0   fake product reviews written by present-day LLMs
  (DeepSeek, GLM, LFM) next to real ones. The Kaggle half the project already has is GPT-2-era, and a detector
  trained on it alone flags only ~4% of modern LLM reviews (per the dataset card).
* janduplessis886/gp_surgery_reviews_fake_and_real  MIT   real vs DeepSeek-written UK GP-surgery feedback - a
  different domain and style, so it is kept for evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "external"

DATASETS = [
    {"name": "modern-fake-reviews-deepseek", "repo": "Flowerly/modern-fake-reviews", "license": "CC BY 4.0", "role": "train",
     "files": ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
     "text_col": "text_", "label_col": "label", "label_map": {"OR": 0, "CG": 1}, "category_col": "category",
     "about": "Real Amazon-style reviews vs fakes written by DeepSeek"},
    {"name": "modern-fake-reviews-glm", "repo": "Flowerly/modern-fake-reviews-glm", "license": "CC BY 4.0", "role": "train",
     "files": ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
     "text_col": "text_", "label_col": "label", "label_map": {"OR": 0, "CG": 1}, "category_col": "category",
     "about": "Real Amazon-style reviews vs fakes written by GLM"},
    {"name": "modern-fake-reviews-lfm", "repo": "Flowerly/modern-fake-reviews-lfm", "license": "CC BY 4.0", "role": "train",
     "files": ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
     "text_col": "text_", "label_col": "label", "label_map": {"OR": 0, "CG": 1}, "category_col": "category",
     "about": "Real Amazon-style reviews vs fakes written by a small LFM model"},
    {"name": "gp-surgery-fake-and-real", "repo": "janduplessis886/gp_surgery_reviews_fake_and_real", "license": "MIT", "role": "eval",
     "files": ["gp_surgery_review_fake_and_real.csv"],
     "text_col": "free_text", "label_col": "label", "label_map": {"0": 0, "1": 1}, "category_col": None,
     "about": "Real UK GP-surgery feedback vs DeepSeek-written fakes (out-of-domain check)"},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_sizes(repo: str) -> dict[str, int]:
    tree = httpx.get(f"https://huggingface.co/api/datasets/{repo}/tree/main", params={"recursive": "true"}, timeout=30).json()
    return {item["path"]: int(item.get("size") or 0) for item in tree if item.get("type") == "file"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the plan and exit")
    args = parser.parse_args()

    manifest = []
    total = 0
    for spec in DATASETS:
        sizes = remote_sizes(spec["repo"])
        wanted = sum(sizes.get(f, 0) for f in spec["files"])
        total += wanted
        print(f"{spec['name']:34} {spec['repo']:55} {spec['license']:10} {wanted / 1e6:6.1f} MB  ({spec['role']})")
        if args.list:
            continue
        folder = TARGET / spec["name"]
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for rel in spec["files"]:
            destination = folder / Path(rel).name
            if not destination.exists() or destination.stat().st_size != sizes.get(rel, -1):
                with httpx.stream("GET", f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{rel}", timeout=120, follow_redirects=True) as response:
                    response.raise_for_status()
                    with destination.open("wb") as out:
                        for chunk in response.iter_bytes(1 << 20):
                            out.write(chunk)
                if sizes.get(rel) and destination.stat().st_size != sizes[rel]:
                    print(f"  size mismatch for {rel}: expected {sizes[rel]}, got {destination.stat().st_size}", file=sys.stderr)
                    return 1
            entries.append({"file": f"{spec['name']}/{destination.name}", "bytes": destination.stat().st_size, "sha256": sha256(destination)})
        manifest.append({**{k: v for k, v in spec.items() if k != "files"}, "files": entries,
                         "source_url": f"https://huggingface.co/datasets/{spec['repo']}"})
    print(f"total {total / 1e6:.1f} MB")
    if not args.list:
        TARGET.mkdir(parents=True, exist_ok=True)
        (TARGET / "manifest.json").write_text(json.dumps({"datasets": manifest}, indent=2), encoding="utf-8")
        print(f"wrote {TARGET / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
