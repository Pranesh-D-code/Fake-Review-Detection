"""Import an external CSV/ZIP review dataset into TrustEngine's SQLite store.

The importer intentionally preserves an external label as supplied. It does
not guess whether values such as 0/1 mean real, fake, AI-generated, or another
class. Add that mapping only after checking the dataset documentation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "trustengine.db"


def review_hash(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def ensure_schema(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS review_observations (
            product_url TEXT NOT NULL,
            platform TEXT NOT NULL,
            sku TEXT,
            review_hash TEXT NOT NULL,
            review_text TEXT NOT NULL,
            rating INTEGER,
            review_timestamp TEXT,
            verified_purchase INTEGER NOT NULL DEFAULT 0,
            collection_source TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            baseline_label TEXT NOT NULL,
            baseline_confidence REAL NOT NULL,
            baseline_signals TEXT NOT NULL,
            external_dataset TEXT,
            external_label TEXT,
            PRIMARY KEY (product_url, review_hash)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS review_labels (
            review_hash TEXT PRIMARY KEY,
            human_label TEXT NOT NULL,
            annotator TEXT,
            note TEXT,
            labeled_at TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS dataset_label_mappings (
            dataset_name TEXT NOT NULL,
            raw_label TEXT NOT NULL,
            normalized_label TEXT NOT NULL,
            label_description TEXT NOT NULL,
            documentation_url TEXT,
            recorded_at TEXT NOT NULL,
            PRIMARY KEY (dataset_name, raw_label)
        )
    """)
    existing = {row[1] for row in db.execute("PRAGMA table_info(review_observations)")}
    for column in ("external_dataset", "external_label"):
        if column not in existing:
            db.execute(f"ALTER TABLE review_observations ADD COLUMN {column} TEXT")


def parse_rating(value: str | None) -> int | None:
    try:
        rating = round(float(value or ""))
        return rating if 1 <= rating <= 5 else None
    except ValueError:
        return None


def rows_from_source(source: Path):
    if source.suffix.casefold() == ".zip":
        with zipfile.ZipFile(source) as archive:
            csv_names = [name for name in archive.namelist() if name.casefold().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError("The ZIP must contain exactly one CSV review dataset.")
            with archive.open(csv_names[0]) as raw:
                yield from csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
    else:
        with source.open("r", encoding="utf-8-sig", newline="") as raw:
            yield from csv.DictReader(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import external review records into TrustEngine SQLite.")
    parser.add_argument("source", type=Path, help="Path to a CSV file or a ZIP containing one CSV file")
    parser.add_argument("--dataset-name", default=None, help="Optional source name stored with each review")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"Dataset not found: {source}")
    dataset_name = args.dataset_name or source.name
    now = datetime.now(timezone.utc).isoformat()
    imported = skipped = 0

    db = sqlite3.connect(DATABASE)
    try:
        ensure_schema(db)
        for row in rows_from_source(source):
            text = (row.get("text") or row.get("review_text") or "").strip()
            if not text:
                skipped += 1
                continue
            category = (row.get("category") or "unclassified").strip()
            product_url = f"external://{dataset_name}/{category}"
            db.execute("""
                INSERT INTO review_observations (
                    product_url, platform, sku, review_hash, review_text, rating,
                    review_timestamp, verified_purchase, collection_source,
                    first_seen_at, last_seen_at, baseline_label,
                    baseline_confidence, baseline_signals, external_dataset,
                    external_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_url, review_hash) DO UPDATE SET
                    last_seen_at=excluded.last_seen_at,
                    rating=excluded.rating,
                    external_dataset=excluded.external_dataset,
                    external_label=excluded.external_label
            """, (
                product_url, "External dataset", category, review_hash(text), text,
                parse_rating(row.get("rating")), None, 0, "external_dataset",
                now, now, "needs-review", 0.0,
                json.dumps(["Imported from external dataset; external label semantics not yet verified."]),
                dataset_name, row.get("label"),
            ))
            imported += 1
        db.commit()
    finally:
        db.close()
    print(f"Imported {imported} review rows; skipped {skipped} empty rows into {DATABASE.name}.")


if __name__ == "__main__":
    main()
