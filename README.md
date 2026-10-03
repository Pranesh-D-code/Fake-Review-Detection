# Unified TrustEngine Platform

A live, evidence-first FastAPI application for analysing product-page metadata, tracking observed prices, and routing cross-platform research without fabricated results.

## Run

Prerequisites: Python 3.11+ and Node.js 20+.

```powershell
git clone https://github.com/YOUR-ACCOUNT/TrustEnginePlatform.git
cd TrustEnginePlatform

# Install the Node.js scraper and its bundled browser.
npm install

# Create an isolated Python environment and install the API.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional but required for Apify-provided Amazon review text.
Copy-Item .env.example .env
# Edit .env locally and add APIFY_API_TOKEN. Never commit this file.

uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`, paste a public product URL, and run an analysis. The interface renders only values exposed by the fetched page. It never substitutes demo products, prices, reviews, or direct-product matches.

## Optional Apify review provider

Amazon often withholds individual review text from ordinary public requests. The
app can therefore use an Apify Amazon-review Actor during the background
enrichment stage. The initial product price remains fast; review collection and
cross-store matching complete afterwards.

1. Create an Apify account and create an API token in **Settings → Integrations**.
2. Copy `.env.example` to `.env`.
3. Put the token in `APIFY_API_TOKEN` and leave `REVIEW_PROVIDER=apify`.
4. Restart the FastAPI server.

Never commit `.env` or paste its token into the browser or source code. The
default Actor is configurable through `APIFY_AMAZON_REVIEWS_ACTOR`. Test the
Actor with an `amazon.in` URL in the Apify Console before relying on it: Amazon
may expose only a small set of product-page review texts for a given listing.

Without `.env`, the app still runs and uses public-page collection only; review
text may be unavailable when a marketplace restricts access.

## Importing an external review dataset

To retain an authorised CSV dataset (or a ZIP containing one CSV) for later
human labelling and ML evaluation:

```powershell
python scripts/import_external_reviews.py path\to\dataset.zip
```

The importer preserves an external `label` column exactly as supplied. Confirm
the dataset documentation before mapping binary labels to genuine, suspicious,
or AI-generated classes.

For the supplied `archive.zip` Fake Reviews Dataset, the documented mapping is
`0 = original/human review` and `1 = computer-generated fake review`. Keep this
source label separate from the app's live “suspicious” signal: a model trained
on this dataset detects patterns represented in that dataset and must be
evaluated separately on live marketplace reviews.

## Design notes

- `app/engines.py` contains the deterministic, explainable baseline. It is production-safe to run locally and exposes clear replacement points for HingBERT/mBERT, Bi-LSTM attention, CLIP, GNN, SHAP, and LIME providers.
- Live collection is deliberately an adapter boundary (`MarketplaceProvider`). Only connect official APIs, licensed feeds, or sources you are authorised to access; do not evade platform protections.
- Direct links are emitted only when both title and image thresholds pass. In demo mode they are labelled as candidates.
- The score uses the supplied weights, with the duplication rate defined as duplicate pairs / total pairs (the inverse of the fraction shown in the specification, which would make more duplicates improve the score).

## API

- `POST /api/analyse` — accepts `{ "url": "..." }`
- `GET /api/report/{analysis_id}` — downloads a PDF report for a completed analysis
- `GET /docs` — interactive API documentation

## Production integration checklist

1. Add authorised per-marketplace catalogue, reviews, and historical-pricing connectors for complete coverage.
2. Register versioned transformer, vision, GNN, SHAP, and LIME providers with evaluation metrics.
3. Move the local SQLite observations to a production database and use a task queue for collection and reports.
4. Add consent, retention controls, rate limits, audit logging, and human review before using any result for consequential action.

## Container run

```powershell
docker build -t trustengine .
docker run -p 8000:8000 trustengine
```
