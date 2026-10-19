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

## Fake-review detection

Every collected review is judged by two independent layers, and the result
keeps both visible:

1. **Rule signals** (`app/nlp_engine.py`): near-duplicate wording, promotional
   phrasing, rating/sentiment mismatch, low-information praise.
2. **Trained text model** (`app/ml_detector.py`): a TF-IDF (word 1-2 grams +
   character 3-5 grams) logistic-regression classifier trained on the labelled
   review dataset in `trustengine.db`. It returns how strongly a review's
   wording resembles the *computer-generated* reviews in that dataset.

A review is **suspicious** if a rule fires or the model is above its
precision-calibrated threshold; **genuine-looking** only if the model is
confidently low and no rule fired; everything else is **needs-review**. Rules
are never overruled by the model. Reviews under ~11 words or in non-Latin
script are not judged by the model (it has no basis for them) and fall back to
the rules.

**What the numbers mean.** Held-out test (6,073 reviews never used for fitting
or threshold choice): accuracy 95.0%, F1 0.951, ROC-AUC 0.991. Flagging at the
default threshold gives 98.2% precision and 89.1% recall. Training with a whole
product category held out gives 91-93% accuracy. The earlier rule-only baseline
flagged none of this dataset's generated reviews. These figures describe
separating *this dataset's* original reviews from its generated ones; the model
has **not** been validated on live marketplace reviews, and a high score is a
wording signal, not proof of fakery. Check it against human-labelled live
reviews (`review_labels` table) before relying on it.

Retrain (about 7 minutes on a laptop) after importing or labelling more data:

```powershell
python scripts/train_review_classifier.py        # writes models/review_detector.joblib + .metrics.json
python -m pytest tests                            # after pip install -r requirements-dev.txt
```

`GET /api/analyse/{id}` and `POST /api/analyse` now include `review_analysis`
(counts, model provenance) and `review_burst_summary` (daily-count z-score
spikes, needs 15+ dated reviews). Each finding carries `ml_fake_probability`.

### Teaching the model with your own labels

Under every review in the list there are **Real** and **Fake** buttons. Your choice is saved in
`review_labels` and used as training data next to the dataset. The detector's own verdict is never
used as a label: a model trained on its own guesses learns nothing new.

- After `AUTO_RETRAIN_MIN_NEW_LABELS` new labels (default 50, and at least 20 labels with 5+ real and
  5+ fake) a retrain starts by itself in the background (`scripts/auto_retrain.py`).
- It trains a candidate model, and only replaces the live one if the candidate's F1 on held-out
  *dataset* rows has not dropped by more than 0.015. The previous model is kept as
  `models/review_detector.prev.joblib`; the outcome is in `models/retrain_status.json`.
- Your labelled reviews count 5x a dataset row (`--live-weight`). With very few labels the numbers are
  noisy, so the live-accuracy figure appears in the metrics file but is not used to accept a model.
- `GET /api/model/status` shows the counts; `POST /api/model/retrain` starts a retrain now.
  Set `AUTO_RETRAIN=false` to retrain only by hand (`python scripts/auto_retrain.py`).

### Analysing a large review set

Amazon shows full review listings only to signed-in users, and this project does
not log in or bypass that. For hundreds of reviews use a licensed route:

- Raise `APIFY_REVIEW_LIMIT` (up to 1000) so the configured Apify Actor returns more.
- Or analyse any export you are entitled to use (a provider's bulk download, a
  seller-centre export, a research dataset):

```powershell
curl -X POST http://127.0.0.1:8000/api/reviews/classify -H "Content-Type: application/json" `
  -d '{"reviews":[{"text":"...","rating":5,"date":"5 March 2024"}]}'
```

## Extended cross-store matching

The original Amazon <-> Flipkart search is unchanged. On top of it,
`app/multistore.py` runs after enrichment and adds:

- precise queries built from brand, model code and key specs (`app/product_identity.py`);
- more stores, chosen by product category from `config/stores.json`:
  Croma, Reliance Digital, Vijay Sales, Tata CLiQ, Snapdeal, Myntra, Ajio, Nykaa, Decathlon
  (plus Amazon and Flipkart when the source is on a different store);
- attribute-aware verification (`app/matcher.py`): accessories, other storage/RAM,
  pack sizes, weights, variants (Pro/Ultra/FE), model numbers, gender and
  implausible prices are rejected or kept unverified, and each match lists why.

Stores are opened as an ordinary visitor would. A store that shows a
CAPTCHA/access wall is reported as `blocked` and skipped; nothing tries to defeat
it. JioMart, Meesho and BigBasket are disabled in `config/stores.json` because
they return no products to a plain headless request. Set `ENHANCED_MATCHING=false`
to run the legacy search only.

## Any store link, every other store, one price comparison

Paste a product link from any online store. `POST /api/analyse` then:

1. **Reads the product** (`app/source_extractor.py`). Amazon and Flipkart keep the original `scraper.js` path.
   For every other store the page's own product markup (schema.org JSON-LD, microdata, OpenGraph) is read
   for title, price, brand, image, GTIN/MPN, rating and availability, using a plain HTTP request first and one
   headless render if that is not enough. A CAPTCHA/access-denied page, a sign-in wall, an HTTP error page
   or a page with no product on it returns `Could not read this page: <reason>`; nothing tries to get around it.
   Local and private addresses are refused.
2. **Searches the other stores** (`app/multistore.py`, `config/stores.json`): never the source store itself,
   Amazon and Flipkart first when the source is elsewhere, up to `ENHANCED_MAX_STORES` (default 8).
3. **Builds `price_comparison`** (`app/price_graph.py`): one entry per store with link, price, rating, match
   reasons/flags and verified / possible status. `lowest` is the cheapest of the pasted listing and *verified*
   matches that have a price and are not out of stock, refurbished/used or in another currency. Possible
   matches are listed apart and can never be the lowest.
4. **Plots only observed prices.** Each store's points are its live price now plus rows genuinely stored in
   the `observations` table for that listing; in-stock cross-store prices are saved after each analysis so
   history grows (`RECORD_CANDIDATE_PRICES=false` turns that off). Nothing is back-filled.

Known limits: the legacy `record_price()` still pads a history of fewer than two stored rows with invented
180/120/60/30-day prices; the new graph ignores it and the UI shows its 6M min/avg/max/trend as "not enough
data" in that case. A store that serves an empty app shell or a wall to automated requests is reported as
unreadable. Sources priced in a currency other than INR are left out of the comparison.

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
