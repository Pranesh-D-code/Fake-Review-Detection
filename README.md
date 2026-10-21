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

### Reviews from any pasted link

`app/review_collectors.py` reads the reviews a store shows to a logged-out visitor, whatever the store:

1. **Store adapters** - Flipkart's public review pages (10 per request, several sort orders, up to ~150 reviews in
   a few seconds). The first answer already carries a quick sample.
2. **Reviews the page declares** in schema.org JSON-LD / microdata (Snapdeal, many others).
3. **A Chrome scan** (`scraper_reviews.js`) for pages that draw their reviews with JavaScript: it scrolls, presses
   the page's own "reviews" / "load more" controls and reads (a) the review JSON the page itself loaded
   (Nykaa) and (b) repeated review blocks that each hold a star rating and prose (Myntra). Blocks are accepted
   only inside a list of look-alikes, never from offers, descriptions or feedback forms.

A store that shows a CAPTCHA / access-denied wall (Ajio) or shows no review text to visitors (Croma, Reliance
Digital, Tata CLiQ, Vijay Sales for most products) simply yields no reviews, and the notice says so. Amazon shows its
full list only to signed-in users, so it keeps the product page plus the Apify provider.
`REVIEW_FETCH_LIMIT` (default 150) caps how many are read per link.

### Speed

- The first answer arrives in a few seconds for any store whose page declares its product (plain HTTP read, with
  Chrome only as a fallback); Flipkart no longer starts Chrome for it.
- Reviews, each store's matches and the Amazon provider run at the same time, and the page shows each as it
  arrives instead of waiting for the slowest. The review model is loaded while the server starts.
- The same link analysed again within 10 minutes is answered from memory in milliseconds.

### Training data and what it can and cannot do

```powershell
python scripts/fetch_review_datasets.py                 # downloads ~42 MB of open datasets into data/external/
python scripts/train_review_classifier.py --holdout-categories 0   # ~20 min; reads the Kaggle set in the database + those datasets
```

| Dataset | Licence | Used for |
|---|---|---|
| Kaggle fake-reviews set (already in `trustengine.db`) | CC BY 4.0 | real reviews + GPT-2-era fakes |
| `Flowerly/modern-fake-reviews` (DeepSeek), `-glm`, `-lfm` | CC BY 4.0 | fakes written by present-day LLMs |
| `janduplessis886/gp_surgery_reviews_fake_and_real` | MIT | **evaluation only** (different domain) |

The reason for the modern sets: a model trained only on the Kaggle fakes catches about 94% of those fakes but only
1-2% of reviews written by today's LLMs (measured in `models/review_detector.metrics.json`, probe `kaggle_only`).
Trained with them, the model catches 97-99% of held-out modern fakes, and when one generator is left out of
training it still catches 87-96% of that generator's fakes.

On real reviews collected by the app (about 45 Amazon and 150 Flipkart reviews, of which only the ones with 10+ words are
judged) it flags 6-17% as likely LLM-written and calls 40-50% genuine-looking. Nobody has
labelled those, so how many of the flags are right is unknown; the polished ones it flags look machine-written, a few
borderline ones look human. To stay cautious on shorter, differently-worded live reviews the flag never goes below
`REVIEW_FLAG_MIN_PROBABILITY` (default 0.9). The previous model is kept as `models/review_detector.prev.joblib`.

What it still cannot do: it does not carry over to other domains. On the UK GP-surgery set it is at chance
(49% accuracy). It learns the style of LLM text *in product reviews*; it is not a general AI-text detector, it has
not been checked on live marketplace reviews, and it cannot spot a human-written paid review. The metrics file
lists all of this next to the headline numbers.

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
it. Stores that return no products to a plain headless request (JioMart, Meesho,
BigBasket, Ajio, ...) are disabled in `config/stores.json` with the reason beside
each. Set `ENHANCED_MATCHING=false` to run the legacy search only.

### Text + photo matching

Every candidate is judged twice, independently, and the two must not contradict:

- **Text** (`app/matcher.py`): brand, model code, RAM/storage/size/weight/SPF, variant words, accessory
  and product-type conflicts, and words that name a different product ("conditioner" against "shampoo").
  `tests/fixtures/match_pairs.json` holds 111 hand-written pairs across 9 categories; `scripts/evaluate_matcher.py` scores them.
- **Photo** (`app/visual_match.py`): both store photos are trimmed to the product and embedded with DINOv2-small
  (local weights, no per-request API call). Fetch the model once with `python scripts/fetch_visual_model.py`
  (about 88 MB, Apache-2.0). Without it the app falls back to a perceptual hash and says so in the match
  reason. A matching photo adds confidence; a clearly different photo blocks the *verified* status. On 1,100 real store photo pairs
(labels from the text matcher, not hand-checked) DINOv2 separates same from different products with AUC 0.84 (0.80 against
same-category look-alikes) against 0.59 for the old hash; at the chosen thresholds a photo verdict of *same* fires on 34% of
true matches and 3% of different products, and *different* fires on 6% of true matches and 23% of different products. A photo
alone cannot tell near-identical siblings (M35 / M36) apart, which is why the text decides and the photo only supports it.

`python scripts/build_visual_pairs.py` collects real store photo pairs and `python scripts/evaluate_visual_matching.py`
measures the photo methods and writes `models/visual/calibration.json` and `evaluation.json`. The pair labels come
from the text matcher, not from a person, so those numbers show that the two signals agree, not hand-checked accuracy.

### Exact and similar products

Matches are split in two, and the page has a tab for each:

- **Exact product**: verified to be the very same product (brand, model code, RAM/storage/size/weight, variant words,
  edition, and the photo all agree). Only these count towards the *lowest price* and the price graph.
- **Similar products**: the same brand and kind of product that differs in a way a shopper may care about, found by
  similar text rather than the exact keyword: `Pro+` instead of `Pro`, 256 GB instead of 128 GB, the next model
  number (`Revolution 6` for `Revolution 7`), `Ultraboost 5X` for `Ultraboost 5`, a special edition, another model
  in the same range. Each card says **what is different**, and similar prices are never used as the lowest price.
  Accessories, other brands, other kinds of product, other genders and second-hand units are not listed at all.

Listings the matcher cannot confirm but also cannot rule out are shown as *Possible, not confirmed* on the similar tab.

### Links that open

Before a match is listed, its link is opened once (`app/link_check.py`). A link the store says is gone (404/410), that
redirects to the home page or to a search/category page, that shows a "not found" page, or that shows a different product
is **dropped**. A page the store will not show to a plain request (403/CAPTCHA/built by script) gets a second look in the
same headless-browser visit that reads ratings and identifiers; if that cannot read it either, the link is still listed
but without the *link checked* tick. Nothing tries to get past a CAPTCHA or access wall. `LINK_CHECK=false` switches this off.

Measured on live runs (a Samsung phone and an Adidas shoe): every link marked *checked* opened on the product when
re-opened in a separate browser; the unchecked ones were Ajio (403 to automated browsers) and Tata CLiQ (page built by
script), and Amazon's pages could not be read by a plain request. Whether those open for a person was not testable here.

### Quiet browsers

Nothing opens a visible window. Chrome is started headless (`browser_launch.js`) and every Python child process is
created without a console (`app/proc.py`). Headless mode is not a way round bot defences: pages that show a
CAPTCHA or access wall are still reported as blocked.

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
