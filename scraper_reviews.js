/*
 * Read the reviews a product page shows to an ordinary logged-out visitor, on any store.
 *
 *   node scraper_reviews.js '{"url": "https://...", "limit": 120}'
 *
 * Output (last stdout line): { reviews: [{text, rating, date, author, verified_purchase}],
 *                              via: [...], reported_total, issues: [...], seconds }
 *
 * How, in order (everything found is merged and de-duplicated):
 *   1. schema.org JSON-LD / microdata reviews in the page;
 *   2. JSON the page itself fetched for its review widget (any JSON object that has review-like text
 *      and a 1-5 rating) - the same data the visitor's screen is drawn from;
 *   3. the review blocks drawn on the page: the smallest repeated element that holds both a star
 *      rating and a paragraph of prose.
 * The page is scrolled and its "reviews" / "load more reviews" controls are pressed a few times, like a
 * visitor reading them. A CAPTCHA / access-denied wall is reported and left alone; there is no sign-in,
 * no bot-protection handling and no calls the page does not make itself.
 */
const puppeteer = require('puppeteer');

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const WALL = /captcha|access denied|are you a robot|unusual traffic|verify you are human|robot check|enter the characters you see/i;

/* ---- JSON: any object with review-like text and a 1-5 rating ------------------------------------------------ */
const TEXT_KEYS = ['reviewBody', 'reviewText', 'review_text', 'reviewtext', 'extract', 'comment', 'comments', 'text', 'body', 'content', 'description', 'message', 'review', 'detail'];
const RATING_MAX_KEYS = ['score_max', 'scoreMax', 'max_score', 'maxScore', 'ratingMax', 'bestRating', 'best_rating'];
const RATING_KEYS = ['ratingValue', 'rating', 'stars', 'star', 'score', 'overallRating', 'userRating', 'reviewRating', 'starRating', 'ratingStar'];
const DATE_KEYS = ['datePublished', 'date', 'created', 'createdAt', 'created_at', 'reviewDate', 'submissionTime', 'submitted_at', 'postedOn', 'timestamp', 'updatedAt'];
const AUTHOR_KEYS = ['author', 'authorName', 'userName', 'username', 'nickname', 'reviewer', 'customerName', 'name'];

function pick(obj, keys) {
  const lower = {};
  for (const k of Object.keys(obj)) lower[k.toLowerCase()] = obj[k];
  for (const k of keys) {
    const v = lower[k.toLowerCase()];
    if (v !== undefined && v !== null && v !== '') return v;
  }
  return undefined;
}

function asRating(v, max) {
  if (v && typeof v === 'object') v = pick(v, ['ratingValue', 'value', 'rating']);
  let n = parseFloat(String(v));
  const top = parseFloat(String(max));
  if (Number.isFinite(n) && Number.isFinite(top) && top > 5 && top <= 100 && n >= 0 && n <= top) n = (n / top) * 5;     // "10 out of 10" -> 5
  return Number.isFinite(n) && n >= 1 && n <= 5 ? Math.round(n) : (Number.isFinite(n) && n > 0 && n < 1 ? 1 : null);
}

function collectFromJson(node, out, depth = 0) {
  if (depth > 12 || node === null) return;
  if (Array.isArray(node)) { node.forEach(n => collectFromJson(n, out, depth + 1)); return; }
  if (typeof node !== 'object') return;
  const text = pick(node, TEXT_KEYS);
  const rating = asRating(pick(node, RATING_KEYS), pick(node, RATING_MAX_KEYS));
  if (typeof text === 'string' && text.trim().length >= 12 && text.length <= 5000 && rating && /[a-z]{3}/i.test(text) && !/^https?:/i.test(text.trim())) {
    const author = pick(node, AUTHOR_KEYS);
    out.push({
      text: text.replace(/\s+/g, ' ').trim(), rating,
      date: String(pick(node, DATE_KEYS) || '') || null,
      author: typeof author === 'string' ? author : (author && typeof author === 'object' ? (author.name || null) : null),
      origin: typeof pick(node, ['domain', 'source']) === 'string' ? pick(node, ['domain', 'source']) : null,
      verified_purchase: /verified|certified/i.test(JSON.stringify(Object.keys(node)) + ' ' + String(pick(node, ['verifiedPurchase', 'isVerifiedPurchase', 'certifiedBuyer', 'verified_buyer', 'isVerified']) || '')),
    });
  }
  for (const v of Object.values(node)) collectFromJson(v, out, depth + 1);
}

/* ---- DOM: repeated blocks that each hold a star rating and a paragraph of prose --------------------------------
 * Strict on purpose: a page full of offers, descriptions and feedback forms must never be mistaken for reviews.
 * A block counts only if it (1) shows a rating, (2) holds prose that is not store furniture, and (3) belongs to a
 * list of at least three look-alike blocks (or two inside something named "review"). */
function extractFromDom(root) {
  const FURNITURE = /\bEMI\b|cash\s?back|\bMRP\b|apply coupon|add to cart|view all|sign in|log in|delivery (by|check)|returnable|price match|payment offers|bank offer|we value your feedback|how would you rate|share your feedback|sort by|filter by|write a review|product description|available offers|terms (and|&) conditions|free shipping|exchange offer|no cost/i;
  const SKIP_ZONE = /offer|coupon|emi|promo|breadcrumb|footer|header|nav|recommend|similar|carousel|related|banner|feedback-form|popup|modal|description|specification/i;
  const classOf = el => (el.className && el.className.toString ? el.className.toString() : '');
  const reviewish = el => {
    for (let n = el, d = 0; n && n.nodeType === 1 && d < 5; n = n.parentElement, d++) {
      if (/review|testimonial|ugc/i.test(classOf(n) + ' ' + (n.id || '') + ' ' + (n.getAttribute('data-testid') || '') + ' ' + (n.getAttribute('itemprop') || ''))) return true;
    }
    return false;
  };
  const inSkipZone = el => {
    for (let n = el, d = 0; n && n.nodeType === 1 && d < 6; n = n.parentElement, d++) {
      if (n.tagName === 'NAV' || n.tagName === 'FOOTER' || n.tagName === 'HEADER') return true;
      if (SKIP_ZONE.test(classOf(n)) && !/review/i.test(classOf(n))) return true;
    }
    return false;
  };
  const proseOk = t => {
    const words = t.split(/\s+/).filter(Boolean);
    const letters = (t.match(/[A-Za-z\u0900-\u097F]/g) || []).length;
    const glued = (t.match(/[a-z][A-Z]/g) || []).length;             // "Most HelpfulSORT BY": labels run together
    return words.length >= 3 && t.length >= 15 && t.length <= 3000 && letters / t.length > 0.6 && glued < 3
      && !FURNITURE.test(t) && (t.match(/[₹$]/g) || []).length < 2 && (t.match(/\d\/5/g) || []).length < 2;   // "5/5 cushioning 5/5 traction" is a score table
  };
  const strongRating = el => {
    const prop = el.querySelector('[itemprop="ratingValue"]');
    if (prop) { const n = parseFloat(prop.getAttribute('content') || prop.textContent); if (n >= 1 && n <= 5) return Math.round(n); }
    for (const a of el.querySelectorAll('[aria-label],[title],img[alt]')) {
      const label = a.getAttribute('aria-label') || a.getAttribute('title') || a.getAttribute('alt') || '';
      const m = label.match(/([1-5](?:\.\d)?)\s*(?:out of 5|\/\s*5|stars?|star rating)/i) || label.match(/rated\s*([1-5](?:\.\d)?)/i);
      if (m) return Math.round(parseFloat(m[1]));
    }
    const filled = el.querySelectorAll('[class*="star" i][class*="fill" i],[class*="star" i][class*="full" i],[class*="star" i][class*="active" i],[class*="star" i][class*="selected" i]');
    return filled.length >= 1 && filled.length <= 5 ? filled.length : null;
  };
  const plainRating = el => {                                         // a lone digit 1-5 / "4.5" / "5 ★" in its own small element
    for (const small of el.querySelectorAll('span,div,i,b,strong,p')) {
      if (small.children.length > 1) continue;
      const t = (small.textContent || '').trim();
      if (/^[1-5](?:\.[0-9])?$/.test(t)) return Math.round(parseFloat(t));
      const m = t.match(/^([1-5](?:\.[0-9])?)\s*(?:★|⭐|\/\s*5)$/);
      if (m) return Math.round(parseFloat(m[1]));
    }
    return null;
  };
  const proseIn = el => {
    let best = '';
    for (const n of el.querySelectorAll('p,span,div,q,blockquote')) {
      if (n.children.length > 2) continue;
      const t = (n.textContent || '').replace(/\s+/g, ' ').trim();
      if (t.length > best.length && proseOk(t) && !/^(helpful|report|was this|verified|certified)/i.test(t)) best = t;
    }
    return best;
  };
  const dateRe = /(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})|((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+(?:\d{1,2},?\s+)?\d{4})|(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4})|(\d+\s+(?:day|week|month|year)s?\s+ago)|(today|yesterday)/i;

  const hits = [];
  for (const el of root.querySelectorAll('div,li,article,section')) {
    const len = (el.innerText || el.textContent || '').length;
    if (len < 40 || len > 3000 || inSkipZone(el)) continue;
    const named = reviewish(el);
    const rating = strongRating(el) || (named ? plainRating(el) : null);
    if (!rating) continue;
    const text = proseIn(el);
    if (!text) continue;
    hits.push({ el, text, rating, named });
  }
  // smallest element per review
  const smallest = hits.filter(h => !hits.some(o => o !== h && h.el.contains(o.el) && o.text === h.text));
  // must belong to a list of look-alikes
  const sig = h => `${h.el.parentElement ? h.el.parentElement.tagName + '.' + classOf(h.el.parentElement) : ''}>${h.el.tagName}.${classOf(h.el)}`;
  const counts = {};
  smallest.forEach(h => { const k = sig(h); counts[k] = (counts[k] || 0) + 1; });
  const listed = smallest.filter(h => counts[sig(h)] >= 3 || (h.named && counts[sig(h)] >= 2));

  const strip = t => t.replace(/^[1-5](?=[A-Z“"'‘’])/, '').trim();       // "4I've been..." -> "I've been..."
  const seen = [];
  const out = [];
  for (const h of listed) {
    const text = strip(h.text);
    const key = text.toLowerCase();
    if (seen.some(k => k === key || k.includes(key) || key.includes(k))) continue;
    seen.push(key);
    const blob = (h.el.innerText || h.el.textContent || '').replace(/\s+/g, ' ');
    out.push({ text, rating: h.rating, date: (blob.match(dateRe) || [null])[0], author: null, verified_purchase: /verified|certified/i.test(blob) });
  }
  return out;
}

function ldAndMicrodata() {
  const out = [];
  const reported = { value: null };
  const walk = n => {
    if (Array.isArray(n)) n.forEach(walk);
    else if (n && typeof n === 'object') {
      if (n.reviewBody) {
        const r = n.reviewRating;
        out.push({ text: String(n.reviewBody).replace(/\s+/g, ' ').trim(), rating: Math.round(parseFloat(r && typeof r === 'object' ? r.ratingValue : r)) || null,
                   date: n.datePublished || null, author: n.author && (n.author.name || n.author) || null, verified_purchase: false });
      }
      if (n.aggregateRating && n.aggregateRating.reviewCount) reported.value = parseInt(String(n.aggregateRating.reviewCount).replace(/,/g, ''), 10);
      Object.values(n).forEach(walk);
    }
  };
  document.querySelectorAll('script[type="application/ld+json"]').forEach(s => { try { walk(JSON.parse(s.textContent)); } catch (e) {} });
  document.querySelectorAll('[itemprop="review"]').forEach(sc => {
    const body = sc.querySelector('[itemprop="reviewBody"]');
    const rv = sc.querySelector('[itemprop="ratingValue"]');
    if (body && body.textContent.trim().length > 10) out.push({ text: body.textContent.replace(/\s+/g, ' ').trim(), rating: rv ? Math.round(parseFloat(rv.getAttribute('content') || rv.textContent)) || null : null, date: null, author: null, verified_purchase: false });
  });
  return { reviews: out, reported: reported.value };
}

async function pressReviewControls(page, rounds) {
  for (let i = 0; i < rounds; i++) {
    const pressed = await page.evaluate(() => {
      const wanted = /^(see|view|read|show|load|more)\b.{0,25}\breviews?\b|^reviews?\s*\(?\d|^(load|show|view|see) more$|^ratings?\s*(&|and)\s*reviews?$|^all reviews?$/i;
      const unwanted = /cart|buy|wishlist|write|login|sign|add|compare|share|notify|emi|offer/i;
      const nodes = [...document.querySelectorAll('button,a,[role="button"],[role="tab"],summary')];
      const target = nodes.find(n => {
        const t = (n.innerText || n.textContent || '').trim();
        return t.length > 0 && t.length < 45 && wanted.test(t) && !unwanted.test(t) && !n.dataset.__pressed;
      });
      if (!target) return false;
      target.dataset.__pressed = '1';
      target.scrollIntoView({ block: 'center' });
      target.click();
      return true;
    }).catch(() => false);
    if (!pressed) break;
    await sleep(1200);
  }
}

async function collect({ url, limit = 120, budget_ms = 30000 }) {
  const started = Date.now();
  const issues = [];
  const via = [];
  const browser = await require('./browser_launch').launchQuiet(puppeteer);
  try {
    const page = await browser.newPage();
    await page.setUserAgent(USER_AGENT);
    await page.setViewport({ width: 1280, height: 900 });
    const bodies = [];
    page.on('response', async r => {
      try {
        const u = r.url();
        if (/\.(js|css|png|jpe?g|webp|gif|svg|woff2?|ico|mp4)(\?|$)/i.test(u) || u.startsWith('data:')) return;
        const type = (r.headers()['content-type'] || '').toLowerCase();
        if (!/json|html/.test(type)) return;
        const text = await r.text();
        if (text.length < 200 || text.length > 5_000_000) return;
        if (/review|rating|ugc|feedback|comment|bazaarvoice|yotpo|judge|powerreview|okendo|stamped|loox|trustpilot|opinion/i.test(u) || text.includes('reviewBody')) bodies.push({ url: u, type, text });
      } catch (e) { /* response body not available */ }
    });

    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 25000 });
    if (response && response.status() >= 400) issues.push(`the page answered HTTP ${response.status()}`);
    await sleep(2500);
    const wall = await page.evaluate(re => new RegExp(re, 'i').test((document.body ? document.body.innerText : '').slice(0, 2500)), WALL.source).catch(() => false);
    if (wall) return { reviews: [], via, reported_total: null, issues: [...issues, 'the store showed a CAPTCHA / access-denied page; nothing was read'], seconds: (Date.now() - started) / 1000 };

    for (let i = 0; i < 7 && Date.now() - started < budget_ms * 0.6; i++) { await page.evaluate(() => window.scrollBy(0, 1000)).catch(() => {}); await sleep(350); }
    await pressReviewControls(page, 4);
    for (let i = 0; i < 4 && Date.now() - started < budget_ms; i++) { await page.evaluate(() => window.scrollBy(0, 1200)).catch(() => {}); await sleep(450); }
    await sleep(800);

    const found = [];
    const markup = await page.evaluate(ldAndMicrodata).catch(() => ({ reviews: [], reported: null }));
    if (markup.reviews.length) { via.push('page markup'); found.push(...markup.reviews); }

    if (process.env.DEBUG_REVIEWS) bodies.forEach(b => { console.error('  captured', b.type.split(';')[0], b.text.length + 'B', b.url.slice(0, 130)); if (process.env.DEBUG_REVIEWS === '2' && /json/.test(b.type)) console.error('     ', b.text.replace(/\s+/g, ' ').slice(0, 700)); });
    const fromJson = [];
    for (const b of bodies) {
      if (/json/.test(b.type) || /^\s*[\[{]/.test(b.text)) { try { collectFromJson(JSON.parse(b.text), fromJson); } catch (e) { /* not JSON after all */ } }
    }
    if (fromJson.length) { via.push('the review data the page loaded'); found.push(...fromJson); }

    const domReviews = await page.evaluate(`(${extractFromDom.toString()})(document)`).catch(() => []);
    if (domReviews.length) { via.push('reviews drawn on the page'); found.push(...domReviews); }

    // review fragments delivered as HTML (e.g. a "getReviews" call): read them in the page as a document
    for (const b of bodies.filter(x => /html/.test(x.type) && !/^\s*[\[{]/.test(x.text))) {
      const frag = await page.evaluate(`(html => (${extractFromDom.toString()})(new DOMParser().parseFromString(html, 'text/html')))(${JSON.stringify(b.text)})`).catch(() => []);
      if (frag.length) { via.push('a review fragment the page loaded'); found.push(...frag); }
    }

    const seen = new Set();
    const reviews = [];
    for (const r of found) {
      const key = r.text.toLowerCase().replace(/\s+/g, ' ').slice(0, 160);
      if (seen.has(key)) continue;
      seen.add(key);
      reviews.push(r);
      if (reviews.length >= limit) break;
    }
    if (!reviews.length) issues.push('no review text was visible to a logged-out visitor on this page');
    return { reviews, via: [...new Set(via)], reported_total: markup.reported, issues, seconds: (Date.now() - started) / 1000 };
  } finally {
    await browser.close().catch(() => {});
  }
}

module.exports = { collect, collectFromJson, extractFromDom };

if (require.main === module) {
  (async () => {
    try {
      const input = JSON.parse(process.argv[2] || '{}');
      const result = await collect(input);
      process.stdout.write(JSON.stringify(result) + '\n', () => process.exit(0));
    } catch (e) {
      console.error(JSON.stringify({ error: String(e.message || e) }));
      process.exit(1);
    }
  })();
}
