/*
 * Multi-store product search for TrustEngine (additive: scraper.js is untouched).
 *
 *   node scraper_search.js search '<json>'   -> candidate listings from several stores
 *   node scraper_search.js deep   '<json>'   -> structured details for specific listing URLs
 *   node scraper_search.js page   '<json>'   -> rendered DOM of one product page ({ "url": "https://..." })
 *
 * search input : { "queries": ["precise", "broad"], "stores": ["Croma", ...], "per_store": 8 }
 * deep input   : { "urls": [{ "platform": "Croma", "url": "https://..." }] }
 *
 * Stores come from config/stores.json. Only public pages are opened. A page that
 * shows a CAPTCHA / access-denied wall is reported as "blocked" and skipped;
 * nothing here tries to defeat bot protection or sign in anywhere.
 * Output: one JSON object on the last stdout line.
 */
const fs = require('fs');
const path = require('path');

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36';
const STORES = JSON.parse(fs.readFileSync(path.join(__dirname, 'config', 'stores.json'), 'utf8')).stores;
const MAX_PARALLEL_PAGES = 5;

const sleep = ms => new Promise(r => setTimeout(r, ms));

function searchUrl(store, query) {
  const slug = query.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return STORES[store].search
    .replace(/\{q\}/g, encodeURIComponent(query))
    .replace(/\{slug\}/g, slug);
}

/* Runs inside the page. Finds product cards without relying on site-specific
 * (and frequently obfuscated) class names: a card is the smallest ancestor of a
 * product link that also holds a price and does not contain a second product. */
function extractCardsInPage(opts) {
  const { linkContains, perStore } = opts;
  const PRICE = /(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{1,2})?)/gi;
  const NOISE_BEFORE_PRICE = /(?:emi|save|off|extra|cashback|from|starting|\/\s*month|per month)\s*[:\-]?\s*$/i;
  const abs = href => { try { return new URL(href, location.href).href.split('#')[0]; } catch (e) { return null; } };
  const auto = linkContains.includes('*');                     // no known URL pattern: judge a link by its shape
  const looksLikeProduct = href => {
    try {
      const u = new URL(href);
      if (u.origin !== location.origin) return false;
      const path = u.pathname;
      if (path.length < 14 || /\/(search|cart|login|account|wishlist|checkout|help|about|contact|offers?|stores?|brands?|categor(y|ies)|collections?|blogs?|pages?)(\/|$)/i.test(path)) return false;
      return /\/(products?|p|pd|item|buy)\//i.test(path) || /\d{4,}/.test(path) || /-[a-z0-9]{8,}$/i.test(path);
    } catch (e) { return false; }
  };
  const isProductLink = href => !!href && (auto ? looksLikeProduct(href) : linkContains.some(p => href.includes(p)));
  const canonical = href => href.replace(/\/ref=[^/?]*/i, '').replace(/[?&](?:ref|pf_rd_[a-z]+|pd_rd_[a-z]+|qid|sr|sprefix|crid|lid|otracker\d*|marketplace|spid|srno|fm|iid|ssid|cmpid|cmp)=[^&]*/gi, '').replace(/\?&/, '?').replace(/\?$/, '');

  const prices = text => {
    const out = [];
    let m;
    PRICE.lastIndex = 0;
    while ((m = PRICE.exec(text)) !== null) {
      const before = text.slice(Math.max(0, m.index - 14), m.index);
      if (NOISE_BEFORE_PRICE.test(before)) continue;
      const value = parseFloat(m[1].replace(/,/g, ''));
      if (value > 0) out.push(value);
    }
    return out;
  };

  const cards = new Map();
  for (const a of document.querySelectorAll('a[href]')) {
    const href = abs(a.getAttribute('href'));
    if (!isProductLink(href)) continue;

    let node = a, card = null;
    for (let depth = 0; depth < 8 && node && node !== document.body; depth++, node = node.parentElement) {
      const distinct = new Set();
      for (const link of node.querySelectorAll('a[href]')) {
        const h = abs(link.getAttribute('href'));
        if (isProductLink(h)) distinct.add(canonical(h));
      }
      if (distinct.size > 1) break;          // grew past a single product
      const nodeText = node.innerText || '';
      if (nodeText.length > 1800) break;     // no longer one card
      if (prices(nodeText).length) card = node;   // keep climbing: the widest single-product node also holds the photo
    }
    if (!card) continue;

    const key = canonical(href);
    if (cards.has(key)) continue;

    const text = (card.innerText || '').replace(/[ \t]+/g, ' ');
    const lines = text.split('\n').map(s => s.trim()).filter(Boolean);
    const img = [...card.querySelectorAll('img')].sort((x, y) => (y.getAttribute('alt') || '').length - (x.getAttribute('alt') || '').length)[0] || null;
    const imgSrc = img ? (img.currentSrc || img.getAttribute('src') || img.getAttribute('data-src') ||
                         (img.getAttribute('srcset') || '').split(' ')[0] || null) : null;
    const junk = /^(?:onlyonline|sponsored|ad|add to compare|new|bestseller|best seller|trending|hot|limited deal|deal of the day|out of stock|compare)$|extra deals|price, product page|product page|or fastest delivery|free delivery|add to cart|buy now|view details/i;
    const tidy = t => String(t || '').replace(/^sponsored ad\s*[-–:]\s*/i, '').replace(/^(?:product )?image of\s+/i, '').replace(/\s+/g, ' ').trim();
    const valid = t => t.length >= 6 && t.length <= 200 && !junk.test(t) && /[a-z]{3}/i.test(t);
    // Headings, title attributes and image alt text are trusted; free text lines are a last resort.
    const trusted = [
      ...[...card.querySelectorAll('h1,h2,h3,h4,[class*="title" i],[class*="name" i]')].map(e => e.innerText),
      a.getAttribute('title'), ...[...card.querySelectorAll('img')].map(i => i.getAttribute('alt')),
    ].map(tidy).filter(valid).sort((x, y) => y.length - x.length);
    const fallback = lines.filter(l => !/(?:₹|Rs\.?)\s*\d/i.test(l)).map(tidy).filter(valid).sort((x, y) => y.length - x.length);
    let title = trusted[0] || fallback[0] || '';
    if (title.length < 18) {
      // Fall back to the product slug in the URL ("samsung-galaxy-a37-5g-8gb-ram" -> readable title).
      const slug = decodeURIComponent(new URL(href).pathname).split('/').filter(seg => /[a-z]+-[a-z0-9]+-/i.test(seg)).sort((x, y) => y.length - x.length)[0];
      if (slug) title = slug.replace(/(?<!\d)(\d)-(\d)(?=-(?:inch|ton|tons|kg|l|ltr|litre|cm|mm)\b)/gi, '$1.$2').replace(/[-_]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).trim();
    }
    if (title.length < 6 || /^(add to compare|sponsored|ad)$/i.test(title)) continue;

    const list = prices(text);
    const ratingMatch = (card.innerText || '').match(/(?:^|\s)([1-5](?:\.\d)?)\s*(?:★|out of 5|stars?)/i)
      || (card.querySelector('[aria-label*="out of 5" i],[aria-label*="stars" i]') || { getAttribute: () => '' })
        .getAttribute('aria-label').match(/([1-5](?:\.\d)?)\s*out of 5/i);
    cards.set(key, {
      title: title.slice(0, 220),
      url: href,
      price: list[0] || null,
      mrp: list.length > 1 ? Math.max(...list) : null,
      rating: ratingMatch ? parseFloat(ratingMatch[1]) : null,
      image_url: imgSrc && !imgSrc.startsWith('data:') ? abs(imgSrc) : null,
      out_of_stock: /out of stock|currently unavailable|sold out/i.test(text),
    });
    if (cards.size >= perStore * 3) break;
  }
  return [...cards.values()];
}

const BLOCK_RE = /captcha|access denied|are you a robot|unusual traffic|verify you are human|robot check|enter the characters you see/i;

async function newPage(browser) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  await page.setUserAgent(USER_AGENT);
  return page;
}

async function searchOneStore(browser, store, queries, perStore, gate) {
  const config = STORES[store];
  const report = { platform: store, status: 'empty', result_count: 0, query: null, attempts: 0 };
  const started = Date.now();
  for (const query of queries) {
    report.attempts++;
    await gate.acquire();
    let page = null;
    try {
      page = await newPage(browser);
      let captured = null;
      if (config.json_capture) {
        page.on('response', async r => {
          if (!r.url().includes(config.json_capture.url_contains)) return;
          try { captured = JSON.parse(await r.text()); } catch (e) { /* not JSON */ }
        });
      }
      const response = await page.goto(searchUrl(store, query), { waitUntil: 'domcontentloaded', timeout: 20000 });
      if (response && response.status() >= 400) { report.status = `http_${response.status()}`; continue; }

      // Single-page apps render results late, so poll for product cards instead of
      // sleeping a fixed time: stop as soon as the card count is stable.
      const deadline = Date.now() + (config.max_wait_ms || 14000);
      let items = [], previous = -1, scrolled = false, blocked = false;
      await sleep(Math.min(config.settle_ms || 2000, 2500));
      while (Date.now() < deadline) {
        blocked = await page.evaluate(re => new RegExp(re, 'i').test((document.body ? document.body.innerText : '').slice(0, 2500)), BLOCK_RE.source).catch(() => false);
        if (blocked) break;
        items = captured ? itemsFromJson(config.json_capture, captured)
                         : await page.evaluate(extractCardsInPage, { linkContains: config.link_contains, perStore }).catch(() => []);
        if (captured) break;                       // the store answered with JSON (possibly an empty result)
        if (items.length && items.length === previous) break;
        previous = items.length;
        if (!scrolled && items.length) { scrolled = true; await page.evaluate(() => window.scrollBy(0, 1400)).catch(() => {}); }
        await sleep(900);
      }
      if (blocked) { report.status = 'blocked'; break; }
      report.query = query;
      if (items.length) {
        report.status = 'ok';
        report.result_count = items.length;
        report.elapsed_ms = Date.now() - started;
        return { report, items: items.slice(0, perStore * 3).map(i => ({ ...i, platform: store, query })) };
      }
    } catch (e) {
      report.status = `error: ${String(e.message || e).slice(0, 80)}`;
    } finally {
      if (page) await page.close().catch(() => {});
      gate.release();
    }
  }
  report.elapsed_ms = Date.now() - started;
  return { report, items: [] };
}

/* Maps the JSON a store's own page requested (see "json_capture" in the registry). */
function itemsFromJson(spec, payload) {
  const at = (obj, dotted) => dotted.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
  const list = at(payload, spec.list_path);
  if (!Array.isArray(list)) return [];
  const absolute = u => (u ? (/^\/\//.test(u) ? 'https:' + u : /^https?:/i.test(u) ? u : spec.base_url + u) : null);
  const number = v => { const n = parseFloat(String(v == null ? '' : v).replace(/,/g, '')); return Number.isFinite(n) && n > 0 ? n : null; };
  return list.map(it => ({
    title: String(at(it, spec.fields.title) || '').trim(),
    url: absolute(at(it, spec.fields.url)),
    price: number(at(it, spec.fields.price)),
    mrp: spec.fields.mrp ? number(at(it, spec.fields.mrp)) : null,
    rating: number(at(it, spec.fields.rating)),
    review_count: number(at(it, spec.fields.review_count)),
    image_url: absolute(at(it, spec.fields.image_url)),
    out_of_stock: it.inStockFlag === false,
  })).filter(i => i.title.length >= 6 && i.url);
}

function makeGate(limit) {
  let active = 0; const waiting = [];
  return {
    async acquire() { if (active >= limit) await new Promise(r => waiting.push(r)); active++; },
    release() { active--; const next = waiting.shift(); if (next) next(); },
  };
}

async function launch() {
  return require('./browser_launch').launchQuiet(require('puppeteer'));
}

async function searchStores({ queries, stores, per_store = 8 }) {
  const wanted = (stores || []).filter(s => STORES[s] && STORES[s].enabled !== false);
  const browser = await launch();
  try {
    const gate = makeGate(MAX_PARALLEL_PAGES);
    // One slow store must not cost the results of the others: each store gets a deadline.
    const storeDeadlineMs = Number(process.env.STORE_DEADLINE_MS || 50000);
    // The deadline timer is cleared as soon as the store answers: left running it keeps the process alive
    // for the full deadline after every result is in, which pushed whole searches past Python's timeout.
    const withDeadline = (store, work) => {
      let timer;
      const deadline = new Promise(resolve => {
        timer = setTimeout(() => resolve({
          report: { platform: store, status: 'timeout', result_count: 0, query: null, attempts: 0, elapsed_ms: storeDeadlineMs }, items: [],
        }), storeDeadlineMs);
      });
      return Promise.race([work, deadline]).finally(() => clearTimeout(timer));
    };
    // Each store's answer is printed the moment it is in ({"store_done": ...}), so the caller can show it at once;
    // the complete result is still printed last.
    const results = await Promise.all(wanted.map(s => withDeadline(s, searchOneStore(browser, s, queries, per_store, gate)).then(r => {
      process.stdout.write(JSON.stringify({ store_done: r.report, items: r.items }) + '\n');
      return r;
    })));
    return { stores: results.map(r => r.report), matches: results.flatMap(r => r.items) };
  } finally {
    await browser.close().catch(() => {});
  }
}

/* Runs inside a product page: prefer schema.org Product JSON-LD. */
function extractProductInPage() {
  const out = { title: null, price: null, rating: null, review_count: null, image_url: null, brand: null, gtin: null, mpn: null, sku: null, availability: null };
  const num = v => { const n = parseFloat(String(v == null ? '' : v).replace(/,/g, '')); return Number.isFinite(n) ? n : null; };
  const visit = node => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(visit); return; }
    const type = [].concat(node['@type'] || []).join(',');
    if (/Product/i.test(type) && !/ProductGroup/i.test(type) || /ProductGroup/i.test(type)) {
      out.title = out.title || node.name || null;
      out.brand = out.brand || (node.brand && (node.brand.name || node.brand)) || null;
      out.gtin = out.gtin || node.gtin13 || node.gtin || node.gtin12 || node.gtin14 || node.gtin8 || null;
      out.mpn = out.mpn || node.mpn || null;
      out.sku = out.sku || node.sku || null;
      const image = Array.isArray(node.image) ? node.image[0] : node.image;
      out.image_url = out.image_url || (image && (image.url || image)) || null;
      const offers = [].concat(node.offers || [])[0];
      if (offers) {
        out.price = out.price || num(offers.price != null ? offers.price : offers.lowPrice);
        out.availability = out.availability || offers.availability || null;
      }
      const agg = node.aggregateRating;
      if (agg) {
        out.rating = out.rating || num(agg.ratingValue);
        out.review_count = out.review_count || num(agg.reviewCount || agg.ratingCount);
      }
    }
    Object.values(node).forEach(v => { if (v && typeof v === 'object') visit(v); });
  };
  document.querySelectorAll('script[type="application/ld+json"]').forEach(s => { try { visit(JSON.parse(s.textContent)); } catch (e) {} });
  const meta = (p) => { const el = document.querySelector(`meta[property="${p}"],meta[name="${p}"]`); return el ? el.getAttribute('content') : null; };
  out.title = out.title || meta('og:title') || document.title || null;
  out.image_url = out.image_url || meta('og:image');
  out.price = out.price || num(meta('product:price:amount'));
  return out;
}

async function deepVisit({ urls }) {
  const browser = await launch();
  try {
    const gate = makeGate(MAX_PARALLEL_PAGES);
    const details = await Promise.all((urls || []).slice(0, 8).map(async ({ platform, url }) => {
      await gate.acquire();
      let page = null;
      try {
        page = await newPage(browser);
        const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
        if (response && response.status() >= 400) return { platform, url, final_url: page.url(), status: `http_${response.status()}` };
        await sleep((STORES[platform] && STORES[platform].settle_ms) ? Math.min(STORES[platform].settle_ms, 3000) : 1500);
        const blocked = await page.evaluate(re => new RegExp(re, 'i').test((document.body ? document.body.innerText : '').slice(0, 2500)), BLOCK_RE.source).catch(() => false);
        if (blocked) return { platform, url, status: 'blocked' };
        return { platform, url, final_url: page.url(), status: 'ok', ...(await page.evaluate(extractProductInPage)) };
      } catch (e) {
        return { platform, url, status: `error: ${String(e.message || e).slice(0, 80)}` };
      } finally {
        if (page) await page.close().catch(() => {});
        gate.release();
      }
    }));
    return { details };
  } finally {
    await browser.close().catch(() => {});
  }
}

/* Renders one product page and returns its final DOM (app/source_extractor.py parses it).
 * Used for source links from stores scraper.js has no selectors for. A wall (CAPTCHA,
 * access denied) is returned as-is: the Python side recognises it and reports it. */
async function pageHtml({ url }) {
  const browser = await launch();
  try {
    const page = await newPage(browser);
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 25000 });
    const hasProductData = () => page.evaluate(() =>
      [...document.querySelectorAll('script[type="application/ld+json"]')].some(s => /"@type"\s*:\s*\[?\s*"(?:Product|ProductGroup)/.test(s.textContent)) ||
      !!document.querySelector('meta[property="product:price:amount"],[itemprop="price"]')).catch(() => false);
    const deadline = Date.now() + 9000;
    await sleep(1200);
    while (Date.now() < deadline && !(await hasProductData())) await sleep(700);
    const html = await page.content();
    return { status: response ? response.status() : null, final_url: page.url(), html: html.slice(0, 3000000) };
  } finally {
    await browser.close().catch(() => {});
  }
}

module.exports = { searchStores, deepVisit, pageHtml, STORES, searchUrl };

if (require.main === module) {
  (async () => {
    const [mode, raw] = process.argv.slice(2);
    try {
      const input = JSON.parse(raw || '{}');
      const result = mode === 'deep' ? await deepVisit(input) : mode === 'page' ? await pageHtml(input) : await searchStores(input);
      process.stdout.write(JSON.stringify(result) + '\n', () => process.exit(0));   // results are final: do not wait on stray handles
    } catch (e) {
      console.error(JSON.stringify({ error: String(e.message || e) }));
      process.exit(1);
    }
  })();
}
