const puppeteer = require('puppeteer-extra');

let mode = process.argv[2];
let targetUrl = process.argv[3];
const BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36';

if (!targetUrl) {
  targetUrl = mode;
  mode = 'extract';
}

if (!targetUrl) {
  console.error(JSON.stringify({ error: "URL argument missing" }));
  process.exit(1);
}

targetUrl = targetUrl.trim();
if (!/^https?:\/\//i.test(targetUrl)) {
  targetUrl = 'https://' + targetUrl;
}

function clean(str) {
  if (!str) return null;
  const s = String(str).replace(/\s+/g, ' ').trim();
  return s.length ? s : null;
}

async function configurePage(page) {
  await page.setViewport({ width: 1280, height: 800 });
  await page.setUserAgent(BROWSER_USER_AGENT);
}

function parseNumber(str) {
  if (!str) return null;
  const m = String(str).match(/(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{1,2})?)/i);
  return m ? parseFloat(m[1].replace(/,/g, '')) : null;
}

function getPlatform(url) {
  try {
    const host = new URL(url).hostname.toLowerCase();
    if (host.includes('amazon')) return 'Amazon';
    if (host.includes('flipkart')) return 'Flipkart';
    if (host.includes('nykaa')) return 'Nykaa';
    if (host.includes('myntra')) return 'Myntra';
    if (host.includes('meesho')) return 'Meesho';
    return host.replace('www.', '');
  } catch (e) {
    return 'Unknown store';
  }
}

function parseUrlSlug(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch (e) {
    return { platform: 'Unknown', title: null, brand: null, sku: null };
  }
  const platform = getPlatform(url);
  const path = decodeURIComponent(parsed.pathname);
  const parts = path.split('/').filter(p => p.trim());
  let title = null, brand = null, sku = null;

  if (platform === 'Amazon') {
    const mAsin = path.match(/\/(?:dp|gp\/product)\/([A-Z0-9]{10})/i) || path.match(/\/([A-Z0-9]{10})(?:[\?\/]|$)/i);
    if (mAsin) sku = mAsin[1].toUpperCase();
    if (parts.length > 0 && !['dp', 'gp'].includes(parts[0])) {
      const raw = parts[0].replace(/-/g, ' ').trim();
      if (raw.length > 3) {
        title = raw.replace(/\b\w/g, l => l.toUpperCase());
        const w = title.split(' ');
        if (w.length) brand = w[0];
      }
    }
  } else if (platform === 'Flipkart') {
    const mSku = path.match(/\/p\/([a-zA-Z0-9]+)/i);
    if (mSku) sku = mSku[1].toUpperCase();
    if (parts.length > 0 && parts[0] !== 'p') {
      const raw = parts[0].replace(/-/g, ' ').trim();
      if (raw.length > 3) {
        title = raw.replace(/\b\w/g, l => l.toUpperCase());
        const w = title.split(' ');
        if (w.length) brand = w[0];
      }
    }
  }
  return { platform, title, brand, sku };
}

(async () => {
  let browser = null;
  try {
    browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--window-position=-10000,-10000',
        '--window-size=1280,800',
        '--disable-gpu'
      ]
    });

    const page = await browser.newPage();
    await configurePage(page);

    // 1. EXTRACT MAIN PRODUCT & LIVE REVIEWS FROM PRODUCT DOM
    const slugMeta = parseUrlSlug(targetUrl);
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout: 16000 });
    await new Promise(r => setTimeout(r, 600));

    await page.evaluate(() => { window.scrollBy(0, 1800); });
    await new Promise(r => setTimeout(r, 600));

    const pageData = await page.evaluate((platform) => {
      function reviewRating(value) {
        const match = String(value || '').match(/([1-5](?:\.0)?)/);
        return match ? Number(match[1]) : null;
      }

      function reviewTimestamp(value) {
        const raw = String(value || '').replace(/^.*?\bon\s+/i, '').trim();
        const parsed = Date.parse(raw);
        return Number.isNaN(parsed) ? null : new Date(parsed).toISOString();
      }
      function getFirstText(selectors) {
        for (const s of selectors) {
          const el = document.querySelector(s);
          if (el) {
            const txt = el.getAttribute('content') || el.getAttribute('value') || el.innerText;
            if (txt && txt.trim()) return txt.trim();
          }
        }
        return null;
      }

      function getFirstAttr(selectors, attr) {
        for (const s of selectors) {
          const el = document.querySelector(s);
          if (el && el.getAttribute(attr)) return el.getAttribute(attr).trim();
        }
        return null;
      }

      let title = getFirstText(['span#productTitle', '#productTitle', 'h1.VU-ZEz', 'span.B_NuCI', 'h1.pdp-name', 'h1.pd-title', 'h1._6ERy96', 'h1']);
      let price = getFirstText(['.a-price .a-offscreen', 'span.a-price-whole', '.Nx9bqj', '._30jeq3', '._16Jk6d', 'div._25bWKC', '.pdp-price strong', '.pdp-price', '.new-price', '.amount']);
      let rating = getFirstText(['#acrPopover .a-icon-alt', 'i.a-icon-star span', '.XQDdHH', '._3LWZlK', '.rating', '.index-overallRating']);
      let reviewCount = getFirstText(['#acrCustomerReviewText', '.Wphh3N', '._2_R_DZ span', '.index-ratingsCount']);
      let image = getFirstAttr(['#landingImage', '#imgBlkFront', 'img._53J4C-', 'img._396cs4', 'img.pdp-image', 'img._2r_T1I', 'img'], 'src');
      let brand = getFirstText(['#bylineInfo', 'a#bylineInfo', '.gUuXy-', '._2Wk1rD', '.pdp-title', '.mBrand']);
      const reviewUrlEl = document.querySelector('#reviews-medley-footer a[href*="/product-reviews/"], a[data-hook="see-all-reviews-link-footer"], a[href*="/product-reviews/"]');
      const reviewUrl = reviewUrlEl ? reviewUrlEl.href : null;

      const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
      for (const s of scripts) {
        try {
          const data = JSON.parse(s.innerText);
          const items = Array.isArray(data) ? data : [data];
          for (const item of items) {
            if (!title && item.name) title = item.name;
            if (!brand && item.brand) brand = typeof item.brand === 'object' ? item.brand.name : item.brand;
            if (!price) {
              const p = item.price || (item.offers ? (item.offers.price || item.offers.lowPrice) : null);
              if (p) price = String(p);
            }
          }
        } catch (e) {}
      }

      const reviews = [];
      const reviewCards = Array.from(document.querySelectorAll('[data-hook="review"], .review, .col.EPCmJX, ._16PBlm, div.cPHRSc, ._27M-vq, .review-card, .user-review, ._2kHMtA, .RcHM54, #cm-cr-dp-review-list .a-section'));
      for (const card of reviewCards) {
        const bodyEl = card.querySelector('[data-hook="review-body"] span, [data-hook="review-body"], .ZmyHeo, .t-ZTKy div, .qwWvyD, .review-text, ._2-N2V1, .review-text-content span');
        const starEl = card.querySelector('[data-hook="review-star-rating"] span, [data-hook="cmps-review-star-rating"] span, i.review-rating span, .XQDdHH, ._3LWZlK, .rating');
        const dateEl = card.querySelector('[data-hook="review-date"], .MztJPv, ._2sc7ZR, .review-date');
        if (bodyEl && bodyEl.innerText.trim().length > 3) {
          reviews.push({
            text: bodyEl.innerText.trim(),
            rating: reviewRating(starEl ? starEl.innerText : null),
            date: dateEl ? dateEl.innerText.trim() : null,
            timestamp: reviewTimestamp(dateEl ? dateEl.innerText : null)
          });
        }
      }

      return { title, price, image, brand, rating, reviewCount, reviewUrl, reviews };
    }, slugMeta.platform);

    // Amazon usually exposes only a few cards on the product page. Read its public
    // review listing pages as well, so analysis receives a current, bounded sample.
    // No CAPTCHA/login bypass is attempted; an unavailable page simply yields no rows.
    const reviewCollection = {
      collected_count: pageData.reviews.length,
      source: slugMeta.platform === 'Amazon' ? 'product page and public Amazon review-listing pages' : 'product page DOM',
      attempted_pages: 0,
      inaccessible_pages: 0,
      issues: []
    };
    if (mode !== 'source' && slugMeta.platform === 'Amazon' && slugMeta.sku) {
      const seenReviewText = new Set(pageData.reviews.map(r => clean(r.text).toLowerCase()));
      const reviewBaseUrl = pageData.reviewUrl || `${new URL(targetUrl).origin}/product-reviews/${slugMeta.sku}/ref=cm_cr_arp_d_viewopt_sr`;
      for (let pageNumber = 1; pageNumber <= 3; pageNumber++) {
        let reviewPage = null;
        try {
          reviewPage = await browser.newPage();
          await configurePage(reviewPage);
          reviewCollection.attempted_pages++;
          const reviewUrl = `${reviewBaseUrl.split('?')[0]}?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=${pageNumber}`;
          const response = await reviewPage.goto(reviewUrl, { waitUntil: 'domcontentloaded', timeout: 12000, referer: targetUrl });
          if (response && response.status() >= 400) {
            reviewCollection.inaccessible_pages++;
            reviewCollection.issues.push(`Amazon review page ${pageNumber} returned HTTP ${response.status()}`);
            continue;
          }
          await reviewPage.waitForSelector('[data-hook="review"]', { timeout: 3500 }).catch(() => null);
          const extracted = await reviewPage.evaluate(() => {
            const ratingOf = value => {
              const match = String(value || '').match(/([1-5](?:\.0)?)/);
              return match ? Number(match[1]) : null;
            };
            const timestampOf = value => {
              const raw = String(value || '').replace(/^.*?\bon\s+/i, '').trim();
              const parsed = Date.parse(raw);
              return Number.isNaN(parsed) ? null : new Date(parsed).toISOString();
            };
            const reviews = Array.from(document.querySelectorAll('[data-hook="review"]')).map(card => {
              const body = card.querySelector('[data-hook="review-body"]');
              const stars = card.querySelector('[data-hook="review-star-rating"] span, [data-hook="cmps-review-star-rating"] span');
              const date = card.querySelector('[data-hook="review-date"]');
              const verified = card.querySelector('[data-hook="avp-badge"]');
              return body && body.innerText.trim().length > 3 ? {
                text: body.innerText.trim(), rating: ratingOf(stars && stars.innerText),
                date: date && date.innerText.trim(), timestamp: timestampOf(date && date.innerText),
                verified_purchase: Boolean(verified)
              } : null;
            }).filter(Boolean);
            const text = document.body ? document.body.innerText.toLowerCase() : '';
            return {
              reviews,
              blocked: /captcha|robot check|enter the characters you see below/.test(text),
              requires_login: /sign in or create account|enter mobile number or email/.test(text)
            };
          });
          if (extracted.requires_login) {
            reviewCollection.inaccessible_pages++;
            reviewCollection.issues.push(`Amazon requires an authenticated review-data source for review page ${pageNumber}.`);
          }
          if (extracted.blocked) {
            reviewCollection.inaccessible_pages++;
            reviewCollection.issues.push(`Amazon did not expose review page ${pageNumber} to this request.`);
          }
          for (const review of extracted.reviews) {
            const key = clean(review.text).toLowerCase();
            if (!seenReviewText.has(key)) {
              seenReviewText.add(key);
              pageData.reviews.push(review);
            }
          }
        } catch (e) {
          // Retain all reviews collected from earlier pages.
          reviewCollection.inaccessible_pages++;
          reviewCollection.issues.push(`Amazon review page ${pageNumber} could not be fetched.`);
        } finally {
          if (reviewPage) await reviewPage.close().catch(() => {});
        }
      }
    }
    reviewCollection.collected_count = pageData.reviews.length;

    // Fast path: return first-party listing details immediately. Expensive review
    // pagination and cross-store matching run in a later enrichment request.
    if (mode === 'source') {
      const finalTitle = clean(pageData.title) || slugMeta.title;
      let finalBrand = clean(pageData.brand) || slugMeta.brand;
      if (finalBrand) finalBrand = finalBrand.replace(/^(?:Visit the|Brand:?)\s*/i, '').replace(/\s+Store$/i, '').trim();
      await browser.close();
      browser = null;
      console.log(JSON.stringify({
        platform: slugMeta.platform, title: finalTitle, url: targetUrl,
        price: parseNumber(pageData.price), currency: 'INR', brand: finalBrand,
        sku: slugMeta.sku, image_url: clean(pageData.image),
        rating: parseNumber(pageData.rating),
        review_count: parseNumber(pageData.reviewCount), reviews: pageData.reviews || [],
        review_collection: { ...reviewCollection, collected_at: new Date().toISOString() },
        competitor_matches: [], competitor_collection: { attempts: [], status: 'deferred' }
      }));
      return;
    }

    // 2. PARALLEL MULTI-ITEM EXTRACTION (STRICT STORE MATRIX ISOLATION)
    const fullTitle = pageData.title || slugMeta.title || "";
    const cleanWords = fullTitle.replace(/[^\w\s]/g, ' ').split(/\s+/).filter(w => w.length > 2);
    const searchKeyword = cleanWords.slice(0, 4).join(' ') || 'product';

    const isBeauty = /serum|lipstick|foundation|shampoo|cream|lotion|nykaa|kajal|eyeliner|skin care|face wash/i.test(fullTitle);
    const isFashion = /shoe|sneaker|running|sandals|apparel|shirt|jeans|dress|jacket|campus|nike|adidas|puma/i.test(fullTitle);

    let searchSites = [];
    if (slugMeta.platform === 'Amazon') {
      searchSites = [{ name: "Flipkart", url: `https://www.flipkart.com/search?q=${encodeURIComponent(searchKeyword)}` }];
      if (isBeauty) {
        searchSites.push({ name: "Nykaa", url: `https://www.nykaa.com/search/result/?q=${encodeURIComponent(searchKeyword)}` });
        searchSites.push({ name: "Myntra", url: `https://www.myntra.com/${encodeURIComponent(searchKeyword)}` });
      } else if (isFashion) {
        searchSites.push({ name: "Myntra", url: `https://www.myntra.com/${encodeURIComponent(searchKeyword)}` });
      }
    } else if (slugMeta.platform === 'Flipkart') {
      searchSites = [{ name: "Amazon", url: `https://www.amazon.in/s?k=${encodeURIComponent(searchKeyword)}` }];
      if (isBeauty) {
        searchSites.push({ name: "Nykaa", url: `https://www.nykaa.com/search/result/?q=${encodeURIComponent(searchKeyword)}` });
        searchSites.push({ name: "Myntra", url: `https://www.myntra.com/${encodeURIComponent(searchKeyword)}` });
      } else if (isFashion) {
        searchSites.push({ name: "Myntra", url: `https://www.myntra.com/${encodeURIComponent(searchKeyword)}` });
      }
    } else {
      searchSites = [
        { name: "Flipkart", url: `https://www.flipkart.com/search?q=${encodeURIComponent(searchKeyword)}` },
        { name: "Amazon", url: `https://www.amazon.in/s?k=${encodeURIComponent(searchKeyword)}` }
      ];
    }

    const rawMatches = [];
    const competitorCollection = { attempts: [] };

    await Promise.all(
      searchSites.map(async (site) => {
        let sPage = null;
        try {
          sPage = await browser.newPage();
          await configurePage(sPage);
          const response = await sPage.goto(site.url, { waitUntil: 'domcontentloaded', timeout: 12000 });
          if (response && response.status() >= 400) {
            throw new Error(`HTTP ${response.status()}`);
          }
          await new Promise(r => setTimeout(r, 600));

          const items = await sPage.evaluate((platformName) => {
            const list = [];
            const seenUrls = new Set();
            const baseHost = platformName === 'Flipkart' ? 'https://www.flipkart.com' : platformName === 'Myntra' ? 'https://www.myntra.com' : platformName === 'Nykaa' ? 'https://www.nykaa.com' : 'https://www.amazon.in';

            // Select product cards directly or anchors
            const cardSelectors = [
              'div[data-id]', 'div.sl_Jkg', 'div.CGtC98', 'div._75WfSc', 'div._1sdW2b', 'div._2kHMtA', 'div._13oc-S', 'div.cPHRSc',
              'div[data-component-type="s-search-result"]', 'div.s-result-item', 'li.product-base', 'div.productWrapper',
              'a[href*="/p/"]', 'a[href*="/dp/"]', 'a[href*="pid="]', 'a[href*="/buy"]'
            ];

            const cards = Array.from(document.querySelectorAll(cardSelectors.join(',')));

            for (const c of cards) {
              const anchor = c.tagName === 'A' ? c : c.querySelector('a[href*="/p/"], a[href*="/dp/"], a[href*="pid="], a[href*="/buy"], a');
              if (!anchor) continue;

              const href = anchor.getAttribute('href');
              if (!href || (!href.includes('/p/') && !href.includes('/dp/') && !href.includes('pid=') && !href.includes('/buy'))) continue;

              const fullUrl = href.startsWith('/') ? baseHost + href : href;
              if (seenUrls.has(fullUrl)) continue;

              let titleEl = c.querySelector('h2, .WpEfc2, .s-line-clamp-2, .product-title, .pdp-name, ._2Wk1rD, span[title], a[title]');
              let titleTxt = (titleEl ? (titleEl.getAttribute('title') || titleEl.innerText) : (anchor.innerText || anchor.getAttribute('title') || '')).trim();
              if (!titleTxt || titleTxt.length < 3 || titleTxt.includes('Add to Compare')) continue;

              let pEl = c.querySelector('.product-discountedPrice, .product-price, .Nx9bqj, ._30jeq3, ._16Jk6d, div._25bWKC, .a-price .a-offscreen, span.a-price-whole, .amount, .new-price');
              let rEl = c.querySelector('.product-ratingsContainer, .XQDdHH, ._3LWZlK, #acrPopover .a-icon-alt, i.a-icon-star span, .rating');
              let imgEl = c.querySelector('img');

              seenUrls.add(fullUrl);
              list.push({
                platform: platformName,
                title: titleTxt.split('\n')[0],
                url: fullUrl,
                price: pEl ? (pEl.getAttribute('content') || pEl.innerText) : null,
                rating: rEl ? rEl.innerText : null,
                image_url: imgEl ? (imgEl.getAttribute('src') || imgEl.getAttribute('data-src')) : null
              });

              if (list.length >= 6) break;
            }

            return list;
          }, site.name);

          await sPage.close();
          competitorCollection.attempts.push({ platform: site.name, result_count: items.length, status: items.length ? 'results collected' : 'no product-card links exposed' });
          if (items && items.length) {
            rawMatches.push(...items.slice(0, 5));
          }
        } catch (e) {
          competitorCollection.attempts.push({ platform: site.name, result_count: 0, status: `unavailable: ${e.message || 'request failed'}` });
          if (sPage) await sPage.close().catch(() => {});
        }
      })
    );

    // Deep Candidate Page Navigation for the FIRST 5 REAL SINGLE PRODUCTS
    const competitorMatches = await Promise.all(
      rawMatches.slice(0, 5).map(async (cand) => {
        let candPage = null;
        try {
          candPage = await browser.newPage();
          await configurePage(candPage);
          await candPage.goto(cand.url, { waitUntil: 'domcontentloaded', timeout: 7000 });
          await new Promise(r => setTimeout(r, 400));

          const deepData = await candPage.evaluate(() => {
            function getFirstText(selectors) {
              for (const s of selectors) {
                const el = document.querySelector(s);
                if (el) {
                  const txt = el.getAttribute('content') || el.getAttribute('value') || el.innerText;
                  if (txt && txt.trim()) return txt.trim();
                }
              }
              return null;
            }

            function getFirstAttr(selectors, attr) {
              for (const s of selectors) {
                const el = document.querySelector(s);
                if (el && el.getAttribute(attr)) return el.getAttribute(attr).trim();
              }
              return null;
            }

            let p = getFirstText(['.a-price .a-offscreen', 'span.a-price-whole', '.Nx9bqj', '._30jeq3', '._16Jk6d', 'div._25bWKC', '.pdp-price strong', '.pdp-price', '.new-price', '.amount']);
            let r = getFirstText(['#acrPopover .a-icon-alt', 'i.a-icon-star span', '.XQDdHH', '._3LWZlK', '.rating', '.index-overallRating']);
            let img = getFirstAttr(['#landingImage', '#imgBlkFront', 'img._53J4C-', 'img._396cs4', 'img.pdp-image', 'img._2r_T1I', 'img'], 'src');
            let title = getFirstText(['span#productTitle', '#productTitle', 'h1.VU-ZEz', 'span.B_NuCI', 'h1.pdp-name', 'h1.pd-title', 'h1._6ERy96', 'h1']);
            let reviewCount = getFirstText(['#acrCustomerReviewText', '.Wphh3N', '._2_R_DZ span', '.index-ratingsCount']);

            // Product pages frequently place the reliable values in Schema.org JSON-LD.
            for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
              try {
                const parsed = JSON.parse(script.textContent || '{}');
                const nodes = Array.isArray(parsed) ? parsed : (parsed['@graph'] || [parsed]);
                for (const node of nodes) {
                  if (!node || (node['@type'] !== 'Product' && !String(node['@type'] || '').includes('Product'))) continue;
                  title = title || node.name || null;
                  const offers = Array.isArray(node.offers) ? node.offers[0] : node.offers;
                  p = p || (offers && (offers.price || offers.lowPrice) != null ? String(offers.price || offers.lowPrice) : null);
                  if (node.image) img = Array.isArray(node.image) ? node.image[0] : node.image;
                  const aggregate = node.aggregateRating || {};
                  r = r || (aggregate.ratingValue != null ? String(aggregate.ratingValue) : null);
                  reviewCount = reviewCount || (aggregate.reviewCount || aggregate.ratingCount ? String(aggregate.reviewCount || aggregate.ratingCount) : null);
                }
              } catch (e) {}
            }

            const reviews = [];
            for (const card of document.querySelectorAll('[data-hook="review"], .review-card, .col.EPCmJX, ._16PBlm, div.cPHRSc')) {
              const body = card.querySelector('[data-hook="review-body"], [data-hook="review-body"] span, .review-text, .ZmyHeo, .t-ZTKy div');
              const stars = card.querySelector('[data-hook="review-star-rating"] span, [data-hook="cmps-review-star-rating"] span, .XQDdHH, ._3LWZlK, .rating');
              if (body && body.innerText.trim().length > 3) {
                const match = String(stars?.innerText || '').match(/([1-5](?:\.0)?)/);
                reviews.push({ text: body.innerText.trim(), rating: match ? Number(match[1]) : null });
              }
              if (reviews.length >= 5) break;
            }

            return { price: p, rating: r, review_count: reviewCount, image_url: img, title: title, reviews };
          });

          await candPage.close();

          return {
            ...cand,
            title: deepData.title ? deepData.title.split('\n')[0] : cand.title,
            price: deepData.price || cand.price,
            rating: deepData.rating || cand.rating,
            review_count: deepData.review_count,
            reviews: deepData.reviews || [],
            image_url: clean(deepData.image_url) || cand.image_url
          };
        } catch (e) {
          if (candPage) await candPage.close().catch(() => {});
          return cand;
        }
      })
    );

    await browser.close();

    const finalTitle = clean(pageData.title) || slugMeta.title;
    let finalBrand = clean(pageData.brand) || slugMeta.brand;
    if (finalBrand) {
      finalBrand = finalBrand.replace(/^(?:Visit the|Brand:?)\s*/i, '').replace(/\s+Store$/i, '').trim();
    }
    const finalPrice = parseNumber(pageData.price);
    const finalRating = parseNumber(pageData.rating);
    const finalReviewCount = parseNumber(pageData.reviewCount);

    const parsedMatches = competitorMatches.map(m => ({
      platform: m.platform,
      title: m.title,
      url: m.url,
      price: parseNumber(m.price),
      rating: parseNumber(m.rating),
      review_count: parseNumber(m.review_count),
      review_sample_count: Array.isArray(m.reviews) ? m.reviews.length : 0,
      reviews: Array.isArray(m.reviews) ? m.reviews : [],
      image_url: clean(m.image_url)
    }));

    const result = {
      platform: slugMeta.platform,
      title: finalTitle,
      url: targetUrl,
      price: finalPrice,
      currency: "INR",
      brand: finalBrand,
      sku: slugMeta.sku,
      image_url: clean(pageData.image),
      rating: finalRating,
      review_count: finalReviewCount ? Math.round(finalReviewCount) : null,
      reviews: pageData.reviews || [],
      review_collection: { ...reviewCollection, collected_at: new Date().toISOString() },
      real_price_history: null,
      competitor_matches: parsedMatches,
      competitor_collection: competitorCollection
    };

    console.log(JSON.stringify(result));

  } catch (err) {
    if (browser) await browser.close();
    console.error(JSON.stringify({ error: err.message }));
    process.exit(1);
  }
})();
