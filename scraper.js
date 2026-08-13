const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

let mode = process.argv[2];
let targetUrl = process.argv[3];

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
        '--disable-blink-features=AutomationControlled',
        '--window-position=-10000,-10000',
        '--window-size=1280,800',
        '--disable-gpu'
      ]
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36');

    // 1. EXTRACT MAIN PRODUCT & LIVE REVIEWS FROM PRODUCT DOM
    const slugMeta = parseUrlSlug(targetUrl);
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout: 16000 });
    await new Promise(r => setTimeout(r, 600));

    await page.evaluate(() => { window.scrollBy(0, 1800); });
    await new Promise(r => setTimeout(r, 600));

    const pageData = await page.evaluate((platform) => {
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
            rating: starEl ? starEl.innerText.trim() : null,
            date: dateEl ? dateEl.innerText.trim() : null
          });
        }
      }

      return { title, price, image, brand, rating, reviewCount, reviews };
    }, slugMeta.platform);

    // 2. PARALLEL MULTI-ITEM EXTRACTION (STRICT STORE MATRIX ISOLATION)
    const fullTitle = pageData.title || slugMeta.title || "";
    const cleanWords = fullTitle.replace(/[^\w\s]/g, ' ').split(/\s+/).filter(w => w.length > 2);
    const searchKeyword = cleanWords.slice(0, 4).join(' ') || 'product';

    const isBeauty = /serum|lipstick|foundation|shampoo|cream|lotion|nykaa|kajal|eyeliner|skin care|face wash/i.test(fullTitle);

    let searchSites = [];
    if (slugMeta.platform === 'Amazon') {
      searchSites = [{ name: "Flipkart", url: `https://www.flipkart.com/search?q=${encodeURIComponent(searchKeyword)}` }];
      if (isBeauty) {
        searchSites.push({ name: "Nykaa", url: `https://www.nykaa.com/search/result/?q=${encodeURIComponent(searchKeyword)}` });
        searchSites.push({ name: "Myntra", url: `https://www.myntra.com/${encodeURIComponent(searchKeyword)}` });
      }
    } else if (slugMeta.platform === 'Flipkart') {
      searchSites = [{ name: "Amazon", url: `https://www.amazon.in/s?k=${encodeURIComponent(searchKeyword)}` }];
      if (isBeauty) {
        searchSites.push({ name: "Nykaa", url: `https://www.nykaa.com/search/result/?q=${encodeURIComponent(searchKeyword)}` });
        searchSites.push({ name: "Myntra", url: `https://www.myntra.com/${encodeURIComponent(searchKeyword)}` });
      }
    } else {
      searchSites = [
        { name: "Flipkart", url: `https://www.flipkart.com/search?q=${encodeURIComponent(searchKeyword)}` },
        { name: "Amazon", url: `https://www.amazon.in/s?k=${encodeURIComponent(searchKeyword)}` }
      ];
    }

    const rawMatches = [];

    await Promise.all(
      searchSites.map(async (site) => {
        let sPage = null;
        try {
          sPage = await browser.newPage();
          await sPage.goto(site.url, { waitUntil: 'domcontentloaded', timeout: 9000 });
          await new Promise(r => setTimeout(r, 600));

          const items = await sPage.evaluate((platformName) => {
            const list = [];
            const seenUrls = new Set();
            // Target ALL product card anchors directly pointing to single product items (/p/ or /dp/)
            const anchors = Array.from(document.querySelectorAll('a[href*="/p/"], a[href*="/dp/"]'));
            const baseHost = platformName === 'Flipkart' ? 'https://www.flipkart.com' : platformName === 'Myntra' ? 'https://www.myntra.com' : platformName === 'Nykaa' ? 'https://www.nykaa.com' : 'https://www.amazon.in';

            for (const a of anchors) {
              const href = a.getAttribute('href');
              if (!href || (!href.includes('/p/') && !href.includes('/dp/'))) continue;
              const text = (a.innerText || a.getAttribute('title') || '').trim();
              if (text.length < 3 || text.includes('Add to Compare')) continue;

              const container = a.closest('div.sl_Jkg, div.CGtC98, div._75WfSc, div._1sdW2b, div._2kHMtA, div._13oc-S, div.cPHRSc, div.tT52d, div[data-id], div._2kHMtA, div._1AtVbE') || a.parentElement;
              let pEl = container ? container.querySelector('.product-discountedPrice, .product-price, .Nx9bqj, ._30jeq3, ._16Jk6d, div._25bWKC, .a-price .a-offscreen, span.a-price-whole, .amount, .new-price') : null;
              let rEl = container ? container.querySelector('.product-ratingsContainer, .XQDdHH, ._3LWZlK, #acrPopover .a-icon-alt, i.a-icon-star span, .rating') : null;
              let imgEl = container ? container.querySelector('img') : null;

              const fullUrl = href.startsWith('/') ? baseHost + href : href;
              if (seenUrls.has(fullUrl)) continue;
              seenUrls.add(fullUrl);

              list.push({
                platform: platformName,
                title: text.split('\n')[0],
                url: fullUrl,
                price: pEl ? (pEl.getAttribute('content') || pEl.innerText) : null,
                rating: rEl ? rEl.innerText : null,
                image_url: imgEl ? (imgEl.getAttribute('src') || imgEl.getAttribute('data-src')) : null
              });

              if (list.length >= 5) break;
            }

            return list;
          }, site.name);

          await sPage.close();
          if (items && items.length) {
            rawMatches.push(...items.slice(0, 5));
          }
        } catch (e) {
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

            return { price: p, rating: r, image_url: img, title: title };
          });

          await candPage.close();

          return {
            ...cand,
            title: deepData.title ? deepData.title.split('\n')[0] : cand.title,
            price: cand.price || deepData.price,
            rating: cand.rating || deepData.rating,
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
      real_price_history: null,
      competitor_matches: parsedMatches
    };

    console.log(JSON.stringify(result));

  } catch (err) {
    if (browser) await browser.close();
    console.error(JSON.stringify({ error: err.message }));
    process.exit(1);
  }
})();
