const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36');
  
  console.log("Navigating to Flipkart search...");
  await page.goto('https://www.flipkart.com/search?q=Purepet+Adult+10kg', { waitUntil: 'domcontentloaded', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2000));
  
  const products = await page.evaluate(() => {
    const items = [];
    const anchors = Array.from(document.querySelectorAll('a[href*="/p/"]'));
    for (const a of anchors) {
      const href = a.getAttribute('href');
      const title = a.getAttribute('title') || a.innerText.trim();
      const priceEl = a.querySelector('.Nx9bqj, ._30jeq3') || (a.parentElement ? a.parentElement.querySelector('.Nx9bqj, ._30jeq3') : null);
      const price = priceEl ? priceEl.innerText.trim() : null;
      if (href && title && title.length > 5) {
        items.push({
          title: title.split('\n')[0],
          price: price,
          url: href.startsWith('/') ? 'https://www.flipkart.com' + href : href
        });
      }
    }
    return items;
  });

  console.log("Found Products:", products.length);
  console.log(JSON.stringify(products.slice(0, 3), null, 2));
  await browser.close();
})();
