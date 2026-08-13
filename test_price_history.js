const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36');
  
  console.log("Navigating to pricehistoryapp.com...");
  await page.goto('https://pricehistoryapp.com/', { waitUntil: 'domcontentloaded' });
  await new Promise(r => setTimeout(r, 1500));
  
  const inputEl = await page.$('input[placeholder*="Paste"], input[type="text"], input');
  if (inputEl) {
    await inputEl.type('https://www.amazon.in/dp/B079TTTWRX');
    await page.keyboard.press('Enter');
    await new Promise(r => setTimeout(r, 4000));
    console.log("Current Page URL:", page.url());
    
    const textData = await page.evaluate(() => document.body.innerText);
    console.log("Extracted Text Preview:", textData.slice(0, 500));
  } else {
    console.log("Search input not found");
  }
  await browser.close();
})();
