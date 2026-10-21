/*
 * One place that starts Chrome for every scraper, always without a visible window.
 *
 * "shell" is Chrome's headless-shell build: it has no user interface at all, so no window can open, flash or
 * appear in the taskbar. If that build is not installed it falls back to Chrome's regular headless mode.
 * PUPPETEER_HEADLESS_MODE=true forces the regular headless mode; "shell" (default) prefers the shell build.
 *
 * This only keeps the browser out of sight. It does nothing to hide the browser from a website and is not
 * used to get past a CAPTCHA or bot check (pages that show one are reported and left alone).
 */
const QUIET_ARGS = [
  '--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--disable-extensions', '--mute-audio', '--hide-scrollbars', '--window-size=1280,900',
];

async function launchQuiet(puppeteer, extraArgs = []) {
  const mode = (process.env.PUPPETEER_HEADLESS_MODE || 'shell').toLowerCase();
  const args = [...QUIET_ARGS, ...extraArgs];
  if (mode === 'true') return puppeteer.launch({ headless: true, args });
  try {
    return await puppeteer.launch({ headless: 'shell', args });
  } catch (e) {
    return puppeteer.launch({ headless: true, args });         // the shell build is not installed
  }
}

module.exports = { launchQuiet, QUIET_ARGS };
