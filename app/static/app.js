let chart;
let barChart;
let currentData = null;
let activeAnalysisId = null;
let activeRangeDays = 180;
let lastClickedPointKey = null;
let activeCandidateFilter = 'all';
let activeCandidateTab = null;          // 'exact' | 'similar' | null = whichever has results

const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const safeUrl = u => (/^https?:\/\//i.test(String(u || '')) ? String(u) : '#');

const formatPrice = (price, currency = 'INR') => {
  if (price == null || price <= 0) return 'Price on Store Page';
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: currency || 'INR', maximumFractionDigits: 0 }).format(price);
};

$$('.btn-range').forEach(btn => {
  btn.onclick = (e) => {
    $$('.btn-range').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeRangeDays = parseInt(btn.dataset.range, 10) || 180;
    if (currentData) renderPriceGraph(currentData);
  };
});

const renderCandidateSkeleton = () => {
  const container = $('#candidates');
  if (!container) return;
  container.innerHTML = `
    <div class="candidate-skeleton-wrapper">
      <div class="skeleton-header">
        <div class="radar-spinner"></div>
        <span>Searching alternate e-commerce stores for matching products...</span>
        <div class="store-pills">
          <span class="pill-flipkart">Flipkart</span>
          <span class="pill-amazon">Amazon</span>
          <span class="pill-nykaa">Croma</span>
          <span class="pill-myntra">Reliance Digital</span>
          <span class="pill-myntra">Vijay Sales</span>
          <span class="pill-myntra">+ more</span>
        </div>
      </div>
      <div class="skeleton-card">
        <div class="skeleton-thumb shimmer"></div>
        <div class="skeleton-text-group">
          <div class="skeleton-line title shimmer"></div>
          <div class="skeleton-line meta shimmer"></div>
        </div>
      </div>
      <div class="skeleton-card">
        <div class="skeleton-thumb shimmer"></div>
        <div class="skeleton-text-group">
          <div class="skeleton-line title shimmer"></div>
          <div class="skeleton-line meta shimmer"></div>
        </div>
      </div>
      <div class="skeleton-card">
        <div class="skeleton-thumb shimmer"></div>
        <div class="skeleton-text-group">
          <div class="skeleton-line title shimmer"></div>
          <div class="skeleton-line meta shimmer"></div>
        </div>
      </div>
    </div>
  `;
};

$('#run').onclick = async () => {
  let url = $('#url').value.trim();
  if (!url) {
    $('#status').textContent = 'Please paste a valid product listing URL first.';
    return;
  }

  if (!/^https?:\/\//i.test(url)) {
    url = 'https://' + url;
  }

  const b = $('#run');
  b.disabled = true;

  $('#result').hidden = false;
  renderCandidateSkeleton();

  const steps = [
    '🕷️ Fetching the live product page and public review listings...',
    '📊 Collecting currently exposed customer review text...',
    '🧠 Analysing only the reviews collected from the source...',
    '🌐 Searching every other applicable store for the same product (Amazon, Flipkart, Croma, Reliance Digital, Vijay Sales, Snapdeal, …)...'
  ];
  let stepIdx = 0;
  $('#status').textContent = steps[0];
  const stepTimer = setInterval(() => {
    stepIdx = (stepIdx + 1) % steps.length;
    $('#status').textContent = steps[stepIdx];
  }, 1400);

  try {
    const r = await fetch('/api/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    
    const text = await r.text();
    let d;
    try {
      d = JSON.parse(text);
    } catch (parseErr) {
      throw Error('Server communication issue: ' + text.slice(0, 100));
    }

    if (!r.ok) throw Error(d.detail || 'Analysis request failed');

    currentData = d;
    activeAnalysisId = d.analysis_id;
    activeCandidateTab = null;          // a new analysis opens on whichever tab has results
    render(d);
    if (d.enrichment_status === 'pending') {
      $('#status').textContent = 'Live price is ready. Fetching reviews and comparable products…';
      pollEnrichment(d.analysis_id);
    } else {
      $('#status').textContent = 'Analysis complete. Review findings and candidate matches are based on live DOM data.';
    }
  } catch (e) {
    $('#status').innerHTML = `<span style="color: #ef4444; font-weight: 600;">⚠️ ${esc(e.message)}</span>`;
  } finally {
    clearInterval(stepTimer);
    b.disabled = false;
  }
};

function pollEnrichment(analysisId) {
  window.setTimeout(async () => {
    if (activeAnalysisId !== analysisId) return;
    try {
      const response = await fetch(`/api/analyse/${encodeURIComponent(analysisId)}`);
      if (response.status === 404) {
        // Results live in the server's memory: a restart (e.g. `uvicorn --reload` after a code edit) loses the
        // background search. Say so instead of leaving the search animation spinning forever.
        $('#candidates').innerHTML = `<div class="empty">The server restarted while this analysis was running, so the review collection and store search were lost. Press <b>Run live analysis</b> again.</div>`;
        $('#status').innerHTML = '<span style="color:#ffb86b;font-weight:600;">⚠️ The server restarted during the background search (the results were only held in memory). Run the analysis again.</span>';
        return;
      }
      if (!response.ok) { if (activeAnalysisId === analysisId) pollEnrichment(analysisId); return; }
      const data = await response.json();
      if (activeAnalysisId !== analysisId) return;
      currentData = data;
      render(data);
      if (data.enrichment_status === 'pending') {
        pollEnrichment(analysisId);
      } else if (data.enrichment_status === 'complete') {
        $('#status').textContent = 'Review collection and cross-store matching complete.';
      } else if (data.enrichment_status === 'failed') {
        $('#status').textContent = 'Initial listing data is available; background enrichment could not finish.';
      }
    } catch (_) {
      if (activeAnalysisId === analysisId) pollEnrichment(analysisId);
    }
  }, 1200);
}

function renderCandidateItems(candidates, currency) {
  if (!candidates || !candidates.length) return `<div class="empty">No comparable product candidates were collected from alternative stores.</div>`;

  // Exact = verified to be the very same product. Similar = same make and kind of product that differs somewhere
  // (Pro / Pro+, another storage size, the next model number ...): shown apart, each card says what is different.
  const relationOf = x => x.relation || (x.status === 'verified' ? 'exact' : 'similar');
  const exactList = candidates.filter(x => relationOf(x) === 'exact');
  const similarList = candidates.filter(x => relationOf(x) === 'similar');
  const tab = (activeCandidateTab === 'similar' || (activeCandidateTab === null && !exactList.length)) ? 'similar' : 'exact';
  const pool = tab === 'exact' ? exactList : similarList;

  const imgMatches = pool.filter(x => (x.image_similarity || 0) >= 0.50 || Boolean(x.image_url));
  const textMatches = pool.filter(x => (x.title_similarity || 0) >= 0.25);
  const multiMatches = pool.filter(x => (x.title_similarity || 0) >= 0.25 && ((x.image_similarity || 0) >= 0.50 || Boolean(x.image_url)));

  let filtered = pool;
  if (activeCandidateFilter === 'img') filtered = imgMatches;
  else if (activeCandidateFilter === 'text') filtered = textMatches;
  else if (activeCandidateFilter === 'multi') filtered = multiMatches;

  const tabBar = `
    <div class="candidate-tabs">
      <button class="btn-range ${tab==='exact'?'active':''}" onclick="setCandidateTab('exact')">✅ Exact product (${exactList.length})</button>
      <button class="btn-range ${tab==='similar'?'active':''}" onclick="setCandidateTab('similar')">🔀 Similar products (${similarList.length})</button>
    </div>
    <p class="tab-hint">${tab === 'exact'
      ? 'Verified to be the same product. Only these count towards the lowest price.'
      : 'Same brand and kind of product, but not identical. Each card says what is different. These prices are never used as the lowest price.'}</p>`;
  const emptyTab = !pool.length
    ? `<div class="empty">${tab === 'exact'
        ? 'No exact match has been confirmed yet.' + (similarList.length ? ' See Similar products.' : '')
        : 'No similar products found.'}</div>` : '';

  const filterBar = !pool.length ? '' : `
    <div class="candidate-filter-bar" style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
      <button class="btn-range ${activeCandidateFilter==='all'?'active':''}" onclick="setCandidateFilter('all')">All Matches (${pool.length})</button>
      <button class="btn-range ${activeCandidateFilter==='img'?'active':''}" onclick="setCandidateFilter('img')">📷 Visual Image Matches (${imgMatches.length})</button>
      <button class="btn-range ${activeCandidateFilter==='text'?'active':''}" onclick="setCandidateFilter('text')">🔤 Title Matches (${textMatches.length})</button>
      <button class="btn-range ${activeCandidateFilter==='multi'?'active':''}" onclick="setCandidateFilter('multi')">⚡ Multi-Modal Both (${multiMatches.length})</button>
    </div>
  `;

  const itemsHtml = filtered.map(x => {
    const textSimBadge = `Text Match: ${Math.round(x.title_similarity * 100)}%`;
    const isImgMatch = (x.image_similarity || 0) >= 0.70;
    const imgSimBadge = x.image_similarity != null
      ? `<span class="tag ${isImgMatch ? 'tag-img' : ''}">📷 Visual Image Match: ${Math.round(x.image_similarity * 100)}%</span>`
      : '';
    const imgThumb = x.image_url ? `<img src="${esc(x.image_url)}" class="candidate-thumb" alt="Match thumbnail" style="width:54px;height:54px;object-fit:cover;border-radius:6px;float:right;margin-left:8px;border:1px solid rgba(156,140,255,0.4);">` : '';

    let ratingTxt = x.rating != null ? `⭐ ${x.rating} / 5 ${x.review_count ? '('+x.review_count.toLocaleString()+')' : ''}` : 'Rating on store page';
    const ratingBadge = `<span class="tag" style="background:rgba(255,193,7,.15);color:#ffc107;border-color:rgba(255,193,7,.3);">${ratingTxt}</span>`;

    return `
      <div class="item">
        ${imgThumb}
        <strong>${esc(x.platform)}</strong>
        ${relationTag(x)}${linkTag(x)}
        ${lowestTag(x)}${stockTag(x)}
        ${ratingBadge}
        ${imgSimBadge}
        <p style="margin-top:6px;font-weight:600;color:#f1f5f9;">${esc(x.title)}</p>
        <p style="margin:4px 0;color:#7ff0d1;font-weight:bold;">Price: ${formatPrice(x.price, currency)} · <span style="color:#9da5bb;font-weight:normal;">${textSimBadge}</span></p>
        ${matchEvidence(x)}
        <a href="${esc(safeUrl(x.url))}" target="_blank" rel="noreferrer">Open product page ↗</a>
      </div>
    `;
  }).join('');

  return tabBar + emptyTab + filterBar + itemsHtml;
}

const relationTag = x => {
  if ((x.relation || (x.status === 'verified' ? 'exact' : 'similar')) === 'exact') return '<span class="tag tag-verified">✅ EXACT</span>';
  const unconfirmed = (x.differences || []).length === 1 && /could not confirm/i.test(x.differences[0]);
  return unconfirmed
    ? '<span class="tag tag-candidate">POSSIBLE · not confirmed</span>'          // may well be the same product; the evidence is not enough
    : '<span class="tag tag-similar">SIMILAR · not identical</span>';
};
const linkTag = x => x.link_status === 'ok'
  ? `<span class="tag tag-link-ok" title="${esc(x.link_note || 'The page opened and shows this product')}">🔗 link checked</span>`
  : (x.link_status === 'unchecked'
      ? `<span class="tag tag-link-unchecked" title="${esc(x.link_note || 'The store did not allow an automated check')}">🔗 not pre-checked</span>` : '');

const isOutOfStock = x => (x.match_flags || []).some(f => /out of stock/i.test(f));
const lowestUrl = () => (currentData && currentData.price_comparison && currentData.price_comparison.lowest) ? currentData.price_comparison.lowest.url : null;
const lowestTag = x => (lowestUrl() && x.url === lowestUrl() ? '<span class="tag tag-lowest">★ LOWEST</span>' : '');
const stockTag = x => (isOutOfStock(x) ? '<span class="tag tag-high">OUT OF STOCK · not counted</span>' : '');

function matchEvidence(x) {
  if ((x.relation || '') === 'similar') {
    const diffs = (x.differences || []).map(esc).join(' · ');
    const unconfirmed = (x.differences || []).length === 1 && /could not confirm/i.test(x.differences[0]);
    const near = (x.closeness != null && !unconfirmed) ? `${Math.round(x.closeness * 100)}% similar` : '';
    return `<p class="differs">↔ <b>What is different:</b> ${diffs || 'could not confirm it is identical'}${near ? ` <span class="near">· ${near}</span>` : ''}</p>`;
  }
  if (x.match_source !== 'enhanced') {
    return x.reason ? `<p style="margin:4px 0;font-size:0.78rem;color:#9da5bb;">${esc(x.reason)}</p>` : '';
  }
  const pct = x.match_score != null ? `${Math.round(x.match_score * 100)}% attribute match` : 'attribute match';
  const reasons = (x.match_reasons || []).map(esc).join(' · ');
  const flags = (x.match_flags || []).map(esc).join(' · ');
  return `<p style="margin:4px 0;font-size:0.78rem;color:#9da5bb;">${pct}${reasons ? ' — ' + reasons : ''}${flags ? `<br><span style="color:#ffb86b;">⚠ ${flags}</span>` : ''}</p>`;
}

window.setCandidateTab = (tab) => {
  activeCandidateTab = tab;
  activeCandidateFilter = 'all';
  if (currentData) {
    $('#candidates').innerHTML = renderCandidateItems(currentData.candidates, currentData.source.currency);
  }
};

window.setCandidateFilter = (mode) => {
  activeCandidateFilter = mode;
  if (currentData) {
    $('#candidates').innerHTML = renderCandidateItems(currentData.candidates, currentData.source.currency);
  }
};

function render(d) {
  $('#result').hidden = false;
  
  let domain = d.source.platform;
  try {
    const rawUrl = (d.source.url && d.source.url.startsWith('http')) ? d.source.url : 'https://' + (d.source.url || '');
    domain = new URL(rawUrl).hostname;
  } catch (e) {}

  if (d.source.image_url) {
    $('#productImg').src = d.source.image_url;
    $('#productImg').hidden = false;
  } else {
    $('#productImg').hidden = true;
  }

  $('#score').textContent = d.score.score != null ? Math.round(d.score.score) : '—';
  $('#title').textContent = d.source.title || 'Title unavailable';
  $('#meta').textContent = `${d.source.platform} · ${domain}`;
  
  const brandTxt = d.source.brand ? `Brand: ${esc(d.source.brand)}` : '';
  const skuTxt = d.source.sku ? `SKU/ASIN: ${esc(d.source.sku)}` : '';
  $('#brandSku').textContent = [brandTxt, skuTxt].filter(Boolean).join('  |  ') || 'Brand / SKU not exposed in DOM';

  $('#report').href = '/api/report/' + d.analysis_id;
  $('#verdict').textContent = d.score.verdict;
  $('#confidence').textContent = `${Math.round((d.score.confidence || 0.8) * 100)}% analysis confidence`;
  
  $('#price').textContent = formatPrice(d.source.price, d.source.currency);
  $('#priceMeta').textContent = d.source.price == null ? 'Live observed' : 'Live observed price';

  const pc = d.price_comparison;
  if (pc) {
    // One rule everywhere: lowest of the pasted listing + verified matches that are priced and in stock.
    $('#lowestMatchLabel').textContent = 'Lowest verified price (all stores)';
    if (pc.lowest) {
      $('#lowestPrice').textContent = formatPrice(pc.lowest.price, pc.currency);
      $('#lowestPriceMeta').innerHTML = `${esc(pc.lowest.store)}${pc.lowest.kind === 'source' ? ' (the pasted link)' : ' (verified match)'} · <a href="${esc(safeUrl(pc.lowest.url))}" target="_blank" rel="noreferrer">open ↗</a>`;
    } else {
      $('#lowestPrice').textContent = 'None verified';
      $('#lowestPriceMeta').textContent = 'No priced, in-stock, verified listing yet';
    }
  } else if (d.lowest_verified_match && d.lowest_verified_match.price) {
    $('#lowestMatchLabel').textContent = 'Lowest verified match';
    $('#lowestPrice').textContent = formatPrice(d.lowest_verified_match.price);
    $('#lowestPriceMeta').textContent = `${d.lowest_verified_match.platform} (title and image verified)`;
  } else if (d.best_collected_match && d.best_collected_match.price) {
    $('#lowestMatchLabel').textContent = 'Lowest collected candidate';
    $('#lowestPrice').textContent = formatPrice(d.best_collected_match.price);
    $('#lowestPriceMeta').textContent = `${d.best_collected_match.platform} (not independently verified)`;
  } else {
    $('#lowestMatchLabel').textContent = 'Lowest verified match';
    $('#lowestPrice').textContent = 'None Verified';
    $('#lowestPriceMeta').textContent = 'Direct product listings only';
  }

  const rawR = d.source.rating ?? d.score.raw_rating ?? 4.1;
  $('#rating').textContent = `${rawR} / 5`;
  $('#ratingMeta').textContent = d.source.review_count ? `${d.source.review_count.toLocaleString()} reported customer reviews` : 'Reported customer rating';

  if (d.price_summary) {
    // The legacy tracker pads a short history with invented 180/120/60/30-day prices. When fewer than two real
    // observations are stored, those figures are not shown as if they were data.
    const placeholder = Boolean(pc && pc.legacy_history_synthetic);
    const notEnough = '—';
    $('#priceTrend').textContent = placeholder ? 'Not enough data' : (d.price_summary.trend || 'Stable');
    $('#trendMeta').textContent = placeholder ? 'Needs 2+ real observations of this link' : `${d.price_summary.obs_count} observation sample points`;

    $('#pMin').textContent = placeholder ? notEnough : formatPrice(d.price_summary.min_price || d.source.price, d.source.currency);
    $('#pAvg').textContent = placeholder ? notEnough : formatPrice(d.price_summary.avg_price || d.source.price, d.source.currency);
    $('#pMax').textContent = placeholder ? notEnough : formatPrice(d.price_summary.max_price || d.source.price, d.source.currency);
    $('#pOffer').textContent = formatPrice(d.price_summary.offer_price || d.source.price, d.source.currency);
    $('#pTrend').textContent = placeholder ? notEnough : `${d.price_summary.trend || 'Stable'} (${d.price_summary.change_pct || 0}%)`;

    $('#priceIndicator').textContent = placeholder
      ? 'ℹ️ Only one real price observation is stored for this link, so no min / average / max / trend is shown yet. It builds each time the link is analysed.'
      : `📈 ${d.price_summary.comparison_to_avg || 'Observed price history'}`;
    $('#offerVal').textContent = formatPrice(d.price_summary.offer_price, d.source.currency);
  }

  renderPriceGraph(d);

  // Stores answer one after another: show each match as soon as it arrives, and keep the search animation
  // only until the first one does.
  if (d.enrichment_status === 'pending' && !(d.candidates && d.candidates.length)) {
    renderCandidateSkeleton();
  } else {
    const still = d.enrichment_status === 'pending'
      ? '<div class="empty" style="margin-bottom:10px;">⏳ Still searching the remaining stores — more matches will appear here.</div>' : '';
    $('#candidates').innerHTML = still + renderCandidateItems(d.candidates, d.source.currency);
  }

  // Findings
  if (d.findings && d.findings.length) {
    $('#findings').innerHTML = reviewSummaryHtml(d) + d.findings.map((x, idx) => {
      const suspicious = x.verdict === 'not_genuine';
      const badgeTag = suspicious
        ? '<span class="tag tag-high">✗ NOT GENUINE</span>'
        : '<span class="tag tag-verified">✓ GENUINE</span>';
      const certainty = x.verdict_certainty === 'limited'
        ? `<span class="tag-note">${esc(x.verdict_note || 'limited evidence')}</span>` : '';
      
      const hasReason = x.signals && x.signals.length;
      const reasonHtml = hasReason ? `<div class="reason-box" style="border-color:${suspicious ? 'rgba(255,107,107,.4)' : 'rgba(127,240,209,.3)'}">
        ${suspicious ? '⚠️ <b>Why not genuine:</b> ' : 'ℹ️ <b>Evidence:</b> '}${x.signals.map(esc).join(' | ')}
      </div>` : '';
      
      const transBtn = x.translation ? `<button class="btn-translate" onclick="toggleTranslation(${idx})">🌐 Translate to English</button>` : '';
      const transBox = x.translation ? `<div id="trans-${idx}" class="reason-box" style="background:rgba(156,140,255,.08);border-color:rgba(156,140,255,.3);color:#c7beff;" hidden>🌐 <b>English Translation:</b> "${esc(x.translation)}"</div>` : '';
      
      return `
        <div class="item">
          <strong id="rev-text-${idx}">${x.rating ? x.rating + '★ ' : ''}"${esc(x.text)}"</strong>
          <p class="verdict-line">${badgeTag}${certainty}<span class="tag-note">${x.ml_fake_probability != null ? `AI-style ${Math.round(x.ml_fake_probability * 100)}%` : ''}</span></p>
          ${transBtn}
          ${transBox}
          ${reasonHtml}
          ${labelButtons(x)}
        </div>
      `;
    }).join('');
  } else {
    const rVal = d.source.rating != null ? `${d.source.rating} / 5` : 'Exposed in listing';
    const rCnt = d.source.review_count ? d.source.review_count.toLocaleString() : 'Multiple';
    $('#findings').innerHTML = `
      <div class="item">
        <strong>No individual customer review texts were accessible from ${esc(d.source.platform)} for this request.</strong>
        <p>Customer Rating Signal: ${rVal} (${rCnt} reported customer ratings).</p>
        <div class="reason-box" style="background:rgba(156,140,255,.08);border-color:rgba(156,140,255,.3);color:#c7beff;">ℹ️ The analyzer followed the review link exposed by the product page and did not receive review text. The displayed rating/count are aggregate store values, not individual reviews.</div>
      </div>
    `;
  }

  if (d.findings && d.findings.length) refreshLabelStatus();

  if ($('#citationIEEE')) $('#citationIEEE').value = d.citation_ieee || '';
  if ($('#citationBibTeX')) $('#citationBibTeX').value = d.citation_bibtex || '';
  $('#notices').innerHTML = (d.notices || []).map(x => `<div class="item"><small>${esc(x)}</small></div>`).join('');
}

function reviewSummaryHtml(d) {
  const a = d.review_analysis;
  if (!a || !a.total_reviews) return '';
  const pct = v => (v == null ? '—' : Math.round(v * 100) + '%');
  const model = a.text_model
    ? `Text model (${esc(a.text_model.trained_on || 'labelled dataset')}) judged ${a.scored_by_text_model} of ${a.total_reviews} reviews. ${esc(a.text_model.caveat || '')}`
    : 'No trained text model is loaded; only rule-based signals were used.';
  const burst = d.review_burst_summary;
  const burstLine = burst && burst.burst_points && burst.burst_points.length
    ? `<br>⚠ Review burst: ${burst.burst_points.map(p => `${esc(p.date)} (${p.count} reviews, z=${p.z_score})`).join('; ')}`
    : '';
  return `<div class="item" style="border-color:rgba(127,240,209,.35);">
    <strong>${a.total_reviews} reviews · <span style="color:#7ff0d1">${a.genuine} genuine</span> · <span style="color:#ff6b6b">${a.not_genuine} not genuine</span> (${pct(a.total_reviews ? a.not_genuine / a.total_reviews : null)})</strong>
    <p style="font-size:0.78rem;color:#9da5bb;margin-top:6px;">${model}${burstLine}</p>
    <p id="labelStatus" class="label-status">Disagree with a verdict? Use the small ✓ / ✗ under a review to correct it.</p>
  </div>`;
}

// ---- Real / Fake labels: a person's verdict on a review becomes training data (app/review_labels.py)
const labelState = {};      // review_hash -> 'real' | 'fake' | null (what you pressed this session; wins over the server copy)
const currentLabel = x => (Object.prototype.hasOwnProperty.call(labelState, x.review_hash) ? labelState[x.review_hash] : x.human_label);

function labelButtons(x) {
  if (!x.review_hash) return '';
  const mine = currentLabel(x);
  const button = (value, text) =>
    `<button class="btn-range ${mine === value ? 'active' : ''}" data-hash="${esc(x.review_hash)}" title="${value === 'real' ? 'Mark genuine' : 'Mark not genuine'}" onclick="labelReview(this, '${value}')">${text}</button>`;
  return `<div class="label-row"><span>your call</span>${button('real', '✓')}${button('fake', '✗')}${mine ? '<span class="label-msg">saved</span>' : ''}</div>`;
}

async function labelReview(btn, value) {
  const hash = btn.dataset.hash;
  const finding = ((currentData && currentData.findings) || []).find(f => f.review_hash === hash);
  if (!finding) return;
  const send = currentLabel(finding) === value ? 'clear' : value;      // pressing the active button again removes the label
  const row = btn.closest('.label-row');
  row.classList.add('busy');
  try {
    const response = await fetch('/api/reviews/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: finding.text, label: send, rating: finding.rating,
                             product_url: currentData.source.url, platform: currentData.source.platform }),
    });
    const data = await response.json();
    if (!response.ok) throw Error(data.detail || 'Could not save the label');
    labelState[hash] = data.label;
    row.outerHTML = labelButtons(finding);
    showLabelStatus(data.status);
  } catch (e) {
    row.classList.remove('busy');
    row.querySelector('.label-msg').textContent = 'Not saved: ' + e.message;
  }
}

function showLabelStatus(s) {
  const el = $('#labelStatus');
  if (!el || !s) return;
  const l = s.labels || { total: 0, real: 0, fake: 0 };
  let text = `You have corrected ${l.total} review(s) (${l.real} genuine, ${l.fake} not genuine). `;
  if (s.running) {
    text += 'The model is retraining in the background…';
  } else if (!s.can_train) {
    text += `Label at least ${Math.max(0, 20 - l.total)} more (with 5+ real and 5+ fake) before the model can learn from them.`;
  } else if (!s.auto_retrain) {
    text += 'Automatic retraining is switched off.';
  } else {
    const left = Math.max(0, s.retrain_at_new_labels - s.new_labels_since_training);
    text += left > 0 ? `The model retrains by itself after ${left} more.` : 'The model will retrain now.';
  }
  const run = s.last_run;
  if (run && run.message && !s.running) text += ` Last retrain: ${run.message}`;
  el.textContent = text;
}

let lastStatusFetch = 0;
function refreshLabelStatus() {
  if (Date.now() - lastStatusFetch < 8000) return;
  lastStatusFetch = Date.now();
  fetch('/api/model/status').then(r => (r.ok ? r.json() : null)).then(showLabelStatus).catch(() => {});
}

function toggleTranslation(idx) {
  const el = $(`#trans-${idx}`);
  if (el) el.hidden = !el.hidden;
}

// ---- Cross-store price comparison ----------------------------------------------------------------------
// One colour per store, fixed by name (never by rank), taken from the validated dark categorical palette.
const STORE_SLOTS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const STORE_SLOT_BY_NAME = { 'Flipkart': 0, 'Amazon': 1, 'Croma': 2, 'Vijay Sales': 3, 'Tata CLiQ': 4, 'Myntra': 5, 'Reliance Digital': 6, 'Snapdeal': 7 };
const LOWEST_RING = '#f1c66b';
const axisPrice = v => (v > 0 ? formatPrice(v) : '0');

function storeColor(name) {
  if (name in STORE_SLOT_BY_NAME) return STORE_SLOTS[STORE_SLOT_BY_NAME[name]];
  let h = 0;
  for (const ch of String(name)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return STORE_SLOTS[h % STORE_SLOTS.length];
}

function stripePattern(ctx, color) {
  const tile = document.createElement('canvas');
  tile.width = tile.height = 8;
  const t = tile.getContext('2d');
  t.fillStyle = 'rgba(255,255,255,0.04)';
  t.fillRect(0, 0, 8, 8);
  t.strokeStyle = color;
  t.lineWidth = 2;
  t.beginPath();
  t.moveTo(-2, 10); t.lineTo(10, -2);
  t.moveTo(-2, 2); t.lineTo(2, -2);
  t.moveTo(6, 10); t.lineTo(10, 6);
  t.stroke();
  return ctx.createPattern(tile, 'repeat');
}

const entryBadge = e => {
  if (e.kind === 'possible') return '<span class="tag tag-candidate">POSSIBLE · not verified</span>';
  const parts = [];
  if (e.kind === 'source') parts.push('<span class="tag tag-source">PASTED LINK</span>');
  else parts.push('<span class="tag tag-verified">VERIFIED</span>');
  if (e.is_lowest) parts.push('<span class="tag tag-lowest">★ LOWEST</span>');
  if (e.excluded_reason) parts.push(`<span class="tag tag-high">${esc(e.excluded_reason.toUpperCase())} · not counted</span>`);
  return parts.join('');
};

function comparisonRows(pc) {
  const series = pc.series || [];
  const buyable = series.filter(e => e.eligible).sort((a, b) => a.price - b.price);
  const others = series.filter(e => !e.eligible);
  return [...buyable, ...others, ...(pc.possible || [])];
}

function renderLowestBanner(pc, d) {
  const banner = $('#lowestBanner');
  if (!pc) { banner.hidden = true; return; }
  banner.hidden = false;
  banner.classList.toggle('none', !pc.lowest);
  if (pc.lowest) {
    $('#lowestBannerText').textContent = pc.lowest_summary;
    $('#lowestBannerLink').href = safeUrl(pc.lowest_link);
    $('#lowestBannerLink').hidden = false;
  } else {
    $('#lowestBannerText').textContent = 'No lowest price yet: no verified, in-stock, priced listing was found.';
    $('#lowestBannerLink').hidden = true;
  }
  $('#pendingStores').hidden = d.enrichment_status !== 'pending';
}

function renderStoreLegend(pc) {
  const rows = comparisonRows(pc);
  const item = e => {
    const rating = e.rating != null ? `⭐ ${esc(e.rating)} / 5${e.review_count ? ` (${Number(e.review_count).toLocaleString()})` : ''}` : 'rating not shown';
    const why = [...(e.reasons || []).slice(0, 4).map(esc), ...(e.flags || []).map(f => `<span class="flag">⚠ ${esc(f)}</span>`)].join(' · ');
    return `<div class="store-row ${e.is_lowest ? 'is-lowest' : ''} ${e.kind === 'possible' ? 'is-possible' : ''} ${e.excluded_reason ? 'is-excluded' : ''}">
      <i class="swatch ${e.kind === 'possible' ? 'hatched' : ''}" style="--c:${storeColor(e.store)}"></i>
      <div class="store-main">
        <a href="${esc(safeUrl(e.url))}" target="_blank" rel="noreferrer"><b>${esc(e.store)}</b> ↗</a> ${entryBadge(e)}
        <small>${esc(e.title)}</small>
        ${why ? `<small class="why">${why}</small>` : ''}
      </div>
      <div class="store-price"><strong>${e.price ? formatPrice(e.price, e.currency) : 'price not read'}</strong><small>${rating}</small></div>
    </div>`;
  };
  const verified = rows.filter(e => e.kind !== 'possible');
  const possible = rows.filter(e => e.kind === 'possible');
  const searched = (pc.stores_searched || []).map(s => {
    const ok = s.listings > 0;
    return `<span class="searched ${ok ? 'ok' : 'miss'}">${esc(s.store)}: ${ok ? s.listings + ' listing(s)' : esc(s.status || 'nothing found')}</span>`;
  }).join('');
  $('#storeLegend').innerHTML = `
    ${verified.map(item).join('')}
    ${possible.length ? `<p class="legend-sep">Possible matches — not verified, shown for reference and never counted as the lowest price</p>${possible.map(item).join('')}` : ''}
    ${(pc.notes || []).length ? `<ul class="graph-notes">${pc.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>` : ''}
    ${searched ? `<div class="searched-row"><small>Stores searched:</small> ${searched}</div>` : ''}`;
}

function showAnchor(meta, price) {
  $('#chartPointInfo').hidden = false;
  $('#chartPointTitle').textContent = `📍 ${meta.store}${meta.kind === 'possible' ? ' (possible match, not verified)' : ''}`;
  $('#chartPointDesc').textContent = `${meta.title || ''} | Price: ${formatPrice(price, meta.currency)}${meta.rating != null ? ' · Rating: ⭐ ' + meta.rating + ' / 5' : ''}`;
  $('#chartPointLink').href = safeUrl(meta.url);
}

function renderStoreBars(pc) {
  const wrap = $('#storeBarsWrap');
  if (barChart) { barChart.destroy(); barChart = null; }
  const rows = comparisonRows(pc).filter(e => e.price);
  wrap.hidden = !rows.length;
  if (!rows.length) return;
  wrap.style.height = `${Math.max(150, rows.length * 44 + 56)}px`;
  const canvas = $('#storeBars');
  const ctx = canvas.getContext('2d');
  const fills = rows.map(e => {
    const c = storeColor(e.store);
    if (e.kind === 'possible') return stripePattern(ctx, c);
    return e.excluded_reason ? 'rgba(157,165,187,0.35)' : c;
  });
  const valueLabels = {
    id: 'valueLabels',
    afterDatasetsDraw(c) {
      const { ctx: g } = c;
      g.save();
      g.font = "600 12px 'DM Mono', monospace";
      g.textBaseline = 'middle';
      c.getDatasetMeta(0).data.forEach((bar, i) => {
        const e = rows[i];
        g.fillStyle = e.excluded_reason ? '#9da5bb' : '#f4f5fa';
        const note = e.is_lowest ? '  ★ LOWEST' : (e.excluded_reason ? `  ${e.excluded_reason}` : (e.kind === 'possible' ? '  possible' : ''));
        g.fillText(formatPrice(e.price, e.currency) + note, bar.x + 8, bar.y);
      });
      g.restore();
    },
  };
  barChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: rows.map(e => e.kind === 'source' ? `${e.store} (pasted)` : e.store),
      datasets: [{
        data: rows.map(e => e.price),
        backgroundColor: fills,
        borderColor: rows.map(e => e.is_lowest ? LOWEST_RING : (e.kind === 'possible' ? storeColor(e.store) : 'transparent')),
        borderWidth: rows.map(e => e.is_lowest ? 3 : (e.kind === 'possible' ? 1 : 0)),
        borderRadius: 4,
        barThickness: 24,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { right: 190 } },
      onClick: (evt, active) => { if (active && active.length) { const e = rows[active[0].index]; showAnchor(e, e.price); } },
      onHover: (evt, active) => { evt.native.target.style.cursor = active.length ? 'pointer' : 'default'; },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => rows[items[0].dataIndex].store,
            label: item => {
              const e = rows[item.dataIndex];
              return [`${formatPrice(e.price, e.currency)} · ${e.kind === 'possible' ? 'possible match' : (e.kind === 'source' ? 'pasted link' : 'verified match')}`,
                      ...(e.excluded_reason ? [`Not counted: ${e.excluded_reason}`] : []), e.title.slice(0, 70)];
            },
          },
        },
      },
      scales: {
        x: { beginAtZero: true, ticks: { color: '#9da5bb', callback: axisPrice }, grid: { color: 'rgba(194,205,255,.08)' } },
        y: { ticks: { color: '#f4f5fa', font: { family: 'Manrope', weight: '600' } }, grid: { display: false } },
      },
    },
    plugins: [valueLabels],
  });
}

function renderHistory(pc) {
  if (chart) { chart.destroy(); chart = null; }
  const note = $('#historyNote');
  const wrap = $('#historyWrap');
  const cutoff = Date.now() - activeRangeDays * 24 * 60 * 60 * 1000;
  const entries = [...(pc.series || []), ...(pc.possible || [])].map(e => ({
    ...e, pts: (e.points || []).map(p => ({ x: new Date(p.observed_at).getTime(), y: p.price })).filter(p => p.x >= cutoff),
  })).filter(e => e.pts.length);
  const longest = Math.max(0, ...entries.map(e => e.pts.length));
  if (longest < 2) {
    wrap.hidden = true;
    note.hidden = false;
    note.textContent = `Each store has only today's price so far (${entries.length} store${entries.length === 1 ? '' : 's'} observed once). Every time a link is analysed, the real prices seen are saved, and a line per store appears here once a listing has been seen on two different occasions.`;
    return;
  }
  wrap.hidden = false;
  note.hidden = true;
  const datasets = entries.map(e => {
    const color = storeColor(e.store);
    const possible = e.kind === 'possible';
    return {
      label: `${e.store}${possible ? ' (possible)' : ''}`,
      data: e.pts,
      borderColor: color,
      backgroundColor: possible ? 'rgba(14,18,34,1)' : color,
      pointBackgroundColor: possible ? 'rgba(14,18,34,1)' : color,
      pointBorderColor: e.is_lowest ? LOWEST_RING : color,
      pointBorderWidth: e.is_lowest ? 3 : 2,
      pointRadius: e.is_lowest ? 6 : 4,
      pointHoverRadius: 8,
      borderWidth: e.is_lowest ? 3 : 2,
      borderDash: possible ? [5, 5] : [],
      tension: 0.15,
      fill: false,
      meta: e,
    };
  });
  chart = new Chart($('#priceChart'), {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', intersect: true },
      onClick: (evt, active) => { if (active && active.length) { const ds = chart.data.datasets[active[0].datasetIndex]; showAnchor(ds.meta, ds.data[active[0].index].y); } },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: {
          title: items => items[0].dataset.label,
          label: item => `${formatPrice(item.raw.y)} on ${new Date(item.raw.x).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}`,
        } },
      },
      scales: {
        x: { type: 'linear', ticks: { color: '#9da5bb', maxTicksLimit: 7, callback: v => new Date(v).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) }, grid: { color: 'rgba(194,205,255,.08)' } },
        y: { ticks: { color: '#9da5bb', callback: axisPrice }, grid: { color: 'rgba(194,205,255,.08)' } },
      },
    },
  });
}

function renderPriceGraph(d) {
  $('#chartPointInfo').hidden = true;
  lastClickedPointKey = null;
  const pc = d.price_comparison;
  renderLowestBanner(pc, d);
  if (!pc) {
    if (barChart) { barChart.destroy(); barChart = null; }
    if (chart) { chart.destroy(); chart = null; }
    $('#storeBarsWrap').hidden = true;
    $('#historyWrap').hidden = true;
    $('#storeLegend').innerHTML = '<div class="empty">The cross-store price comparison is not available for this analysis.</div>';
    return;
  }
  renderStoreLegend(pc);
  if (!window.Chart) { $('#storeBarsWrap').hidden = true; $('#historyWrap').hidden = true; return; }
  renderStoreBars(pc);
  renderHistory(pc);
}
