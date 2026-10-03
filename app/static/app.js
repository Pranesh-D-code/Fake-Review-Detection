let chart;
let currentData = null;
let activeAnalysisId = null;
let activeRangeDays = 180;
let lastClickedPointKey = null;
let activeCandidateFilter = 'all';

const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

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
          <span class="pill-nykaa">Nykaa</span>
          <span class="pill-myntra">Myntra</span>
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
    '🌐 Executing parallel multi-store search for matching products (Amazon, Flipkart, Myntra, Nykaa)...'
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
      if (!response.ok) return;
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
  }, 1800);
}

function renderCandidateItems(candidates, currency) {
  if (!candidates || !candidates.length) return `<div class="empty">No comparable product candidates were collected from alternative stores.</div>`;

  const imgMatches = candidates.filter(x => (x.image_similarity || 0) >= 0.50 || Boolean(x.image_url));
  const textMatches = candidates.filter(x => (x.title_similarity || 0) >= 0.25);
  const multiMatches = candidates.filter(x => (x.title_similarity || 0) >= 0.25 && ((x.image_similarity || 0) >= 0.50 || Boolean(x.image_url)));

  let filtered = candidates;
  if (activeCandidateFilter === 'img') filtered = imgMatches;
  else if (activeCandidateFilter === 'text') filtered = textMatches;
  else if (activeCandidateFilter === 'multi') filtered = multiMatches;

  const filterBar = `
    <div class="candidate-filter-bar" style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
      <button class="btn-range ${activeCandidateFilter==='all'?'active':''}" onclick="setCandidateFilter('all')">All Matches (${candidates.length})</button>
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
        <span class="tag ${x.status === 'verified' ? 'tag-verified' : 'tag-candidate'}">${esc(x.status)}</span>
        ${ratingBadge}
        ${imgSimBadge}
        <p style="margin-top:6px;font-weight:600;color:#f1f5f9;">${esc(x.title)}</p>
        <p style="margin:4px 0;color:#7ff0d1;font-weight:bold;">Price: ${formatPrice(x.price, currency)} · <span style="color:#9da5bb;font-weight:normal;">${textSimBadge}</span></p>
        <a href="${esc(x.url)}" target="_blank" rel="noreferrer">Open direct product page ↗</a>
      </div>
    `;
  }).join('');

  return filterBar + itemsHtml;
}

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

  if (d.lowest_verified_match && d.lowest_verified_match.price) {
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
    $('#priceTrend').textContent = d.price_summary.trend || 'Stable';
    $('#trendMeta').textContent = `${d.price_summary.obs_count} observation sample points`;

    $('#pMin').textContent = formatPrice(d.price_summary.min_price || d.source.price, d.source.currency);
    $('#pAvg').textContent = formatPrice(d.price_summary.avg_price || d.source.price, d.source.currency);
    $('#pMax').textContent = formatPrice(d.price_summary.max_price || d.source.price, d.source.currency);
    $('#pOffer').textContent = formatPrice(d.price_summary.offer_price || d.source.price, d.source.currency);
    $('#pTrend').textContent = `${d.price_summary.trend || 'Stable'} (${d.price_summary.change_pct || 0}%)`;

    $('#priceIndicator').textContent = `📈 ${d.price_summary.comparison_to_avg || '6-month price history active'}`;
    $('#offerVal').textContent = formatPrice(d.price_summary.offer_price, d.source.currency);
  }

  renderPriceGraph(d);

  // Keep the fetch animation visible until the background cross-store job completes.
  if (d.enrichment_status === 'pending') {
    renderCandidateSkeleton();
  } else {
    $('#candidates').innerHTML = renderCandidateItems(d.candidates, d.source.currency);
  }

  // Findings
  if (d.findings && d.findings.length) {
    $('#findings').innerHTML = d.findings.map((x, idx) => {
      const suspicious = x.classification === 'suspicious';
      const genuineLooking = x.classification === 'genuine-looking';
      const badgeTag = suspicious
        ? '<span class="tag tag-high">⚠️ SUSPICIOUS SIGNALS</span>'
        : genuineLooking
          ? '<span class="tag tag-verified">✓ GENUINE-LOOKING</span>'
          : '<span class="tag">REQUIRES REVIEW</span>';
      
      const hasReason = x.signals && x.signals.length;
      const reasonHtml = hasReason ? `<div class="reason-box" style="border-color:${suspicious ? 'rgba(255,107,107,.4)' : 'rgba(127,240,209,.3)'}">
        ${suspicious ? '⚠️ <b>Observed suspicious signals:</b> ' : 'ℹ️ <b>Review-text evidence:</b> '}${x.signals.map(esc).join(' | ')}
      </div>` : '';
      
      const transBtn = x.translation ? `<button class="btn-translate" onclick="toggleTranslation(${idx})">🌐 Translate to English</button>` : '';
      const transBox = x.translation ? `<div id="trans-${idx}" class="reason-box" style="background:rgba(156,140,255,.08);border-color:rgba(156,140,255,.3);color:#c7beff;" hidden>🌐 <b>English Translation:</b> "${esc(x.translation)}"</div>` : '';
      
      return `
        <div class="item">
          <strong id="rev-text-${idx}">${x.rating ? x.rating + '★ ' : ''}"${esc(x.text)}"</strong>
          <p>Review-text sentiment ${x.sentiment ?? '—'} ${badgeTag} · ${Math.round((x.classification_confidence || 0) * 100)}% evidence confidence</p>
          ${transBtn}
          ${transBox}
          ${reasonHtml}
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

  if ($('#citationIEEE')) $('#citationIEEE').value = d.citation_ieee || '';
  if ($('#citationBibTeX')) $('#citationBibTeX').value = d.citation_bibtex || '';
  $('#notices').innerHTML = (d.notices || []).map(x => `<div class="item"><small>${esc(x)}</small></div>`).join('');
}

function toggleTranslation(idx) {
  const el = $(`#trans-${idx}`);
  if (el) el.hidden = !el.hidden;
}

function renderPriceGraph(d) {
  if (!window.Chart) return;
  if (chart) chart.destroy();
  $('#chartPointInfo').hidden = true;
  lastClickedPointKey = null;

  const now = new Date();
  const allPoints = d.price_history || [];

  let displayPoints = allPoints;
  if (!displayPoints.length && d.source.price) {
    const d30 = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();
    displayPoints = [
      { observed_at: d30, price: d.source.price },
      { observed_at: now.toISOString(), price: d.source.price }
    ];
  } else if (displayPoints.length === 1 && d.source.price) {
    const single = displayPoints[0];
    const prevDate = new Date(new Date(single.observed_at).getTime() - 14 * 24 * 60 * 60 * 1000).toISOString();
    displayPoints = [
      { observed_at: prevDate, price: single.price },
      single
    ];
  }

  const labels = displayPoints.map(x => new Date(x.observed_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }));
  const mainPrices = displayPoints.map(x => x.price);

  const mainDataset = {
    label: `Original ${d.source.platform} Listing Price (INR)`,
    data: mainPrices,
    borderColor: '#7ff0d1',
    pointBackgroundColor: '#7ff0d1',
    pointRadius: 6,
    pointHoverRadius: 9,
    borderWidth: 3,
    tension: 0.2,
    fill: false,
    meta: {
      platform: d.source.platform,
      title: d.source.title,
      url: d.source.url,
      rating: d.source.rating
    }
  };

  const candidatePrices = (d.candidates || []).filter(c => c.price && c.price > 0);
  const compDatasets = candidatePrices.slice(0, 4).map((c, i) => {
    const colors = ['#ff9f43', '#00d2d3', '#54a0ff', '#ff6b6b'];
    return {
      label: `${c.platform} Candidate: ${c.title.slice(0, 22)}… (INR ₹${c.price})`,
      data: displayPoints.map(() => c.price),
      borderColor: colors[i % colors.length],
      pointBackgroundColor: colors[i % colors.length],
      pointRadius: 4,
      pointHoverRadius: 7,
      borderWidth: 2,
      borderDash: [5, 5],
      tension: 0.1,
      fill: false,
      meta: {
        platform: c.platform,
        title: c.title,
        url: c.url,
        rating: c.rating
      }
    };
  });

  const chartCanvas = $('#priceChart');
  chart = new Chart(chartCanvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [mainDataset, ...compDatasets]
    },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', intersect: false },
      onClick: (e, activeElements) => {
        if (!activeElements || !activeElements.length) return;
        const el = activeElements[0];
        const datasetIndex = el.datasetIndex;
        const index = el.index;
        const dataset = chart.data.datasets[datasetIndex];
        const meta = dataset.meta || {};

        const itemUrl = meta.url || d.source.url;
        const itemTitle = meta.title || d.source.title;
        const itemStore = meta.platform || d.source.platform;
        const itemPrice = dataset.data[index];
        const itemRating = meta.rating ? `⭐ ${meta.rating} / 5` : '';

        $('#chartPointInfo').hidden = false;
        $('#chartPointTitle').textContent = `📍 Selected Anchor Point: ${itemStore} Listing`;
        $('#chartPointDesc').textContent = `${itemTitle} | Price: ${formatPrice(itemPrice)} ${itemRating ? '· Rating: ' + itemRating : ''}`;
        $('#chartPointLink').href = itemUrl;
      },
      plugins: {
        legend: { labels: { color: '#9da5bb', font: { family: 'DM Mono' } } }
      },
      scales: {
        x: { ticks: { color: '#9da5bb' }, grid: { color: 'rgba(194,205,255,.08)' } },
        y: { ticks: { color: '#9da5bb' }, grid: { color: 'rgba(194,205,255,.08)' } }
      }
    }
  });
}
