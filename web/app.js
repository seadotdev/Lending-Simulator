// ============================================================================
// Loanville Arena — Application
// Single-page app with hash routing, dynamic views, SVG charts
// ============================================================================

// --- Globals ---
let data = null;
let leaderboard = null;
let matchCache = {};
let currentSort = { key: 'profit_elo', dir: 'desc' };

// --- Model Colors (from sample data) ---
function getModelMeta(modelId) {
  if (!data) return { color: '#64748b', icon: '?', short: modelId };
  const m = data.models.find(m => m.model_id === modelId);
  return m || { color: '#64748b', icon: modelId.slice(0, 2).toUpperCase(), short: modelId.split('/').pop(), display_name: modelId.split('/').pop() };
}

// --- Router ---
const router = {
  navigate(view, params = {}) {
    const hash = '#/' + view + (params.id ? '/' + encodeURIComponent(params.id) : '');
    window.location.hash = hash;
  },

  parse() {
    const hash = window.location.hash.replace('#/', '') || 'dashboard';
    const parts = hash.split('/');
    return {
      view: parts[0],
      id: parts[1] ? decodeURIComponent(parts[1]) : null,
    };
  },

  init() {
    window.addEventListener('hashchange', () => this.render());
    this.render();
  },

  render() {
    const { view, id } = this.parse();
    const content = document.getElementById('main-content');

    // Update nav active state
    document.querySelectorAll('.nav-link').forEach(link => {
      link.classList.toggle('active', link.dataset.view === view ||
        (view === 'model' && link.dataset.view === 'leaderboard') ||
        (view === 'match' && link.dataset.view === 'matches'));
    });

    // Render view
    switch (view) {
      case 'dashboard': content.innerHTML = views.dashboard(); break;
      case 'leaderboard': content.innerHTML = views.leaderboard(); break;
      case 'model': content.innerHTML = views.modelDetail(id); break;
      case 'matches': content.innerHTML = views.matchList(); break;
      case 'match': content.innerHTML = views.matchDetail(id); break;
      case 'compare': content.innerHTML = views.compare(); break;
      default: content.innerHTML = views.dashboard();
    }

    // Add entrance animation
    content.classList.remove('view-enter');
    void content.offsetWidth; // force reflow
    content.classList.add('view-enter');

    // Post-render hooks (charts, etc.)
    this.postRender(view, id);

    // Scroll to top
    window.scrollTo(0, 0);
  },

  postRender(view, id) {
    if (view === 'model' && id) {
      drawEloHistoryChart(id);
    }
    if (view === 'compare') {
      setupCompareHandlers();
    }
  }
};

// --- Utility Functions ---
function elo(v) { return Math.round(v); }
function pct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '\u2014'; }
function pctRaw(v) { return v != null ? v.toFixed(1) + '%' : '\u2014'; }
function dollar(v) {
  if (v == null) return '\u2014';
  const abs = Math.abs(Math.round(v));
  const formatted = abs >= 1000000
    ? '$' + (abs / 1000000).toFixed(1) + 'M'
    : abs >= 1000
      ? '$' + (abs / 1000).toFixed(0) + 'K'
      : '$' + abs;
  return v < 0 ? '-' + formatted : formatted;
}

function eloColorClass(v) {
  return v >= 1520 ? 'positive' : v <= 1480 ? 'negative' : 'neutral';
}

function formatDate(isoStr) {
  const d = new Date(isoStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatTime(isoStr) {
  const d = new Date(isoStr);
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function shortModelName(modelId) {
  return getModelMeta(modelId).short || modelId.split('/').pop();
}

// --- SVG Sparkline ---
function sparklineSVG(history, key, w = 80, h = 22, color = null) {
  if (!history || history.length < 2) return '';
  const vals = history.map(h => h[key]);
  const min = Math.min(...vals) - 5;
  const max = Math.max(...vals) + 5;
  const range = max - min || 1;
  const points = vals.map((v, i) =>
    `${(i / (vals.length - 1)) * w},${h - ((v - min) / range) * (h - 4) - 2}`
  ).join(' ');
  const last = vals[vals.length - 1];
  const c = color || (last >= 1500 ? 'var(--green-400)' : 'var(--red-400)');
  const lastX = w;
  const lastY = h - ((last - min) / range) * (h - 4) - 2;

  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="vertical-align:middle">
    <polyline fill="none" stroke="${c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" points="${points}" opacity="0.8"/>
    <circle cx="${lastX}" cy="${lastY}" r="2" fill="${c}"/>
  </svg>`;
}

// --- Large Elo Chart (SVG) ---
function drawEloHistoryChart(modelId) {
  const container = document.getElementById('elo-chart-container');
  if (!container) return;

  const hist = (leaderboard.elo_history || {})[modelId] || [];
  if (hist.length < 2) {
    container.innerHTML = '<div class="empty-state"><p>Not enough data for chart</p></div>';
    return;
  }

  const W = 700, H = 220, PAD_L = 50, PAD_R = 20, PAD_T = 20, PAD_B = 40;
  const keys = ['profit_elo', 'credit_elo', 'dealshare_elo'];
  const colors = ['var(--blue-400)', 'var(--green-400)', 'var(--purple-400)'];
  const labels = ['Profit', 'Credit', 'DealShare'];

  const allVals = keys.flatMap(k => hist.map(h => h[k]));
  const min = Math.min(...allVals, 1490) - 15;
  const max = Math.max(...allVals, 1510) + 15;
  const range = max - min || 1;

  const x = i => PAD_L + (i / (hist.length - 1)) * (W - PAD_L - PAD_R);
  const y = v => PAD_T + (1 - (v - min) / range) * (H - PAD_T - PAD_B);

  let svg = `<svg width="100%" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="max-width:${W}px">`;

  // Grid lines
  const step = range > 100 ? 50 : range > 40 ? 20 : 10;
  for (let v = Math.ceil(min / step) * step; v <= max; v += step) {
    const yv = y(v);
    svg += `<line x1="${PAD_L}" y1="${yv}" x2="${W - PAD_R}" y2="${yv}" stroke="var(--border-subtle)" stroke-dasharray="4"/>`;
    svg += `<text x="${PAD_L - 8}" y="${yv + 4}" fill="var(--text-muted)" font-size="10" font-family="var(--font-mono)" text-anchor="end">${Math.round(v)}</text>`;
  }

  // 1500 baseline
  const y1500 = y(1500);
  svg += `<line x1="${PAD_L}" y1="${y1500}" x2="${W - PAD_R}" y2="${y1500}" stroke="var(--text-dim)" stroke-dasharray="2" stroke-width="1" opacity="0.5"/>`;
  svg += `<text x="${W - PAD_R + 4}" y="${y1500 + 3}" fill="var(--text-dim)" font-size="9" font-family="var(--font-mono)">1500</text>`;

  // X-axis labels (match numbers)
  hist.forEach((h, i) => {
    if (i % Math.max(1, Math.floor(hist.length / 6)) === 0 || i === hist.length - 1) {
      svg += `<text x="${x(i)}" y="${H - 8}" fill="var(--text-muted)" font-size="9" font-family="var(--font-mono)" text-anchor="middle">#${i + 1}</text>`;
    }
  });

  // Area fills (subtle gradient effect)
  keys.forEach((k, ki) => {
    const pts = hist.map((h, i) => `${x(i)},${y(h[k])}`);
    const areaPoints = pts.join(' ') + ` ${x(hist.length - 1)},${H - PAD_B} ${x(0)},${H - PAD_B}`;
    svg += `<polygon fill="${colors[ki]}" opacity="0.04" points="${areaPoints}"/>`;
  });

  // Lines
  keys.forEach((k, ki) => {
    const points = hist.map((h, i) => `${x(i)},${y(h[k])}`).join(' ');
    svg += `<polyline fill="none" stroke="${colors[ki]}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="${points}" opacity="0.9"/>`;

    // End dots
    const lastH = hist[hist.length - 1];
    svg += `<circle cx="${x(hist.length - 1)}" cy="${y(lastH[k])}" r="4" fill="${colors[ki]}" stroke="var(--bg-surface)" stroke-width="2"/>`;
  });

  // Hover circles (interactive hit areas)
  hist.forEach((h, i) => {
    keys.forEach((k, ki) => {
      svg += `<circle cx="${x(i)}" cy="${y(h[k])}" r="8" fill="transparent" class="chart-dot" data-match="${i}" data-key="${k}" data-value="${Math.round(h[k])}">
        <title>Match #${i + 1}: ${labels[ki]} ${Math.round(h[k])}</title>
      </circle>`;
    });
  });

  // Legend
  const legendY = H - 2;
  labels.forEach((label, ki) => {
    const lx = PAD_L + ki * 110;
    svg += `<circle cx="${lx}" cy="${legendY - 3}" r="4" fill="${colors[ki]}"/>`;
    svg += `<text x="${lx + 8}" y="${legendY}" fill="var(--text-secondary)" font-size="11" font-family="var(--font-body)">${label}: <tspan font-weight="600" fill="${colors[ki]}">${Math.round(hist[hist.length - 1][keys[ki]])}</tspan></text>`;
  });

  svg += '</svg>';
  container.innerHTML = svg;
}

// ============================================================================
// Views
// ============================================================================
const views = {};

// --- Dashboard ---
views.dashboard = function () {
  const s = leaderboard.standings;
  const matches = data.matches;
  const top3 = s.slice(0, 3);
  const recentMatches = [...matches].reverse().slice(0, 3);

  // Stats
  const totalDealsWon = s.reduce((sum, m) => {
    const cm = m.confusion_agg;
    return sum + cm.good.approved + cm.bad.approved + cm.fraud.approved;
  }, 0);
  const totalFraudsCaught = s.reduce((sum, m) => sum + m.confusion_agg.fraud.rejected, 0);
  const avgRaroc = s.length > 0 ? s.reduce((sum, m) => sum + m.avg_raroc, 0) / s.length : 0;

  // Reorder podium: [2nd, 1st, 3rd]
  const podiumOrder = top3.length >= 3 ? [top3[1], top3[0], top3[2]] : top3;
  const podiumClasses = top3.length >= 3 ? ['second', 'first', 'third'] : ['first', 'second', 'third'];
  const podiumRanks = top3.length >= 3 ? ['2', '1', '3'] : ['1', '2', '3'];

  let html = `
    <div class="dashboard-hero">
      <div class="hero-overline">AI Lending Simulation</div>
      <h1 class="hero-title">Loanville Arena</h1>
      <p class="hero-subtitle">Where AI models compete as commercial lenders. Approve loans, set rates, dodge fraud — and prove who&rsquo;s the best underwriter.</p>
    </div>

    <div class="stats-strip">
      <div class="stat-card">
        <div class="stat-value gold">${leaderboard.n_matches}</div>
        <div class="stat-label">Matches Played</div>
      </div>
      <div class="stat-card">
        <div class="stat-value blue">${s.length}</div>
        <div class="stat-label">Competing Models</div>
      </div>
      <div class="stat-card">
        <div class="stat-value green">${totalFraudsCaught}</div>
        <div class="stat-label">Frauds Caught</div>
      </div>
      <div class="stat-card">
        <div class="stat-value purple">${avgRaroc >= 0 ? '+' : ''}${avgRaroc.toFixed(1)}%</div>
        <div class="stat-label">Avg RAROC</div>
      </div>
    </div>

    <div class="podium-section">
      <div class="section-header">
        <h2 class="section-title">Top Performers</h2>
        <a class="section-action" onclick="router.navigate('leaderboard')">
          View full leaderboard &rarr;
        </a>
      </div>
      <div class="podium-grid">
        ${podiumOrder.map((m, i) => {
          const meta = getModelMeta(m.model_id);
          return `
            <div class="podium-card ${podiumClasses[i]}" onclick="router.navigate('model', {id: '${m.model_id}'})">
              <div class="podium-rank">#${podiumRanks[i]}</div>
              <div class="podium-avatar" style="background:${meta.color}">
                ${meta.icon}
              </div>
              <div class="podium-name">${m.display_name}</div>
              <div class="podium-model-id">${m.model_id}</div>
              <div class="podium-elo ${eloColorClass(m.profit_elo)}">${elo(m.profit_elo)}</div>
              <div class="podium-elo-label">Profit Elo</div>
              <div class="podium-stats">
                <div>
                  <div class="podium-stat-value elo-cell ${eloColorClass(m.credit_elo)}">${elo(m.credit_elo)}</div>
                  <div class="podium-stat-label">Credit</div>
                </div>
                <div>
                  <div class="podium-stat-value elo-cell ${eloColorClass(m.dealshare_elo)}">${elo(m.dealshare_elo)}</div>
                  <div class="podium-stat-label">DealShare</div>
                </div>
                <div>
                  <div class="podium-stat-value">${m.matches_played}</div>
                  <div class="podium-stat-label">Matches</div>
                </div>
              </div>
            </div>`;
        }).join('')}
      </div>
    </div>

    <div class="recent-matches">
      <div class="section-header">
        <h2 class="section-title">Recent Matches</h2>
        <a class="section-action" onclick="router.navigate('matches')">
          View all matches &rarr;
        </a>
      </div>
      ${recentMatches.map(match => {
        const sorted = [...match.results].sort((a, b) => (b.raroc_score || 0) - (a.raroc_score || 0));
        return `
          <div class="match-card" onclick="router.navigate('match', {id: '${match.match_id}'})">
            <div class="match-card-header">
              <div>
                <span style="font-weight:600">${formatDate(match.timestamp_utc)}</span>
                <span style="color:var(--text-muted);margin-left:8px">${formatTime(match.timestamp_utc)}</span>
              </div>
              <div class="match-card-meta">
                <span class="badge badge-mix">${match.mix}</span>
                <span class="badge badge-count">${match.n_borrowers} borrowers</span>
              </div>
            </div>
            <div class="match-card-results">
              ${sorted.map((r, i) => {
                const meta = getModelMeta(r.model_id);
                return `
                  <div class="match-result-item">
                    <div class="match-result-avatar" style="background:${meta.color}">${meta.icon}</div>
                    <div class="match-result-info">
                      <div class="match-result-name">${i === 0 ? '\u2B50 ' : ''}${shortModelName(r.model_id)}</div>
                      <div class="match-result-score ${r.raroc_score >= 0 ? 'text-positive' : 'text-negative'}">
                        RAROC ${r.raroc_score >= 0 ? '+' : ''}${r.raroc_score.toFixed(1)}%
                        &middot; ${r.deals_won} won &middot; ${r.frauds_funded} fraud${r.frauds_funded !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </div>`;
              }).join('')}
            </div>
          </div>`;
      }).join('')}
    </div>
  `;

  return html;
};

// --- Leaderboard ---
views.leaderboard = function () {
  const s = [...leaderboard.standings];
  const hist = leaderboard.elo_history || {};

  // Sort
  s.sort((a, b) => {
    const av = a[currentSort.key] ?? 0;
    const bv = b[currentSort.key] ?? 0;
    return currentSort.dir === 'desc' ? bv - av : av - bv;
  });

  const sortIcon = (key) => {
    const active = currentSort.key === key;
    const arrow = active ? (currentSort.dir === 'desc' ? '\u25BC' : '\u25B2') : '\u25BC';
    return `<span class="sort-icon">${arrow}</span>`;
  };

  const sortClass = (key) => currentSort.key === key ? 'sorted' : '';

  let html = `
    <h1 class="page-title">Leaderboard</h1>
    <p class="page-subtitle">${leaderboard.n_matches} matches played &middot; K=${leaderboard.config.k} &middot; ${s.length} models</p>

    <div class="leaderboard-controls">
      <button class="elo-toggle ${currentSort.key === 'profit_elo' ? 'active' : ''}" onclick="sortLeaderboard('profit_elo')">Profit Elo</button>
      <button class="elo-toggle ${currentSort.key === 'credit_elo' ? 'active' : ''}" onclick="sortLeaderboard('credit_elo')">Credit Elo</button>
      <button class="elo-toggle ${currentSort.key === 'dealshare_elo' ? 'active' : ''}" onclick="sortLeaderboard('dealshare_elo')">DealShare Elo</button>
      <button class="elo-toggle ${currentSort.key === 'avg_raroc' ? 'active' : ''}" onclick="sortLeaderboard('avg_raroc')">Avg RAROC</button>
    </div>

    <div class="data-table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th style="width:40px">#</th>
            <th>Model</th>
            <th class="${sortClass('profit_elo')}" onclick="sortLeaderboard('profit_elo')">Profit Elo ${sortIcon('profit_elo')}</th>
            <th class="${sortClass('credit_elo')}" onclick="sortLeaderboard('credit_elo')">Credit Elo ${sortIcon('credit_elo')}</th>
            <th class="${sortClass('dealshare_elo')}" onclick="sortLeaderboard('dealshare_elo')">DealShare ${sortIcon('dealshare_elo')}</th>
            <th class="${sortClass('matches_played')}" onclick="sortLeaderboard('matches_played')">Matches ${sortIcon('matches_played')}</th>
            <th class="${sortClass('avg_raroc')}" onclick="sortLeaderboard('avg_raroc')">Avg RAROC ${sortIcon('avg_raroc')}</th>
            <th>Confusion</th>
            <th>Trend</th>
          </tr>
        </thead>
        <tbody>
          ${s.map((m, i) => {
            const meta = getModelMeta(m.model_id);
            const h = hist[m.model_id] || [];
            const cm = m.confusion_agg || {};
            const rankClass = i === 0 ? 'top-1' : i === 1 ? 'top-2' : i === 2 ? 'top-3' : '';

            return `
              <tr onclick="router.navigate('model', {id: '${m.model_id}'})">
                <td class="rank-cell ${rankClass}">${i + 1}</td>
                <td>
                  <div class="model-cell">
                    <div class="model-avatar" style="background:${meta.color}">${meta.icon}</div>
                    <div>
                      <div class="model-name">${m.display_name}</div>
                      <div class="model-id-sub">${m.model_id}</div>
                    </div>
                  </div>
                </td>
                <td class="elo-cell ${eloColorClass(m.profit_elo)}">${elo(m.profit_elo)}</td>
                <td class="elo-cell ${eloColorClass(m.credit_elo)}">${elo(m.credit_elo)}</td>
                <td class="elo-cell ${eloColorClass(m.dealshare_elo)}">${elo(m.dealshare_elo)}</td>
                <td class="matches-cell">${m.matches_played}</td>
                <td class="raroc-cell ${m.avg_raroc >= 0 ? 'text-positive' : 'text-negative'}">${m.avg_raroc >= 0 ? '+' : ''}${m.avg_raroc.toFixed(1)}%</td>
                <td>
                  <div class="mini-cm">
                    <div class="mini-cm-cell correct" title="Good→Approve">${cm.good?.approved || 0}</div>
                    <div class="mini-cm-cell incorrect" title="Bad→Approve">${cm.bad?.approved || 0}</div>
                    <div class="mini-cm-cell incorrect" title="Fraud→Approve">${cm.fraud?.approved || 0}</div>
                  </div>
                </td>
                <td class="sparkline-cell">${sparklineSVG(h, currentSort.key.includes('elo') ? currentSort.key : 'profit_elo')}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;

  return html;
};

// --- Model Detail ---
views.modelDetail = function (modelId) {
  const s = leaderboard.standings.find(s => s.model_id === modelId);
  if (!s) return '<div class="empty-state"><div class="empty-state-icon">\u2753</div><div class="empty-state-text">Model not found</div></div>';

  const meta = getModelMeta(modelId);
  const hist = (leaderboard.elo_history || {})[modelId] || [];
  const cm = s.confusion_agg || {};
  const ra = s.rate_analysis || {};

  // Compute deltas from initial
  const initial = leaderboard.config.initial_elo;
  const profitDelta = s.profit_elo - initial;
  const creditDelta = s.credit_elo - initial;
  const dealshareDelta = s.dealshare_elo - initial;

  // Compute confusion accuracy
  function accuracy(cat) {
    const a = (cm[cat] || {}).approved || 0;
    const r = (cm[cat] || {}).rejected || 0;
    const total = a + r;
    const correct = cat === 'good' ? a : r;
    return total > 0 ? Math.round(correct / total * 100) : null;
  }

  // Find matches this model participated in
  const modelMatches = data.matches.filter(m =>
    m.results.some(r => r.model_id === modelId)
  ).reverse();

  let html = `
    <div class="view-header">
      <a class="back-link" onclick="router.navigate('leaderboard')">&larr; Back to leaderboard</a>
      <div class="model-header">
        <div class="model-header-avatar" style="background:${meta.color}">${meta.icon}</div>
        <div class="model-header-info">
          <h1>${s.display_name}</h1>
          <div class="model-id-tag">${s.model_id} &middot; ${s.matches_played} matches</div>
        </div>
      </div>
    </div>

    <div class="elo-cards">
      <div class="elo-card profit">
        <div class="elo-card-label">Profit Elo</div>
        <div class="elo-card-value">${elo(s.profit_elo)}</div>
        <div class="elo-card-delta ${profitDelta >= 0 ? 'up' : 'down'}">${profitDelta >= 0 ? '\u25B2' : '\u25BC'} ${Math.abs(profitDelta).toFixed(0)} from start</div>
        <div class="elo-card-sparkline">${sparklineSVG(hist, 'profit_elo', 180, 30, 'var(--blue-400)')}</div>
      </div>
      <div class="elo-card credit">
        <div class="elo-card-label">Credit Elo</div>
        <div class="elo-card-value">${elo(s.credit_elo)}</div>
        <div class="elo-card-delta ${creditDelta >= 0 ? 'up' : 'down'}">${creditDelta >= 0 ? '\u25B2' : '\u25BC'} ${Math.abs(creditDelta).toFixed(0)} from start</div>
        <div class="elo-card-sparkline">${sparklineSVG(hist, 'credit_elo', 180, 30, 'var(--green-400)')}</div>
      </div>
      <div class="elo-card dealshare">
        <div class="elo-card-label">DealShare Elo</div>
        <div class="elo-card-value">${elo(s.dealshare_elo)}</div>
        <div class="elo-card-delta ${dealshareDelta >= 0 ? 'up' : 'down'}">${dealshareDelta >= 0 ? '\u25B2' : '\u25BC'} ${Math.abs(dealshareDelta).toFixed(0)} from start</div>
        <div class="elo-card-sparkline">${sparklineSVG(hist, 'dealshare_elo', 180, 30, 'var(--purple-400)')}</div>
      </div>
    </div>

    <div class="chart-section">
      <div class="chart-title">Elo Trajectory</div>
      <div class="chart-container" id="elo-chart-container">
        <div class="skeleton" style="height:220px"></div>
      </div>
    </div>

    <div class="grid-2 mb-xl">
      <div class="chart-section">
        <div class="chart-title">Confusion Matrix</div>
        <div class="cm-grid">
          <div class="cm-header"></div>
          <div class="cm-header">Approved</div>
          <div class="cm-header">Rejected</div>
          <div class="cm-header">Acc.</div>

          <div class="cm-row-label"><span class="badge badge-good">Good</span></div>
          <div class="cm-cell correct-strong">${cm.good?.approved || 0}</div>
          <div class="cm-cell incorrect-light">${cm.good?.rejected || 0}</div>
          <div class="cm-accuracy text-positive">${accuracy('good') != null ? accuracy('good') + '%' : '\u2014'}</div>

          <div class="cm-row-label"><span class="badge badge-bad">Bad</span></div>
          <div class="cm-cell incorrect-strong">${cm.bad?.approved || 0}</div>
          <div class="cm-cell correct-strong">${cm.bad?.rejected || 0}</div>
          <div class="cm-accuracy text-positive">${accuracy('bad') != null ? accuracy('bad') + '%' : '\u2014'}</div>

          <div class="cm-row-label"><span class="badge badge-fraud">Fraud</span></div>
          <div class="cm-cell incorrect-strong">${cm.fraud?.approved || 0}</div>
          <div class="cm-cell correct-strong">${cm.fraud?.rejected || 0}</div>
          <div class="cm-accuracy text-positive">${accuracy('fraud') != null ? accuracy('fraud') + '%' : '\u2014'}</div>
        </div>
      </div>

      <div class="chart-section">
        <div class="chart-title">Rate Analysis</div>
        ${ra.avg_rate_good != null || ra.avg_rate_bad != null ? `
          <div class="rate-bars">
            ${ra.avg_rate_good != null ? `
              <div class="rate-bar-item">
                <div class="rate-bar-label">Good borrowers</div>
                <div class="rate-bar-track">
                  <div class="rate-bar-fill good" style="width:${Math.min(ra.avg_rate_good * 100 / 0.20 * 100, 100)}%">
                    ${(ra.avg_rate_good * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div style="font-size:0.8rem;color:var(--text-muted);margin-left:136px">${ra.n_good_offers} offers</div>
            ` : ''}
            ${ra.avg_rate_bad != null ? `
              <div class="rate-bar-item mt-md">
                <div class="rate-bar-label">Bad/Fraud borr.</div>
                <div class="rate-bar-track">
                  <div class="rate-bar-fill bad" style="width:${Math.min(ra.avg_rate_bad * 100 / 0.20 * 100, 100)}%">
                    ${(ra.avg_rate_bad * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div style="font-size:0.8rem;color:var(--text-muted);margin-left:136px">${ra.n_bad_offers} offers</div>
            ` : ''}
            ${ra.avg_rate_good != null && ra.avg_rate_bad != null ? `
              <div class="mt-md" style="font-size:0.9rem">
                <span style="color:var(--text-muted)">Spread:</span>
                <span class="text-mono" style="font-weight:600;color:${(ra.avg_rate_bad - ra.avg_rate_good) > 0 ? 'var(--green-400)' : 'var(--red-400)'}">
                  ${((ra.avg_rate_bad - ra.avg_rate_good) * 100).toFixed(1)}pp
                </span>
              </div>
            ` : ''}
          </div>
        ` : '<div class="text-muted">No rate data available</div>'}
      </div>
    </div>

    <div class="chart-section">
      <div class="chart-title">Match History</div>
      <div class="data-table-wrapper" style="border:none">
        <table class="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Date</th>
              <th>Mix</th>
              <th>Opponents</th>
              <th>RAROC</th>
              <th>Deals Won</th>
              <th>Frauds</th>
              <th>P Elo</th>
              <th>Delta</th>
            </tr>
          </thead>
          <tbody>
            ${modelMatches.map((match, idx) => {
              const r = match.results.find(r => r.model_id === modelId);
              if (!r) return '';
              const histEntry = hist[modelMatches.length - 1 - idx];
              const prevEntry = hist[modelMatches.length - 2 - idx];
              const eloVal = histEntry ? histEntry.profit_elo : null;
              const prevElo = prevEntry ? prevEntry.profit_elo : initial;
              const delta = eloVal != null ? eloVal - prevElo : null;
              const opponents = match.results.filter(rr => rr.model_id !== modelId).map(rr => shortModelName(rr.model_id)).join(', ');

              return `
                <tr onclick="router.navigate('match', {id: '${match.match_id}'})">
                  <td class="rank-cell">${modelMatches.length - idx}</td>
                  <td>${formatDate(match.timestamp_utc)}</td>
                  <td><span class="badge badge-mix">${match.mix}</span></td>
                  <td style="font-size:0.8rem">${opponents}</td>
                  <td class="raroc-cell ${r.raroc_score >= 0 ? 'text-positive' : 'text-negative'}">${r.raroc_score >= 0 ? '+' : ''}${r.raroc_score.toFixed(1)}%</td>
                  <td class="text-mono">${r.deals_won}</td>
                  <td class="${r.frauds_funded > 0 ? 'text-negative' : 'text-muted'} text-mono">${r.frauds_funded}</td>
                  <td class="elo-cell ${eloVal ? eloColorClass(eloVal) : ''}">${eloVal ? elo(eloVal) : '\u2014'}</td>
                  <td class="text-mono ${delta != null ? (delta >= 0 ? 'text-positive' : 'text-negative') : ''}">${delta != null ? (delta >= 0 ? '+' : '') + delta.toFixed(0) : '\u2014'}</td>
                </tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;

  return html;
};

// --- Match List ---
views.matchList = function () {
  const matches = [...data.matches].reverse();

  let html = `
    <h1 class="page-title">Match History</h1>
    <p class="page-subtitle">${matches.length} matches played</p>

    <div class="match-list">
      ${matches.map((match, idx) => {
        const sorted = [...match.results].sort((a, b) => (b.raroc_score || 0) - (a.raroc_score || 0));
        const winner = sorted[0];
        const winnerMeta = getModelMeta(winner.model_id);

        return `
          <div class="match-list-item" onclick="router.navigate('match', {id: '${match.match_id}'})">
            <div style="display:flex;align-items:center;gap:var(--space-md)">
              <div class="match-list-avatars">
                ${match.results.map(r => {
                  const meta = getModelMeta(r.model_id);
                  return `<div class="model-avatar" style="background:${meta.color};width:28px;height:28px;font-size:0.55rem">${meta.icon}</div>`;
                }).join('')}
              </div>
              <div class="match-list-info">
                <h3>${formatDate(match.timestamp_utc)}</h3>
                <p>${match.mix} mix &middot; ${match.n_borrowers} borrowers</p>
              </div>
            </div>
            <div class="match-list-winner">
              <div class="match-list-winner-label">Winner</div>
              <div class="match-list-winner-name">${shortModelName(winner.model_id)}</div>
            </div>
            <div class="match-list-arrow">&rsaquo;</div>
          </div>`;
      }).join('')}
    </div>
  `;

  return html;
};

// --- Match Detail ---
views.matchDetail = function (matchId) {
  const match = data.matches.find(m => m.match_id === matchId);
  if (!match) return '<div class="empty-state"><div class="empty-state-icon">\u2753</div><div class="empty-state-text">Match not found</div></div>';

  const sorted = [...match.results].sort((a, b) => (b.raroc_score || 0) - (a.raroc_score || 0));
  const winnerModelId = sorted[0].model_id;

  // Collect all borrower IDs
  const borrowerIds = new Set();
  match.results.forEach(r => {
    Object.keys(r.per_borrower || {}).forEach(bid => borrowerIds.add(bid));
  });
  const bids = [...borrowerIds].sort();

  // Build borrower info map
  const borrowerInfo = {};
  bids.forEach(bid => {
    const b = data.borrowers.find(b => b.id === bid);
    borrowerInfo[bid] = b || { name: bid, sector: '', outcome: 'unknown' };
  });

  let html = `
    <div class="view-header">
      <a class="back-link" onclick="router.navigate('matches')">&larr; Back to matches</a>
      <div class="match-header-bar">
        <div>
          <h1 class="page-title">Match Detail</h1>
          <div class="page-subtitle" style="margin-bottom:0">
            ${formatDate(match.timestamp_utc)} ${formatTime(match.timestamp_utc)} &middot;
            <span class="badge badge-mix">${match.mix}</span>
            <span class="badge badge-count">${match.n_borrowers} borrowers</span>
          </div>
        </div>
      </div>
    </div>

    <div class="match-model-cards">
      ${sorted.map((r, i) => {
        const meta = getModelMeta(r.model_id);
        const isWinner = i === 0;
        const cm = r.confusion_matrix || {};

        return `
          <div class="match-model-card ${isWinner ? 'winner' : ''}" onclick="router.navigate('model', {id: '${r.model_id}'})">
            ${isWinner ? '<div class="winner-badge">\u2B50 Winner</div>' : ''}
            <div style="display:flex;align-items:center;gap:var(--space-sm);margin-bottom:var(--space-md)">
              <div class="model-avatar" style="background:${meta.color}">${meta.icon}</div>
              <div>
                <div class="match-card-model-name">${shortModelName(r.model_id)}</div>
                <div class="match-card-model-id">${r.model_id}</div>
              </div>
            </div>
            <div class="match-card-stats">
              <div class="match-stat">
                <span class="match-stat-label">RAROC</span>
                <span class="match-stat-value ${r.raroc_score >= 0 ? 'text-positive' : 'text-negative'}">${r.raroc_score >= 0 ? '+' : ''}${r.raroc_score.toFixed(1)}%</span>
              </div>
              <div class="match-stat">
                <span class="match-stat-label">Net P&L</span>
                <span class="match-stat-value ${r.net_pnl >= 0 ? 'text-positive' : 'text-negative'}">${dollar(r.net_pnl)}</span>
              </div>
              <div class="match-stat">
                <span class="match-stat-label">Deals Won</span>
                <span class="match-stat-value">${r.deals_won}</span>
              </div>
              <div class="match-stat">
                <span class="match-stat-label">Deployed</span>
                <span class="match-stat-value">${dollar(r.deployed)}</span>
              </div>
              <div class="match-stat">
                <span class="match-stat-label">Frauds</span>
                <span class="match-stat-value ${r.frauds_funded > 0 ? 'text-negative' : ''}">${r.frauds_funded}</span>
              </div>
              <div class="match-stat">
                <span class="match-stat-label">Defaults</span>
                <span class="match-stat-value ${r.defaults > 0 ? 'text-negative' : ''}">${r.defaults}</span>
              </div>
            </div>
          </div>`;
      }).join('')}
    </div>

    <div class="chart-section">
      <div class="chart-title">Borrower-by-Borrower Decisions</div>
      <div class="borrower-table-wrapper" style="border:none">
        <table class="borrower-table">
          <thead>
            <tr>
              <th>Borrower</th>
              <th>Sector</th>
              <th>Truth</th>
              <th>Amount</th>
              ${match.results.map(r => `<th>${shortModelName(r.model_id)}</th>`).join('')}
              <th>Winner</th>
            </tr>
          </thead>
          <tbody>
            ${bids.map(bid => {
              const bInfo = borrowerInfo[bid];
              const gt = match.results[0]?.per_borrower?.[bid]?.ground_truth || bInfo.outcome || '?';

              // Find winner for this borrower
              let winnerModel = null;
              match.results.forEach(r => {
                const bd = r.per_borrower?.[bid];
                if (bd && bd.decision_state === 'won') winnerModel = r.model_id;
              });

              return `
                <tr>
                  <td>
                    <div class="borrower-name">${bInfo.name || bid}</div>
                  </td>
                  <td class="borrower-sector">${bInfo.sector || ''}</td>
                  <td><span class="badge badge-${gt}">${gt}</span></td>
                  <td class="text-mono" style="font-size:0.8rem">${bInfo.loan_amount ? dollar(bInfo.loan_amount) : '\u2014'}</td>
                  ${match.results.map(r => {
                    const bd = r.per_borrower?.[bid];
                    if (!bd) return '<td>\u2014</td>';

                    const approved = bd.decision_state === 'won' || bd.decision_state === 'lost';
                    const correct = (gt === 'good' && approved) || (gt !== 'good' && !approved);
                    const isWinner = r.model_id === winnerModel;

                    let cellClass, content;
                    if (bd.decision_state === 'declined') {
                      cellClass = correct ? 'declined-correct' : 'declined-incorrect';
                      content = 'DECLINE';
                    } else {
                      cellClass = correct ? 'approved-correct' : 'approved-incorrect';
                      const rateStr = bd.rate_offered != null ? (bd.rate_offered * 100).toFixed(1) + '%' : '';
                      content = `<span class="rate">${rateStr}</span>${isWinner ? '<span class="star">\u2605</span>' : ''}`;
                    }

                    return `<td><div class="decision-cell ${cellClass}">${content}</div></td>`;
                  }).join('')}
                  <td>
                    ${winnerModel
                      ? `<span style="color:var(--gold-400);font-weight:600;font-size:0.8rem">${shortModelName(winnerModel)}</span>`
                      : '<span class="text-muted">\u2014</span>'}
                  </td>
                </tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>
    </div>

    <div class="grid-3 mt-xl">
      ${match.results.map(r => {
        const meta = getModelMeta(r.model_id);
        const cm = r.confusion_matrix || {};

        function acc(cat) {
          const a = (cm[cat] || {}).approved || 0;
          const rej = (cm[cat] || {}).rejected || 0;
          const total = a + rej;
          const correct = cat === 'good' ? a : rej;
          return total > 0 ? Math.round(correct / total * 100) + '%' : '\u2014';
        }

        return `
          <div class="chart-section">
            <div class="chart-title" style="display:flex;align-items:center;gap:8px">
              <div class="model-avatar" style="background:${meta.color};width:24px;height:24px;font-size:0.5rem">${meta.icon}</div>
              ${shortModelName(r.model_id)} — Confusion Matrix
            </div>
            <div class="cm-grid" style="max-width:300px">
              <div class="cm-header"></div>
              <div class="cm-header">Appr</div>
              <div class="cm-header">Rej</div>
              <div class="cm-header">Acc</div>
              <div class="cm-row-label"><span class="badge badge-good">Good</span></div>
              <div class="cm-cell correct-strong">${cm.good?.approved || 0}</div>
              <div class="cm-cell incorrect-light">${cm.good?.rejected || 0}</div>
              <div class="cm-accuracy text-positive">${acc('good')}</div>
              <div class="cm-row-label"><span class="badge badge-bad">Bad</span></div>
              <div class="cm-cell incorrect-strong">${cm.bad?.approved || 0}</div>
              <div class="cm-cell correct-strong">${cm.bad?.rejected || 0}</div>
              <div class="cm-accuracy text-positive">${acc('bad')}</div>
              <div class="cm-row-label"><span class="badge badge-fraud">Fraud</span></div>
              <div class="cm-cell incorrect-strong">${cm.fraud?.approved || 0}</div>
              <div class="cm-cell correct-strong">${cm.fraud?.rejected || 0}</div>
              <div class="cm-accuracy text-positive">${acc('fraud')}</div>
            </div>
          </div>`;
      }).join('')}
    </div>
  `;

  return html;
};

// --- Compare View ---
views.compare = function () {
  const s = leaderboard.standings;

  let html = `
    <h1 class="page-title">Head-to-Head Compare</h1>
    <p class="page-subtitle">Compare two models side by side</p>

    <div class="compare-selector">
      <label>Model A:</label>
      <select id="compare-a">
        ${s.map((m, i) => `<option value="${m.model_id}" ${i === 0 ? 'selected' : ''}>${m.display_name}</option>`).join('')}
      </select>
      <span style="font-family:var(--font-display);font-size:1.2rem;color:var(--text-muted);padding:0 8px">vs</span>
      <label>Model B:</label>
      <select id="compare-b">
        ${s.map((m, i) => `<option value="${m.model_id}" ${i === 1 ? 'selected' : ''}>${m.display_name}</option>`).join('')}
      </select>
      <button class="elo-toggle active" onclick="updateCompare()">Compare</button>
    </div>

    <div id="compare-content"></div>
  `;

  return html;
};

function setupCompareHandlers() {
  setTimeout(() => updateCompare(), 50);
}

function updateCompare() {
  const aId = document.getElementById('compare-a')?.value;
  const bId = document.getElementById('compare-b')?.value;
  const container = document.getElementById('compare-content');
  if (!aId || !bId || !container) return;

  const a = leaderboard.standings.find(s => s.model_id === aId);
  const b = leaderboard.standings.find(s => s.model_id === bId);
  if (!a || !b) return;

  const metaA = getModelMeta(aId);
  const metaB = getModelMeta(bId);

  function compareBar(label, aVal, bVal, suffix = '', higher = true) {
    const max = Math.max(Math.abs(aVal), Math.abs(bVal), 1);
    const aPct = Math.abs(aVal) / max * 100;
    const bPct = Math.abs(bVal) / max * 100;
    const aWins = higher ? aVal > bVal : aVal < bVal;
    const bWins = higher ? bVal > aVal : bVal < aVal;

    return `
      <div style="margin-bottom:var(--space-lg)">
        <div class="flex-between mb-md">
          <span class="text-mono" style="font-weight:600;${aWins ? 'color:var(--gold-400)' : ''}">${typeof aVal === 'number' ? (aVal >= 0 && suffix !== '%' ? '' : '') : ''}${aVal}${suffix}</span>
          <span style="font-size:0.8rem;color:var(--text-muted);font-weight:500">${label}</span>
          <span class="text-mono" style="font-weight:600;${bWins ? 'color:var(--gold-400)' : ''}">${typeof bVal === 'number' ? (bVal >= 0 && suffix !== '%' ? '' : '') : ''}${bVal}${suffix}</span>
        </div>
        <div style="display:flex;gap:4px;height:8px">
          <div style="flex:1;display:flex;justify-content:flex-end">
            <div style="width:${aPct}%;background:${metaA.color};border-radius:4px;transition:width 0.5s var(--ease-out)"></div>
          </div>
          <div style="flex:1">
            <div style="width:${bPct}%;background:${metaB.color};border-radius:4px;transition:width 0.5s var(--ease-out)"></div>
          </div>
        </div>
      </div>`;
  }

  const cmA = a.confusion_agg || {};
  const cmB = b.confusion_agg || {};
  const totalCorrectA = (cmA.good?.approved || 0) + (cmA.bad?.rejected || 0) + (cmA.fraud?.rejected || 0);
  const totalCorrectB = (cmB.good?.approved || 0) + (cmB.bad?.rejected || 0) + (cmB.fraud?.rejected || 0);
  const totalA = Object.values(cmA).reduce((s, c) => s + (c.approved || 0) + (c.rejected || 0), 0);
  const totalB = Object.values(cmB).reduce((s, c) => s + (c.approved || 0) + (c.rejected || 0), 0);
  const accA = totalA > 0 ? Math.round(totalCorrectA / totalA * 100) : 0;
  const accB = totalB > 0 ? Math.round(totalCorrectB / totalB * 100) : 0;

  container.innerHTML = `
    <div class="compare-vs">
      <div style="text-align:center">
        <div class="model-avatar" style="background:${metaA.color};width:48px;height:48px;font-size:0.9rem;margin:0 auto var(--space-sm)">${metaA.icon}</div>
        <div style="font-weight:600">${a.display_name}</div>
      </div>
      <div class="compare-vs-divider">VS</div>
      <div style="text-align:center">
        <div class="model-avatar" style="background:${metaB.color};width:48px;height:48px;font-size:0.9rem;margin:0 auto var(--space-sm)">${metaB.icon}</div>
        <div style="font-weight:600">${b.display_name}</div>
      </div>
    </div>

    <div class="chart-section">
      <div class="chart-title">Elo Ratings</div>
      ${compareBar('Profit Elo', elo(a.profit_elo), elo(b.profit_elo))}
      ${compareBar('Credit Elo', elo(a.credit_elo), elo(b.credit_elo))}
      ${compareBar('DealShare Elo', elo(a.dealshare_elo), elo(b.dealshare_elo))}
    </div>

    <div class="chart-section">
      <div class="chart-title">Performance</div>
      ${compareBar('Avg RAROC', a.avg_raroc.toFixed(1), b.avg_raroc.toFixed(1), '%')}
      ${compareBar('Matches Played', a.matches_played, b.matches_played)}
      ${compareBar('Overall Accuracy', accA, accB, '%')}
      ${compareBar('Frauds Approved', cmA.fraud?.approved || 0, cmB.fraud?.approved || 0, '', false)}
    </div>

    <div class="grid-2">
      <div class="chart-section">
        <div class="chart-title" style="display:flex;align-items:center;gap:8px">
          <div class="model-avatar" style="background:${metaA.color};width:20px;height:20px;font-size:0.45rem">${metaA.icon}</div>
          ${a.display_name}
        </div>
        <div class="cm-grid" style="max-width:300px">
          <div class="cm-header"></div>
          <div class="cm-header">Appr</div>
          <div class="cm-header">Rej</div>
          <div class="cm-header"></div>
          <div class="cm-row-label"><span class="badge badge-good">Good</span></div>
          <div class="cm-cell correct-strong">${cmA.good?.approved || 0}</div>
          <div class="cm-cell incorrect-light">${cmA.good?.rejected || 0}</div>
          <div class="cm-accuracy"></div>
          <div class="cm-row-label"><span class="badge badge-bad">Bad</span></div>
          <div class="cm-cell incorrect-strong">${cmA.bad?.approved || 0}</div>
          <div class="cm-cell correct-strong">${cmA.bad?.rejected || 0}</div>
          <div class="cm-accuracy"></div>
          <div class="cm-row-label"><span class="badge badge-fraud">Fraud</span></div>
          <div class="cm-cell incorrect-strong">${cmA.fraud?.approved || 0}</div>
          <div class="cm-cell correct-strong">${cmA.fraud?.rejected || 0}</div>
          <div class="cm-accuracy"></div>
        </div>
      </div>
      <div class="chart-section">
        <div class="chart-title" style="display:flex;align-items:center;gap:8px">
          <div class="model-avatar" style="background:${metaB.color};width:20px;height:20px;font-size:0.45rem">${metaB.icon}</div>
          ${b.display_name}
        </div>
        <div class="cm-grid" style="max-width:300px">
          <div class="cm-header"></div>
          <div class="cm-header">Appr</div>
          <div class="cm-header">Rej</div>
          <div class="cm-header"></div>
          <div class="cm-row-label"><span class="badge badge-good">Good</span></div>
          <div class="cm-cell correct-strong">${cmB.good?.approved || 0}</div>
          <div class="cm-cell incorrect-light">${cmB.good?.rejected || 0}</div>
          <div class="cm-accuracy"></div>
          <div class="cm-row-label"><span class="badge badge-bad">Bad</span></div>
          <div class="cm-cell incorrect-strong">${cmB.bad?.approved || 0}</div>
          <div class="cm-cell correct-strong">${cmB.bad?.rejected || 0}</div>
          <div class="cm-accuracy"></div>
          <div class="cm-row-label"><span class="badge badge-fraud">Fraud</span></div>
          <div class="cm-cell incorrect-strong">${cmB.fraud?.approved || 0}</div>
          <div class="cm-cell correct-strong">${cmB.fraud?.rejected || 0}</div>
          <div class="cm-accuracy"></div>
        </div>
      </div>
    </div>
  `;
}

// --- Leaderboard Sorting ---
function sortLeaderboard(key) {
  if (currentSort.key === key) {
    currentSort.dir = currentSort.dir === 'desc' ? 'asc' : 'desc';
  } else {
    currentSort.key = key;
    currentSort.dir = 'desc';
  }
  router.render();
}

// ============================================================================
// Initialization
// ============================================================================
function init() {
  // Load data
  if (typeof SAMPLE_DATA !== 'undefined') {
    data = SAMPLE_DATA;
    leaderboard = data.leaderboard;
  }

  // Update meta counters
  document.getElementById('meta-matches').textContent = leaderboard ? leaderboard.n_matches : 0;
  document.getElementById('meta-models').textContent = leaderboard ? leaderboard.standings.length : 0;

  // Dismiss loading screen
  const loadingScreen = document.getElementById('loading-screen');
  const app = document.getElementById('app');

  setTimeout(() => {
    loadingScreen.classList.add('fade-out');
    app.classList.remove('hidden');

    setTimeout(() => {
      loadingScreen.style.display = 'none';
    }, 600);

    // Start router
    router.init();
  }, 800);
}

// Boot
document.addEventListener('DOMContentLoaded', init);
