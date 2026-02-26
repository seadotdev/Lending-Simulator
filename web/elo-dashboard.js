/**
 * Elo Leaderboard Dashboard
 *
 * Renders Elo standings with sub-leaderboard filters (by lender, by match type).
 */

// ---- Helpers ----

function fmt(n, decimals = 1) {
  if (n == null) return '-';
  return Number(n).toFixed(decimals);
}

function fmtPnl(n) {
  if (n == null) return '-';
  const sign = n >= 0 ? '+' : '';
  return `${sign}$${Number(n).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

function eloClass(elo) {
  if (elo >= 1100) return 'positive';
  if (elo <= 900) return 'negative';
  return '';
}

function sparklineSvg(history, key, width = 80, height = 20) {
  if (!history || history.length < 2) return '';
  const vals = history.map(h => h[key] || 1000);
  const min = Math.min(...vals) - 5;
  const max = Math.max(...vals) + 5;
  const range = max - min || 1;
  const points = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="vertical-align:middle">
    <polyline points="${points}" fill="none" stroke="var(--gold)" stroke-width="1.5" stroke-linejoin="round"/>
  </svg>`;
}

// ---- Confusion Matrix ----

function renderConfusionMatrix(cm) {
  if (!cm) return '';
  const categories = ['good', 'bad', 'fraud'];
  let html = `<table class="confusion-table">
    <tr><th></th><th>Approved</th><th>Rejected</th></tr>`;
  for (const cat of categories) {
    const data = cm[cat] || {};
    html += `<tr>
      <th>${cat}</th>
      <td>${data.approved || 0}</td>
      <td>${data.rejected || 0}</td>
    </tr>`;
  }
  html += '</table>';
  return html;
}

// ---- Standings Table ----

function renderStandingsTable(standings, eloHistory) {
  if (!standings || !standings.length) {
    return '<div class="stats-empty">No standings data</div>';
  }

  let html = `<table class="stats-comparison-table elo-table">
    <thead>
      <tr>
        <th>#</th>
        <th style="text-align:left">Model</th>
        <th>Composite</th>
        <th>Profit</th>
        <th>Credit</th>
        <th>Dealshare</th>
        <th>Matches</th>
        <th>Avg P&L</th>
        <th>Trend</th>
      </tr>
    </thead>
    <tbody>`;

  standings.forEach((s, i) => {
    const hist = eloHistory ? eloHistory[s.model_id] : null;
    const spark = sparklineSvg(hist, 'profit_elo');
    html += `<tr>
      <td style="text-align:center;color:var(--gold);font-weight:700">${i + 1}</td>
      <td class="stats-table-name">
        ${s.display_name}
        <div class="stats-table-model">${s.model_id}</div>
      </td>
      <td class="${eloClass(s.composite_elo)}">${fmt(s.composite_elo)}</td>
      <td class="${eloClass(s.profit_elo)}">${fmt(s.profit_elo)}</td>
      <td class="${eloClass(s.credit_elo)}">${fmt(s.credit_elo)}</td>
      <td class="${eloClass(s.dealshare_elo)}">${fmt(s.dealshare_elo)}</td>
      <td style="text-align:center">${s.matches_played}${s.mock_matches ? ` <span style="color:var(--amber);font-size:9px" title="${s.mock_matches} mock">(${s.mock_matches}m)</span>` : ''}</td>
      <td class="${(s.avg_net_pnl || 0) >= 0 ? 'positive' : 'negative'}">${fmtPnl(s.avg_net_pnl)}</td>
      <td>${spark}</td>
    </tr>`;
  });

  html += '</tbody></table>';
  return html;
}

// ---- Match History ----

function renderMatchHistory(matchIds) {
  if (!matchIds || !matchIds.length) return '';
  let html = `<div class="stats-section">
    <div class="stats-section-title">Match History (${matchIds.length} matches)</div>
    <div class="elo-match-list">`;
  for (const mid of matchIds.slice().reverse()) {
    const ts = mid.replace(/_[a-f0-9]+$/, '').replace(/T/, ' ').replace(/-/g, (m, offset) => offset > 9 ? ':' : '-');
    html += `<div class="elo-match-item">${ts}</div>`;
  }
  html += '</div></div>';
  return html;
}

// ---- Filter Bar ----

function renderFilterBar(leaderboard, activeFilter, activeKey) {
  const subLb = leaderboard.sub_leaderboards || {};
  const byLender = subLb.by_lender || {};
  const byMatchType = subLb.by_match_type || {};

  let html = '<div class="elo-filter-bar">';
  html += `<button class="elo-filter-btn ${activeFilter === 'overall' ? 'active' : ''}" data-filter="overall">Overall</button>`;

  // By Lender buttons
  for (const key of Object.keys(byLender)) {
    const isActive = activeFilter === 'by_lender' && activeKey === key;
    html += `<button class="elo-filter-btn ${isActive ? 'active' : ''}" data-filter="by_lender" data-key="${key}">${key}</button>`;
  }

  // By Match Type buttons
  for (const key of Object.keys(byMatchType)) {
    const label = key.charAt(0).toUpperCase() + key.slice(1);
    const isActive = activeFilter === 'by_match_type' && activeKey === key;
    html += `<button class="elo-filter-btn ${isActive ? 'active' : ''}" data-filter="by_match_type" data-key="${label}">${label}</button>`;
  }

  html += '</div>';
  return html;
}

// ---- Main Export ----

export function renderElo(container, leaderboard) {
  let activeFilter = 'overall';
  let activeKey = '';

  function render() {
    let standings, eloHistory;

    if (activeFilter === 'overall') {
      standings = leaderboard.standings;
      eloHistory = leaderboard.elo_history;
    } else {
      const subLb = leaderboard.sub_leaderboards || {};
      const group = subLb[activeFilter] || {};
      standings = group[activeKey] || [];
      eloHistory = null;
    }

    let html = `<div class="stats-content">`;

    // Header
    html += `<div class="stats-section">
      <div class="stats-section-title">Elo Leaderboard</div>
      <div class="stats-summary-grid">
        <div class="stats-summary-card">
          <div class="stats-summary-label">Models</div>
          <div class="stats-summary-value">${(standings || []).length}</div>
        </div>
        <div class="stats-summary-card">
          <div class="stats-summary-label">Matches</div>
          <div class="stats-summary-value">${leaderboard.n_matches || 0}</div>
        </div>
        <div class="stats-summary-card">
          <div class="stats-summary-label">K Factor</div>
          <div class="stats-summary-value">${leaderboard.config?.k || '-'}</div>
        </div>
        <div class="stats-summary-card">
          <div class="stats-summary-label">Computed</div>
          <div class="stats-summary-value" style="font-size:11px">${(leaderboard.computed_at || '').replace('T', ' ').replace('Z', '')}</div>
        </div>
      </div>
    </div>`;

    // Filter bar
    html += renderFilterBar(leaderboard, activeFilter, activeKey);

    // Standings table
    html += `<div class="stats-section">`;
    html += renderStandingsTable(standings, eloHistory);
    html += '</div>';

    // Confusion matrices (only for overall view)
    if (activeFilter === 'overall' && standings) {
      const withCm = standings.filter(s => s.confusion_agg && Object.keys(s.confusion_agg).length);
      if (withCm.length) {
        html += `<div class="stats-section">
          <div class="stats-section-title">Aggregate Confusion Matrices</div>
          <div class="elo-confusion-grid">`;
        for (const s of withCm) {
          html += `<div class="stats-confusion-block">
            <div class="stats-confusion-title">${s.display_name}</div>
            ${renderConfusionMatrix(s.confusion_agg)}
          </div>`;
        }
        html += '</div></div>';
      }
    }

    // Match history
    if (activeFilter === 'overall') {
      html += renderMatchHistory(leaderboard.match_ids);
    }

    html += '</div>';
    container.innerHTML = html;

    // Bind filter buttons
    container.querySelectorAll('.elo-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const filter = btn.dataset.filter;
        const key = btn.dataset.key || '';
        if (filter === 'overall') {
          activeFilter = 'overall';
          activeKey = '';
        } else {
          activeFilter = filter;
          // Resolve case: button label may be capitalized but key is lowercase
          const subLb = leaderboard.sub_leaderboards || {};
          const group = subLb[filter] || {};
          activeKey = group[key] ? key : Object.keys(group).find(k => k.toLowerCase() === key.toLowerCase()) || key;
        }
        render();
      });
    });
  }

  render();
}
