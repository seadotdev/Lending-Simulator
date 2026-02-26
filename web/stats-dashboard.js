/**
 * Stats Dashboard — Aggregate analytics tab for season data.
 *
 * Renders: radar charts, confusion heatmap, learning curves,
 * rate analysis, and model comparison tables.
 * Pure SVG/HTML — no external dependencies.
 */

// ---- SVG Helpers ----

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function polarToCart(cx, cy, r, angleDeg) {
  const rad = (angleDeg - 90) * Math.PI / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

// ---- Colors ----

const CHART_COLORS = [
  '#e84a4a', '#4ae84a', '#4a90d9', '#e8a838', '#9a6ae8',
  '#e84ab0', '#4ae8d0', '#d9904a', '#6a9ae8', '#b8e84a',
];

function esc(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function fmtNum(n) {
  return Math.round(n).toLocaleString('en-US');
}

// ---- Main Render ----

export function renderStats(container, season) {
  container.innerHTML = '';
  if (!season || !season.weeks?.length) {
    container.innerHTML = '<div class="stats-empty">No season data loaded</div>';
    return;
  }

  const lenders = season.lenders;
  const weeks = season.weeks;
  const lastWeek = weeks[weeks.length - 1];

  // Build sections
  container.appendChild(buildSummaryCards(lenders, weeks));
  container.appendChild(buildRadarChart(lenders, weeks));

  const row = document.createElement('div');
  row.className = 'stats-row';
  row.appendChild(buildConfusionHeatmap(lenders, weeks));
  row.appendChild(buildRateAnalysis(lenders, weeks));
  container.appendChild(row);

  container.appendChild(buildLearningCurves(lenders, weeks));
  container.appendChild(buildCostAnalysis(lenders, weeks));
  container.appendChild(buildModelComparisonTable(lenders, weeks));
}

// ---- Summary Cards ----

function buildSummaryCards(lenders, weeks) {
  const section = document.createElement('div');
  section.className = 'stats-section';

  const totalDeals = lenders.reduce((s, l) => s + l.deals_won, 0);
  const totalDefaults = lenders.reduce((s, l) => s + (l.defaults || 0), 0);
  const totalFrauds = lenders.reduce((s, l) => s + (l.frauds_funded || 0), 0);
  const totalPnl = lenders.reduce((s, l) => s + l.net_pnl, 0);
  const totalBorrowers = weeks.reduce((s, w) => s + w.borrowers.length, 0);

  // Count cost data from decisions
  let totalCost = 0;
  let totalTokens = 0;
  for (const w of weeks) {
    for (const d of w.decisions) {
      totalCost += d.cost_usd || 0;
      totalTokens += (d.tokens_in || 0) + (d.tokens_out || 0);
    }
  }

  const cards = [
    { label: 'Weeks', value: weeks.length, cls: '' },
    { label: 'Borrowers', value: totalBorrowers, cls: '' },
    { label: 'Deals Booked', value: totalDeals, cls: '' },
    { label: 'Defaults', value: totalDefaults, cls: 'negative' },
    { label: 'Frauds Funded', value: totalFrauds, cls: 'negative' },
    { label: 'Total P&L', value: `$${fmtNum(totalPnl)}`, cls: totalPnl >= 0 ? 'positive' : 'negative' },
  ];

  if (totalCost > 0) {
    cards.push({ label: 'LLM Cost', value: `$${totalCost.toFixed(2)}`, cls: '' });
  }
  if (totalTokens > 0) {
    cards.push({ label: 'Total Tokens', value: fmtNum(totalTokens), cls: '' });
  }

  const grid = document.createElement('div');
  grid.className = 'stats-summary-grid';
  for (const c of cards) {
    const card = document.createElement('div');
    card.className = 'stats-summary-card';
    card.innerHTML = `
      <div class="stats-summary-label">${c.label}</div>
      <div class="stats-summary-value ${c.cls}">${c.value}</div>
    `;
    grid.appendChild(card);
  }
  section.appendChild(grid);
  return section;
}

// ---- Radar Chart ----

function buildRadarChart(lenders, weeks) {
  const section = document.createElement('div');
  section.className = 'stats-section';
  section.innerHTML = '<h3 class="stats-section-title">Model Comparison — Radar</h3>';

  const lastWeek = weeks[weeks.length - 1];
  const maxCapital = Math.max(...lenders.map(l => l.total_capital || 1));

  // Compute per-lender normalized metrics (0-1)
  const metrics = ['Win Rate', 'Profit ROI', 'Fraud Detection', 'Deploy Rate', 'Selectivity'];
  const lenderData = lenders.map(l => {
    const snap = lastWeek.lender_snapshots[l.id] || l;
    const totalDecisions = (snap.deals_won || 0) + (snap.deals_rejected || 0) + (snap.deals_lost || 0);
    const winRate = totalDecisions > 0 ? (snap.deals_won || 0) / totalDecisions : 0;
    const deployed = snap.deployed || 0;
    const roi = deployed > 0 ? Math.max(0, snap.net_pnl / deployed) : 0;
    const fraudsAvoided = 1 - ((snap.frauds_funded || 0) / Math.max(1, totalFraudsInSeason(weeks)));
    const deployRate = (l.total_capital || 1) > 0 ? deployed / (l.total_capital || 1) : 0;
    const selectivity = totalDecisions > 0 ? (snap.deals_rejected || 0) / totalDecisions : 0;

    return {
      name: l.name,
      values: [
        Math.min(1, winRate),
        Math.min(1, roi * 5),  // scale ROI up so it's visible
        Math.min(1, fraudsAvoided),
        Math.min(1, deployRate),
        Math.min(1, selectivity),
      ],
    };
  });

  // Build SVG
  const size = 300;
  const cx = size / 2, cy = size / 2, maxR = 110;
  const svg = svgEl('svg', { width: size, height: size, viewBox: `0 0 ${size} ${size}` });
  svg.style.display = 'block';
  svg.style.margin = '0 auto';

  // Grid rings
  for (let ring = 1; ring <= 4; ring++) {
    const r = (ring / 4) * maxR;
    const points = metrics.map((_, i) => {
      const p = polarToCart(cx, cy, r, (360 / metrics.length) * i);
      return `${p.x},${p.y}`;
    }).join(' ');
    svg.appendChild(svgEl('polygon', {
      points, fill: 'none', stroke: '#2a3a55', 'stroke-width': 0.5,
    }));
  }

  // Axes + labels
  metrics.forEach((m, i) => {
    const angle = (360 / metrics.length) * i;
    const end = polarToCart(cx, cy, maxR, angle);
    svg.appendChild(svgEl('line', {
      x1: cx, y1: cy, x2: end.x, y2: end.y,
      stroke: '#2a3a55', 'stroke-width': 0.5,
    }));
    const labelPos = polarToCart(cx, cy, maxR + 18, angle);
    const text = svgEl('text', {
      x: labelPos.x, y: labelPos.y,
      'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: '#6b7080', 'font-size': '9', 'font-family': 'Inter, sans-serif',
    });
    text.textContent = m;
    svg.appendChild(text);
  });

  // Data polygons
  lenderData.forEach((ld, li) => {
    const color = CHART_COLORS[li % CHART_COLORS.length];
    const points = ld.values.map((v, i) => {
      const p = polarToCart(cx, cy, v * maxR, (360 / metrics.length) * i);
      return `${p.x},${p.y}`;
    }).join(' ');
    svg.appendChild(svgEl('polygon', {
      points, fill: color, 'fill-opacity': 0.15,
      stroke: color, 'stroke-width': 1.5,
    }));
  });

  // Legend
  const legend = document.createElement('div');
  legend.className = 'stats-legend';
  lenderData.forEach((ld, i) => {
    const item = document.createElement('span');
    item.className = 'stats-legend-item';
    item.innerHTML = `<span class="stats-legend-dot" style="background:${CHART_COLORS[i % CHART_COLORS.length]}"></span>${esc(ld.name)}`;
    legend.appendChild(item);
  });

  section.appendChild(svg);
  section.appendChild(legend);
  return section;
}

function totalFraudsInSeason(weeks) {
  let count = 0;
  for (const w of weeks) {
    for (const b of w.borrowers) {
      if (b.true_outcome === 'fraud') count++;
    }
  }
  return count;
}

// ---- Confusion Heatmap ----

function buildConfusionHeatmap(lenders, weeks) {
  const section = document.createElement('div');
  section.className = 'stats-section stats-half';
  section.innerHTML = '<h3 class="stats-section-title">Confusion Matrix</h3>';

  // Build per-lender confusion data
  for (const lender of lenders) {
    const matrix = { good: { approve: 0, reject: 0 }, bad: { approve: 0, reject: 0 }, fraud: { approve: 0, reject: 0 } };

    for (const w of weeks) {
      for (const d of w.decisions) {
        if (d.lender_id !== lender.id) continue;
        const b = w.borrowers.find(b => b.id === d.borrower_id);
        if (!b) continue;
        const outcome = b.true_outcome;
        const action = d.decision === 'APPROVE' ? 'approve' : 'reject';
        if (matrix[outcome]) matrix[outcome][action]++;
      }
    }

    const table = document.createElement('div');
    table.className = 'stats-confusion-block';
    table.innerHTML = `
      <div class="stats-confusion-title">${esc(lender.name)}</div>
      <table class="confusion-table">
        <tr><th></th><th>Approve</th><th>Reject</th></tr>
        <tr>
          <th>Good</th>
          <td class="confusion-good-approve">${matrix.good.approve}</td>
          <td class="confusion-good-reject">${matrix.good.reject}</td>
        </tr>
        <tr>
          <th>Bad</th>
          <td class="confusion-bad-approve">${matrix.bad.approve}</td>
          <td class="confusion-bad-reject">${matrix.bad.reject}</td>
        </tr>
        <tr>
          <th>Fraud</th>
          <td class="confusion-fraud-approve">${matrix.fraud.approve}</td>
          <td class="confusion-fraud-reject">${matrix.fraud.reject}</td>
        </tr>
      </table>
    `;

    // Color intensity
    const cells = table.querySelectorAll('td');
    cells.forEach(cell => {
      const val = parseInt(cell.textContent) || 0;
      if (cell.className.includes('good-approve') || cell.className.includes('bad-reject') || cell.className.includes('fraud-reject')) {
        cell.style.background = `rgba(74, 232, 74, ${Math.min(0.4, val * 0.08)})`;
      } else if (val > 0) {
        cell.style.background = `rgba(232, 74, 74, ${Math.min(0.4, val * 0.08)})`;
      }
    });

    section.appendChild(table);
  }

  return section;
}

// ---- Rate Analysis ----

function buildRateAnalysis(lenders, weeks) {
  const section = document.createElement('div');
  section.className = 'stats-section stats-half';
  section.innerHTML = '<h3 class="stats-section-title">Rate Analysis</h3>';

  for (const lender of lenders) {
    const rates = { good: [], bad: [], fraud: [] };

    for (const w of weeks) {
      for (const d of w.decisions) {
        if (d.lender_id !== lender.id || d.decision !== 'APPROVE' || !d.term_sheet) continue;
        const b = w.borrowers.find(b => b.id === d.borrower_id);
        if (!b) continue;
        const rate = d.term_sheet.rate;
        if (rate && rates[b.true_outcome]) {
          rates[b.true_outcome].push(rate * 100);
        }
      }
    }

    const avgRate = arr => arr.length > 0 ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : '—';

    const block = document.createElement('div');
    block.className = 'stats-rate-block';
    block.innerHTML = `
      <div class="stats-confusion-title">${esc(lender.name)}</div>
      <div class="stats-rate-row">
        <span class="stats-rate-label">Good:</span>
        <span class="stats-rate-value positive">${avgRate(rates.good)}%</span>
        <span class="stats-rate-count">(${rates.good.length} loans)</span>
      </div>
      <div class="stats-rate-row">
        <span class="stats-rate-label">Bad:</span>
        <span class="stats-rate-value negative">${avgRate(rates.bad)}%</span>
        <span class="stats-rate-count">(${rates.bad.length} loans)</span>
      </div>
      <div class="stats-rate-row">
        <span class="stats-rate-label">Fraud:</span>
        <span class="stats-rate-value negative">${avgRate(rates.fraud)}%</span>
        <span class="stats-rate-count">(${rates.fraud.length} loans)</span>
      </div>
    `;
    section.appendChild(block);
  }

  return section;
}

// ---- Learning Curves ----

function buildLearningCurves(lenders, weeks) {
  const section = document.createElement('div');
  section.className = 'stats-section';
  section.innerHTML = '<h3 class="stats-section-title">Learning Curves — Weekly Performance</h3>';

  if (weeks.length < 2) {
    section.innerHTML += '<div class="stats-empty">Need 2+ weeks for learning curves</div>';
    return section;
  }

  // Build three charts: P&L trajectory, Approval Rate, Default Rate
  const charts = [
    { title: 'Cumulative P&L', metric: (snap) => snap.net_pnl, format: v => `$${fmtNum(v)}` },
    { title: 'Approval Rate %', metric: (snap) => {
      const total = (snap.deals_won || 0) + (snap.deals_rejected || 0) + (snap.deals_lost || 0);
      return total > 0 ? ((snap.deals_won || 0) / total * 100) : 0;
    }, format: v => `${v.toFixed(0)}%` },
    { title: 'Deployment %', metric: (snap) => {
      const cap = snap.effective_capital || snap.total_capital || 1;
      return ((snap.deployed || 0) / cap) * 100;
    }, format: v => `${v.toFixed(0)}%` },
  ];

  for (const chart of charts) {
    const chartEl = buildLineChart(chart.title, lenders, weeks, chart.metric, chart.format);
    section.appendChild(chartEl);
  }

  return section;
}

function buildLineChart(title, lenders, weeks, metricFn, formatFn) {
  const container = document.createElement('div');
  container.className = 'stats-chart-container';

  const label = document.createElement('div');
  label.className = 'stats-chart-label';
  label.textContent = title;
  container.appendChild(label);

  const w = 600, h = 200, pad = { top: 20, right: 20, bottom: 30, left: 60 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // Compute series
  const series = lenders.map((l, li) => {
    const points = [];
    for (let wi = 0; wi < weeks.length; wi++) {
      const snap = weeks[wi].lender_snapshots[l.id];
      if (snap) {
        points.push({ x: wi, y: metricFn(snap) });
      }
    }
    return { name: l.name, color: CHART_COLORS[li % CHART_COLORS.length], points };
  });

  // Y range
  let yMin = Infinity, yMax = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
  }
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * 0.1;
  yMin -= yPad; yMax += yPad;

  const xScale = i => pad.left + (i / Math.max(1, weeks.length - 1)) * plotW;
  const yScale = v => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const svg = svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}` });
  svg.style.display = 'block';
  svg.style.maxWidth = '100%';

  // Grid lines
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const yVal = yMin + (yMax - yMin) * (i / yTicks);
    const y = yScale(yVal);
    svg.appendChild(svgEl('line', {
      x1: pad.left, y1: y, x2: w - pad.right, y2: y,
      stroke: '#2a3a55', 'stroke-width': 0.5,
    }));
    const text = svgEl('text', {
      x: pad.left - 6, y: y + 3,
      'text-anchor': 'end', fill: '#6b7080', 'font-size': '9', 'font-family': 'Inter, sans-serif',
    });
    text.textContent = formatFn(yVal);
    svg.appendChild(text);
  }

  // X axis labels
  for (let i = 0; i < weeks.length; i++) {
    if (weeks.length > 15 && i % 2 !== 0 && i !== weeks.length - 1) continue;
    const text = svgEl('text', {
      x: xScale(i), y: h - 6,
      'text-anchor': 'middle', fill: '#6b7080', 'font-size': '9', 'font-family': 'Inter, sans-serif',
    });
    text.textContent = `W${i + 1}`;
    svg.appendChild(text);
  }

  // Lines
  for (const s of series) {
    if (s.points.length < 2) continue;
    const d = s.points.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`).join(' ');
    svg.appendChild(svgEl('path', {
      d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round',
    }));
    // Dots
    for (const p of s.points) {
      svg.appendChild(svgEl('circle', {
        cx: xScale(p.x), cy: yScale(p.y), r: 3,
        fill: s.color, stroke: '#0a0e17', 'stroke-width': 1,
      }));
    }
  }

  container.appendChild(svg);

  // Legend
  const legend = document.createElement('div');
  legend.className = 'stats-legend';
  series.forEach(s => {
    const item = document.createElement('span');
    item.className = 'stats-legend-item';
    item.innerHTML = `<span class="stats-legend-dot" style="background:${s.color}"></span>${esc(s.name)}`;
    legend.appendChild(item);
  });
  container.appendChild(legend);

  return container;
}

// ---- Cost Analysis ----

function buildCostAnalysis(lenders, weeks) {
  const section = document.createElement('div');
  section.className = 'stats-section';
  section.innerHTML = '<h3 class="stats-section-title">LLM Cost Analysis</h3>';

  // Check if any cost data exists
  let hasCostData = false;
  for (const w of weeks) {
    for (const d of w.decisions) {
      if (d.cost_usd || d.tokens_in) { hasCostData = true; break; }
    }
    if (hasCostData) break;
  }

  if (!hasCostData) {
    section.innerHTML += '<div class="stats-empty">No cost data available (run with LOS mode to capture token usage)</div>';
    return section;
  }

  const grid = document.createElement('div');
  grid.className = 'stats-cost-grid';

  for (const lender of lenders) {
    let totalCost = 0, totalTokensIn = 0, totalTokensOut = 0, totalToolCalls = 0, decisionCount = 0;

    for (const w of weeks) {
      for (const d of w.decisions) {
        if (d.lender_id !== lender.id) continue;
        decisionCount++;
        totalCost += d.cost_usd || 0;
        totalTokensIn += d.tokens_in || 0;
        totalTokensOut += d.tokens_out || 0;
        totalToolCalls += d.tool_calls || 0;
      }
    }

    const avgCost = decisionCount > 0 ? totalCost / decisionCount : 0;

    const card = document.createElement('div');
    card.className = 'stats-cost-card';
    card.innerHTML = `
      <div class="stats-confusion-title">${esc(lender.name)}</div>
      <div class="detail-grid">
        <div class="detail-stat"><span class="detail-stat-label">Total Cost</span><span class="detail-stat-value">$${totalCost.toFixed(2)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Avg/Decision</span><span class="detail-stat-value">$${avgCost.toFixed(3)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Tokens In</span><span class="detail-stat-value">${fmtNum(totalTokensIn)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Tokens Out</span><span class="detail-stat-value">${fmtNum(totalTokensOut)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Tool Calls</span><span class="detail-stat-value">${totalToolCalls}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Decisions</span><span class="detail-stat-value">${decisionCount}</span></div>
      </div>
    `;
    grid.appendChild(card);
  }

  section.appendChild(grid);
  return section;
}

// ---- Model Comparison Table ----

function buildModelComparisonTable(lenders, weeks) {
  const section = document.createElement('div');
  section.className = 'stats-section';
  section.innerHTML = '<h3 class="stats-section-title">Model Comparison</h3>';

  const lastWeek = weeks[weeks.length - 1];
  const table = document.createElement('table');
  table.className = 'stats-comparison-table';

  const header = document.createElement('tr');
  for (const h of ['Lender', 'Model', 'P&L', 'RAROC', 'Won', 'Rejected', 'Defaults', 'Fraud', 'Deploy %']) {
    const th = document.createElement('th');
    th.textContent = h;
    header.appendChild(th);
  }
  table.appendChild(header);

  const sorted = [...lenders].sort((a, b) => b.net_pnl - a.net_pnl);

  for (const l of sorted) {
    const snap = lastWeek.lender_snapshots[l.id] || l;
    const deployed = snap.deployed || 0;
    const cap = l.total_capital || 1;
    const raroc = deployed > 0 ? ((snap.net_pnl / deployed) * 100).toFixed(1) : '0.0';
    const deployPct = ((deployed / cap) * 100).toFixed(0);

    const row = document.createElement('tr');
    const pnlClass = snap.net_pnl >= 0 ? 'positive' : 'negative';
    const pnlStr = snap.net_pnl >= 0 ? `+$${fmtNum(snap.net_pnl)}` : `-$${fmtNum(Math.abs(snap.net_pnl))}`;

    row.innerHTML = `
      <td class="stats-table-name">${esc(l.name)}</td>
      <td class="stats-table-model">${esc(l.model)}</td>
      <td class="${pnlClass}">${pnlStr}</td>
      <td>${raroc}%</td>
      <td>${snap.deals_won || 0}</td>
      <td>${snap.deals_rejected || 0}</td>
      <td class="negative">${snap.defaults || 0}</td>
      <td class="negative">${snap.frauds_funded || 0}</td>
      <td>${deployPct}%</td>
    `;
    table.appendChild(row);
  }

  section.appendChild(table);
  return section;
}
