/**
 * Dashboard 2D — Information-dense canvas-based alternative to 3D town.
 *
 * Shows the lending market as a flow diagram: lenders on left, borrowers
 * flowing through, loans as animated connections. Optimized for data density.
 * Uses HTML/SVG for sharp text rendering.
 */

const COLORS = [
  '#e84a4a', '#4ae84a', '#4a90d9', '#e8a838', '#9a6ae8',
  '#e84ab0', '#4ae8d0', '#d9904a', '#6a9ae8', '#b8e84a',
];

const OUTCOME_COLORS = {
  good: '#4ae84a',
  bad: '#e8a838',
  fraud: '#e84a4a',
  unknown: '#6b7080',
};

function esc(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function fmtNum(n) {
  return Math.round(n).toLocaleString('en-US');
}

function formatRatePercent(rawRate) {
  if (rawRate == null) return '—';
  const rate = Number(rawRate);
  if (!Number.isFinite(rate)) return '—';
  const percent = rate <= 1 ? rate * 100 : rate;
  return `${percent.toFixed(1)}%`;
}

export class Dashboard2D {
  constructor(container) {
    this.container = container;
    this.season = null;
    this.currentWeek = -1;
  }

  setSeason(season) {
    this.season = season;
    this.currentWeek = -1;
    this.render();
  }

  setWeek(weekIndex) {
    this.currentWeek = weekIndex;
    this.render();
  }

  render() {
    this.container.innerHTML = '';
    if (!this.season) {
      this.container.innerHTML = '<div class="stats-empty">No season data loaded</div>';
      return;
    }

    const lenders = this.season.lenders;
    const weeks = this.season.weeks;
    const upTo = Math.min(this.currentWeek, weeks.length - 1);

    // Main layout: top summary bar, then flow diagram
    this.container.appendChild(this._buildTopBar(lenders, weeks, upTo));
    this.container.appendChild(this._buildFlowDiagram(lenders, weeks, upTo));
    this.container.appendChild(this._buildWeekTimeline(weeks, upTo));
  }

  _buildTopBar(lenders, weeks, upTo) {
    const bar = document.createElement('div');
    bar.className = 'dash2d-topbar';

    const sorted = [...lenders].map((l, i) => {
      const snap = upTo >= 0 ? (weeks[upTo].lender_snapshots[l.id] || l) : l;
      return { ...l, ...snap, color: COLORS[i % COLORS.length] };
    }).sort((a, b) => b.net_pnl - a.net_pnl);

    for (const l of sorted) {
      const deployed = l.deployed || 0;
      const cap = l.total_capital || 1;
      const pnlStr = l.net_pnl >= 0 ? `+$${fmtNum(l.net_pnl)}` : `-$${fmtNum(Math.abs(l.net_pnl))}`;
      const pnlCls = l.net_pnl >= 0 ? 'positive' : 'negative';

      const card = document.createElement('div');
      card.className = 'dash2d-lender-card';
      card.style.borderLeftColor = l.color;
      card.innerHTML = `
        <div class="dash2d-lender-name">${esc(l.name)}</div>
        <div class="dash2d-lender-model">${esc(l.model)}</div>
        <div class="dash2d-lender-stats">
          <span class="${pnlCls}">${pnlStr}</span>
          <span>${l.deals_won || 0}W / ${l.deals_rejected || 0}R</span>
          <span>${((deployed / cap) * 100).toFixed(0)}% deployed</span>
        </div>
        <div class="dash2d-capital-bar">
          <div class="dash2d-capital-fill" style="width:${((deployed / cap) * 100).toFixed(1)}%; background:${l.color}"></div>
        </div>
      `;
      bar.appendChild(card);
    }

    return bar;
  }

  _buildFlowDiagram(lenders, weeks, upTo) {
    const container = document.createElement('div');
    container.className = 'dash2d-flow';

    if (upTo < 0) {
      container.innerHTML = '<div class="stats-empty">Play the season to see the flow diagram</div>';
      return container;
    }

    const week = weeks[upTo];

    // Flow: Borrowers → Decisions → Outcomes
    // Left column: borrowers, Center: decision arrows, Right: lender targets

    const flowGrid = document.createElement('div');
    flowGrid.className = 'dash2d-flow-grid';

    // Borrower column
    const borrowerCol = document.createElement('div');
    borrowerCol.className = 'dash2d-col dash2d-col-borrowers';
    borrowerCol.innerHTML = '<div class="dash2d-col-header">Borrowers (Week ' + week.week + ')</div>';

    for (const b of week.borrowers) {
      const outcomeColor = OUTCOME_COLORS[b.true_outcome] || OUTCOME_COLORS.unknown;
      const loan = week.booked_loans.find(l => l.borrower_id === b.id);
      const funded = !!loan;
      const lender = funded ? lenders.find(l => l.id === loan.lender_id) : null;

      const el = document.createElement('div');
      el.className = `dash2d-borrower ${funded ? 'funded' : 'rejected'}`;
      el.innerHTML = `
        <div class="dash2d-borrower-top">
          <span class="dash2d-outcome-dot" style="background:${outcomeColor}" title="${b.true_outcome}"></span>
          <span class="dash2d-borrower-name">${esc(b.name)}</span>
          <span class="dash2d-borrower-amount">$${fmtNum(b.amount)}</span>
        </div>
        <div class="dash2d-borrower-flow">
          ${funded
            ? `<span class="dash2d-arrow funded">→ ${esc(lender?.name || '?')} @ ${formatRatePercent(loan.interest_rate)}</span>`
            : `<span class="dash2d-arrow rejected">✗ No offer</span>`
          }
        </div>
      `;
      borrowerCol.appendChild(el);
    }

    // Decision matrix column
    const matrixCol = document.createElement('div');
    matrixCol.className = 'dash2d-col dash2d-col-matrix';
    matrixCol.innerHTML = '<div class="dash2d-col-header">Decision Matrix</div>';

    const matrixTable = document.createElement('table');
    matrixTable.className = 'dash2d-matrix-table';

    // Header row
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th></th>' + week.borrowers.map(b =>
      `<th class="dash2d-matrix-header" title="${esc(b.name)}">${esc(b.name.substring(0, 8))}</th>`
    ).join('');
    matrixTable.appendChild(headerRow);

    // Lender rows
    for (let li = 0; li < lenders.length; li++) {
      const l = lenders[li];
      const color = COLORS[li % COLORS.length];
      const row = document.createElement('tr');
      row.innerHTML = `<td class="dash2d-matrix-lender" style="border-left: 3px solid ${color}">${esc(l.name.split(' ')[0])}</td>`;

      for (const b of week.borrowers) {
        const dec = week.decisions.find(d => d.lender_id === l.id && d.borrower_id === b.id);
        const loan = week.booked_loans.find(bl => bl.lender_id === l.id && bl.borrower_id === b.id);
        let cellClass = 'dash2d-cell-none';
        let cellContent = '—';

        if (dec) {
          const rawRate = dec.term_sheet?.rate ?? dec.term_sheet?.interest_rate;
          if (loan) {
            cellClass = 'dash2d-cell-won';
            cellContent = formatRatePercent(rawRate);
          } else if (dec.decision === 'APPROVE') {
            cellClass = 'dash2d-cell-lost';
            cellContent = formatRatePercent(rawRate);
          } else {
            cellClass = 'dash2d-cell-reject';
            cellContent = '✗';
          }
        }

        row.innerHTML += `<td class="${cellClass}" title="${dec?.decision || 'N/A'}">${cellContent}</td>`;
      }
      matrixTable.appendChild(row);
    }

    matrixCol.appendChild(matrixTable);

    flowGrid.appendChild(borrowerCol);
    flowGrid.appendChild(matrixCol);
    container.appendChild(flowGrid);

    // Cumulative stats bar below
    const cumBar = document.createElement('div');
    cumBar.className = 'dash2d-cumulative';

    const allBooked = weeks.slice(0, upTo + 1).reduce((s, w) => s + w.booked_loans.length, 0);
    const allBorrowers = weeks.slice(0, upTo + 1).reduce((s, w) => s + w.borrowers.length, 0);
    const allDefaults = (() => {
      let d = 0;
      for (const w of weeks.slice(0, upTo + 1)) {
        for (const e of w.events) {
          if (e.startsWith('DEFAULT')) d++;
        }
      }
      return d;
    })();

    cumBar.innerHTML = `
      <span>Through Week ${week.week}: </span>
      <span>${allBorrowers} applications</span>
      <span class="dash2d-sep">|</span>
      <span class="positive">${allBooked} funded</span>
      <span class="dash2d-sep">|</span>
      <span>${allBorrowers - allBooked} rejected</span>
      <span class="dash2d-sep">|</span>
      <span class="negative">${allDefaults} defaults</span>
    `;
    container.appendChild(cumBar);

    return container;
  }

  _buildWeekTimeline(weeks, upTo) {
    const container = document.createElement('div');
    container.className = 'dash2d-timeline';

    for (let i = 0; i < weeks.length; i++) {
      const dot = document.createElement('div');
      dot.className = 'dash2d-week-dot';
      if (i === upTo) dot.classList.add('active');
      else if (i < upTo) dot.classList.add('done');

      const funded = weeks[i].booked_loans.length;
      const total = weeks[i].borrowers.length;
      dot.innerHTML = `
        <div class="dash2d-week-num">W${i + 1}</div>
        <div class="dash2d-week-stat">${funded}/${total}</div>
      `;
      container.appendChild(dot);
    }

    return container;
  }
}
