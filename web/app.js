/**
 * Loanville — 3D Town Visualization + Trace Viewer
 *
 * Loads season JSON (from --json export) and/or trace JSON via drag-and-drop.
 * Season data drives the 3D town game view with playback.
 * Trace data drives the LLM trace inspection tab.
 */

import { AssetLoader } from './asset-loader.js';
import { TownScene } from './town-scene.js';
import { generateLayout } from './town-layout.js';
import { renderStats } from './stats-dashboard.js';
import { Dashboard2D } from './dashboard-2d.js';

// ---- Constants ----

const LENDER_COLORS = [
  '#e84a4a', '#4ae84a', '#4a90d9', '#e8a838', '#9a6ae8',
  '#e84ab0', '#4ae8d0', '#d9904a', '#6a9ae8', '#b8e84a',
];

const SPEED_LEVELS = [1, 2, 4];

// ---- App ----

class App {
  constructor() {
    // Data
    this.traces = [];
    this.season = null;  // parsed season JSON

    // State machine
    this.currentWeek = -1;  // index into season.weeks
    this.playing = false;
    this.speedIndex = 0;
    this.phaseTimer = null;
    this.stepping = false;  // true while a week is animating

    // 3D
    this.assetLoader = new AssetLoader();
    this.townScene = null;
    this.layout = null;

    // Trace viewer state
    this.lenderColorMap = {};
    this.nextColor = 0;

    // 2D Dashboard
    this.dashboard2d = null;

    this.dom = {};
    this._initDOM();
    this._bindEvents();
    this._populateSeasonDropdown();
  }

  async _populateSeasonDropdown() {
    const select = this.dom.selectSeason;
    try {
      const resp = await fetch('seasons/index.json');
      if (!resp.ok) return;
      const files = await resp.json(); // string[] sorted newest first
      if (!files.length) return;
      for (const f of files) {
        const opt = document.createElement('option');
        opt.value = `seasons/${f}`;
        // Format: "20260225_160530_Heritage-Velocity-Meridian.json" → readable label
        const label = f.replace('.json', '').replace(/_/g, ' ').replace(/-/g, ', ');
        opt.textContent = label;
        select.appendChild(opt);
      }
      // Auto-select and load the latest (first entry)
      select.value = `seasons/${files[0]}`;
      await this._loadSeasonFromUrl(`seasons/${files[0]}`);
    } catch (e) { /* no index available */ }
  }

  async _loadSeasonFromUrl(url) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.type === 'season') {
        this._reset();
        this.season = data;
        const name = url.split('/').pop();
        this._logEvent('system', `Loaded ${name}: ${data.weeks.length} timeline steps, ${data.lenders.length} lenders`);
        this.dom.dropOverlay.hidden = true;
        this.dashboard2d = null;
        await this._initTown();
        this.dom.controlsGroup.hidden = false;
        this.dom.packSelector.hidden = false;
        this._buildTimeline();
        this._renderLeaderboard(-1);
        this._extractSeasonTraces();
        this._switchTab('game');
      }
    } catch (e) {
      console.error('Failed to load season:', e);
    }
  }

  _initDOM() {
    const $ = id => document.getElementById(id);
    this.dom = {
      dropOverlay: $('drop-overlay'),
      fileInput: $('file-input'),
      tabGame: $('tab-game'),
      tabDashboard: $('tab-dashboard'),
      tabStats: $('tab-stats'),
      tabTrace: $('tab-trace'),
      pageGame: $('page-game'),
      pageDashboard: $('page-dashboard'),
      pageStats: $('page-stats'),
      pageTrace: $('page-trace'),
      dashboardContent: $('dashboard-content'),
      statsContent: $('stats-content'),
      townContainer: $('town-container'),
      btnPlay: $('btn-play'),
      btnSpeed: $('btn-speed'),
      btnReset: $('btn-reset'),
      weekLabel: $('week-label'),
      progressBar: $('progress-bar'),
      timeline: $('timeline'),
      phaseLabel: $('phase-label'),
      leaderboard: $('leaderboard'),
      detailPanel: $('detail-panel'),
      eventLogBody: $('event-log-body'),
      controlsGroup: $('controls-group'),
      // Season selector
      selectSeason: $('select-season'),
      // Pack selector
      packSelector: $('pack-selector'),
      selectBuildingPack: $('select-building-pack'),
      selectCharacterPack: $('select-character-pack'),
      // Trace tab
      traceToolbar: $('trace-toolbar'),
      traceContent: $('trace-content'),
      filterLender: $('filter-lender'),
      filterModel: $('filter-model'),
      filterDecision: $('filter-decision'),
      filterSearch: $('filter-search'),
      traceStats: $('trace-stats'),
    };
  }

  _bindEvents() {
    // Drag & drop
    document.body.addEventListener('dragover', e => {
      e.preventDefault();
      this.dom.dropOverlay.classList.add('dragover');
      this.dom.dropOverlay.hidden = false;
    });
    this.dom.dropOverlay.addEventListener('dragleave', e => {
      if (!e.relatedTarget || !this.dom.dropOverlay.contains(e.relatedTarget)) {
        this.dom.dropOverlay.classList.remove('dragover');
        if (this.season || this.traces.length) this.dom.dropOverlay.hidden = true;
      }
    });
    this.dom.dropOverlay.addEventListener('drop', e => {
      e.preventDefault();
      this.dom.dropOverlay.classList.remove('dragover');
      this._loadFiles(e.dataTransfer.files);
    });
    this.dom.fileInput.addEventListener('change', () => {
      this._loadFiles(this.dom.fileInput.files);
      this.dom.fileInput.value = '';
    });

    // Tabs
    this.dom.tabGame.addEventListener('click', () => this._switchTab('game'));
    this.dom.tabDashboard.addEventListener('click', () => this._switchTab('dashboard'));
    this.dom.tabStats.addEventListener('click', () => this._switchTab('stats'));
    this.dom.tabTrace.addEventListener('click', () => this._switchTab('trace'));

    // Playback
    this.dom.btnPlay.addEventListener('click', () => this._togglePlay());
    this.dom.btnSpeed.addEventListener('click', () => this._cycleSpeed());
    this.dom.btnReset.addEventListener('click', () => this._reset());

    // Trace filters
    for (const el of [this.dom.filterLender, this.dom.filterModel, this.dom.filterDecision]) {
      el.addEventListener('change', () => this._renderTraces());
    }
    this.dom.filterSearch.addEventListener('input', () => this._renderTraces());

    // Season selector
    this.dom.selectSeason.addEventListener('change', () => {
      const url = this.dom.selectSeason.value;
      if (url) this._loadSeasonFromUrl(url);
    });

    // Pack selector
    this.dom.selectBuildingPack.addEventListener('change', () => this._changePack());
    this.dom.selectCharacterPack.addEventListener('change', () => this._changePack());
  }

  // ==== File Loading ====

  async _loadFiles(fileList) {
    for (const file of fileList) {
      try {
        const data = JSON.parse(await file.text());
        if (data.type === 'season') {
          this.season = data;
          this._logEvent('system', `Loaded season: ${data.weeks.length} timeline steps, ${data.lenders.length} lenders`);
        } else if (Array.isArray(data)) {
          for (const t of data) { t._file = file.name; this._assignColor(t.lender_name); }
          this.traces.push(...data);
          this._logEvent('system', `Loaded ${data.length} traces from ${file.name}`);
        } else if (data.match_id) {
          // Legacy single-match format — wrap as 1-week season
          this._legacyMatchToSeason(data);
        }
      } catch (err) {
        console.error(`Failed to parse ${file.name}:`, err);
      }
    }

    this.dom.dropOverlay.hidden = true;

    if (this.season) {
      this.dashboard2d = null;
      await this._initTown();
      this.dom.controlsGroup.hidden = false;
      this.dom.packSelector.hidden = false;
      this._buildTimeline();
      this._renderLeaderboard(-1);
      this._extractSeasonTraces();
      this._switchTab('game');
    }
    if (this.traces.length) {
      this._updateTraceFilters();
      this._renderTraces();
    }
  }

  _legacyMatchToSeason(match) {
    // Convert old match_id format to season format
    const lenders = (match.models || []).map((m, i) => ({
      id: m.model_id || `lender-${i}`,
      name: m.display_name || m.model_id,
      model: m.model_id || 'unknown',
      total_capital: 1000000,
      net_pnl: 0, deployed: 0, deals_won: 0, deals_rejected: 0,
      deals_lost: 0, cumulative_interest: 0, cumulative_losses: 0,
      cumulative_fees: 0, defaults: 0, frauds_funded: 0,
      weekly_utilization: [], weekly_snapshots: [],
    }));

    const decisions = [];
    const borrowerSet = new Set();
    for (const result of (match.results || [])) {
      if (result.per_borrower) {
        for (const [bid, info] of Object.entries(result.per_borrower)) {
          borrowerSet.add(bid);
          decisions.push({
            lender_id: result.model_id,
            borrower_id: bid,
            decision: info.decision_state === 'won' ? 'APPROVE'
              : info.decision_state === 'declined' ? 'REJECT' : 'REJECT',
            reasoning: '',
            term_sheet: null,
          });
        }
      }
    }

    this.season = {
      type: 'season',
      config: { weeks: 1, cohort_size: borrowerSet.size, months_per_week: 2, season_mix: 'unknown', seed: 0 },
      lenders,
      weeks: [{
        week: 1,
        borrowers: [...borrowerSet].map(id => ({ id, name: id, sector: 'unknown', amount: 0, true_outcome: 'unknown' })),
        decisions,
        booked_loans: [],
        events: [],
        lender_snapshots: {},
      }],
    };
  }

  _assignColor(name) {
    if (!name || this.lenderColorMap[name]) return;
    this.lenderColorMap[name] = LENDER_COLORS[this.nextColor++ % LENDER_COLORS.length];
  }

  // ==== 3D Town ====

  async _initTown() {
    if (this.townScene) this.townScene.dispose();

    const lenderCount = this.season.lenders.length;

    this.townScene = new TownScene();
    this.townScene.init(this.dom.townContainer);

    try {
      await this.assetLoader.loadManifest();
      const manifest = this.assetLoader.manifest;

      // Sync dropdown values to match manifest defaults
      this.dom.selectBuildingPack.value = manifest.activePack;
      this.dom.selectCharacterPack.value = manifest.characterPack;

      // Use buildingTiers (curated order) if available, else fall back to buildings list
      const packConfig = this.assetLoader.getActivePack();
      const buildingFiles = packConfig.buildingTiers || packConfig.buildings;
      this.layout = generateLayout(lenderCount, buildingFiles);

      // Always set layout on the scene so animations work even without 3D models
      this.townScene.layout = this.layout;

      const names = this.season.lenders.map(l => l.name);
      await this.townScene.buildTown(this.layout, names, this.assetLoader);
    } catch (err) {
      console.warn('Asset loading failed, continuing without 3D models:', err);
      // Fallback layout without curated buildings
      this.layout = generateLayout(lenderCount);
      this.townScene.layout = this.layout;
    }

    this.townScene.onBuildingClick = idx => this._showDetail(idx);
  }

  async _changePack() {
    if (!this.season) return;
    this.assetLoader.manifest.activePack = this.dom.selectBuildingPack.value;
    this.assetLoader.manifest.characterPack = this.dom.selectCharacterPack.value;
    // Rebuild town with new assets
    this._reset();
    await this._initTown();
  }

  // ==== State Machine ====

  _setPhase(phase) {
    this.dom.phaseLabel.textContent = phase;
    this.dom.phaseLabel.className = `phase-label phase-${phase.toLowerCase().replace(/\s+/g, '_')}`;
  }

  async _stepWeek() {
    if (this.stepping) return;
    this.stepping = true;

    this.currentWeek++;
    if (this.currentWeek >= this.season.weeks.length) {
      this._setPhase('SEASON END');
      this._showSeasonEnd();
      this.playing = false;
      this.dom.btnPlay.textContent = '\u25B6';
      this.stepping = false;
      return;
    }

    const week = this.season.weeks[this.currentWeek];
    const lenders = this.season.lenders;

    // Final resolution is exported as a dedicated timeline step.
    if (week.final_resolution) {
      this._setPhase('FINAL RESOLUTION');
      this.dom.weekLabel.textContent = 'Final Resolution';
      this._updateProgress();
      this._highlightTimelineDot(this.currentWeek);
      this._logEvent('system', '--- Final Resolution ---');
      for (const evt of week.events) {
        this._logEvent(this._eventTypeForMessage(evt), evt);
      }
      await this._wait(800);
      this._renderLeaderboard(this.currentWeek);

      for (let i = 0; i < lenders.length; i++) {
        const snap = week.lender_snapshots[lenders[i].id];
        if (snap) {
          const totalDec = (snap.deals_won || 0) + (snap.deals_rejected || 0) + (snap.deals_lost || 0);
          const approvalRate = totalDec > 0 ? ((snap.deals_won || 0) / totalDec) * 100 : null;
          this.townScene?.updateSignpost(i, snap.name, approvalRate, snap.net_pnl);
        }
      }

      this._logEvent('system', 'Final resolution complete');
      await this._wait(600);
      this.stepping = false;
      if (this.playing) {
        this.phaseTimer = setTimeout(() => this._stepWeek(), 200);
      }
      return;
    }

    // ---- WEEK INTRO ----
    this._setPhase('WEEK INTRO');
    this.dom.weekLabel.textContent = `Week ${week.week}`;
    this._updateProgress();
    this._highlightTimelineDot(this.currentWeek);
    this._logEvent('system', `--- Week ${week.week} ---`);

    // Show loan resolution events from this week
    for (const evt of week.events) {
      this._logEvent(this._eventTypeForMessage(evt), evt);
    }
    await this._wait(800);

    // ---- BORROWERS ARRIVE ----
    this._setPhase('BORROWERS ARRIVE');
    this.townScene?.clearBorrowers();

    const totalBorrowers = week.borrowers.length;
    const spawnPos = this.layout?.spawnPoint || { x: 0, z: -10 };
    const borrowerSpawns = week.borrowers.map((b, i) =>
      this.townScene?.addBorrower(b.id, this.assetLoader, { name: b.name, amount: b.amount }, i, totalBorrowers, spawnPos)
    );
    await Promise.all(borrowerSpawns.filter(Boolean));
    this._logEvent('loan', `${week.borrowers.length} borrowers arrived: ${week.borrowers.map(b => b.name).join(', ')}`);
    await this._wait(400);

    // ---- TRAVEL (residential → junction) ----
    this._setPhase('TRAVEL');
    const junction = { x: 0, z: 0 };
    const travelWalks = week.borrowers.map((b, i) => {
      const spread = (i - (totalBorrowers - 1) / 2) * 0.6;
      return this.townScene?.animateBorrowerWalk(b.id, { x: junction.x + spread, z: junction.z }, 1.0 / SPEED_LEVELS[this.speedIndex]);
    });
    await Promise.all(travelWalks.filter(Boolean));
    await this._wait(200);

    // ---- EVALUATION ----
    this._setPhase('EVALUATION');
    if (this.townScene) {
      await this.townScene.pulseBuildings(800 / SPEED_LEVELS[this.speedIndex]);
    }

    // ---- ADJUDICATION ----
    this._setPhase('ADJUDICATION');
    for (const dec of week.decisions) {
      const idx = lenders.findIndex(l => l.id === dec.lender_id);
      if (idx >= 0) {
        const approved = dec.decision === 'APPROVE';
        this.townScene?.showDecision(idx, approved);
      }
      const borrower = week.borrowers.find(b => b.id === dec.borrower_id);
      const bName = borrower?.name || dec.borrower_id;
      const verb = dec.decision === 'APPROVE' ? 'approved' : 'rejected';
      const lender = lenders.find(l => l.id === dec.lender_id);
      this._logEvent('loan', `${lender?.name || dec.lender_id} ${verb} ${bName}`);
    }
    await this._wait(1200);

    // ---- ALLOCATION ----
    this._setPhase('ALLOCATION');
    const bookedSet = new Set(week.booked_loans.map(l => l.borrower_id));
    const lenderQueues = {};  // lenderIndex → count
    const allocWalks = [];

    const rejectTarget = this.layout?.spawnPoint || { x: 0, z: -10 };
    for (const b of week.borrowers) {
      const loan = week.booked_loans.find(l => l.borrower_id === b.id);
      if (loan) {
        // Funded — walk from junction along +X to winning lender's building
        const lenderIdx = lenders.findIndex(l => l.id === loan.lender_id);
        if (lenderIdx >= 0 && this.townScene?.layout?.buildings?.[lenderIdx]) {
          const bld = this.townScene.layout.buildings[lenderIdx];
          const queueCount = lenderQueues[lenderIdx] || 0;
          lenderQueues[lenderIdx] = queueCount + 1;
          const sign = Math.sign(bld.z) || 1;
          const queueZ = sign * (Math.abs(bld.z) - queueCount * 0.6);
          allocWalks.push(
            this.townScene.animateBorrowerWalk(b.id, { x: bld.x, z: queueZ }, 0.8 / SPEED_LEVELS[this.speedIndex])
          );
        }
        const lender = lenders.find(l => l.id === loan.lender_id);
        this._logEvent('loan', `BOOKED: ${b.name} \u2192 ${lender?.name || loan.lender_id} ($${fmtNum(loan.principal)} @ ${loan.interest_rate.toFixed(1)}%)`);
      } else {
        // Rejected — walk back toward residential zone and fade
        if (this.townScene) {
          const walkOff = this.townScene.animateBorrowerWalk(b.id, rejectTarget, 0.6 / SPEED_LEVELS[this.speedIndex])
            .then(() => this.townScene.fadeBorrower(b.id, 400 / SPEED_LEVELS[this.speedIndex]));
          allocWalks.push(walkOff);
        }
        this._logEvent('loan', `REJECTED: ${b.name} — no winning offer`);
      }
    }
    await Promise.all(allocWalks);

    // ---- WEEK SUMMARY ----
    this._setPhase('WEEK SUMMARY');
    this.townScene?.resetBuildings();
    this._renderLeaderboard(this.currentWeek);

    // Update bank signposts with approval rate + P&L
    for (let i = 0; i < lenders.length; i++) {
      const snap = week.lender_snapshots[lenders[i].id];
      if (snap) {
        const totalDec = (snap.deals_won || 0) + (snap.deals_rejected || 0) + (snap.deals_lost || 0);
        const approvalRate = totalDec > 0 ? ((snap.deals_won || 0) / totalDec) * 100 : null;
        this.townScene?.updateSignpost(i, snap.name, approvalRate, snap.net_pnl);
      }
    }

    this._logEvent('system', `Week ${week.week} complete: ${week.booked_loans.length} loans booked`);

    // Sync 2D dashboard if it exists
    if (this.dashboard2d) this.dashboard2d.setWeek(this.currentWeek);

    await this._wait(600);

    this.stepping = false;

    // Auto-advance if playing
    if (this.playing) {
      this.phaseTimer = setTimeout(() => this._stepWeek(), 200);
    }
  }

  _togglePlay() {
    if (!this.season) return;
    this.playing = !this.playing;
    this.dom.btnPlay.textContent = this.playing ? '\u23F8' : '\u25B6';
    if (this.playing && !this.stepping) {
      if (this.currentWeek >= this.season.weeks.length - 1) {
        this.currentWeek = -1;  // restart from beginning
      }
      this._stepWeek();
    }
  }

  _cycleSpeed() {
    this.speedIndex = (this.speedIndex + 1) % SPEED_LEVELS.length;
    this.dom.btnSpeed.textContent = `${SPEED_LEVELS[this.speedIndex]}x`;
  }

  _reset() {
    this.playing = false;
    this.stepping = false;
    clearTimeout(this.phaseTimer);
    this.dom.btnPlay.textContent = '\u25B6';
    this.currentWeek = -1;
    this._setPhase('IDLE');
    this.dom.weekLabel.textContent = 'Ready';
    this.dom.progressBar.style.width = '0%';
    this.townScene?.clearBorrowers();
    this.townScene?.resetBuildings();
    this._renderLeaderboard(-1);
    this._clearTimeline();
    this._buildTimeline();
    this.dom.eventLogBody.innerHTML = '';
    this.dom.detailPanel.innerHTML = '<div class="detail-placeholder">Click a building to inspect</div>';

    // Reset bank signposts
    if (this.season && this.townScene) {
      for (let i = 0; i < this.season.lenders.length; i++) {
        this.townScene.updateSignpost(i, this.season.lenders[i].name, null, null);
      }
    }
  }

  _wait(baseMs) {
    return new Promise(r => setTimeout(r, baseMs / SPEED_LEVELS[this.speedIndex]));
  }

  // ==== UI ====

  _switchTab(tab) {
    this.dom.tabGame.classList.toggle('active', tab === 'game');
    this.dom.tabDashboard.classList.toggle('active', tab === 'dashboard');
    this.dom.tabStats.classList.toggle('active', tab === 'stats');
    this.dom.tabTrace.classList.toggle('active', tab === 'trace');
    this.dom.pageGame.hidden = tab !== 'game';
    this.dom.pageDashboard.hidden = tab !== 'dashboard';
    this.dom.pageStats.hidden = tab !== 'stats';
    this.dom.pageTrace.hidden = tab !== 'trace';
    if (tab === 'game' && this.townScene) this.townScene._onResize();
    if (tab === 'stats' && this.season) renderStats(this.dom.statsContent, this.season);
    if (tab === 'dashboard') this._updateDashboard2d();
  }

  _buildTimeline() {
    this.dom.timeline.innerHTML = '';
    if (!this.season) return;
    for (let i = 0; i < this.season.weeks.length; i++) {
      const week = this.season.weeks[i];
      const dot = document.createElement('div');
      dot.className = 'timeline-dot';
      dot.title = week.final_resolution ? 'Final Resolution' : `Week ${week.week}`;
      dot.addEventListener('click', () => {
        if (!this.playing && !this.stepping) {
          this.currentWeek = i - 1;
          this._stepWeek();
        }
      });
      this.dom.timeline.appendChild(dot);
    }
  }

  _clearTimeline() {
    this.dom.timeline.querySelectorAll('.timeline-dot').forEach(d => d.classList.remove('active', 'done'));
  }

  _highlightTimelineDot(idx) {
    const dots = this.dom.timeline.querySelectorAll('.timeline-dot');
    dots.forEach((d, i) => {
      d.classList.toggle('active', i === idx);
      d.classList.toggle('done', i < idx);
    });
  }

  _updateProgress() {
    if (!this.season) return;
    const pct = ((this.currentWeek + 1) / this.season.weeks.length) * 100;
    this.dom.progressBar.style.width = `${pct}%`;
  }

  // ==== Leaderboard ====

  _renderLeaderboard(upToWeekIndex) {
    if (!this.season) { this.dom.leaderboard.innerHTML = ''; return; }

    // Use lender snapshots from the specified week, or final stats
    let stats;
    if (upToWeekIndex >= 0 && upToWeekIndex < this.season.weeks.length) {
      const snapshots = this.season.weeks[upToWeekIndex].lender_snapshots;
      stats = this.season.lenders.map(l => ({
        ...l,
        ...(snapshots[l.id] || {}),
      }));
    } else {
      // Use final stats from season.lenders (includes everything)
      stats = [...this.season.lenders];
    }

    stats.sort((a, b) => b.net_pnl - a.net_pnl);
    this.dom.leaderboard.innerHTML = '';

    for (let rank = 0; rank < stats.length; rank++) {
      const s = stats[rank];
      const card = document.createElement('div');
      card.className = 'lb-card';

      const lenderIdx = this.season.lenders.findIndex(l => l.id === s.id);
      card.addEventListener('click', (e) => {
        e.stopPropagation();
        if (lenderIdx >= 0) this._showDetail(lenderIdx);
      });

      const pnlClass = s.net_pnl >= 0 ? 'positive' : 'negative';
      const pnlStr = s.net_pnl >= 0 ? `+$${fmtNum(s.net_pnl)}` : `-$${fmtNum(Math.abs(s.net_pnl))}`;
      const deployed = s.deployed || 0;
      const raroc = deployed > 0 ? ((s.net_pnl / deployed) * 100).toFixed(1) : '0.0';

      const costStr = s.cost_usd > 0 ? `$${s.cost_usd.toFixed(2)}` : '';

      card.innerHTML = `
        <div class="lb-rank">#${rank + 1}</div>
        <div class="lb-info">
          <div class="lb-name">${esc(s.name)}</div>
          <div class="lb-model">${esc(s.model)}</div>
        </div>
        <div class="lb-stats">
          <span class="lb-pnl ${pnlClass}">${pnlStr}</span>
          <span class="lb-raroc">RAROC ${raroc}%</span>
          <span class="lb-deals">${s.deals_won}W / ${s.deals_rejected}R</span>
          ${costStr ? `<span class="lb-cost">${costStr}</span>` : ''}
        </div>
      `;
      this.dom.leaderboard.appendChild(card);
    }
  }

  // ==== Detail Panel ====

  _showDetail(lenderIndex) {
    if (!this.season) return;
    const lender = this.season.lenders[lenderIndex];
    if (!lender) return;

    this.townScene?.highlightBuilding(lenderIndex);

    // Get snapshot at current week or final
    let s;
    if (this.currentWeek >= 0 && this.currentWeek < this.season.weeks.length) {
      const snap = this.season.weeks[this.currentWeek].lender_snapshots[lender.id];
      s = snap ? { ...lender, ...snap } : lender;
    } else {
      s = lender;
    }

    const pnlClass = s.net_pnl >= 0 ? 'positive' : 'negative';
    const pnlStr = s.net_pnl >= 0 ? `+$${fmtNum(s.net_pnl)}` : `-$${fmtNum(Math.abs(s.net_pnl))}`;
    const deployed = s.deployed || 0;
    const raroc = deployed > 0 ? ((s.net_pnl / deployed) * 100).toFixed(1) : '0.0';

    let html = `
      <div class="detail-header">
        <h3>${esc(s.name)}</h3>
        <div class="detail-model">${esc(s.model)}</div>
      </div>
      <div class="detail-grid">
        <div class="detail-stat"><span class="detail-stat-label">Net P&L</span><span class="detail-stat-value ${pnlClass}">${pnlStr}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">RAROC</span><span class="detail-stat-value">${raroc}%</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Capital</span><span class="detail-stat-value">$${fmtNum(s.total_capital || 0)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Deployed</span><span class="detail-stat-value">$${fmtNum(deployed)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Won</span><span class="detail-stat-value">${s.deals_won || 0}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Rejected</span><span class="detail-stat-value">${s.deals_rejected || 0}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Lost</span><span class="detail-stat-value">${s.deals_lost || 0}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Defaults</span><span class="detail-stat-value negative">${s.defaults || 0}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Frauds</span><span class="detail-stat-value negative">${s.frauds_funded || 0}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Interest</span><span class="detail-stat-value positive">$${fmtNum(s.cumulative_interest || 0)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Losses</span><span class="detail-stat-value negative">$${fmtNum(s.cumulative_losses || 0)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Active Loans</span><span class="detail-stat-value">${s.active_loans || 0}</span></div>
        ${s.cost_usd > 0 ? `
        <div class="detail-stat"><span class="detail-stat-label">API Cost</span><span class="detail-stat-value">$${(s.cost_usd || 0).toFixed(2)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Tokens In</span><span class="detail-stat-value">${fmtNum(s.tokens_in || 0)}</span></div>
        <div class="detail-stat"><span class="detail-stat-label">Tokens Out</span><span class="detail-stat-value">${fmtNum(s.tokens_out || 0)}</span></div>
        ` : ''}
      </div>
    `;

    // Per-week P&L sparkline (text-based)
    if (lender.weekly_snapshots?.length > 0) {
      html += `<div class="detail-section-title">Weekly P&L</div><div class="sparkline-row">`;
      for (const snap of lender.weekly_snapshots) {
        const pnl = snap.net_pnl;
        const cls = pnl >= 0 ? 'positive' : 'negative';
        html += `<span class="sparkline-bar ${cls}" title="Week ${snap.week}: $${fmtNum(pnl)}">${pnl >= 0 ? '+' : '-'}</span>`;
      }
      html += `</div>`;
    }

    // This week's decisions for this lender
    if (this.currentWeek >= 0) {
      const week = this.season.weeks[this.currentWeek];
      const thisLenderDecs = week.decisions.filter(d => d.lender_id === lender.id);
      if (thisLenderDecs.length > 0) {
        html += `<div class="detail-section-title">Week ${week.week} Decisions</div><div class="borrower-grid">`;
        for (const dec of thisLenderDecs) {
          const b = week.borrowers.find(b => b.id === dec.borrower_id);
          const cls = dec.decision === 'APPROVE' ? 'state-won' : 'state-declined';
          html += `<div class="borrower-cell">
            <span class="borrower-cell-id">${esc(b?.name || dec.borrower_id)}</span>
            <span class="borrower-cell-state ${cls}">${dec.decision}</span>
          </div>`;
        }
        html += `</div>`;
      }
    }

    this.dom.detailPanel.innerHTML = html;
  }

  // ==== Event Log ====

  _logEvent(type, message) {
    const entry = document.createElement('div');
    entry.className = `event-entry event-${type}`;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    entry.innerHTML = `<span class="event-time">${time}</span> <span class="event-msg">${esc(message)}</span>`;
    this.dom.eventLogBody.appendChild(entry);
    this.dom.eventLogBody.scrollTop = this.dom.eventLogBody.scrollHeight;
  }

  _eventTypeForMessage(evt) {
    if (evt.includes('DEFAULT')) return 'error';
    if (evt.startsWith('REPAID') || evt.startsWith('PREPAID') || evt.startsWith('PAYMENT')) return 'payment';
    return 'loan';
  }

  // ==== Season End ====

  _showSeasonEnd() {
    if (!this.season) return;
    const sorted = [...this.season.lenders].sort((a, b) => b.net_pnl - a.net_pnl);
    const winner = sorted[0];

    let overlay = document.getElementById('season-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'season-overlay';
      overlay.className = 'season-overlay';
      document.body.appendChild(overlay);
    }

    let html = `<div class="season-overlay-content">
      <h2>Season Complete</h2>
      <div class="season-winner">Winner: ${esc(winner.name)}</div>
      <div class="season-rankings">`;

    for (let i = 0; i < sorted.length; i++) {
      const s = sorted[i];
      const pnlStr = s.net_pnl >= 0 ? `+$${fmtNum(s.net_pnl)}` : `-$${fmtNum(Math.abs(s.net_pnl))}`;
      html += `<div class="season-rank-row">
        <span class="season-rank">#${i + 1}</span>
        <span class="season-rank-name">${esc(s.name)}</span>
        <span class="season-rank-pnl ${s.net_pnl >= 0 ? 'positive' : 'negative'}">${pnlStr}</span>
      </div>`;
    }

    html += `</div><button class="btn btn-primary" id="btn-close-overlay">Close</button></div>`;
    overlay.innerHTML = html;
    overlay.hidden = false;

    document.getElementById('btn-close-overlay').addEventListener('click', () => {
      overlay.hidden = true;
    });
  }

  // ==== 2D Dashboard ====

  _updateDashboard2d() {
    if (!this.season) return;
    if (!this.dashboard2d) {
      this.dashboard2d = new Dashboard2D(this.dom.dashboardContent);
      this.dashboard2d.setSeason(this.season);
    }
    this.dashboard2d.setWeek(this.currentWeek);
  }

  // ==== Trace Viewer ====

  /**
   * Extract decision data from season JSON into the trace array,
   * enriched with cost and chain-of-thought data when available.
   */
  _extractSeasonTraces() {
    if (!this.season) return;
    // Remove any previous season-derived traces
    this.traces = this.traces.filter(t => t._source !== 'season');

    for (const week of this.season.weeks) {
      if (!week.decisions) continue;
      for (const dec of week.decisions) {
        const lender = this.season.lenders.find(l => l.id === dec.lender_id);
        const borrower = week.borrowers?.find(b => b.id === dec.borrower_id);
        this.traces.push({
          _source: 'season',
          _file: `Season Week ${week.week}`,
          _week: week.week,
          lender_name: lender?.name || dec.lender_id,
          lender_id: dec.lender_id,
          model: lender?.model || '',
          borrower_id: dec.borrower_id,
          borrower_name: borrower?.name || dec.borrower_id,
          decision: dec.decision,
          reasoning: dec.reasoning || '',
          term_sheet: dec.term_sheet,
          tokens_in: dec.tokens_in || 0,
          tokens_out: dec.tokens_out || 0,
          cost_usd: dec.cost_usd || 0,
          tool_calls_count: dec.tool_calls || 0,
          chain_of_thought: dec.chain_of_thought || '',
          true_outcome: borrower?.true_outcome || 'unknown',
        });
        this._assignColor(lender?.name);
      }
    }

    if (this.traces.length) {
      this._updateTraceFilters();
      this._renderTraces();
    }
  }

  _updateTraceFilters() {
    const lenders = new Set();
    const models = new Set();
    for (const t of this.traces) {
      if (t.lender_name) lenders.add(t.lender_name);
      if (t.model) models.add(t.model);
    }
    this._populateSelect(this.dom.filterLender, 'All lenders', [...lenders].sort());
    this._populateSelect(this.dom.filterModel, 'All models', [...models].sort());
  }

  _populateSelect(el, defaultLabel, options) {
    const current = el.value;
    el.innerHTML = `<option value="">${defaultLabel}</option>`;
    for (const opt of options) {
      const o = document.createElement('option');
      o.value = opt; o.textContent = opt;
      el.appendChild(o);
    }
    el.value = current;
  }

  _getFilteredTraces() {
    const lender = this.dom.filterLender.value;
    const model = this.dom.filterModel.value;
    const decision = this.dom.filterDecision.value;
    const search = this.dom.filterSearch.value.toLowerCase();

    return this.traces.filter(t => {
      if (lender && t.lender_name !== lender) return false;
      if (model && t.model !== model) return false;
      if (decision === 'ERROR' && !t.error) return false;
      if (decision && decision !== 'ERROR' && t.decision !== decision) return false;
      if (search) {
        const haystack = [t.lender_name, t.borrower_name, t.model, t.decision,
          t.system_prompt, t.user_prompt, t.raw_response, t.error, t.reasoning
        ].filter(Boolean).join(' ').toLowerCase();
        if (!haystack.includes(search)) return false;
      }
      return true;
    });
  }

  _renderTraces() {
    const content = this.dom.traceContent;
    content.innerHTML = '';

    const filtered = this._getFilteredTraces();
    const byFile = new Map();
    for (const t of filtered) {
      const f = t._file || 'unknown';
      if (!byFile.has(f)) byFile.set(f, []);
      byFile.get(f).push(t);
    }

    for (const [fileName, traces] of byFile) {
      const group = document.createElement('div');
      group.className = 'file-group';

      const header = document.createElement('div');
      header.className = 'file-group-header';
      header.textContent = `${fileName} (${traces.length} traces)`;
      group.appendChild(header);

      for (const trace of traces) {
        group.appendChild(this._renderTraceCard(trace));
      }
      content.appendChild(group);
    }

    const approves = filtered.filter(t => t.decision === 'APPROVE').length;
    const rejects = filtered.filter(t => t.decision === 'REJECT').length;
    const errors = filtered.filter(t => t.error).length;
    this.dom.traceStats.textContent =
      `${filtered.length} traces | ${approves} approvals | ${rejects} rejections` +
      (errors ? ` | ${errors} errors` : '');
    this.dom.traceToolbar.hidden = filtered.length === 0 && this.traces.length === 0;
  }

  _renderTraceCard(trace) {
    const card = document.createElement('div');
    card.className = 'trace-card';

    const color = this.lenderColorMap[trace.lender_name] || '#888';
    const decisionClass = trace.error ? 'error' : (trace.decision || '').toLowerCase();
    const decisionLabel = trace.error ? 'ERROR' : (trace.decision || '?');

    // Cost badge
    let costBadge = '';
    if (trace.cost_usd > 0) {
      costBadge += `<span class="trace-card-cost">$${trace.cost_usd.toFixed(4)}</span>`;
    }
    if (trace.tokens_in > 0 || trace.tokens_out > 0) {
      costBadge += `<span class="trace-card-tokens">${fmtNum(trace.tokens_in + trace.tokens_out)} tok</span>`;
    }

    // True outcome badge for season traces
    let outcomeBadge = '';
    if (trace.true_outcome && trace.true_outcome !== 'unknown') {
      const outcomeClass = `outcome-${trace.true_outcome}`;
      outcomeBadge = `<span class="trace-card-outcome ${outcomeClass}">${trace.true_outcome}</span>`;
    }

    const header = document.createElement('div');
    header.className = 'trace-card-header';
    header.innerHTML = `
      <span class="trace-card-arrow">&#x25B6;</span>
      <span class="trace-card-lender-dot" style="background:${color}"></span>
      <span class="trace-card-lender">${esc(trace.lender_name || trace.lender_id)}</span>
      <span class="trace-card-model">${esc(trace.model || '')}</span>
      <span class="trace-card-borrower">${esc(trace.borrower_name || trace.borrower_id)}</span>
      ${outcomeBadge}
      <span class="trace-card-decision ${decisionClass}">${decisionLabel}</span>
      ${costBadge}
    `;
    header.addEventListener('click', () => card.classList.toggle('expanded'));
    card.appendChild(header);

    const body = document.createElement('div');
    body.className = 'trace-card-body';

    // Chain-of-thought (show prominently if available)
    if (trace.chain_of_thought) {
      body.appendChild(this._makeSection('Chain of Thought', () => {
        const el = document.createElement('div');
        el.className = 'trace-cot';
        el.innerHTML = `<div class="trace-cot-content">${esc(trace.chain_of_thought)}</div>`;
        return el;
      }, true));
    }

    // Reasoning summary
    if (trace.reasoning) {
      body.appendChild(this._makeSection('Reasoning', () => pre(trace.reasoning), true));
    }

    // Term sheet
    if (trace.term_sheet) {
      body.appendChild(this._makeSection('Term Sheet', () => {
        const el = document.createElement('div');
        el.className = 'detail-grid';
        const rate = trace.term_sheet.rate != null ? trace.term_sheet.rate : trace.term_sheet.interest_rate;
        const rateStr = rate != null ? (rate > 1 ? rate.toFixed(1) : (rate * 100).toFixed(1)) : '—';
        el.innerHTML = `
          <div class="detail-stat"><span class="detail-stat-label">Amount</span><span class="detail-stat-value">$${fmtNum(trace.term_sheet.amount || trace.term_sheet.principal || 0)}</span></div>
          <div class="detail-stat"><span class="detail-stat-label">Rate</span><span class="detail-stat-value">${rateStr}%</span></div>
          <div class="detail-stat"><span class="detail-stat-label">Term</span><span class="detail-stat-value">${trace.term_sheet.term_months || '—'}mo</span></div>
        `;
        return el;
      }, true));
    }

    // Cost details
    if (trace.tokens_in > 0 || trace.cost_usd > 0) {
      body.appendChild(this._makeSection('Cost & Tokens', () => {
        const el = document.createElement('div');
        el.className = 'detail-grid';
        el.innerHTML = `
          <div class="detail-stat"><span class="detail-stat-label">Cost</span><span class="detail-stat-value">$${(trace.cost_usd || 0).toFixed(4)}</span></div>
          <div class="detail-stat"><span class="detail-stat-label">Tokens In</span><span class="detail-stat-value">${fmtNum(trace.tokens_in || 0)}</span></div>
          <div class="detail-stat"><span class="detail-stat-label">Tokens Out</span><span class="detail-stat-value">${fmtNum(trace.tokens_out || 0)}</span></div>
          <div class="detail-stat"><span class="detail-stat-label">Tool Calls</span><span class="detail-stat-value">${trace.tool_calls_count || 0}</span></div>
        `;
        return el;
      }, false));
    }

    // Original trace data (from file-based traces)
    if (trace.system_prompt) body.appendChild(this._makeSection('System Prompt', () => pre(trace.system_prompt), false));
    if (trace.user_prompt) body.appendChild(this._makeSection('User Prompt', () => pre(trace.user_prompt), false));
    if (trace.tool_calls?.length) {
      body.appendChild(this._makeSection(`Tool Calls (${trace.tool_calls.length})`, () => {
        const frag = document.createDocumentFragment();
        for (const tc of trace.tool_calls) {
          const block = document.createElement('div');
          block.className = 'tool-call-block'; block.textContent = tc;
          frag.appendChild(block);
        }
        if (trace.tool_rounds !== undefined) {
          const badge = document.createElement('div');
          badge.className = 'tool-rounds-badge';
          badge.textContent = `${trace.tool_rounds} tool round(s)`;
          frag.appendChild(badge);
        }
        return frag;
      }, true));
    }
    if (trace.raw_response) body.appendChild(this._makeSection('Raw Response', () => pre(trace.raw_response), true));
    if (trace.parsed_json) body.appendChild(this._makeSection('Parsed JSON', () => pre(JSON.stringify(trace.parsed_json, null, 2), 'json-block'), true));
    if (trace.error) body.appendChild(this._makeSection('Error', () => pre(typeof trace.error === 'string' ? trace.error : JSON.stringify(trace.error, null, 2), 'error-block'), true));

    card.appendChild(body);
    return card;
  }

  _makeSection(label, buildContent, startOpen) {
    const section = document.createElement('div');
    section.className = 'trace-section' + (startOpen ? ' open' : '');

    const toggle = document.createElement('div');
    toggle.className = 'trace-section-toggle';
    toggle.innerHTML = `<span class="trace-section-arrow">&#x25B6;</span><span class="trace-section-label">${esc(label)}</span>`;

    const wrap = document.createElement('div');
    wrap.className = 'trace-section-content';

    let built = false;
    const ensureBuilt = () => { if (!built) { wrap.appendChild(buildContent()); built = true; } };
    if (startOpen) ensureBuilt();

    toggle.addEventListener('click', () => { section.classList.toggle('open'); ensureBuilt(); });
    section.appendChild(toggle);
    section.appendChild(wrap);
    return section;
  }
}

// ---- Helpers ----

function esc(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function fmtNum(n) {
  return Math.round(n).toLocaleString('en-US');
}

function pre(text, extraClass = '') {
  const el = document.createElement('pre');
  el.className = 'trace-pre' + (extraClass ? ' ' + extraClass : '');
  el.textContent = text;
  return el;
}

// Boot
new App();
