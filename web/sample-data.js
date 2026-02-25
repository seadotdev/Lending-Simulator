// ============================================================================
// Loanville Arena — Sample Data
// 8 models, 10 matches, 15 borrowers per match, full Elo histories
// ============================================================================

const SAMPLE_DATA = (() => {
  // --- Model Registry ---
  const MODELS = [
    { model_id: "deepseek/deepseek-v3", display_name: "DeepSeek V3", short: "DeepSeek V3", color: "#7C3AED", icon: "DS" },
    { model_id: "google/gemini-2.5-flash", display_name: "Gemini 2.5 Flash", short: "Gemini 2.5", color: "#3B82F6", icon: "GF" },
    { model_id: "meta-llama/llama-3.3-70b", display_name: "Llama 3.3 70B", short: "Llama 70B", color: "#F59E0B", icon: "LL" },
    { model_id: "anthropic/claude-3.5-haiku", display_name: "Claude 3.5 Haiku", short: "Claude Haiku", color: "#D97706", icon: "CH" },
    { model_id: "qwen/qwen-2.5-72b", display_name: "Qwen 2.5 72B", short: "Qwen 72B", color: "#10B981", icon: "QW" },
    { model_id: "mistralai/mistral-large", display_name: "Mistral Large", short: "Mistral Lg", color: "#EF4444", icon: "ML" },
    { model_id: "openai/gpt-4o-mini", display_name: "GPT-4o Mini", short: "GPT-4o Mini", color: "#06B6D4", icon: "GP" },
    { model_id: "nvidia/nemotron-70b", display_name: "Nemotron 70B", short: "Nemotron", color: "#84CC16", icon: "NM" },
  ];

  // --- Borrower Templates ---
  const BORROWERS = [
    { id: "bor_aero_001", name: "SkyBridge Aero-Logistics", sector: "Aero-Logistics", outcome: "good", loan_amount: 450000 },
    { id: "bor_bio_002", name: "GeneSynth Biologics", sector: "Bio-Synthetics", outcome: "good", loan_amount: 320000 },
    { id: "bor_quantum_003", name: "QubitCore Computing", sector: "Quantum Computing", outcome: "good", loan_amount: 580000 },
    { id: "bor_green_004", name: "Verdant Solar Systems", sector: "Green Energy", outcome: "good", loan_amount: 275000 },
    { id: "bor_fin_005", name: "NexPay Financial", sector: "FinTech", outcome: "good", loan_amount: 390000 },
    { id: "bor_med_006", name: "PulsePoint Medical", sector: "MedTech", outcome: "good", loan_amount: 510000 },
    { id: "bor_cyber_007", name: "IronClad Security", sector: "Cybersecurity", outcome: "good", loan_amount: 340000 },
    { id: "bor_log_008", name: "SwiftRoute Delivery", sector: "Logistics", outcome: "good", loan_amount: 220000 },
    { id: "bor_mfg_009", name: "TitanForge Manufacturing", sector: "Advanced Mfg", outcome: "bad", loan_amount: 480000 },
    { id: "bor_retail_010", name: "LuxeVault Retail", sector: "Retail Tech", outcome: "bad", loan_amount: 360000 },
    { id: "bor_prop_011", name: "ClearView Properties", sector: "PropTech", outcome: "bad", loan_amount: 520000 },
    { id: "bor_food_012", name: "FarmDirect Foods", sector: "AgriTech", outcome: "bad", loan_amount: 190000 },
    { id: "bor_ghost_013", name: "Phantom Trade Corp", sector: "Import/Export", outcome: "fraud", loan_amount: 670000 },
    { id: "bor_shell_014", name: "Oceanic Holdings LLC", sector: "Maritime", outcome: "fraud", loan_amount: 430000 },
    { id: "bor_wash_015", name: "GreenWash Industries", sector: "Green Energy", outcome: "fraud", loan_amount: 550000 },
  ];

  // --- Seeded RNG for reproducibility ---
  let _seed = 42;
  function seededRandom() {
    _seed = (_seed * 16807 + 0) % 2147483647;
    return (_seed - 1) / 2147483646;
  }
  function randRange(min, max) { return min + seededRandom() * (max - min); }
  function randInt(min, max) { return Math.floor(randRange(min, max + 1)); }
  function pick(arr) { return arr[randInt(0, arr.length - 1)]; }

  // --- Model Behavior Profiles ---
  // Each model has tendencies that shape their decisions
  const PROFILES = {
    "deepseek/deepseek-v3":       { approve_good: 0.92, approve_bad: 0.45, approve_fraud: 0.35, rate_base: 7.5, rate_spread: 2.5, aggression: 0.85 },
    "google/gemini-2.5-flash":    { approve_good: 0.88, approve_bad: 0.30, approve_fraud: 0.15, rate_base: 8.5, rate_spread: 3.0, aggression: 0.65 },
    "meta-llama/llama-3.3-70b":   { approve_good: 0.78, approve_bad: 0.20, approve_fraud: 0.10, rate_base: 9.5, rate_spread: 3.5, aggression: 0.40 },
    "anthropic/claude-3.5-haiku": { approve_good: 0.85, approve_bad: 0.25, approve_fraud: 0.08, rate_base: 8.8, rate_spread: 3.2, aggression: 0.55 },
    "qwen/qwen-2.5-72b":         { approve_good: 0.82, approve_bad: 0.38, approve_fraud: 0.28, rate_base: 8.0, rate_spread: 2.8, aggression: 0.70 },
    "mistralai/mistral-large":    { approve_good: 0.84, approve_bad: 0.22, approve_fraud: 0.05, rate_base: 9.0, rate_spread: 3.3, aggression: 0.50 },
    "openai/gpt-4o-mini":         { approve_good: 0.80, approve_bad: 0.42, approve_fraud: 0.30, rate_base: 7.8, rate_spread: 2.6, aggression: 0.75 },
    "nvidia/nemotron-70b":        { approve_good: 0.86, approve_bad: 0.28, approve_fraud: 0.12, rate_base: 8.6, rate_spread: 3.1, aggression: 0.60 },
  };

  // --- Generate Match Decisions ---
  function generateMatchDecisions(matchModels, borrowers) {
    const decisions = {};
    for (const model of matchModels) {
      const profile = PROFILES[model.model_id];
      decisions[model.model_id] = {};

      for (const b of borrowers) {
        let approveProb;
        if (b.outcome === "good") approveProb = profile.approve_good;
        else if (b.outcome === "bad") approveProb = profile.approve_bad;
        else approveProb = profile.approve_fraud;

        // Add some randomness
        approveProb += randRange(-0.15, 0.15);
        approveProb = Math.max(0.05, Math.min(0.98, approveProb));

        const approved = seededRandom() < approveProb;
        const rate = approved
          ? Math.max(4.0, profile.rate_base + randRange(-profile.rate_spread, profile.rate_spread * 0.5)) / 100
          : null;

        decisions[model.model_id][b.id] = {
          approved,
          rate_offered: rate,
          decision_state: approved ? "pending" : "declined",
          ground_truth: b.outcome,
          utility: 0,
        };
      }
    }

    // Adjudicate: borrower picks lowest rate among approvers
    for (const b of borrowers) {
      let bestRate = Infinity;
      let bestModel = null;

      for (const model of matchModels) {
        const d = decisions[model.model_id][b.id];
        if (d.approved && d.rate_offered < bestRate) {
          bestRate = d.rate_offered;
          bestModel = model.model_id;
        }
      }

      for (const model of matchModels) {
        const d = decisions[model.model_id][b.id];
        if (!d.approved) continue;
        if (model.model_id === bestModel) {
          d.decision_state = "won";
          // Compute utility
          if (b.outcome === "good") {
            d.utility = b.loan_amount * d.rate_offered * 2; // 2yr interest
          } else if (b.outcome === "bad") {
            d.utility = -(b.loan_amount * 0.6); // partial loss
          } else {
            d.utility = -(b.loan_amount * 0.95); // fraud loss
          }
        } else {
          d.decision_state = "lost";
        }
      }
    }

    return decisions;
  }

  // --- Compute Match Results ---
  function computeMatchResults(matchModels, borrowers, decisions) {
    const results = [];

    for (const model of matchModels) {
      const cm = { good: { approved: 0, rejected: 0 }, bad: { approved: 0, rejected: 0 }, fraud: { approved: 0, rejected: 0 } };
      let dealsWon = 0, dealsRejected = 0, dealsLost = 0;
      let deployed = 0, netPnl = 0, fraudsFunded = 0, defaults = 0;
      const perBorrower = {};

      for (const b of borrowers) {
        const d = decisions[model.model_id][b.id];
        const action = d.approved ? "approved" : "rejected";
        cm[b.outcome][action]++;

        if (d.decision_state === "won") {
          dealsWon++;
          deployed += b.loan_amount;
          if (b.outcome === "good") {
            netPnl += b.loan_amount * d.rate_offered * 2;
          } else if (b.outcome === "bad") {
            netPnl -= b.loan_amount * 0.6;
            defaults++;
          } else {
            netPnl -= b.loan_amount * 0.95;
            fraudsFunded++;
            defaults++;
          }
        } else if (d.decision_state === "declined") {
          dealsRejected++;
        } else {
          dealsLost++;
        }

        perBorrower[b.id] = {
          decision_state: d.decision_state,
          ground_truth: b.outcome,
          utility: d.utility,
          rate_offered: d.rate_offered,
        };
      }

      // RAROC calculation (simplified)
      const capital = 5000000;
      const fundingCost = deployed * 0.04 * 2;
      const riskFreeReturn = capital * 0.05 * 2;
      const rawReturn = netPnl - fundingCost;
      const raroc = ((rawReturn - riskFreeReturn * 0.5) / Math.max(capital * 0.1, 1)) * 100;

      results.push({
        model_id: model.model_id,
        raroc_score: Math.round(raroc * 100) / 100,
        deals_won: dealsWon,
        deals_rejected: dealsRejected,
        deals_lost: dealsLost,
        deals_errored: 0,
        frauds_funded: fraudsFunded,
        defaults: defaults,
        deployed: deployed,
        net_pnl: Math.round(netPnl),
        confusion_matrix: cm,
        per_borrower: perBorrower,
      });
    }

    return results;
  }

  // --- Generate All Matches ---
  const MIXES = ["balanced", "realistic", "hard", "fraud", "analyst", "stress", "easy", "balanced", "hard", "realistic"];

  function generateMatches() {
    const matches = [];
    const baseDate = new Date("2025-11-01T14:00:00Z");

    for (let i = 0; i < 10; i++) {
      _seed = 42 + i * 137; // Reset seed per match for variety

      // Pick 3 models for this match (rotating roster)
      const roster = [];
      const startIdx = (i * 2) % MODELS.length;
      for (let j = 0; j < 3; j++) {
        roster.push(MODELS[(startIdx + j) % MODELS.length]);
      }

      // Shuffle borrowers slightly per match
      const matchBorrowers = [...BORROWERS].sort(() => seededRandom() - 0.5);
      const subset = matchBorrowers.slice(0, randInt(12, 15));

      const decisions = generateMatchDecisions(roster, subset);
      const results = computeMatchResults(roster, subset, decisions);

      const matchDate = new Date(baseDate.getTime() + i * 3 * 24 * 60 * 60 * 1000 + randInt(0, 8) * 3600000);
      const timestamp = matchDate.toISOString().replace(/\.\d{3}Z/, "Z");
      const tsafe = timestamp.replace(/:/g, "-").slice(0, 19);
      const hash = (i * 7 + 0xab3f).toString(16).padStart(6, "0").slice(0, 6);
      const matchId = `${tsafe}_${hash}`;

      matches.push({
        match_id: matchId,
        timestamp_utc: timestamp,
        mix: MIXES[i],
        n_borrowers: subset.length,
        models: roster.map(m => ({ model_id: m.model_id, display_name: m.display_name })),
        results: results,
        validation: { valid: true, errors: [] },
      });
    }

    return matches;
  }

  // --- Compute Leaderboard from Matches ---
  function computeLeaderboard(matches) {
    const K = 32;
    const INITIAL = 1500;

    const profitElo = {};
    const creditElo = {};
    const dealshareElo = {};
    const matchCounts = {};
    const totalRaroc = {};
    const aggConfusion = {};
    const rateStats = {};
    const eloHistory = {};
    const allModels = {};
    const matchIds = [];

    // Init all models
    for (const m of MODELS) {
      const mid = m.model_id;
      allModels[mid] = m.display_name;
      profitElo[mid] = INITIAL;
      creditElo[mid] = INITIAL;
      dealshareElo[mid] = INITIAL;
      matchCounts[mid] = 0;
      totalRaroc[mid] = 0;
      aggConfusion[mid] = { good: { approved: 0, rejected: 0 }, bad: { approved: 0, rejected: 0 }, fraud: { approved: 0, rejected: 0 } };
      rateStats[mid] = { good_rates: [], bad_rates: [] };
      eloHistory[mid] = [];
    }

    // Elo update helpers
    function expectedScore(ra, rb) { return 1 / (1 + Math.pow(10, (rb - ra) / 400)); }

    function updatePairwise(ratings, results, scoreKey) {
      const newRatings = { ...ratings };
      for (let i = 0; i < results.length; i++) {
        for (let j = i + 1; j < results.length; j++) {
          const a = results[i].model_id;
          const b = results[j].model_id;
          const sa = results[i][scoreKey];
          const sb = results[j][scoreKey];

          const ea = expectedScore(newRatings[a], newRatings[b]);
          const eb = 1 - ea;

          let wa, wb;
          if (sa > sb) { wa = 1; wb = 0; }
          else if (sa < sb) { wa = 0; wb = 1; }
          else { wa = 0.5; wb = 0.5; }

          newRatings[a] += K * (wa - ea);
          newRatings[b] += K * (wb - eb);
        }
      }
      return newRatings;
    }

    // Replay matches
    for (const match of matches) {
      matchIds.push(match.match_id);

      // Compute derived scores for Elo updates
      const resultsWithScores = match.results.map(r => {
        const cm = r.confusion_matrix;
        const totalDecisions = Object.values(cm).reduce((s, c) => s + c.approved + c.rejected, 0);
        const correctDecisions = cm.good.approved + cm.bad.rejected + cm.fraud.rejected;
        const creditScore = totalDecisions > 0 ? correctDecisions / totalDecisions : 0;

        return {
          model_id: r.model_id,
          profit_score: r.raroc_score,
          credit_score: creditScore * 100,
          dealshare_score: r.deals_won,
        };
      });

      // Update Elo ratings
      const profitResults = resultsWithScores.map(r => ({ model_id: r.model_id, score: r.profit_score }));
      const creditResults = resultsWithScores.map(r => ({ model_id: r.model_id, score: r.credit_score }));
      const dealshareResults = resultsWithScores.map(r => ({ model_id: r.model_id, score: r.dealshare_score }));

      const newProfit = updatePairwise(profitElo, profitResults, "score");
      const newCredit = updatePairwise(creditElo, creditResults, "score");
      const newDealshare = updatePairwise(dealshareElo, dealshareResults, "score");

      Object.assign(profitElo, newProfit);
      Object.assign(creditElo, newCredit);
      Object.assign(dealshareElo, newDealshare);

      // Aggregate stats
      for (const r of match.results) {
        const mid = r.model_id;
        matchCounts[mid]++;
        totalRaroc[mid] += r.raroc_score;

        for (const cat of ["good", "bad", "fraud"]) {
          for (const action of ["approved", "rejected"]) {
            aggConfusion[mid][cat][action] += (r.confusion_matrix[cat] || {})[action] || 0;
          }
        }

        for (const [bid, bdata] of Object.entries(r.per_borrower)) {
          if (bdata.rate_offered != null) {
            if (bdata.ground_truth === "good") rateStats[mid].good_rates.push(bdata.rate_offered);
            else rateStats[mid].bad_rates.push(bdata.rate_offered);
          }
        }
      }

      // Snapshot Elo for all models
      for (const mid of Object.keys(allModels)) {
        eloHistory[mid].push({
          match_id: match.match_id,
          profit_elo: Math.round(profitElo[mid] * 10) / 10,
          credit_elo: Math.round(creditElo[mid] * 10) / 10,
          dealshare_elo: Math.round(dealshareElo[mid] * 10) / 10,
        });
      }
    }

    // Build standings
    const standings = Object.keys(allModels).map(mid => {
      const n = matchCounts[mid];
      const gr = rateStats[mid].good_rates;
      const br = rateStats[mid].bad_rates;

      return {
        model_id: mid,
        display_name: allModels[mid],
        profit_elo: Math.round(profitElo[mid] * 10) / 10,
        credit_elo: Math.round(creditElo[mid] * 10) / 10,
        dealshare_elo: Math.round(dealshareElo[mid] * 10) / 10,
        matches_played: n,
        avg_raroc: n > 0 ? Math.round(totalRaroc[mid] / n * 100) / 100 : 0,
        confusion_agg: aggConfusion[mid],
        rate_analysis: {
          avg_rate_good: gr.length > 0 ? Math.round(gr.reduce((a, b) => a + b, 0) / gr.length * 10000) / 10000 : null,
          avg_rate_bad: br.length > 0 ? Math.round(br.reduce((a, b) => a + b, 0) / br.length * 10000) / 10000 : null,
          n_good_offers: gr.length,
          n_bad_offers: br.length,
        },
      };
    });

    standings.sort((a, b) => b.profit_elo - a.profit_elo);

    return {
      computed_at: new Date().toISOString(),
      n_matches: matchIds.length,
      config: { k: K, initial_elo: INITIAL, utility_epsilon: 500 },
      standings,
      match_ids: matchIds,
      elo_history: eloHistory,
    };
  }

  // --- Build Everything ---
  const matches = generateMatches();
  const leaderboard = computeLeaderboard(matches);

  return {
    models: MODELS,
    borrowers: BORROWERS,
    matches,
    leaderboard,
  };
})();
