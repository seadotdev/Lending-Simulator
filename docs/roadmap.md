# Loanville2 Roadmap

## Current State of Main (post-merge)

### What's Solid

| Layer | Components | Status |
|-------|-----------|--------|
| **Data** | 36 hand-crafted borrowers, 8 mix presets, procedural gen for seasons | Complete |
| **Engine** | Origination → adjudication → resolution pipeline, LOS-only eval path | Complete |
| **Scoring** | RAROC with funding cost, fraud/concentration penalties, 3 economics presets | Complete |
| **Elo** | Triple Elo (Profit/Credit/DealShare), composite blend, ±40 caps, epsilon validation | Complete |
| **Season** | Multi-week loop with capital decay, adequacy elimination, briefings, tooling phase, escalating mixes | Complete |
| **Leaderboard** | Git-native JSON match files, deterministic replay, 13 matches across ~15 models | Complete |
| **Web** | Town animation viewer for seasons | Shipped |
| **Infrastructure** | Champion/challenger model, scorecard (3-layer gates), flywheel CLI, run schema, cost tracking | Structural |

### What's Broken

**1. Composite Elo never emitted to leaderboard.json**
`compute_composite_elo()` exists in code. `leaderboard/core.py` imports it. But standings output has no `composite_elo` field — every model lacks its single-number ranking.

**2. Leaderboard Elo replay uses placeholder payoffs**
`core.py:196,211` stubs `per_applicant_payoffs` with `0.0`, so when standings are recomputed from match history, the triple Elo ratings barely move. The actual data is in the match JSON files but isn't extracted during replay.

**3. Zero passing tests on main**
`test_lite_scenarios.py` imports `loanville.llm` (moved to `legacy/llm.py` in PR #26). `test_scoring_scenarios.py` has no `test_` functions (it's a script, not pytest). The other 4 test files need external fixtures or APIs. Effectively no CI safety net.

**4. Info asymmetry is a no-op in production**
`_create_borrower_views()` only applies on the direct-LLM path in `engine.py`. Since PR #26 made LOS the default, the `info_asymmetry` config does nothing in live runs — the dossier sent to `/v1/underwrite` is always unfiltered.

**5. Custom tools are cosmetic in live mode**
Mock mode auto-generates tools. Live mode has a `pass` statement. `LenderToolkit.get_tool_definitions()` output is never injected into LOS requests or lender personas.

---

## Roadmap

### Tier 1 — Fix What's Broken (highest ROI, lowest effort)

| # | Item | Why It Matters |
|---|------|---------------|
| **1** | **Emit composite Elo to leaderboard standings** | The single-number ranking is the whole point of the composite. One function call missing in `core.py`. |
| **2** | **Fix Elo replay from match history** | Without real payoff extraction, recomputing the leaderboard from match files produces flat ratings. The data is there, the deserialization isn't. |
| **3** | **Fix or replace broken test suite** | Zero passing tests = zero safety net. At minimum: fix the `loanville.llm` import, add a mock-mode integration test that exercises engine → scoring → season in one pass. |

### Tier 2 — Activate Half-Built Features (high leverage)

| # | Item | Why It Matters |
|---|------|---------------|
| **4** | **Season → leaderboard bridge** | Season mode produces rich per-week match data but never emits leaderboard records. A `season_week_to_match_record()` adapter would multiply match volume for Elo stability — every season week becomes a rated match. |
| **5** | **Wire info asymmetry through LOS adapter** | `serialize_dossier()` in `los_adapter.py` needs to accept pre-filtered borrower views. Without this, the `partial_statements` and `redacted` modes are dead code for all live runs. |
| **6** | **Inject custom tools into LOS evaluation** | Append `toolkit.get_tool_definitions()` to the lender persona or LOS request payload. This turns the tooling phase from bookkeeping into actual strategy evolution — the core Skirmish mechanic. |
| **7** | **Wire inter-match feedback in tournament loop** | `_build_feedback_prompt()` and `make_lender(feedback=...)` exist, but `run_tournament()` doesn't pass previous-round results through. One dict plumbing fix to close the coaching loop. |

### Tier 3 — Deepen the Game (strategic)

| # | Item | Why It Matters |
|---|------|---------------|
| **8** | **Strategy memo phase** | Between season weeks, prompt each lender to write a brief strategy statement ("I will tighten fraud checks and lower rates on tech deals"). This creates observable adaptation and makes the season narrative legible. |
| **9** | **Adversarial borrower generation** | Current borrowers are static. An adversarial generator that reads the leaderboard's confusion matrices and crafts borrowers targeting the top model's blind spots (e.g., "this model never catches structured deposits") would create Red Queen dynamics. |
| **10** | **Dynamic concentration enforcement** | Sector limits are only penalized at final scoring. Enforcing them during adjudication (reject a deal if it would breach concentration) creates real-time portfolio management pressure. |
| **11** | **Prepayment modeling** | Good loans can prepay (the hash-based mechanism exists) but there's no lender-visible impact. Making prepayment reduce realized yield would create an asymmetry: aggressive pricing wins deals but risks early payoff. |

### Tier 4 — Research Extensions

| # | Item | What It Tests |
|---|------|--------------|
| **12** | **Syndicated lending** | Two lenders split a large loan. Tests negotiation and cooperation within competition. |
| **13** | **Market regime shifts** | Mid-season shocks (rate hikes, sector downturns) that change ground truth. Tests regime-change detection vs. anchoring on stale priors. |
| **14** | **Explainability scoring** | Score reasoning trace quality, not just the decision. "Approved because DSCR > 1.5x" should beat "approved because it seems good" even when both approve correctly. |
| **15** | **Multi-season meta-learning** | Run 3 seasons, carry Elo + champion policy across. Tests whether models improve over longer horizons or regress to mean. |

---

## Recommended Sequence

**1 → 2 → 3** (one session, pure fixes) → **4 → 7** (close the data flywheel) → **5 → 6** (activate dead features) → **8 → 9** (make seasons interesting) → rest as research appetite allows.

Items 1-3 are quick fixes that unblock everything downstream. Item 4 is probably the single highest-leverage new feature — it turns every season week into Elo training data.
