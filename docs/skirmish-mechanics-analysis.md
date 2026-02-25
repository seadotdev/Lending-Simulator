# Skirmish vs Loanville2: ELO, Turn Mechanics, and Game Design Lessons

## Context

Analysis of [LLM Skirmish](https://github.com/llmskirmish/skirmish) — a Screeps-inspired RTS game where LLMs write JavaScript strategies to compete in real-time combat — compared to Loanville2's competitive lending simulation. The goal is to identify game mechanics insights we can adopt, particularly around ELO rating and temporal/turn structure.

---

## 1. ELO Systems Compared

### Skirmish: Single ELO, Clean and Simple
- **One rating**: Standard Elo (K=32, Initial=1500)
- **Win/Loss/Draw**: 1.0 / 0.0 / 0.5
- **One signal per match**: The match outcome is the only input
- **Sequential processing**: Matches sorted chronologically, updates applied in order
- **Victory is unambiguous**: Either you destroyed the spawn or you didn't (timeout → score comparison)

### Loanville2: Triple ELO with Per-Applicant Pairwise Signals
- **Three ratings**: Profit Elo, Credit Elo, DealShare Elo
- **Per-applicant signals**: For N borrowers in a 3-way match → `N × C(3,2) = 3N` pairwise updates
- **K-factor scaling**: `pair_k = K / ((n-1) × N)` to bound total movement
- **Tie epsilon**: $500 utility band for Profit Elo ties
- **Same underlying formula**: `E(A) = 1 / (1 + 10^((R_B - R_A) / 400))`

### Key Differences

| Aspect | Skirmish | Loanville2 |
|--------|----------|------------|
| Signals per match | 1 | 3N (e.g., 48 for 16 borrowers × 3 pairs) |
| Rating dimensions | 1 | 3 (Profit, Credit, DealShare) |
| Match format | 1v1 | 3-way (triplets) |
| Convergence | ~30-50 matches | ~50+ matches (faster per-match due to signal density) |
| Interpretability | Instant ("who wins fights") | Requires explanation ("why 3 ratings?") |
| Win definition | Destroy spawn or outscore | Per-borrower utility comparison |

### What We Can Learn

**a) Skirmish proves that a single, legible rating works for engagement.** Their leaderboard is one number. People understand it immediately. Our 3-Elo system is analytically richer but creates confusion about "which rating matters."

**Actionable idea**: Consider a **composite Elo** as the primary public-facing metric (e.g., weighted combination: 50% Profit + 30% Credit + 20% DealShare) while keeping the three dimensions as drill-down analytics. This gives us Skirmish's leaderboard simplicity without losing our analytical depth.

**b) Skirmish's 1-signal-per-match is noisier but more robust.** Our per-applicant decomposition gives faster convergence but introduces correlated signals (a model that's universally bad loses on every borrower, creating an avalanche of negative updates in one match). Skirmish's approach naturally bounds single-match volatility.

**Actionable idea**: Consider adding an **Elo movement cap per match** (e.g., max ±40 points per match regardless of borrower count) to prevent catastrophic single-match swings, similar to how Skirmish's single-signal design naturally provides this.

**c) Skirmish's draw handling is clean.** Draw = 0.5, no epsilon. Our $500 utility epsilon for ties is a necessary complexity given financial outcomes, but it introduces a tuning parameter that could bias results. Worth validating that the epsilon doesn't systematically favor conservative or aggressive strategies.

---

## 2. Turn/Tick Mechanics Compared

### Skirmish: Real-Time Tick Loop (2,000 ticks)
```
Each tick:
  Phase 1: Intent Processing (all players, alternating priority)
  Phase 2: Movement Resolution (collision detection)
  Phase 3: Object Ticks (lifecycle updates)
```
- **Simultaneous execution**: All actions processed together each tick
- **Alternating priority**: Player order alternates by tick for fairness
- **Deterministic**: Same inputs → same outputs (sorted by ID for consistency)
- **2,000 tick limit**: Creates natural time pressure
- **Continuous economy**: Energy regenerates (10/tick), harvesting (2/tick), spawning (3 ticks/part)

### Loanville2: Parallel Evaluation → Sequential Resolution
```
Single-run:
  Phase 1-2: All lenders evaluate all borrowers (parallel, no interaction)
  Phase 3: Adjudication (lowest rate wins, capital limits enforced)
  Phase 4: Booking
  Phase 5: Resolution (fast-forward 24 months)

Season mode:
  Week N: Resolve existing loans → Portfolio briefing → New cohort → Adjudicate → Book
  Repeat for 10 weeks
```
- **No real interaction between agents**: Decisions are independent and simultaneous
- **One-shot decisions**: No ability to react to opponent moves
- **Season mode adds temporal depth**: Multi-week capital carryforward
- **No time pressure within a decision** (except optional latency scoring)

### Key Differences

| Aspect | Skirmish | Loanville2 |
|--------|----------|------------|
| Temporal granularity | 2,000 discrete ticks | 1 decision point per borrower (or 10 weeks in season) |
| Agent interaction | Real-time reactive (see enemy, respond) | Blind simultaneous (no visibility of competitors) |
| Adaptation within match | Continuous (every tick) | None (single-run) / Limited (season mode) |
| Strategic depth | Spatial + temporal + economic | Analytical + pricing + portfolio |
| Feedback frequency | Every tick (immediate) | End of match (delayed) |
| Time pressure | Hard (2,000 tick cap) | Soft (latency scoring optional) |

### What We Can Learn

**a) Skirmish's multi-round iteration loop is the biggest gap.** In Skirmish, LLMs play a match → receive logs → reflect on performance → write improved strategy → play again. This creates a **self-improvement flywheel** that Loanville2's tournament mode lacks.

Currently in Loanville2, each match is independent — a model doesn't learn from its previous matches within a tournament. Season mode adds week-to-week portfolio context, but models don't get feedback on their decision quality or competitor behavior.

**Actionable idea**: Implement a **tournament iteration mode** where between matches, models receive:
- Their confusion matrix from the previous match
- Whether they lost deals (and to what rate)
- Which borrowers defaulted vs performed
- A brief "coaching prompt" asking them to adjust strategy

This directly mirrors Skirmish's `NEXT_ROUND.md` prompt pattern where LLMs review match logs and previous strategies before generating improved ones.

**b) Skirmish's alternating priority creates fairness in simultaneous execution.** Their tick processor alternates which player's intents are processed first, preventing systematic first-mover advantage.

In Loanville2, adjudication is already fair (lowest rate wins regardless of order), but the LLM evaluation order could theoretically affect API rate limits or context window behavior. Worth confirming this isn't introducing bias.

**c) Skirmish's tick limit creates natural pacing.** The 2,000-tick cap forces strategic trade-offs between early aggression (rush strategy) and late-game economy (boom strategy).

**Actionable idea**: In season mode, consider a **capital decay or time-value mechanic** where undeployed capital slowly loses value (beyond the existing volume floor penalty). This creates Skirmish-style tension between "deploy early" and "wait for better opportunities."

---

## 3. Strategic Dynamics Compared

### Skirmish: Code Generation + Spatial Strategy
- LLMs **write JavaScript code** that executes autonomously
- Strategy space: unit composition, attack timing, economy balance, spatial positioning
- Classic RTS archetypes emerge: rush, turtle, boom, all-in
- **The LLM's job**: Author a program, not make individual decisions
- **Emergent complexity**: Simple rules create deep strategy through unit interactions

### Loanville2: Decision-Making Under Uncertainty
- LLMs **analyze financial data** and make approve/reject decisions
- Strategy space: risk tolerance, pricing aggression, fraud detection, portfolio construction
- Classic lending archetypes emerge: conservative, aggressive, balanced
- **The LLM's job**: Make informed decisions per borrower
- **Emergent complexity**: Competitive pricing, capital allocation, portfolio diversification

### What We Can Learn

**a) Skirmish's "code as strategy" creates much richer strategy differentiation.** Two LLMs given the same Skirmish rules will write vastly different code. Two LLMs given the same Loanville borrower will often make similar decisions (especially on obviously good/bad borrowers). The strategy space is narrower.

**Actionable idea**: Increase strategic differentiation by:
- Adding **persona-driven strategy constraints** (e.g., one lender must maintain sector focus, another has higher capital but stricter risk limits)
- Introducing **information asymmetry** (different lenders see different subsets of borrower data)
- Adding **custom tool development** (already in SeasonConfig as `custom_tools: bool`) where models can build analysis tools that carry across rounds, similar to Skirmish's code iteration

**b) Skirmish's scoring at timeout is an elegant tiebreaker.** When no spawn is destroyed, the remaining health/unit count determines the winner. This prevents draws from being too common while still rewarding defensive play.

In Loanville2, the RAROC scoring already serves this purpose well. But Skirmish's approach suggests we could add a simpler, more intuitive "who's ahead?" metric for real-time tournament visualization (e.g., running P&L during season mode displayed as a live score).

**c) Skirmish's surrender mechanic acknowledges lost positions.** If an LLM recognizes it's losing, it can surrender. In Loanville2, a model can't opt out mid-season — it must continue making decisions even if hopelessly behind on capital.

**Actionable idea**: In season mode, consider a **capital adequacy threshold** below which a lender is automatically eliminated (similar to real banking). This creates genuine elimination dynamics and prevents zombie lenders from noise-polluting the tournament.

---

## 4. Concrete Recommendations (Prioritized)

### High Priority
1. **Add a composite Elo for leaderboard display** — Single number for public ranking, triple Elo as analytical detail
   - Files: `legacy/benchmarks/elo_benchmark.py`, `loanville/leaderboard/core.py`
   - Formula: `composite = 0.50 * profit_elo + 0.30 * credit_elo + 0.20 * dealshare_elo`

2. **Implement inter-match feedback in tournaments** — Give models their previous match results as context for the next match (Skirmish's core iteration loop)
   - Files: `legacy/benchmarks/elo_benchmark.py` (run_match), `loanville/llm.py` (prompt construction)
   - Pattern: After each match, append a "previous performance summary" to the system prompt

3. **Add per-match Elo movement caps** — Prevent catastrophic single-match swings from correlated per-applicant signals
   - Files: `legacy/benchmarks/elo_benchmark.py` (update_*_elo functions)
   - Implementation: After computing all per-borrower updates for a match, clamp total delta to +/-40

### Medium Priority
4. **Season mode capital adequacy elimination** — Lenders below threshold are eliminated, creating genuine stakes
   - Files: `loanville/season.py`
   - Threshold: e.g., capital < 20% of initial -> eliminated

5. **Information asymmetry modes** — Different lenders see different data subsets per borrower
   - Files: `loanville/engine.py`, `loanville/data.py`
   - Modes: Full/partial bank statements, delayed financials, redacted fields

6. **Live scoring dashboard for season mode** — Real-time "who's ahead?" view inspired by Skirmish's tick-by-tick game state
   - Files: New visualization on top of `loanville/season.py` state snapshots

### Lower Priority
7. **Validate $500 epsilon neutrality** — Ensure UTILITY_EPSILON doesn't systematically favor conservative/aggressive strategies
8. **Capital time-value decay** — Undeployed capital loses value over season weeks beyond volume floor
9. **Custom tool development phase** — Enable `custom_tools` in season mode for LLM-authored analysis tools

---

## 5. Summary Table

| Skirmish Mechanic | Loanville2 Equivalent | Gap / Opportunity |
|---|---|---|
| Single Elo (K=32) | Triple Elo (Profit/Credit/DealShare) | Add composite for public display |
| 2,000-tick matches | Single-shot evaluation | Season mode partially addresses; add iteration feedback |
| Code iteration between rounds | No inter-match learning | Implement match-to-match feedback prompts |
| Alternating tick priority | Parallel evaluation | Already fair by design; verify API ordering bias |
| Spawn destruction win condition | RAROC/Elo ranking | Consider capital elimination in season mode |
| Draw = 0.5 | $500 epsilon tie band | Validate epsilon isn't introducing bias |
| Energy economy + harvesting | Capital deployment + interest | Add capital time-value mechanics |
| Surrender mechanic | No opt-out | Capital adequacy elimination serves same purpose |
| JavaScript strategy authoring | JSON decision outputs | Custom tool development (already planned) |
| 100x100 spatial grid | 24 borrower pool | Information asymmetry adds strategic "terrain" |
