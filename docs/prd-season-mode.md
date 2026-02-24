# PRD: Season Mode

**Status:** Draft
**Author:** Design Session
**Date:** 2026-02-22

---

## 1. Problem Statement

Loanville currently resets capital every match. Each lender starts with a clean book, so the benchmark only tests **application-level underwriting** — can the model spot fraud and price risk on a single dossier?

This leaves three critical lending competencies unmeasured:

| Competency | Why it matters | Current gap |
|---|---|---|
| **Concentration management over time** | Real lenders must manage sector exposure as their book grows | Exposure resets every match; no carry-forward |
| **Capital rationing** | Saying "yes" to deal #3 might mean saying "no" to a better deal #8 | All 12 deals arrive simultaneously; no sequencing pressure |
| **Path dependency** | Early defaults should change risk appetite for later deals | No memory between evaluations |

Additionally, the current system gives every lender the same two tools (`run_bash`, `analyse_bank_statements`). In practice, the best underwriters build reusable analytical workflows. We want to measure whether agents can **build custom tooling** that makes them faster and more accurate over time.

---

## 2. Goals

1. **Season = multi-round game** where capital, exposures, and loan outcomes carry forward across sequential borrower cohorts ("weeks")
2. **Speed-to-offer scoring** — fewer tool calls = faster offer = competitive advantage in deal adjudication
3. **Custom tool creation** — lenders can use Pi Agent Core to build reusable analysis tools that persist across weeks
4. **Preserve the existing single-match mode** — Season is additive, not a replacement

### Non-Goals (v1)

- Real-time multiplayer / interactive bidding
- Inter-lender communication or syndication
- Secondary market / loan trading
- Dynamic interest rate environment (macro shocks)

---

## 3. Design

### 3.1 Season Structure

A **Season** is a sequence of **Weeks**. Each week presents a new cohort of borrowers. State carries forward.

```
Season (e.g. 10 weeks)
├── Week 1: Cohort A (4-6 borrowers)
│   ├── Phase 1-2: Pipeline + Underwriting (tool calls counted)
│   ├── Phase 3: Deal Adjudication (speed-to-offer tiebreaker)
│   ├── Phase 4: Ledger Booking
│   └── Phase 5: Partial Resolution (mature loans only)
├── Week 2: Cohort B (4-6 borrowers)
│   ├── Carry-forward: capital, exposures, active loans
│   ├── Resolution: Week 1 loans age by 1 period
│   └── ... same phases ...
├── ...
└── Week 10: Cohort J
    └── Final Resolution: all remaining loans fast-forwarded
```

**Default configuration:**

| Parameter | Default | Rationale |
|---|---|---|
| `season_length` | 10 weeks | Long enough for portfolio effects; short enough for single API session |
| `cohort_size` | 4-6 borrowers/week | Smaller than current 12 to create scarcity and force rationing |
| `total_borrowers` | ~50 | Drawn from an expanded pool (current 18 + new additions) |
| `resolution_cadence` | Rolling | Loans age each week; defaults/payoffs happen at their scheduled month |
| `starting_capital` | Same as current per-lender ($3M-$4M) | But it must now last the whole season |

### 3.2 Carry-Forward State

Between weeks, each lender's state persists:

```python
@dataclass
class SeasonLenderState:
    lender_id: str

    # Capital
    total_capital: float           # Fixed for season
    deployed_capital: float        # Sum of outstanding loan principals
    available_capital: float       # total - deployed (repayments add back)

    # Portfolio
    active_loans: list[ActiveLoan] # Loans not yet fully resolved
    resolved_loans: list[LoanOutcome]  # Completed (repaid or defaulted)

    # Exposure
    sector_exposure: dict[str, float]  # Current outstanding by sector

    # Cumulative P&L
    cumulative_interest: float     # All interest collected to date
    cumulative_losses: float       # All principal lost to date

    # Efficiency
    total_tool_calls: int          # Across all evaluations this season
    total_evaluations: int         # Number of borrowers evaluated

    # Custom tools (see 3.5)
    custom_tools: list[CustomTool] # Tools built by this lender
```

### 3.3 Rolling Loan Resolution

Instead of fast-forwarding all loans at the end, loans **age in real time** relative to the season clock:

- Each week advances the simulation clock by `N` months (configurable; default 2)
- Active loans collect payments or default according to their schedule
- When a "bad" loan hits its `months_before_default`, it defaults that week
- Fraud loans still default immediately upon booking
- Repaid principal returns to `available_capital`

This creates **information signals** lenders can react to:

```
Week 1: Lender funds Borrower X (bad, defaults at month 6)
Week 2: Borrower X makes payment → interest income, no alarm
Week 3: Borrower X makes payment → still performing
Week 4: Borrower X defaults → principal loss, capital freed (partially)
         Lender now knows: their fraud/risk detection was wrong
         Remaining capital and risk appetite should adjust
```

**Between-week briefing:** At the start of each week (after resolution), every lender receives an updated portfolio summary:

```
WEEKLY PORTFOLIO UPDATE (Week 5)
================================
Available Capital: $1,450,000 (was $1,800,000)
Active Loans: 7 (2 new last week, 1 defaulted)

EVENTS SINCE LAST WEEK:
  - LOAN-003 (SkyFreight): DEFAULT at month 6. Lost $180,000 principal.
  - LOAN-001 (NovaBio): Payment received. $12,400 interest. On track.
  - LOAN-005 (UrbanGreens): Payment received. $8,200 interest. On track.

CURRENT SECTOR EXPOSURE:
  - Aero-Logistics: 28.5% (limit: 30%) ⚠️ NEAR LIMIT
  - Bio-Synthetics: 15.2% (limit: 25%)
  - Green Energy: 8.0% (limit: 30%)

SEASON PERFORMANCE:
  - Net P&L: +$42,300
  - Deals Won: 5 / Rejected: 8 / Lost: 2
  - Defaults: 1 / Frauds Funded: 0
```

### 3.4 Speed-to-Offer Scoring

**Concept:** In real lending, speed matters. A borrower who gets an offer in 24 hours is less likely to shop around. We simulate this by counting tool calls as a proxy for processing time.

**Mechanics:**

Each lender's evaluation of a borrower produces a `tool_call_count`. This is used as a **tiebreaker in deal adjudication** and as a **scoring modifier**.

#### 3.4.1 Deal Adjudication with Speed

Current adjudication: sort by (lowest rate, highest amount).

New adjudication adds speed as a factor:

```
For each borrower with multiple offers:
  1. Filter to lenders with sufficient capital
  2. Score each offer:
     offer_score = rate_competitiveness + speed_bonus

     rate_competitiveness:
       normalized inverse of offered rate vs. range of offers

     speed_bonus:
       Offers made in ≤ 3 tool calls: +0.15 bonus
       Offers made in 4-6 tool calls: +0.05 bonus
       Offers made in 7+ tool calls:   0.00 bonus

  3. Highest offer_score wins
  4. Ties broken by: lower rate → higher amount → random
```

The speed bonus is calibrated so that rate still dominates — a lender can't win by being fast but expensive. But between similar offers, the faster lender wins, simulating borrower preference for quick decisions.

#### 3.4.2 Efficiency Score Component

At season end, efficiency contributes to the final score:

```
efficiency_metric = total_evaluations / total_tool_calls
```

This is reported but given a small weight (5-10% of final score) so it rewards efficiency without overwhelming credit quality.

### 3.5 Custom Tool Creation (Pi Agent Core)

**The big differentiator.** Lenders can create reusable tools using Pi Agent Core's tool-building primitives. These tools persist across weeks within a season, letting lenders build an "analyst toolkit" that gets better over time.

#### 3.5.1 How It Works

At the start of each week (before borrower evaluation), lenders get a **tooling phase** where they can create or update custom tools:

```
TOOLING PHASE (Week 3)
=======================
You may create or update custom analysis tools before this week's
applications arrive. Tools persist for the rest of the season.

Your current custom tools:
  1. revenue_trend_check - Analyzes 12-month deposit trends for growth/decline
  2. vendor_concentration - Flags single-vendor revenue dependency

You can:
  - CREATE a new tool (define name, description, bash implementation)
  - UPDATE an existing tool
  - SKIP and proceed to evaluations

Tool creation uses the Pi Agent Core create_tool function.
```

#### 3.5.2 Custom Tool Schema

```python
@dataclass
class CustomTool:
    name: str              # e.g. "fraud_pattern_scanner"
    description: str       # Natural language description for LLM
    implementation: str    # Bash script template
    created_week: int
    times_used: int = 0

    # The implementation is a bash script that:
    # - Reads from /data/bank_statements.json (always available)
    # - Can use jq, awk, grep, sort, uniq, etc.
    # - Outputs a structured result to stdout
    # - Runs in the same sandbox as run_bash
```

#### 3.5.3 Tool Creation Mechanics

Custom tools are implemented as **bash script wrappers** that the lender defines:

```json
{
  "name": "monthly_volatility",
  "description": "Calculate coefficient of variation for monthly deposits and withdrawals. Returns volatility score and flags months with >2 std dev swings.",
  "implementation": "jq '[.[].total_deposits] | {mean: (add/length), values: .} | .std = ((.values | map(. - .mean) | map(.*.) | add / length) | sqrt) | {volatility_cv: (.std/.mean), mean_deposits: .mean, std_deposits: .std}' /data/bank_statements.json"
}
```

When invoked during evaluation, the custom tool:
1. Executes the implementation script in the sandbox
2. Returns stdout to the LLM
3. **Counts as 1 tool call** (same as `run_bash`) for speed scoring

This means well-designed custom tools give the lender an efficiency advantage: one tool call does the work that might otherwise take 3-5 `run_bash` calls.

#### 3.5.4 Tool Creation Cost

Creating a tool has a cost to prevent spam:

- Each tool creation costs the equivalent of 2 tool calls toward the season's efficiency metric
- Tool updates cost 1 tool call equivalent
- Maximum 5 custom tools per lender per season
- Tool creation happens outside the evaluation timer (doesn't affect speed-to-offer on individual deals)

#### 3.5.5 Pi Agent Core Integration

The tooling phase uses Pi Agent Core's agent primitives:

```python
from pi_agent_core import Tool, ToolRegistry

class LenderToolkit:
    """Per-lender toolkit that persists across season weeks."""

    def __init__(self, lender_id: str):
        self.registry = ToolRegistry()
        self.lender_id = lender_id

        # Register base tools
        self.registry.register(run_bash_tool)

    def create_tool(self, name: str, description: str, implementation: str) -> Tool:
        """Create a custom tool backed by a bash script."""
        tool = Tool(
            name=name,
            description=description,
            handler=lambda args: self._run_script(implementation, args),
        )
        self.registry.register(tool)
        return tool

    def get_openai_tools(self) -> list[dict]:
        """Export all tools (base + custom) in OpenAI function-calling format."""
        return self.registry.to_openai_format()
```

The key insight: the LLM decides what tools to build based on patterns it noticed in previous weeks. A lender that funded a fraud in Week 2 might build a `round_number_detector` tool in Week 3. This creates an **emergent learning loop** without explicit training.

---

## 4. Scoring Changes

### 4.1 Season Score Formula

The season score extends the current match score with portfolio lifecycle and efficiency components:

```
Season Score = Credit Quality (70%) + Portfolio Management (20%) + Efficiency (10%)
```

**Credit Quality (70%)** — same formula as current match scoring, but accumulated over all weeks:
- Net P&L vs risk-free benchmark
- Fraud penalties (25% of fraud principal)
- Concentration penalties (5% of excess exposure)
- Yield drag

**Portfolio Management (20%)** — new dimensions:
- **Capital utilization**: % of capital deployed on average across the season (too low = missed opportunity, too high = over-leveraged)
- **Concentration discipline**: number of weeks where any sector was within 5% of its limit without breaching (shows awareness)
- **Recovery from defaults**: did the lender adjust behavior after experiencing a default?

**Efficiency (10%)** — speed and tooling:
- **Tool call efficiency**: evaluations per tool call (higher = better)
- **Custom tool adoption**: bonus for creating tools that are actually used in subsequent weeks
- **Speed-to-offer win rate**: % of competitive deals won via speed bonus

### 4.2 Portfolio Management Sub-Scores

```python
def score_portfolio_management(state: SeasonLenderState, season_length: int) -> float:
    """
    Score range: 0-100, mapped to 20% of final score.
    """
    score = 0.0

    # Capital Utilization (0-40 points)
    # Sweet spot: 60-85% average utilization
    avg_utilization = state.avg_utilization_pct
    if 60 <= avg_utilization <= 85:
        score += 40
    elif 40 <= avg_utilization < 60 or 85 < avg_utilization <= 95:
        score += 25
    else:
        score += 10

    # Concentration Discipline (0-30 points)
    # Reward staying within limits throughout the season
    weeks_with_violations = state.weeks_with_concentration_violations
    discipline_pct = 1 - (weeks_with_violations / season_length)
    score += discipline_pct * 30

    # Adaptive Behavior (0-30 points)
    # Did lender adjust after defaults?
    # Measured by: rejection rate increase in the 2 weeks after a default
    score += state.adaptation_score * 30

    return score
```

---

## 5. Borrower Pool Expansion

A 10-week season with 4-6 borrowers/week needs ~50 borrowers. The current pool has 18. We need to either:

**Option A: Procedural generation** — Generate borrower dossiers from templates with parameterized financials. Each season gets a unique but statistically consistent borrower population.

**Option B: Expanded static pool** — Hand-craft additional borrowers to reach 50+, maintaining the good/bad/fraud distribution.

**Recommendation: Option A (procedural)** with the following approach:

```python
@dataclass
class BorrowerTemplate:
    sector: str
    outcome: str  # good, bad, fraud
    revenue_range: tuple[float, float]
    margin_range: tuple[float, float]
    loan_range: tuple[float, float]
    narrative_template: str
    fraud_signals: list[str]  # Only for fraud borrowers

def generate_season_cohort(
    week: int,
    size: int,
    mix: str,
    seed: int,
) -> list[Borrower]:
    """Generate a cohort of borrowers for a given week.

    Uses deterministic seeding so seasons are reproducible:
    season_seed + week_number = cohort seed
    """
```

### 5.1 Season Mix Presets

| Preset | Good% | Bad% | Fraud% | Description |
|---|---|---|---|---|
| `gentle` | 70% | 20% | 10% | Warm-up: mostly good deals, few traps |
| `realistic` | 55% | 30% | 15% | Mirrors real pipeline quality |
| `adversarial` | 40% | 35% | 25% | Stress test: high default/fraud rate |
| `escalating` | varies | varies | varies | Starts gentle, ends adversarial |

The `escalating` preset is the most interesting for seasons — it tests whether lenders can detect the environment shifting:

```
Weeks 1-3:  80% good, 15% bad,  5% fraud
Weeks 4-6:  55% good, 30% bad, 15% fraud
Weeks 7-10: 35% good, 35% bad, 30% fraud
```

---

## 6. CLI Interface

```bash
# Run a 10-week season with default settings
python -m loanville --season

# Configure season parameters
python -m loanville --season --weeks 10 --cohort-size 5 --season-mix realistic

# Season with escalating difficulty
python -m loanville --season --season-mix escalating

# Season with custom tools enabled (requires Pi Agent Core)
python -m loanville --season --custom-tools

# Season with speed scoring enabled
python -m loanville --season --speed-scoring

# Full season experience
python -m loanville --season --weeks 10 --custom-tools --speed-scoring --season-mix escalating

# Reproduce a specific season (deterministic borrower generation)
python -m loanville --season --seed 42

# Existing modes still work unchanged
python -m loanville --mock           # Single match, mock
python -m loanville --rotate         # Model rotation tournament
```

---

## 7. Implementation Phases

### Phase 1: Season Loop + Carry-Forward (Core)

- `SeasonEngine` class wrapping the existing `SimulationEngine`
- `SeasonLenderState` dataclass for carry-forward state
- Rolling resolution (loans age per week)
- Between-week portfolio briefing in system prompt
- `--season` CLI flag
- No new tools, no speed scoring yet — just persistent capital

**This alone tests concentration limits, capital rationing, and path dependency.**

### Phase 2: Speed-to-Offer

- Track `tool_call_count` per evaluation (already partially tracked in traces)
- Speed bonus in deal adjudication
- Efficiency component in season scoring
- `--speed-scoring` CLI flag

### Phase 3: Custom Tool Creation

- `CustomTool` model and `LenderToolkit` class
- Tooling phase between weeks
- Pi Agent Core integration
- Custom tool persistence across weeks
- Tool creation cost accounting
- `--custom-tools` CLI flag

### Phase 4: Procedural Borrower Generation

- `BorrowerTemplate` and generation functions
- Season mix presets including `escalating`
- Deterministic seeding for reproducibility
- Expanded to ~50+ unique borrowers per season

---

## 8. Key Architectural Decisions

### 8.1 Season Engine wraps, not replaces, SimulationEngine

The `SeasonEngine` orchestrates the weekly loop and state carry-forward but delegates each week's origination/adjudication to the existing `SimulationEngine` (with modified lender configs reflecting current state). This preserves backward compatibility.

```python
class SeasonEngine:
    def __init__(self, config: SeasonConfig, lenders: list[LenderConfig], ...):
        self.config = config
        self.lender_states: dict[str, SeasonLenderState] = {}

    async def run_season(self):
        for week in range(1, self.config.weeks + 1):
            # 1. Resolve aging loans
            self._resolve_week(week)

            # 2. Brief lenders on portfolio state
            lenders = self._build_week_lenders(week)

            # 3. (Optional) Tooling phase
            if self.config.custom_tools:
                await self._tooling_phase(week)

            # 4. Generate this week's cohort
            cohort = generate_season_cohort(week, ...)

            # 5. Run standard match
            engine = SimulationEngine(cohort, lenders, ...)
            await engine.run()

            # 6. Update carry-forward state
            self._update_state(engine)

        # Final resolution
        self._final_resolution()
        self._print_season_report()
```

### 8.2 Tool calls are the speed unit, not wall-clock time

Wall-clock time varies by model provider latency, rate limits, and network conditions — none of which reflect the lender agent's actual decision quality. Tool calls are deterministic, reproducible, and directly measure analytical effort.

### 8.3 Custom tools are bash scripts, not Python

Keeping custom tools as bash scripts means they run in the existing `just-bash` sandbox with no additional security surface. The LLM writes `jq`/`awk`/`grep` pipelines, which is exactly what it already does with `run_bash` — custom tools just let it name and reuse them.

---

## 9. Success Metrics

| Metric | Target | How measured |
|---|---|---|
| Concentration violations are non-trivial | >30% of lenders breach a limit at least once | Season run logs |
| Capital exhaustion happens | At least 1 lender runs out of capital before season end | Season state tracking |
| Default adaptation is measurable | Rejection rate changes post-default vs pre-default | Decision trace analysis |
| Speed bonus changes outcomes | >10% of competitive deals decided by speed tiebreaker | Adjudication logs |
| Custom tools get reused | >50% of created tools used in subsequent weeks | Tool usage tracking |
| Scoring spread increases | Season mode score variance > single-match variance | Score distributions |

---

## 10. Open Questions

1. **Months per week:** How many simulated months should each week advance? 2 months gives a 20-month season (10 weeks); 3 months gives 30. Needs tuning.

2. **Tooling phase token budget:** Should tool creation have its own token/cost budget separate from evaluation? Or is it all one pool?

3. **Information asymmetry between weeks:** Should lenders see other lenders' portfolio performance (league table), or only their own? A league table creates pressure but also enables gaming.

4. **Borrower memory:** Should borrowers who were rejected in Week N reappear later (perhaps with updated financials)? This would add a "relationship lending" dimension.

5. **Pi Agent Core availability:** Need to confirm the Pi Agent Core SDK is available and compatible. If not, custom tools can be implemented with a simpler bespoke registry as a v1 fallback.

---

## 11. Future Considerations (Post-v1)

### 11.1 Competitive Refinancing

As a portfolio matures, the most interesting competitive dynamic in real lending is **poaching**: Lender B targets Lender A's best-performing borrowers and offers refinancing at a lower rate. The performing borrower is now a known quantity — the credit risk is proven — so the refinancing lender can rationally offer better terms.

This creates several design tensions:

**The static-financials problem.** Borrower financials are currently predetermined at season start. If Lender B offers a lower rate, that changes the borrower's debt service costs, which should improve their cash flow, which should change their financial profile. But we can't retroactively rewrite the bank statements.

**Recommended approach: treat financials as static, model refinancing as a pure rate/exposure event.**

- A performing loan (6+ months of on-time payments) becomes **refinancing-eligible**
- Competing lenders see a simplified "refi dossier": sector, original terms, months performing, current balance — but NOT the original bank statements
- The refinancing decision is: "Given this borrower has performed for N months at X% rate, will I offer Y% to take the loan?"
- If a refi is accepted, the original lender loses the loan (remaining principal returned to their available capital) and the new lender books it at the new rate
- The borrower's underlying outcome (good/bad) doesn't change — a "bad" borrower that hasn't defaulted yet can still be refinanced, and will still default at their predetermined month

**Why this is hard but worth exploring:**

| Challenge | Mitigation |
|---|---|
| Financials don't reflect the new rate | Treat financials as static; the refi decision is based on payment history, not updated projections |
| Predetermined defaults create unfair refi traps | Only allow refi on loans past 50% of their `months_before_default` — if it would default at month 6, it's not eligible until month 3 |
| Adds significant complexity to adjudication | Implement as a separate "refi round" after new origination each week |
| Scoring becomes circular | Score refi'd loans based on the terms and remaining life under the new lender |

**The strategic depth this adds:** A lender who rejected a borrower in Week 2 might refinance that same borrower from a competitor in Week 6 — at proven-performing terms. This rewards patience and punishes lenders who underprice good deals (their best borrowers get poached).

### 11.2 Phased Pipeline & Bandwidth Management

Currently, all borrowers in a cohort arrive simultaneously and lenders evaluate them all in parallel. In reality, loan applications arrive in a stream, and underwriters must **triage** — deciding where to spend their limited analytical bandwidth.

**The concept: skim vs. deep underwrite as a resource allocation problem.**

#### 11.2.1 Two-Tier Evaluation

Instead of one evaluation phase per borrower, split into two tiers:

```
TIER 1 — SKIM (cheap, fast)
  - Lender sees: company name, sector, loan amount, narrative, annual totals
  - NO quarterly breakdowns, NO bank statement access
  - Cost: 0 tool calls (pure prompt, single LLM call)
  - Output: PASS (proceed to deep underwrite) or SKIP (decline to evaluate further)

TIER 2 — DEEP UNDERWRITE (expensive, thorough)
  - Full dossier: quarterly income + bank statement tools
  - Cost: N tool calls (as today)
  - Output: APPROVE with term sheet, or REJECT
```

**Bandwidth constraint:** Each lender has a maximum number of **deep underwrite slots per week** (e.g., 3 out of 5 available borrowers). This forces triage:

```python
@dataclass
class BandwidthConfig:
    skim_slots: int = -1       # Unlimited skims (they're cheap)
    deep_uw_slots: int = 3     # Max full underwrite per week
    # A lender that uses all 3 slots on mediocre deals
    # misses the great deal that arrived last
```

#### 11.2.2 Phased Arrival

Borrowers don't all arrive at once. Within a week, they arrive in **phases**:

```
Week 5:
  Phase A (Monday):  Borrower 21, Borrower 22  → lenders skim
  Phase B (Wednesday): Borrower 23, Borrower 24 → lenders skim
  Phase C (Friday):  Borrower 25              → lenders skim

  After all phases: lenders allocate deep UW slots
  Then: full evaluation + adjudication
```

This tests **pipeline management**: do you burn your deep UW slots on the first borrowers you see, or do you wait to see the full pipeline? Waiting is safer but means you might lose a competitive deal to a lender who already deep-underwrote and made an offer.

**The speed-to-offer interaction:** A lender who skims Phase A and immediately deep-underwrites Borrower 21 gets a speed bonus on that deal. A lender who waits until Phase C to decide gets better information but slower offers. This is a genuine strategic trade-off.

#### 11.2.3 Why Phased Arrival Matters for the Benchmark

This moves the benchmark from "can you analyze one dossier well?" toward "can you manage an underwriting desk?" — which is fundamentally a resource allocation problem:

| Decision | Trade-off |
|---|---|
| Skim aggressively, deep UW early | Fast offers, but might waste slots on mediocre deals |
| Skim everything, deep UW selectively | Better selection, but slower offers and risk of losing competitive deals |
| Build custom tools to make skims more informative | Upfront investment (tool creation cost) for better triage later |

The phased model also creates natural synergy with custom tools: a lender who builds a good skim-stage analytical tool (e.g., `quick_sector_health_check`) can make more informed triage decisions without burning deep UW slots.

### 11.3 Borrower Financial Maturation

A related question to refinancing: should borrower financials change over time?

**The simplest model: static financials, dynamic payment history.**

- Bank statements and quarterly income remain frozen at origination
- But the season tracks payment history: months performing, total interest paid, any missed payments
- This payment history is what other lenders see when considering refinancing
- The underlying outcome (good/bad/fraud) is still predetermined

**A more ambitious model: evolving financials.**

If we procedurally generate borrowers (Phase 4 of implementation), we could also procedurally evolve their financials:

- Good borrowers: revenue grows 2-5% per season quarter, margins stable
- Bad borrowers: revenue flat or declining, margins compress before default
- Fraud borrowers: financials remain artificially clean until sudden collapse

This would let lenders do **portfolio monitoring** — reviewing their existing borrowers' updated financials each week and deciding whether to increase exposure (offer more capital) or pull back. But it adds significant complexity to the data generation pipeline and risks making the simulation feel arbitrary if the generation isn't convincing.

**Recommendation: defer to post-v1.** Static financials with dynamic payment history is sufficient for v1 season mode. The payment history alone gives lenders meaningful signals to react to. Evolving financials can be layered on in v2 once the procedural generation pipeline is proven.

### 11.4 How These Features Interact

These three ideas compound each other in interesting ways:

```
                    Phased Arrival
                         │
              "Which deals deserve my bandwidth?"
                         │
                    ┌────┴────┐
                    ▼         ▼
               Skim Only   Deep UW
                    │         │
                    │    ┌────┴────┐
                    │    ▼         ▼
                    │  Reject   Approve
                    │              │
                    │         Book Loan
                    │              │
                    │    ┌────────┴────────┐
                    │    ▼                 ▼
                    │  Performing       Defaulting
                    │    │                 │
                    │    ▼                 │
                    │  Refi-eligible       │
                    │    │                 │
                    │    ▼                 │
                    │  Competitor offers   │
                    │  lower rate          │
                    │    │                 │
                    │    ▼                 ▼
                    │  Loan moves to      Loss realized
                    │  new lender
                    │
                    ▼
               Missed deal
          (borrower went to competitor)
```

**Implementation order if pursued:**
1. Phased arrival + bandwidth (extends Season v1 naturally)
2. Refinancing (requires performing loan tracking, which Season v1 already has)
3. Financial maturation (requires procedural generation pipeline from Phase 4)

Each can be shipped independently. Phased arrival is the highest-impact addition because it transforms the benchmark from "credit analysis" to "underwriting desk management" — a much richer test of agent capability.

### 11.5 TODO: More Realistic Economics (Season Mode)

Season mode introduces timing and reinvestment, so the "economics layer" will matter more than in single-match mode. Consider (post-v1) upgrading the economics model with:

- **Cashflow realism:** explicit fee schedules (origination/servicing/prepay), collections/workout costs, and recovery timing
- **Time value:** discounting/NPV of delayed recoveries and long-duration cashflows (not just total interest)
- **Duration + reinvestment:** prepayment/refi changing loan duration, capital recycling, and competitive repricing
- **Balance-sheet constraints:** dynamic funding curve, liquidity/bandwidth constraints, and explicit unit economics per underwrite
- **Capital + provisioning:** RWA/cost of equity, CECL-style reserves, and penalties that depend on portfolio mix and macro regime
- **Correlated risk:** sector/macro shocks that drive correlated defaults, not i.i.d. borrower outcomes
