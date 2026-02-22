# Loanville Benchmark: Design & Methodology

## 1. Overview

Loanville is a synthetic commercial lending simulation that benchmarks LLM models as autonomous middle-market loan underwriters. Each model receives the same loan applications, analyses financial data, and makes approve/reject decisions with proposed terms. Models are scored on **RAROC (Risk-Adjusted Return on Capital)** and ranked via a **three-Elo tournament** system with per-applicant pairwise comparisons.

### Three-Elo Rating System

The benchmark tracks three separate Elo ratings per model, each measuring a different dimension of underwriting quality:

| Rating | What It Measures | Win Condition (per borrower) |
|---|---|---|
| **Profit Elo** | Economic utility — aligned with RAROC | Higher risk-adjusted utility (realized profit or benchmark return) |
| **Credit Elo** | Decision correctness vs ground truth | Correct approve/reject given borrower's true outcome |
| **DealShare Elo** | Market participation / bid aggressiveness | Won the deal (regardless of profitability) |

**Profit Elo** is the recommended headline ranking — it aligns Elo with RAROC economics while preserving the fast convergence of per-applicant pairwise signals.

---

## 2. How Businesses Are Created

Borrowers are **hand-crafted synthetic businesses** defined in `loanville/data.py`. Each has:

- **Identity**: Company name, sector, years in business, employee count
- **Financials**: 12 months of `(deposit_total, withdrawal_total)` figures, from which:
  - **12 monthly bank statements** are procedurally generated with named customers/vendors, individual transactions, and running balances
  - **4 quarterly income statements** are aggregated (revenue, expenses, gross profit, margins)
  - **Annual totals** are pre-computed
- **Narrative**: A 3-5 sentence pitch describing the business and loan purpose
- **Ground truth**: `true_outcome` ("good", "bad", or "fraud") and `months_before_default` — hidden from models

### 24 Borrowers Total

| Category | Count | IDs | Challenge |
|---|---|---|---|
| Good | 10 | BRW-001 to BRW-005, BRW-013 to BRW-017 | Healthy businesses that will fully repay |
| Bad (standard) | 4 | BRW-006 to BRW-009 | Customer concentration, margin compression, grant dependency, declining revenue |
| Bad (over-leverage) | 6 | BRW-019 to BRW-024 | Legitimate revenue but DSCR < 1.0 — loan creates unsustainable debt |
| Fraud | 4 | BRW-010 to BRW-012, BRW-018 | Round numbers, circular transfers, fabricated statements, structuring |

### Sample Good Business

> **SkyFreight Solutions** (Aero-Logistics, 8 years, 45 employees)
> Annual revenue $2.44M, net income $659K (27% margin). Steady monthly growth from $185K to $222K deposits. Requesting $500K for fleet expansion.
> True outcome: **good** (full repayment)

### Sample Bad Business (Over-Leverage)

> **Pacific Rim Importers** (Import/Distribution, 13 years, 22 employees)
> Revenue $3.49M looks impressive but net margin is 5.1% ($177K net income). Requesting $500K — debt service ($276K/yr) would exceed net income. DSCR ~0.64.
> True outcome: **bad** (defaults at month 11)

### Fraud Types

| Pattern | Borrower | Detection Method |
|---|---|---|
| **Round-number deposits** | BRW-010 (CloudNet Logistics) | All deposits are $50K, $75K, $100K, etc. |
| **Circular transfers** | BRW-011 (BioGenesis Research) | 70% of deposits from related entities |
| **Fabricated consistency** | BRW-012 (QubitTech Solutions) | Monthly figures vary by <$400 (statistically implausible) |
| **Structuring (smurfing)** | BRW-018 (Orion Fleet Services) | All deposits broken into sub-$10K transactions |

---

## 3. Scenario Suites

Borrowers are grouped into **mix presets** that test different capabilities:

| Mix | Good | Bad | Fraud | Total | Primary Test |
|---|---|---|---|---|---|
| `analyst` | 10 | 6 (over-leverage) | 0 | 16 | DSCR analysis, margin compression |
| `fraud` | 5 | 0 | 4 | 9 | Bank statement fraud detection |
| `easy` | 9 | 2 | 1 | 12 | Realistic commercial pipeline |
| `balanced` | 7 | 4 | 4 | 15 | Stressed market with all risk types |
| `hard` | 5 | 4 | 4 | 13 | Adversarial stress test |
| `concentration` | 10 | 4 | 2 | 16 | Sector limit management |
| `stress` | 5 | 8 | 4 | 17 | Kitchen-sink: all risk types combined |
| `all` | 10 | 4 | 4 | 18 | Complete pool (no over-leverage bad) |

### Borrower Sampling

To reduce overfitting to fixed borrower pools, the `--sample-borrowers N` flag randomly samples N borrowers per match using **stratified sampling** (maintaining the good/bad/fraud ratio). This introduces match-to-match variation so Elo reflects general underwriting skill rather than memorization of 16 specific businesses.

---

## 4. How Underwriting Works

Each model is instantiated as a lender agent via OpenRouter's API (`loanville/llm.py`). The flow per application:

1. **System prompt**: Sets the lender persona, target yield (11%), capital ($3.5M), sector limits (25% each), max single loan ($700K), and analysis instructions
2. **User prompt**: Presents the full financial dossier — quarterly income statements, annual totals, company narrative, and loan request
3. **Tool use**: Models can call `run_bash` to execute sandboxed `jq` queries against the borrower's raw 12-month bank statements at `/data/bank_statements.json`
4. **Decision**: Model returns a JSON object:

```json
{
  "decision": "APPROVE",
  "reasoning": "Strong cash flow, diversified client base, reasonable leverage...",
  "term_sheet": {
    "loan_amount": 500000,
    "interest_rate": 9.5,
    "term_months": 24
  }
}
```

Up to 3 tool-call rounds are allowed; models are forced to produce a final answer on the last round. All evaluations run concurrently via `asyncio`.

### Deal Adjudication (Competitive Market)

When multiple models approve the same borrower, the borrower **picks the lowest interest rate** (breaking ties by highest loan amount). Only one lender wins each deal — this simulates competitive market dynamics. Capital limits are enforced: if the preferred lender lacks capacity, the deal falls to the next-best offer.

### Scope: Application-Level Underwriting

Each match starts with a **clean book** ($0 deployed, $3.5M available). Capital and exposures do **not** carry forward across matches. This benchmarks application-level underwriting judgment, not portfolio lifecycle management. Models are not tested on capital rationing over time, path dependency, or concentration management across sequential deals.

---

## 5. RAROC Scoring

The final score (`loanville/scoring.py`) is **RAROC vs a risk-free benchmark**:

```
Score = (Adjusted_PnL / Available_Capital) * 100 - Benchmark%
```

Where:
- **Benchmark** = 5% risk-free rate x (24 months / 12) = **10.0%**
- **Adjusted P&L** = Net P&L - Funding Cost - Fraud Penalty - Concentration Penalty - Yield Drag - Risk Penalty - Volume Penalty - Hard Constraint Penalty

### Component Breakdown

| Component | Formula / Mechanism | Constants |
|---|---|---|
| **Net P&L** | Interest earned minus principal lost (amortizing schedule) | — |
| **Funding cost** | Outstanding balance x 4% annual, per-month on amortizing principal | `FUNDING_RATE = 0.04` |
| **Risk penalty** | lambda x sigma x sqrt(n) — portfolio volatility scaling | `RISK_LAMBDA = 0.5` |
| **Volume floor** | Deploy >= 20% of capital; penalty = lambda x shortfall^2 x capital | `MIN_DEPLOYMENT_RATIO = 0.20` |
| **Default rate cap** | Default rate > 30% triggers quadratic penalty | `MAX_DEFAULT_RATE = 0.30` |
| **Min ROE** | ROE below -10% triggers quadratic penalty | `MIN_ROE_THRESHOLD = -0.10` |
| **Fraud penalty** | 25% of principal on fraud loans (regulatory/reputational cost) | — |
| **Concentration penalty** | 5% of excess exposure above sector limits | — |
| **Yield drag** | Penalty for loans priced below target yield, proportional to shortfall | — |

### Risk Penalty: Why `sigma x sqrt(n)`

The risk penalty uses `sigma x sqrt(n)` rather than `sigma x n`. This is analogous to portfolio standard deviation scaling: the standard deviation of the mean of n independent observations scales with `sqrt(n)`, not `n`. Using `sigma x n` would double-count scale and excessively penalize diversified portfolios. The `sqrt(n)` formulation is closer to a portfolio VaR-style term.

### Scoring Dynamics

- A model that **rejects everything** gets a volume penalty plus the opportunity cost of missing the risk-free benchmark — scoring approximately **-10%**.
- A model that **approves everything** gets crushed by defaults, risk penalties, and hard-constraint penalties.
- The optimal strategy requires selective approval of creditworthy borrowers at appropriate rates.

### Baselines

Two baselines validate that the scoring system is well-calibrated:

| Baseline | Method | Purpose |
|---|---|---|
| **Oracle** | Perfect foresight: approves all good, rejects all bad/fraud, prices at target yield | Upper bound — shows the theoretical max under constraints |
| **Heuristic** | Simple rules: reject if net margin < 10%, DSCR < 1.25, or leverage > 1.5x net income | Shows the task is solvable by straightforward financial analysis, not LLM-specific reasoning |

### Per-Loan Payoff Computation

Each loan outcome is computed via `compute_loan_payoff()`:

- **Good loans**: Full amortization schedule. Profit = total interest earned - funding cost.
- **Bad loans**: Partial payments for `months_before_default` months, then remaining principal is lost. Profit = interest earned - principal lost - funding cost.
- **Fraud loans**: Immediate total loss of principal, plus 25% fraud penalty. Profit = -(principal + funding cost + fraud penalty).

---

## 6. Three-Elo Rating System

### Why Three Ratings?

A single Elo rating conflates multiple dimensions of lending performance. The original system ("won deal = win") produced rankings anti-correlated with RAROC — aggressive bidders dominated Elo while losing money. Rather than patch one Elo formula, we track three orthogonal ratings:

### 6.1 Profit Elo (Recommended Ranking)

**Measures: Economic utility per borrower, aligned with RAROC.**

For each borrower, each model's utility is:
- **Won deal**: Realized net profit (can be negative for bad loans)
- **Declined**: Risk-free benchmark return on the notional capital (`principal x risk_free_rate x term/12`)
- **Lost deal (outbid)**: Same as declined (capital wasn't deployed)

Pairwise comparison:
- A wins if `Utility(A) > Utility(B) + epsilon` (epsilon = $500)
- Tie if `|Utility(A) - Utility(B)| <= epsilon`
- B wins otherwise

**Why this works**: Correctly declining a bad borrower earns the benchmark return (~$50K on a $500K loan over 2 years), which beats the large negative profit from funding a defaulting borrower. This eliminates the participation bias of the original system.

### 6.2 Credit Elo (Decision Quality)

**Measures: Correctness of approve/reject decisions vs ground truth.**

For each borrower:
- **Correct decision**: Approve good borrowers, reject bad/fraud
- **Incorrect decision**: Reject good borrowers, approve bad/fraud

Pairwise: correct beats incorrect; same correctness = tie.

**Why this is useful**: Pure signal on underwriting judgment, independent of pricing, competitive dynamics, or market share. Directly measures whether the model can distinguish creditworthy from uncreditworthy borrowers.

### 6.3 DealShare Elo (Market Participation)

**Measures: Who wins deals, regardless of profitability.**

Pairwise: winning a deal = 1.0 vs any non-winner; all non-winners tie.

**Why this is useful**: Characterizes bidding aggressiveness and pricing competitiveness. High DealShare + low Profit = aggressive volume player. Low DealShare + high Profit = conservative optimizer.

### Per-Applicant Pairwise Signals

All three Elo systems use per-applicant pairwise signals rather than aggregate match comparisons. For N borrowers in a 3-way match:
- Each pair of models generates N pairwise signals (one per borrower)
- Total: `N x C(3,2) = N x 3` signals per match
- K-factor scaling: `pair_k = K / ((n-1) x N)` keeps total Elo movement per match bounded at ~K

This provides dramatically faster convergence than single aggregate comparisons.

### Standard Elo Formula

```
Expected score: E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
Rating update:  R_A' = R_A + pair_k * (S_A - E(A))
```

### Matchup Generation

Triplets are drawn with balanced participation — models with fewer scheduled matches get priority.

---

## 7. Reporting

### Confusion Matrix

For each model, the system tracks a confusion matrix by borrower ground truth:

|  | Good Borrowers | Bad/Fraud Borrowers |
|---|---|---|
| **Approved** | True Positive (correct) | False Positive (error) |
| **Rejected** | False Negative (missed opportunity) | True Negative (correct) |

This is reported in the standings as `Good✓` (good borrowers correctly approved) and `Bad✓` (bad/fraud borrowers correctly rejected).

### Penalty Decomposition

Per-model breakdown of which RAROC penalties are driving negative scores:

- Funding cost, fraud penalty, concentration penalty, yield drag
- Risk penalty (volatility), volume penalty (under-deployment)
- Hard constraint penalty (default rate or ROE violations)

This identifies whether a model's poor score comes from bad credit decisions (defaults) vs bad pricing (yield drag) vs insufficient deployment (volume penalty).

### Bootstrap Confidence Intervals

RAROC scores include bootstrap 95% confidence intervals computed by resampling loan outcomes with replacement (1000 iterations). This quantifies scoring uncertainty — particularly important with small portfolio sizes where a single default can swing the score dramatically.

### Pricing Analysis

The system tracks interest rates offered by each model per borrower, enabling analysis of:
- Average rate on good vs bad borrowers (does the model price risk correctly?)
- Rate distributions and outliers
- Competitive pricing dynamics (who is consistently cheapest?)

---

## 8. Methodology Notes

### Controlled Conditions
All lenders in each match get identical config (persona, capital, limits). The only variable is the model.

### Capital Resets
Each match starts with a clean book. This benchmarks **application-level underwriting**, not portfolio lifecycle management. Models are not tested on capital rationing over time or path dependency.

### Deterministic Outcomes
Borrower ground truth is fixed — the same business always has the same outcome. Variation comes from:
- Which 3 of N models face off (stochastic matchups with balanced participation)
- Optional borrower sampling per match (`--sample-borrowers`)
- Model stochasticity (temperature 0.3)

### Temperature
All models use temperature 0.3 — low enough for consistent analysis, high enough for some pricing variation.

---

## 9. Architecture

```
elo_benchmark.py          # 3-Elo tournament runner, matchup generation
benchmark_models.py       # Model lists, single-run benchmark, Pareto analysis
loanville/
  data.py                 # 24 hand-crafted borrowers, 8 mix presets, statement generation
  models.py               # Dataclasses: Borrower, LenderConfig, LenderDecision, LenderScore, etc.
  engine.py               # SimulationEngine: origination, adjudication, booking, resolution
  llm.py                  # OpenRouter client, prompt construction, tool-use loop, cost tracking
  scoring.py              # RAROC scoring, baselines, confusion matrix, bootstrap CI, penalties
  mock_llm.py             # Deterministic mock for testing without API calls
```

---

## 10. Design Decisions & Trade-offs

### Why Three Elo Ratings Instead of One

A single "utility Elo" would be simpler but loses information. The divergence between ratings is itself diagnostic:

| Pattern | Interpretation |
|---|---|
| High Profit + High Credit + Low DealShare | Conservative optimizer: makes correct decisions but loses deals on pricing |
| High DealShare + Low Profit + Low Credit | Aggressive volume player: wins deals but picks bad borrowers |
| High Credit + High Profit + High DealShare | Ideal: correct decisions, good pricing, wins deals |
| Low everything | Weak model: bad decisions, bad pricing, loses deals |

### Why Benchmark Return for Declines (Profit Elo)

In Profit Elo, declining a borrower earns the risk-free benchmark return on the notional capital, not zero. This reflects the opportunity cost framing: capital not deployed in a bad deal is available for risk-free investment. Setting decline utility to zero would under-reward correct rejections and re-introduce the participation bias.

### Why `sqrt(n)` for Risk Penalty

The original `sigma x n` formulation was flagged as unusual — standard portfolio theory says the standard deviation of the portfolio mean scales with `sqrt(n)`, not linearly. The `sqrt(n)` formulation means:
- A well-diversified portfolio of 10 loans is penalized less than the old formula
- A concentrated portfolio of 2-3 loans is penalized more (relatively)
- This better captures the actual risk reduction from diversification

### Why Fixed Borrower Pools (and How to Mitigate)

The system uses 24 hand-crafted borrowers rather than procedural generation because:
- Each borrower encodes a specific analytical challenge (DSCR, concentration, fraud pattern)
- Procedural generation risks unrealistic financial profiles
- Quality control is easier with curated data

**Mitigation for overfitting**: The `--sample-borrowers` flag randomly samples a subset per match, and multiple mix presets test different skill dimensions.

### Why Separate Scenario Suites

Fraud detection, DSCR analysis, and concentration management are distinct skills. Lumping them into one mix makes it impossible to diagnose model weaknesses. The scenario suites (`analyst`, `fraud`, `concentration`, `stress`) isolate each capability:
- `analyst`: Can the model compute debt service ratios?
- `fraud`: Can the model detect bank statement anomalies?
- `concentration`: Does the model respect portfolio limits?
- `stress`: How does the model handle all risk types simultaneously?
