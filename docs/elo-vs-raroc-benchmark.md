# Loanville Elo vs RAROC Benchmark: Full Write-Up

## 1. Overview

Loanville is a synthetic commercial lending simulation that benchmarks LLM models as autonomous middle-market loan underwriters. Each model receives the same loan applications, analyses financial data, and makes approve/reject decisions with proposed terms. Models are then scored on a **RAROC (Risk-Adjusted Return on Capital)** basis and ranked via an **Elo tournament** system with per-applicant pairwise comparisons.

**Tournament Run Metadata:**

| Field | Value |
|---|---|
| Date | 2026-02-22 |
| Models | 6 |
| Matches | 20 (3-way head-to-head) |
| Borrower Mix | `analyst` (16 businesses: 10 good, 6 bad, 0 fraud) |
| Total API Cost | $0.79 |
| K-factor | 32 |
| Initial Elo | 1500 |
| Capital per Lender | $3,500,000 |
| Sim Horizon | 24 months |

---

## 2. The Models

Six models competed, spanning budget to mid-range on OpenRouter:

| Model | Approx. Output $/M tokens |
|---|---|
| GPT-4.1 Nano | $0.40 |
| GPT-4o Mini | $0.60 |
| Gemini 2.5 Flash | $2.50 |
| Qwen3-30B | $0.28 |
| Llama 3 8B | $0.05 |
| Mistral Nemo | $0.04 |

---

## 3. How Businesses Are Created

Borrowers are **hand-crafted synthetic businesses** defined in `loanville/data.py`. Each has:

- **Identity**: Company name, sector, years in business, employee count
- **Financials**: 12 months of `(deposit_total, withdrawal_total)` figures, from which:
  - **12 monthly bank statements** are procedurally generated with named customers/vendors, individual transactions, and running balances
  - **4 quarterly income statements** are aggregated (revenue, expenses, gross profit, margins)
  - **Annual totals** are pre-computed
- **Narrative**: A 3-5 sentence pitch describing the business and loan purpose
- **Ground truth**: `true_outcome` ("good", "bad", or "fraud") and `months_before_default` — hidden from models

### Sample Good Business

> **SkyFreight Solutions** (Aero-Logistics, 8 years, 45 employees)
> Annual revenue $2.44M, net income $659K (27% margin). Steady monthly growth from $185K to $222K deposits. Requesting $500K for fleet expansion. Customers: Meridian Airways, TransGlobal Shipping, etc.
> True outcome: **good** (full repayment)

### Sample Bad Business (Over-Leverage)

> **Pacific Rim Importers** (Import/Distribution, 13 years, 22 employees)
> Revenue $3.49M looks impressive but net margin is 5.1% ($177K net income). Requesting $500K — debt service ($276K/yr) would exceed net income. DSCR approximately 0.64.
> True outcome: **bad** (defaults at month 11)

### Fraud Types (not in this mix, but in the system)

Four fraud patterns are supported:

- **Round-number deposits**: All deposits are suspiciously round ($50K, $75K, $100K)
- **Circular transfers**: 70% of deposits come from related entities (e.g., "BioGenesis Holdings LLC", "BGH Capital Partners")
- **Fabricated statements**: Monthly figures are unnaturally consistent with near-zero variance
- **Structured deposits (smurfing)**: Large sums broken into many sub-$10K transactions to avoid Currency Transaction Report thresholds

### The "Analyst" Mix

The `analyst` preset used in this tournament contains **no fraud** — all 6 "bad" businesses are legitimate companies where the **requested loan would create unsustainable debt service** relative to free cash flow. This specifically tests whether models can compute DSCR, spot margin compression, and recognize thin-margin businesses that can't absorb new debt.

The 16 borrowers:

| Category | Count | IDs | Challenge |
|---|---|---|---|
| Good | 10 | BRW-001 through BRW-005, BRW-013 through BRW-017 | Healthy businesses that will fully repay |
| Bad (over-leverage) | 6 | BRW-019 through BRW-024 | Legitimate revenue but DSCR < 1.0 — loan would be unsustainable |
| Fraud | 0 | — | Not present in this mix |

---

## 4. How Underwriting Works

Each model is instantiated as a lender agent via OpenRouter's API (`loanville/llm.py`). The flow per application:

1. **System prompt**: Sets the lender persona, target yield (11%), capital ($3.5M), sector limits (25% each), max single loan ($700K), and analysis instructions
2. **User prompt**: Presents the full financial dossier — quarterly income statements, annual totals, company narrative, and loan request
3. **Tool use**: Models can call `run_bash` to execute sandboxed `jq` queries against the borrower's raw 12-month bank statements at `/data/bank_statements.json` (powered by `just-bash`)
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

Up to 3 tool-call rounds are allowed; models are forced to produce a final answer on the last round. All evaluations for all lenders run concurrently via `asyncio`.

### Deal Adjudication (Competitive Market)

When multiple models approve the same borrower, the borrower **picks the lowest interest rate** (breaking ties by highest loan amount). Only one lender wins each deal — this is not parallel-universe underwriting; it simulates competitive market dynamics. Capital limits are enforced: if the preferred lender lacks capacity, the deal falls to the next-best offer.

---

## 5. RAROC Scoring

The final score (`loanville/scoring.py`) is **RAROC vs a risk-free benchmark**, computed as:

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
| **Funding cost** | Outstanding balance x 4% annual, computed per-month on amortizing principal | `FUNDING_RATE = 0.04` |
| **Risk penalty** | lambda x sigma x n (penalizes per-loan outcome variance, not portfolio size) | `RISK_LAMBDA = 0.5` |
| **Volume floor** | Must deploy >= 20% of capital; penalty = lambda x shortfall^2 x capital | `MIN_DEPLOYMENT_RATIO = 0.20` |
| **Default rate cap** | Default rate > 30% triggers quadratic penalty scaling with severity | `MAX_DEFAULT_RATE = 0.30` |
| **Min ROE** | ROE below -10% triggers quadratic penalty | `MIN_ROE_THRESHOLD = -0.10` |
| **Fraud penalty** | 25% of principal on fraud loans (regulatory/reputational cost) | — |
| **Concentration penalty** | 5% of excess exposure above sector limits | — |
| **Yield drag** | Penalty for loans priced below target yield, proportional to shortfall | — |

### Scoring Dynamics

- A model that **rejects everything** gets a volume penalty plus the opportunity cost of missing the risk-free benchmark — scoring approximately **-10%**.
- A model that **approves everything** gets crushed by defaults, risk penalties, and hard-constraint penalties.
- The optimal strategy requires selective approval of creditworthy borrowers at appropriate rates — balancing deployment volume against default risk.

### Per-Loan Payoff Computation

Each loan outcome is computed via `compute_loan_payoff()`:

- **Good loans**: Full amortization schedule, all payments made. Profit = total interest earned - funding cost.
- **Bad loans**: Partial payments for `months_before_default` months, then remaining principal is lost. Profit = interest earned - principal lost - funding cost.
- **Fraud loans**: Immediate total loss of principal, plus 25% fraud penalty. Profit = -(principal + funding cost + fraud penalty).

---

## 6. Elo System

### Per-Applicant Pairwise Elo

Rather than comparing models only on aggregate match score, the system (`elo_benchmark.py`) generates **per-borrower Elo signals**. For each of the 16 borrowers in each 3-way match, every pair of models is compared:

| State (Model A vs B) | Outcome |
|---|---|
| Both declined | Tie (0.5 / 0.5) |
| A won deal, B declined | A wins (1.0 / 0.0) |
| A won deal, B lost (outbid) | A wins (1.0 / 0.0) |
| Both lost to third model | Tie (0.5 / 0.5) |
| A lost, B declined | Tie (0.5 / 0.5) |

**K-factor scaling**: `pair_k = K / ((n-1) * n_borrowers)` = 32 / (2 x 16) = 1.0 per borrower-pair, keeping total Elo movement per match bounded at approximately K.

This generates **16 x C(3,2) = 48 pairwise signals per match** instead of a single aggregate comparison, providing dramatically faster Elo convergence.

### Standard Elo Formula

```
Expected score: E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
Rating update:  R_A' = R_A + pair_k * (S_A - E(A))
```

Where S_A is the actual outcome (1.0 for win, 0.5 for tie, 0.0 for loss).

### Matchup Generation

Triplets are drawn with balanced participation — models with fewer scheduled matches get priority. Each model appeared in exactly **10 of the 20 matches**.

---

## 7. Results

### Final Elo Standings

| Rank | Model | Elo | Matches | Wins | Win% | Avg Score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **GPT-4.1 Nano** | **1565** | 10 | 2 | 20% | -41.80% |
| 2 | **Llama 3 8B** | **1528** | 10 | 0 | 0% | -54.85% |
| 3 | **Gemini 2.5 Flash** | **1485** | 10 | 3 | 30% | -32.60% |
| 4 | **GPT-4o Mini** | **1479** | 10 | 2 | 20% | -41.82% |
| 5 | **Qwen3-30B** | **1477** | 10 | 5 | 50% | -23.95% |
| 6 | **Mistral Nemo** | **1467** | 10 | 8 | 80% | -17.81% |

### The Elo vs RAROC Tension

| Model | Elo Rank | RAROC Rank | Avg Approvals | Avg Deals Won | Avg Defaults | Avg Deployed | Avg P&L |
|---|---|---|---|---|---|---|---|
| GPT-4.1 Nano | 1st | 4th | 12.1 | 9.0 | 2.2 | $3,155,000 | -$269,083 |
| Llama 3 8B | 2nd | 6th (worst) | 10.1 | 6.9 | 3.1 | $2,750,000 | -$535,874 |
| Gemini 2.5 Flash | 3rd | 3rd | 8.4 | 3.5 | 1.1 | $1,327,500 | -$155,521 |
| GPT-4o Mini | 4th | 5th | 11.2 | 3.5 | 1.6 | $1,320,000 | -$306,746 |
| Qwen3-30B | 5th | 2nd | 9.7 | 3.0 | 0.8 | $1,105,000 | -$101,890 |
| Mistral Nemo | 6th | 1st (best) | 2.2 | 1.9 | 0.2 | $602,500 | +$1,957 |

### Key Observations

**Elo rating and RAROC score diverge significantly**, and this is the central finding:

- **Mistral Nemo** has the best RAROC score (-17.81% avg) and highest win rate (80%) but the **lowest Elo** (1467). Why? It approves only ~2.2 out of 16 applicants per match, deploying just $602K on average. It is cautious and profitable when it lends, but the per-applicant Elo system rewards models that **win deals** — and Mistral Nemo declines most borrowers, generating neutral "tie" (0.5/0.5) signals on the vast majority of applications.

- **GPT-4.1 Nano** has the highest Elo (1565) despite a poor RAROC (-41.80%). It approves 12.1/16 applicants, deploys $3.15M on average, and wins 9.0 deals per match. In per-applicant Elo, it accumulates "win" signals on many borrowers because it simply bids on and wins more deals — even though some of those deals default.

- **Llama 3 8B** ranks #2 by Elo despite the worst RAROC (-54.85%) and zero outright match wins. Its high Elo comes from aggressive deployment (10.1 approvals, $2.75M deployed), which generates many "won deal" Elo signals even though the financial outcomes are poor.

- **Qwen3-30B** occupies the middle ground — moderate approval rate (9.7/16), low defaults (0.8 avg), and the second-best RAROC. It also has the highest match win rate at 50%, suggesting it balances selectivity with profitability. Its Elo is held back because it loses deals competitively to more aggressive bidders.

### Cost Efficiency

| Model | API Cost (20 matches) | Cost per Match |
|---|---|---|
| Mistral Nemo | $0.018 | $0.002 |
| Llama 3 8B | $0.038 | $0.004 |
| GPT-4.1 Nano | $0.099 | $0.010 |
| GPT-4o Mini | $0.145 | $0.014 |
| Qwen3-30B | $0.150 | $0.015 |
| Gemini 2.5 Flash | $0.340 | $0.034 |

### Why All Scores Are Negative

All RAROC scores are negative because the `analyst` mix is intentionally challenging — 37.5% of borrowers (6/16) are designed to default through over-leverage, and even perfect foresight only scores around +5-8% above the risk-free benchmark. The over-leverage businesses present realistic-looking financials where the default risk is hidden in DSCR analysis that requires careful computation.

---

## 8. Methodology Notes

- **Controlled conditions**: All lenders in each match get identical config (persona, capital, limits). The only variable is the model.
- **Deterministic borrowers**: The same 16 businesses appear in every match; outcomes are fixed by ground truth.
- **Stochastic matchups**: Which 3 of 6 models face off varies, with balanced participation.
- **No existing portfolio**: Tournament lenders start with a clean book ($0 deployed, $3.5M available) — no legacy position bias.
- **Tool access**: Models can query raw bank statement JSON via sandboxed bash/jq, but tool usage varies by model.
- **Temperature**: 0.3 for all models.
- **Total decisions**: 20 matches x 3 models x 16 borrowers = **960 individual underwriting decisions** across the tournament.

---

## 9. Architecture

```
elo_benchmark.py          # Tournament runner, Elo math, matchup generation
benchmark_models.py       # Model lists, single-run benchmark, Pareto analysis
loanville/
  data.py                 # 24 hand-crafted borrowers, mix presets, statement generation
  models.py               # Dataclasses: Borrower, LenderConfig, LenderDecision, LenderScore, etc.
  engine.py               # SimulationEngine: origination, adjudication, booking, resolution
  llm.py                  # OpenRouter client, prompt construction, tool-use loop, cost tracking
  scoring.py              # RAROC scoring, per-loan payoff, penalties, final report
  mock_llm.py             # Deterministic mock for testing without API calls
```

---

## 10. Conclusions

1. **Per-applicant Elo rewards market participation over profitability.** Models that bid aggressively accumulate Elo even when deals lose money. This mirrors real lending markets where volume-focused lenders can dominate market share while sacrificing risk-adjusted returns.

2. **RAROC correctly captures economic value.** Mistral Nemo — the only model to achieve positive average P&L (+$1,957) — ranks first on RAROC despite last on Elo. Its extreme selectivity (2.2/16 approvals) is rewarded by the financial scoring but penalized by the competitive Elo framework.

3. **The analyst mix exposes DSCR blindness.** Most models struggle to distinguish healthy businesses from over-leveraged ones when the revenue looks strong but debt-service capacity is inadequate. This is a realistic failure mode — revenue alone does not determine creditworthiness.

4. **Cost does not predict performance.** Gemini 2.5 Flash ($0.34 total) does not outperform GPT-4.1 Nano ($0.10) or Qwen3-30B ($0.15) on either metric, suggesting diminishing returns in the budget-to-midrange tier for structured financial analysis tasks.

5. **The tension between Elo and RAROC is itself informative.** A model's position in both rankings characterizes its "lending personality" — aggressive market-taker (high Elo, low RAROC) vs conservative portfolio optimizer (low Elo, high RAROC). Real lending institutions face this same trade-off between market share and risk-adjusted returns.
