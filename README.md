# Loanville2 — LLM Lending Benchmark

A synthetic commercial lending simulation that benchmarks LLM models as autonomous middle-market loan underwriters. Models compete head-to-head in randomly matched triplets, each receiving identical borrower applications and lender constraints. The only variable is the model's analytical and pricing ability.

## How It Works

1. **Pipeline Distribution** — 24 hand-crafted business dossiers (with 12-month bank statements, quarterly income statements, and narratives) are broadcast to all lender agents.
2. **Underwriting** — Each LLM evaluates every application using a sandboxed bash tool to query raw bank statement data, then outputs a JSON decision (APPROVE/REJECT) with an optional term sheet (amount, rate, term).
3. **Deal Adjudication** — Borrowers pick the lowest interest rate offer. Only one lender wins each deal (winner-takes-all competitive market). Capital limits enforced.
4. **Resolution** — The engine fast-forwards loan lifecycles. Good businesses repay in full, bad businesses default partway through, and frauds default immediately.
5. **Scoring** — Lenders are scored on RAROC (risk-adjusted return on capital) with penalties for fraud, concentration, under-deployment, and volatility. Models are ranked via a three-Elo tournament system.

### The Borrower Pool (24 businesses)

| Type | Count | Behavior |
|------|-------|----------|
| Good | 10 | Full repayment with interest — healthy cash flow, diversified clients |
| Bad (standard) | 4 | Default after partial payments — customer concentration, margin compression, grant dependency, revenue decline |
| Bad (over-leverage) | 6 | Legitimate revenue but DSCR < 1.0 — loan creates unsustainable debt service |
| Fraud | 4 | Immediate default — round-number deposits, circular transfers, fabricated consistency, structuring |

### Scenario Mixes

| Mix | Good | Bad | Fraud | Primary Test |
|-----|------|-----|-------|-------------|
| `analyst` | 10 | 6 | 0 | DSCR analysis, margin compression |
| `fraud` | 5 | 0 | 4 | Bank statement fraud detection |
| `easy` | 9 | 2 | 1 | Realistic commercial pipeline |
| `balanced` | 7 | 4 | 4 | Stressed market with all risk types |
| `hard` | 5 | 4 | 4 | Adversarial stress test |
| `stress` | 5 | 8 | 4 | Kitchen-sink: all risk types combined |

### Three Elo Ratings

| Rating | What It Measures | Win Condition |
|--------|-----------------|---------------|
| **Profit Elo** (primary) | Economic utility aligned with RAROC | Higher risk-adjusted profit per borrower |
| **Credit Elo** | Decision correctness vs ground truth | Correct approve/reject given true outcome |
| **DealShare Elo** | Market participation and bid aggressiveness | Won the deal (regardless of profitability) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

## Run

```bash
# Elo tournament (recommended)
python elo_benchmark.py --mix analyst --matches 50

# Deterministic mock replay (fixed borrower order)
python -m loanville --mock --mix fraud --data-mode lite --seed 42

# Resume from previous results
python elo_benchmark.py --mix analyst --matches 80 --resume elo_results.json

# View standings from saved results
python elo_benchmark.py --standings elo_results.json

# Single-run benchmark with Pareto analysis
python benchmark_models.py --mix analyst
```

### Run SIM with Open LOS backend

```bash
# 1) Start Open LOS API (from ../LOS)
cd ../LOS
npm install
DB_PATH=file:./sim-integration.db PORT=3100 npm run start --workspace=packages/api

# 2) Run SIM against LOS API (from ../SIM)
cd ../SIM
python -m loanville --mix fraud --seed 42 --sample-size 6 \
  --underwriting-backend los --los-base-url http://localhost:3100

# 3) Integration smoke test
python scripts/test_los_integration.py --los-base-url http://localhost:3100
```

## Configuration

Set `OPENROUTER_API_KEY` in your `.env` file. Get one at [openrouter.ai](https://openrouter.ai/).

Models are defined in `benchmark_models.py`. The tournament uses all models from `SMALL_MODELS` by default; pass `--full` for the complete `BENCHMARK_MODELS` list, or `--models model1 model2` for specific models.

## Project Structure

```
elo_benchmark.py          # 3-Elo tournament runner, matchup generation, standings
benchmark_models.py       # Model lists, single-run benchmark, Pareto analysis
loanville/
├── __main__.py           # CLI entry point (legacy single-run mode)
├── models.py             # Dataclasses: Borrower, LenderConfig, LenderDecision, LenderScore
├── data.py               # 24 hand-crafted borrowers, 8 mix presets, statement generation
├── llm.py                # OpenRouter client, prompt construction, tool-use loop, cost tracking
├── los_adapter.py        # Open LOS API adapter for live underwriting integration
├── engine.py             # SimulationEngine: origination, adjudication, booking, resolution
├── scoring.py            # RAROC scoring, baselines, confusion matrix, bootstrap CI, penalties
└── mock_llm.py           # Deterministic mock for testing without API calls
docs/
├── elo-vs-raroc-benchmark.md   # Full design & methodology documentation
└── lending-sim-design.md       # Original design doc and roadmap
```

## Documentation

See [docs/elo-vs-raroc-benchmark.md](docs/elo-vs-raroc-benchmark.md) for the full design document covering borrower construction, underwriting mechanics, RAROC scoring, the three-Elo rating system, baselines, and pricing analysis.
