# Lending Simulator — LLM Lending Benchmark

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
make setup        # Clone submodules + install Python & Node deps
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

## Run

```bash
make sim          # Full LOS simulation (realistic mix)
make mock         # Mock mode (no API key, no LOS, deterministic)
make season       # Multi-week season via LOS
make season-mock  # Multi-week season in mock mode
make smoke        # Smoke test (1 week, 3 borrowers per model)
make view         # Browser-based season visualizer
```

Or call the CLI directly:

```bash
# LOS simulation
python -m loanville --mix realistic --los-mode full

# Mock mode (no API key needed)
python -m loanville --mock --allow-non-los-formal

# Preset-driven season run
python -m loanville \
  --economics balanced \
  --scenario realistic-10w \
  --lenders budget-league \
  --allow-non-los-formal --mock
```

Run `make help` for all available targets.

## Configuration

Set `OPENROUTER_API_KEY` in your `.env` file. Get one at [openrouter.ai](https://openrouter.ai/).

## Project Structure

```
loanville/                        # Core package
├── __main__.py                   # CLI entry point
├── models.py                     # Dataclasses: Borrower, LenderConfig, LenderDecision, SeasonConfig
├── data.py                       # 24 hand-crafted borrowers, mix presets, lender definitions
├── llm.py                        # OpenRouter client, prompt construction, tool-use loop
├── engine.py                     # SimulationEngine: origination, adjudication, booking, resolution
├── scoring.py                    # RAROC scoring, baselines, confusion matrix, bootstrap CI, penalties
├── season.py                     # Season mode: multi-week game with carry-forward capital/exposures
├── borrower_gen.py               # Procedural borrower generation + hybrid pool management
├── custom_tools.py               # Lender-created reusable tools that persist across season weeks
├── los_adapter.py                # Open LOS REST API adapter (SIM ↔ LOS translation)
├── scorecard.py                  # 3-layer scoring: hard gates, underwriting quality, market behavior
├── run_schema.py                 # UnderwritingRun contract (unifies Benchmark, Simulator, LOS)
├── run_logger.py                 # Append-only run storage and replay
├── champion.py                   # Champion/challenger operating model for policy changes
├── cost_tracking.py              # Token cost estimation for season mode
├── contracts.py                  # Contract helpers, APR normalization
├── benchmark_items.py            # Gold-labeled cases for evaluation harness
├── flywheel_cli.py               # Unified CLI for the LOS → Benchmark → Simulator loop
└── mock_llm.py                   # Deterministic mock for testing without API calls

open-los/                         # LOS submodule (Loan Origination System)
                                  # REST API, chat bot commands, CRM test tool schema

configs/                          # Presets
├── lenders/                      # Persona→model presets (e.g. budget-league)
└── scenarios/                    # Named season bundles (e.g. realistic-10w)

Makefile                          # Common tasks (make help)

scripts/                          # Utilities and experiment runners
├── check_mock_replay.py          # Determinism verification (hash comparison)
├── test_los_integration.py       # LOS integration tests
├── test_los_smoke.py             # LOS smoke tests
├── validate_contract_fixtures.py # Contract fixture validation
├── run_smoke_test.py             # 1-week, 3-borrower smoke test per model
├── run_season_test.py            # Conservative 3-model season (gentle mix)
└── ...                           # Additional experiment runners

contracts/                        # JSON schemas for integration artifacts
fixtures/                         # Test fixture data
tests/                            # Regression tests
leaderboard/                      # Elo leaderboard (config, matches, standings)
web/                              # Browser-based season visualizer (town scene + dashboards)
docs/                             # Design docs, roadmap, methodology
```

## Documentation

See [docs/elo-vs-raroc-benchmark.md](docs/elo-vs-raroc-benchmark.md) for the full design document covering borrower construction, underwriting mechanics, RAROC scoring, the three-Elo rating system, baselines, and pricing analysis.
