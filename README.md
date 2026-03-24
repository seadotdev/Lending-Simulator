# Lending Simulator — LLM Lending Benchmark

A synthetic commercial lending simulation that benchmarks LLM models as autonomous loan underwriters. Models compete head-to-head, each receiving identical borrower applications and lender constraints. The only variable is the model's analytical and pricing ability.

## How It Works

1. **Pipeline** — Business dossiers (12-month bank statements, quarterly income, narratives) are broadcast to all lender agents.
2. **Underwriting** — Each LLM evaluates every application, then outputs a structured decision (APPROVE/REJECT) with an optional term sheet (amount, rate, term).
3. **Adjudication** — Borrowers pick the lowest rate. Winner-takes-all. Capital limits enforced.
4. **Resolution** — The engine fast-forwards loan lifecycles. Good businesses repay, bad businesses default partway through, frauds default immediately.
5. **Scoring** — Lenders are scored on RAROC (risk-adjusted return on capital) with penalties for fraud, concentration, under-deployment, and volatility.

## Quick Start

```bash
make setup                # Clone submodules, install Python & Node deps
cp .env.example .env      # Add your API key (e.g. OPENROUTER_API_KEY)
```

Try it without an API key first:

```bash
make mock                 # Deterministic mock mode — no API key, no LOS
```

Then run a real simulation:

```bash
make sim                  # Full LOS simulation with default models
```

Run `make help` for all targets.

## CLI

Everything runs through `python -m loanville`. Key flags:

```bash
# Single run
python -m loanville --mix balanced --los-mode full

# Multi-week season
python -m loanville --season --weeks 10 --cohort-size 10 --season-mix realistic

# Use preset configs (see configs/ directory)
python -m loanville --season --scenario realistic-10w --lenders competitive-league

# Override the model for all lenders
python -m loanville --los-model anthropic/claude-sonnet-4

# Mock mode (no API key, deterministic)
python -m loanville --mock --allow-non-los-formal
```

## Configuration

### API Keys

Set at least one LLM provider key in `.env`:

```
OPENROUTER_API_KEY=...    # Recommended — access to 100+ models
ANTHROPIC_API_KEY=...     # Direct Anthropic access
OPENAI_API_KEY=...        # Direct OpenAI access
```

### Lender Presets

Lender presets in `configs/lenders/` map personas to models:

```yaml
# configs/lenders/competitive-league.yaml
- id: LND-001
  model: openai/gpt-4.1-mini
- id: LND-002
  model: google/gemini-2.5-flash
- id: LND-003
  model: deepseek/deepseek-chat-v3-0324
```

Available presets: `budget-league`, `competitive-league`, `default-league`, `sonnet-vs-flash`, `broken-league`

### Scenario Presets

Scenario presets in `configs/scenarios/` configure season parameters:

```yaml
# configs/scenarios/fast-pipeline.yaml
weeks: 5
cohort_size: 5
season_mix: realistic
los_config:
  disabled_guards: [spread_created, documents_uploaded]
```

Available presets: `fast-pipeline`, `realistic-10w`, `stress-20w`

### Borrower Mixes

Control the composition of the borrower pool:

| Mix | Good | Bad | Fraud | Focus |
|-----|------|-----|-------|-------|
| `realistic` | 9 | 2 | 1 | Typical commercial pipeline |
| `analyst` | 10 | 6 | 0 | DSCR analysis, margin compression |
| `fraud` | 5 | 0 | 4 | Bank statement fraud detection |
| `balanced` | 7 | 4 | 4 | All risk types |
| `hard` | 5 | 4 | 4 | Adversarial stress test |
| `stress` | 5 | 8 | 4 | Kitchen-sink |

### Scoring

Three Elo ratings track different dimensions:

| Rating | Measures | Win Condition |
|--------|----------|---------------|
| **Profit Elo** | RAROC-aligned economic utility | Higher risk-adjusted profit |
| **Credit Elo** | Decision correctness vs ground truth | Correct approve/reject |
| **DealShare Elo** | Market participation | Won the deal |

## Project Structure

```
loanville/          Core Python package (CLI, engine, scoring, LLM client)
open-los/           LOS submodule — Loan Origination System REST API
configs/            Lender and scenario presets (YAML)
contracts/          JSON schemas for integration artifacts
scripts/            Experiment runners and utilities
tests/              Regression tests
fixtures/           Test fixture data
leaderboard/        Elo leaderboard config and standings
web/                Browser-based season visualizer
docs/               Design docs and methodology
Makefile            Common tasks (make help)
```

## Documentation

See [docs/elo-vs-raroc-benchmark.md](docs/elo-vs-raroc-benchmark.md) for the full design document covering borrower construction, underwriting mechanics, RAROC scoring, the three-Elo rating system, baselines, and pricing analysis.

## License

[MIT](LICENSE)
