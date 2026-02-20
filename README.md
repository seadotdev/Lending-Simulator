# Loanville2 — The LLM Lending Simulator

A multi-agent simulation where distinct LLM personas compete as business lenders. They originate loans from a pool of hard-coded businesses, balancing risk, detecting fraud, managing portfolio concentration, and outbidding each other on terms.

## How It Works

1. **Pipeline Distribution** — 12 pre-built business dossiers (with 12-month bank statements) are broadcast to all lender agents.
2. **Underwriting** — Each LLM evaluates every application for fraud, creditworthiness, and portfolio fit, then outputs a JSON decision with an optional term sheet.
3. **Deal Adjudication** — When multiple lenders approve the same borrower, the best offer wins. Single-offer deals are auto-booked.
4. **Resolution** — The engine fast-forwards loan lifecycles. Good businesses repay in full, bad businesses default partway through, and frauds default immediately.
5. **Scoring** — Lenders are ranked by risk-adjusted return with penalties for funding fraud and breaching sector concentration limits.

### The Borrower Pool

| Type  | Count | Behavior |
|-------|-------|----------|
| Good  | 5     | Full repayment with interest |
| Bad   | 4     | Partial payments then default (hidden red flags: customer concentration, margin compression, grant dependency, revenue decline) |
| Fraud | 3     | Immediate default (red flags: round-number deposits, circular transfers, fabricated consistency) |

### The Lenders

| Lender | Persona | Default Model |
|--------|---------|---------------|
| Velocity Capital | Aggressive growth fintech | `meta-llama/llama-3.1-8b-instruct` |
| Heritage Trust Bank | Conservative regional bank | `google/gemma-2-9b-it` |
| Meridian Partners | Balanced middle-market | `mistralai/mistral-7b-instruct` |

Each lender starts with a pre-existing portfolio that creates sector concentration pressure, forcing genuine trade-offs.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

## Run

```bash
python run.py
# or
python -m loanville
```

## Configuration

Set `OPENROUTER_API_KEY` in your `.env` file. Get one at [openrouter.ai](https://openrouter.ai/).

The default models are cheap/fast for smoke testing. To change models, edit the `model` field in `loanville/data.py` under `_build_lenders()`.

## Project Structure

```
loanville/
├── __main__.py   # CLI entry point
├── models.py     # Data classes (Borrower, Lender, Loan, etc.)
├── data.py       # Hard-coded borrower dataset & lender configs
├── llm.py        # OpenRouter client, prompt construction, JSON parsing
├── engine.py     # Simulation engine (origination, adjudication, resolution)
└── scoring.py    # Final scoring and reporting
```
