# YC-Bench vs Loanville2: Long-Horizon Business Simulation

## Context

Analysis of [YC-Bench](https://github.com/collinear-ai/yc-bench) — a deterministic benchmark where LLM agents play CEO of an AI startup for 1-3 simulated years, managing cash flow, employees, and task pipelines via CLI commands against a SQLite-backed discrete-event simulation — compared to Loanville2's competitive lending simulation.

---

## 1. What YC-Bench Does

- Agent starts with $250K, hires employees, accepts tasks from a market
- Tasks have domain-specific prestige requirements (research, inference, data, training)
- Monthly payroll auto-deducts; employees get 1% raises on task completion
- Prestige gates lock higher-value work behind progression ladders
- Bankruptcy (negative funds) = elimination
- Multi-turn: hundreds of CLI interactions over simulated months

### Results (3 models × 3 seeds)

| Model | Survival Rate | Max Return | Key Pattern |
|-------|--------------|------------|-------------|
| Gemini 3 Flash | 8/9 | ~$35M | Most consistent |
| GPT-5.2 | 7/9 | $43.5M | Highest ceiling, volatile |
| Sonnet 4.6 | 5/9 | ~$25M | Highest variance |

Critical finding: **>58% task win rate → survived. <40% → bankrupt every time.**

---

## 2. Relevance to Loanville2

### a) Persistent Scratchpad — **IMPLEMENTED**

YC-Bench's most important design pattern: agents maintain a **persistent scratchpad** (free-text notes) that survives context window truncation. Only the last 20 conversation rounds are retained, but the scratchpad persists across the entire simulation.

This directly addresses a Loanville2 gap: in season mode, models evaluate 5+ borrowers per week for 10 weeks, but have no persistent memory. A model that notices "sector X defaults often" in week 3 has lost that insight by week 7 due to context truncation.

**Implementation**: `SeasonLenderState.scratchpad` — a persistent string that:
- Appears in the portfolio briefing each week
- In mock mode: auto-populated from performance patterns (defaults by sector, pricing gaps, concentration warnings)
- In live mode: extracted from `SCRATCHPAD_UPDATE: <text>` directives in LLM reasoning
- Capped at ~500 chars to avoid prompt bloat
- Included in JSON export for analysis

### b) Compounding Economic Pressure

YC-Bench creates escalating pressure through auto-payroll + salary growth. Loanville2 has analogous mechanics:
- **Capital decay** (already implemented) — idle capital loses value each week
- **Capital adequacy elimination** (already implemented) — effective capital below threshold = eliminated
- **Portfolio aging** — existing loans mature, default, or prepay, constantly shifting available capital

The parallel validates our capital decay and elimination mechanics as effective sources of strategic pressure.

### c) Prestige/Progression Gating

YC-Bench locks high-value tasks behind prestige levels, forcing agents to bootstrap through easier work. Loanville2's borrower pool has a natural equivalent: good borrowers are easier to evaluate (clear financials) while fraud/bad borrowers require deeper analysis. Models that reject everything avoid losses but miss the progression to profitable lending.

**Future idea**: Formalize this as "underwriting reputation" — lenders who demonstrate good credit judgment (low default rate) could gain access to exclusive deal flow or better pricing.

### d) Difficulty Tiers with Seed Variation

YC-Bench runs each difficulty tier with 3 different seeds, giving cleaner variance estimates. Loanville2's season mode already supports seeds (`--seed`) but doesn't systematically run multiple seeds per configuration.

**Future idea**: Season tournament mode that runs each model configuration across 3+ seeds and reports mean ± std for more reliable rankings.

---

## 3. Key Differences

| Aspect | YC-Bench | Loanville2 |
|--------|----------|------------|
| Agent role | CEO (general business) | Loan underwriter (domain-specific) |
| Interaction model | CLI commands against SQLite | LLM tool calls against borrower data |
| Competition | Single-agent vs environment | Multi-agent competitive market |
| Duration | 100-300+ turns | 10 weeks × 5 borrowers = 50 decisions |
| Failure mode | Bankruptcy (cash=0) | Capital adequacy elimination |
| Memory | Scratchpad + last 20 rounds | Portfolio briefing + scratchpad (now implemented) |
| Scoring | Survival + total returns | RAROC + triple Elo |

---

## 4. Actionable Ideas Adopted

1. **Persistent scratchpad** — Implemented as `SeasonLenderState.scratchpad` with auto-population (mock) and LLM-driven updates (live)
2. **Context truncation resilience** — Scratchpad carries strategic memory across weeks regardless of context window limits
3. **Compounding pressure validation** — Confirms capital decay + adequacy elimination create meaningful strategic tension
