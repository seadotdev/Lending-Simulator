# Can LLMs Detect Financial Fraud? Lessons from Building a Multi-Agent Lending Simulator

*How we built Loanville2, a competitive lending simulation where LLM agents evaluate loan applications, detect fraud, and manage portfolio risk — and what we learned about tool use, model sizing, and the surprising fragility of giving models "more data."*

---

## The Setup

Loanville2 is a multi-agent lending simulation. Three LLM-powered lenders — each with a distinct risk persona, portfolio constraints, and sector exposure limits — compete to originate loans from a pool of synthetic businesses. Some businesses are healthy. Some have hidden red flags (customer concentration, margin compression, grant dependency). Some are outright fraudulent (circular transfers, fabricated deposits, impossibly consistent revenue).

The lenders don't know which is which. They get financial dossiers and have to figure it out.

After all decisions are made, the engine fast-forwards loan lifecycles: good businesses repay, bad ones default partway through, and frauds default immediately. Lenders are scored on risk-adjusted P&L, with penalties for funding fraud and breaching sector concentration limits. A "reject everything" strategy scores -8.3% (the opportunity cost of earning zero while a risk-free benchmark returns 5% annually). You have to actually lend to win — but lend to the wrong borrower and you're punished hard.

### The Fraud Signals

The three fraud types are designed around distinct detection patterns:

- **Round-number deposits**: CloudNet Logistics claims $3.5M revenue at 37% margins, but bank statements show suspiciously round deposit amounts ($50,000, $100,000, etc.) — a hallmark of fabricated financials.
- **Circular transfers**: BioGenesis Research receives 68% of its deposits from affiliated entities (BioGenesis Holdings LLC, BGH Capital Partners). The revenue is an internal shell game.
- **Fabricated consistency**: QubitTech Solutions shows deposits that vary by only +/- $200 across 12 months. Real businesses have seasonal variation, late payments, lumpy contracts. A coefficient of variation below 1% is a statistical impossibility.

Each fraud type requires different analytical techniques to detect. Round numbers need filtering. Circular transfers need source analysis. Fabricated consistency needs variance computation.

---

## Experiment 1: Does Model Size Matter?

We tested six models across four tiers, from DeepSeek V3 (~685B MoE) down to Qwen 2.5 7B:

| Model | Score | Frauds Funded | Defaults | Cost/Run |
|-------|-------|---------------|----------|----------|
| DeepSeek V3 (685B MoE) | -1.65% | 0 | 1 | $0.04 |
| Llama 3.3 70B | -25.6% | 1 | 2 | $0.02 |
| Nemotron 49B | -20.8% | 1 | 2 | $0.02 |
| Qwen3 30B (3B active) | -18.3% | 1 | 2 | $0.01 |
| Llama 3.1 8B | -4.7% | 0 | 0 | $0.01 |
| Qwen 2.5 7B | -6.2% | 0 | 0 | $0.01 |

Two surprises here:

**Small models (7-8B) scored better than medium models (49-70B).** Not because they're better at analysis, but because they reject almost everything. In a pool with 58% risky borrowers (hard mix), extreme conservatism accidentally works. It's the "broken clock" effect — always saying no avoids all losses, and you only pay the opportunity cost penalty.

**Frontier models (685B) are the only ones that can actively lend profitably.** DeepSeek V3 scored -1.65% by actually deploying $2.5M in capital while avoiding all frauds. It earned real returns instead of hiding. The gap between -1.65% and the perfect score (+5.1%) represents the deals it priced too aggressively or the bad businesses it couldn't distinguish from good ones.

### The Real Capability Cliff

Tool use (calling the bank statement analysis function) worked down to 7-8B models. Below 3B, tool use support disappears entirely. But the quality of *reasoning about tool results* degrades much earlier. A 70B model can call the tool and get data back — but it often can't synthesize a 100KB JSON dump into actionable fraud signals. This leads us to our central finding.

---

## Experiment 2: Do Bank Statements Help or Hurt?

This was the experiment that surprised us most.

We tested four data modes, controlling what financial evidence models receive:

1. **Full**: Quarterly income statement + tool to retrieve raw 12-month bank statements
2. **Quarterly only**: Just the quarterly summary table, no bank data
3. **Statements inline**: Raw bank statements dumped directly into the prompt
4. **Aggregate only**: Just annual totals + company narrative

### The Counterintuitive Result

At the 49-70B model tier, bank statements **hurt performance by 3-7 percentage points.**

| Model | Full (with statements) | Quarterly Only |
|-------|----------------------|----------------|
| Llama 3.3 70B | -26.98% | -19.83% |
| Nemotron 49B | -20.84% | -17.88% |

Why? Two failure modes:

1. **Tool avoidance**: Nemotron 49B never called the analyse_bank_statements tool in any of its 36 evaluations. The tool was available but unused. Yet the system prompt saying "you SHOULD call the tool" made the model less confident in its decisions, leading to worse calibration.

2. **Context pollution**: Llama 3.3 70B called the tool but received ~100KB of raw JSON per borrower. Buried in hundreds of individual transactions, the model couldn't reliably extract the signal. It's like giving someone a phone book when they need a phone number — more data made the task harder.

### Frontier Models Are Different

At the frontier tier, the picture changes. We tested DeepSeek V3 and Qwen3-235B:

| Model | Full (with statements) | Quarterly Only | Delta |
|-------|----------------------|----------------|-------|
| DeepSeek V3 | -16.57% | -12.47% | -4.1pp (hurt) |
| Qwen3-235B | -15.34% | -20.46% | +5.1pp (helped) |

Qwen3-235B is the only model where bank statements added genuine signal. It caught one additional fraud that was invisible in quarterly data alone. DeepSeek V3, despite using the tool on every evaluation, was still distracted by the raw data volume.

The takeaway: **raw data is not information.** Dumping 100KB of JSON into a context window doesn't help unless the model can process it analytically rather than impressionistically.

---

## Experiment 3: Sandboxed Bash — Teaching Models to Compute

This finding led us to the key architectural insight: instead of asking models to *read* bank statements, let them *query* bank statements.

We replaced the `analyse_bank_statements` tool (which returned raw JSON) with a `run_bash` tool powered by [just-bash](https://github.com/AshkanAe/just-bash), a Python port of a sandboxed bash executor. Each borrower evaluation gets its own sandbox with the bank statement data pre-loaded at `/data/bank_statements.json`. The model can run `jq`, `awk`, `grep`, `sort`, `uniq` — whatever it needs to compute statistics programmatically.

### What Models Actually Do With a Bash Sandbox

In a single evaluation of BioGenesis Research (circular transfer fraud), DeepSeek V3 spontaneously ran five jq queries across three tool rounds:

```bash
# 1. Check for round-number deposits
jq '[.[] | .deposits[] | select(.amount % 1000 == 0)]' /data/bank_statements.json

# 2. List unique deposit sources — spotted affiliated entities
jq '[.[] | .deposits[] | .description] | unique' /data/bank_statements.json

# 3. Monthly deposit totals
jq '[.[] | {month, total_deposits}]' /data/bank_statements.json

# 4. Deposit concentration by source
jq '[.[] | .deposits[] | {d: .description, a: .amount}]
  | group_by(.d)
  | map({source: .[0].d, total: (map(.a) | add), count: length})
  | sort_by(-.total)' /data/bank_statements.json

# 5. Count affiliated transfers specifically (unprompted follow-up)
jq '[.[] | .deposits[]
  | select(.description == "Transfer from BioGenesis Holdings LLC"
        or .description == "Transfer from BGH Capital Partners")]
  | length' /data/bank_statements.json
```

The concentration query returned:

```json
[
  { "source": "Transfer from BioGenesis Holdings LLC", "total": 1161720.0, "count": 12 },
  { "source": "Transfer from BGH Capital Partners", "total": 774480.0, "count": 12 },
  { "source": "Genova Pharmaceuticals", "total": 421486.66, "count": 12 },
  { "source": "LifeScience Direct", "total": 408313.34, "count": 12 }
]
```

68% of deposits from affiliated entities. The model correctly rejected with reasoning: *"significant revenue concentration from related-party transfers, which raises concerns about circular financing and revenue authenticity."*

For QubitTech (fabricated consistency), a single jq query computes the coefficient of variation:

```bash
jq '[.[] | .total_deposits]
  | (add / length) as $m
  | (map(. - $m | . * .) | add / length | sqrt) as $s
  | {mean: $m, stddev: $s, cv: ($s / $m * 100)}'
```

QubitTech: CV = **0.24%**. A healthy business (SkyFreight): CV = **5.7%**. The fabrication is obvious when computed, invisible when eyeballed.

### Sandbox Results vs Raw JSON

| Model | Old (Raw JSON dump) | New (Bash sandbox) | Quarterly Only |
|-------|--------------------|--------------------|----------------|
| DeepSeek V3 | -16.57% (hurt 4.1pp) | **-30.78%** (neutral) | -31.07% |
| Qwen3-235B | -15.34% (helped 5.1pp) | **-13.05%** (helped 2.4pp) | -15.41% |

The sandbox eliminated the noise problem for DeepSeek V3 entirely — from hurting by 4.1pp to essentially neutral (0.3pp). Qwen3-235B with the sandbox achieved the best absolute score of any configuration: **-13.05%**.

Note: DeepSeek V3's elevated numbers in this run were driven by Velocity Capital's aggressive persona having a bad variance run (-60.72%). Heritage Trust Bank and Meridian Partners both performed solidly. Qwen3-235B showed more consistent results across all three lender personas.

---

## What We Learned

### 1. More Data ≠ Better Decisions

The single most surprising finding: giving models access to detailed bank statements — the very data that contains the fraud signals — made most models perform *worse*. The bottleneck isn't data availability; it's data processing capability.

This has implications beyond lending. Any system that hands an LLM a large document and says "find the anomaly" is likely suffering from the same problem. The model sees everything but understands nothing specific.

### 2. Tools Should Enable Computation, Not Just Retrieval

The old `analyse_bank_statements` tool was a retrieval tool — it returned raw data. The new `run_bash` sandbox is a computation tool — it lets the model specify exactly what analysis to run and get back a precise answer.

This distinction matters enormously. With retrieval, the model must process 100KB of JSON in-context. With computation, the model writes a 50-character jq query and gets back a 5-line result. The cognitive load is fundamentally different.

Models with sandbox access ran 3-5 targeted queries per evaluation. They computed concentration ratios, checked for round numbers, calculated variance statistics. They worked like an analyst with a terminal, not a human trying to read a spreadsheet.

### 3. Model Size Thresholds Are Real but Non-Linear

We expected a smooth degradation curve from large to small models. Instead we found:

- **Frontier (>200B MoE)**: Can actively lend profitably. Detects most fraud. Benefits from computational tools.
- **Large (49-70B)**: Can reason about financials but drowns in raw data. Benefits from less information, not more.
- **Small (7-8B)**: Rejects everything. Accidentally safe but economically useless.
- **Tiny (<3B)**: Tool use breaks entirely.

The "sweet spot" for cost-effective lending analysis appears to be frontier MoE models (DeepSeek V3, Qwen3-235B) paired with computational tools. These models cost $0.50-2.80 per million tokens — an entire 12-borrower simulation costs $0.15-0.35.

### 4. Competitive Dynamics Reveal Persona Weaknesses

The multi-lender competition surfaces failure modes invisible in single-agent testing:

- **Velocity Capital** (aggressive) deploys capital quickly and earns high interest, but occasionally funds bad businesses that conservative lenders correctly reject.
- **Heritage Trust Bank** (conservative) catches more fraud but loses deals on pricing, leaving money on the table.
- **Meridian Partners** (balanced) consistently falls in the middle — never the worst but rarely the best.

The scoring system makes these trade-offs explicit: rejecting everything scores -8.3%, but funding one fraud can cost 25% of the loan's principal in regulatory penalty. The winning strategy is selective aggression — approve with conviction, reject with evidence.

### 5. Prompt Design Matters More Than You Think

The original system prompt said: *"You SHOULD call the analyse_bank_statements tool before making your decision."* This phrasing made models less confident even when they had enough information from quarterly data alone. The implied obligation to gather more data created decision paralysis.

The sandbox-era prompt is more directive: it suggests specific jq queries and explains what each one reveals. This turned the tool from an obligation into a capability — models used it when analytically motivated, not out of compliance.

---

## The Numbers in Context

The perfect score in our hard mix (5 good, 4 bad, 3 fraud) is approximately +5% — the theoretical return if an omniscient lender approved only good borrowers at its target yield and rejected everything else. No model achieved this. The best result was **-1.65%** (DeepSeek V3, easy mix) and **-13.05%** (Qwen3-235B with sandbox, hard mix).

The gap to perfect represents the real cost of imperfect information, miscalibrated pricing, and competitive losses. Closing that gap — from -13% to +5% — is the frontier of LLM-powered financial analysis.

---

## Try It Yourself

Loanville2 is designed as a benchmark. If you're evaluating LLM capabilities for financial analysis, tool use, or multi-agent coordination, you can run it against any OpenRouter-compatible model:

```bash
# Quick smoke test (mock mode, no API key needed)
python -m loanville --mock --mix hard

# Live run with a specific model
python test_bank_statements.py --live --model your/model-here --mix hard --modes full quarterly_only

# Model capability degradation test
python test_models.py
```

The scoring framework rewards the right things: profitable lending with fraud avoidance. The borrower pool has enough variety to test genuine analytical reasoning. And the sandbox integration means you can test whether your model actually benefits from computational tools or just generates noise.

---

*All experiments run via OpenRouter. Total API spend for the full experimental program: approximately $2.50.*
