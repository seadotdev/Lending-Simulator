# Interesting Comparisons

Findings from controlled experiments run against the Loanville simulation.

---

## 1. Quarterly Financials Beat Raw Bank Statements for Business Underwriting

**Experiment**: [`test_underwriting_modes.py`](../test_underwriting_modes.py) — compares two data modes across 4 models on the `balanced` mix (7 good, 4 bad, 4 fraud borrowers):

- **`statements_inline`**: Raw 12-month bank statements embedded in the prompt (no quarterly summaries)
- **`quarterly_only`**: Structured quarterly income statements only (no raw bank data)

**Finding**: When isolating business underwriting quality (ignoring fraud detection), structured quarterly financials consistently outperform raw bank statements. Three of four models caught more legitimately bad businesses with quarterly data.

### Bad Businesses Funded (lower is better)

4 bad businesses in the pool. Non-fraud defaults = total defaults minus frauds funded.

| Model | Bank Statements | Quarterly Financials |
|---|---|---|
| DeepSeek v3 (685B MoE) | 3 of 4 bad funded | 1 of 4 bad funded |
| Llama 3.3 70B | **4 of 4 bad funded** | **0 of 4 bad funded** |
| Nemotron 49B | 1 of 4 bad funded | 2 of 4 bad funded |
| Qwen3 30B (3B active) | 2 of 4 bad funded | 1 of 4 bad funded |

### Financial Impact

| Model | Mode | Total Deployed | Principal Lost | Net P&L |
|---|---|---|---|---|
| DeepSeek v3 | Statements | $5,350,000 | $2,502,628 | -$2,150,632 |
| DeepSeek v3 | Quarterly | $3,350,000 | $763,766 | -$467,923 |
| Llama 3.3 70B | Statements | $4,600,000 | $1,431,713 | -$1,004,508 |
| Llama 3.3 70B | Quarterly | $2,050,000 | $350,000 | -$158,630 |
| Nemotron 49B | Statements | $2,850,000 | $764,410 | -$489,910 |
| Nemotron 49B | Quarterly | $4,200,000 | $1,462,785 | -$1,081,756 |
| Qwen3 30B | Statements | $4,300,000 | $1,532,747 | -$1,195,580 |
| Qwen3 30B | Quarterly | $3,600,000 | $1,043,429 | -$733,121 |

### Why This Happens

Structured quarterly income statements surface the metrics that matter for credit analysis: margin trends, revenue concentration, cash burn trajectory. Raw bank statements contain this information implicitly, but it's buried in transaction-level noise — models have to mentally reconstruct P&Ls from deposit and withdrawal patterns, and most fail to do so reliably.

The standout result is **Llama 3.3 70B**: it missed every bad business with bank statements (4/4 funded) but caught all four with quarterly financials (0/4 funded). Its quarterly-mode principal loss ($350K) came entirely from the one fraud it missed — its business underwriting was essentially perfect.

### The Exception

**Nemotron 49B** bucked the trend — better at spotting bad businesses from raw statements (1 miss) than from quarterly financials (2 misses). This suggests some architectures extract more signal from transactional patterns than structured summaries, though Nemotron's advantage reversed on overall P&L once fraud losses were factored in.

### Replication (Run 2)

A second independent run of the same experiment confirmed the original findings. All four directional results replicated:

| Model | Run 1 Score (Stmts → Fin) | Run 2 Score (Stmts → Fin) | Direction |
|---|---|---|---|
| DeepSeek v3 | -80% → -30% (Fin +50pp) | -67% → -29% (Fin +38pp) | Financials win (both) |
| Llama 3.3 70B | -49% → -31% (Fin +18pp) | -89% → -32% (Fin +56pp) | Financials win (both) |
| **Nemotron 49B** | **-30% → -58% (Stmt +28pp)** | **-39% → -57% (Stmt +18pp)** | **Statements win (both)** |
| Qwen3 30B | -45% → -39% (Fin +6pp) | -54% → -42% (Fin +13pp) | Financials win (both) |

Detection rates (catch %) also replicated: Nemotron 49B caught 79% with statements vs 71% with financials (run 2), remaining the only model where raw bank statements outperform structured financials. Llama 70B again showed the most dramatic gap — 46% catch with statements vs 92% with financials.

The effect sizes vary between runs (expected with stochastic LLM outputs), but the rank ordering is stable: Nemotron 49B is consistently the only exception to the financials-win pattern.

### Implication

For production underwriting pipelines focused on credit quality (not fraud), structured financial summaries should be the primary input. Raw bank statements add value for fraud detection (see data mode design in the [benchmark doc](elo-vs-raroc-benchmark.md#42-data-modes)) but can actually hurt business credit assessment by overwhelming models with noise.

---

## 2. Nemotron 49B's Bank Statement Advantage Does Not Generalise Across the Family

**Experiment**: [`test_nemotron_generalization.py`](../test_nemotron_generalization.py) — runs the same `statements_inline` vs `quarterly_only` comparison across the Nemotron model family to test whether the 49B's unusual raw-statement advantage (Section 1) is a family-level architectural trait.

**Models tested**: Nemotron Ultra 253B, Nemotron Nano 9B (free tier). The 70B, 49B, 30B, and 12B VL models failed mid-experiment due to OpenRouter weekly key limits. The 49B's original data (Section 1) provides a third reference point.

**Finding**: The 49B's raw-statement advantage is specific to that model, not a Nemotron family trait. The 253B Ultra shows the standard pattern (financials strongly beat statements), and the 9B Nano shows no difference on bad detection, with financials slightly ahead on fraud and overall score.

### Bad Business Detection (3 lenders × 4 bad = 12 decisions)

| Model | Params | Bank Stmts Catch | Financials Catch | Winner |
|---|---|---|---|---|
| Nemotron Ultra 253B | 253B | 9/12 (75%) | 11/12 (92%) | **Financials** (+17pp) |
| Nemotron Super 49B* | 49B | 3/4 (75%)* | 2/4 (50%)* | **Statements** (+25pp) |
| Nemotron Nano 9B | 9B | 11/12 (92%) | 11/12 (92%) | Tie |

*49B data from original experiment (Section 1), single-lender view (4 decisions, not 12).

### Fraud Detection

| Model | Params | Bank Stmts Catch | Financials Catch | Winner |
|---|---|---|---|---|
| Nemotron Ultra 253B | 253B | 8/12 (67%) | **12/12 (100%)** | **Financials** (+33pp) |
| Nemotron Nano 9B | 9B | 6/12 (50%) | 7/12 (58%) | Financials (+8pp) |

### Overall Score

| Model | Params | Bank Stmts Score | Financials Score | Delta |
|---|---|---|---|---|
| Nemotron Ultra 253B | 253B | -54.48% | **-28.44%** | Financials +26pp |
| Nemotron Super 49B* | 49B | -$490K P&L | -$1,082K P&L | **Statements better** |
| Nemotron Nano 9B | 9B | -67.93% | **-51.36%** | Financials +17pp |

### What This Means

The 49B Super's raw-statement advantage appears to be model-specific rather than architectural. Across the Nemotron family:

- **253B Ultra** follows the standard pattern: structured financials dominate for both credit quality (+17pp) and fraud detection (+33pp). Its financials-mode performance was strong — only 1 default, zero frauds funded, and a manageable -$80K P&L across $3M deployed.
- **9B Nano** shows no meaningful difference on bad business detection (both modes catch 92%), but struggles badly with false positives (48–52% FP rate) and fraud detection (50–58%) regardless of mode. At -51% to -68% scores, it is well below the -10% reject-all baseline and not useful for underwriting in either mode.
- **49B Super** remains the only Nemotron model where raw bank statements outperform structured financials on credit quality. One possible explanation: the 49B's architecture (Llama 3.3 base with Nemotron fine-tuning) hits a sweet spot where the model is large enough to process transaction-level patterns but not so large that it over-indexes on the noise.

### Size Floor

The 9B Nano can produce valid underwriting decisions (correct JSON format, approve/reject with reasoning), but its 48–52% false positive rate makes it unusable — it rejects half the good businesses. It scores -51% to -68%, far below the -10% reject-all baseline. The minimum useful Nemotron size for this task appears to be above 9B, likely at the 49B+ tier based on the original experiment data.

### Incomplete Models

The 70B Instruct, 30B Nano, and 12B VL Nano could not be tested due to API key weekly spending limits. The 12B VL is a vision-language model that may behave differently on text-only financial analysis. Re-running with a fresh key would fill these gaps and clarify whether the pattern is strictly 49B-specific or shared by any mid-size Nemotron variant.
