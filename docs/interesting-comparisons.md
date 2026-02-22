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

### Implication

For production underwriting pipelines focused on credit quality (not fraud), structured financial summaries should be the primary input. Raw bank statements add value for fraud detection (see data mode design in the [benchmark doc](elo-vs-raroc-benchmark.md#42-data-modes)) but can actually hurt business credit assessment by overwhelming models with noise.
