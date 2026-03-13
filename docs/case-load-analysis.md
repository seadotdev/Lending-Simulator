# Case Load Analysis: Lessons from llmenron for Loanville Agent Capacity

**Status:** Draft
**Author:** Design Session
**Date:** 2026-03-13

---

## 1. Problem Statement

Loanville currently distributes borrower pipelines to lender agents with fixed concurrency knobs (`max_concurrent_per_lender=5`, CRM mode at 12) but has no empirical model for **when quality degrades as case load increases**. We configure throughput limits based on API rate-limit ergonomics, not on measured underwriting quality breakpoints.

The `strangeloopcanon/llmenron` project ran systematic experiments measuring exactly this — how LLM agent quality degrades as concurrent task load scales from comfortable (N=35) to stressful (N=105+) levels, calibrated against the Enron email corpus as a human-workload baseline. Their findings are directly applicable to how we should think about lender agent capacity in Loanville.

---

## 2. Summary of llmenron Findings

### 2.1 What they measured

llmenron simulates organizational inbox triage: prioritizing, routing, and responding to email threads. Agents process messages sequentially, each receiving the email content plus up to `k` recent messages from the same thread. Decisions are scored against gold labels for priority accuracy, reply type accuracy, action item extraction, hallucination, and SLA compliance.

### 2.2 Key experimental parameters

| Parameter | Capacity Eval | Scratchpad Frontier Eval |
|---|---|---|
| N-values (concurrent threads) | 35, 50, 70, 100 | 35, 50, 70, 105, 140 |
| Episodes per N | 3 | 40 |
| Messages per episode | 50 | 220 |
| Max API calls | 1,000 | 100,000 |
| Thread context | k=4 recent messages | k=4 recent messages |

### 2.3 Findings that matter for Loanville

1. **There is a measurable breakpoint (N\*) where quality degrades.** For email triage, this lands around N=105 concurrent threads — the P90 stress level from the Enron corpus. Quality curves are smooth until N\*, then fall off sharply.

2. **Explicit per-task state outperforms scratchpad/memory.** At N=105, agents with structured per-thread state significantly outperformed agents relying on a shared scratchpad. The scratchpad's char budget (5,000 chars) becomes a bottleneck as thread count grows — the agent cannot maintain enough context to make good decisions.

3. **"More agents" is not the first lever.** Their central thesis: *"The first big unlock for agent-native systems is better state. The second is shared state. 'More agents' comes later."* The org simulator (3-6 agents with a shared task board) showed that routing alone doesn't fix quality — the board's shared state is what matters.

4. **Sequential processing dominates.** Despite managing 100+ concurrent threads, all API calls are made sequentially within each episode. Parallelism is at the experimental level (multiple episodes), not at the agent level. This is a deliberate design choice — it isolates the state-management problem from the concurrency problem.

5. **SLA compliance is the canary.** Priority SLA thresholds (P0 < 5min, P1 < 20min, P2 < 120min) are the first metric to degrade as load increases, before raw accuracy drops. Time-sensitive decisions fail before analytical quality fails.

---

## 3. Mapping to Loanville

### 3.1 What is "case load" for a Loanville lender?

In llmenron, case load = concurrent email threads. In Loanville, the analogous dimension is **active deals requiring underwriting attention**, which varies by mode:

| Mode | Current load | Concurrency model |
|---|---|---|
| Single-run | 9-24 borrowers, all at once | 5 concurrent evals (semaphore) |
| CRM sim | 120 borrowers, streamed | 12 concurrent per lender |
| Season | 5/week × 10 weeks = 50 total | Sequential weeks, concurrent within-week |

### 3.2 Where llmenron's breakpoint maps

The llmenron breakpoint at N=105 is for a lightweight decision (triage an email). Underwriting a loan is a heavier cognitive task — reading financial statements, computing DSCR, spotting fraud signals, pricing risk. We should expect Loanville's N\* to be **much lower** than 105.

Concretely:
- **Single-run mode** (9-24 borrowers) is likely well below N\* — each evaluation is independent and the model has no cross-deal state to manage.
- **CRM sim** (120 borrowers, 12 concurrent) is the first mode likely to hit capacity limits, especially when 35% of cases require follow-up (incomplete intake). The model must track which cases are pending, which need follow-up docs, and maintain pricing consistency across the batch.
- **Season mode** (50 deals over 10 weeks) is the most exposed, because the lender must carry forward portfolio state: capital deployed, sector exposure, prior defaults, and strategy adjustments. This is exactly the scratchpad-vs-structured-state problem that llmenron identified.

### 3.3 State management is the bottleneck, not throughput

Loanville already has the right instinct here. Season mode's `SeasonLenderState` tracks capital, portfolio, exposure, and scratchpad as structured state. But there's an important nuance from llmenron: **the scratchpad (free-text strategy notes) will become the bottleneck before the structured fields do.**

Currently, season mode gives lenders a persistent scratchpad with no explicit size budget. As the season progresses and the portfolio grows, the model's system prompt balloons with portfolio history. llmenron found that a 5,000-char scratchpad budget is insufficient at N=105; Loanville's per-deal context (financial statements, bank data) is far richer, so the effective context pressure is higher even at lower N.

---

## 4. Recommendations

### 4.1 Measure N\* empirically for underwriting quality

Run a capacity sweep analogous to llmenron's frontier eval:

```
N-values:  6, 12, 18, 24, 36, 48
Episodes:  10 per N
Metric:    RAROC, fraud detection F1, pricing accuracy (APR tolerance)
```

Use the CRM sim infrastructure (`--crm-sim --crm-cases N`) with the `stress` mix, which has the highest bad/fraud ratio. Plot quality curves per metric against N. The point where RAROC or fraud-F1 drops by >1 standard deviation from the N=6 baseline is our N\*.

### 4.2 Add a scratchpad char budget to season mode

Introduce `--scratchpad-char-budget` (default: 4,000 chars) to season mode. When the lender's scratchpad exceeds the budget, the engine should prompt the model to compress it before the next week's evaluation. This mirrors llmenron's finding that unbounded scratchpad degrades quality at scale.

### 4.3 Structured per-deal state over narrative memory

For season mode, replace or supplement the free-text scratchpad with structured per-deal state:

```python
@dataclass
class DealMemory:
    borrower_id: str
    decision: str           # APPROVE/REJECT/PASS
    key_risk: str           # one-line risk summary
    dscr: float | None
    sector: str
    amount: float
    week: int
```

The lender's system prompt would include a compact table of `DealMemory` records instead of (or alongside) the narrative scratchpad. This is the "explicit per-task state" that outperformed scratchpad in llmenron.

### 4.4 Add SLA-style canary metrics to CRM sim

llmenron found SLA compliance degrades before accuracy. Add time-sensitive metrics to CRM sim:

- **Follow-up latency**: how quickly the lender requests missing documents for incomplete applications
- **Decision consistency**: whether the same borrower profile gets the same decision across different positions in the queue (early vs late in the batch)
- **Capital allocation drift**: whether pricing discipline degrades as the lender processes more applications (fatigue proxy)

These serve as early-warning signals for quality degradation before RAROC or fraud detection measurably drops.

### 4.5 Evaluate shared state for multi-lender coordination (future)

llmenron's org simulator found that a shared task board with routing, assignee tracking, and status was more valuable than adding more agents. When Loanville eventually adds inter-lender features (syndication, secondary market), the design should prioritize a shared deal board over independent agent proliferation.

---

## 5. Implementation Priority

| Priority | Change | Effort | Depends on |
|---|---|---|---|
| P0 | Capacity sweep experiment (4.1) | Low — uses existing CRM infra | Nothing |
| P1 | Scratchpad char budget (4.2) | Low — add CLI arg + compression prompt | Nothing |
| P1 | Structured deal memory (4.3) | Medium — new dataclass + prompt engineering | Season mode |
| P2 | Canary metrics (4.4) | Medium — new scoring dimensions in CRM | CRM sim |
| P3 | Shared state design (4.5) | Design only — no implementation yet | Syndication roadmap |

---

## 6. Open Questions

1. **What is the right N\* sweep range for season mode?** The within-week cohort is small (5), but the accumulated portfolio state is the real load. Should we sweep `weeks × cohort_size` as total N, or treat weekly cohort as the unit?

2. **Should scratchpad compression be model-driven or rule-based?** Model-driven (ask the LLM to summarize its own notes) is more flexible but costs an extra API call per week. Rule-based (FIFO eviction of oldest entries) is cheaper but may discard strategically important notes.

3. **How does tool-call depth interact with case load?** llmenron's agents made simple triage decisions. Loanville agents can run multi-step bash queries on bank statements. Does deeper tool use make agents more resilient to high case load (better evidence) or less resilient (more context consumed)?

---

## Appendix A: llmenron Repo Structure

Key source files from `strangeloopcanon/llmenron`:

| File | Purpose |
|---|---|
| `scripts/llm_capacity_eval.py` | Main inbox capacity evaluation harness |
| `scripts/scratchpad_frontier_eval.py` | Scratchpad-vs-thread-state evaluation at scale |
| `scripts/agent_org_simulator.py` | Multi-agent org simulation with shared boards |
| `scripts/regime_task_scenario.py` | Enron regime shift detection and scenario gen |
| `scripts/judge_llm_capacity_run.py` | LLM-as-judge evaluation of run outputs |

## Appendix B: Current Loanville Capacity Knobs

| Parameter | Default | CLI flag | Mode |
|---|---|---|---|
| `max_concurrent_per_lender` | 5 | `--max-concurrent-per-lender` | Single-run |
| CRM cases | 120 | `--crm-cases` | CRM sim |
| CRM concurrency | 12 | `--crm-concurrency` | CRM sim |
| Season weeks | 10 | `--weeks` | Season |
| Cohort size | 5 | `--cohort-size` | Season |
| Deep UW slots | 0 (unlimited) | `--deep-uw-slots` | Season |
| Consistency samples | 0 (disabled) | `--consistency-samples` | Season |
