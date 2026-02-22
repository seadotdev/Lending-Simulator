# LOS Cutover Guide

> How to replace `llm.py` with your production LOS while keeping Loanville's benchmark, scoring, and flywheel infrastructure intact.

---

## 1) Architecture recap

The current system has three layers:

```
┌─────────────┐      ┌──────────────┐      ┌──────────────────┐
│  Benchmark   │      │     LOS      │      │    Simulator     │
│  (borrowers, │─────▶│  (llm.py)    │─────▶│  (engine.py +    │
│   gold labels)│      │              │      │   scoring.py)    │
└─────────────┘      └──────────────┘      └──────────────────┘
                            │
                            ▼
                     UnderwritingRun
                     (run_schema.py)
```

**The Run is the seam.** The old LOS (`llm.py`) and the new LOS both plug in at the same point: they consume a `Borrower` + `LenderConfig` and produce an `UnderwritingRun`. Everything downstream — scoring, Elo, champion/challenger, disagreement mining — consumes Runs, not `LenderDecision`s.

---

## 2) What the old LOS does

`llm.py` implements a single function that matters:

```python
async def evaluate_borrower(
    client, lender, borrower, semaphore, data_mode
) -> LenderDecision
```

It:
1. Builds a system prompt from `LenderConfig` (persona, limits, portfolio)
2. Builds a user prompt from `Borrower` (dossier, financials, narrative)
3. Calls OpenRouter with tool definitions (`run_bash` for jq queries)
4. Loops up to 3 tool-call rounds (max 5 parallel calls per round)
5. Extracts JSON from the final response → `LenderDecision`

Then `engine.py:run_origination()` wraps each `LenderDecision` into an `UnderwritingRun` via the `build_run()` bridge:

```python
run = build_run(borrower=borrower, lender=lender, decision=decision, source="simulator")
```

---

## 3) What the new LOS must produce

The new LOS must return an `UnderwritingRun` directly. No bridge needed.

### 3a) Required fields

These fields must be populated for the scorecard, Elo, and flywheel to function:

| Field | Type | Used by | Notes |
|---|---|---|---|
| `case.case_id` | str | Elo pairing, disagree mining | Must match `Borrower.id` |
| `case.source` | str | Filtering | `"los"`, `"simulator"`, `"production"` |
| `policy.policy_id` | str | Champion tracker, run logger | Unique per LOS config version |
| `policy.model` | str | Cost tracking, Elo grouping | OpenRouter model ID |
| `decision.action` | str | Gates, UW quality, Elo | `"approve"` or `"decline"` |
| `decision.terms.amount` | float | Gates, business score | Loan amount in dollars |
| `decision.terms.apr` | float | Gates, terms MAE | As decimal (0.095 = 9.5%) |
| `decision.terms.tenor_months` | int | Gates, payoff calc | Loan term |
| `decision.rationale.summary` | str | Gates (trace_supports_decision) | 2-4 sentence explanation |

### 3b) Fields that unlock higher scores

These are optional but strongly affect the scorecard:

| Field | Type | Score impact | Current old-LOS value |
|---|---|---|---|
| `decision.rationale.key_factors` | list[str] | +0.3 explainability (Layer B) | `[]` (scores 0.0) |
| `decision.rationale.what_would_change` | list[str] | +0.3 explainability (Layer B) | `[]` (scores 0.0) |
| `decision.confidence` | float 0-1 | Disagree mining severity ranking | `0.0` |
| `decision.prob_default_12m` | float 0-1 | Brier calibration score | `0.0` |
| `decision.risk_grade` | str A-F | Terms accuracy rubric | `""` |
| `decision.conditions` | list[str] | Gate: required_docs | `[]` |
| `decision.covenants` | list[str] | Benchmark rubric | `[]` |
| `trace.steps` | list[TraceStep] | Audit trail, gate checks | `[]` (traces in separate file) |
| `trace.cost.tokens_in` | int | Ops component (Layer C) | `0` |
| `trace.cost.estimated_cost_usd` | float | Cost budgeting | `0.0` |
| `trace.latency_ms` | int | Ops component (Layer C) | `0` |
| `inputs.missing_info` | list[str] | Gate: required_docs | `[]` |

### 3c) Fields you don't need to set

These are filled by other parts of the system:

| Field | Set by | When |
|---|---|---|
| `labels.gold` | `build_run()` or benchmark harness | At run creation (from `Borrower.true_outcome`) |
| `labels.outcome` | `engine._finalize_runs()` | After loan resolution |
| `scores.*` | `scorecard.score_run()` | When scoring is invoked |

---

## 4) The cutover: step by step

### Step 1: Implement the LOS interface

Create your LOS module. It must expose one async function:

```python
# new_los.py (or wherever your LOS lives)

from loanville.run_schema import (
    UnderwritingRun, RunCase, RunPolicy, RunInputs, RunTrace,
    RunDecision, DecisionTerms, DecisionRationale, TraceStep, TraceCost,
)
from loanville.models import Borrower, LenderConfig

async def evaluate(
    borrower: Borrower,
    lender: LenderConfig,
    api_key: str = "",
) -> UnderwritingRun:
    """Evaluate a borrower and return a complete UnderwritingRun."""
    run = UnderwritingRun()
    run.case = RunCase.from_borrower(borrower, source="los")
    run.policy = RunPolicy.from_lender(lender)
    run.inputs = RunInputs.from_dossier(borrower.dossier)

    # --- Your underwriting logic here ---
    # Call your model, run your tools, build your decision
    # ...

    run.decision = RunDecision(
        action="approve",  # or "decline"
        terms=DecisionTerms(amount=..., apr=..., tenor_months=...),
        rationale=DecisionRationale(
            summary="...",
            key_factors=["dscr", "revenue_trend", ...],
            what_would_change=["2+ NSFs", "revenue decline >15%", ...],
        ),
        confidence=0.85,
        prob_default_12m=0.03,
        risk_grade="B",
        conditions=["provide_tax_returns"],
        covenants=["min_dscr_1_20"],
    )

    run.trace = RunTrace(
        steps=[
            TraceStep(t="...", type="tool_call", name="run_bash",
                      args={"command": "jq ..."}, result={"stdout": "..."}),
            TraceStep(t="...", type="reasoning", content="DSCR = 1.35..."),
        ],
        latency_ms=4200,
        cost=TraceCost(tokens_in=15000, tokens_out=800, estimated_cost_usd=0.015),
    )

    return run
```

### Step 2: Wire it into the engine

Replace the origination phase in `engine.py`. The change is in `run_origination()`:

```python
# engine.py — in run_origination()

# BEFORE (old LOS):
from .llm import run_lender_evaluations
results = await asyncio.gather(*[
    run_lender_evaluations(self.client, lender, self.borrowers, ...)
    for lender in self.lenders
])
for lender, decisions in zip(self.lenders, results):
    self.all_decisions[lender.id] = decisions

# AFTER (new LOS):
from new_los import evaluate
for lender in self.lenders:
    decisions = []
    runs = []
    for borrower in self.borrowers:
        run = await evaluate(borrower, lender, api_key)
        runs.append(run)
        # Convert Run back to LenderDecision for adjudication compatibility
        decisions.append(LenderDecision(
            lender_id=lender.id,
            borrower_id=borrower.id,
            decision="APPROVE" if run.decision.action == "approve" else "REJECT",
            reasoning=run.decision.rationale.summary,
            term_sheet=TermSheet(
                loan_amount=run.decision.terms.amount,
                interest_rate=run.decision.terms.apr * 100,  # Run uses decimal, TermSheet uses %
                term_months=run.decision.terms.tenor_months,
            ) if run.decision.action == "approve" else None,
        ))
    self.all_decisions[lender.id] = decisions
    self.runs.extend(runs)
```

**Important**: Phases 3-5 (adjudication, booking, resolution) consume `LenderDecision` + `BookedLoan` + `LoanOutcome`. They don't touch the LOS. You only replace Phase 1-2.

### Step 3: Remove the `build_run()` bridge

Once the new LOS emits Runs directly, delete the bridge code at the end of `run_origination()`:

```python
# DELETE this block from run_origination():
borrower_map = {b.id: b for b in self.borrowers}
for lender in self.lenders:
    for decision in self.all_decisions.get(lender.id, []):
        borrower = borrower_map.get(decision.borrower_id)
        if borrower:
            run = build_run(borrower=borrower, lender=lender, decision=decision, source="simulator")
            self.runs.append(run)
```

The new LOS populates `self.runs` directly in Step 2.

### Step 4: Verify with benchmark

Run against the smallest mix to verify:

```bash
# Quick smoke test (9 borrowers, cheapest)
python -m loanville --mix fraud

# Then score the runs
python -m loanville.flywheel_cli score --policy <your-policy-id>

# Check that gates pass and scores are reasonable
python -m loanville.flywheel_cli status
```

What to check:
- All 10 gates pass (especially `sanity_apr` — remember Run stores APR as decimal, not percentage)
- `explainability > 0.0` (confirms `key_factors` and `what_would_change` are populated)
- Decision accuracy matches expectations for the mix
- Traces are populated in replay: `python -m loanville.flywheel_cli replay <run-id>`

### Step 5: Run champion/challenger against old LOS

Register both as policies and compare:

```python
import asyncio
from loanville.data import get_borrowers, get_lenders
from loanville.run_schema import UnderwritingRun, build_run
from loanville.run_logger import RunLogger, find_disagreements
from loanville.champion import ChampionTracker
from loanville.scorecard import score_runs

borrowers = get_borrowers("balanced")
lender = get_lenders()[0]
logger = RunLogger()
tracker = ChampionTracker()

# 1. Run old LOS (already has runs from previous sims)
old_runs = logger.load_by_policy(f"p_{lender.id}_{lender.model.replace('/', '_')}")

# 2. Run new LOS
from new_los import evaluate
new_policy_id = "new_los_v1"
new_runs = []
for b in borrowers:
    run = asyncio.run(evaluate(b, lender))
    run.policy.policy_id = new_policy_id
    logger.log(run)
    new_runs.append(run)

# 3. Score both
old_cards = score_runs(old_runs)
new_cards = score_runs(new_runs)

print(f"Old LOS avg: {sum(c.overall_score for c in old_cards)/len(old_cards):.1f}")
print(f"New LOS avg: {sum(c.overall_score for c in new_cards)/len(new_cards):.1f}")

# 4. Find where they disagree
disag = find_disagreements(old_runs, new_runs)
print(f"Disagreements: {len(disag)}")
for d in disag[:5]:
    print(f"  {d['case_id']}: old={d['decisions']['a']} new={d['decisions']['b']}")

# 5. Register and promote if better
tracker.register_challenger(new_policy_id, model=lender.model)
tracker.update_scores(new_policy_id, new_cards)
should, reason = tracker.should_promote(new_policy_id)
print(f"Promote? {should} — {reason}")
```

### Step 6: Cut over

Once the new LOS passes all criteria:

```bash
# Promote
python -m loanville.flywheel_cli promote --policy new_los_v1

# Verify
python -m loanville.flywheel_cli status
```

Then update `engine.py` to import from the new LOS by default and remove the `llm.py` import.

---

## 5) APR convention: the one gotcha

The old LOS and `TermSheet` use **percentage** (9.5 = 9.5%).
The Run schema `DecisionTerms.apr` uses **decimal** (0.095 = 9.5%).

The scorecard handles both:
```python
# scorecard.py:_check_sanity_apr
if apr > 1.0:
    apr_pct = apr          # percentage scale
else:
    apr_pct = apr * 100    # decimal scale
```

But be consistent. If your new LOS stores APR as decimal in the Run, and you convert back to `TermSheet` for adjudication, multiply by 100:

```python
term_sheet=TermSheet(
    interest_rate=run.decision.terms.apr * 100,  # 0.095 → 9.5
)
```

---

## 6) What you can delete after cutover

Once the new LOS is champion and stable:

| File/function | Safe to delete | Reason |
|---|---|---|
| `llm.py` (entire file) | Yes | Replaced by new LOS |
| `mock_llm.py` | Keep | Still useful for offline dev/testing |
| `build_run()` in `run_schema.py` | Yes | Bridge no longer needed |
| `run_lender_evaluations()` import in `engine.py` | Yes | New LOS wired directly |
| `TOOL_ANALYSE_BANK_STATEMENTS` | Yes | Legacy tool definition |
| `TOOL_RUN_BASH` | Maybe | Keep if new LOS uses same sandbox |
| `MODEL_PRICING` dict | Move | Move to new LOS or shared config |

Do **not** delete:
- `engine.py` (adjudication, booking, resolution are LOS-independent)
- `scoring.py` (RAROC scoring is LOS-independent)
- `run_schema.py` (the contract)
- `scorecard.py` (evaluation harness)
- `champion.py` (operating model)
- `run_logger.py` (observability)
- `flywheel_cli.py` (CLI)
- `data.py` (borrower pool — these are your benchmark cases)

---

## 7) Parallel running (recommended)

Rather than a hard swap, run both LOS implementations in parallel for N matches:

```python
# In engine.py or a test script:
from loanville.llm import evaluate_borrower as old_evaluate
from new_los import evaluate as new_evaluate

# Run both on same borrower
old_decision = await old_evaluate(client, lender, borrower, semaphore)
new_run = await new_evaluate(borrower, lender, api_key)

# Compare
old_run = build_run(borrower, lender, old_decision)
diff = diff_runs(old_run, new_run)
if not diff["decision_match"]:
    print(f"DISAGREE on {borrower.id}: old={diff['decisions']['a']} new={diff['decisions']['b']}")
```

This is exactly what `find_disagreements()` was built for. Disagreements become your next batch of benchmark cases, closing the flywheel loop.

---

## 8) Checklist

```
[ ] New LOS returns UnderwritingRun directly
[ ] case.case_id matches Borrower.id
[ ] policy.policy_id is unique per config version
[ ] decision.action is "approve" or "decline"
[ ] decision.terms.apr is decimal (0.095 not 9.5)
[ ] decision.rationale.summary is non-empty
[ ] decision.rationale.key_factors populated (for explainability)
[ ] decision.rationale.what_would_change populated (for explainability)
[ ] trace.steps populated with tool calls
[ ] All 10 scorecard gates pass
[ ] Smoke test on fraud mix (9 borrowers) succeeds
[ ] Side-by-side comparison with old LOS complete
[ ] Disagreements reviewed and understood
[ ] Champion/challenger promotion criteria met
[ ] engine.py updated to use new LOS
[ ] build_run() bridge removed
[ ] Old llm.py deleted or archived
```
