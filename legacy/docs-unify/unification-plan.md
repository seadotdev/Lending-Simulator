# Unification Plan: LOS + SIM + UW Bench Feedback Loop

## Context

Three distinct repos need to work together as a closed-loop system for B2B lending simulation, benchmarking, and LOS validation — while remaining independently publishable as open-source projects.

| Repo | Language | Purpose |
|------|----------|---------|
| **LOS** (`seadotdev/open-los`) | TypeScript | Production loan origination system (28-table schema, REST API, agent orchestration) |
| **SIM** (`seadotdev/loanville2`) | Python | Lending simulation + LLM benchmark (36 borrowers, RAROC scoring, Elo tournament, champion/challenger) |
| **UW Bench** (`seadotdev/rl-benchmarks`) | Python | Real-world financial extraction benchmark (~60 companies, ~1,200 documents, real loan outcomes) |

**The goal:** Replace the SIM's built-in `llm.py` with calls to the actual Open LOS, connect UW Bench's real documents into the pipeline, and close the feedback loop where simulation outcomes improve both the LOS and the benchmarks.

**Why now:** The SIM has a clean integration seam (`UnderwritingRun` in `run_schema.py`), the LOS has a working REST API + agent scaffolding (801-line `agent.ts`), and the LOS cutover guide (`docs/los-cutover.md`) already specifies the exact contract.

---

## Architecture Overview

```
       ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐
       │      SIM      │  │     LOS      │  │    UW Bench      │
       │  (loanville2) │  │  (open-los)  │  │ (rl-benchmarks)  │
       │   Python      │  │  TypeScript  │  │   Python         │
       └───┬───┬───────┘  └──┬───▲───────┘  └──────┬───────────┘
           │   │   HTTP REST │   │                  │
           │   └─────────────┘   │                  │
           │          ▲          │                  │
           └──────────┼──────────┼──────────────────┘
              Shared benchmark   │
              data + scoring     │
                                 │
                    LOS owns the canonical schemas:
                    schemas/underwriting-run.schema.json
                    schemas/borrower-dossier.schema.json
                    schemas/extraction-result.schema.json
                    SIM + UW Bench conform to these
```

### Data Flow Loops

1. **SIM → LOS → SIM** (underwriting loop): SIM sends borrowers to LOS API → LOS agent evaluates → returns `UnderwritingRun` → SIM runs adjudication, booking, resolution, scoring
2. **UW Bench → SIM** (real data injection): Real company documents/financials → SIM `Borrower` objects with real ground truth outcomes
3. **UW Bench → LOS** (extraction quality): Real documents uploaded to LOS → extraction results compared against ground truth → feeds into scorecard
4. **SIM → UW Bench** (feedback): Disagreement mining produces new benchmark cases; simulation outcomes with real borrowers become default prediction data points
5. **SIM → LOS roadmap** (observability): Friction analysis on agent traces identifies which CLI capabilities to build next (capability-gated roadmap from `lending-sim-design.md` section 7.4)

---

## Phase 0: Canonical Schemas in the LOS (Week 1)

### What
Define the shared contracts (`UnderwritingRun`, `BorrowerDossier`, `ExtractionResult`) as JSON Schemas in the LOS repo — where they naturally belong. The SIM and UW Bench conform to these schemas via Python dataclasses that match the LOS definitions.

### Why
The LOS is the product and the most likely to be open-sourced first. It already has a `schemas/` directory with 10 JSON Schema files and a full OpenAPI spec (`openapi/v1.yaml`). The contracts for how underwriting works should be defined by the LOS, not by a separate package. The SIM and UW Bench are evaluation harnesses — they conform to the LOS's API, not the other way around.

### New Schemas Added to LOS

```
open-los/schemas/
├── deal.schema.json                 # (existing)
├── entity.schema.json               # (existing)
├── document.schema.json             # (existing)
├── ... (8 more existing)
├── underwriting-run.schema.json     # NEW — the canonical contract
├── borrower-dossier.schema.json     # NEW — what gets submitted for evaluation
└── extraction-result.schema.json    # NEW — structured extraction output
```

The `underwriting-run.schema.json` codifies the contract currently defined in Python at `loanville2/loanville/run_schema.py`: RunCase, RunPolicy, RunInputs, RunTrace, RunDecision, RunLabels, RunScores — translated to JSON Schema.

### Key Actions

1. **Translate** `loanville2/loanville/run_schema.py` → `open-los/schemas/underwriting-run.schema.json`
2. **Translate** borrower-related types from `loanville2/loanville/models.py` → `open-los/schemas/borrower-dossier.schema.json`
3. **Create** `open-los/schemas/extraction-result.schema.json` with UW Bench's 8 extraction fields (Revenue, COGS, EBITDA, NetIncome, CashAtBank, TradeDebtors, TradeCreditors, Stock)
4. **Add LOS TypeScript types** generated from these schemas in `open-los/packages/core/src/schema/contracts.ts`
5. **Update the OpenAPI spec** (`openapi/v1.yaml`) to reference these schemas for the new `/evaluate` endpoint
6. **Update SIM's `run_schema.py`** to conform to the LOS JSON Schema (add any new fields, adjust conventions to match)
7. **Update UW Bench** extraction types to match the LOS `extraction-result.schema.json`

### Contract Enforcement

- The LOS conformance tests (`packages/conformance/`) gain new test cases validating the evaluate endpoint produces valid `UnderwritingRun` JSON
- SIM integration tests validate that `run_schema.py` dataclasses round-trip through the LOS JSON Schema
- Schema versioning follows the LOS release cycle; new optional fields are backward-compatible

### Files Modified

| Repo | File | Change |
|------|------|--------|
| LOS | `schemas/underwriting-run.schema.json` | New — canonical contract |
| LOS | `schemas/borrower-dossier.schema.json` | New — evaluation input |
| LOS | `schemas/extraction-result.schema.json` | New — extraction output |
| LOS | `openapi/v1.yaml` | Reference new schemas in evaluate endpoint |
| LOS | `packages/core/src/schema/contracts.ts` | Generated TS types from schemas |
| SIM | `loanville/run_schema.py` | Updated to conform to LOS schema |
| SIM | `loanville/models.py` | Borrower/FinancialDossier aligned with LOS schema |
| UW Bench | `rubric_scoring.py` | Extraction fields aligned with LOS schema |

---

## Phase 1: LOS Adapter — Minimum Viable Integration (Weeks 2-3)

### What
Replace `llm.py` in SIM with an HTTP adapter that calls the Open LOS REST API, creating the first closed loop.

### New File: `loanville2/loanville/los_adapter.py`

Implements the same contract as `llm.py` but calls Open LOS instead of OpenRouter directly.

**API call sequence for one `(borrower, lender)` evaluation:**

| Step | LOS Endpoint | Purpose |
|------|-------------|---------|
| 1 | `POST /v1/entities` | Create company entity for borrower |
| 2 | `POST /v1/deals` | Create deal with borrower metadata |
| 3 | `POST /v1/deals/{id}/documents` (x2) | Upload bank statements JSON + quarterly income JSON |
| 4 | `POST /v1/deals/{id}/spread` | Create financial spread from dossier |
| 5 | `POST /v1/deals/{id}/stage-transitions` (x2) | Advance: broker → origination → underwriting |
| 6 | `POST /v1/deals/{id}/evaluate` **(NEW)** | Trigger LOS agent to evaluate |
| 7 | `GET /v1/deals/{id}` | Read back decision + `_context` |
| 8 | `GET /v1/deals/{id}/audit-events` | Get trace for `RunTrace.steps` |

Each lender gets its own tenant (`X-Tenant-Id: lender_{id}`) for natural isolation.

### New LOS Endpoint: `POST /v1/deals/{dealId}/evaluate`

Added to `open-los/packages/api/src/routes/underwriting.ts`.

This endpoint triggers the agent's `getRecommendation()` flow (from `agent.ts` line ~244) with:
- The lender's policy configuration (persona, model, limits, sector constraints)
- The deal's full context (documents, spreads, entity graph)

Returns structured response mapping to `UnderwritingRun`:
```json
{
  "decision": { "action": "approve", "terms": { "amount": 250000, "apr": 0.095, "tenor_months": 24 } },
  "rationale": { "summary": "...", "key_factors": [...], "what_would_change": [...] },
  "trace": { "steps": [...], "latency_ms": 4200, "cost": { "tokens_in": 15000 } },
  "confidence": 0.85,
  "risk_grade": "B"
}
```

### Wiring Agent to Real Services

The LOS `agent.ts` (801 lines) has real logic but placeholder service interfaces at the bottom (lines ~700-801). These must be wired to the actual services from `packages/core`:

- `DealService` → `packages/core/src/services/deals.ts`
- `StageService` → `packages/core/src/services/stages.ts`
- `DocumentService` → `packages/core/src/services/documents.ts`
- `SpreadService` → `packages/core/src/services/spreads.ts`
- etc.

Additionally, the `LLMClient` interface needs concrete implementations for multiple providers — **configurable via the evaluate request or LOS environment**:
- **OpenRouter** (multi-model, consistent with SIM's Elo benchmarks)
- **Anthropic** (direct Claude access, lower latency)

The `/evaluate` endpoint accepts an optional `provider` field. The LOS instantiates the matching `LLMClient` implementation. This enables the SIM to benchmark different models through the LOS (e.g., run Elo tournaments where each lender uses a different model via the LOS agent).

### Engine.py Modification

In `loanville2/loanville/engine.py`, add a `--los` code path alongside the existing `--mock` path:

```python
# In run_origination(), after the existing mock/live branches:
elif self.use_los:
    from .los_adapter import evaluate_via_los
    for lender in self.lenders:
        decisions = []
        for borrower in self.borrowers:
            run = await evaluate_via_los(borrower, lender, self.los_url)
            self.runs.append(run)
            decisions.append(run_to_decision(run))  # Convert back for adjudication
        self.all_decisions[lender.id] = decisions
```

Phases 3-5 (adjudication, booking, resolution) remain untouched — they consume `LenderDecision`, not the LOS.

### State Management

**Phase 1 approach:** Fresh SQLite in-memory database per simulation run. The LOS starts as a subprocess, runs evaluations, then shuts down. No persistent state.

**Phase 2+ approach (Season Mode):** Persistent tenants with carry-forward state between weeks.

### CLI Changes

```bash
# New flags in __main__.py
python -m loanville --los --mix fraud          # Use Open LOS
python -m loanville --los --los-url http://...  # Custom LOS URL
python -m loanville --mock                      # Existing mock mode (unchanged)
python -m loanville                             # Existing OpenRouter mode (unchanged)
```

### APR Convention
Per `los-cutover.md` section 5: `UnderwritingRun.DecisionTerms.apr` uses **decimal** (0.095 = 9.5%). `TermSheet.interest_rate` uses **percentage** (9.5 = 9.5%). The adapter converts: `term_sheet.interest_rate = run.decision.terms.apr * 100`.

### Verification

```bash
# Terminal 1
cd open-los && npm run dev

# Terminal 2
cd loanville2 && python -m loanville --los --mix fraud
# Expected: 9 borrowers evaluated via LOS, UnderwritingRun artifacts emitted

# Then score
python -m loanville.flywheel_cli score --policy los_v1
python -m loanville.flywheel_cli status
```

Check: all 10 scorecard gates pass, source="los", traces populated.

---

## Phase 2: UW Bench Document Bridge (Weeks 2-3, parallel with Phase 1)

### What
Convert real company data from UW Bench into SIM-compatible `Borrower` objects, enabling simulation runs with real ground truth.

### New File: `rl-benchmarks/exporters/dossier_exporter.py`

Reads from UW Bench's parquet data (configured in `rl-benchmarks/config.py`) and produces `Borrower` objects:

| UW Bench Source | SIM Target | Mapping |
|----------------|------------|---------|
| `FinancialInputs.Revenue` | `QuarterlyIncome.revenue` | Group by quarter |
| `FinancialInputs.Cogs` | `QuarterlyIncome.expenses` (partial) | Direct |
| `FinancialInputs.NetIncome` | `QuarterlyIncome.net_income` | Direct |
| `FinancialInputs.CashAtBank` | `MonthlyStatement.ending_balance` | Balance sheet proxy |
| `Companies.CompanyName` | `FinancialDossier.company_name` | Direct |
| `Loans.LoanStatus = "Defaulted"` | `Borrower.true_outcome = "bad"` | Outcome mapping |
| `Loans.LoanStatus = "Redeemed"` | `Borrower.true_outcome = "good"` | Outcome mapping |
| `Loans.DefaultDate` | `Borrower.months_before_default` | Computed from loan start |

### New File: `loanville2/loanville/data_real.py`

```python
def get_real_borrowers(min_periods: int = 4) -> list[Borrower]:
    """Load real borrowers from UW Bench exported data."""
```

### CLI Addition

```bash
python -m loanville --los --real-data       # Real borrowers + Open LOS
python -m loanville --real-data --mix easy   # Real borrowers + OpenRouter
```

### Constraint
Start with the subset of ~47 companies that have funded loans (real default/repayment outcomes). These are the only ones with genuine `true_outcome` ground truth.

---

## Phase 3: Extraction Quality in Scorecard (Week 4)

### What
When the LOS processes real documents (from Phase 2), compare its financial extraction against UW Bench ground truth. This becomes a new scoring dimension.

### Modification: `loanville2/loanville/scorecard.py`

Add `ExtractionQuality` to Layer B (UW Quality):

```python
class ExtractionQuality:
    weighted_score: float = 0.0       # From rubric_scoring.evaluate_extraction()
    field_scores: dict[str, float]    # Per-field accuracy
    error_types: dict[str, int]       # column_selection, unit_scaling, etc.
```

Uses the existing `rubric_scoring.evaluate_extraction()` function from UW Bench — no need to rewrite scoring logic.

**Updated scorecard weights:**
- Gates: Hard pass/fail (unchanged)
- UW Quality (55%): decision accuracy (30%) + explainability (10%) + **extraction quality (15%)**
- Business (35%): unchanged
- Ops (10%): unchanged

---

## Phase 4: Feedback Loop Automation (Weeks 5-6)

### What
Close the full loop: disagreement mining creates new benchmark cases, simulation outcomes feed back as ground truth, LOS friction analysis drives the capability roadmap.

### 4a. Disagreement Mining → Benchmark Cases

**Modify:** `loanville2/loanville/flywheel_cli.py` — add `export` command

```bash
los mine --top 20                    # Existing: find disagreements
los export --format uwbench          # NEW: export as UW Bench tasks
```

High-severity disagreements (where champion and challenger disagree on approve/decline) become new `BenchmarkItem` objects in UW Bench format.

### 4b. Simulation Outcomes → Default Prediction Data

**New file:** `rl-benchmarks/importers/sim_outcome_importer.py`

When SIM runs with real borrower data, the `UnderwritingRun.labels.outcome` (attached after resolution in `engine._finalize_runs()`) provides new data points for UW Bench's loan default prediction benchmark (the 47-deal dataset with 12 defaults).

### 4c. LOS Observability → Roadmap Engine

**New file:** `loanville2/loanville/observer.py`

Analyzes `RunTrace.steps` across all runs to identify friction:
- **Token hotspots**: Which tool calls consume the most tokens?
- **Repeated patterns**: Same jq query across different borrowers → candidate for a CLI tool
- **Missing capabilities**: Trace steps that fail or return empty → unmet needs
- **Stuck stages**: Deals that take >N tool rounds → UX problems

This maps directly to the capability-gated roadmap table:

| Friction Signal | Gates New Feature |
|----------------|-------------------|
| Repeated multi-bank queries | `los bank consolidate/anomalies` |
| AR/AP timing confusion | `los wc snapshot/reconcile` |
| Multi-facility waterfall errors | `los exposure/payment apply/covenants` |

---

## Phase 5: Integration Infrastructure (Start Week 1, complete by Week 3)

### Development Monorepo

```
unify/                              # This repo (private)
├── repos/
│   ├── open-los/                   # Git submodule → seadotdev/open-los (owns schemas)
│   ├── loanville2/                 # Git submodule → seadotdev/loanville2
│   └── rl-benchmarks/              # Git submodule → seadotdev/rl-benchmarks
├── integration/
│   ├── docker-compose.yml          # LOS + SIM + data stores
│   ├── integration_test.py         # End-to-end: borrower → LOS → score → mine
│   └── Makefile                    # make setup, make test, make run-loop
├── .github/workflows/
│   └── integration.yml
└── README.md
```

### Docker Compose

```yaml
services:
  los:
    build: ./repos/open-los
    ports: ["3000:3000"]
    environment:
      DB_PATH: ":memory:"
  sim:
    build: ./repos/loanville2
    depends_on: [los]
    environment:
      LOS_URL: "http://los:3000"
```

### CI Strategy

- **Per-repo CI** (unchanged): each repo runs its own tests independently
- **Integration CI** (new, in `unify`): starts LOS, runs SIM with `--los`, verifies full loop
- **Schema CI** (in LOS): JSON Schema validation + conformance tests for the evaluate endpoint. SIM and UW Bench CI includes a round-trip test against the LOS schemas.

---

## Phase 6: Season Mode with LOS Integration (Weeks 7-10, after Phases 1-4)

Implements `prd-season-mode.md` on top of the LOS integration:

- `SeasonEngine` wraps `SimulationEngine` for multi-week play
- Each lender gets a **persistent tenant** in the LOS (not in-memory sandbox)
- Portfolio state (capital, exposures, active loans) lives in LOS and carries forward between weeks
- Rolling resolution uses LOS loan account state machine (PENDING_APPROVAL → ACTIVE → CLOSED)
- Speed-to-offer scoring uses `trace.steps` count from the LOS agent
- Weekly portfolio briefings generated from LOS `GET /v1/deals?stage=monitoring` + covenant status

This is where the LOS integration pays off most — the LOS becomes a real system-of-record across weeks, not just a per-evaluation tool.

---

## Parallelism Summary

```
Week 1:      [Phase 0: Contracts] ──────── [Phase 5: Infrastructure]
Weeks 2-3:   [Phase 1: LOS Adapter] ────── [Phase 2: UW Bench Bridge]
Week 4:      [Phase 3: Extraction Scoring]
Weeks 5-6:   [Phase 4: Feedback Loop]
Weeks 7-10:  [Phase 6: Season Mode]
```

**Minimum viable integration** (closes the loop): Phase 0 + Phase 1. This alone replaces `llm.py` with the real LOS and enables scoring, Elo, and champion/challenger on LOS-produced runs.

---

## Risk Assessment

### High Risk

**LOS Agent Maturity.** The `agent.ts` (801 lines) has real orchestration logic but the service interfaces at lines ~700-801 are placeholders. The `LLMClient` interface has no concrete implementation. Before Phase 1 can work, the agent must be wired to real core services and a real LLM provider. This is the critical path.

**Mitigation:** Phase 1 includes a "thin agent" fallback — a simpler endpoint that uses the LOS services directly (create spread → compute ratios → apply rules) without the full agent orchestration. This lets the integration work while the full agent matures.

### Medium Risk

**Performance at scale.** SIM evaluates `N_borrowers x N_lenders` cases. With HTTP round-trips + LLM calls per evaluation, a 36-borrower x 3-lender run could take hours instead of seconds.

**Mitigation:** `--max-concurrent-los` flag for concurrent evaluations. Batch entity/document creation. For benchmarking, use `--mock` or `--los-lite` (rules-only, no LLM).

**Schema drift across repos.** Three repos evolving independently will diverge on data model assumptions.

**Mitigation:** LOS owns the canonical JSON Schemas. SIM and UW Bench CI includes a round-trip test against the LOS schemas. Schema changes go through the LOS repo first.

**Document format bridging.** SIM has pre-computed structured dossiers. LOS expects raw documents. Converting SIM → LOS → SIM is a lossy round-trip.

**Mitigation:** Use JSON as interchange format. The LOS already handles JSON document uploads and CSV spread ingestion.

### What to Defer

- TigerBeetle ledger integration (design doc Release 0) — use SIM's simple loan math for now
- Fraud packs (Release 3) — existing 4 fraud borrowers sufficient
- Multi-tenant book purchase (Release 4+) — far future
- Custom tool creation in Season Mode — implement after basic season loop works
- PDF document processing — start with JSON/CSV only

---

## Verification Plan

### Phase 1 Smoke Test
```bash
# Start LOS
cd repos/open-los && npm run dev

# Run SIM against LOS with smallest mix
cd repos/loanville2 && python -m loanville --los --mix fraud

# Verify
python -m loanville.flywheel_cli score --policy los_v1
python -m loanville.flywheel_cli status
# CHECK: all 10 gates pass, source="los", traces populated
```

### Phase 2 Validation
```bash
# Export real borrowers from UW Bench
cd repos/rl-benchmarks && python -m exporters.dossier_exporter --output ../loanville2/data/real_borrowers.json

# Run SIM with real data
cd repos/loanville2 && python -m loanville --los --real-data
# CHECK: real company names, real default outcomes match
```

### Full Loop Test
```bash
# Integration test runs the complete cycle
cd unify && make test-integration
# 1. SIM sends borrowers to LOS
# 2. LOS evaluates, returns UnderwritingRuns
# 3. SIM scores runs, runs Elo
# 4. Mine disagreements, export as UW Bench cases
# 5. Observer analyzes friction
# CHECK: all 5 stages complete, new benchmark cases produced
```

### Champion/Challenger Comparison
```bash
# Run old LOS (llm.py) and new LOS against same borrowers
python -m loanville --mix balanced           # old: OpenRouter direct
python -m loanville --los --mix balanced     # new: via Open LOS

# Compare
python -m loanville.flywheel_cli diff --run-a old_run_id --run-b new_run_id
# CHECK: disagreements identified, scores comparable
```
