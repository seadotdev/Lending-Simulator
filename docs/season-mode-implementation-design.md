# Season Mode Implementation Design (LOS-First, Extensible)

**Status:** Draft  
**Date:** 2026-02-24  
**Source baseline:** `SIM/docs/prd-season-mode.md`

---

## 1. Why this design

This design takes the Season PRD and makes three constraints first-class:

1. **Realism must keep growing** (fraud, economics, financial complexity, competitive dynamics).
2. **LOS hardening is a product output** (sim pressure should improve real lender workflows outside the sim).
3. **Underwriting inputs must scale** from synthetic to real-company artifacts and richer business data.

The design keeps single-match mode intact while introducing a new season kernel that can absorb complexity without repeated rewrites.

---

## 2. Architecture decision

### Decision: Strangler refactor into a `SeasonKernel`, not a one-shot rewrite

Current `SimulationEngine` in `SIM/loanville/engine.py` is useful but monolithic (origination, adjudication, booking, resolution all in one flow).  
A big-bang rewrite risks breaking scoring parity and LOS integration progress.

**Approach:**
- Extract reusable modules from the existing engine.
- Add a `SeasonEngine` that orchestrates week-by-week play.
- Keep `SimulationEngine` behavior stable for existing modes.
- Move progressively toward a modular architecture where week simulation is composable.

This gives fast delivery now and still supports a full internal rewrite over time.

---

## 3. Target architecture

```
SeasonEngine
  ├─ WeekPlanner
  │   ├─ CaseSource (static/synthetic/real-hybrid)
  │   └─ RegimeModel (macro/fraud difficulty schedule)
  ├─ UnderwritingGateway
  │   ├─ LOSGateway (preferred)
  │   ├─ DirectLLMGateway
  │   └─ MockGateway
  ├─ MarketEngine
  │   ├─ AdjudicationPolicy (rate/amount/speed)
  │   └─ CompetitiveDynamics (future: phased arrivals/refi)
  ├─ PortfolioLedger
  │   ├─ Capital state
  │   ├─ Exposure state
  │   └─ Active loan positions
  ├─ ResolutionEngine
  │   └─ Rolling aging/default/prepay/recovery
  ├─ BriefingEngine
  │   └─ Weekly lender context packet
  ├─ SeasonScorer
  │   └─ Credit quality + portfolio mgmt + efficiency
  └─ SeasonJournal (append-only events for replay/debug)
```

---

## 4. Core domain model

Add new models in a dedicated season module (for example: `SIM/loanville/season_models.py`):

```python
@dataclass
class SeasonConfig:
    season_id: str
    weeks: int = 10
    cohort_size_min: int = 4
    cohort_size_max: int = 6
    months_per_week: int = 2
    season_mix: str = "realistic"   # gentle/realistic/adversarial/escalating
    speed_scoring: bool = False
    custom_tools: bool = False
    seed: int = 42

@dataclass
class ActiveLoanState:
    loan_id: str
    lender_id: str
    borrower_id: str
    sector: str
    principal_outstanding: float
    apr_pct: float
    term_months: int
    age_months: int = 0
    months_before_default: int | None = None
    true_outcome: str = "good"
    status: str = "active"  # active/defaulted/repaid/prepaid

@dataclass
class SeasonLenderState:
    lender_id: str
    total_capital: float
    available_capital: float
    deployed_capital: float
    sector_exposure: dict[str, float]
    active_loans: list[ActiveLoanState]
    resolved_loans: list[LoanOutcome]
    weekly_snapshots: list[dict]
    total_tool_calls: int = 0
    total_evaluations: int = 0
    tool_creation_equiv_calls: int = 0
```

Design rule: season state is authoritative for carry-forward; weekly match output is an event source.

---

## 5. Engine boundaries and interfaces

Define explicit interfaces to support LOS-first execution and future complexity.

### 5.1 `UnderwritingGateway`
- Input: lender policy context + borrower artifact bundle + weekly briefing.
- Output: `UnderwritingRun` + `LenderDecision` + `EvaluationTelemetry`.
- Implementations:
  - `LOSGateway`: default path for realism and product hardening.
  - `DirectLLMGateway`: compatibility path.
  - `MockGateway`: deterministic testing path.

### 5.2 `CaseSource`
- Input: `SeasonConfig`, `week`.
- Output: borrower cohort + hidden truth labels + artifact manifest.
- Implementations:
  - `StaticCaseSource` (existing borrowers, deterministic shuffle).
  - `TemplateCaseSource` (procedural synthetic generation).
  - `HybridCaseSource` (real + synthetic + transformed fraud variants).

### 5.3 `MarketEngine`
- Input: all offers + lender capital + telemetry.
- Output: booked winners + deal outcomes (`booked`, `lost`, `abandoned`, `no_capital`).
- Rules plug-in:
  - baseline rate/amount sort
  - optional speed bonus
  - future phased arrival/deep-underwrite slots/refinancing rounds

### 5.4 `ResolutionEngine`
- Input: active loans + `months_per_week`.
- Output: week cashflows and resolved events (default/repay/prepay/recovery/workout).
- Must be deterministic and replayable for benchmark reproducibility.

---

## 6. LOS-first integration design

Season mode should harden real LOS workflows, not bypass them.

### 6.1 Tenant and policy strategy
- One persistent LOS tenant per lender per season (`tenant = lender_{id}_season_{season_id}`).
- Policy/version metadata includes season context:
  - `policy.params.season_id`
  - `policy.params.week`
  - `policy.params.case_source`

### 6.2 Lifecycle integration
- Keep using LOS underwriting APIs via `SIM/loanville/los_adapter.py`.
- Add season-aware lifecycle sync:
  - booking hooks to LOS loan/facility APIs
  - weekly resolution cashflows posted as LOS transactions
  - monitoring snapshots pulled from LOS audit/events endpoints

### 6.3 Hardening outputs for external lenders
Every season run emits:
- **Flow failure taxonomy** (stage guard failures, missing docs, invalid spreads).
- **Trace friction report** (repeated tool patterns, high-latency branches, high-token paths).
- **Conformance replay pack** (minimal reproducer cases for LOS regressions).

This converts simulation findings directly into production LOS improvements.

---

## 7. Financial artifact pipeline (synthetic + real + complex)

Create an artifact abstraction so underwriting complexity can increase without touching engine internals.

### 7.1 Artifact bundle contract

```python
@dataclass
class BorrowerArtifactBundle:
    borrower_id: str
    source: str   # static | synthetic | real | hybrid
    dossier: FinancialDossier
    raw_artifacts: list[dict]         # statements, ledgers, invoices, AR/AP, tax docs
    derived_features: dict[str, float]
    fraud_tags: list[str]
    provenance: dict[str, str]
```

### 7.2 Complexity growth model
- **V1:** current dossier + statements + quarterly.
- **V2:** AR/AP aging, invoice trails, payroll summaries, tax deltas.
- **V3:** multi-entity and multi-account structures, artifact inconsistencies, cross-doc contradictions.

This supports continuous realism upgrades without rewriting scoring or adjudication.

---

## 8. Speed scoring and efficiency telemetry

Use a normalized telemetry layer so speed scoring is consistent across LOS and non-LOS modes.

```python
@dataclass
class EvaluationTelemetry:
    lender_id: str
    borrower_id: str
    tool_calls: int
    doc_requests: int
    latency_ms: int
    tokens_in: int
    tokens_out: int
    llm_cost_usd: float
```

Rules:
- `tool_calls` for LOS path = count of `trace.steps` where `type=="tool_call"`.
- `tool_calls` for direct LLM path = instrumented call trace count.
- custom tool creation/update contributes equivalent calls to season efficiency, not per-deal speed.

---

## 9. Scoring design

Keep compatibility with current RAROC-oriented scoring while adding season-specific components.

### 9.1 Score structure
- **Credit Quality (70%)**: existing economic scoring accumulated across all resolved events.
- **Portfolio Management (20%)**:
  - average capital utilization banding
  - concentration discipline over weeks
  - post-default adaptation signal
- **Efficiency (10%)**:
  - evaluations per tool call
  - speed-tiebreak win rate
  - custom tool reuse score

### 9.2 Anti-gaming constraints
- Maintain hard constraints (default cap, ROE floor) from current scorer.
- Apply penalties if efficiency gains come with materially higher fraud/default incidence.

---

## 10. Implementation phases

### Phase 0: Seam extraction (1 week)
- Extract adjudication/resolution logic from `SimulationEngine` into reusable modules.
- Add `EvaluationTelemetry` capture path.
- Add `SeasonJournal` event logger.

**Exit:** existing single-match outputs unchanged.

### Phase 1: Season core (1-2 weeks)
- Implement `SeasonConfig`, `SeasonLenderState`, `SeasonEngine`.
- Rolling week loop with carry-forward capital/exposure/active loans.
- Weekly briefing generation.
- CLI: `--season`, `--weeks`, `--months-per-week`, `--season-mix`.

**Exit:** deterministic multi-week run with current static borrower set.

### Phase 2: LOS-backed season hardening (1-2 weeks)
- Persistent season tenants in LOS mode.
- Lifecycle sync for booked loans and weekly cashflow events.
- Reconciliation report between season ledger and LOS state.

**Exit:** LOS-backed season runs with no unreconciled balance drift.

### Phase 3: Speed scoring + telemetry parity (1 week)
- Add speed bonus in adjudication.
- Add efficiency component to season score.
- Enforce telemetry parity between LOS and direct modes.

**Exit:** >10% of competitive deals decided by speed in stress presets.

### Phase 4: Artifact pipeline expansion (2+ weeks)
- Implement `CaseSource` abstraction with static + template generators.
- Add hybrid/real artifact adapters.
- Add fraud transform packs with deterministic seeding.

**Exit:** reproducible ~50-case season with mixed artifact provenance.

### Phase 5: Custom tools (1-2 weeks)
- Add tool runtime interface:
  - `PiCoreToolRuntime` (preferred)
  - `ScriptToolRuntime` fallback
- Tooling phase between weeks.
- Tool creation/update cost accounting and reuse metrics.

**Exit:** tool creation persists week-over-week and impacts efficiency score.

### Phase 6: Advanced competitive dynamics (ongoing)
- Phased arrival and deep-underwrite slots.
- Refinancing rounds for performing loans.
- Dynamic funding/liquidity constraints and correlated macro regimes.

---

## 11. Verification strategy

### Determinism and replay
- fixed-seed replay test for full season output equality (events + score).
- week-by-week snapshot hash tests.

### Contract and LOS integration
- validate all run artifacts against `SIM/contracts/underwriting-run.v1.json`.
- validate LOS outputs against `LOS/integration/contracts/los-underwrite-result.v1.json`.
- ledger reconciliation tests between season state and LOS transactions.

### Regression safety
- keep existing single-match tests green.
- add parity tests proving non-season modes are unchanged.

---

## 12. Immediate build order (next sprint)

1. Create season module with config/state/journal dataclasses.  
2. Split current `SimulationEngine` into reusable weekly primitives.  
3. Implement `SeasonEngine.run_season()` with rolling resolution and weekly briefings.  
4. Add CLI flags and a minimal season report.  
5. Add deterministic tests for 2-week and 10-week runs.

This sequence delivers useful Season Mode quickly while preserving headroom for realism and LOS hardening.

