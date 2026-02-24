# LOS + SIM + UW Bench Unification Plan

Date: February 23, 2026

---

## 1. Context and Objective

Three repositories need to operate as a closed feedback loop for B2B lending simulation, LOS validation, and underwriting benchmarking — while remaining independently open-sourceable.

| Repo | Language | Role in the Loop |
|------|----------|-----------------|
| **LOS** (`seadotdev/open-los`) | TypeScript | System of record — lifecycle execution, accounting state, servicing, audit trail. Owns the canonical schemas. |
| **SIM** (`seadotdev/loanville2`) | Python | Orchestration — scenario packs, scorecards, Elo tournaments, champion/challenger, disagreement mining. |
| **UW Bench** (`seadotdev/rl-benchmarks`) | Python | Document intelligence — extraction quality benchmarking, real ground truth from analyst judgments and loan outcomes. |

**Primary goals** (from `lending-sim-design.md`):

1. Validate LOS operational value, throughput, and token efficiency under realistic lending workflows.
2. Evaluate LLM underwriting quality with reproducible benchmark runs (LOS-tooled vs raw-file holdouts).
3. Close the feedback loop so disagreements and failures automatically create better LOS capabilities and better benchmarks.
4. Objective 3 (shrewd operator gameplay) deferred until objectives 1+2 are stable.

**Design truth sources:**
- `SIM/docs/lending-sim-design.md` — master architecture
- `SIM/docs/los-cutover.md` — LOS integration contract
- `SIM/docs/prd-season-mode.md` — multi-round season mode
- `LOS/openapi/v1.yaml` + `LOS/schemas/` — API surface and existing schemas
- `UW Bench/BENCHMARK.md` — extraction methodology and results

---

## 2. Non-Negotiable Design Principles

1. **No cross-repo source coupling.** Repos integrate via versioned JSON Schema contracts and HTTP APIs. No importing internal code across repos.
2. **Contract-first integration.** JSON Schema + strict validation. Every shared artifact has a schema, an owner, and CI enforcement.
3. **Deterministic replay.** Benchmark decisions must be reproducible from seeded scenario packs with fixed manifests.
4. **Objective 1+2 before objective 3.** Business shrewdness gameplay only after LOS validation and underwriting evaluation are stable.
5. **Capability-gated complexity.** New world complexity (multi-bank, AR/AP, multi-facility) only ships when the corresponding LOS CLI capability exists.
6. **LOS is the product.** The LOS defines the canonical schemas for what a deal, decision, and underwriting run look like. SIM and UW Bench conform to these contracts.

---

## 3. Target End-State Data Flow

```
┌───────────────┐     ┌──────────────┐     ┌──────────────────┐
│      SIM      │     │     LOS      │     │    UW Bench      │
│  (loanville2) │     │  (open-los)  │     │ (rl-benchmarks)  │
│  Orchestrator │     │   Product    │     │  Doc Intelligence │
└──┬──┬─────────┘     └──┬──▲───────┘     └──────┬───────────┘
   │  │    HTTP REST     │  │                     │
   │  └─────────────────►┘  │                     │
   │         ▲              │                     │
   └─────────┼──────────────┼─────────────────────┘
             │              │
             │   LOS owns canonical schemas:
             │   schemas/underwriting-run.schema.json
             │   schemas/borrower-dossier.schema.json
             │   schemas/extraction-result.schema.json
             │   schemas/los-event-envelope.schema.json
             │
        SIM + UW Bench conform to these
```

**Loop sequence:**

1. **SIM generates** case/scenario pack (borrowers, seeds, mix metadata).
2. **SIM sends** each case to LOS via HTTP adapter (entity → deal → documents → spread → evaluate).
3. **LOS executes** underwriting (agent evaluates using real services + LLM) and emits `UnderwritingRun`.
4. **SIM scores** outcomes (quality, business, ops) via scorecard, runs Elo, tracks champion/challenger.
5. **UW Bench evaluates** extraction quality for linked cases (real documents vs ground truth).
6. **SIM merges** LOS + UW signals into promotion decisions.
7. **SIM mines** disagreements → new benchmark cases fed back to UW Bench + new conformance tests for LOS.
8. **Observer analyzes** friction in agent traces → identifies next LOS CLI capabilities to build (roadmap engine).

---

## 4. Ownership Matrix

| Artifact | Owner | Consumers | Format |
|----------|-------|-----------|--------|
| `underwriting-run.schema.json` | LOS | SIM, UW Bench | JSON Schema in `open-los/schemas/` |
| `borrower-dossier.schema.json` | LOS | SIM | JSON Schema in `open-los/schemas/` |
| `extraction-result.schema.json` | LOS | UW Bench, SIM | JSON Schema in `open-los/schemas/` |
| `los-event-envelope.schema.json` | LOS | SIM | JSON Schema in `open-los/schemas/` |
| Case/scenario packs | SIM | LOS (via adapter) | Python dataclasses conforming to LOS schema |
| Feedback candidates | SIM | UW Bench, LOS | Disagreement objects from `flywheel_cli.py` |
| Extraction ground truth | UW Bench | SIM scorecard | Parquet + exported JSON |
| Loan outcome ground truth | UW Bench | SIM scoring | Parquet (47 funded deals, 12 defaults) |
| Scorecard + Elo + champion/challenger | SIM | All (reporting) | Python (`scorecard.py`, `champion.py`, `elo_benchmark.py`) |
| Observer / friction reports | SIM | LOS roadmap | Python (`observer.py`) |

### Simulation Ownership Boundary

Both repos have simulation capabilities that must not conflict:
- **SIM (`loanville2`)**: Owns benchmark orchestration, scenario packs, scoring, Elo, champion/challenger, the flywheel loop.
- **LOS (`packages/simulation`)**: Internal CRM scenario testing only — used for LOS conformance and capability testing, not for benchmark scoring. Must not duplicate SIM's orchestration or scoring logic.

---

## 5. Shared Contract Layer

### 5.1 Schemas Added to LOS

```
open-los/schemas/
├── deal.schema.json                  # (existing)
├── entity.schema.json                # (existing)
├── document.schema.json              # (existing)
├── ... (8 more existing)
├── underwriting-run.schema.json      # NEW — canonical evaluation artifact
├── borrower-dossier.schema.json      # NEW — what gets submitted for evaluation
├── extraction-result.schema.json     # NEW — structured extraction output
└── los-event-envelope.schema.json    # NEW — domain events (stage transitions, facilities, loans, covenants)
```

The `underwriting-run.schema.json` codifies the contract currently defined in Python at `loanville2/loanville/run_schema.py`: RunCase, RunPolicy, RunInputs, RunTrace, RunDecision, RunLabels, RunScores.

### 5.2 Contract Versioning

- Schemas follow SemVer, tied to the LOS release cycle.
- New optional fields are backward-compatible (minor bump).
- Breaking changes require major bump + migration period where both old and new are accepted.
- Every schema file includes a `$schema` version field.
- Strict unit conventions enforced: APR is **decimal** (0.095 = 9.5%) in all contracts. The SIM's `TermSheet.interest_rate` (percentage scale) is an internal representation; the adapter converts at the boundary.

### 5.3 Contract Fixtures

Each schema ships with a fixture corpus in `open-los/conformance/fixtures/contracts/`:
- Happy-path examples (valid, all fields populated)
- Edge-path examples (optional fields omitted, boundary values)
- Invalid examples (for negative testing)

All three repos validate against these fixtures in CI.

### 5.4 Cross-Language Type Generation

The LOS (TypeScript) generates TypeScript types from JSON Schema in `packages/core/src/schema/contracts.ts`. The SIM and UW Bench (Python) maintain conforming dataclasses. The workflow for schema changes:

1. Change made in `open-los/schemas/*.schema.json`
2. LOS CI regenerates TypeScript types, runs conformance tests
3. SIM and UW Bench CI pull latest fixtures, validate their dataclasses round-trip correctly
4. If validation fails → breaking change detected → requires coordinated update

---

## 6. Implementation Roadmap

### Phase 0: Program Setup + Schema Foundation (Week 1)

**Deliverables:**
- Architecture decision records for: contract ownership, ledger strategy, simulation boundary
- Ownership map (repo, module, contract owners — per section 4)
- Data classification and open-source boundary policy
- JSON Schemas for all 4 shared artifacts (section 5.1)
- Fixture corpus (happy path + edge path)
- Python/TypeScript validators
- Contract CI checks added to all three repos

**Key actions — LOS:**
- Translate `loanville2/loanville/run_schema.py` → `open-los/schemas/underwriting-run.schema.json`
- Translate borrower types from `loanville2/loanville/models.py` → `open-los/schemas/borrower-dossier.schema.json`
- Create `open-los/schemas/extraction-result.schema.json` with UW Bench's 8 extraction fields
- Create `open-los/schemas/los-event-envelope.schema.json` for domain events
- Generate TypeScript types in `packages/core/src/schema/contracts.ts`
- Update `openapi/v1.yaml` to reference new schemas
- **Fix API-contract drift**: audit runtime routes against published OpenAPI and close gaps

**Key actions — SIM:**
- Update `loanville/run_schema.py` to conform to LOS JSON Schema
- Update `loanville/models.py` — align Borrower/FinancialDossier with LOS `borrower-dossier.schema.json`
- Add CI workflow (currently missing) with contract fixture validation

**Key actions — UW Bench:**
- Align extraction types in `rubric_scoring.py` with LOS `extraction-result.schema.json`
- Add output mode producing `extraction-result.v1.json` with case/policy/run correlation fields
- Add CI workflow (currently missing) with contract fixture validation

**Exit criteria:**
- All three repos pass the same fixture suite in CI.
- OpenAPI spec matches runtime routes (no drift).
- Ownership matrix and governance policy approved.

**Runs in parallel with:** Phase 5 (infrastructure).

---

### Phase 1: LOS Adapter — Minimum Viable Integration (Weeks 2-3)

**Deliverables:**
- SIM adapter calling LOS REST API and returning `UnderwritingRun`
- New LOS endpoint: `POST /v1/deals/{dealId}/evaluate`
- LOS agent wired to real core services and configurable LLM provider
- Dual-run mode: legacy `llm.py` vs LOS-backed output on identical cases
- Automated disagreement diff report between old and new

#### New File: `loanville2/loanville/los_adapter.py`

Replaces `llm.py` as the LOS integration point. API call sequence for one `(borrower, lender)` evaluation:

| Step | LOS Endpoint | Purpose |
|------|-------------|---------|
| 1 | `POST /v1/entities` | Create company entity for borrower |
| 2 | `POST /v1/deals` | Create deal with borrower metadata in `custom_fields` |
| 3 | `POST /v1/deals/{id}/documents` (x2) | Upload bank statements JSON + quarterly income JSON |
| 4 | `POST /v1/deals/{id}/spread` | Create financial spread from dossier line items |
| 5 | `POST /v1/deals/{id}/stage-transitions` (x2) | Advance: broker → origination → underwriting |
| 6 | `POST /v1/deals/{id}/evaluate` **(NEW)** | Trigger LOS agent to evaluate with lender policy |
| 7 | `GET /v1/deals/{id}` | Read back decision + `_context` |
| 8 | `GET /v1/deals/{id}/audit-events` | Get trace for `RunTrace.steps` |

Each lender gets its own tenant (`X-Tenant-Id: lender_{id}`) for natural isolation.

#### New LOS Endpoint: `POST /v1/deals/{dealId}/evaluate`

Added to `open-los/packages/api/src/routes/underwriting.ts`. Triggers the agent's evaluation flow with:
- The lender's policy configuration (persona, model, limits, sector constraints)
- The deal's full context (documents, spreads, entity graph)
- Configurable LLM provider (`provider` field: `"openrouter"` or `"anthropic"`)

Returns structured response conforming to `underwriting-run.schema.json`:
```json
{
  "decision": { "action": "approve", "terms": { "amount": 250000, "apr": 0.095, "tenor_months": 24 } },
  "rationale": { "summary": "...", "key_factors": [...], "what_would_change": [...] },
  "trace": { "steps": [...], "latency_ms": 4200, "cost": { "tokens_in": 15000 } },
  "confidence": 0.85,
  "risk_grade": "B"
}
```

#### Wiring Agent to Real Services (Critical Path)

The LOS `agent.ts` (801 lines) has real orchestration logic but **placeholder service interfaces at lines ~700-801** and **no concrete `LLMClient` implementation**. This is the single biggest technical blocker. Before Phase 1 can complete:

1. Wire placeholder interfaces to actual services from `packages/core`:
   - `DealService` → `packages/core/src/services/deals.ts`
   - `StageService` → `packages/core/src/services/stages.ts`
   - `DocumentService` → `packages/core/src/services/documents.ts`
   - `SpreadService` → `packages/core/src/services/spreads.ts`
   - `AuditService` → `packages/core/src/services/audit.ts`

2. Implement `LLMClient` with configurable providers:
   - `OpenRouterLLMClient` — multi-model, consistent with SIM's Elo benchmarks
   - `AnthropicLLMClient` — direct Claude access, lower latency
   - Provider selected per-request via the `/evaluate` endpoint's `provider` field

**Fallback strategy:** If the full agent wiring takes longer than expected, implement a "thin agent" path — a simpler endpoint that uses LOS services directly (create spread → compute ratios → apply deterministic rules) without LLM orchestration. This lets the integration work while the full agent matures.

#### Engine.py Modification

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

#### Data Model Mapping

| SIM (Python) | LOS (TypeScript) | Notes |
|---|---|---|
| `Borrower.id` | `deal.custom_fields.sim_borrower_id` | Stored for round-trip |
| `FinancialDossier.company_name` | `deal.borrower_name` | Direct |
| `FinancialDossier.annual_revenue` | `spread.line_items[category=revenue]` | Via spread API |
| `FinancialDossier.bank_statements` | Document upload (JSON) | Serialized as JSON blob |
| `FinancialDossier.quarterly_income` | Document upload (JSON) | Serialized |
| `LenderConfig.persona` | Evaluation policy config | Passed to `/evaluate` |
| `LenderConfig.model` | Agent LLM configuration | Sets which model the agent uses |
| `LenderDecision.decision` | `run.decision.action` | "approve"/"decline" |
| `TermSheet.interest_rate` | `run.decision.terms.apr * 100` | Decimal → percentage conversion |

#### State Management

**Phase 1:** Fresh SQLite in-memory database per simulation run. The LOS starts as a subprocess, runs evaluations, then shuts down. No persistent state needed.

**Season Mode (Phase 6):** Persistent tenants with carry-forward state between weeks.

#### CLI Changes

```bash
# New flags in __main__.py
python -m loanville --los --mix fraud           # Use Open LOS
python -m loanville --los --los-url http://...  # Custom LOS URL
python -m loanville --los --provider openrouter  # Specify LLM provider
python -m loanville --mock                       # Existing mock mode (unchanged)
python -m loanville                              # Existing OpenRouter direct mode (unchanged)
```

**Exit criteria:**
- `>=95%` scorecard gate pass rate on LOS-backed runs.
- No critical behavior regressions vs legacy `llm.py` on same cases.
- Replay determinism preserved (same seed → same case sequence).
- All `UnderwritingRun` artifacts validate against `underwriting-run.schema.json`.
- Dual-run disagreement report published and reviewed.

**Runs in parallel with:** Phase 2.

---

### Phase 2: UW Bench Document Bridge (Weeks 2-3, parallel with Phase 1)

**Deliverables:**
- Real company data from UW Bench exported as SIM-compatible `Borrower` objects
- SIM can run with real ground truth (funded loans with actual default/repayment outcomes)
- SIM → UW Bench manifest adapter and UW → SIM result adapter with case/run correlation

#### New File: `rl-benchmarks/exporters/dossier_exporter.py`

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

#### New File: `loanville2/loanville/data_real.py`

```python
def get_real_borrowers(min_periods: int = 4) -> list[Borrower]:
    """Load real borrowers from UW Bench exported data."""
```

Start with the ~47 companies with funded loans (real default/repayment outcomes).

#### Evaluation Signal Quality

UW Bench ground truth is **analyst-adjusted**, not raw document values. This means:
- Some "errors" are unavoidable (models can't replicate discretionary EBITDA adjustments)
- Scoring must separate **raw extraction accuracy** from **adjustment replication accuracy**
- Revenue and balance sheet fields are "cleaner" targets; EBITDA is noisiest

The extraction scoring component (Phase 3) must account for this split.

#### CLI Addition

```bash
python -m loanville --los --real-data        # Real borrowers + Open LOS
python -m loanville --real-data --mix easy    # Real borrowers + OpenRouter
```

**Exit criteria:**
- Real borrower export covers all ~47 funded companies.
- Linked UW artifact coverage for all designated benchmark cases.
- Real outcomes (default/repayment) match loan records.
- SIM runs successfully with real data through full pipeline.

---

### Phase 3: Extraction Quality + Scorecard Enrichment (Week 4)

**Deliverables:**
- Extraction quality scoring integrated into SIM scorecard
- UW Bench rubric results linked to SIM runs via case/policy/run identifiers
- Split scoring: raw extraction vs analyst-adjusted fields

#### Modification: `loanville2/loanville/scorecard.py`

Add `ExtractionQuality` to Layer B (UW Quality):

```python
@dataclass
class ExtractionQuality:
    raw_field_score: float = 0.0       # Clean fields (Revenue, CashAtBank, TradeDebtors)
    adjusted_field_score: float = 0.0  # Analyst-adjusted fields (EBITDA, NetIncome)
    weighted_score: float = 0.0        # Combined from rubric_scoring.evaluate_extraction()
    field_scores: dict[str, float]     # Per-field accuracy
    error_types: dict[str, int]        # column_selection, unit_scaling, wrong_line_item, etc.
```

Uses the existing `rubric_scoring.evaluate_extraction()` function from UW Bench — no rewrite needed.

**Updated scorecard weights:**
- Gates: Hard pass/fail (unchanged)
- UW Quality (55%): decision accuracy (30%) + explainability (10%) + **extraction quality (15%)**
- Business (35%): unchanged
- Ops (10%): unchanged

The `prob_default_12m` field on `RunDecision` is scored against real default outcomes via Brier calibration (already in `scorecard.py` lines ~295-298).

**Exit criteria:**
- Extraction quality scores populated for all real-document runs.
- Raw vs adjusted split reported separately.
- No regression in existing scorecard dimensions.

---

### Phase 4: Ledger and Servicing Truth Integration (Weeks 5-6)

**Deliverables:**
- SIM booking/resolution integrated with LOS lifecycle endpoints (facilities, loans, transactions)
- Reconciliation reports between SIM's fast-forward resolution and LOS's loan account state
- Canonical accounting path defined (TigerBeetle strategy decision)

#### Why This Phase Matters

`lending-sim-design.md` section 2.1 states: "TigerBeetle is the canonical source of truth for money movement." Currently neither the SIM nor the LOS uses TigerBeetle — both have their own loan math.

#### Implementation

**Step 1: Wire SIM booking to LOS facilities/loans** (required)
- When SIM books a loan (`engine.py` Phase 4), call LOS:
  - `POST /v1/deals/{id}/facilities` — create facility with terms
  - `POST /v1/loans` — create loan account (PENDING_APPROVAL → APPROVED → ACTIVE)
- When SIM resolves loans (`engine.py` Phase 5), record in LOS:
  - `POST /v1/loans/{id}/transactions` — payment, default, recovery events
  - Read back loan state to verify consistency

**Step 2: Reconciliation** (required)
- After each simulation run, compare SIM's `LoanOutcome` against LOS's loan account state
- Report: any unexplained mismatches in principal, interest, default status

**Step 3: TigerBeetle integration** (decision point)
- **If pursuing now:** Add TigerBeetle as the LOS's backing store for `LoanAccountService`, replacing in-memory/SQLite loan accounting. SIM trusts the LOS's numbers.
- **If deferring:** Document the decision explicitly. SIM's simple loan math remains the source of truth for scoring, and the LOS integration in this phase serves as a shadow ledger with reconciliation checking.
- **Recommendation:** Defer TigerBeetle to Phase 6 (Season Mode), where persistent multi-week state makes a real ledger essential. For single-run benchmarks, SIM's loan math is sufficient and simpler.

**Exit criteria:**
- No unexplained accounting mismatches on seeded regression packs.
- Reconciliation report passes on all scenario mixes.
- TigerBeetle decision documented as ADR with rationale.

---

### Phase 5: Integration Infrastructure (Start Week 1, complete by Week 3)

**Deliverables:**
- Development monorepo with submodules
- Docker Compose for running all three together
- Integration test suite
- Cross-repo nightly CI

#### Development Monorepo

```
unify/
├── repos/
│   ├── open-los/                    # Git submodule → seadotdev/open-los (owns schemas)
│   ├── loanville2/                  # Git submodule → seadotdev/loanville2
│   └── rl-benchmarks/               # Git submodule → seadotdev/rl-benchmarks
├── integration/
│   ├── docker-compose.yml           # LOS + SIM + data stores
│   ├── integration_test.py          # End-to-end loop test
│   └── Makefile                     # make setup, make test, make run-loop
├── .github/workflows/
│   └── integration.yml              # Nightly cross-repo integration
└── README.md
```

#### Docker Compose

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

#### CI Strategy

**Per-repo CI:**
- LOS: TypeScript lint + 161 conformance tests + type check + **schema fixture validation** (existing CI enhanced)
- SIM: **New CI workflow** — Python tests + mock simulation + scorecard tests + contract fixture validation
- UW Bench: **New CI workflow** — Python tests + benchmark dry-run + contract fixture validation

**Cross-repo nightly integration:**
1. Start LOS
2. Run SIM mini-pack against LOS adapter (`--los --mix fraud`)
3. Run UW mini-manifest extraction against LOS
4. Publish joined compatibility report with drift flags
5. Fail if contract validation or gate thresholds breached

---

### Phase 6: Closed Feedback Loop + Season Mode (Weeks 7-10)

**Deliverables:**
- Automated disagreement mining → benchmark case generation
- Simulation outcomes → UW Bench default prediction data
- LOS friction observer → capability roadmap
- Season Mode with LOS integration (from `prd-season-mode.md`)

#### 6a. Disagreement Mining → Benchmark Cases

**Modify:** `loanville2/loanville/flywheel_cli.py` — add `export` command

```bash
los mine --top 20                     # Existing: find disagreements
los export --format uwbench           # NEW: export as UW Bench tasks
los export --format los-conformance   # NEW: export as LOS conformance tests
```

High-severity disagreements (champion vs challenger disagree on approve/decline) become:
- New `BenchmarkItem` objects in UW Bench format
- New conformance test cases suggested for LOS

Severity thresholds + human triage rubric prevent feedback-loop poisoning from low-signal disagreements.

#### 6b. Simulation Outcomes → Default Prediction Data

**New file:** `rl-benchmarks/importers/sim_outcome_importer.py`

When SIM runs with real borrower data, `UnderwritingRun.labels.outcome` provides new data points for UW Bench's default prediction benchmark (47-deal dataset with 12 defaults).

#### 6c. LOS Friction Observer → Roadmap Engine

**New file:** `loanville2/loanville/observer.py`

Analyzes `RunTrace.steps` across all runs to identify friction:
- **Token hotspots**: Which tool calls consume the most tokens?
- **Repeated patterns**: Same jq query across different borrowers → candidate for a CLI tool
- **Missing capabilities**: Trace steps that fail or return empty → unmet needs
- **Stuck stages**: Deals that take >N tool rounds → UX problems

Maps directly to the capability-gated roadmap (from `lending-sim-design.md` section 7.4):

| Friction Signal | Gates New Feature |
|----------------|-------------------|
| Repeated multi-bank queries | `los bank consolidate/anomalies` |
| AR/AP timing confusion | `los wc snapshot/reconcile` |
| Multi-facility waterfall errors | `los exposure/payment apply/covenants` |
| Book transfer confusion | `los transfer reconcile` + `audit pack` |

#### 6d. Season Mode with LOS Integration

Implements `prd-season-mode.md` on top of the LOS integration:

- `SeasonEngine` wraps `SimulationEngine` for multi-week play
- Each lender gets a **persistent tenant** in the LOS (not in-memory sandbox)
- Portfolio state (capital, exposures, active loans) lives in LOS and carries forward between weeks
- Rolling resolution uses LOS loan account state machine (PENDING_APPROVAL → ACTIVE → CLOSED)
- Speed-to-offer scoring uses `trace.steps` count from the LOS agent
- Weekly portfolio briefings generated from LOS `GET /v1/deals?stage=monitoring` + covenant status
- **TigerBeetle integration here** if deferred from Phase 4 — multi-week persistent state demands a real ledger

**Exit criteria:**
- End-to-end feedback loop runs without manual data reshaping.
- Disagreement → benchmark case pipeline produces valid UW Bench tasks.
- Observer friction report generated from real run traces.
- Season Mode runs 10-week season with LOS-backed evaluation and carry-forward state.

---

### Phase 7: Capability-Gated Complexity Expansion (Ongoing)

Sequencing — only when corresponding LOS capability ships:

| World Complexity | Gates On | LOS CLI Capability |
|-----------------|----------|-------------------|
| Multi-bank accounts | `los bank consolidate/anomalies` ships | Bank consolidation + anomaly detection |
| AR/AP timing | `los wc snapshot/reconcile` ships | Working capital snapshot + reconciliation |
| Multi-facility waterfalls | `los exposure/payment apply/covenants` ships | Exposure tracking + payment allocation |
| Book transfer / multi-tenant | `los transfer reconcile` + `audit pack` ships | Portfolio transfer + audit reporting |
| Fraud packs v2 | Extraction + categorization quality stable | Advanced fraud detection tools |
| Objective 3: shrewd operators | Objectives 1+2 KPIs stable for 3+ months | Business gameplay layer |

---

## 7. Parallelism Summary

```
Week 1:      [Phase 0: Schemas + Governance] ──── [Phase 5: Infrastructure]
Weeks 2-3:   [Phase 1: LOS Adapter] ───────────── [Phase 2: UW Bench Bridge]
Week 4:      [Phase 3: Extraction Scoring]
Weeks 5-6:   [Phase 4: Ledger Integration]
Weeks 7-10:  [Phase 6: Feedback Loop + Season Mode]
Ongoing:     [Phase 7: Capability-Gated Expansion]
```

**Minimum viable integration** (closes the loop): Phase 0 + Phase 1. This alone replaces `llm.py` with the real LOS and enables scoring, Elo, and champion/challenger on LOS-produced runs.

---

## 8. Risk Assessment

| # | Risk | Evidence | Likelihood | Impact | Mitigation |
|---|------|----------|:---:|:---:|------------|
| 1 | **LOS agent placeholder services** | `agent.ts` lines ~700-801 are interface stubs; no `LLMClient` implementation | High | Critical | "Thin agent" fallback (rules-only evaluation) while full agent matures. This is the single biggest blocker. |
| 2 | **API-contract drift** | LOS runtime routes exceed published OpenAPI | High | High | Phase 0: audit and close drift. CI blocks merges on schema mismatch. |
| 3 | **Dual simulation ownership** | LOS has `packages/simulation`; SIM is also orchestrator | High | High | Explicit boundary: SIM = benchmark orchestration, LOS simulation = internal capability testing only. |
| 4 | **Missing CI in SIM/UW** | Neither repo has GitHub Actions workflows | High | High | Phase 0: add baseline CI with contract fixture validation. |
| 5 | **Canonical ledger undefined** | Design doc requires TigerBeetle; neither repo uses it | High | High | Phase 4 reconciliation; TigerBeetle decision as ADR. Defer to Phase 6 if single-run benchmarks are sufficient without it. |
| 6 | **Performance at scale** | N_borrowers × N_lenders HTTP + LLM calls | High | Medium | `--max-concurrent-los` flag; batch API calls; `--los-lite` rules-only mode for benchmarking. Budget caps and per-run cost SLOs. |
| 7 | **Evaluation signal contamination** | UW ground truth is analyst-adjusted, not raw | High | Medium | Split scoring into raw extraction and adjustment replication. Revenue/balance sheet are clean targets; EBITDA is noisiest. |
| 8 | **Model/provider nondeterminism** | Provider models evolve; behavior changes between runs | High | Medium | Model version pinning, replay archives, variance thresholds for promotion decisions. |
| 9 | **Schema drift across repos** | Independent release cycles, two languages | Medium | High | LOS owns schemas. SIM/UW CI validates against LOS fixtures. Compatibility matrix maintained. |
| 10 | **Document format bridging** | SIM has structured dossiers; LOS expects raw documents | Medium | Medium | JSON as interchange format. LOS already handles JSON uploads and CSV spread ingestion. |
| 11 | **Feedback-loop poisoning** | Auto-mined disagreements can be low-signal noise | Medium | Medium | Severity thresholds + human triage rubric before benchmark promotion. |
| 12 | **Tenant/auth immaturity** | Header-based actor/tenant patterns are easy to misuse | Medium | High | Strengthen authN/authZ before production-grade loop. Acceptable for dev/benchmark use. |
| 13 | **Cost and latency blowouts** | Multiple LLM-heavy stages across LOS/SIM/UW | High | Medium | Budget caps, lite manifests, per-run cost SLOs, `--los-lite` mode. |
| 14 | **OSS packaging ambiguity** | Public/private boundaries not codified | Medium | Medium | Phase 0: data classification policy. Synthetic fixtures public; private borrower data stays private. |

---

## 9. Hard Gates Before Full Unification

| Gate | Criteria | Phase |
|------|----------|-------|
| **Contract Gate** | All repos pass shared contract fixtures in CI | Phase 0 |
| **Shadow Gate** | >=95% scorecard gate pass rate on LOS-backed runs; no critical regressions vs legacy | Phase 1 |
| **Repro Gate** | Same manifest replay within agreed variance bands | Phase 1 |
| **Extraction Gate** | Linked UW extraction artifacts for all benchmark cases; split scoring working | Phase 3 |
| **Ledger Gate** | No unexplained accounting mismatches on seeded packs; TigerBeetle ADR filed | Phase 4 |
| **Loop Gate** | End-to-end feedback loop runs without manual reshaping | Phase 6 |
| **Safety Gate** | Auth/tenant/audit controls hardened beyond header-based patterns | Before production |
| **Economics Gate** | Budget and latency SLO enforcement active; cost per run tracked | Before production |

---

## 10. Open-Source and Repo Separation Strategy

1. **Core logic in each repo is independently releasable.** No cross-repo source imports.
2. **Share only:** contracts (JSON Schema), fixtures, and generated clients.
3. **Sensitive connectors/data loaders** are modular and separately controlled (e.g., UW Bench's S3/parquet connectors are private; the harness and scoring are public).
4. **Publish synthetic fixtures/manifests.** Never publish private borrower artifacts.
5. **Maintain a compatibility matrix:** `LOS vX.Y` × `SIM vA.B` × `UW Bench vC.D` — tested versions.
6. **Licensing:** Each repo has its own license (LOS is MIT). Verify compatibility.

---

## 11. Governance

| Responsibility | Owner |
|---------------|-------|
| Contract schemas (JSON Schema) | LOS repo maintainer |
| Phase gate approval | Program lead |
| Champion/challenger promotion policy | SIM repo maintainer |
| Extraction benchmark integrity | UW Bench repo maintainer |
| Integration failure triage | Rotating weekly on-call |
| Architecture review | Monthly, all maintainers |

---

## 12. Success Metrics

**Program-level:**
- Time from disagreement discovery → benchmark/test insertion (target: < 1 week)
- Champion promotion cadence without regression (target: monthly)
- Reproducibility rate across reruns (target: >99% on seeded packs)
- Cost per validated improvement (tracked, no initial target)

**Operational:**
- LOS throughput: deals processed per unit time under sim load
- UW quality: loss rate at fixed approval rate, AUC improvement over baseline
- Extraction quality: weighted accuracy improvement per LOS release
- Drift incidents: unresolved cross-repo breakages per month (target: 0)

---

## 13. Verification Plan

### Phase 1 Smoke Test
```bash
# Start LOS
cd repos/open-los && npm run dev

# Run SIM against LOS with smallest mix
cd repos/loanville2 && python -m loanville --los --mix fraud

# Verify
python -m loanville.flywheel_cli score --policy los_v1
python -m loanville.flywheel_cli status
# CHECK: all 10 gates pass, source="los", traces populated, schema-valid
```

### Dual-Run Comparison (Phase 1)
```bash
# Old LOS
python -m loanville --mix balanced

# New LOS
python -m loanville --los --mix balanced

# Compare
python -m loanville.flywheel_cli diff --run-a old_run_id --run-b new_run_id
# CHECK: disagreements identified and reviewed
```

### Phase 2 Validation
```bash
# Export real borrowers
cd repos/rl-benchmarks && python -m exporters.dossier_exporter --output ../loanville2/data/real_borrowers.json

# Run SIM with real data
cd repos/loanville2 && python -m loanville --los --real-data
# CHECK: real company names, real default outcomes, extraction scores populated
```

### Full Loop Test (Phase 6)
```bash
cd unify && make test-integration
# 1. SIM sends borrowers to LOS
# 2. LOS evaluates, returns schema-valid UnderwritingRuns
# 3. SIM scores runs, runs Elo
# 4. UW Bench scores extraction quality
# 5. Mine disagreements, export as benchmark cases + LOS conformance tests
# 6. Observer analyzes friction, produces roadmap report
# CHECK: all 6 stages complete, new benchmark cases produced, no manual reshaping
```

---

## 14. Immediate Next Actions

1. Approve ownership matrix and contract list (section 4).
2. Stand up shared schema fixtures and CI checks in all three repos (Phase 0).
3. Begin LOS agent service wiring — this is the critical path (Phase 1 prerequisite).
4. Start Phase 1 with LOS-in-SIM shadow mode on a small fixed fraud+analyst pack.
5. Start Phase 5 infrastructure in parallel (Docker Compose, integration test harness).
6. Publish first compatibility report (`LOS x SIM x UW`) and iterate weekly.
