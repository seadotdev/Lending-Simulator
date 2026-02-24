# LOS + SIM + UW Bench Unification Plan

Date: February 23, 2026

## 1. Objective

Unify three repositories into one operating feedback loop while keeping each repository distinct and independently open-sourceable:

- LOS: `https://github.com/seadotdev/open-los`
- SIM: `https://github.com/seadotdev/loanville2`
- UW Bench: `https://github.com/seadotdev/rl-benchmarks`

Primary goals (aligned to design docs):

1. Validate LOS operational value and throughput under realistic lending workflows.
2. Evaluate underwriting quality with reproducible benchmark runs.
3. Close the feedback loop so disagreements and failures automatically create better LOS capabilities and better benchmarks.

## 2. Source of Design Truth

This plan is aligned to:

- `SIM/docs/lending-sim-design.md`
- `SIM/docs/los-cutover.md`
- Supporting architecture state in LOS and UW Bench codebases.

## 3. Target End-State Architecture

Keep all repos separate. Integrate via versioned contracts and APIs.

- `open-los` (LOS): system of record for lifecycle execution, accounting state, servicing operations, and audit trail.
- `loanville2` (SIM): orchestration, scenario packs, scorecards, champion/challenger, disagreement mining.
- `rl-benchmarks` (UW Bench): document-intelligence evaluation and extraction quality benchmarking.

Data/control flow:

1. SIM generates case/scenario pack.
2. LOS executes underwriting + lifecycle actions and emits execution artifacts.
3. SIM scores outcomes (quality, business, ops) and computes disagreements.
4. UW Bench evaluates document extraction/evidence quality for linked cases.
5. SIM merges LOS + UW signals into promotion decisions and next-case generation.

## 4. Non-Negotiable Design Principles

1. No cross-repo source coupling: no importing internal code across repos.
2. Contract-first integration: JSON Schema + strict validation.
3. Deterministic replay for benchmark decisions.
4. Objective 1+2 before objective 3 expansion.
5. Capability-gated complexity: only add world complexity when corresponding LOS capability exists.

## 5. Current-State Findings (What This Plan Accounts For)

- SIM already has a viable unification seam via `UnderwritingRun` and flywheel tooling (`run_schema.py`, `run_logger.py`, `flywheel_cli.py`).
- SIM cutover design already prescribes replacing legacy `llm.py` with LOS-backed evaluation returning `UnderwritingRun`.
- LOS runtime route coverage is broader than its published OpenAPI surface (contract drift risk).
- UW Bench already emits traceable artifacts (manifest, trace JSON/JSONL), but not yet SIM-compatible run artifacts by default.
- LOS has an internal simulation package, while SIM is also a simulation/orchestration engine (ownership boundary must be explicit).

## 6. Shared Contract Layer (Must Be Built First)

Create versioned contract artifacts (single source of truth):

1. `underwriting-run.v1.json`
- Owner: SIM.
- Canonical underwriting artifact consumed by SIM scoring/promotion and produced by LOS adapter.

2. `case-pack.v1.json`
- Owner: SIM.
- Scenario payload: borrower profile, docs, request, seed, mix metadata.

3. `los-event-envelope.v1.json`
- Owner: LOS.
- Domain events for stage transitions, facility state, loan transactions, monitoring, covenant outcomes.

4. `uw-extraction-result.v1.json`
- Owner: UW Bench.
- Extraction/evidence outcome linked to case/policy/run identifiers.

5. `feedback-candidate.v1.json`
- Owner: SIM.
- Standard disagreement object for backlog and benchmark promotion.

## 7. Implementation Roadmap

### Phase 0: Program Setup (1 week)

Deliverables:

- Architecture decision records.
- Ownership map (repo, module, contract owners).
- Data classification and open-source boundary policy.
- Versioning policy for contracts and APIs.

Exit criteria:

- Approved governance and compatibility strategy.

### Phase 1: Contract Hardening (2 weeks)

Deliverables:

- JSON Schemas for all shared artifacts.
- Fixture corpus (happy path + edge path).
- Python/TypeScript validators.
- Contract CI checks in all three repos.

Exit criteria:

- Every repo validates the same fixture suite in CI.

### Phase 2: LOS-in-SIM Shadow Cutover (2-3 weeks)

Deliverables:

- SIM adapter calling LOS interfaces and returning `UnderwritingRun`.
- Dual-run mode: legacy SIM LOS vs LOS-backed output on identical cases.
- Automated disagreement diff report.

Exit criteria:

- >=95% scorecard gate pass rate.
- No critical behavior regressions.
- Replay determinism preserved.

### Phase 3: Ledger and Servicing Truth Integration (2-3 weeks)

Deliverables:

- SIM booking/resolution integrated with LOS lifecycle endpoints (facilities/loans/transactions/arrears).
- Reconciliation reports between expected and realized balances.

Exit criteria:

- No unexplained accounting mismatches on seeded regression packs.

### Phase 4: UW Bench Integration (2-3 weeks)

Deliverables:

- SIM -> UW manifest adapter.
- UW -> SIM result adapter (with case/run correlation).
- Scorecard enrichment in SIM for extraction/evidence quality.

Exit criteria:

- Linked UW artifact coverage for all designated benchmark cases.

### Phase 5: Closed Feedback Loop Automation (3 weeks)

Deliverables:

- Automated disagreement mining with severity scoring.
- Auto-suggested LOS conformance tests from severe disagreements.
- Auto-suggested UW benchmark additions from extraction failures.
- Promotion workflow integration (champion/challenger gating).

Exit criteria:

- End-to-end loop runs without manual data reshaping.

### Phase 6: Capability-Gated Complexity Expansion (ongoing)

Sequencing (only when capability exists):

- Multi-bank accounts -> consolidation/anomaly capabilities.
- AR/AP timing -> working capital snapshot/reconcile capabilities.
- Multi-facility waterfalls -> exposure/payment allocation capabilities.
- Book transfer/multi-tenant cutover -> transfer + audit pack capabilities.
- Objective 3 (shrewd operator gameplay) only after objectives 1+2 are stable.

## 8. Repo-Specific Workstreams

### LOS (`open-los`)

- Close API-contract drift (runtime routes and OpenAPI must match).
- Publish stable integration subset for SIM.
- Expose consistent event export contract.
- Ensure policy/version metadata is traceable in emitted artifacts.

### SIM (`loanville2`)

- Elevate `UnderwritingRun` to external contract package.
- Build pluggable evaluator modes (`legacy_llm`, `los_adapter`).
- Keep champion/challenger and promotion logic as central governance.
- Merge LOS and UW signals into single scorecard/promotion decisions.

### UW Bench (`rl-benchmarks`)

- Add output mode producing `uw-extraction-result.v1`.
- Add strict case/policy/run correlation fields.
- Separate public synthetic benchmark packs from private connectors.
- Add deterministic SIM-triggered benchmark mode.

## 9. CI/CD, Release, and Quality Gates

Per-repo required checks:

- Contract compatibility tests.
- Fixture replay determinism checks.
- Schema backward compatibility checks.
- Minimal smoke tests.

Cross-repo nightly integration:

1. Start LOS.
2. Run SIM mini-pack against LOS adapter.
3. Run UW mini-manifest.
4. Publish joined compatibility report with drift flags.

Promotion guardrails:

- Gate pass rate threshold.
- Minimum run count threshold.
- Replay variance threshold.
- Budget and latency threshold.

## 10. Open-Source and Repo Separation Strategy

1. Keep core logic in each repo independently releasable.
2. Share only contracts, fixtures, and generated clients.
3. Keep sensitive connectors/data loaders modular and separately controlled.
4. Publish synthetic fixtures/manifests; avoid publishing private borrower artifacts.
5. Maintain a compatibility matrix: `LOS x SIM x UW` supported versions.

## 11. Governance and Ownership

Define accountable owner for:

- Each contract schema.
- Each phase gate.
- Champion/challenger promotion policy.
- Incident management for integration failures.

Establish monthly architecture review and weekly integration triage.

## 12. Success Metrics

Program-level:

- Time from disagreement -> benchmark/test insertion.
- Champion promotion cadence without regression.
- Reproducibility rate across reruns.
- Cost per validated improvement.

Operational:

- LOS throughput/task completion improvements in sim.
- UW quality improvements at fixed approval rates.
- Lower unresolved drift incidents between repos.

## 13. Key Risks and Concerns Assessment

Overall posture: medium-high until contracts, CI coverage, and ledger strategy are hardened.

| Risk / Concern | Evidence | Likelihood | Impact | Concern | Mitigation |
|---|---|---:|---:|---|---|
| Canonical ledger mismatch vs design | Design docs require TigerBeetle-backed canonical truth; runtime integration not present yet | High | High | Undermines ledger-as-physics premise and reconciliation guarantees | Decide canonical ledger implementation now; validate with reconciliation suites |
| API contract drift | LOS runtime routes exceed published OpenAPI | High | High | Integrations can silently diverge and fail | Generate/validate OpenAPI from route tests; block merges on drift |
| Dual simulation ownership ambiguity | LOS has `packages/simulation`; SIM is also orchestration owner | High | High | Duplicate logic and roadmap conflict | Set clear ownership: SIM for benchmark orchestration, LOS simulation for internal capability checks only |
| Missing CI in SIM/UW | SIM and UW currently lack formal GitHub workflows | High | High | Regressions ship without guardrails | Add baseline CI immediately in both repos |
| Semantic schema/unit drift | Known term conventions differ (e.g., APR scale concerns) | Medium | High | Silent scoring/adjudication errors | Add strict unit validation + golden fixtures for terms semantics |
| Evaluation signal contamination | UW ground truth includes analyst-adjusted values | High | Medium | Optimizes against noisy targets | Split scoring into raw extraction and adjustment replication objectives |
| Security and tenant model maturity | Header-based actor/tenant patterns are easy to misuse | Medium | High | Audit and multi-tenant integrity risk | Add stronger authN/authZ hardening before production-grade loop |
| Eventing/outbox immaturity | Limited async/event backbone across repos | Medium | High | Brittle loop and delayed feedback | Introduce versioned event envelope + outbox/webhook path |
| Cost and latency blowouts | Multiple LLM-heavy stages across LOS/SIM/UW | High | Medium | Slower iteration and budget instability | Budget caps, lite manifests, and per-run cost SLOs |
| Nondeterminism from model/provider drift | Provider models and behavior evolve rapidly | High | Medium | Unstable promotion decisions | Model pinning, replay archives, and variance thresholds |
| Feedback-loop poisoning/noise | Auto-mined disagreements can be low-signal | Medium | Medium | Roadmap can chase artifacts | Severity thresholds + human triage rubric |
| Version fragmentation | Independent release cycles across three repos | Medium | High | Cross-repo breakage and stalled adoption | Maintain explicit compatibility matrix and semver discipline |
| OSS packaging ambiguity | Public/private boundaries not fully codified | Medium | Medium | Legal/release friction | Add licensing, contribution policy, data publication boundaries |
| Scope creep to objective 3 too early | Design priorities explicitly favor objectives 1+2 first | High | Medium | Diluted execution focus | Enforce objective-3 entry gate based on objective-1/2 KPIs |
| Governance ambiguity | Cross-repo changes require aligned decisions | Medium | High | Slow decisions and inconsistent architecture | Assign single accountable owner per contract and phase gate |

## 14. Hard Gates Before Full Unification

1. Contract Gate: all repos pass shared contract fixtures in CI.
2. Ledger Gate: canonical accounting strategy finalized and validated.
3. Repro Gate: same manifest replay within agreed variance bands.
4. Safety Gate: auth/tenant/audit controls accepted.
5. Economics Gate: budget and latency SLO enforcement active.

## 15. Immediate Next Actions

1. Approve the ownership matrix and contract list.
2. Stand up shared schema fixtures and CI checks.
3. Start Phase 2 with LOS-in-SIM shadow mode on a small fixed fraud+analyst pack.
4. Publish first compatibility report (`LOS x SIM x UW`) and iterate weekly.

