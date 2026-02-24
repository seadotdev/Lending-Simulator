# Unification Plan V2 Final: LOS + SIM + UW Bench

Date: February 23, 2026
Status: Final draft for expert review

## 1) Executive Summary

This plan unifies three repositories into a closed feedback loop while preserving repository independence and open-source optionality:

- LOS (`seadotdev/open-los`): execution system of record.
- SIM (`seadotdev/loanville2`): simulation, adjudication, scoring, champion/challenger.
- UW Bench (`seadotdev/rl-benchmarks`): document-intelligence evaluation and extraction benchmarking.

The loop is:

1. SIM generates cases and policies.
2. LOS executes underwriting/lifecycle operations and emits traceable outputs.
3. UW Bench evaluates extraction/evidence quality for relevant cases.
4. SIM aggregates outcomes, mines disagreements, and promotes/demotes policies.
5. Severe disagreements generate new benchmark and conformance cases.

This V2 resolves issues identified in prior drafts:

- Corrects endpoint mismatches (e.g., audit endpoint naming).
- Clarifies two-layer contract ownership boundaries (LOS production output vs SIM evaluation run).
- Removes implicit cross-repo code coupling.
- Adds missing CI requirements for SIM and UW Bench.
- Treats LOS OpenAPI/runtime drift as a hard blocker before broad integration.

## 2) Scope and Design Alignment

Primary objectives (in priority order):

1. Validate LOS operational value and workflow throughput.
2. Evaluate underwriting quality with reproducible benchmark runs.
3. Enable evidence-based roadmap iteration via disagreement-driven feedback.

Design alignment sources:

- `loanville2/docs/lending-sim-design.md`
- `loanville2/docs/los-cutover.md`

Out of scope for initial unification:

- Full objective-3 gameplay (“shrewd operator behavior”) before objective-1/2 gates are stable.
- Broad productization of multi-party negotiation mechanics before LOS execution loop is proven.

## 3) Repo Boundaries and Responsibilities

| Repo | Primary Responsibility | Must Not Own |
|---|---|---|
| LOS | Deal/facility/loan lifecycle, accounting operations, monitoring, audit artifacts, API contracts | Benchmark policy governance and Elo/champion logic |
| SIM | Case orchestration, run scoring, adjudication economics, champion/challenger, disagreement mining, evaluation run contract | Production LOS domain state ownership |
| UW Bench | Extraction/evidence quality benchmarking, trace artifacts, benchmark manifests | LOS lifecycle execution logic |

Operational rule:

- Repos integrate through versioned contracts and API calls only.
- No direct source imports from one repo into another.

## 4) Current-State Facts and Concerns (Validated)

1. SIM has an explicit cutover seam via `UnderwritingRun` in `run_schema.py`.
2. LOS runtime routes are richer than what is currently represented in `openapi/v1.yaml`.
3. LOS does not currently expose `/v1/deals/{dealId}/evaluate`; it must be added for clean integration.
4. LOS audit route is `GET /v1/deals/:dealId/audit` (not `/audit-events`).
5. SIM and UW Bench currently need explicit CI pipeline additions.
6. Dataset counts differ across docs/code paths; unification must standardize via explicit manifests, not prose assumptions.

## 5) Architecture and Data Flow

Target flow:

1. SIM emits `case-pack` and invokes LOS adapter.
2. LOS executes underwriting and returns `los-underwrite-result` (production decision artifact).
3. SIM maps `los-underwrite-result` into SIM-owned `underwriting-run` and performs adjudication/lifecycle scoring.
4. UW Bench receives linked tasks/manifests and returns `uw-extraction-result`.
5. SIM merges LOS + UW signals into scorecards and disagreement exports.

High-level interface shape:

- Control plane: HTTP API between SIM and LOS.
- Data plane: versioned JSON artifacts (`case-pack`, `los-underwrite-result`, `underwriting-run`, `uw-extraction-result`, `feedback-candidate`).
- Observability plane: trace/event artifacts and run log correlation IDs.

## 6) Contract Strategy and Ownership

Use contract-first integration with explicit owners:

1. `los-underwrite-result.v1.json`
- Owner: LOS (production decision/output contract).
- Producers: LOS `/evaluate` endpoint.
- Consumers: SIM LOS adapter, LOS-native consumers.
- Notes: Contains decision, terms, rationale, trace, cost/latency, and references needed to audit LOS behavior. Does not contain benchmark labels/scores.

2. `underwriting-run.v1.json`
- Owner: SIM (benchmark/evaluation semantic contract).
- Producers: SIM legacy adapter, SIM LOS adapter (mapping from `los-underwrite-result`), other evaluators.
- Consumers: SIM scoring, champion/challenger, replay tools.
- Notes: Includes evaluation-only fields such as labels and scores; not a LOS production API contract.

3. `case-pack.v1.json`
- Owner: SIM.
- Producers: SIM case generator.
- Consumers: LOS adapter, UW Bench bridge.

4. `los-event-envelope.v1.json`
- Owner: LOS.
- Producers: LOS.
- Consumers: SIM observability/reconciliation pipeline.

5. `uw-extraction-result.v1.json`
- Owner: UW Bench.
- Producers: UW Bench runs.
- Consumers: SIM scorecard enrichment.

6. `feedback-candidate.v1.json`
- Owner: SIM.
- Producers: SIM disagreement miner.
- Consumers: LOS conformance backlog and UW benchmark backlog.

Versioning policy:

- Semantic versioning per contract.
- Backward-compatible additions only in minor versions.
- Breaking changes require coordinated major bump and compatibility matrix update.

## 7) Integration Endpoints (Corrected Baseline)

Minimum LOS API sequence per borrower/lender evaluation:

1. `POST /v1/entities`
2. `POST /v1/deals`
3. `POST /v1/deals/{dealId}/documents`
4. `POST /v1/deals/{dealId}/spread`
5. `POST /v1/deals/{dealId}/stage-transitions`
6. `POST /v1/deals/{dealId}/evaluate` (new endpoint to add)
7. `GET /v1/deals/{dealId}`
8. `GET /v1/deals/{dealId}/audit` (existing endpoint)

Required LOS additions:

- Add `/evaluate` endpoint contract and implementation path.
- Include trace and cost metadata needed by SIM run scoring.
- Ensure response validates against `los-underwrite-result.v1`.

## 8) Phased Delivery Plan

### Phase 0: Program Setup and ADRs (Week 1)

Deliverables:

- Architecture decision records (ownership, contracts, ledger strategy, security posture).
- Compatibility matrix template (`LOS x SIM x UW`).
- Data publication/open-source boundary policy.
- Baseline manifest inventory with canonical counts.

Exit criteria:

- All ADRs signed.
- Scope and repo ownership approved.

### Phase 1: Contract Hardening + API Drift Closure (Weeks 1-2)

Deliverables:

- Contract schemas and fixtures in all three repos.
- OpenAPI-runtime parity plan in LOS.
- Schema validators in TypeScript and Python.
- Contract conformance tests.

Exit criteria:

- LOS OpenAPI/runtime parity achieved for integration endpoints.
- All repos validate shared fixtures in CI.

### Phase 2: LOS Adapter in SIM (Weeks 2-3)

Deliverables:

- `los_adapter.py` in SIM.
- Dual-run mode: legacy vs LOS-backed.
- Correlated `UnderwritingRun` emission from LOS adapter via deterministic mapping from `los-underwrite-result`.
- Decision conversion path for existing adjudication pipeline.

Exit criteria:

- Gate pass rate threshold achieved.
- Replay determinism within agreed variance.
- No critical regression against baseline mixes.

### Phase 3: UW Bench Bridge (Weeks 2-4, parallel)

Deliverables:

- UW export -> SIM borrower/case conversion pipeline.
- SIM->UW task/manifest bridge.
- UW result import into SIM with stable IDs.

Exit criteria:

- End-to-end linked run IDs and case IDs.
- Reproducible UW enrichment artifacts for sampled scenarios.

### Phase 4: Unified Scorecard Enrichment (Week 4)

Deliverables:

- SIM scorecard includes extraction/evidence dimensions using UW artifacts.
- Weighting strategy finalized and documented.
- Disagreement severity rubric updated.

Exit criteria:

- Promotion decisions include LOS + UW signals.
- Scorecard reproducibility validated across reruns.

### Phase 5: Feedback Loop Automation (Weeks 5-6)

Deliverables:

- Disagreement export pipeline to benchmark candidates.
- LOS conformance test suggestion generation.
- UW task suggestion generation.
- Triage workflow with severity thresholds.

Exit criteria:

- Automated loop operates without manual reshaping.
- High-severity issues route to actionable backlog artifacts.

### Phase 6: Season Mode and Persistent State (Weeks 7-10)

Deliverables:

- Persistent tenant strategy for multi-week simulation.
- Portfolio carry-forward support.
- Weekly reporting and operational telemetry.

Exit criteria:

- Stable multi-week runs with reproducible scoring deltas.

## 9) CI/CD and Release Governance

Per-repo mandatory CI:

- Contract fixture validation.
- Schema compatibility checks.
- Determinism/replay smoke checks.
- Integration smoke test hooks.

Cross-repo integration CI:

1. Start LOS.
2. Run SIM small mix through LOS adapter.
3. Run UW mini-manifest.
4. Assert artifact linkage and publish compatibility report.

Release controls:

- No major contract bump without compatibility matrix update.
- Promotion pipelines blocked on failed conformance/integration checks.

## 10) Open-Source and Packaging Strategy

Rules:

1. Keep each repo releasable independently.
2. Publish contract specs and synthetic fixtures openly.
3. Keep sensitive connectors/data access modules separable.
4. Avoid private-data assumptions in public contracts.
5. Track license and contribution policy per repo explicitly.

## 11) Risk Register (V2)

| Risk | Likelihood | Impact | Why It Matters | Mitigation |
|---|---:|---:|---|---|
| OpenAPI/runtime drift in LOS | High | High | Integration breaks or behaves inconsistently | Treat parity as Phase-1 hard gate |
| Missing `/evaluate` endpoint | High | High | SIM cannot cleanly cut over to LOS | Add endpoint early with contract tests |
| Ledger strategy ambiguity | High | High | Invalidates canonical truth/reconciliation claims | ADR and reconciliation test suite before broader rollout |
| Cross-repo coupling creep | Medium | High | Breaks independent OSS release model | Artifact/API-only integration policy |
| Missing CI in SIM/UW | High | High | Regressions pass unnoticed | Add baseline CI immediately |
| Dataset baseline inconsistency | Medium | Medium | Invalid comparison and benchmarking claims | Manifest-backed source-of-truth metrics |
| Nondeterminism from provider drift | High | Medium | Unstable promotion decisions | Model pinning + replay variance thresholds |
| Cost/latency blowout | High | Medium | Slows iteration and increases spend | Budget caps, lite manifests, tiered test modes |
| Security/tenant weaknesses | Medium | High | Audit and isolation risk in realistic runs | Strengthen authN/authZ and tenant enforcement gates |
| Feedback-loop noise | Medium | Medium | Backlog polluted by low-signal disagreements | Severity thresholds + human triage rules |
| Contract semantic leakage (prod vs eval) | Medium | High | Benchmark-only fields can pollute LOS API or force tight coupling | Enforce two-layer contracts: LOS `los-underwrite-result`, SIM `underwriting-run` |

## 12) Hard Gates Before Full Unification

1. Contract Gate: all shared fixtures pass in all repos.
2. API Gate: LOS OpenAPI equals runtime integration surface.
3. Ledger Gate: canonical accounting strategy tested and reconciled.
4. Repro Gate: replay stability inside approved variance.
5. Safety Gate: auth/tenant/audit controls accepted.
6. Economics Gate: cost and latency SLOs enforced.

## 13) Implementation Backlog (Immediate)

Week-1 actionable items:

1. Create ADR set for ownership, contracts, and ledger strategy.
2. Add `los-underwrite-result.v1`, `underwriting-run.v1`, `case-pack.v1`, `uw-extraction-result.v1`, `feedback-candidate.v1`, `los-event-envelope.v1`.
3. Add LOS `/evaluate` endpoint stub with schema validation.
4. Add SIM `los_adapter.py` skeleton and dual-run flag plumbing.
5. Add CI workflows in SIM and UW Bench.
6. Add compatibility report generator in integration workspace.

Week-2 actionable items:

1. Wire LOS adapter request sequence end-to-end.
2. Validate corrected audit route ingestion (`/v1/deals/{dealId}/audit`).
3. Run fraud/easy smoke mixes through LOS-backed path.
4. Produce first disagreement export and triage report.

## 14) Acceptance Criteria for Expert Review Sign-Off

The plan is accepted when:

1. Contract ownership and boundaries are unambiguous.
2. Endpoint and route references match real implementation.
3. CI and release gates are explicit and enforceable.
4. Risk register is concrete and tied to mitigations and gates.
5. Open-source separation strategy is technically workable.

---

## Appendix A: Corrected Items from Prior Draft

1. Audit route corrected to `/v1/deals/{dealId}/audit`.
2. Explicitly states `/v1/deals/{dealId}/evaluate` must be added.
3. Introduces two-layer contract model:
   LOS owns `los-underwrite-result` (production output), SIM owns `underwriting-run` (evaluation artifact).
4. Removes direct cross-repo code reuse assumptions.
5. Replaces prose-based dataset assumptions with manifest-backed baseline requirement.
6. Treats LOS OpenAPI/runtime drift as a hard blocker.
