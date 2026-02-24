# Unification Plan V3: Ready To Implement (LOS + SIM + UW Bench)

Date: February 24, 2026
Status: In progress. SIM-to-LOS thin-agent loop is live; full-agent cutover and parity debt are still open.

## 0) What Changed In V3

This version explicitly resolves the six reviewer findings and adds measurable implementation gates.

Resolved in V3:

1. Workspace execution blocker is now a hard preflight gate with repair commands.
2. API parity is now bidirectional and full-surface (runtime-only and spec-only drift).
3. APR unit normalization is now a dedicated contract gate with fixtures.
4. Dataset manifest is now a Phase 0 hard exit gate.
5. SIM and UW CI are now specified with concrete workflows, commands, seed policy, and tolerances.
6. Hard gates now have numeric thresholds and named owners.

## 0.1 Reality Update (February 24, 2026)

This plan has been corrected against current repo state:

1. API parity is not yet zero-drift (`runtime_only=56`, `spec_only=2`, `method_mismatch=1`).
2. Release parity gate is now configured as a no-regression ratchet using `integration/parity/parity_baseline.json`.
3. `/v1/deals/{dealId}/evaluate` currently runs deterministic thin-agent rules; OpenAPI now reflects this and defaults `mode=rules_only`.
4. `packages/agent/src/agent.ts` still has placeholder service-interface sections and requires production wiring work beyond "just add LLM client".
5. SIM already supports LOS execution mode (`--underwriting-backend los`, plus `--los` alias) through `loanville/los_adapter.py`.

## 1) Objectives And Scope

Priority objectives:

1. Validate LOS operational value and workflow throughput.
2. Evaluate underwriting quality with reproducible benchmark runs.
3. Close feedback loop from disagreements to roadmap and benchmarks.

Out of scope until objective-1/2 gates are stable:

- Full objective-3 shrewd-operator gameplay.
- Multi-party negotiation-heavy mechanics.

## 2) Repo Boundaries (Unchanged Principle)

| Repo | Primary Responsibility | Must Not Own |
|---|---|---|
| LOS (`open-los`) | Product lifecycle execution, accounting operations, monitoring, audit/event contracts, production decision contract | Benchmark promotion logic, Elo/champion governance |
| SIM (`loanville2`) | Case orchestration, adjudication/scoring, champion/challenger, disagreement mining, evaluation run contract | Product system-of-record state |
| UW Bench (`rl-benchmarks`) | Extraction/evidence benchmarking and trace outputs | LOS execution logic |

Rules:

- Integration is artifact/API-based only.
- No cross-repo source imports.

## 3) Contract Model (Two-Layer, Explicit)

1. LOS-owned production output contract: `los-underwrite-result.v1.json`
2. SIM-owned evaluation contract: `underwriting-run.v1.json`
3. SIM-owned scenario contract: `case-pack.v1.json`
4. LOS-owned lifecycle event contract: `los-event-envelope.v1.json`
5. UW-owned extraction output contract: `uw-extraction-result.v1.json`
6. SIM-owned disagreement export contract: `feedback-candidate.v1.json`

Semantics boundary:

- LOS contract contains production decision data, trace, cost, and references.
- SIM contract may include benchmark-only fields (`labels.*`, `scores.*`) and must not be forced into LOS product API.

## 4) Phase -1: Workspace And Repo Readiness Gate (New, Blocking)

This gate is mandatory before any implementation work.

Current blocker in this workspace:

- `/Users/mattbook-air/unify/open-los/.git/HEAD` points to `refs/heads/.invalid`.
- `git rev-parse --verify HEAD` fails.

### 4.1 Preflight Checks

Run:

```bash
git -C /Users/mattbook-air/unify/open-los rev-parse --verify HEAD
git -C /Users/mattbook-air/unify/open-los status
ls -la /Users/mattbook-air/unify/open-los
```

### 4.2 Repair Procedure (Preferred)

```bash
git -C /Users/mattbook-air/unify/open-los fetch origin --prune
git -C /Users/mattbook-air/unify/open-los checkout -B main origin/main
git -C /Users/mattbook-air/unify/open-los rev-parse --verify HEAD
git -C /Users/mattbook-air/unify/open-los status
```

### 4.3 Repair Procedure (Fallback)

If checkout cannot be repaired in-place:

```bash
mv /Users/mattbook-air/unify/open-los /Users/mattbook-air/unify/open-los.broken.$(date +%Y%m%d%H%M%S)
git clone https://github.com/seadotdev/open-los.git /Users/mattbook-air/unify/open-los
```

### 4.4 Exit Criteria (Owner: Integration Lead)

All must be true:

1. `rev-parse --verify HEAD` succeeds for all three repos.
2. Working trees contain tracked files (not `.git` only).
3. Default branches pinned in `integration/repo-lock.json` with commit SHAs.

## 5) API Parity Policy: Full-Surface, Bidirectional (Updated)

### 5.1 Problem Statement

Drift can be:

1. Runtime-only route (implemented, undocumented).
2. Spec-only route (documented, unimplemented).
3. Method/path mismatch (same path but wrong verbs).

Any of these can break integration.

### 5.2 Required Parity Artifacts

1. `integration/parity/runtime_routes.json`
2. `integration/parity/openapi_routes.json`
3. `integration/parity/parity_report.json`
4. `integration/parity/parity_allowlist.yaml` (temporary exceptions only)

### 5.3 Required Checks

Generate runtime inventory from LOS route files.
Generate OpenAPI inventory from `openapi/v1.yaml`.
Diff both directions.

Report fields:

- `runtime_only_count`
- `spec_only_count`
- `method_mismatch_count`
- `allowlisted_count`

### 5.4 Enforceable Gate (Owner: LOS API Owner)

For release branch (current ratchet policy):

1. `unexpected_count = 0` versus `integration/parity/parity_baseline.json`
2. Allowlist must be empty in release mode.

Target end state (explicitly not yet met):

1. `runtime_only_count = 0`
2. `spec_only_count = 0`
3. `method_mismatch_count = 0`

For temporary development windows only:

- Max allowlisted exceptions: `<= 3`
- Each allowlist entry must include: owner, reason, created_at, expires_at.
- Maximum allowlist age: `14 days`.

## 6) APR Normalization Contract Gate (New, Explicit)

Known risk:

- LOS/SIM paths can represent APR as decimal (`0.095`) or percent (`9.5`).

### 6.1 Required Rule

Canonical representation in contracts:

- `underwriting-run.decision.terms.apr` uses decimal.

Adapter conversion rules:

- If source APR `<= 1.0`, treat as decimal.
- If source APR `> 1.0`, treat as percent and normalize to decimal.

### 6.2 Fixture Matrix (Must Pass 100%)

Required fixtures:

1. Input APR `0.095` -> normalized decimal `0.095`
2. Input APR `9.5` -> normalized decimal `0.095`
3. Input APR `0.0` with decline action -> accepted
4. Input APR `55` -> gate failure (sanity limit breach)
5. Round-trip decimal->termsheet->run keeps semantic equivalence
6. Mixed provider payloads still normalize identically

### 6.3 Gate Definition (Owner: SIM Tech Lead)

- APR fixture pass rate: `100%`
- No silent fallback accepted for unknown APR scale
- Any APR-scale parse ambiguity is a hard failure in CI

## 7) Dataset Manifest Baseline Gate (Promoted To Hard Exit Gate)

### 7.1 Required Artifacts

1. `integration/manifests/dataset-baseline-v1.json`
2. `integration/manifests/dataset-baseline-v1.md`

Minimum baseline fields:

- Repo + dataset source
- count of borrowers/cases/doc-period pairs
- extraction field list
- mix definitions
- generation/query command
- seed value
- timestamp
- owner

### 7.2 Phase 0 Exit Criteria (Owner: Data Steward)

All must be true:

1. Baseline manifest is committed.
2. Counts are generated by reproducible command(s), not prose.
3. Cross-repo comparisons and promotions are blocked if baseline is missing.

## 8) CI Bootstrap Plan For SIM And UW (Concrete, Week 1)

Both repos currently require greenfield CI setup.

### 8.1 SIM CI (`loanville2/.github/workflows/ci.yml`)

Trigger:

- pull_request
- push to default branch

Matrix:

- OS: `ubuntu-latest`
- Python: `3.11`

Required jobs and commands:

1. Static/import sanity

```bash
pip install -r requirements.txt
python -m compileall loanville
```

2. Deterministic mock simulation smoke

```bash
python -m loanville --mock --mix fraud --data-mode lite
```

3. Underwriting mode regression smoke (mock/offline)

```bash
python test_underwriting_modes.py
```

4. Contract fixtures validation (new script to add)

```bash
python scripts/validate_contract_fixtures.py
```

Artifacts:

- `runs/**`
- simulation logs

Policy:

- No live API calls in PR CI.
- PR CI cost target: `$0`.

### 8.2 UW CI (`rl-benchmarks/.github/workflows/ci.yml`)

Trigger:

- pull_request
- push to default branch

Matrix:

- OS: `ubuntu-latest`
- Python: `3.11`

Required jobs and commands:

1. Static/import sanity

```bash
pip install -r requirements.txt
python -m compileall harness
python -m compileall benchmark_runner.py
```

2. Offline harness smoke (new script to add; no API keys)

```bash
python smoke_test_offline.py
```

3. Contract fixtures validation (new script to add)

```bash
python scripts/validate_contract_fixtures.py
```

Policy:

- `smoke_test.py` (live OpenRouter) runs only in nightly with secrets.
- PR CI must be secret-free and deterministic.

### 8.3 Cross-Repo Integration CI (`unify/.github/workflows/integration.yml`)

Matrix:

- OS: `ubuntu-latest`
- Node: `20.x` (LOS)
- Python: `3.11` (SIM + UW)

PR job (offline/mocked):

1. Verify Phase -1 readiness checks.
2. Start LOS.
3. Run SIM LOS adapter on mini-pack in mock/offline mode.
4. Run UW offline mini-manifest checks.
5. Run parity + APR fixture gates.

Nightly job (live, budget-controlled):

1. Fixed mini manifest.
2. Fixed model list.
3. Budget cap enforcement.

### 8.4 Seed And Replay Policy (New, Explicit)

Seed policy:

1. UW Bench uses explicit seed pinning in CI/nightly (`--seed 42` unless changed via ADR).
2. SIM PR CI uses deterministic mock mode and fixed borrower mixes as temporary policy.
3. SIM Week-1 deliverable: add `--seed` support for deterministic borrower ordering/sampling in LOS-adapter paths.

Replay policy:

1. PR CI replay check: rerun same job twice and compare run hash/artifacts.
2. Mock mode tolerance: exact equality required.
3. Live nightly tolerance: action agreement and score-drift thresholds from Gate Table apply.

## 9) Hard Gates With Numeric Thresholds And Owners

| Gate | Metric | Threshold | Owner |
|---|---|---|---|
| Readiness | Repo HEAD validity | 100% repos valid (`rev-parse` passes) | Integration Lead |
| API parity | Runtime/spec drift | `runtime_only=0`, `spec_only=0`, `method_mismatch=0` (release branch) | LOS API Owner |
| APR normalization | Fixture pass rate | `100%` required | SIM Tech Lead |
| Manifest baseline | Baseline artifact presence + reproducibility | Required before Phase 1 starts | Data Steward |
| Repro (mock) | Run hash identity on fixed inputs | `100%` identical across two reruns | SIM Tech Lead |
| Repro (live nightly) | Action agreement | `>= 97%` action agreement on fixed mini-pack | Benchmark Owner |
| Repro (live nightly) | Aggregate score drift | `<= 2.0` points vs prior nightly baseline | Benchmark Owner |
| Safety | Tenant isolation tests | `100%` pass in conformance suite | LOS Security Owner |
| Safety | Open critical/high security findings for integration code | `0` | LOS Security Owner |
| Economics (PR) | PR CI run cost | `$0` (offline/mock only) | Integration Lead |
| Economics (nightly) | Live benchmark budget | `<= $20/day` | Benchmark Owner |
| Economics (nightly) | p95 LOS evaluate latency on mini-pack | `<= 20s` | LOS API Owner |

## 10) Delivery Plan (Updated)

### Phase 0: Program Setup (Week 1)

Deliverables:

1. ADRs for ownership, contracts, parity, APR policy, security, economics.
2. Dataset baseline manifest.
3. CI bootstrap PRs for SIM and UW.
4. Parity report scaffolding.

Exit criteria:

- Phase -1 gate passed.
- Dataset baseline committed and reproducible.
- CI running in SIM and UW.

### Phase 1: Contract + Parity Hardening (Weeks 1-2)

Deliverables:

1. All v1 contracts published.
2. LOS `/evaluate` endpoint added.
3. Full-surface parity checks active.
4. APR fixture tests active.

Exit criteria:

- API parity gate passes.
- APR normalization gate passes.

### Phase 2: LOS Adapter Cutover In SIM (Weeks 2-3)

Deliverables:

1. `los_adapter.py` and deterministic mapping to `underwriting-run`.
2. Dual-run mode and disagreement diff.
3. Fixed mini-pack integration smoke.

Exit criteria:

- Mock repro gate passes.
- No critical regressions on fraud/easy mini-pack.

### Phase 3: UW Bridge + Score Enrichment (Weeks 2-4, parallel)

Deliverables:

1. UW export/import bridge with stable IDs.
2. SIM scorecard enrichment with UW artifacts.
3. Cross-repo artifact linkage report.

Exit criteria:

- End-to-end linkage completeness on mini-pack: `100%`.

### Phase 4: Feedback Automation (Weeks 5-6)

Deliverables:

1. Disagreement export pipeline.
2. Benchmark/conformance candidate generation.
3. Triage workflow with severity thresholds.

Exit criteria:

- High-severity disagreement turnaround path fully automated.

### Phase 5: Season Mode (Weeks 7-10)

Deliverables:

1. Persistent tenant strategy.
2. Carry-forward portfolio state.
3. Weekly reporting and observability.

Exit criteria:

- Stable multi-week runs under repro and economics gates.

## 11) Ready-To-Implement Definition

This plan is considered ready to implement when all of the following are true:

1. Phase -1 readiness gate is green.
2. Phase 0 exit criteria are complete.
3. Gate owners are assigned by name.
4. CI workflows are merged in all three repos (or in-flight with approved blocking PRs).

## 12) Appendix: Finding-To-Fix Traceability

| Reviewer Finding | V3 Resolution |
|---|---|
| 1. Workspace execution blocker | Phase -1 gate with explicit repair commands and exit criteria |
| 2. Incomplete API parity treatment | Full-surface bidirectional parity policy + allowlist controls |
| 3. APR unit conversion risk | Dedicated APR normalization contract gate + fixture matrix |
| 4. Dataset manifest not gated | Dataset baseline now required Phase 0 hard exit gate |
| 5. CI underspecified for SIM/UW | Concrete workflow matrix, commands, policy, and artifacts |
| 6. Hard gates subjective | Numeric thresholds + explicit owners for each gate |
