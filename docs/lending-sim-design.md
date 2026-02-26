# LLM-Enabled B2B Lending Simulation + LOS Roadmap

> Detailed doc synthesized from initial design conversations; raw user questions in appendix.

---

## 1) Purpose

Build a ledger-correct, reproducible simulation that serves three goals:

1. **Validate the LOS** you're building for agentic business lending — especially workflow throughput, correctness, and token efficiency when managing many concurrent deals and negotiations.
2. **Evaluate LLM underwriting ability** on realistic financial statements + bank statements, comparing LOS-tools vs raw-file holdouts, and scoring outcomes against simulated ground truth.
3. **(Later) Evaluate shrewdness of models as business operators** — competing, borrowing, surviving shocks, and potentially gaming incentives.

The simulation is not "game-first." It's a benchmark harness that generates credible underwriting work and operational load, and that drives an evidence-based LOS roadmap.

---

## 2) Core design principles

### 2.1 Ledger is the physics engine; LOS is perception

- TigerBeetle is the canonical source of truth for money movement: disbursements, repayments, interest accrual, fees, delinquency tracking, charge-offs, recoveries.
- Lenders/LOS instances may have their own internal ledgers/views that can drift or be wrong (intentionally), but reconciliation against TigerBeetle is always possible and auditable.

### 2.2 Benchmarks are snapshots/runs first

- Primary scoring uses frozen scenario packs with fixed seeds and replay.
- An "always-on Smallville" universe is optional later as a marketing/demo layer, not as the scoring substrate.

### 2.3 Start clean; add mess and adversaries progressively

- v0 begins with clean, standardized artifacts (P&L, balance sheet, bank statement).
- Complexity (multi-bank accounts, AR/AP timing, document messiness, fraud transforms) is introduced as staged releases.

### 2.4 Capability-gated world complexity (roadmap driver)

You only turn on new world complexity when you ship LOS/CLI capabilities that make it tractable:

- **Summarize** (compact, token-efficient facts)
- **Drill down** (evidence pointers: transactions, doc anchors, IDs)
- **Diff** (before/after, reconcile, compare)
- **Audit** ("book is in order" pack)

This turns the sim into a roadmap engine: it manufactures the operational pain only when you also ship the feature that removes it.

### 2.5 Optimize objectives 1+2 first

Business "shrewdness" gameplay (objective 3) is postponed until:

- ledger correctness is proven
- underwriting evaluation is stable
- LOS ops value is demonstrated

---

## 3) Simulation architecture

### 3.1 Three-layer separation

#### A) World simulation (economy + events)

- Produces business operating outcomes: revenue, COGS, opex, cash conversion.
- Injects macro events and sector conditions (baseline vs stress scenarios).
- Generates realistic flows that roll up into financial artifacts.

#### B) Financial system (ledger + loan accounting)

- TigerBeetle-backed transfers enforce correctness.
- Deterministic posting rules map "world events" → ledger movements:
  - customer receipts, vendor payments, payroll, taxes
  - loan disbursement and scheduled payments
  - interest accrual and payment waterfall (fees → interest → principal)
  - delinquency/DPD and collections actions

#### C) Agent harness (LLM + tools + evaluation)

- Runs multiple agent types:
  - **LOS-agent** uses your LOS/CLI tool surface.
  - **Raw-file holdout** sees only files (CSV/PDF) + sandbox tools (bash allowed).
  - **Baselines** (rules/heuristics) for underwriting and servicing.
- Logs outcomes: decisions, tokens, tool calls, time, task success, errors, audit exceptions.

### 3.2 "Catan-like" business world (minimal but expandable)

- A procedurally seeded regional economy (e.g., "Open-Lend Midlands") with sectors and markets.
- Businesses are agents with:
  - latent "management quality" / behavioral quirks (kept simple initially)
  - balance sheet and bank feed derived from truth
- Early version: template-driven operations that produce plausible statements; later can grow into richer economics.

### 3.3 Holdouts and information asymmetry

- A fixed % of borrowers refuse LOS adoption (raw doc flows, CSV/email style), forcing lenders to operate via fallback tools.
- "Hidden GDP" / macro regime can be invisible to incumbents (inference required) while visible to "god mode" or new entrants (optional design lever).

---

## 4) Tasks and evaluation

### 4.1 Objective 1: LOS validation (ops + ergonomics)

**Focus:** "How good is the LOS at helping a lender keep track of multiple loan applications, negotiations, and next steps?"

**Ops task stream:**

- New application arrivals
- Doc uploads, missing items
- Borrower Q&A
- Negotiation changes (counteroffers, conditions)
- Payment failure and servicing tasks post-close

**Score components:**

- **Throughput:** number of deals advanced correctly per unit time
- **Workflow correctness:** no dropped deals, consistent stages, accurate checklists
- **Token/tool efficiency:** low chatter, compact summaries, bounded evidence
- **Auditability:** reproducible state transitions, traceability, clean reconciliation

### 4.2 Objective 2: Underwriting evaluation

Compare models across conditions:

- **LOS condition:** tool access + aggregated metrics → drill-down evidence
- **Holdout condition:** raw files + sandbox bash, no LOS computations

**Artifacts (v0):**

- P&L (12 months monthly)
- Balance sheet (most recent)
- Bank statement (6 months, one account)

**Outcomes:**

- Defaults/delinquency and recoveries under baseline and macro stress
- Underwriting quality measured as:
  - loss rate at fixed approval rate
  - default prediction separability (AUC/PR) if you want probabilistic scoring
  - decision consistency and evidence quality

### 4.3 Objective 3: Shrewd operator behavior (later)

Track model-run businesses on:

- profit/survival/market share
- debt service behavior
- expansion timing and competitive strategies

Initially hardcode business policies; later allow LLM operators.

---

## 5) Macro stress and fraud

### 5.1 Macro stress (must-have early)

Include at least one scenario analogous to a systemic shock ("Covid/GFC-like"):

- demand collapse / margin compression
- liquidity tightening
- worse recoveries / longer workout timelines

**Purpose:** test underwriting robustness and servicing workload.

### 5.2 Fraud packs (essential)

Fraud is treated as a first-class axis:

- Start clean; then introduce fraud via controlled transforms.
- Score detection vs false positives and "time-to-suspicion."

Early fraud archetypes can include:

- statement tampering inconsistencies
- P&L inflation without corresponding cashflows
- expense suppression / timing tricks
- bust-out post-funding behavior

(Expanded using your compiled list over time.)

---

## 6) Multi-tenant and book purchase scenarios

You want to test:

- how the LOS handles multi-tenant situations
- one lender buying another lender's book
- whether coordination is sensible (later)

### 6.1 v0–v1 focus: LOS merge mechanics, not negotiation

Core deliverable: portfolio transfer with auditability:

- seller → buyer data snapshot
- servicing rights cutover rules
- reconcile buyer's imported view against canonical ledger truth
- generate "book is in order" report

### 6.2 v1+ focus: coordination failure modes

Later add lender-to-lender negotiation and coordination scaffolding (commitments, acknowledgements, checklists), informed by multi-agent coordination benchmarks (e.g., CooperBench).

### 6.3 Missing detail to enable direct lender-borrower negotiation

To make negotiation measurable (not just free-form chat), define explicit protocol objects and outcome rules:

- **Negotiation primitives:** offer, counteroffer, accept, reject, withdraw, expire, escalate.
- **Term vector schema:** amount, APR/rate type, tenor, amortization, collateral, guarantor scope, covenants, reporting cadence, fees.
- **Constraint model:** lender hard limits (concentration, policy caps, minimum DSCR) and borrower hard limits (max payment, required liquidity buffer).
- **Evidence linkage:** each proposed term change must include cited evidence (financial metric, bank pattern, covenant rationale).
- **Round mechanics:** bounded turns, response SLA, and timeout/default behavior to avoid infinite bargaining.
- **Commitment semantics:** signed intent vs binding commitment, with explicit revocation/penalty rules.
- **Escalation path:** committee review, risk override, and final adjudicator actions with audit reasons.
- **Scoring dimensions:** close rate, time-to-close, concession efficiency, post-close performance, and negotiation harm (overly coercive or unsupported terms).

Borrowed patterns to use:

- From social deduction benchmarks: bounded rounds + explicit vote/decision transitions.
- From auction/market benchmarks: standardized bid/ask objects and expiry windows.
- From Skirmish-style environments: deterministic state transitions, replay logs, and strict action APIs.

---

## 7) CLI tools as product surface and roadmap lever

### 7.1 CLI design principle

CLI tools should be:

- deterministic, idempotent, replayable
- compact by default (JSON facts), with optional evidence/explanations
- diff/audit native (`--dry-run`, `--diff`, `--evidence`)
- budget controllable (`--top-k`, `--fields`, `--max-bytes`)

These tools simultaneously serve:

- sim harness tools for LLMs
- real LOS ops tooling
- integration contracts

### 7.2 Most valuable lender-side CLI tools (revenue + sim efficiency)

- `los inbox` / `los deal` — pipeline management at scale; SLAs; stuck detection
- `los ingest` + `validate` — doc/bank ingestion and consistency checks
- `los risk snapshot` — token-efficient underwriting "facts view" + evidence pointers
- `los servicing` — DPD, schedule, task lists under stress
- `los portfolio` — cohort reporting + stress reports
- `los transfer` + `los audit` — book purchase import/reconcile/audit pack
- `los consortium publish/query` — anonymized fraud and application signal exchange
- `los borrower graph` — cross-lender entity-link and request-spam detection

### 7.3 Most valuable business-side CLI tools

- `biz pack` — borrower artifact pack generator and validator
- `biz bank reconcile` — bank-to-financial reconciliation, anomaly flags
- Later: `biz forecast`, `biz covenant check`, `biz offers compare`

### 7.4 Capability-gated complexity examples

| World Complexity | Gates On CLI Capability |
|---|---|
| Multi-bank accounts | `los bank consolidate/anomalies` ships |
| AR/AP timing | `los wc snapshot/reconcile` ships |
| Multi-facility | `los exposure/payment apply/covenants` ships |
| Book purchase | `los transfer reconcile` + `audit pack` ships |

---

## 8) Roadmap (sensible sequence)

### Release 0 — Foundation

- TigerBeetle schema + posting rules + replay harness
- Scenario pack format + seeded runner
- Minimal CLI: ingest, deal/inbox, risk snapshot, audit/reconcile

### Release 1 — Underwriting + Ops MVP (Objectives 1+2 begin)

- Clean artifacts; one bank account; template WC; single facility type
- Approve/decline outputs
- Benchmark Pack A: clean baseline + LOS vs holdout

### Release 1.1 — LOS Observability (roadmap engine)

- "Observer" that reports friction: token hotspots, repeated tool calls, stuck stages
- Produces ranked feature candidates and before/after comparisons across LOS versions

### Release 2 — Macro Stress + Servicing

- Add shock pack; collections load; portfolio reporting
- Benchmark Pack B: baseline vs stress

### Release 3 — Fraud Pack v1

- Fraud transforms + provenance
- Benchmark Pack C: fraud detection and workflow response

### Release 4+ — Capability-gated expansions

- Multi-bank accounts ⇄ consolidation/anomaly CLI
- AR/AP timing ⇄ working-cap snapshot/reconcile CLI
- Multi-facility/waterfall ⇄ exposure + payment apply CLI
- Book purchase multi-tenant ⇄ transfer import/reconcile + audit pack
- Coordination/negotiation layer (later)
- Shrewd operator gameplay (last)

### 8.1 Precedent-driven extensions (social benchmarks)

Use social benchmark precedents (for example Among AIs-style runs in web-native deterministic environments) to add a new evaluation axis: social decision quality under uncertainty, not just single-agent credit accuracy.

| Precedent Pattern | Why It Matters | Loanville Adaptation | Gate / Release |
|---|---|---|---|
| Deterministic fixed-timestep environment + standardized actions | Makes runs replayable and auditable while preserving multi-agent dynamics | Add phase-based lender committee loop with explicit actions (`propose`, `challenge`, `request_evidence`, `vote`, `skip`) | Release 2.5 |
| Partial observability ("fog of war") | Prevents trivial omniscient policies and exposes trust calibration failures | Restrict each lender to scoped evidence windows; require explicit evidence fetches to expand context | Release 2.5 |
| Bounded discussion rounds before vote | Pressure-tests persuasion, coordination, and decision discipline | Add capped underwriting committee meetings (e.g., 3 turns) before approve/decline/escalate | Release 2.5 |
| Accuracy and harm tracked separately | Avoids rewarding loud but unsafe agents | Split metrics into decision accuracy vs social harm (wrongful declines, wrongful approvals, scapegoating) | Release 3 |
| Leadership vs bandwagoning signatures | Surfaces stable style differences across models | Track initiative rate, herding rate, vote-switch rate, and skip rate by model and role | Release 3 |
| Role-conditioned behavior ("chameleon" shifts) | Detects strategic behavior changes when incentives flip | Run lender, reviewer, and adversarial borrower roles; measure delta in proactivity and harm by role | Release 3.5 |
| Highlight/replay artifacts | Enables qualitative review and debugging of failures | Auto-export per-episode meeting transcripts, event timeline, and key replay clips | Release 3.5 |

### Release 2.5 — Committee Protocol Pack (new)

- Add a structured committee phase to underwriting episodes: discuss -> vote -> action.
- Benchmark Pack D: committee baseline vs independent underwriter baseline.
- Publish role-scoped action logs and deterministic replay for each episode.

### Release 3.5 — Adversarial Social Robustness Pack (new)

- Add adversarial borrower and noisy-witness behaviors to induce disagreement and deception pressure.
- Score persuasion quality with safety constraints: evidence-backed influence > confidence-only influence.
- Add guardrail ablations: evidence thresholding, veto rights, and minimum quorum before adverse actions.

### Release 4.5 — Consortium Collaboration Pack (new)

- Add lender collaboration channels:
  - anonymized suspected-fraud signal sharing
  - anonymized application fingerprint sharing to detect multi-lender spam
- Add a privacy-preserving signal schema with confidence, evidence class, TTL, and revocation.
- Benchmark Pack E: isolated underwriting vs consortium-enabled underwriting (measure fraud catch uplift, false-positive spillover, and latency impact).
- Add anti-collusion and anti-poisoning controls (source reputation, quorum thresholds, and delayed influence weighting).

### 8.2 Scaling methodology for collaboration at larger network size

Current methodology is small-N and episode-centric; consortium behavior requires network-scale evaluation.

| Current Method | Scaling Risk | Scale Upgrade |
|---|---|---|
| Triplet lender matches | Too few counterparties to test network effects | Move to many-lender cohorts (e.g., 25, 50, 100+) with partitioned communication graphs |
| Per-episode local state | Misses repeated spam/fraud campaigns | Persist borrower/application identity graph across episodes and seasons |
| Raw transcript-heavy logging | Costly and hard to aggregate at high volume | Event-sourced compact logs + sampled transcript retention |
| Single-run metrics | Fails to capture propagation delays | Add time-series metrics: alert precision@time, spread velocity, and downstream decision impact |
| Uniform trust across agents | Vulnerable to poisoning | Add source trust scores and adversarial signal injection tests |

Scale targets to stage:

- Stage A: 25 lenders, 10k applications, 1 season.
- Stage B: 100 lenders, 100k applications, repeated borrower identities.
- Stage C: 250+ lenders, 1M application events, mixed honest/adversarial signal providers.

### Release 5 — Policy Code Arena (Skirmish-inspired, new)

- Allow agents to submit executable policy code modules (pricing, fraud triage, negotiation policy) rather than only one-shot text decisions.
- Run policies in a deterministic sandbox with strict limits (CPU/memory/time/network-off) and stable tool APIs.
- Add iterative rounds: policy -> run -> diagnostics -> policy revision (Skirmish-style strategy loop).
- Benchmark Pack F: prompt-only agents vs code-policy agents under equal data and budget constraints.
- Score policy quality across utility, safety, robustness, and operational efficiency (runtime + token spend).

### 8.3 Code-policy detail required before rollout

- **Policy API contract:** exact observation schema, action schema, and allowed state persistence.
- **Sandbox model:** language/runtime choices, dependency policy, deterministic randomness handling.
- **Review and safety gates:** static checks, banned operations, and reproducibility requirements before execution.
- **Version governance:** signed policy artifacts, rollback support, and lineage tracking per run.
- **Fairness constraints:** equal compute budget and equal information access across policy and prompt baselines.
- **Failure handling:** timeouts, runtime exceptions, and fallback behavior to keep tournaments progressing.

---

## 9) Key compromises explicitly accepted

- Prefer reproducible benchmark runs over always-on universe.
- Start with simple underwriting outputs (approve/decline) and one product.
- Keep business world minimal initially; realism comes from financial artifact fidelity.
- Defer negotiation/coordination mechanics; first nail LOS merge + audit.
- Add complexity only when tooling exists to make it cheap and measurable.

---

## Appendix A — Raw user questions and prompts (verbatim)

1. *"https://arxiv.org/abs/2304.03442 — I want a comprehensive write up of all the best open code frameworks in this line of LLM enabled simulation. I want to use one as the underpinning of a b2b lending simulation I'm planning"*

2. *"Objective is as follows — 3 objective: 1. Test the value, functionality and token efficiency of the LOS I'm working on designed for agentic tools for business lending 2. Evaluate LLM models on ability to do business credit underwriting based on finance and bank statements 3. Evaluate the shrewdness of models for operating businesses, taking loans, beating competitors … I want the business side of the world to be something simple like Catan… but my primary objective is 1+2…"*

3. *"Review the plan above, and ask any questions necessary to tighten the spec and make some key decisions"*

4. *"What would be the most valuable from revenue perspective CLI tools to add to the lender side and business side to enable the sim to be more efficient and also improve our LOS from a capability perspective"*

5. *"One principle to add around CLI — We should introduce new world complexity as we release CLI capabilities. Eg if we add multi bank account then we do it when we launch a CLI that supports consolidation? This drives our roadmap?"*

6. *"Yes lay out a roadmap with decision criteria for each and something sensible that reflects the discussion"*

7. *"Give me a brief outline of the objective, compromises and roadmap"*

8. *"Turn this full chat into a detailed doc with the raw questions from me as appendix"*
