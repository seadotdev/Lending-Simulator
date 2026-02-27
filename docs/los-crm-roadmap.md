# LOS Formality + CRM Simulation Roadmap

This document maps the three objectives into implementation tracks.

## Objective 1: Formal LOS-only interactions

### Goal
Only formal loan actions (submission/offer) are valid when produced through the LOS pipeline and LOS tool calls.

### Implemented
- CLI now defaults to **formal LOS-only** mode.
- Formal mode blocks legacy paths:
  - `--mock`
  - `--underwrite-only`
  - `--los-mode rules_only`
- Approvals without LOS `tool_call` trace evidence are downgraded to non-formal (`PASS`), so they cannot become booked offers.
- Enforcement applies to both single-run and season mode.

### Escape hatch
- `--allow-non-los-formal` enables legacy behavior for debugging/comparison runs.

## Objective 2: CRM tests as simulations (high-volume)

### Goal
Shift from test-style assertions to simulation-style workloads that measure:
- speed (throughput/latency)
- correctness (formulaic decision accuracy)
- responsiveness (follow-up behavior on incomplete cases)
- LOS formal compliance (offers backed by tool traces)

### Implemented
- New high-volume CRM simulation mode:
  - `python -m loanville --crm-sim`
- Inputs are procedurally generated borrower queues with configurable incomplete-intake ratio.
- Scoring reports:
  - decision accuracy
  - pricing accuracy (APR tolerance against deterministic target formula)
  - follow-up responsiveness
  - formal-offer compliance
  - throughput and latency
- Simulation report includes visibility into the upstream Open LOS CRM scenario corpus:
  - `open-los/packages/simulation/src/crm-test/scenarios.ts`

## Objective 3: Deeper financials (plan)

### Goal
Use `rl-benchmarks`-sourced patterns to synthesize realistic multi-entity portfolios including defaults, early redemption, and known fraud while keeping data synthetic and high fidelity.

### Plan
1. Data ingestion and schema bridge
- Add an ingestion adapter that maps `rl-benchmarks` entities/events into Loanville dossier schema.
- Preserve provenance metadata per generated borrower and per synthetic statement row.

2. Multi-entity synthetic generator
- Generate parent/subsidiary/guarantor structures with cross-entity cash movement and shared counterparty exposures.
- Add configurable event overlays: default chains, early redemptions, and fraud signatures.

3. High-fidelity bank statement cloning
- Clone temporal and categorical transaction patterns from source distributions while anonymizing identifiers.
- Add controllable noise so generated statements remain realistic but non-identifying.

4. LOS spread + underwrite workflow pressure
- Add simulation gates that require both LOS spread and LOS underwrite tool usage for high scores.
- Track token-efficiency KPIs:
  - tokens per decision
  - tokens per booked loan
  - evidence-to-token efficiency score

5. Rollout
- Phase A: shadow mode (new data generated but not scored)
- Phase B: scored optional pack
- Phase C: default season pack for deep-financial benchmarks

