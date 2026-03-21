# Terra Nova vs Loanville2: Comprehensive Challenge Environments

## Context

Analysis of [Terra Nova](https://trevormcinroe.github.io/terra_nova_research) (arXiv:2511.15378, McInroe 2025) — a Civilization V-inspired reinforcement learning challenge environment designed as a "Comprehensive Challenge Environment" (CCE) where multiple canonical RL difficulties emerge simultaneously from a single unified environment.

---

## 1. What Terra Nova Does

Terra Nova is a RL environment that deliberately combines multiple challenge types into one game:

- **Partial observability** — fog of war, incomplete information
- **Credit assignment** — delayed rewards, many interacting causes
- **Representation learning** — complex state spaces requiring learned abstractions
- **Enormous action spaces** — many possible moves per turn
- **Long-horizon planning** — decisions compound over hundreds of turns

The key design principle: these challenges interact and compound. A multitask benchmark that tests each in isolation "primarily assesses whether an agent can catalog and switch among unrelated policies rather than test an agent's ability to perform deep reasoning across many interacting variables."

---

## 2. Relevance to Loanville2

### a) CCE Design Philosophy — Validates Our Approach

Terra Nova's core argument is that real intelligence requires handling multiple interacting challenges simultaneously, not switching between isolated tasks. Loanville2 is already a CCE by this definition:

| Challenge | How It Manifests in Loanville2 |
|-----------|-------------------------------|
| Partial observability | Bank statements may be incomplete, narratives may be misleading |
| Credit assignment | Was a default caused by bad underwriting or market conditions? |
| Representation learning | Models must build internal representations of "risky borrower" from heterogeneous signals |
| Large action space | Approve/reject × pricing × amount × term creates a continuous decision space |
| Long-horizon planning | Season mode: week-to-week capital management, portfolio construction |

**Insight**: Loanville2 is closer to Terra Nova's CCE vision than to typical LLM benchmarks that test isolated capabilities. This is a positioning strength — we test integrated financial reasoning, not just "can the model do math" or "can it follow instructions."

### b) Partial Observability as a Feature

Terra Nova treats partial observability as a core challenge dimension. Loanville2's `info_asymmetry` modes (already implemented) create this:
- `partial_statements` — each lender sees a random subset of bank statement months
- `redacted` — some financial fields are hidden per-lender

This directly mirrors Terra Nova's fog-of-war mechanic: different agents have different views of the same ground truth.

### c) Credit Assignment Problem

Terra Nova highlights that delayed, multi-cause rewards are hard for agents. In Loanville2, a lender that approves a loan doesn't learn the outcome for several season weeks. The season mode's portfolio briefing partially addresses this by showing which loans defaulted, but the model still must attribute outcomes to its own decisions vs market conditions.

**Future idea**: Add a "decision attribution" field to the briefing that explicitly links past decisions to outcomes — "Your W3 approval of TechCo defaulted in W6; the DSCR was 0.9x at origination."

---

## 3. Key Differences

| Aspect | Terra Nova | Loanville2 |
|--------|-----------|------------|
| Agent type | RL agents (policy networks) | LLM agents (prompted models) |
| Environment | Civ-V grid world | Financial simulation |
| Learning | Weight updates (training) | In-context learning only |
| Multi-agent | Competitive (multiple civs) | Competitive (multiple lenders) |
| Evaluation | RL metrics (reward, win rate) | Business metrics (RAROC, Elo) |
| Open-source | Yes (environment) | Yes (full sim + borrower data) |

---

## 4. Takeaway

Terra Nova validates Loanville2's design as a comprehensive challenge environment rather than a narrow benchmark. The interacting challenges of credit analysis, fraud detection, pricing strategy, portfolio construction, and capital management create the kind of "integrated, long-horizon understanding across many interacting variables" that Terra Nova argues is the right way to evaluate intelligence.

No direct features to implement — this is primarily a conceptual validation and positioning reference.
