# OASIS vs Loanville2: Large-Scale Multi-Agent Social Simulation

## Context

Analysis of [OASIS](https://github.com/camel-ai/oasis) (CAMEL-AI) — a scalable open-source social media simulator that uses LLM agents to mimic up to one million users on Twitter/Reddit-like platforms. Agents autonomously create posts, comment, follow, like, and transact, enabling research on information dissemination, group polarization, and behavioral contagion.

---

## 1. What OASIS Does

- **Scale**: Up to 1M concurrent LLM-powered agents on simulated social platforms
- **23 distinct actions**: post, comment, follow, like, mute, search, transact, etc.
- **Recommendation algorithms**: Interest-based and hot-score-based content discovery
- **Platform templates**: Twitter-like (tweets, retweets, engagement) and Reddit-like (subreddits, ranking, communities)
- **Electronic mall**: Transaction simulation on the Reddit platform variant
- **Research focus**: Information flow, polarization, contagion, synthetic dataset generation

### Architecture

- Asynchronous execution for agent parallelism
- Database-backed state persistence (agent memories, relationships, content)
- Modular agent graph — configurable agent types and interaction patterns
- Both LLM-driven autonomous actions and manual/scripted injection

---

## 2. Relevance to Loanville2

### LOW for current benchmarking — HIGH for future business simulation evolution

OASIS operates in a fundamentally different domain (social media) but its architectural patterns are highly relevant if Loanville2 evolves toward richer market simulation.

### a) Agent-to-Agent Communication — Future Lending Market

OASIS agents communicate through posts and comments, creating emergent information networks. Loanville2's agents currently operate in isolation — they see borrower data but not each other's behavior (except indirectly via competition feedback).

**Future idea**: A "lending market" where:
- Borrowers post loan requests publicly (like OASIS posts)
- Lenders can comment/negotiate terms (multi-round offers)
- Other lenders see deal flow and pricing signals
- Information cascades emerge — if one lender rejects a borrower, does that signal something to others?

This would transform Loanville2 from "parallel evaluation" into a true multi-agent market simulation.

### b) Recommendation/Matching Algorithms

OASIS uses recommendation algorithms to surface content to agents. The lending equivalent is **deal flow curation** — how borrowers get matched to lenders. Currently Loanville2 broadcasts all borrowers to all lenders. A recommendation layer could:
- Route borrowers to lenders based on sector expertise and portfolio fit
- Create information asymmetry through differential deal flow (some lenders see deals first)
- Model real-world broker/marketplace dynamics

### c) Scalability Patterns

OASIS scales to 1M agents via:
- Async execution (Loanville2 already uses asyncio)
- Database-backed state (Loanville2 uses in-memory dataclasses)
- Modular agent graphs (Loanville2 has fixed lender/borrower roles)

**Future idea**: If Loanville2 scales to 50+ lenders or multi-season tournaments, adopting database-backed state persistence would enable:
- Resume from checkpoint after crashes
- Historical analysis across seasons
- Real-time dashboards querying live state

### d) Behavioral Contagion and Herding

OASIS studies how behaviors spread through agent networks. In lending, **herding** is a well-documented phenomenon:
- If several lenders reject a borrower, others follow (credit rationing)
- If one lender offers aggressive pricing, others race to match (yield compression)
- Sector booms: one successful loan in a sector triggers concentration by all lenders

Loanville2's scratchpad + competition feedback create the conditions for herding to emerge naturally. Measuring and reporting herding behavior could be a valuable analytical dimension.

### e) Electronic Mall — Transaction Simulation

OASIS includes a basic transaction/marketplace feature. This validates the concept of embedding financial transactions into agent simulations. Loanville2 is already more sophisticated here (full loan lifecycle, RAROC, amortization), but the pattern of "agents transacting within a simulated economy" is shared.

---

## 3. Key Differences

| Aspect | OASIS | Loanville2 |
|--------|-------|------------|
| Domain | Social media | Commercial lending |
| Agent count | Up to 1M | 3-10 lenders |
| Agent actions | 23 social actions | Approve/reject + pricing |
| Communication | Public (posts, comments) | None (blind evaluation) |
| Economy | Basic transactions (e-mall) | Full loan lifecycle (RAROC, amortization) |
| Information flow | Recommendation algorithms | Broadcast to all |
| State persistence | Database-backed | In-memory dataclasses |
| Evaluation | Social dynamics metrics | Financial performance + Elo |
| Open-source | Yes | Yes |

---

## 4. Actionable Ideas (Future — Business Simulation Side)

These are all medium-to-large efforts, documented here for roadmap consideration:

1. **Multi-round negotiation**: Borrowers and lenders exchange offers/counteroffers rather than one-shot decisions. Models the real lending process more accurately.

2. **Information cascade modeling**: Track whether rejection by one lender influences others' decisions. Requires agent-to-agent signaling (even indirect, via market-level stats).

3. **Deal flow curation / recommendation**: Route borrowers to lenders based on fit rather than broadcasting to all. Creates strategic value in specialization.

4. **Database-backed state**: Migrate from in-memory dataclasses to SQLite for checkpoint/resume, cross-season analysis, and real-time dashboards.

5. **Herding detection**: Measure and report sector concentration correlation across lenders as a behavioral metric (do all lenders pile into the same sectors?).

---

## 5. Takeaway

OASIS is more relevant to a future evolution of Loanville2 toward rich market simulation than to the current benchmarking use case. Its strongest contributions are architectural patterns for agent communication, scalability, and emergent social dynamics — all of which have direct analogues in lending markets. The current Loanville2 benchmarking mode (parallel evaluation, RAROC scoring, Elo tournaments) doesn't need these patterns, but a "Loanville3: Market Simulation" could borrow heavily from OASIS's agent graph and communication architecture.
