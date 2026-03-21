# InteractiveBench vs Loanville2: Multi-Turn Evaluation and Consistency

## Context

Analysis of [InteractiveBench](https://github.com/interactivebench/InteractiveBench) — a benchmarking framework that evaluates LLMs through interactive, multi-turn tasks (situation puzzles, math proofs, trust games, poker) rather than single-pass evaluations — compared to Loanville2's competitive lending simulation.

---

## 1. What InteractiveBench Does

Four evaluation domains:

| Domain | Method | What It Tests |
|--------|--------|---------------|
| Situation Puzzles | Multi-turn Q&A with a judge | Iterative reasoning, hypothesis refinement |
| Interactive Math | Models can request intermediate verification (proof-style) | Self-correction, tool use discipline |
| Trust Game | Repeated cooperation/defection with multiple agents + human baselines | Game-theoretic reasoning, reputation modeling |
| Poker | Strategic play under uncertainty | Bluffing, expected value reasoning |

Key innovation: **pass@k evaluation** — sampling multiple completions and measuring agreement as a signal of model confidence and calibration.

---

## 2. Relevance to Loanville2

### a) Pass@k Decision Consistency — **IMPLEMENTED**

InteractiveBench's pass@k methodology directly addresses a gap in Loanville2: we measure *what* a model decides but not *how stable* that decision is.

A model that approves a borderline borrower 8/10 times vs 5/10 times reveals critical calibration information:
- **High agreement (>80%)**: Model has confident internal threshold — decision is reliable
- **Low agreement (<60%)**: Model is near its decision boundary — the approval may be noise
- **Asymmetric disagreement**: Approve→Reject flips vs Reject→Approve flips reveal directional bias

**Implementation**: `--consistency-samples K` re-evaluates a random subset of each week's borrowers K additional times and tracks per-lender agreement rate. This is expensive (multiplies API calls by ~K×sample_pct) so it's opt-in and samples only a fraction of the cohort.

### b) Trust Game Tournament — Conceptual Parallel

InteractiveBench's trust game evaluates whether agents can build cooperative relationships across repeated interactions. In Loanville2, the pricing dynamic has a trust-game structure:

- **Cooperation**: All lenders price fairly → all earn reasonable returns
- **Defection**: One lender undercuts aggressively → wins deals but may take on bad risk
- **Punishment**: Market learns aggressive lender exists → others adjust

The season mode's competition feedback already enables this dynamic. The trust game analysis suggests we could formalize this by tracking whether models exhibit tit-for-tat, always-defect, or other identifiable strategies across seasons.

### c) Interactive Proof-Style Evaluation

InteractiveBench's math domain lets models request intermediate verification before committing to answers. Loanville2 already supports this via the bash tool — models can query bank statements, run calculations, and verify hypotheses before issuing a decision.

**Insight**: InteractiveBench found that interactive-proof solving significantly outperforms naive single-pass solving. This validates Loanville2's tool-use architecture where models that use `run_bash` to verify DSCR calculations before deciding tend to perform better than models that reason purely from the narrative.

### d) Human Baselines

InteractiveBench includes human baselines in its trust game evaluation, giving the ratings intuitive calibration. Loanville2 could add human underwriter decisions as fixed reference points on the Elo leaderboard — e.g., a junior analyst baseline and a senior credit officer baseline.

---

## 3. Key Differences

| Aspect | InteractiveBench | Loanville2 |
|--------|-----------------|------------|
| Primary metric | Task accuracy per domain | RAROC / triple Elo |
| Multi-turn depth | 5-20 turns per task | 1 decision per borrower (10 weeks in season) |
| Agent interaction | With environment (judge, verifier) | With borrower data (tools, statements) |
| Competitive element | Trust game only | Core mechanic (pricing competition) |
| Pass@k | Core methodology | Implemented as opt-in consistency check |
| Cost model | Flat (per-task) | Realistic (token costs tracked, pricing API) |

---

## 4. Actionable Ideas Adopted

1. **Pass@k consistency scoring** — Implemented as `--consistency-samples` in season mode
2. **Human baselines** — Future work: add expert underwriter decisions as Elo reference points
3. **Agreement-rate as reliability signal** — Tracked per-lender across the season, included in JSON export and season report
