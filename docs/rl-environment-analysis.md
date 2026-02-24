# Loanville as an RL Environment: Lessons from Scale AI's Vending Machine

**Date:** 2026-02-24
**Context:** Analysis of Scale AI roundtable discussion on RL environments (hosted by Akash Bajwa, featuring Matt and Thomas from Scale AI) and how it maps to Loanville's architecture.

---

## 1. The Structural Parallel

The vending machine environment and Loanville share a remarkably similar loop:

| Vending Machine | Loanville (Current) | Loanville (Season Mode) |
|---|---|---|
| **Observation:** inventory, cash, time, delta log | **Observation:** borrower dossiers, lender constraints, capital | **Observation:** dossiers + portfolio state, P&L history, sector exposure |
| **Action:** JSON restock orders + price changes | **Action:** JSON APPROVE/REJECT + term sheet (amount, rate, term) | Same, plus tool creation and triage decisions |
| **Step function:** update simulator, return new state | **Engine:** adjudicate deals, resolve loans, compute outcomes | **Season loop:** weekly cohorts, rolling resolution, carry-forward |
| **Reward:** cash balance (continuous) | **Reward:** RAROC score (continuous) | RAROC + portfolio management + efficiency |

Loanville is already closer to a proper RL environment than the vending machine was at hackathon time. The key differences: Loanville uses LLM inference (not weight updates) and evaluates pre-trained models rather than training them. The question is what changes if we want to support actual RL training.

---

## 2. What Loanville Already Gets Right

### 2.1 Dense, Continuous Rewards

The Scale AI team emphasized that sparse binary rewards (1/0 at episode end) are compute-wasteful. Loanville's RAROC scoring is already dense and continuous:

- **Per-loan payoff** (`compute_loan_payoff`): each loan produces a continuous P&L signal, not binary pass/fail
- **Multi-component score**: interest earned, principal lost, funding cost, fraud penalty — all real-valued
- **Penalty decomposition** (`compute_penalty_decomposition`): seven distinct penalty types, each continuous and independently measurable
- **Season mode's rolling resolution** (PRD): intermediate rewards every week as loans age

This is exactly the "continuous reward instead of binary" improvement they described. Loanville naturally produces ~24 reward signals per episode (one per borrower decision) rather than a single end-of-episode number.

### 2.2 Operating Costs Prevent Inaction

The vending machine team baked in operating costs so "doing nothing = consistent losses." Loanville has the equivalent:

- **Volume penalty**: deploying <20% of capital incurs quadratic penalty
- **Opportunity cost**: undeployed capital benchmarked against risk-free rate (5%)
- **Funding cost**: 4% cost of funds on deployed capital

A model that rejects everything scores negative. A model that approves everything gets crushed by defaults. The optimal strategy requires genuine credit analysis — the scoring design already prevents the trivial equilibria that plague simpler environments.

### 2.3 Verifiable Outcomes

Scale AI identified coding as the "gold standard" for RL because outputs are programmatically testable. Lending has a similar property: **loan outcomes are deterministic given the borrower's ground truth.** There's no subjective judgment needed to score — either the borrower repaid or defaulted, and the P&L follows mechanically.

This puts Loanville in the "domains that work well with RL" category alongside coding and tool interaction, not in the "needs rubrics" category like creative writing or strategy consulting.

---

## 3. What We Can Learn

### 3.1 The Competition Gap (Critical)

> "The real problem was environment realism: the simulation had no competition. In the real world, high-margin strategies attract competitors who arbitrage profits away."

This was the vending machine team's key finding and **Loanville already partially addresses it** through competitive deal adjudication (borrowers pick the lowest rate). But there are gaps:

**What Loanville has:**
- Winner-takes-all competitive bidding on each deal
- Borrowers prefer lowest rate → pricing aggression is constrained

**What's missing for RL training:**
- **No competitor adaptation.** In the current sim, competitors are other LLM instances with fixed weights. In a real RL training loop, all agents would be improving simultaneously (self-play). The vending machine had zero competition; Loanville has static competition.
- **No market equilibrium pressure.** If a model learns to price at 8% and wins everything, there's no mechanism for competitors to respond with 7.5% in the next episode.

**Recommendation for RL:** Implement **self-play** or **population-based training** where multiple copies of the same model train against each other. The Elo tournament system (`elo_benchmark.py`) already provides the infrastructure for multi-agent evaluation — the gap is connecting this to a training loop rather than just using it for benchmarking pre-trained models.

### 3.2 Reward Hacking Vulnerabilities

The vending machine team found a specific exploit: one item had low cost + high demand, so the agent filled every slot with it. They caught it by **manually reviewing generation logs.**

Loanville has analogous vulnerabilities:

| Potential Exploit | Current Mitigation | Gap |
|---|---|---|
| Approve all good, reject all bad (if ground truth leaks) | Ground truth hidden from LLM | Prompt injection via dossier content could reveal patterns |
| Fund only one sector at maximum margin | Concentration limits (sector caps) | Caps are generous (25-30%); a smart agent could load up to the limit in every sector |
| Price every loan at maximum rate | Yield drag penalty | But yield drag is small (just shortfall % × principal × term); agent could price at exactly target yield |
| Reject everything to avoid losses | Volume floor (20% minimum deployment) | 20% is low; agent could approve 2 small safe loans and reject the rest |
| Approve everything for volume, accept default penalty | Hard constraint: 30% max default rate | 30% is generous; an agent could fund 7 good + 3 bad out of 10 and still be under the cap |

**Recommendation:** Build a **reward hacking detection suite** that:
1. Flags monotonic strategies (all-approve, all-reject, single-sector loading)
2. Compares trained model behavior against the DSCR heuristic baseline — if the trained model converges to the exact same decisions as the heuristic, it may have found a shortcut rather than learning deeper credit analysis
3. Uses the **dual-benchmark approach** described in the discussion: train on mix A, monitor on mix B. If performance on mix B degrades while mix A improves, the model is overfitting to the environment rather than learning lending.

### 3.3 Environment Realism Gaps

The discussion identified that the vending machine lacked:
- Competition (addressed above)
- Delivery lag (they added it — orders arrive in future steps)
- Expiration costs
- Market dynamics

Mapping to lending, Loanville's realism gaps for RL training:

| Gap | Impact on RL Training | Difficulty |
|---|---|---|
| **No macro environment** — interest rates, recessions, sector cycles | Agent can't learn to adapt to changing conditions | Medium (add a `MacroState` that shifts default probabilities per week) |
| **Predetermined outcomes** — borrower fate is fixed at creation | Agent learns to pattern-match dossier features to static labels, not to assess dynamic risk | High (requires evolving borrower financials — acknowledged in Season PRD §11.3) |
| **No relationship effects** — rejected borrowers don't come back | No learning about relationship value or reputation | Low (add borrower re-application with memory of prior rejection) |
| **Perfect information within a dossier** — all financial data is available | Real underwriting involves incomplete information and verification costs | Medium (Season PRD §11.2 phased pipeline partially addresses this) |
| **No prepayment/early exit** — good loans always run full term | Misses a real source of portfolio uncertainty | Low (add stochastic early repayment for good borrowers) |

### 3.4 Dense vs. Sparse Reward Design

The discussion's hierarchy:
1. **Sparse binary** (worst): did you make money? yes/no
2. **Sparse continuous**: total cash at episode end
3. **Dense continuous** (best): intermediate step rewards + curriculum shaping

Loanville's current position:

- **Single-match mode**: effectively sparse continuous — one RAROC score at the end
- **Season mode (PRD)**: dense continuous — weekly P&L updates, rolling resolution, portfolio briefings

Season mode as designed in the PRD already implements what the Scale AI team recommended. The weekly portfolio briefing (PRD §3.3) is literally the "delta log" from the vending machine observation space.

**For RL training specifically**, we'd want to emit a **per-step reward** at each decision point:

```
Step = one borrower evaluation
Observation = {current_portfolio_state, borrower_dossier, capital_remaining, sector_exposure}
Action = {decision: APPROVE|REJECT, term_sheet: {amount, rate, term}}
Reward = immediate signal (see below)
```

The challenge: individual loan outcomes aren't known until resolution (months later). Options for intermediate reward:

| Approach | Signal Quality | Compute Cost | Risk |
|---|---|---|---|
| **Wait for outcome** (sparse per-loan) | High | Low | Slow training signal |
| **Predicted payoff at booking** (use `compute_loan_payoff` with ground truth) | Perfect but cheats | Zero | Agent learns oracle, not credit analysis |
| **Heuristic credit score at booking** | Medium | Low | Agent learns to match heuristic, not to be a good lender |
| **Season-end RAROC with per-step shaping** | High | Medium | Shaping bias, but tractable |

**Recommendation:** For RL training, use **season-end RAROC as the primary reward** with **dense shaping from per-week portfolio P&L changes**. This is the lending equivalent of the vending machine's "cash balance at each step." The weekly portfolio briefing already computes this.

### 3.5 Observation/Action Space Formalization

To function as a proper RL environment (e.g., Gymnasium-compatible), Loanville needs a formal observation/action space. Currently, observations are free-text prompts and actions are JSON extracted from LLM completions.

For LLM-based RL (GRPO, RLHF, etc.), the text-based interface is actually correct — the observation is the prompt, the action is the completion. But the environment needs to expose:

```python
class LoanvilleEnv:
    def reset(self, seed=None) -> Observation:
        """Start a new season/episode. Return initial state."""

    def step(self, action: str) -> tuple[Observation, float, bool, dict]:
        """Process one lending decision. Return (obs, reward, done, info)."""

    @property
    def action_space(self) -> TextSpace:
        """JSON schema for valid actions."""

    @property
    def observation_space(self) -> TextSpace:
        """Description of what the agent sees."""
```

This is a thin wrapper around the existing `SimulationEngine` / `SeasonEngine`. The key architectural decision is **granularity of steps** — is one step one borrower decision, or one full week?

**Recommendation:** One step = one borrower decision within a season. This gives maximum training signal density. The season loop becomes the episode, and each borrower evaluation is a step.

### 3.6 Training Pipeline Efficiency

The discussion emphasized that RL is inherently sequential (simulate → train → repeat) and that environment step functions must be fast to not block training.

Loanville's current bottleneck: **LLM inference.** Each evaluation requires an API call to OpenRouter, which takes 5-30 seconds. For RL training, we need:

1. **Local model inference** — can't use API calls in the training loop. The model being trained must run locally (vLLM, SGLang, etc.)
2. **Fast environment stepping** — the simulation logic (adjudication, resolution, scoring) is pure Python math. This is already fast.
3. **Batched evaluation** — GRPO processes groups of trajectories. Loanville needs to support running N parallel episodes simultaneously.

The `mock_llm.py` module already provides fast deterministic evaluations without API calls. For RL training, we'd need a similar interface that calls a local model server instead.

**Compute profile estimate for training:**
- Episode = 10-week season × 5 borrowers/week = 50 steps
- GRPO batch = 8-16 parallel episodes per gradient step
- Training run = 50-200 gradient steps
- Total inference calls = 50 × 16 × 200 = 160,000
- At ~500 tokens/call, that's ~80M tokens of inference per training run

This is feasible on a single 8×H100 node running a 7B-70B model. The CPU bottleneck the discussion raised (for complex OS-level simulations) is not an issue here — Loanville's step function is lightweight arithmetic.

---

## 4. The Gymnasium Interface

A concrete spec for making Loanville a Gymnasium-compatible environment:

```python
import gymnasium as gym

class LoanvilleSeasonEnv(gym.Env):
    """
    RL environment for commercial lending.

    Episode = one season (10 weeks, ~50 borrower decisions)
    Step = one borrower evaluation
    Observation = text prompt (portfolio state + borrower dossier)
    Action = text completion (JSON decision + term sheet)
    Reward = per-step shaping + end-of-episode RAROC
    """

    metadata = {"render_modes": ["text"]}

    def __init__(self, config: dict):
        self.season_config = SeasonConfig(**config)
        self.lender_config = config["lender"]  # single-agent for training
        # Competitors are fixed policies (mock or frozen model snapshots)
        self.competitor_policies = config.get("competitors", [])

    def reset(self, seed=None) -> tuple[str, dict]:
        """Initialize a new season. Returns first observation."""
        self._rng = np.random.default_rng(seed)
        self._state = SeasonLenderState(...)
        self._week = 1
        self._borrower_idx = 0
        self._cohort = self._generate_cohort(self._week)

        obs = self._build_observation()
        info = {"week": 1, "borrower_idx": 0}
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """
        Process one lending decision.

        action: JSON string with decision and optional term sheet
        Returns: (observation, reward, terminated, truncated, info)
        """
        # 1. Parse action
        decision = self._parse_action(action)

        # 2. Run competitors on same borrower
        competitor_decisions = self._run_competitors(self._current_borrower)

        # 3. Adjudicate
        deal_result = self._adjudicate(decision, competitor_decisions)

        # 4. Compute step reward (shaping)
        reward = self._compute_step_reward(decision, deal_result)

        # 5. Advance to next borrower or next week
        self._borrower_idx += 1
        if self._borrower_idx >= len(self._cohort):
            self._end_week()
            self._week += 1
            if self._week > self.season_config.weeks:
                # Episode done — add terminal reward
                terminal_reward = self._compute_season_raroc()
                reward += terminal_reward
                return "", reward, True, False, self._build_info()
            self._cohort = self._generate_cohort(self._week)
            self._borrower_idx = 0

        obs = self._build_observation()
        return obs, reward, False, False, self._build_info()

    def _compute_step_reward(self, decision, deal_result) -> float:
        """Dense reward shaping per decision."""
        reward = 0.0

        # Small reward for each decision to keep training signal flowing
        if deal_result["outcome"] == "booked" and deal_result["winner"] == "agent":
            # Won a deal — small positive signal proportional to expected margin
            loan = deal_result["loan"]
            expected_interest = loan.principal * (loan.interest_rate / 100) * (loan.term_months / 12)
            reward += expected_interest * 0.01  # Scale down to not overwhelm terminal reward

        # Penalty shaping for obvious errors (optional, trades off with reward hacking risk)
        # reward -= fraud_detection_shaping(...)

        return reward
```

### 4.1 Episode Structure Options

| Option | Steps/Episode | Signal Density | Training Speed | Realism |
|---|---|---|---|---|
| **Single match** (current) | 12-24 | Low | Fast | Low |
| **Season (10 weeks × 5)** | 50 | High | Medium | High |
| **Mini-season (4 weeks × 4)** | 16 | Medium | Fast | Medium |

**Recommendation:** Start with mini-seasons (16 steps) for initial training runs, scale to full seasons once the reward signal is validated.

---

## 5. Reward Hacking Prevention Checklist

Drawing from the Scale AI discussion, a concrete checklist for Loanville:

### 5.1 Manual Review Protocol
- [ ] After every training run, sample 10 episodes and read raw decision traces
- [ ] Check for monotonic strategies (all-approve, all-reject, single-rate)
- [ ] Verify the model is actually reading financial data (check tool call content, not just decisions)

### 5.2 Dual-Benchmark Monitoring
- [ ] Train on mix A (e.g., `realistic`), monitor on mix B (e.g., `adversarial`)
- [ ] Track both mixes per training step — divergence = hacking signal
- [ ] Also monitor on `fraud`-only and `analyst`-only mixes for capability-specific regression

### 5.3 Environment Robustness
- [ ] Randomize borrower presentation order within cohorts
- [ ] Randomize sector names and company names between episodes (to prevent memorization)
- [ ] Add noise to financial figures (±2-5%) between episodes for the same borrower archetype
- [ ] Verify the model doesn't key on borrower ID strings or other metadata that correlates with outcome

### 5.4 Reward Function Auditing
- [ ] Test each penalty component independently — can the agent minimize penalty X without gaming it?
- [ ] Verify that the volume floor (20%) can't be satisfied by two small safe loans while rejecting everything else
- [ ] Check that concentration limits interact correctly (agent shouldn't be able to load every sector to its limit)
- [ ] Confirm the scoring function is deterministic given the same decisions (no hidden randomness that could be exploited)

---

## 6. Multi-Agent Considerations

The Scale AI discussion clarified that their 10 evaluation runs were independent runs of the same model, not separate agents. Loanville's Elo tournament is already more sophisticated — it runs true multi-agent competition.

For RL training, the multi-agent dimension creates options:

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **Single-agent + fixed competitors** | Train one model against frozen opponents (mock or snapshots) | Simple, stable | Doesn't learn competitive dynamics |
| **Self-play** | All agents share weights, train simultaneously | Learns competitive pricing | Unstable, can collapse |
| **Population-based** | Maintain a population of N policies, train against random opponents from the pool | Best of both worlds | N× compute |
| **League training** (AlphaStar-style) | Main agent + exploiter agents + league of past versions | Most robust | Very complex |

**Recommendation for v1:** Single-agent with frozen competitors. Use the existing mock LLM or snapshots of pre-trained models as opponents. This isolates the credit analysis learning signal from the competitive dynamics. Add self-play in v2 once the single-agent training loop is stable.

---

## 7. Loanville's Unique RL Advantages Over the Vending Machine

Several properties make Loanville a *better* RL environment than the vending machine case study:

1. **Natural competition.** The vending machine had none. Loanville's deal adjudication creates genuine competitive pressure, preventing the "globally optimal strategy in an unrealistic environment" problem.

2. **Rich observation space.** The vending machine had ~20 numbers (inventory slots, prices, cash). Loanville has 12 months of bank statements, quarterly income, narratives — thousands of tokens of structured financial data. This means the model must learn information extraction, not just numerical optimization.

3. **Heterogeneous decision space.** Vending machine: adjust prices and restock (two continuous controls). Loanville: binary approve/reject + continuous pricing (rate, amount, term) + portfolio-level decisions (concentration, capital rationing). More axes of decision quality to differentiate models.

4. **Verifiable without being trivial.** The vending machine was solvable by basic optimization (the model crushed it in 50 steps). Loanville's fraud detection and over-leverage analysis require genuine analytical reasoning that simple optimizers can't replicate — the heuristic baseline proves the task is solvable, but the gap between heuristic and perfect foresight (~5-10 RAROC points) leaves room for models to show differentiated capability.

5. **Domain-relevant.** The Scale AI team acknowledged that vending machines don't drive enough economic value to justify investment. Commercial lending is a $4T+ market. An RL environment that produces better lending models has direct enterprise applicability.

---

## 8. Implementation Roadmap

### Phase 1: Gymnasium Wrapper (enables RL experimentation)
- Implement `LoanvilleSeasonEnv` wrapping the existing `SeasonEngine` (once Season Mode is built)
- Single-agent mode with mock competitors
- Text-based observation/action spaces
- Dense reward: per-step shaping + terminal RAROC
- Deterministic seeding for reproducibility

### Phase 2: Training Integration
- vLLM/SGLang local inference backend (replacing OpenRouter API calls)
- GRPO training script using the Gymnasium env
- Reward hacking detection suite (dual-benchmark monitoring, trace sampling)
- Baseline comparisons: pre-trained model vs. RL-finetuned model

### Phase 3: Multi-Agent
- Self-play mode: N copies of the training model compete
- Population-based training with league of past checkpoints
- Elo-based evaluation against fixed model population

### Phase 4: Environment Enrichment
- Macro state (interest rate cycles, sector shocks)
- Stochastic borrower financials (noise between episodes)
- Phased arrival / bandwidth management (from Season PRD §11.2)
- Refinancing dynamics (from Season PRD §11.1)

---

## 9. Key Takeaways

| Scale AI Lesson | Loanville Status | Action Required |
|---|---|---|
| Dense continuous rewards > sparse binary | Already done (RAROC, per-loan payoff) | Formalize per-step reward in Gym wrapper |
| Operating costs prevent trivial equilibria | Already done (volume penalty, funding cost) | Audit for edge cases where penalties are too weak |
| Competition prevents unrealistic strategies | Partially done (deal adjudication) | Add self-play for RL training |
| Reward hacking is caught by manual review | Not yet formalized | Build trace sampling + dual-benchmark monitoring |
| Environment realism bounds strategy quality | Moderate realism | Season mode addresses the biggest gaps |
| CPU bottleneck for complex environments | Not an issue (lightweight step function) | Focus on inference throughput instead |
| GRPO works with short training runs (50 steps) | N/A (not training yet) | Promising — Loanville episodes are similar length |
| Constrained reasoning prevents benchmark crushing | Not implemented | Consider limiting tool calls or reasoning tokens in RL training |

The core insight: **Loanville is already well-designed as an evaluation benchmark. The gap to being an RL training environment is primarily infrastructure (Gymnasium wrapper, local inference, training loop), not environment design.** Season mode closes most of the remaining environment design gaps. The Scale AI discussion validates the architectural choices already made in the RAROC scoring system and the Season Mode PRD.
