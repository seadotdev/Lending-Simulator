"""
Gymnasium-style RL environment adapter for Loanville.

Wraps the SimulationEngine and SeasonEngine into a step/reset interface
compatible with RL training loops.  This is the primary integration point
for frameworks like verifiers, trl, or custom GRPO implementations.

Design:
  - ``reset()`` initializes a new episode (borrower cohort + lender state)
  - ``step(action)`` processes one underwriting decision and returns
    (observation, reward, done, truncated, info)
  - Observations are dicts (text-based, for LLM agents)
  - Actions are dicts (JSON decisions, for LLM agents)
  - Rewards come from the pluggable verifier/rubric system

This does NOT depend on gymnasium as a hard requirement — it follows
the interface pattern so that a gymnasium wrapper is trivial to add.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..data import get_borrowers, MIX_PRESETS
from ..models import (
    Borrower,
    EconomicsConfig,
    ECONOMICS_PRESETS,
    LenderConfig,
    LenderDecision,
    SeasonConfig,
    TermSheet,
)
from ..scoring import compute_loan_payoff
from .rollout import Rollout, Step
from .verifier import CreditQualityVerifier, RewardResult, Rubric


@dataclass
class EnvConfig:
    """Configuration for the RL environment.

    Controls episode structure, borrower generation, and reward computation.
    """
    # Episode structure
    episode_type: str = "single"  # "single" | "season"
    n_borrowers: int = 10
    # Season-specific
    weeks: int = 10
    cohort_size: int = 5
    months_per_week: int = 2
    season_mix: str = "realistic"
    # Economics
    economics: str = "balanced"  # preset name or "custom"
    economics_config: Optional[dict] = None
    # Reproducibility
    seed: int = 42
    # Observation mode
    data_mode: str = "full"
    # Reward shaping
    dense_rewards: bool = True  # per-step reward signals
    reward_scale: float = 1.0
    # Lender persona (used for system prompt generation)
    lender_persona: str = "balanced"
    target_yield_pct: float = 10.0
    max_single_loan: float = 500_000.0
    total_capital: float = 5_000_000.0


class LoanvilleEnv:
    """Gymnasium-compatible RL environment for Loanville.

    Presents borrower dossiers as observations and accepts
    APPROVE/REJECT decisions as actions. Rewards are computed
    by the configured verifier rubric.

    This environment is designed for single-agent training where
    one LLM-lender makes sequential underwriting decisions against
    a stream of borrowers. Multi-agent (competitive) training can
    be implemented by running multiple LoanvilleEnv instances with
    shared borrower pools.

    Usage::

        env = LoanvilleEnv(EnvConfig(n_borrowers=10, seed=42))
        obs, info = env.reset()
        while True:
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        rubric: Optional[Rubric] = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.rubric = rubric or self._default_rubric()
        self._rng = random.Random(self.config.seed)

        # Episode state (populated on reset)
        self._borrowers: list[Borrower] = []
        self._current_idx: int = 0
        self._decisions: list[dict] = []
        self._outcomes: list[dict] = []
        self._rollout: Optional[Rollout] = None
        self._done: bool = True

        # Economics config
        if self.config.economics_config:
            eco_params = dict(self.config.economics_config)
            eco_params.setdefault("name", "custom")
            self._economics = EconomicsConfig(**{
                k: v for k, v in eco_params.items()
                if k in EconomicsConfig.__dataclass_fields__
            })
        else:
            self._economics = ECONOMICS_PRESETS.get(
                self.config.economics, EconomicsConfig()
            )

        # Cumulative tracking (reset each episode)
        self._total_deployed: float = 0.0
        self._total_interest: float = 0.0
        self._total_losses: float = 0.0
        self._total_fees: float = 0.0
        self._deals_won: int = 0
        self._deals_rejected: int = 0
        self._frauds_funded: int = 0
        self._defaults_count: int = 0

    def _default_rubric(self) -> Rubric:
        """Build default reward rubric."""
        rubric = Rubric()
        rubric.add(CreditQualityVerifier(), weight=1.0, name="credit_quality")
        return rubric

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment for a new episode.

        Returns (observation, info) for the first borrower.
        """
        if seed is not None:
            self.config.seed = seed
        self._rng = random.Random(self.config.seed)

        # Get borrowers from the data module
        # Map season-style mix names to data-module mix names
        mix_map = {
            "gentle": "easy",
            "realistic": "balanced",
            "adversarial": "hard",
            "stress": "stress",
        }
        data_mix = mix_map.get(self.config.season_mix, self.config.season_mix)
        if data_mix not in MIX_PRESETS:
            data_mix = "easy"
        all_borrowers = get_borrowers(data_mix, seed=self.config.seed)
        self._borrowers = all_borrowers[:self.config.n_borrowers]
        self._current_idx = 0
        self._decisions = []
        self._outcomes = []
        self._done = False

        # Reset accumulators
        self._total_deployed = 0.0
        self._total_interest = 0.0
        self._total_losses = 0.0
        self._total_fees = 0.0
        self._deals_won = 0
        self._deals_rejected = 0
        self._frauds_funded = 0
        self._defaults_count = 0

        # Initialize rollout capture
        from datetime import datetime, timezone
        self._rollout = Rollout(
            episode_type=self.config.episode_type,
            scenario=self.config.season_mix,
            seed=self.config.seed,
            config=self._config_dict(),
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        obs = self._make_observation(self._borrowers[0])
        info = {
            "episode_type": self.config.episode_type,
            "n_borrowers": len(self._borrowers),
            "borrower_idx": 0,
        }
        return obs, info

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Process one underwriting decision.

        Args:
            action: dict with keys:
                - decision: "APPROVE" | "REJECT"
                - term_sheet: {loan_amount, interest_rate, term_months} (if approved)
                - reasoning: str (optional)

        Returns:
            (observation, reward, done, truncated, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        borrower = self._borrowers[self._current_idx]
        decision_str = action.get("decision", "REJECT").upper()
        reasoning = action.get("reasoning", "")

        # Process the decision
        step_reward = 0.0
        step_outcome: dict[str, Any] = {}

        if decision_str == "APPROVE":
            ts = action.get("term_sheet", {})
            amount = float(ts.get("loan_amount", 0.0))
            rate = float(ts.get("interest_rate", 0.0))
            term = int(ts.get("term_months", 24))

            if amount > 0 and rate > 0:
                # Compute loan payoff given ground truth
                payoff = compute_loan_payoff(
                    principal=amount,
                    interest_rate=rate,
                    term_months=term,
                    true_outcome=borrower.true_outcome,
                    months_before_default=borrower.months_before_default,
                    economics=self._economics,
                )

                self._total_deployed += amount
                self._total_interest += payoff["interest_earned"]
                self._total_fees += payoff.get("fees_earned", 0.0)
                self._total_losses += payoff["principal_lost"]
                self._deals_won += 1
                if borrower.true_outcome == "fraud":
                    self._frauds_funded += 1
                if borrower.true_outcome in ("bad", "fraud"):
                    self._defaults_count += 1

                step_outcome = payoff
                if self.config.dense_rewards:
                    step_reward = payoff["net_profit"] / max(1.0, amount)
                    step_reward = max(-1.0, min(1.0, step_reward))
            else:
                # Invalid approval (no valid terms) → treat as reject
                decision_str = "REJECT"
                self._deals_rejected += 1
        else:
            self._deals_rejected += 1

        # Record the decision
        self._decisions.append({
            "borrower_id": borrower.id,
            "decision": decision_str,
            "reasoning": reasoning,
            "term_sheet": action.get("term_sheet"),
        })
        self._outcomes.append(step_outcome)

        # Record step in rollout
        if self._rollout is not None:
            obs_for_record = self._make_observation(borrower)
            self._rollout.add_step(
                observation=obs_for_record,
                action=action,
                reward=step_reward * self.config.reward_scale,
                reward_breakdown=step_outcome if step_outcome else {},
                ground_truth={
                    "true_outcome": borrower.true_outcome,
                    "months_before_default": borrower.months_before_default,
                },
            )

        # Advance to next borrower
        self._current_idx += 1
        done = self._current_idx >= len(self._borrowers)
        self._done = done

        # Build next observation (or empty if done)
        if done:
            obs = {}
            # Compute terminal reward via rubric
            terminal_state = self._build_verifier_state()
            terminal_result = self.rubric.score(terminal_state)
            terminal_reward = terminal_result.reward * self.config.reward_scale

            # Finalize rollout
            if self._rollout is not None:
                self._rollout.finalize(
                    terminal_reward=terminal_reward,
                    terminal_rewards=terminal_result.rewards,
                )

            info = {
                "terminal_reward": terminal_reward,
                "terminal_rewards": terminal_result.rewards,
                "rollout": self._rollout,
                "total_deployed": self._total_deployed,
                "total_interest": self._total_interest,
                "total_losses": self._total_losses,
                "deals_won": self._deals_won,
                "deals_rejected": self._deals_rejected,
                "frauds_funded": self._frauds_funded,
                "defaults_count": self._defaults_count,
            }
        else:
            obs = self._make_observation(self._borrowers[self._current_idx])
            info = {
                "borrower_idx": self._current_idx,
                "borrowers_remaining": len(self._borrowers) - self._current_idx,
            }

        return obs, step_reward * self.config.reward_scale, done, False, info

    def _make_observation(self, borrower: Borrower) -> dict[str, Any]:
        """Build the observation dict for a borrower.

        This is what the LLM agent sees. It includes the borrower's
        financial dossier and the lender's current portfolio state.
        Ground truth (true_outcome) is NOT included.
        """
        d = borrower.dossier
        obs: dict[str, Any] = {
            "borrower_id": borrower.id,
            "company_name": d.company_name,
            "sector": d.sector,
            "years_in_business": d.years_in_business,
            "employee_count": d.employee_count,
            "annual_revenue": d.annual_revenue,
            "annual_expenses": d.annual_expenses,
            "net_income": d.net_income,
            "loan_request_amount": d.loan_request_amount,
            "loan_purpose": d.loan_purpose,
            "narrative": d.narrative,
        }

        # Quarterly income
        if d.quarterly_income and self.config.data_mode != "aggregate_only":
            obs["quarterly_income"] = [
                {
                    "quarter": q.quarter,
                    "revenue": q.revenue,
                    "expenses": q.expenses,
                    "net_income": q.net_income,
                    "net_margin_pct": q.net_margin_pct,
                }
                for q in d.quarterly_income
            ]

        # Bank statements (summary or full depending on data_mode)
        if d.bank_statements and self.config.data_mode in ("full", "statements_inline"):
            obs["bank_statements"] = [
                {
                    "month": s.month,
                    "opening_balance": s.opening_balance,
                    "ending_balance": s.ending_balance,
                    "total_deposits": s.total_deposits,
                    "total_withdrawals": s.total_withdrawals,
                }
                for s in d.bank_statements
            ]

        # Portfolio context (what the lender knows about their own state)
        obs["portfolio_context"] = {
            "total_capital": self.config.total_capital,
            "deployed_capital": self._total_deployed,
            "available_capital": self.config.total_capital - self._total_deployed,
            "deals_approved": self._deals_won,
            "deals_rejected": self._deals_rejected,
            "target_yield_pct": self.config.target_yield_pct,
            "max_single_loan": self.config.max_single_loan,
        }

        return obs

    def _build_verifier_state(self) -> dict[str, Any]:
        """Build the state dict for the terminal verifier."""
        return {
            "total_interest": self._total_interest,
            "total_losses": self._total_losses,
            "total_fees": self._total_fees,
            "total_deployed": self._total_deployed,
            "available_capital": self.config.total_capital,
            "frauds_funded": self._frauds_funded,
            "defaults_count": self._defaults_count,
            "deals_won": self._deals_won,
            "deals_rejected": self._deals_rejected,
            "economics": asdict(self._economics),
        }

    def _config_dict(self) -> dict[str, Any]:
        """Serialize config for rollout metadata."""
        return {
            "episode_type": self.config.episode_type,
            "n_borrowers": self.config.n_borrowers,
            "season_mix": self.config.season_mix,
            "economics": self.config.economics,
            "data_mode": self.config.data_mode,
            "seed": self.config.seed,
            "dense_rewards": self.config.dense_rewards,
            "reward_scale": self.config.reward_scale,
        }

    @property
    def rollout(self) -> Optional[Rollout]:
        """Access the current episode's rollout (available after episode ends)."""
        return self._rollout

    @property
    def is_done(self) -> bool:
        return self._done
