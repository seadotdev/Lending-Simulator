"""
Verifier protocol — pluggable reward functions for RL training.

When the ``verifiers`` package is installed (``pip install verifiers``),
this module integrates directly with ``vf.Rubric`` and the verifiers
framework's reward function protocol.

When ``verifiers`` is not installed, standalone equivalents are provided
so the reward logic can still be used for evaluation and offline scoring.

The domain-specific verifiers (CreditQualityVerifier, PortfolioVerifier,
GateVerifier) work identically in both modes — they are pure functions
over a state dict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

# ---------------------------------------------------------------------------
# Detect verifiers framework
# ---------------------------------------------------------------------------

try:
    import verifiers as vf

    HAS_VERIFIERS = True
except ImportError:
    vf = None  # type: ignore[assignment]
    HAS_VERIFIERS = False


# ---------------------------------------------------------------------------
# Reward result — works with or without verifiers
# ---------------------------------------------------------------------------

@dataclass
class RewardResult:
    """Structured reward output from a verifier.

    Compatible with Harbor's ``VerifierResult`` (``rewards: dict``)
    and the verifiers framework's scalar return convention.
    """
    reward: float = 0.0
    rewards: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "rewards": dict(self.rewards),
            "info": dict(self.info),
        }

    def to_harbor_json(self) -> dict:
        """Export in Harbor reward.json format."""
        return {"reward": self.reward, **self.rewards}


class VerifierFunc(Protocol):
    """Protocol for a single reward function.

    Compatible with both standalone usage and ``vf.Rubric.add_reward_func()``.
    """

    def __call__(self, state: dict[str, Any]) -> RewardResult: ...


# ---------------------------------------------------------------------------
# Rubric — uses vf.Rubric when available, standalone otherwise
# ---------------------------------------------------------------------------

class Rubric:
    """Compose multiple verifier functions with weights.

    When ``verifiers`` is installed, this wraps ``vf.Rubric`` and converts
    between RewardResult and the framework's float-based reward convention.
    When standalone, it provides equivalent weighted-sum logic.

    Usage::

        rubric = Rubric()
        rubric.add(CreditQualityVerifier(), weight=0.7, name="credit")
        rubric.add(PortfolioVerifier(), weight=0.2, name="portfolio")
        result = rubric.score(state)
    """

    def __init__(self) -> None:
        self._funcs: list[tuple[VerifierFunc, float, str]] = []
        self._vf_rubric: Any = None

        if HAS_VERIFIERS:
            self._vf_rubric = vf.Rubric()

    def add(
        self,
        func: VerifierFunc,
        weight: float = 1.0,
        name: str = "",
    ) -> None:
        label = name or getattr(func, "__name__", "") or func.__class__.__name__
        self._funcs.append((func, weight, label))

        # Register with vf.Rubric if available
        if self._vf_rubric is not None:
            # Wrap our VerifierFunc into the verifiers framework's convention:
            # async def reward_fn(state, **kwargs) -> float
            verifier = func

            async def _vf_reward(state: dict, **kwargs: Any) -> float:
                result = verifier(state)
                return result.reward

            _vf_reward.__name__ = label
            self._vf_rubric.add_reward_func(_vf_reward, weight=weight)

    def score(self, state: dict[str, Any]) -> RewardResult:
        """Run all verifiers and produce a weighted-sum reward."""
        total = 0.0
        rewards: dict[str, float] = {}
        info: dict[str, Any] = {}

        for func, weight, label in self._funcs:
            result = func(state)
            component = result.reward * weight
            total += component
            rewards[label] = result.reward
            rewards[f"{label}_weighted"] = component
            if result.info:
                info[label] = result.info

        return RewardResult(reward=total, rewards=rewards, info=info)

    @property
    def vf_rubric(self) -> Any:
        """Access the underlying ``vf.Rubric`` (None if verifiers not installed).

        Use this when you need to pass the rubric directly to verifiers
        framework components like ``Environment`` or ``GRPOTrainer``.
        """
        return self._vf_rubric

    @property
    def names(self) -> list[str]:
        return [label for _, _, label in self._funcs]

    @property
    def weights(self) -> list[float]:
        return [w for _, w, _ in self._funcs]


# ---------------------------------------------------------------------------
# Built-in verifiers — domain-specific reward functions for Loanville
# ---------------------------------------------------------------------------

class CreditQualityVerifier:
    """Reward based on RAROC-style credit quality.

    Computes a normalized score from the lender's P&L outcomes,
    incorporating interest earned, losses, fraud penalties, and
    funding costs.  This is the primary reward signal for RL training.

    Expected state keys:
      - total_interest, total_losses, total_fees, total_deployed: float
      - frauds_funded, defaults_count, deals_won: int
      - available_capital: float
      - economics: dict (optional EconomicsConfig fields)
    """

    def __call__(self, state: dict[str, Any]) -> RewardResult:
        interest = float(state.get("total_interest", 0.0))
        losses = float(state.get("total_losses", 0.0))
        fees = float(state.get("total_fees", 0.0))
        deployed = float(state.get("total_deployed", 0.0))
        available = float(state.get("available_capital", 0.0))
        frauds = int(state.get("frauds_funded", 0))
        defaults = int(state.get("defaults_count", 0))

        eco = state.get("economics", {})
        funding_rate = float(eco.get("funding_rate", 0.04))
        fraud_penalty_rate = float(eco.get("fraud_penalty_rate", 0.25))
        risk_free_rate = float(eco.get("risk_free_rate", 0.05))
        horizon_months = int(eco.get("sim_horizon_months", 24))

        horizon_years = horizon_months / 12.0

        # Gross P&L
        funding_cost = deployed * funding_rate * horizon_years
        net_pnl = interest + fees - losses - funding_cost

        # Fraud penalty (spread across funded deals)
        deals_won = max(1, int(state.get("deals_won", 1)))
        avg_deal_size = deployed / deals_won if deals_won else 0
        fraud_penalty = frauds * fraud_penalty_rate * avg_deal_size

        # Opportunity cost benchmark
        opportunity = available * risk_free_rate * horizon_years

        # RAROC: (net_pnl - fraud_penalty) / available_capital
        if available > 0:
            raroc = (net_pnl - fraud_penalty) / available
        else:
            raroc = 0.0

        # Normalize to [0, 1]: -20% -> 0.0, 0% -> 0.5, +20% -> 1.0
        normalized = max(0.0, min(1.0, (raroc + 0.20) / 0.40))

        return RewardResult(
            reward=normalized,
            rewards={
                "raroc": raroc,
                "net_pnl": net_pnl,
                "funding_cost": funding_cost,
                "fraud_penalty": fraud_penalty,
                "opportunity_cost": opportunity,
            },
            info={
                "deployed": deployed,
                "frauds_funded": frauds,
                "defaults_count": defaults,
            },
        )


class PortfolioVerifier:
    """Reward based on portfolio management quality.

    Evaluates capital deployment efficiency, sector concentration
    discipline, and error avoidance.

    Expected state keys:
      - avg_utilization: float (0-1)
      - concentration_violations: int
      - total_weeks: int
      - deals_won, deals_rejected, deals_errored: int
    """

    def __call__(self, state: dict[str, Any]) -> RewardResult:
        utilization = float(state.get("avg_utilization", 0.0))
        violations = int(state.get("concentration_violations", 0))
        total_weeks = max(1, int(state.get("total_weeks", 1)))
        deals_won = int(state.get("deals_won", 0))
        deals_rejected = int(state.get("deals_rejected", 0))
        deals_errored = int(state.get("deals_errored", 0))

        # Utilization: 0.5 is optimal
        util_score = max(0.0, 1.0 - 2.0 * abs(utilization - 0.5))

        # Concentration discipline
        concentration_score = max(0.0, 1.0 - violations / total_weeks)

        # Error penalty
        total_decisions = max(1, deals_won + deals_rejected + deals_errored)
        error_score = 1.0 - deals_errored / total_decisions

        score = 0.4 * util_score + 0.4 * concentration_score + 0.2 * error_score

        return RewardResult(
            reward=score,
            rewards={
                "utilization": util_score,
                "concentration_discipline": concentration_score,
                "error_rate": error_score,
            },
            info={
                "avg_utilization": utilization,
                "concentration_violations": violations,
                "deals_errored": deals_errored,
            },
        )


class GateVerifier:
    """Hard-gate verifier: 1.0 if all gates pass, 0.0 otherwise.

    Mirrors the scorecard's Layer A.  In RL, this acts as a constraint —
    episodes failing gates get zero reward.

    Expected state keys:
      - gate_results: list[dict] with {gate_id, passed}  OR
      - gates_passed: bool
    """

    def __call__(self, state: dict[str, Any]) -> RewardResult:
        if "gates_passed" in state:
            passed = bool(state["gates_passed"])
            return RewardResult(
                reward=1.0 if passed else 0.0,
                rewards={"gates_passed": 1.0 if passed else 0.0},
            )

        gate_results = state.get("gate_results", [])
        if not gate_results:
            return RewardResult(reward=1.0, rewards={"gates_passed": 1.0})

        failures = [g["gate_id"] for g in gate_results if not g.get("passed", True)]
        passed = len(failures) == 0

        return RewardResult(
            reward=1.0 if passed else 0.0,
            rewards={"gates_passed": 1.0 if passed else 0.0},
            info={"failures": failures} if failures else {},
        )


class CompoundVerifier:
    """Compose verifiers, merging reward dicts (for multi-objective RL).

    Unlike Rubric (weighted scalar), preserves all rewards in a flat dict.
    Useful for Harbor's reward.json format or multi-objective training.
    """

    def __init__(self) -> None:
        self._verifiers: list[tuple[str, VerifierFunc]] = []

    def add(self, name: str, verifier: VerifierFunc) -> None:
        self._verifiers.append((name, verifier))

    def verify(self, state: dict[str, Any]) -> RewardResult:
        all_rewards: dict[str, float] = {}
        all_info: dict[str, Any] = {}

        for name, verifier in self._verifiers:
            result = verifier(state)
            all_rewards[name] = result.reward
            for k, v in result.rewards.items():
                all_rewards[f"{name}/{k}"] = v
            if result.info:
                all_info[name] = result.info

        if self._verifiers:
            primary = sum(all_rewards[name] for name, _ in self._verifiers) / len(self._verifiers)
        else:
            primary = 0.0

        return RewardResult(reward=primary, rewards=all_rewards, info=all_info)

    def __call__(self, state: dict[str, Any]) -> RewardResult:
        return self.verify(state)


# ---------------------------------------------------------------------------
# Factory: build the default Loanville rubric
# ---------------------------------------------------------------------------

def build_default_rubric(
    credit_weight: float = 0.7,
    portfolio_weight: float = 0.2,
    gate_weight: float = 0.1,
) -> Rubric:
    """Build the standard Loanville RL rubric.

    Returns a Rubric with the three core verifiers at the recommended
    weights.  The rubric's ``vf_rubric`` property gives you the
    underlying ``vf.Rubric`` object when verifiers is installed.
    """
    rubric = Rubric()
    rubric.add(CreditQualityVerifier(), weight=credit_weight, name="credit_quality")
    rubric.add(PortfolioVerifier(), weight=portfolio_weight, name="portfolio_mgmt")
    rubric.add(GateVerifier(), weight=gate_weight, name="gates")
    return rubric
