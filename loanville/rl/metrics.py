"""
Composable reward aggregation metrics.

Mirrors Harbor's metrics architecture (BaseMetric with mean/max/min/sum)
but tailored for Loanville's multi-component reward signals.

These operate on batches of rollouts to produce aggregate statistics
for training monitoring, leaderboard integration, and reward analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

T = TypeVar("T", float, int)


class BaseMetric(ABC, Generic[T]):
    """Abstract base for reward aggregation metrics.

    Mirrors Harbor's ``BaseMetric`` interface: accepts a list of
    reward values (possibly None for failed/skipped episodes) and
    returns a dict of computed statistics.
    """

    @abstractmethod
    def compute(self, rewards: list[T | None]) -> dict[str, float]:
        """Compute metric from a batch of reward values."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class MeanMetric(BaseMetric[float]):
    """Average reward across episodes (ignoring None)."""

    def compute(self, rewards: list[float | None]) -> dict[str, float]:
        valid = [r for r in rewards if r is not None]
        if not valid:
            return {"mean": 0.0, "count": 0.0}
        return {
            "mean": sum(valid) / len(valid),
            "count": float(len(valid)),
        }


class MaxMetric(BaseMetric[float]):
    """Maximum reward across episodes."""

    def compute(self, rewards: list[float | None]) -> dict[str, float]:
        valid = [r for r in rewards if r is not None]
        if not valid:
            return {"max": 0.0, "count": 0.0}
        return {
            "max": max(valid),
            "count": float(len(valid)),
        }


class MinMetric(BaseMetric[float]):
    """Minimum reward across episodes."""

    def compute(self, rewards: list[float | None]) -> dict[str, float]:
        valid = [r for r in rewards if r is not None]
        if not valid:
            return {"min": 0.0, "count": 0.0}
        return {
            "min": min(valid),
            "count": float(len(valid)),
        }


class SumMetric(BaseMetric[float]):
    """Sum of rewards across episodes."""

    def compute(self, rewards: list[float | None]) -> dict[str, float]:
        valid = [r for r in rewards if r is not None]
        return {
            "sum": sum(valid),
            "count": float(len(valid)),
        }


class WeightedMetric(BaseMetric[float]):
    """Weighted combination of named reward components.

    Operates on a batch of reward dicts (not scalar values) and
    computes a weighted sum across named components.

    Usage::

        metric = WeightedMetric({
            "credit_quality": 0.7,
            "portfolio_mgmt": 0.2,
            "efficiency": 0.1,
        })
        result = metric.compute_from_dicts(reward_dicts)
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = dict(weights)

    def compute(self, rewards: list[float | None]) -> dict[str, float]:
        """Compute mean of scalar rewards (fallback for simple usage)."""
        valid = [r for r in rewards if r is not None]
        if not valid:
            return {"weighted_mean": 0.0, "count": 0.0}
        return {
            "weighted_mean": sum(valid) / len(valid),
            "count": float(len(valid)),
        }

    def compute_from_dicts(
        self,
        reward_dicts: list[dict[str, float] | None],
    ) -> dict[str, float]:
        """Compute weighted combination from reward breakdown dicts."""
        valid = [d for d in reward_dicts if d is not None]
        if not valid:
            return {"weighted_score": 0.0, "count": 0.0}

        scores = []
        per_component: dict[str, list[float]] = {k: [] for k in self._weights}

        for d in valid:
            weighted_sum = 0.0
            for key, weight in self._weights.items():
                val = d.get(key, 0.0)
                weighted_sum += val * weight
                per_component[key].append(val)
            scores.append(weighted_sum)

        result: dict[str, float] = {
            "weighted_score": sum(scores) / len(scores),
            "count": float(len(valid)),
        }
        # Add per-component averages
        for key, values in per_component.items():
            if values:
                result[f"{key}_mean"] = sum(values) / len(values)

        return result

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


class MetricSuite:
    """Run multiple metrics on a batch of rewards.

    Collects results from all registered metrics into a single dict.
    """

    def __init__(self) -> None:
        self._metrics: list[tuple[str, BaseMetric]] = []

    def add(self, name: str, metric: BaseMetric) -> None:
        self._metrics.append((name, metric))

    def compute(self, rewards: list[float | None]) -> dict[str, dict[str, float]]:
        """Run all metrics on scalar rewards."""
        return {name: metric.compute(rewards) for name, metric in self._metrics}

    def compute_summary(self, rewards: list[float | None]) -> dict[str, float]:
        """Flat dict with prefixed keys for easy logging."""
        result: dict[str, float] = {}
        for name, metric in self._metrics:
            for k, v in metric.compute(rewards).items():
                result[f"{name}/{k}"] = v
        return result
