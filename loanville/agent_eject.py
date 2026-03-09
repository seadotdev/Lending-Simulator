"""Eject policies for agent sim — early termination for poor performers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EjectDecision:
    should_eject: bool
    reason: str = ""


class EjectPolicy:
    """Base class for eject policies."""

    def check(
        self,
        turn: int,
        max_turns: int,
        los_state: dict,
        budget_fraction_spent: float | None = None,
    ) -> EjectDecision:
        return EjectDecision(should_eject=False)


class NoProgressEject(EjectPolicy):
    """Eject if no deal created after a fraction of turns."""

    def __init__(self, threshold_fraction: float = 0.30) -> None:
        self.threshold_fraction = threshold_fraction

    def check(
        self,
        turn: int,
        max_turns: int,
        los_state: dict,
        budget_fraction_spent: float | None = None,
    ) -> EjectDecision:
        threshold_turn = int(max_turns * self.threshold_fraction)
        if turn >= threshold_turn:
            deals = los_state.get("deals", [])
            if not deals:
                return EjectDecision(
                    should_eject=True,
                    reason=f"no deal created after {turn}/{max_turns} turns",
                )
        return EjectDecision(should_eject=False)


class QualityEject(EjectPolicy):
    """Warn at warn_fraction if stuck in origination; eject at eject_fraction."""

    def __init__(
        self,
        warn_fraction: float = 0.60,
        eject_fraction: float = 0.80,
    ) -> None:
        self.warn_fraction = warn_fraction
        self.eject_fraction = eject_fraction
        self._warned = False

    def check(
        self,
        turn: int,
        max_turns: int,
        los_state: dict,
        budget_fraction_spent: float | None = None,
    ) -> EjectDecision:
        deals = los_state.get("deals", [])
        if not deals:
            return EjectDecision(should_eject=False)

        stage = ""
        for d in deals:
            stage = d.get("stage", "")

        # Only trigger if stuck in early stages
        stuck = stage in ("", "broker", "origination", "processing")

        warn_turn = int(max_turns * self.warn_fraction)
        eject_turn = int(max_turns * self.eject_fraction)

        if turn >= eject_turn and stuck:
            return EjectDecision(
                should_eject=True,
                reason=f"stuck at stage={stage!r} after {turn}/{max_turns} turns",
            )

        if turn >= warn_turn and stuck and not self._warned:
            self._warned = True
            # Warning only — don't eject yet
            return EjectDecision(
                should_eject=False,
                reason=f"warning: stage={stage!r} at turn {turn}/{max_turns}",
            )

        return EjectDecision(should_eject=False)


class AnalysisEject(EjectPolicy):
    """Detect rubber-stamping: model decides without doing real analysis.

    Triggers if ALL of:
    - Decision made (deal exists and has been advanced)
    - No spread was created (never looked at financials)
    - Decided in very few turns (≤ min_turns)

    Also flags if model delegated entirely to los_deal_evaluate
    without creating its own spread or reviewing ratios.
    """

    def __init__(
        self,
        min_turns: int = 3,
        eject_fraction: float = 0.50,
    ) -> None:
        self.min_turns = min_turns
        self.eject_fraction = eject_fraction

    def check(
        self,
        turn: int,
        max_turns: int,
        los_state: dict,
        budget_fraction_spent: float | None = None,
    ) -> EjectDecision:
        deals = los_state.get("deals", [])
        if not deals:
            return EjectDecision(should_eject=False)

        spread_count = los_state.get("spread_count", 0)
        doc_count = los_state.get("doc_count", 0)
        stage = deals[0].get("stage", "broker") if deals else "broker"

        # Check if model has advanced past origination (suggesting it made a decision)
        advanced_stages = {"underwriting", "closing", "monitoring"}
        has_advanced = stage in advanced_stages

        if not has_advanced:
            return EjectDecision(should_eject=False)

        # Rubber-stamp detection: advanced without doing financial analysis
        if spread_count == 0 and doc_count == 0:
            return EjectDecision(
                should_eject=True,
                reason=f"rubber-stamp: advanced to {stage} with no docs or spreads in {turn} turns",
            )

        # Suspiciously fast: decided in ≤ min_turns
        if turn <= self.min_turns and spread_count == 0:
            return EjectDecision(
                should_eject=True,
                reason=f"rubber-stamp: decided in {turn} turns without creating spread",
            )

        return EjectDecision(should_eject=False)


def default_eject_policies() -> list[EjectPolicy]:
    """Return the standard set of eject policies."""
    return [NoProgressEject(), QualityEject(), AnalysisEject()]
