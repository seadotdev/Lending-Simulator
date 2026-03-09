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


def default_eject_policies() -> list[EjectPolicy]:
    """Return the standard set of eject policies."""
    return [NoProgressEject(), QualityEject()]
