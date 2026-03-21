"""Eject policies for agent sim — early termination for poor performers.

Loanville-specific policies built on agent-preflight's EjectPolicy base class.
Domain logic (LOS deal/stage checks) is passed via the ``context`` dict.
"""

from __future__ import annotations

from agent_preflight import (  # noqa: F401
    BudgetExceededEject,
    BudgetFractionEject,
    CompositeEject,
    EjectDecision,
    EjectPolicy,
    IdleTimeoutEject,
    default_eject_policies as library_default_eject_policies,
)
from typing import Any


class NoProgressEject(EjectPolicy):
    """Eject if no deal created after a fraction of max turns.

    Expects ``context`` to contain:
    - ``turn``: current turn number (1-based)
    - ``max_turns``: maximum turns allowed
    - ``los_state``: dict with ``deals`` list from LOS inspection
    """

    def __init__(self, threshold_fraction: float = 0.30) -> None:
        self.threshold_fraction = threshold_fraction

    def check(self, *, context: dict, **_kwargs: Any) -> EjectDecision:
        turn = context.get("turn", 0)
        max_turns = context.get("max_turns", 20)
        los_state = context.get("los_state", {})

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
    """Warn at warn_fraction if stuck in origination; eject at eject_fraction.

    Expects ``context`` to contain:
    - ``turn``: current turn number (1-based)
    - ``max_turns``: maximum turns allowed
    - ``los_state``: dict with ``deals`` list from LOS inspection
    """

    def __init__(
        self,
        warn_fraction: float = 0.60,
        eject_fraction: float = 0.80,
    ) -> None:
        self.warn_fraction = warn_fraction
        self.eject_fraction = eject_fraction
        self._warned = False

    def check(self, *, context: dict, **_kwargs: Any) -> EjectDecision:
        turn = context.get("turn", 0)
        max_turns = context.get("max_turns", 20)
        los_state = context.get("los_state", {})

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
    """Return the standard set of Loanville eject policies."""
    return [NoProgressEject(), QualityEject()]
