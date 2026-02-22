"""
Champion / Challenger operating model.

- Always have a frozen champion policy
- Every change becomes a challenger
- Challengers must beat champion on benchmark gates, then compete in Elo
- Promote only if it wins in both worlds

State is stored in runs/champion.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .run_logger import RunLogger
from .scorecard import Scorecard, score_runs


CHAMPION_FILE = "champion.json"


@dataclass
class PolicyRecord:
    """A record of a policy with its aggregate performance."""
    policy_id: str
    model: str
    promoted_at: str = ""
    avg_score: float = 0.0
    n_runs: int = 0
    gates_pass_rate: float = 0.0
    decision_acc: float = 0.0
    elo_profit: float = 1500.0
    elo_credit: float = 1500.0
    elo_dealshare: float = 1500.0
    notes: str = ""


@dataclass
class ChampionState:
    """Persistent state for champion/challenger tracking."""
    champion: Optional[PolicyRecord] = None
    challengers: list[PolicyRecord] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # promotion log

    def to_dict(self) -> dict:
        return {
            "champion": asdict(self.champion) if self.champion else None,
            "challengers": [asdict(c) for c in self.challengers],
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChampionState:
        state = cls()
        if data.get("champion"):
            state.champion = PolicyRecord(**{
                k: v for k, v in data["champion"].items()
                if k in PolicyRecord.__dataclass_fields__
            })
        state.challengers = [
            PolicyRecord(**{k: v for k, v in c.items()
                           if k in PolicyRecord.__dataclass_fields__})
            for c in data.get("challengers", [])
        ]
        state.history = data.get("history", [])
        return state


class ChampionTracker:
    """Manages the champion/challenger operating model."""

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.base_dir / CHAMPION_FILE
        self.state = self._load_state()

    def _load_state(self) -> ChampionState:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            return ChampionState.from_dict(data)
        return ChampionState()

    def _save_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2, default=str)
        )

    def get_champion(self) -> Optional[PolicyRecord]:
        return self.state.champion

    def register_challenger(
        self,
        policy_id: str,
        model: str,
        notes: str = "",
    ) -> PolicyRecord:
        """Register a new challenger policy."""
        record = PolicyRecord(
            policy_id=policy_id,
            model=model,
            notes=notes,
        )
        # Remove existing if same policy_id
        self.state.challengers = [
            c for c in self.state.challengers
            if c.policy_id != policy_id
        ]
        self.state.challengers.append(record)
        self._save_state()
        return record

    def update_scores(
        self,
        policy_id: str,
        scorecards: list[Scorecard],
        elo_profit: float = 1500.0,
        elo_credit: float = 1500.0,
        elo_dealshare: float = 1500.0,
    ) -> Optional[PolicyRecord]:
        """Update a policy's aggregate scores from scorecards."""
        if not scorecards:
            return None

        avg_score = sum(c.overall_score for c in scorecards) / len(scorecards)
        gates_pass = sum(1 for c in scorecards if c.gates.passed) / len(scorecards)
        dec_acc = sum(c.uw_quality.decision_acc for c in scorecards) / len(scorecards)

        # Find the record (champion or challenger)
        record = None
        if self.state.champion and self.state.champion.policy_id == policy_id:
            record = self.state.champion
        else:
            for c in self.state.challengers:
                if c.policy_id == policy_id:
                    record = c
                    break

        if record is None:
            return None

        record.avg_score = avg_score
        record.n_runs = len(scorecards)
        record.gates_pass_rate = gates_pass
        record.decision_acc = dec_acc
        record.elo_profit = elo_profit
        record.elo_credit = elo_credit
        record.elo_dealshare = elo_dealshare

        self._save_state()
        return record

    def promote(self, policy_id: str, reason: str = "") -> bool:
        """Promote a challenger to champion.

        Returns True if promotion succeeded.
        """
        # Find the challenger
        challenger = None
        for c in self.state.challengers:
            if c.policy_id == policy_id:
                challenger = c
                break

        if challenger is None:
            return False

        # Record promotion
        now = datetime.now(timezone.utc).isoformat()
        promotion_event = {
            "timestamp": now,
            "new_champion": policy_id,
            "old_champion": self.state.champion.policy_id if self.state.champion else None,
            "reason": reason,
            "new_scores": {
                "avg_score": challenger.avg_score,
                "gates_pass_rate": challenger.gates_pass_rate,
                "decision_acc": challenger.decision_acc,
                "elo_profit": challenger.elo_profit,
            },
        }
        if self.state.champion:
            promotion_event["old_scores"] = {
                "avg_score": self.state.champion.avg_score,
                "gates_pass_rate": self.state.champion.gates_pass_rate,
                "decision_acc": self.state.champion.decision_acc,
                "elo_profit": self.state.champion.elo_profit,
            }

        self.state.history.append(promotion_event)

        # Demote current champion to challenger
        if self.state.champion:
            self.state.challengers.append(self.state.champion)

        # Promote
        challenger.promoted_at = now
        self.state.champion = challenger
        self.state.challengers = [
            c for c in self.state.challengers
            if c.policy_id != policy_id
        ]

        self._save_state()
        return True

    def should_promote(self, challenger_id: str) -> tuple[bool, str]:
        """Evaluate whether a challenger should be promoted.

        Promotion criteria:
        1. Gates pass rate >= 95%
        2. Overall score > champion score (or no champion)
        3. Elo profit >= champion Elo profit (or close)
        4. At least 10 runs

        Returns (should_promote, reason).
        """
        challenger = None
        for c in self.state.challengers:
            if c.policy_id == challenger_id:
                challenger = c
                break

        if challenger is None:
            return False, "challenger not found"

        if challenger.n_runs < 10:
            return False, f"insufficient runs ({challenger.n_runs} < 10)"

        if challenger.gates_pass_rate < 0.95:
            return False, f"gates pass rate too low ({challenger.gates_pass_rate:.0%} < 95%)"

        if self.state.champion is None:
            return True, "no existing champion"

        champ = self.state.champion

        # Must beat champion on overall score
        if challenger.avg_score <= champ.avg_score:
            return False, (
                f"avg score {challenger.avg_score:.1f} <= "
                f"champion {champ.avg_score:.1f}"
            )

        # Must not lose badly on Elo
        elo_gap = challenger.elo_profit - champ.elo_profit
        if elo_gap < -50:
            return False, (
                f"Elo profit gap too large: "
                f"{challenger.elo_profit:.0f} vs {champ.elo_profit:.0f}"
            )

        return True, (
            f"beats champion: score {challenger.avg_score:.1f} vs "
            f"{champ.avg_score:.1f}, Elo gap {elo_gap:+.0f}"
        )

    def print_status(self) -> None:
        """Print current champion/challenger status."""
        print(f"\n{'=' * 60}")
        print(f"  CHAMPION / CHALLENGER STATUS")
        print(f"{'=' * 60}")

        if self.state.champion:
            c = self.state.champion
            print(f"\n  CHAMPION: {c.policy_id}")
            print(f"    Model: {c.model}")
            print(f"    Score: {c.avg_score:.1f} | Gates: {c.gates_pass_rate:.0%} | "
                  f"DecAcc: {c.decision_acc:.0%}")
            print(f"    Elo: P={c.elo_profit:.0f} C={c.elo_credit:.0f} "
                  f"D={c.elo_dealshare:.0f}")
            print(f"    Runs: {c.n_runs}")
            if c.promoted_at:
                print(f"    Promoted: {c.promoted_at}")
        else:
            print(f"\n  No champion set.")

        if self.state.challengers:
            print(f"\n  CHALLENGERS:")
            for ch in self.state.challengers:
                print(f"    {ch.policy_id} ({ch.model})")
                print(f"      Score: {ch.avg_score:.1f} | Gates: {ch.gates_pass_rate:.0%} | "
                      f"Runs: {ch.n_runs}")
        else:
            print(f"\n  No challengers registered.")

        if self.state.history:
            print(f"\n  PROMOTION HISTORY ({len(self.state.history)} promotions):")
            for h in self.state.history[-5:]:
                print(f"    {h['timestamp'][:10]}: {h.get('old_champion', 'none')} "
                      f"→ {h['new_champion']}")
                if h.get("reason"):
                    print(f"      Reason: {h['reason']}")

        print(f"{'=' * 60}")
