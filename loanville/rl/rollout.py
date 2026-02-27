"""
Rollout capture — trajectory recording for offline RL and GRPO.

Records the full interaction trajectory of a Loanville episode:
observations, actions, rewards, and metadata.  Rollouts are
JSON-serializable and can be exported for training frameworks
like verifiers/trl or converted to Harbor-format artifacts.

Design goals:
  - Append-only: steps are added incrementally during an episode
  - Immutable after finalization: ``finalize()`` freezes the rollout
  - Serializable: ``to_dict()`` / ``to_json()`` for storage
  - Compatible with verifiers framework's state dict convention
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class Step:
    """A single step in an RL rollout.

    Maps to one underwriting decision within a Loanville episode.
    The observation is the borrower dossier + portfolio state presented
    to the model; the action is the model's decision + term sheet;
    the reward is the per-step signal from the verifier.
    """
    step_idx: int
    # What the model saw
    observation: dict[str, Any] = field(default_factory=dict)
    # What the model did
    action: dict[str, Any] = field(default_factory=dict)
    # Reward signal (dense, per-step)
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    # Ground truth (hidden from model during inference)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    # Metadata
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Rollout:
    """A complete RL episode trajectory.

    Contains the full sequence of steps from a Loanville simulation
    (single match or one season), plus episode-level metadata and
    the terminal reward.

    Compatible with:
      - verifiers framework: can be converted to vf.State dicts
      - Harbor format: terminal reward written to reward.json
      - HuggingFace datasets: list-of-dicts serialization
    """
    rollout_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Episode identification
    episode_type: str = "single"  # "single" | "season"
    scenario: str = ""
    seed: int = 0
    # Agent identification
    agent_id: str = ""
    model: str = ""
    policy_id: str = ""
    # Steps
    steps: list[Step] = field(default_factory=list)
    # Terminal reward (episode-level, from verifier)
    terminal_reward: float = 0.0
    terminal_rewards: dict[str, float] = field(default_factory=dict)
    # Timing
    started_at: str = ""
    finished_at: str = ""
    # Episode config (for reproducibility)
    config: dict[str, Any] = field(default_factory=dict)
    # Whether the rollout has been finalized
    finalized: bool = False

    def add_step(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        reward: float = 0.0,
        reward_breakdown: Optional[dict[str, float]] = None,
        ground_truth: Optional[dict[str, Any]] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> Step:
        """Append a step to the rollout. Raises if already finalized."""
        if self.finalized:
            raise RuntimeError("Cannot add steps to a finalized rollout")

        step = Step(
            step_idx=len(self.steps),
            observation=observation,
            action=action,
            reward=reward,
            reward_breakdown=reward_breakdown or {},
            ground_truth=ground_truth or {},
            info=info or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.steps.append(step)
        return step

    def finalize(
        self,
        terminal_reward: float,
        terminal_rewards: Optional[dict[str, float]] = None,
    ) -> None:
        """Freeze the rollout with terminal reward signals."""
        self.terminal_reward = terminal_reward
        self.terminal_rewards = terminal_rewards or {}
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.finalized = True

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def cumulative_reward(self) -> float:
        """Sum of per-step rewards (dense signal)."""
        return sum(s.reward for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "rollout_id": self.rollout_id,
            "episode_type": self.episode_type,
            "scenario": self.scenario,
            "seed": self.seed,
            "agent_id": self.agent_id,
            "model": self.model,
            "policy_id": self.policy_id,
            "n_steps": self.n_steps,
            "steps": [s.to_dict() for s in self.steps],
            "terminal_reward": self.terminal_reward,
            "terminal_rewards": self.terminal_rewards,
            "cumulative_reward": self.cumulative_reward,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "finalized": self.finalized,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> Rollout:
        rollout = cls(
            rollout_id=data.get("rollout_id", str(uuid.uuid4())),
            episode_type=data.get("episode_type", "single"),
            scenario=data.get("scenario", ""),
            seed=data.get("seed", 0),
            agent_id=data.get("agent_id", ""),
            model=data.get("model", ""),
            policy_id=data.get("policy_id", ""),
            terminal_reward=data.get("terminal_reward", 0.0),
            terminal_rewards=data.get("terminal_rewards", {}),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            config=data.get("config", {}),
            finalized=data.get("finalized", False),
        )
        for s in data.get("steps", []):
            step = Step(
                step_idx=s.get("step_idx", 0),
                observation=s.get("observation", {}),
                action=s.get("action", {}),
                reward=s.get("reward", 0.0),
                reward_breakdown=s.get("reward_breakdown", {}),
                ground_truth=s.get("ground_truth", {}),
                info=s.get("info", {}),
                timestamp=s.get("timestamp", ""),
            )
            rollout.steps.append(step)
        return rollout

    def to_harbor_reward(self) -> dict:
        """Export terminal reward in Harbor reward.json format.

        Harbor verifiers write either a plain float to reward.txt
        or a JSON dict to reward.json.  This produces the JSON variant.
        """
        return {
            "reward": self.terminal_reward,
            **self.terminal_rewards,
        }

    def to_verifier_states(self) -> list[dict[str, Any]]:
        """Convert to verifiers-framework state dicts.

        Each step becomes a state dict that can be passed to
        ``vf.Rubric.score_rollout()`` or similar.
        """
        states = []
        for step in self.steps:
            state: dict[str, Any] = {
                "step_idx": step.step_idx,
                "observation": step.observation,
                "action": step.action,
                "reward": step.reward,
                "info": step.info,
            }
            # Merge observation keys into top-level for verifier compatibility
            state.update(step.observation)
            states.append(state)
        return states


class RolloutLogger:
    """Persist rollouts to disk for offline training.

    Writes rollouts as individual JSON files into a directory,
    with an index file for batch loading.
    """

    def __init__(self, output_dir: str | Path = "rollouts") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def log(self, rollout: Rollout) -> Path:
        """Write a rollout to disk. Returns the file path."""
        filename = f"{rollout.rollout_id}.json"
        path = self.output_dir / filename
        path.write_text(rollout.to_json())
        self._count += 1
        self._update_index()
        return path

    def _update_index(self) -> None:
        """Rebuild the index file listing all rollouts."""
        files = sorted(
            [f.name for f in self.output_dir.glob("*.json") if f.name != "index.json"]
        )
        index_path = self.output_dir / "index.json"
        index_path.write_text(json.dumps(files, indent=2))

    def load(self, rollout_id: str) -> Optional[Rollout]:
        """Load a single rollout by ID."""
        path = self.output_dir / f"{rollout_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return Rollout.from_dict(data)

    def load_all(self) -> list[Rollout]:
        """Load all rollouts from the output directory."""
        rollouts = []
        for path in sorted(self.output_dir.glob("*.json")):
            if path.name == "index.json":
                continue
            data = json.loads(path.read_text())
            rollouts.append(Rollout.from_dict(data))
        return rollouts

    @property
    def count(self) -> int:
        return self._count
