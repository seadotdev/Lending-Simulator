"""
RL training integration for Loanville.

This package provides the interfaces and adapters needed to use Loanville
as a reinforcement learning environment.

Two integration paths:

1. **With verifiers framework** (``pip install verifiers[rl]``):
   - ``vf_adapter.LoanvilleVerifierEnv``: native vf.SingleTurnEnv subclass
   - ``vf_adapter.build_vf_rubric()``: native vf.Rubric with Loanville rewards
   - ``vf_adapter.build_loanville_dataset()``: HuggingFace Dataset of tasks
   - Use with ``vf.GRPOTrainer`` or ``vf.HarborEnv`` for RL training

2. **Standalone** (no extra dependencies):
   - ``verifier``: pluggable reward functions (CreditQuality, Portfolio, Gate)
   - ``rollout``: trajectory recording for offline RL / GRPO
   - ``metrics``: composable reward aggregation (mean, max, weighted)
   - ``tasks``: export scenarios as Harbor-format tasks (task.toml + test.sh)
   - ``env``: lightweight Gymnasium-style step/reset interface

Harbor task format is always available (it's just files, no install needed).
The verifiers adapter requires ``pip install verifiers[rl]``.
"""

# Always-available standalone components
from .verifier import (
    RewardResult,
    VerifierFunc,
    Rubric,
    CreditQualityVerifier,
    PortfolioVerifier,
    GateVerifier,
    CompoundVerifier,
    build_default_rubric,
    HAS_VERIFIERS,
)
from .rollout import (
    Step,
    Rollout,
    RolloutLogger,
)
from .env import (
    LoanvilleEnv,
    EnvConfig,
)
from .metrics import (
    BaseMetric,
    MeanMetric,
    MaxMetric,
    MinMetric,
    WeightedMetric,
    MetricSuite,
)
from .tasks import (
    TaskConfig,
    export_harbor_task,
    export_harbor_dataset,
)

__all__ = [
    # Feature detection
    "HAS_VERIFIERS",
    # Verifiers (standalone)
    "RewardResult",
    "VerifierFunc",
    "Rubric",
    "CreditQualityVerifier",
    "PortfolioVerifier",
    "GateVerifier",
    "CompoundVerifier",
    "build_default_rubric",
    # Rollouts
    "Step",
    "Rollout",
    "RolloutLogger",
    # Environment
    "LoanvilleEnv",
    "EnvConfig",
    # Metrics
    "BaseMetric",
    "MeanMetric",
    "MaxMetric",
    "MinMetric",
    "WeightedMetric",
    "MetricSuite",
    # Tasks
    "TaskConfig",
    "export_harbor_task",
    "export_harbor_dataset",
]


def __getattr__(name: str):
    """Lazy import for verifiers-framework components."""
    if name in ("LoanvilleVerifierEnv", "build_vf_rubric", "build_loanville_dataset"):
        from . import vf_adapter
        return getattr(vf_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
