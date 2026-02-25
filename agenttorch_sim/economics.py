"""
Economics configuration for AgentTorch lending simulation.

Ports Loanville2's EconomicsConfig to tensor-friendly format. All parameters
are either YAML environment variables (global) or substep arguments (per-module).
"""

from dataclasses import dataclass, asdict


@dataclass
class TorchEconomicsConfig:
    """Mirror of Loanville2 EconomicsConfig with same defaults."""
    name: str = "balanced"
    risk_free_rate: float = 0.05
    sim_horizon_months: int = 24
    funding_rate: float = 0.04
    risk_lambda: float = 0.35
    min_deployment_ratio: float = 0.35
    volume_penalty_lambda: float = 0.8
    max_default_rate: float = 0.25
    min_roe_threshold: float = -0.08
    hard_constraint_base_pct: float = 5.0
    fraud_penalty_rate: float = 0.25
    recovery_rate_bad: float = 0.25
    recovery_rate_fraud: float = 0.02
    workout_cost_rate: float = 0.03
    concentration_penalty_rate: float = 0.05
    origination_fee_rate: float = 0.01
    servicing_cost_rate_annual: float = 0.003

    def to_env_overrides(self) -> dict:
        """Return dict of environment variable overrides for the YAML config."""
        return {
            "funding_rate": self.funding_rate,
            "risk_free_rate": self.risk_free_rate,
            "recovery_rate_bad": self.recovery_rate_bad,
            "recovery_rate_fraud": self.recovery_rate_fraud,
            "origination_fee_rate": self.origination_fee_rate,
            "workout_cost_rate": self.workout_cost_rate,
            "fraud_penalty_rate": self.fraud_penalty_rate,
            "servicing_cost_rate": self.servicing_cost_rate_annual,
        }

    def to_dict(self) -> dict:
        return asdict(self)


TORCH_ECONOMICS_PRESETS = {
    "balanced": TorchEconomicsConfig(name="balanced"),
    "aggressive": TorchEconomicsConfig(
        name="aggressive",
        risk_free_rate=0.03,
        funding_rate=0.025,
        risk_lambda=0.2,
        min_deployment_ratio=0.50,
        volume_penalty_lambda=1.5,
        max_default_rate=0.40,
        min_roe_threshold=-0.15,
        fraud_penalty_rate=0.15,
        recovery_rate_bad=0.18,
        recovery_rate_fraud=0.01,
        workout_cost_rate=0.04,
        origination_fee_rate=0.008,
        servicing_cost_rate_annual=0.0025,
    ),
    "conservative": TorchEconomicsConfig(
        name="conservative",
        risk_free_rate=0.07,
        funding_rate=0.06,
        risk_lambda=0.8,
        min_deployment_ratio=0.20,
        volume_penalty_lambda=0.3,
        max_default_rate=0.15,
        min_roe_threshold=-0.05,
        fraud_penalty_rate=0.40,
        concentration_penalty_rate=0.10,
        recovery_rate_bad=0.35,
        recovery_rate_fraud=0.03,
        workout_cost_rate=0.02,
        origination_fee_rate=0.0125,
        servicing_cost_rate_annual=0.0035,
    ),
}
