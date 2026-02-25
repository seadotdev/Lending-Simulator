"""
Substep 5: Market Update / Macro Dynamics

Updates environment state based on aggregate behavior:
  - Advances the month counter
  - Updates market cycle phase based on default rates
  - Adjusts lender risk tolerances based on cycle phase (emergent credit cycles)

This is the emergent dynamics layer — behavior that only appears at population scale.
"""

import re
import torch
import torch.nn as nn

try:
    from agent_torch.core.registry import Registry
    from agent_torch.core.substep import SubstepTransition
    from agent_torch.core.helpers import get_by_path
except ImportError:
    from agenttorch_sim import Registry
    SubstepTransition = nn.Module

    def get_by_path(state, path):
        for key in path:
            state = state[key]
        return state


def _get(state, var_path):
    return get_by_path(state, re.split("/", var_path))


@Registry.register_substep("market_dynamics", "transition")
class MarketDynamics(SubstepTransition if not isinstance(SubstepTransition, type(nn.Module)) else nn.Module):
    """Emergent credit cycle dynamics.

    Feedback loop:
      high defaults → market_cycle_phase increases (contraction)
      → lenders reduce risk_tolerance
      → fewer loans originated
      → fewer defaults
      → market_cycle_phase decreases (expansion)
      → lenders increase risk_tolerance
      → more loans → cycle repeats

    Learnable parameters:
      - cycle_alpha: sensitivity of cycle phase to default rate deviation
      - cycle_beta: how much cycle phase affects lender risk tolerance
      - target_default_rate: equilibrium default rate
    """

    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None, **kwargs):
        super().__init__()
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or []
        self.args = arguments or {}

        def _param(key, default):
            v = self.args.get(key, default)
            if isinstance(v, dict):
                v = v.get("value", default)
            return nn.Parameter(torch.tensor(float(v)))

        self.cycle_alpha = _param("cycle_alpha", 0.1)
        self.cycle_beta = _param("cycle_beta", 0.05)

        target = self.args.get("target_default_rate", 0.05)
        if isinstance(target, dict):
            target = target.get("value", 0.05)
        self.target_default_rate = float(target)

    def forward(self, state, action=None):
        current_month = _get(state, self.input_variables.get(
            "current_month", "environment/current_month")).clone()
        market_cycle_phase = _get(state, self.input_variables.get(
            "market_cycle_phase", "environment/market_cycle_phase")).clone()
        total_defaults_this_step = _get(state, self.input_variables.get(
            "total_defaults_this_step", "environment/total_defaults_this_step"))
        loan_status = _get(state, self.input_variables.get(
            "loan_status", "agents/borrowers/loan_status"))
        risk_tolerance = _get(state, self.input_variables.get(
            "risk_tolerance", "agents/lenders/risk_tolerance")).clone()

        # --- Advance month ---
        current_month = current_month + 1

        # --- Compute current default rate ---
        num_performing = (loan_status == 1).float().sum()
        num_defaults = total_defaults_this_step.sum()
        # Default rate this period (avoid division by zero)
        current_default_rate = num_defaults / (num_performing + num_defaults + 1e-8)

        # --- Update credit cycle phase ---
        # Phase increases when defaults exceed target, decreases when below
        deviation = current_default_rate - self.target_default_rate
        market_cycle_phase = market_cycle_phase + self.cycle_alpha * deviation
        market_cycle_phase = market_cycle_phase.clamp(0.0, 1.0)

        # --- Adjust lender risk tolerances ---
        # During contraction (high phase), lenders tighten
        # During expansion (low phase), lenders loosen
        # The adjustment is relative to each lender's current tolerance
        risk_adjustment = -self.cycle_beta * (market_cycle_phase - 0.5)
        risk_tolerance = (risk_tolerance + risk_adjustment).clamp(0.1, 0.95)

        # Pack outputs
        keys = self.output_variables
        values = [current_month, market_cycle_phase, risk_tolerance]
        return {k: v for k, v in zip(keys, values)}
