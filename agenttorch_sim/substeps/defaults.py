"""
Substep 4: Differentiable Default Process

Replaces Loanville2's deterministic good/bad/fraud outcomes with a differentiable
hazard model. Each performing loan has a time-dependent default probability:

  hazard = sigmoid((months_elapsed - months_to_default) * steepness) * default_prob
  fraud_hazard = fraud_prob * sigmoid(months_elapsed * fraud_steepness)
  combined = 1 - (1 - hazard) * (1 - fraud_hazard)

Defaults are sampled using a differentiable Bernoulli (straight-through estimator
for gradient flow). On default, recovery and losses are computed per Loanville2's
economics.
"""

import re
import torch
import torch.nn as nn

try:
    from agent_torch.core.registry import Registry
    from agent_torch.core.substep import SubstepTransition
    from agent_torch.core.helpers import get_by_path
    from agent_torch.core.distributions import StraightThroughBernoulli as DiffBernoulli
except ImportError:
    from agenttorch_sim import Registry
    SubstepTransition = nn.Module
    DiffBernoulli = None

    def get_by_path(state, path):
        for key in path:
            state = state[key]
        return state


def _get(state, var_path):
    return get_by_path(state, re.split("/", var_path))


def _differentiable_bernoulli(prob):
    """Sample from Bernoulli with straight-through gradient estimator.

    Falls back to regular torch.bernoulli if AgentTorch distributions unavailable.
    """
    if DiffBernoulli is not None:
        return DiffBernoulli.apply(prob)
    # Straight-through estimator: forward = hard sample, backward = pass gradient through
    sample = torch.bernoulli(prob.clamp(0.0, 1.0))
    return sample + prob - prob.detach()  # straight-through trick


@Registry.register_substep("default_process", "transition")
class DefaultProcess(SubstepTransition if not isinstance(SubstepTransition, type(nn.Module)) else nn.Module):
    """Differentiable default and fraud detection process.

    For each performing loan:
      1. Compute months elapsed since origination
      2. Compute time-dependent default hazard
      3. Compute fraud hazard (fires early)
      4. Sample default event (differentiable Bernoulli)
      5. On default: compute recovery, workout cost, principal lost
      6. Update lender accumulators

    Learnable parameters:
      - hazard_steepness: controls how sharply the default hazard activates
      - fraud_steepness: controls how quickly fraud hazard activates
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

        self.hazard_steepness = _param("hazard_steepness", 5.0)
        self.fraud_steepness = _param("fraud_steepness", 10.0)

    def forward(self, state, action=None):
        # Read state
        outstanding_debt = _get(state, self.input_variables.get(
            "outstanding_debt", "agents/borrowers/outstanding_debt")).clone()
        months_remaining = _get(state, self.input_variables.get(
            "months_remaining", "agents/borrowers/months_remaining"))
        loan_status = _get(state, self.input_variables.get(
            "loan_status", "agents/borrowers/loan_status")).clone()
        assigned_lender = _get(state, self.input_variables.get(
            "assigned_lender", "agents/borrowers/assigned_lender"))
        default_prob = _get(state, self.input_variables.get(
            "default_prob", "agents/borrowers/default_prob"))
        months_to_default = _get(state, self.input_variables.get(
            "months_to_default", "agents/borrowers/months_to_default"))
        fraud_prob = _get(state, self.input_variables.get(
            "fraud_prob", "agents/borrowers/fraud_prob"))
        origination_month = _get(state, self.input_variables.get(
            "loan_origination_month", "agents/borrowers/loan_origination_month"))

        total_principal_lost = _get(state, self.input_variables.get(
            "total_principal_lost", "agents/lenders/total_principal_lost")).clone()
        total_recovery = _get(state, self.input_variables.get(
            "total_recovery", "agents/lenders/total_recovery")).clone()
        total_workout_cost = _get(state, self.input_variables.get(
            "total_workout_cost", "agents/lenders/total_workout_cost")).clone()
        defaults_count = _get(state, self.input_variables.get(
            "defaults_count", "agents/lenders/defaults_count")).clone()
        fraud_count = _get(state, self.input_variables.get(
            "fraud_count", "agents/lenders/fraud_count")).clone()

        recovery_rate_bad = _get(state, self.input_variables.get(
            "recovery_rate_bad", "environment/recovery_rate_bad"))
        recovery_rate_fraud = _get(state, self.input_variables.get(
            "recovery_rate_fraud", "environment/recovery_rate_fraud"))
        workout_cost_rate = _get(state, self.input_variables.get(
            "workout_cost_rate", "environment/workout_cost_rate"))
        current_month = _get(state, self.input_variables.get(
            "current_month", "environment/current_month"))
        total_defaults_this_step = torch.zeros(1, device=loan_status.device)

        # Only process performing loans
        performing = (loan_status == 1).float()  # (N, 1)

        # Months elapsed since origination
        months_elapsed = (current_month.float() - origination_month).clamp(min=0.0)  # (N, 1)

        # --- Time-dependent default hazard ---
        # Hazard increases as months_elapsed approaches months_to_default
        hazard = torch.sigmoid(
            (months_elapsed - months_to_default) * self.hazard_steepness
        ) * default_prob  # (N, 1)

        # --- Fraud hazard (activates early) ---
        fraud_hazard = fraud_prob * torch.sigmoid(
            months_elapsed * self.fraud_steepness
        )  # (N, 1)

        # --- Combined hazard ---
        combined_hazard = 1.0 - (1.0 - hazard) * (1.0 - fraud_hazard)
        combined_hazard = combined_hazard * performing  # only for performing loans
        combined_hazard = combined_hazard.clamp(0.0, 1.0)

        # --- Sample default events ---
        did_default = _differentiable_bernoulli(combined_hazard)  # (N, 1)

        # --- Compute recovery and losses for defaulted loans ---
        # Determine if default is fraud-type (fraud_prob > 0.5)
        is_fraud = (fraud_prob > 0.5).float()  # (N, 1)
        recovery_rate = is_fraud * recovery_rate_fraud + (1.0 - is_fraud) * recovery_rate_bad
        recovery = outstanding_debt * recovery_rate * did_default
        workout_cost = outstanding_debt * workout_cost_rate * did_default
        principal_lost = (outstanding_debt - recovery) * did_default
        principal_lost = principal_lost.clamp(min=0.0)

        # --- Update borrower state ---
        # Defaulted loans: status -> 2, debt -> 0
        loan_status = torch.where(
            did_default > 0.5,
            torch.full_like(loan_status, 2.0),
            loan_status,
        )
        outstanding_debt = outstanding_debt * (1.0 - did_default)

        # --- Aggregate to lenders ---
        N = loan_status.shape[0]
        M = total_principal_lost.shape[0]
        lender_idx = assigned_lender.squeeze(-1).long()

        new_defaults_total = 0.0
        for m in range(M):
            defaulted_for_m = (did_default.squeeze(-1) > 0.5) & (lender_idx == m)
            if defaulted_for_m.any():
                total_principal_lost[m] += principal_lost[defaulted_for_m].sum()
                total_recovery[m] += recovery[defaulted_for_m].sum()
                total_workout_cost[m] += workout_cost[defaulted_for_m].sum()
                n_defaults = defaulted_for_m.sum().float()
                defaults_count[m] += n_defaults
                new_defaults_total += n_defaults.item()

                # Count fraud defaults
                fraud_defaults = defaulted_for_m & (is_fraud.squeeze(-1) > 0.5)
                fraud_count[m] += fraud_defaults.sum().float()

        total_defaults_this_step = torch.tensor([new_defaults_total], device=loan_status.device)

        # Pack outputs
        keys = self.output_variables
        values = [
            outstanding_debt, loan_status,
            total_principal_lost, total_recovery, total_workout_cost,
            defaults_count, fraud_count,
            total_defaults_this_step,
        ]
        return {k: v for k, v in zip(keys, values)}
