"""
Substep 0: Loan Application

Borrowers without active loans generate demand based on their financials
and current market conditions. During contraction phases, fewer borrowers
apply; during expansion, more seek credit.

This substep only activates borrowers with loan_status == 0 (no active loan).
"""

import re
import torch
import torch.nn as nn

try:
    from agent_torch.core.registry import Registry
    from agent_torch.core.substep import SubstepAction, SubstepTransition
    from agent_torch.core.helpers import get_by_path
except ImportError:
    from agenttorch_sim import Registry
    SubstepAction = nn.Module
    SubstepTransition = nn.Module

    def get_by_path(state, path):
        for key in path:
            state = state[key]
        return state


def _get(state, var_path):
    return get_by_path(state, re.split("/", var_path))


@Registry.register_substep("generate_demand", "policy")
class GenerateDemand(SubstepAction if not isinstance(SubstepAction, type(nn.Module)) else nn.Module):
    """Borrowers decide whether to apply for a loan.

    Policy: demand = sigmoid((net_income / loan_request - market_contraction) * sensitivity)
    Only borrowers with loan_status == 0 generate demand.
    """

    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None, **kwargs):
        super().__init__()
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or ["applying"]
        self.args = arguments or {}
        # Rate sensitivity: how much market conditions affect demand
        rate_sensitivity = self.args.get("rate_sensitivity", 1.0)
        if isinstance(rate_sensitivity, dict):
            val = rate_sensitivity.get("value", 1.0)
            self.rate_sensitivity = nn.Parameter(torch.tensor(float(val)))
        elif isinstance(rate_sensitivity, (int, float)):
            self.rate_sensitivity = nn.Parameter(torch.tensor(float(rate_sensitivity)))
        else:
            self.rate_sensitivity = nn.Parameter(torch.tensor(1.0))

    def forward(self, state, observation=None):
        net_income = _get(state, self.input_variables.get("net_income", "agents/borrowers/net_income"))
        loan_request = _get(state, self.input_variables.get("loan_request", "agents/borrowers/loan_request"))
        loan_status = _get(state, self.input_variables.get("loan_status", "agents/borrowers/loan_status"))
        market_phase = _get(state, self.input_variables.get("market_cycle_phase", "environment/market_cycle_phase"))

        # Only borrowers without active loans can apply
        eligible = (loan_status == 0).float()

        # Demand based on income-to-request ratio and market conditions
        income_ratio = net_income / (loan_request + 1e-8)
        market_dampening = 1.0 - 0.5 * market_phase  # expansion=1.0, contraction=0.5

        demand = torch.sigmoid(
            (income_ratio - 0.5) * self.rate_sensitivity * market_dampening
        )

        # Mask out ineligible borrowers
        applying = demand * eligible

        output_key = self.output_variables[0] if self.output_variables else "applying"
        return {output_key: applying}


@Registry.register_substep("mark_applicants", "transition")
class MarkApplicants(SubstepTransition if not isinstance(SubstepTransition, type(nn.Module)) else nn.Module):
    """Transition: no state change needed in this substep.

    The application decision is passed to subsequent substeps via the action profile.
    Loan status remains 0 until adjudication books the loan.
    """

    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None, **kwargs):
        super().__init__()
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or ["loan_status"]

    def forward(self, state, action=None):
        loan_status = _get(state, self.input_variables.get("loan_status", "agents/borrowers/loan_status"))
        # Pass through unchanged — actual booking happens in adjudication
        return {self.output_variables[0]: loan_status}
