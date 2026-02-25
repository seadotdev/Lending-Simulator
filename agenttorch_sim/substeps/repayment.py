"""
Substep 3: Monthly Repayment (Amortization)

All performing loans (loan_status == 1) make their monthly payment.
Interest goes to the lender, principal reduces the outstanding balance.
Loans that reach months_remaining == 0 are marked as repaid (status 3).

This is a vectorized port of Loanville2's amortization math from engine.py.
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


@Registry.register_substep("monthly_amortization", "transition")
class MonthlyAmortization(SubstepTransition if not isinstance(SubstepTransition, type(nn.Module)) else nn.Module):
    """Vectorized monthly amortization for all performing loans.

    For each loan with status == 1:
      interest_portion = outstanding_debt * (interest_rate / 100 / 12)
      principal_portion = monthly_payment - interest_portion
      outstanding_debt -= principal_portion
      months_remaining -= 1

    Lender accumulators updated:
      total_interest_earned += interest_portion
      total_funding_cost += outstanding_debt * funding_rate / 12

    If months_remaining reaches 0, loan_status -> 3 (repaid).
    """

    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None, **kwargs):
        super().__init__()
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or []

    def forward(self, state, action=None):
        # Read state
        outstanding_debt = _get(state, self.input_variables.get(
            "outstanding_debt", "agents/borrowers/outstanding_debt")).clone()
        monthly_payment = _get(state, self.input_variables.get(
            "monthly_payment", "agents/borrowers/monthly_payment"))
        months_remaining = _get(state, self.input_variables.get(
            "months_remaining", "agents/borrowers/months_remaining")).clone()
        interest_rate = _get(state, self.input_variables.get(
            "interest_rate", "agents/borrowers/interest_rate"))
        loan_status = _get(state, self.input_variables.get(
            "loan_status", "agents/borrowers/loan_status")).clone()
        assigned_lender = _get(state, self.input_variables.get(
            "assigned_lender", "agents/borrowers/assigned_lender"))

        total_interest = _get(state, self.input_variables.get(
            "total_interest_earned", "agents/lenders/total_interest_earned")).clone()
        total_funding = _get(state, self.input_variables.get(
            "total_funding_cost", "agents/lenders/total_funding_cost")).clone()
        funding_rate = _get(state, self.input_variables.get(
            "funding_rate", "environment/funding_rate"))

        # Mask: only performing loans
        performing = (loan_status == 1).float()  # (N, 1)

        # Monthly interest rate
        monthly_rate = interest_rate / 100.0 / 12.0  # (N, 1)

        # Interest and principal portions
        interest_portion = outstanding_debt * monthly_rate * performing  # (N, 1)
        principal_portion = (monthly_payment - interest_portion) * performing  # (N, 1)

        # Ensure principal portion doesn't exceed remaining balance
        principal_portion = torch.min(principal_portion, outstanding_debt * performing)

        # Update outstanding debt
        outstanding_debt = outstanding_debt - principal_portion

        # Decrement months remaining
        months_remaining = months_remaining - performing

        # Check for matured loans (months_remaining <= 0 and was performing)
        matured = performing * (months_remaining <= 0).float()  # (N, 1)
        # Mature: remaining debt goes to 0, status -> 3
        outstanding_debt = outstanding_debt * (1.0 - matured)
        loan_status = torch.where(
            matured > 0.5,
            torch.full_like(loan_status, 3.0),
            loan_status,
        )

        # Funding cost: cost of carrying the outstanding balance
        funding_cost = outstanding_debt * (funding_rate / 12.0) * performing

        # Aggregate interest and funding costs to lenders
        N = loan_status.shape[0]
        M = total_interest.shape[0]
        lender_idx = assigned_lender.squeeze(-1).long()  # (N,)

        for m in range(M):
            is_this_lender = (lender_idx == m) & (performing.squeeze(-1) > 0.5)
            if is_this_lender.any():
                total_interest[m] += interest_portion[is_this_lender].sum()
                total_funding[m] += funding_cost[is_this_lender].sum()

        # Pack outputs
        keys = self.output_variables
        values = [outstanding_debt, months_remaining, loan_status, total_interest, total_funding]
        return {k: v for k, v in zip(keys, values)}
