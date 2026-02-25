"""
Substep 1: Underwriting (Rule-Based Credit Model)

Each lender evaluates each applicant using a differentiable credit scoring model
based on Loanville2's heuristic baseline: DSCR, margin, and leverage checks.

The output is a (num_borrowers, num_lenders) approval probability matrix and
a (num_borrowers, num_lenders) offered rate matrix.

Phase 2 will replace this with LLM archetype-driven decisions.
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


@Registry.register_substep("credit_evaluate", "policy")
class CreditEvaluate(SubstepAction if not isinstance(SubstepAction, type(nn.Module)) else nn.Module):
    """Differentiable credit scoring model.

    For each (borrower, lender) pair, computes:
      - DSCR score: soft threshold on debt service coverage ratio
      - Margin score: soft threshold on net margin
      - Leverage score: soft threshold on loan-to-income ratio
      - Combined approval probability
      - Risk-adjusted offered interest rate

    Produces two matrices stored on the state for adjudication to consume:
      approval_matrix: (num_borrowers, num_lenders) — probability of approval
      rate_matrix: (num_borrowers, num_lenders) — offered annual rate
    """

    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None, **kwargs):
        super().__init__()
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or ["approval_prob", "offered_rate"]
        self.args = arguments or {}

        # Learnable thresholds (initialized from Loanville2 heuristic baseline)
        def _param(key, default):
            v = self.args.get(key, default)
            if isinstance(v, dict):
                v = v.get("value", default)
            return nn.Parameter(torch.tensor(float(v)))

        self.margin_threshold = _param("margin_threshold", 0.10)
        self.leverage_cap = _param("leverage_cap", 1.5)
        self.dscr_floor = _param("dscr_floor", 1.25)

    def forward(self, state, observation=None):
        # Borrower features: (N, 1)
        net_income = _get(state, self.input_variables.get("net_income", "agents/borrowers/net_income"))
        net_margin = _get(state, self.input_variables.get("net_margin", "agents/borrowers/net_margin"))
        loan_request = _get(state, self.input_variables.get("loan_request", "agents/borrowers/loan_request"))
        loan_status = _get(state, self.input_variables.get("loan_status", "agents/borrowers/loan_status"))

        # Lender features: (M, 1)
        risk_tolerance = _get(state, self.input_variables.get("risk_tolerance", "agents/lenders/risk_tolerance"))
        target_yield = _get(state, self.input_variables.get("target_yield", "agents/lenders/target_yield"))
        funding_rate = _get(state, self.input_variables.get("funding_rate", "environment/funding_rate"))

        N = net_income.shape[0]  # num borrowers
        M = risk_tolerance.shape[0]  # num lenders

        # Only evaluate borrowers with loan_status == 0 (applying)
        eligible_mask = (loan_status == 0).float()  # (N, 1)

        # --- Differentiable credit scores (all borrowers at once) ---

        # DSCR: assumes 24-month term, monthly rate ~= target_yield / 12
        # Approximate amortization factor for DSCR
        approx_monthly_payment = loan_request / 24.0  # simplified
        dscr = (net_income / 12.0) / (approx_monthly_payment + 1e-8)  # (N, 1)
        dscr_score = torch.sigmoid((dscr - self.dscr_floor) * 10.0)  # (N, 1)

        # Margin check: soft threshold
        margin_score = torch.sigmoid((net_margin - self.margin_threshold) * 20.0)  # (N, 1)

        # Leverage check: loan / annual net income < leverage_cap
        leverage_ratio = loan_request / (net_income + 1e-8)  # (N, 1)
        leverage_score = torch.sigmoid((self.leverage_cap - leverage_ratio) * 10.0)  # (N, 1)

        # Combined borrower quality score: (N, 1)
        borrower_quality = dscr_score * margin_score * leverage_score

        # --- Per-lender modulation ---
        # risk_tolerance shapes approval threshold: (N, M)
        # Broadcast: borrower_quality (N, 1) * risk_tolerance.T (1, M)
        approval_prob = borrower_quality * risk_tolerance.T  # (N, M)
        approval_prob = approval_prob * eligible_mask  # mask ineligible
        approval_prob = torch.clamp(approval_prob, 0.0, 1.0)

        # --- Offered rate: risk premium inversely proportional to quality ---
        # Higher quality = lower rate, higher risk tolerance = lower rate
        risk_premium = (1.0 - borrower_quality) * 5.0  # 0-5% risk premium
        # Each lender's base rate is their target yield
        # offered_rate = target_yield + (1 - quality) * spread
        offered_rate = target_yield.T + risk_premium  # (N, M)
        # Add some per-lender noise for competitive diversity
        offered_rate = offered_rate + funding_rate * 0.1  # small funding cost pass-through

        # Store matrices on state for adjudication to consume
        # We store them as environment variables (will be overwritten each step)
        output = {}
        if len(self.output_variables) > 0:
            output[self.output_variables[0]] = approval_prob
        if len(self.output_variables) > 1:
            output[self.output_variables[1]] = offered_rate

        # Also attach to state for cross-substep communication
        return output


@Registry.register_substep("store_evaluations", "transition")
class StoreEvaluations(SubstepTransition if not isinstance(SubstepTransition, type(nn.Module)) else nn.Module):
    """Pass-through transition for the underwriting substep.

    The actual evaluation results (approval_prob, offered_rate matrices) are
    communicated via the action profile to the adjudication substep.
    """

    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None, **kwargs):
        super().__init__()
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or ["loan_status"]

    def forward(self, state, action=None):
        loan_status = _get(state, self.input_variables.get("loan_status", "agents/borrowers/loan_status"))
        return {self.output_variables[0]: loan_status}
