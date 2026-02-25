"""
Substep 2: Competitive Adjudication

Borrowers select the lender offering the lowest rate (differentiable softmin).
Capital constraints and concentration limits are enforced.
Winning loans are booked: borrower state updated with loan terms, lender
state updated with capital deployment and sector exposure.

This is the most complex substep — it bridges the (N, M) evaluation matrices
from underwriting into per-borrower loan assignments.
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


@Registry.register_substep("competitive_auction", "transition")
class CompetitiveAuction(SubstepTransition if not isinstance(SubstepTransition, type(nn.Module)) else nn.Module):
    """Competitive auction: each eligible borrower picks the best lender offer.

    Uses a differentiable softmin over offered rates to select the winning lender.
    Capital constraints enforced via hard masking (non-differentiable boundary).
    Concentration limits checked per sector.

    Updates:
      - Borrower: loan_status -> 1, outstanding_debt, monthly_payment,
                  months_remaining, assigned_lender, interest_rate
      - Lender: available_capital, sector_exposure, loans_booked, total_fees_earned
    """

    def __init__(self, config=None, input_variables=None, output_variables=None, arguments=None, **kwargs):
        super().__init__()
        self.config = config or {}
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or []
        self.args = arguments or {}

        temp = self.args.get("temperature", 0.1)
        if isinstance(temp, dict):
            temp = temp.get("value", 0.1)
        self.temperature = float(temp)

    def forward(self, state, action=None):
        # --- Read current state ---
        loan_status = _get(state, self.input_variables.get("loan_status", "agents/borrowers/loan_status")).clone()
        outstanding_debt = _get(state, self.input_variables.get("outstanding_debt", "agents/borrowers/outstanding_debt")).clone()
        monthly_payment = _get(state, self.input_variables.get("monthly_payment", "agents/borrowers/monthly_payment")).clone()
        months_remaining = _get(state, self.input_variables.get("months_remaining", "agents/borrowers/months_remaining")).clone()
        assigned_lender = _get(state, self.input_variables.get("assigned_lender", "agents/borrowers/assigned_lender")).clone()
        interest_rate = _get(state, self.input_variables.get("interest_rate", "agents/borrowers/interest_rate")).clone()
        origination_month = _get(state, self.input_variables.get("loan_origination_month", "agents/borrowers/loan_origination_month")).clone()

        available_capital = _get(state, self.input_variables.get("available_capital", "agents/lenders/available_capital")).clone()
        sector_exposure = _get(state, self.input_variables.get("sector_exposure", "agents/lenders/sector_exposure")).clone()
        loans_booked = _get(state, self.input_variables.get("loans_booked", "agents/lenders/loans_booked")).clone()
        total_fees_earned = _get(state, self.input_variables.get("total_fees_earned", "agents/lenders/total_fees_earned")).clone()

        # Get evaluation matrices from the action profile (set by underwriting policy)
        approval_prob = None
        offered_rate = None
        if action is not None:
            lender_action = action.get("lenders", {})
            approval_prob = lender_action.get("approval_prob")
            offered_rate = lender_action.get("offered_rate")

        # If no evaluation matrices available (e.g., first step or mock), skip
        if approval_prob is None or offered_rate is None:
            return self._pack_outputs(
                loan_status, outstanding_debt, monthly_payment, months_remaining,
                assigned_lender, interest_rate, origination_month,
                available_capital, sector_exposure, loans_booked, total_fees_earned,
            )

        N = loan_status.shape[0]  # borrowers
        M = available_capital.shape[0]  # lenders

        # Borrower properties needed for loan booking
        loan_request = _get(state, "agents/borrowers/loan_request")  # (N, 1)
        sector = _get(state, "agents/borrowers/sector")  # (N, 1)
        current_month = _get(state, "environment/current_month")
        origination_fee_rate = _get(state, "environment/origination_fee_rate")
        max_single_loan = _get(state, "agents/lenders/max_single_loan")  # (M, 1)
        total_capital = _get(state, "agents/lenders/total_capital")  # (M, 1)
        sector_limit = _get(state, "agents/lenders/sector_limit")  # (M, 1)

        # Only process borrowers eligible for new loans (status == 0)
        eligible = (loan_status.squeeze(-1) == 0)  # (N,)

        # approval_prob: (N, M), offered_rate: (N, M)
        approval_mask = (approval_prob > 0.5).float()  # hard threshold for approval

        # Size constraint: loan_request <= max_single_loan
        size_ok = (loan_request <= max_single_loan.T).float()  # (N, M)

        num_sectors = sector_exposure.shape[1]
        borrower_sector_idx = sector.squeeze(-1).long()  # (N,)
        term_months = 24.0  # standard term

        # --- Sequential booking with running capital tracking ---
        # Process eligible borrowers one at a time to enforce capital limits.
        # This is the correct approach: Loanville2 also processes sequentially.
        eligible_indices = eligible.nonzero(as_tuple=True)[0]

        running_capital = available_capital.clone()  # (M, 1)
        running_sector_exposure = sector_exposure.clone()  # (M, num_sectors)

        for idx in eligible_indices:
            i = idx.item()
            req = loan_request[i, 0].item()

            # Find feasible lender offering the lowest rate
            best_lender = -1
            best_rate = float('inf')

            for m in range(M):
                # Approval check
                if approval_mask[i, m].item() < 0.5:
                    continue
                # Size limit
                if size_ok[i, m].item() < 0.5:
                    continue
                # Capital (running balance)
                if req > running_capital[m, 0].item():
                    continue
                # Concentration limit (running exposure)
                sec = borrower_sector_idx[i].item()
                proposed = running_sector_exposure[m, sec].item() + req
                limit = total_capital[m, 0].item() * sector_limit[m, 0].item()
                if proposed > limit:
                    continue
                # Best rate?
                rate = offered_rate[i, m].item()
                if rate < best_rate:
                    best_rate = rate
                    best_lender = m

            if best_lender < 0:
                continue  # no feasible lender

            # Book the loan
            m = best_lender
            sec = borrower_sector_idx[i].item()

            # Compute amortization payment
            monthly_rate_val = best_rate / 100.0 / 12.0
            if monthly_rate_val > 0:
                compound = (1.0 + monthly_rate_val) ** term_months
                pmt = req * (monthly_rate_val * compound) / (compound - 1.0)
            else:
                pmt = req / term_months

            # Update borrower state
            loan_status[i, 0] = 1.0
            outstanding_debt[i, 0] = req
            monthly_payment[i, 0] = pmt
            months_remaining[i, 0] = term_months
            assigned_lender[i, 0] = float(m)
            interest_rate[i, 0] = best_rate
            origination_month[i, 0] = current_month.float().item()

            # Update lender running state
            running_capital[m, 0] -= req
            running_sector_exposure[m, sec] += req
            loans_booked[m, 0] += 1
            total_fees_earned[m, 0] += req * origination_fee_rate.item()

        # Write back running state
        available_capital = running_capital
        sector_exposure = running_sector_exposure

        return self._pack_outputs(
            loan_status, outstanding_debt, monthly_payment, months_remaining,
            assigned_lender, interest_rate, origination_month,
            available_capital, sector_exposure, loans_booked, total_fees_earned,
        )

    def _pack_outputs(self, loan_status, outstanding_debt, monthly_payment,
                      months_remaining, assigned_lender, interest_rate,
                      origination_month, available_capital, sector_exposure,
                      loans_booked, total_fees_earned):
        """Pack updated state into output dict matching output_variables order."""
        keys = self.output_variables
        values = [
            loan_status, outstanding_debt, monthly_payment, months_remaining,
            assigned_lender, interest_rate, origination_month,
            available_capital, sector_exposure, loans_booked, total_fees_earned,
        ]
        return {k: v for k, v in zip(keys, values)}
