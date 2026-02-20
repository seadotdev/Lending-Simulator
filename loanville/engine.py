"""
Simulation engine: orchestrates the origination loop, deal adjudication,
ledger booking, and loan resolution (fast-forward).
"""

import asyncio
import json
from openai import AsyncOpenAI

from .models import (
    BookedLoan,
    Borrower,
    LenderConfig,
    LenderDecision,
    LoanOutcome,
)
from .llm import run_lender_evaluations, get_call_traces, clear_call_traces
from .mock_llm import mock_evaluate_all


class SimulationEngine:
    def __init__(
        self,
        borrowers: list[Borrower],
        lenders: list[LenderConfig],
        openrouter_api_key: str = "",
        max_concurrent_per_lender: int = 5,
        mock: bool = False,
    ):
        self.borrowers = borrowers
        self.lenders = lenders
        self.mock = mock
        self.max_concurrent = max_concurrent_per_lender

        if not mock:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
        else:
            self.client = None

        # State
        self.all_decisions: dict[str, list[LenderDecision]] = {}  # lender_id -> decisions
        self.booked_loans: list[BookedLoan] = []
        self.loan_outcomes: list[LoanOutcome] = []
        self.deal_results: dict[str, dict] = {}  # borrower_id -> adjudication info

    # ------------------------------------------------------------------
    # Phase 1 & 2: Pipeline Distribution + Underwriting
    # ------------------------------------------------------------------
    async def run_origination(self) -> None:
        """Run all lender evaluations in parallel."""
        print("\n" + "=" * 70)
        print("PHASE 1-2: PIPELINE DISTRIBUTION & UNDERWRITING")
        print("=" * 70)

        borrower_names = {b.id: b.dossier.company_name for b in self.borrowers}
        print(f"\nBroadcasting {len(self.borrowers)} applications to {len(self.lenders)} lenders...")
        for b in self.borrowers:
            print(f"  - {b.id}: {b.dossier.company_name} ({b.dossier.sector}) "
                  f"requesting ${b.dossier.loan_request_amount:,.0f}")

        if self.mock:
            print("\n[MOCK MODE] Simulating LLM evaluations...\n")
            self.all_decisions = mock_evaluate_all(self.lenders, self.borrowers)
        else:
            # Run all lenders in parallel via OpenRouter
            tasks = [
                run_lender_evaluations(self.client, lender, self.borrowers, self.max_concurrent)
                for lender in self.lenders
            ]
            print("\nLenders are evaluating applications...\n")
            results = await asyncio.gather(*tasks)
            for lender, decisions in zip(self.lenders, results):
                self.all_decisions[lender.id] = decisions

        # Print results
        for lender in self.lenders:
            decisions = self.all_decisions[lender.id]
            approvals = sum(1 for d in decisions if d.decision == "APPROVE")
            rejections = len(decisions) - approvals
            print(f"  {lender.name} ({lender.model}):")
            print(f"    Approved: {approvals} | Rejected: {rejections}")
            for d in decisions:
                status = "APPROVED" if d.decision == "APPROVE" else "REJECTED"
                bname = borrower_names.get(d.borrower_id, d.borrower_id)
                rate_str = ""
                if d.term_sheet:
                    rate_str = f" @ {d.term_sheet.interest_rate}% for {d.term_sheet.term_months}mo"
                print(f"      {bname}: {status}{rate_str}")
                if d.reasoning and not d.reasoning.startswith("[SYSTEM"):
                    print(f"        Reasoning: {d.reasoning}")

        # Write trace file for live (non-mock) runs
        if not self.mock:
            traces = get_call_traces()
            if traces:
                trace_path = "loanville_trace.json"
                with open(trace_path, "w") as f:
                    json.dump(traces, f, indent=2)
                print(f"\n  Trace log written to {trace_path} ({len(traces)} calls)")
                clear_call_traces()

    # ------------------------------------------------------------------
    # Phase 3: Deal Adjudication
    # ------------------------------------------------------------------
    def adjudicate_deals(self) -> None:
        """Determine which lender wins each deal based on competitive offers."""
        print("\n" + "=" * 70)
        print("PHASE 3: DEAL ADJUDICATION")
        print("=" * 70)

        borrower_map = {b.id: b for b in self.borrowers}
        lender_map = {l.id: l for l in self.lenders}
        loan_counter = 0

        for borrower in self.borrowers:
            bid = borrower.id
            bname = borrower.dossier.company_name

            # Collect all approvals for this borrower
            approvals: list[LenderDecision] = []
            for lender_id, decisions in self.all_decisions.items():
                for d in decisions:
                    if d.borrower_id == bid and d.decision == "APPROVE" and d.term_sheet:
                        approvals.append(d)

            if not approvals:
                self.deal_results[bid] = {"outcome": "no_takers", "winner": None}
                print(f"\n  {bname}: NO OFFERS - all lenders rejected")
                continue

            if len(approvals) == 1:
                winner = approvals[0]
                print(f"\n  {bname}: SINGLE OFFER from {lender_map[winner.lender_id].name}")
            else:
                # Multiple offers - borrower picks the best deal
                # Best = lowest effective cost (interest_rate * loan_amount_requested / loan_amount_offered)
                # Simplified: borrower prefers lowest interest rate, ties broken by highest amount
                approvals.sort(key=lambda a: (a.term_sheet.interest_rate, -a.term_sheet.loan_amount))
                winner = approvals[0]
                print(f"\n  {bname}: COMPETITIVE - {len(approvals)} offers")
                for a in approvals:
                    lname = lender_map[a.lender_id].name
                    ts = a.term_sheet
                    marker = " <-- WINNER" if a is winner else ""
                    print(f"    {lname}: ${ts.loan_amount:,.0f} @ {ts.interest_rate}% "
                          f"for {ts.term_months}mo{marker}")

            # Book the winning deal
            loan_counter += 1
            ts = winner.term_sheet
            b = borrower_map[bid]
            loan = BookedLoan(
                id=f"LOAN-{loan_counter:03d}",
                borrower_id=bid,
                lender_id=winner.lender_id,
                borrower_name=bname,
                sector=b.dossier.sector,
                principal=ts.loan_amount,
                interest_rate=ts.interest_rate,
                term_months=ts.term_months,
                true_outcome=b.true_outcome,
                months_before_default=b.months_before_default,
            )
            self.booked_loans.append(loan)

            self.deal_results[bid] = {
                "outcome": "booked",
                "winner": winner.lender_id,
                "loan_id": loan.id,
            }

            # Notify results
            winner_name = lender_map[winner.lender_id].name
            print(f"    --> Deal won by {winner_name}: {loan.id}")

            # Notify losers
            for a in approvals:
                if a is not winner:
                    loser_name = lender_map[a.lender_id].name
                    print(f"    --> {loser_name}: Deal lost to competitor")

    # ------------------------------------------------------------------
    # Phase 4: Ledger Booking (summary)
    # ------------------------------------------------------------------
    def print_booked_ledger(self) -> None:
        """Print the global ledger of booked loans."""
        print("\n" + "=" * 70)
        print("PHASE 4: LEDGER BOOKING")
        print("=" * 70)

        lender_map = {l.id: l.name for l in self.lenders}

        if not self.booked_loans:
            print("\n  No loans were booked.")
            return

        print(f"\n  {len(self.booked_loans)} loans booked to the Global Ledger:\n")
        for loan in self.booked_loans:
            lname = lender_map.get(loan.lender_id, loan.lender_id)
            print(f"  {loan.id}: {loan.borrower_name} ({loan.sector})")
            print(f"    Lender: {lname}")
            print(f"    Principal: ${loan.principal:,.2f} @ {loan.interest_rate}% "
                  f"for {loan.term_months} months")
            print(f"    [Hidden: outcome={loan.true_outcome}]")

    # ------------------------------------------------------------------
    # Resolution: Fast-forward the future
    # ------------------------------------------------------------------
    def resolve_loans(self) -> None:
        """Fast-forward through loan lifecycles and determine outcomes."""
        print("\n" + "=" * 70)
        print("RESOLUTION: FAST-FORWARDING LOAN LIFECYCLES")
        print("=" * 70)

        for loan in self.booked_loans:
            monthly_rate = loan.interest_rate / 100.0 / 12.0

            if loan.true_outcome == "fraud":
                # Instant default - principal lost entirely
                outcome = LoanOutcome(
                    loan_id=loan.id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    sector=loan.sector,
                    principal=loan.principal,
                    total_interest_paid=0.0,
                    principal_recovered=0.0,
                    principal_lost=loan.principal,
                    defaulted=True,
                    was_fraud=True,
                    months_paid=0,
                )
                print(f"\n  {loan.id} ({loan.borrower_name}): FRAUD - Immediate default!")
                print(f"    Principal lost: ${loan.principal:,.2f}")

            elif loan.true_outcome == "bad":
                # Partial payments then default
                months_paid = min(loan.months_before_default or 6, loan.term_months)

                # Calculate amortization payments
                if monthly_rate > 0:
                    payment = loan.principal * (monthly_rate * (1 + monthly_rate) ** loan.term_months) / \
                              ((1 + monthly_rate) ** loan.term_months - 1)
                else:
                    payment = loan.principal / loan.term_months

                total_paid = payment * months_paid
                # Break down into interest and principal components
                remaining_principal = loan.principal
                total_interest = 0.0
                total_principal_paid = 0.0
                for _ in range(months_paid):
                    interest_portion = remaining_principal * monthly_rate
                    principal_portion = payment - interest_portion
                    total_interest += interest_portion
                    total_principal_paid += principal_portion
                    remaining_principal -= principal_portion

                principal_lost = max(0, remaining_principal)
                outcome = LoanOutcome(
                    loan_id=loan.id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    sector=loan.sector,
                    principal=loan.principal,
                    total_interest_paid=round(total_interest, 2),
                    principal_recovered=round(total_principal_paid, 2),
                    principal_lost=round(principal_lost, 2),
                    defaulted=True,
                    was_fraud=False,
                    months_paid=months_paid,
                )
                print(f"\n  {loan.id} ({loan.borrower_name}): DEFAULT after {months_paid} months")
                print(f"    Interest collected: ${total_interest:,.2f}")
                print(f"    Principal recovered: ${total_principal_paid:,.2f}")
                print(f"    Principal lost: ${principal_lost:,.2f}")

            else:  # good
                # Full amortization - all payments made
                if monthly_rate > 0:
                    payment = loan.principal * (monthly_rate * (1 + monthly_rate) ** loan.term_months) / \
                              ((1 + monthly_rate) ** loan.term_months - 1)
                else:
                    payment = loan.principal / loan.term_months

                total_paid = payment * loan.term_months
                total_interest = total_paid - loan.principal
                outcome = LoanOutcome(
                    loan_id=loan.id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    sector=loan.sector,
                    principal=loan.principal,
                    total_interest_paid=round(total_interest, 2),
                    principal_recovered=round(loan.principal, 2),
                    principal_lost=0.0,
                    defaulted=False,
                    was_fraud=False,
                    months_paid=loan.term_months,
                )
                print(f"\n  {loan.id} ({loan.borrower_name}): FULLY REPAID over {loan.term_months} months")
                print(f"    Interest collected: ${total_interest:,.2f}")
                print(f"    Principal recovered: ${loan.principal:,.2f}")

            self.loan_outcomes.append(outcome)

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Execute the full simulation."""
        await self.run_origination()
        self.adjudicate_deals()
        self.print_booked_ledger()
        self.resolve_loans()
