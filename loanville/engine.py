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
from .run_schema import UnderwritingRun, build_run
from .run_logger import RunLogger


class SimulationEngine:
    def __init__(
        self,
        borrowers: list[Borrower],
        lenders: list[LenderConfig],
        openrouter_api_key: str = "",
        max_concurrent_per_lender: int = 5,
        mock: bool = False,
        data_mode: str = "full",
    ):
        self.borrowers = borrowers
        self.lenders = lenders
        self.mock = mock
        self.data_mode = data_mode
        self.max_concurrent = max_concurrent_per_lender

        if not mock:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                default_headers={
                    "X-Title": "Loanville",
                    "HTTP-Referer": "https://github.com/seadotdev/Loanville2",
                },
            )
        else:
            self.client = None

        # State
        self.all_decisions: dict[str, list[LenderDecision]] = {}  # lender_id -> decisions
        self.booked_loans: list[BookedLoan] = []
        self.loan_outcomes: list[LoanOutcome] = []
        self.deal_results: dict[str, dict] = {}  # borrower_id -> adjudication info

        # Flywheel: Run artifacts emitted per (lender, borrower) evaluation
        self.runs: list[UnderwritingRun] = []
        self.run_logger: RunLogger | None = None

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
            print(f"\n[MOCK MODE] Simulating LLM evaluations (data_mode={self.data_mode})...\n")
            self.all_decisions = mock_evaluate_all(
                self.lenders, self.borrowers, self.data_mode,
            )
        else:
            # Run all lenders in parallel via OpenRouter
            tasks = [
                run_lender_evaluations(
                    self.client, lender, self.borrowers, self.max_concurrent,
                    self.data_mode,
                )
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

        # Emit UnderwritingRun artifacts for every (lender, borrower) evaluation
        borrower_map = {b.id: b for b in self.borrowers}
        for lender in self.lenders:
            for decision in self.all_decisions.get(lender.id, []):
                borrower = borrower_map.get(decision.borrower_id)
                if borrower:
                    run = build_run(
                        borrower=borrower,
                        lender=lender,
                        decision=decision,
                        source="simulator",
                    )
                    self.runs.append(run)

    # ------------------------------------------------------------------
    # Phase 3: Deal Adjudication
    # ------------------------------------------------------------------
    def adjudicate_deals(self) -> None:
        """Determine which lender wins each deal based on competitive offers.

        Enforces capital limits: a lender cannot deploy more than its available
        capital (total_capital minus existing portfolio).  If the preferred
        lender lacks capacity, the deal falls to the next-best offer.
        """
        print("\n" + "=" * 70)
        print("PHASE 3: DEAL ADJUDICATION")
        print("=" * 70)

        borrower_map = {b.id: b for b in self.borrowers}
        lender_map = {l.id: l for l in self.lenders}
        loan_counter = 0

        # Track remaining deployable capital per lender
        remaining_capital: dict[str, float] = {}
        for lender in self.lenders:
            existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
            remaining_capital[lender.id] = lender.total_capital - existing_deployed

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

            # Sort by borrower preference: lowest rate, then highest amount
            approvals.sort(key=lambda a: (a.term_sheet.interest_rate, -a.term_sheet.loan_amount))

            if len(approvals) > 1:
                print(f"\n  {bname}: COMPETITIVE - {len(approvals)} offers")
                for a in approvals:
                    lname = lender_map[a.lender_id].name
                    ts = a.term_sheet
                    cap = remaining_capital[a.lender_id]
                    cap_note = "" if ts.loan_amount <= cap else f" [OVER CAPITAL: ${cap:,.0f} remaining]"
                    print(f"    {lname}: ${ts.loan_amount:,.0f} @ {ts.interest_rate}% "
                          f"for {ts.term_months}mo{cap_note}")

            # Pick the best offer from a lender that has enough capital
            winner = None
            for a in approvals:
                if a.term_sheet.loan_amount <= remaining_capital[a.lender_id]:
                    winner = a
                    break

            if winner is None:
                # No lender has enough capital — deal falls through
                self.deal_results[bid] = {"outcome": "no_capital", "winner": None}
                if len(approvals) == 1:
                    lname = lender_map[approvals[0].lender_id].name
                    print(f"\n  {bname}: SINGLE OFFER from {lname} — "
                          f"DECLINED (insufficient capital)")
                else:
                    print(f"    --> NO DEAL — all interested lenders at capital limit")
                continue

            if len(approvals) == 1:
                print(f"\n  {bname}: SINGLE OFFER from {lender_map[winner.lender_id].name}")
            else:
                # Mark winner in the list
                for a in approvals:
                    if a is winner:
                        lname = lender_map[a.lender_id].name
                        print(f"    --> {lname} selected")
                        break

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
            remaining_capital[winner.lender_id] -= ts.loan_amount

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
    # Run artifact finalization
    # ------------------------------------------------------------------
    def _finalize_runs(self) -> None:
        """Attach outcome labels to runs after loan resolution."""
        # Index outcomes by (lender_id, borrower_id)
        outcome_map: dict[tuple[str, str], LoanOutcome] = {}
        for loan in self.booked_loans:
            outcome = next(
                (o for o in self.loan_outcomes if o.loan_id == loan.id), None
            )
            if outcome:
                outcome_map[(loan.lender_id, loan.borrower_id)] = outcome

        for run in self.runs:
            key = (
                run.policy.params.get("_lender_id", ""),
                run.case.case_id,
            )
            # Try to match by iterating outcomes
            for (lid, bid), outcome in outcome_map.items():
                if bid == run.case.case_id and lid in run.policy.policy_id:
                    run.labels.outcome = {
                        "defaulted": outcome.defaulted,
                        "was_fraud": outcome.was_fraud,
                        "months_paid": outcome.months_paid,
                        "interest_paid": outcome.total_interest_paid,
                        "principal_lost": outcome.principal_lost,
                        "principal_recovered": outcome.principal_recovered,
                    }
                    break

        # Log runs if logger is configured
        if self.run_logger:
            for run in self.runs:
                self.run_logger.log(run)

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Execute the full simulation."""
        await self.run_origination()
        self.adjudicate_deals()
        self.print_booked_ledger()
        self.resolve_loans()
        self._finalize_runs()
