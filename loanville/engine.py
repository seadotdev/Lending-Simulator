"""
Simulation engine: orchestrates the origination loop, deal adjudication,
ledger booking, and loan resolution (fast-forward).
"""

import asyncio
import hashlib
import json
import math
from openai import AsyncOpenAI

from .models import (
    BookedLoan,
    Borrower,
    EconomicsConfig,
    LenderConfig,
    LenderDecision,
    LoanOutcome,
    ResolveResult,
)
from .mock_llm import mock_evaluate_all
from .run_schema import UnderwritingRun, build_run
from .run_logger import RunLogger


def _load_llm_functions():
    """Import LLM helpers lazily so LOS mode doesn't require direct-LLM deps."""
    try:
        from .llm import run_lender_evaluations, get_call_traces, clear_call_traces
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "unknown")
        raise RuntimeError(
            f"Missing dependency '{missing}' required for direct OpenRouter mode. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return run_lender_evaluations, get_call_traces, clear_call_traces


def _stable_u01(*parts: str) -> float:
    """Deterministic U[0,1) derived from stable identifiers (no RNG state)."""
    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    # Use 8 bytes for a stable, reproducible float in [0,1).
    return int.from_bytes(h[:8], "big") / float(2**64)


def _run_lender_id(run: UnderwritingRun) -> str:
    """Best-effort lender_id extraction from a Run."""
    lid = (run.policy.params or {}).get("_lender_id", "")
    if lid:
        return str(lid)
    pid = run.policy.policy_id or ""
    # policy_id format: "p_{lender_id}_{model}"
    parts = pid.split("_", 2)
    if len(parts) >= 2 and parts[0] == "p":
        return parts[1]
    return pid


def resolve_loan_period(
    principal: float,
    interest_rate: float,
    term_months: int,
    true_outcome: str,
    months_before_default: int | None,
    months_already_elapsed: int,
    months_to_advance: int,
    economics: EconomicsConfig,
) -> ResolveResult:
    """Advance a loan by N months.

    Returns interest collected, principal repaid, whether default/prepay/maturity
    occurred, and remaining balance. Uses the same amortization math as
    resolve_loans().
    """
    eco = economics
    monthly_rate = interest_rate / 100.0 / 12.0
    orig_fee = max(0.0, principal * eco.origination_fee_rate) if months_already_elapsed == 0 else 0.0

    # Compute amortization payment (based on full term from origination)
    if monthly_rate > 0 and term_months > 0:
        payment = principal * (monthly_rate * (1 + monthly_rate) ** term_months) / \
                  ((1 + monthly_rate) ** term_months - 1)
    else:
        payment = principal / term_months if term_months > 0 else 0.0

    # Compute remaining principal at the start of this period
    remaining = principal
    for _ in range(months_already_elapsed):
        interest_portion = remaining * monthly_rate
        principal_portion = payment - interest_portion
        remaining -= principal_portion

    if true_outcome == "fraud" and months_already_elapsed == 0:
        # Instant default (no payments)
        recovery_amt = max(0.0, principal * eco.recovery_rate_fraud)
        workout_cost = max(0.0, principal * eco.workout_cost_rate)
        principal_lost = max(0.0, principal - recovery_amt)
        return ResolveResult(
            interest=0.0,
            principal_repaid=0.0,
            remaining_balance=0.0,
            principal_lost=principal_lost,
            recovery_amount=recovery_amt,
            workout_cost=workout_cost,
            fees=orig_fee,
            defaulted=True,
            matured=False,
            prepaid=False,
            months_actually_advanced=0,
        )

    months_remaining_in_term = term_months - months_already_elapsed
    months_to_run = min(months_to_advance, months_remaining_in_term)

    # Check if default occurs during this period
    default_month = None
    if true_outcome in ("bad", "fraud") and months_before_default is not None:
        months_until_default = months_before_default - months_already_elapsed
        if months_until_default <= months_to_run and months_until_default > 0:
            default_month = months_until_default
        elif months_until_default <= 0:
            # Already past default point — default immediately
            default_month = 0

    # Check prepayment for good loans
    prepay_month = None
    if true_outcome == "good":
        monthly_prepay_hazard = max(0.0, min(0.95, eco.prepayment_rate_annual / 12.0))
        if monthly_prepay_hazard > 0 and months_remaining_in_term > 1:
            # Deterministic prepayment based on stable hash
            u = _stable_u01("prepay_period", str(principal), str(interest_rate),
                           str(months_already_elapsed))
            t = int(math.log(max(1e-15, 1.0 - u)) / math.log(max(1e-15, 1.0 - monthly_prepay_hazard))) + 1
            if t <= months_to_run:
                prepay_month = max(1, t)

    # Determine how many performing months to simulate
    if default_month is not None and default_month == 0:
        performing_months = 0
    elif default_month is not None:
        performing_months = default_month
        if prepay_month is not None:
            performing_months = min(performing_months, prepay_month)
    elif prepay_month is not None:
        performing_months = prepay_month
    else:
        performing_months = months_to_run

    # Simulate performing months
    total_interest = 0.0
    total_principal_paid = 0.0
    for _ in range(performing_months):
        interest_portion = remaining * monthly_rate
        principal_portion = payment - interest_portion
        total_interest += interest_portion
        total_principal_paid += principal_portion
        remaining -= principal_portion

    defaulted = False
    matured = False
    prepaid = False
    principal_lost = 0.0
    recovery_amt = 0.0
    workout_cost = 0.0
    fees = orig_fee

    if default_month is not None and (prepay_month is None or default_month <= prepay_month):
        # Default occurred
        defaulted = True
        if true_outcome == "fraud":
            recovery_amt = max(0.0, remaining * eco.recovery_rate_fraud)
        else:
            recovery_amt = max(0.0, remaining * eco.recovery_rate_bad)
        workout_cost = max(0.0, remaining * eco.workout_cost_rate)
        principal_lost = max(0.0, remaining - recovery_amt)
        remaining = 0.0
    elif prepay_month is not None and (default_month is None or prepay_month < default_month):
        # Prepayment
        prepaid = True
        if remaining > 0:
            fees += max(0.0, remaining * eco.prepayment_penalty_rate)
            total_principal_paid += remaining
            remaining = 0.0
    elif months_already_elapsed + performing_months >= term_months:
        # Loan matured
        matured = True
        if remaining > 0:
            total_principal_paid += remaining
            remaining = 0.0

    return ResolveResult(
        interest=round(total_interest, 2),
        principal_repaid=round(total_principal_paid, 2),
        remaining_balance=round(max(0.0, remaining), 2),
        principal_lost=round(principal_lost, 2),
        recovery_amount=round(recovery_amt, 2),
        workout_cost=round(workout_cost, 2),
        fees=round(fees, 2),
        defaulted=defaulted,
        matured=matured,
        prepaid=prepaid,
        months_actually_advanced=performing_months,
    )


class SimulationEngine:
    def __init__(
        self,
        borrowers: list[Borrower],
        lenders: list[LenderConfig],
        openrouter_api_key: str = "",
        max_concurrent_per_lender: int = 5,
        mock: bool = False,
        data_mode: str = "full",
        use_los: bool = False,
        los_url: str = "http://localhost:3000",
        los_provider: str = "openrouter",
        los_mode: str = "rules_only",
        underwrite_only: bool = False,
        los_model: str | None = None,
        economics: EconomicsConfig | None = None,
    ):
        self.borrowers = borrowers
        self.lenders = lenders
        self.mock = mock
        self.use_los = use_los
        self.los_url = los_url
        self.los_provider = los_provider
        self.los_mode = los_mode
        self.underwrite_only = underwrite_only
        self.los_model = los_model
        self.data_mode = data_mode
        self.max_concurrent = max_concurrent_per_lender
        self.economics = economics or EconomicsConfig()

        if not mock and not use_los:
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

        if self.use_los:
            from .los_adapter import evaluate_all_via_los, run_to_decision

            mode_label = "underwrite-only" if self.underwrite_only else self.los_mode
            print(f"\n[LOS MODE] Evaluating via Open LOS at {self.los_url} "
                  f"(mode={mode_label})...\n")
            tasks = [
                evaluate_all_via_los(
                    lender, self.borrowers, self.los_url,
                    max_concurrent=self.max_concurrent,
                    provider=self.los_provider,
                    mode=self.los_mode,
                    underwrite_only=self.underwrite_only,
                    los_model=self.los_model,
                )
                for lender in self.lenders
            ]
            results = await asyncio.gather(*tasks)
            for lender, (decisions, runs) in zip(self.lenders, results):
                self.all_decisions[lender.id] = decisions
                self.runs.extend(runs)

        elif self.mock:
            print(f"\n[MOCK MODE] Simulating LLM evaluations (data_mode={self.data_mode})...\n")
            self.all_decisions = mock_evaluate_all(
                self.lenders, self.borrowers, self.data_mode,
            )
        else:
            # Run all lenders in parallel via OpenRouter
            run_lender_evaluations, _, _ = _load_llm_functions()
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
        if not self.mock and not self.use_los:
            _, get_call_traces, clear_call_traces = _load_llm_functions()
            traces = get_call_traces()
            if traces:
                trace_path = "loanville_trace.json"
                with open(trace_path, "w") as f:
                    json.dump(traces, f, indent=2)
                print(f"\n  Trace log written to {trace_path} ({len(traces)} calls)")
                clear_call_traces()

        # Emit UnderwritingRun artifacts for every (lender, borrower) evaluation
        # Skip if LOS mode — runs already emitted by the adapter
        if not self.use_los:
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
    def adjudicate_deals(
        self,
        remaining_capital: dict[str, float] | None = None,
        speed_scoring: bool = False,
        tool_call_counts: dict[tuple[str, str], int] | None = None,
    ) -> None:
        """Determine which lender wins each deal based on competitive offers.

        Enforces capital limits: a lender cannot deploy more than its available
        capital (total_capital minus existing portfolio).  If the preferred
        lender lacks capacity, the deal falls to the next-best offer.

        If remaining_capital is provided, uses those values instead of computing
        from existing_portfolio (used by season mode).

        If speed_scoring is True and tool_call_counts provided, computes
        offer_score = rate_competitiveness + speed_bonus for sorting.
        """
        print("\n" + "=" * 70)
        print("PHASE 3: DEAL ADJUDICATION")
        print("=" * 70)

        borrower_map = {b.id: b for b in self.borrowers}
        lender_map = {l.id: l for l in self.lenders}
        loan_counter = 0

        # Track remaining deployable capital per lender
        if remaining_capital is None:
            remaining_capital = {}
            for lender in self.lenders:
                existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
                remaining_capital[lender.id] = lender.total_capital - existing_deployed
        else:
            remaining_capital = dict(remaining_capital)  # copy to avoid mutation

        # Index runs by (lender_id, borrower_id) for friction modeling (LOS trace)
        run_index: dict[tuple[str, str], UnderwritingRun] = {}
        for run in self.runs:
            lid = _run_lender_id(run)
            bid = run.case.case_id
            if lid and bid:
                run_index[(lid, bid)] = run

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

            def _base_offer_sort_key(offer: LenderDecision) -> tuple[float, float]:
                return (offer.term_sheet.interest_rate, -offer.term_sheet.loan_amount)

            def _speed_bonus(offer: LenderDecision) -> float:
                if not (speed_scoring and tool_call_counts):
                    return 0.0
                tc = tool_call_counts.get((offer.lender_id, bid), 7)
                if tc <= 3:
                    return 0.15
                if tc <= 6:
                    return 0.05
                return 0.0

            def _speed_offer_sort_key(offer: LenderDecision) -> tuple[float, float]:
                return (
                    offer.term_sheet.interest_rate - _speed_bonus(offer),
                    -offer.term_sheet.loan_amount,
                )

            ranked_approvals = (
                sorted(approvals, key=_speed_offer_sort_key)
                if speed_scoring and tool_call_counts
                else sorted(approvals, key=_base_offer_sort_key)
            )
            baseline_approvals = sorted(approvals, key=_base_offer_sort_key)

            if len(ranked_approvals) > 1:
                print(f"\n  {bname}: COMPETITIVE - {len(ranked_approvals)} offers")
                for a in ranked_approvals:
                    lname = lender_map[a.lender_id].name
                    ts = a.term_sheet
                    cap = remaining_capital[a.lender_id]
                    cap_note = "" if ts.loan_amount <= cap else f" [OVER CAPITAL: ${cap:,.0f} remaining]"
                    print(f"    {lname}: ${ts.loan_amount:,.0f} @ {ts.interest_rate}% "
                          f"for {ts.term_months}mo{cap_note}")

            # Pick the best offer from a lender that has enough capital.
            # Optional realism: a borrower may abandon an offer if underwriting friction is high
            # (e.g., too many doc requests / slow processing).
            eco = self.economics
            friction_enabled = (
                eco.abandonment_base_rate > 0
                or eco.abandonment_per_doc_request > 0
                or eco.abandonment_per_second_latency > 0
            )
            speed_bonus_decisive = False
            speed_baseline_winner = None

            def _pick_winner(
                candidates: list[LenderDecision],
                *,
                emit_logs: bool,
            ) -> tuple[LenderDecision | None, bool, list[str]]:
                abandoned: list[str] = []
                has_capacity_offer = False
                for candidate in candidates:
                    if candidate.term_sheet.loan_amount <= remaining_capital[candidate.lender_id]:
                        has_capacity_offer = True

                        if friction_enabled:
                            run = run_index.get((candidate.lender_id, bid))
                            doc_requests = 0
                            latency_s = 0.0
                            if run:
                                doc_requests = sum(1 for s in run.trace.steps if s.type == "doc_request")
                                latency_s = (run.trace.latency_ms or 0) / 1000.0

                            p = (
                                eco.abandonment_base_rate
                                + doc_requests * eco.abandonment_per_doc_request
                                + latency_s * eco.abandonment_per_second_latency
                            )
                            p = max(0.0, min(eco.abandonment_cap, p))
                            if p > 0.0:
                                u = _stable_u01("abandon", bid, candidate.lender_id)
                                if u < p:
                                    abandoned.append(candidate.lender_id)
                                    if emit_logs:
                                        lname = lender_map[candidate.lender_id].name
                                        print(
                                            f"    --> {lname} offer abandoned "
                                            f"(p={p*100:.1f}%, doc_requests={doc_requests}, latency={latency_s:.1f}s)"
                                        )
                                    continue

                        return candidate, has_capacity_offer, abandoned
                return None, has_capacity_offer, abandoned

            winner, had_capacity_offer, abandoned_offers = _pick_winner(
                ranked_approvals, emit_logs=True
            )

            if speed_scoring and tool_call_counts and len(ranked_approvals) > 1:
                baseline_winner, _, _ = _pick_winner(
                    baseline_approvals, emit_logs=False
                )
                if baseline_winner:
                    speed_baseline_winner = baseline_winner.lender_id
                if (
                    winner
                    and baseline_winner
                    and winner.lender_id != baseline_winner.lender_id
                ):
                    speed_bonus_decisive = True

            if winner is None:
                # No lender has enough capital — deal falls through
                if had_capacity_offer and abandoned_offers:
                    self.deal_results[bid] = {
                        "outcome": "abandoned",
                        "winner": None,
                        "abandoned_offers": abandoned_offers,
                        "competitive": len(ranked_approvals) > 1,
                        "speed_bonus_decisive": False,
                    }
                    if len(ranked_approvals) == 1:
                        lname = lender_map[ranked_approvals[0].lender_id].name
                        print(f"\n  {bname}: SINGLE OFFER from {lname} — "
                              f"ABANDONED (underwriting friction)")
                    else:
                        print(f"    --> NO DEAL — borrower abandoned all viable offers")
                else:
                    self.deal_results[bid] = {
                        "outcome": "no_capital",
                        "winner": None,
                        "competitive": len(ranked_approvals) > 1,
                        "speed_bonus_decisive": False,
                    }
                    if len(ranked_approvals) == 1:
                        lname = lender_map[ranked_approvals[0].lender_id].name
                        print(f"\n  {bname}: SINGLE OFFER from {lname} — "
                              f"DECLINED (insufficient capital)")
                    else:
                        print(f"    --> NO DEAL — all interested lenders at capital limit")
                continue

            if len(ranked_approvals) == 1:
                print(f"\n  {bname}: SINGLE OFFER from {lender_map[winner.lender_id].name}")
            else:
                # Mark winner in the list
                for a in ranked_approvals:
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
                "competitive": len(ranked_approvals) > 1,
                "speed_bonus_decisive": speed_bonus_decisive,
                "speed_baseline_winner": speed_baseline_winner,
            }

            # Notify results
            winner_name = lender_map[winner.lender_id].name
            print(f"    --> Deal won by {winner_name}: {loan.id}")

            # Notify losers
            for a in ranked_approvals:
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

        eco = self.economics
        for loan in self.booked_loans:
            monthly_rate = loan.interest_rate / 100.0 / 12.0
            orig_fee = max(0.0, loan.principal * eco.origination_fee_rate)

            if loan.true_outcome == "fraud":
                # Instant default (no payments); allow small recovery + workout costs
                recovery_amt = max(0.0, loan.principal * eco.recovery_rate_fraud)
                workout_cost = max(0.0, loan.principal * eco.workout_cost_rate)
                principal_lost = max(0.0, loan.principal - recovery_amt)
                outcome = LoanOutcome(
                    loan_id=loan.id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    borrower_id=loan.borrower_id,
                    sector=loan.sector,
                    principal=loan.principal,
                    total_interest_paid=0.0,
                    principal_recovered=round(recovery_amt, 2),
                    principal_lost=round(principal_lost, 2),
                    defaulted=True,
                    was_fraud=True,
                    months_paid=0,
                    total_fees_paid=round(orig_fee, 2),
                    recovery_amount=round(recovery_amt, 2),
                    workout_cost=round(workout_cost, 2),
                    prepaid=False,
                )
                print(f"\n  {loan.id} ({loan.borrower_name}): FRAUD - Immediate default!")
                if recovery_amt > 0:
                    print(f"    Recovery:       ${recovery_amt:,.2f}")
                if workout_cost > 0:
                    print(f"    Workout cost:   ${workout_cost:,.2f}")
                print(f"    Principal lost: ${principal_lost:,.2f}")

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

                recovery_amt = max(0.0, remaining_principal * eco.recovery_rate_bad)
                workout_cost = max(0.0, remaining_principal * eco.workout_cost_rate)
                principal_lost = max(0.0, remaining_principal - recovery_amt)
                principal_recovered = min(loan.principal, total_principal_paid + recovery_amt)
                outcome = LoanOutcome(
                    loan_id=loan.id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    borrower_id=loan.borrower_id,
                    sector=loan.sector,
                    principal=loan.principal,
                    total_interest_paid=round(total_interest, 2),
                    principal_recovered=round(principal_recovered, 2),
                    principal_lost=round(principal_lost, 2),
                    defaulted=True,
                    was_fraud=False,
                    months_paid=months_paid,
                    total_fees_paid=round(orig_fee, 2),
                    recovery_amount=round(recovery_amt, 2),
                    workout_cost=round(workout_cost, 2),
                    prepaid=False,
                )
                print(f"\n  {loan.id} ({loan.borrower_name}): DEFAULT after {months_paid} months")
                print(f"    Interest collected: ${total_interest:,.2f}")
                if recovery_amt > 0:
                    print(f"    Recovery:           ${recovery_amt:,.2f}")
                if workout_cost > 0:
                    print(f"    Workout cost:       ${workout_cost:,.2f}")
                print(f"    Principal recovered: ${principal_recovered:,.2f}")
                print(f"    Principal lost: ${principal_lost:,.2f}")

            else:  # good
                # Good loan — full amortization or deterministic prepayment
                prepaid = False
                months_paid = loan.term_months
                monthly_prepay_hazard = max(0.0, min(0.95, eco.prepayment_rate_annual / 12.0))
                if monthly_prepay_hazard > 0 and loan.term_months > 1:
                    u = _stable_u01("prepay", loan.borrower_id, loan.lender_id)
                    # Geometric with per-month hazard:
                    #   P(T > t) = (1-h)^t, for t months with no event.
                    t = int(math.log(1.0 - u) / math.log(1.0 - monthly_prepay_hazard)) + 1
                    if t < loan.term_months:
                        prepaid = True
                        months_paid = max(1, t)

                # Calculate amortization payments (payment based on original term)
                if monthly_rate > 0:
                    payment = loan.principal * (monthly_rate * (1 + monthly_rate) ** loan.term_months) / \
                              ((1 + monthly_rate) ** loan.term_months - 1)
                else:
                    payment = loan.principal / loan.term_months if loan.term_months > 0 else 0.0

                remaining_principal = loan.principal
                total_interest = 0.0
                total_principal_paid = 0.0
                for _ in range(months_paid):
                    interest_portion = remaining_principal * monthly_rate
                    principal_portion = payment - interest_portion
                    total_interest += interest_portion
                    total_principal_paid += principal_portion
                    remaining_principal -= principal_portion

                fees = orig_fee
                if prepaid and remaining_principal > 0:
                    fees += max(0.0, remaining_principal * eco.prepayment_penalty_rate)
                    # Pay off the remaining principal at prepayment.
                    total_principal_paid += remaining_principal
                    remaining_principal = 0.0
                outcome = LoanOutcome(
                    loan_id=loan.id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    borrower_id=loan.borrower_id,
                    sector=loan.sector,
                    principal=loan.principal,
                    total_interest_paid=round(total_interest, 2),
                    principal_recovered=round(loan.principal, 2),
                    principal_lost=0.0,
                    defaulted=False,
                    was_fraud=False,
                    months_paid=months_paid,
                    total_fees_paid=round(fees, 2),
                    recovery_amount=0.0,
                    workout_cost=0.0,
                    prepaid=prepaid,
                )
                if prepaid:
                    print(f"\n  {loan.id} ({loan.borrower_name}): PREPAID after {months_paid} months")
                else:
                    print(f"\n  {loan.id} ({loan.borrower_name}): FULLY REPAID over {loan.term_months} months")
                print(f"    Interest collected: ${total_interest:,.2f}")
                if outcome.total_fees_paid > 0:
                    print(f"    Fees collected:     ${outcome.total_fees_paid:,.2f}")
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
                        "fees_paid": outcome.total_fees_paid,
                        "principal_lost": outcome.principal_lost,
                        "principal_recovered": outcome.principal_recovered,
                        "recovery_amount": outcome.recovery_amount,
                        "workout_cost": outcome.workout_cost,
                        "prepaid": outcome.prepaid,
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
