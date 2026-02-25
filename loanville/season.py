"""
Season mode engine: multi-week simulation with carry-forward portfolio state.

Phase 1 scope:
- Week loop with persistent lender capital/exposure state
- Rolling loan resolution (aging each week)
- Weekly portfolio briefings injected into lender persona
- Deterministic cohort generation from existing borrower pools
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from .data import get_borrowers
from .engine import SimulationEngine
from .models import (
    BookedLoan,
    Borrower,
    EconomicsConfig,
    ExistingLoan,
    LenderConfig,
    LenderDecision,
    LoanOutcome,
)
from .run_schema import UnderwritingRun


SEASON_MIX_CHOICES = ("gentle", "realistic", "adversarial", "escalating")
_BASE_MIX_MAP = {
    "gentle": "easy",
    "realistic": "realistic",
    "adversarial": "hard",
}


@dataclass
class SeasonConfig:
    weeks: int = 10
    cohort_size: int = 5
    months_per_week: int = 2
    season_mix: str = "realistic"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.weeks <= 0:
            raise ValueError("weeks must be > 0")
        if self.cohort_size <= 0:
            raise ValueError("cohort_size must be > 0")
        if self.months_per_week <= 0:
            raise ValueError("months_per_week must be > 0")
        if self.season_mix not in SEASON_MIX_CHOICES:
            raise ValueError(
                f"season_mix must be one of {SEASON_MIX_CHOICES}, got '{self.season_mix}'"
            )


@dataclass
class ActiveLoanState:
    loan_id: str
    lender_id: str
    borrower_id: str
    borrower_name: str
    sector: str
    original_principal: float
    principal_outstanding: float
    interest_rate_pct: float
    term_months: int
    true_outcome: str
    months_before_default: int | None
    months_paid: int = 0
    total_interest_paid: float = 0.0
    total_principal_paid: float = 0.0
    total_fees_paid: float = 0.0
    status: str = "active"


@dataclass
class SeasonLenderState:
    lender_id: str
    lender_name: str
    model: str
    base_capital: float
    baseline_deployed: float
    deployed_capital: float
    available_capital: float
    sector_exposure: dict[str, float] = field(default_factory=dict)
    active_loans: list[ActiveLoanState] = field(default_factory=list)
    resolved_loans: list[LoanOutcome] = field(default_factory=list)
    cumulative_interest: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_losses: float = 0.0
    cumulative_workout_cost: float = 0.0
    deals_won: int = 0
    deals_lost: int = 0
    deals_rejected: int = 0
    total_evaluations: int = 0
    total_tool_calls: int = 0
    recent_events: list[str] = field(default_factory=list)
    pending_events: list[str] = field(default_factory=list)
    weekly_snapshots: list[dict] = field(default_factory=list)


class SeasonEngine:
    """Runs a multi-week season while preserving lender state across weeks."""

    def __init__(
        self,
        config: SeasonConfig,
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
        self.config = config
        self.base_lenders = copy.deepcopy(lenders)
        self.openrouter_api_key = openrouter_api_key
        self.max_concurrent = max_concurrent_per_lender
        self.mock = mock
        self.data_mode = data_mode
        self.use_los = use_los
        self.los_url = los_url
        self.los_provider = los_provider
        self.los_mode = los_mode
        self.underwrite_only = underwrite_only
        self.los_model = los_model
        self.economics = economics or EconomicsConfig()

        self._rng = random.Random(self.config.seed)
        self._mix_queues: dict[str, list[Borrower]] = {}
        self._loan_counter = 0

        self.lender_states: dict[str, SeasonLenderState] = {}
        self._init_lender_states(self.base_lenders)

        self.all_decisions: dict[str, list[LenderDecision]] = {
            lender.id: [] for lender in self.base_lenders
        }
        self.booked_loans: list[BookedLoan] = []
        self.loan_outcomes: list[LoanOutcome] = []
        self.deal_results: dict[str, dict] = {}
        self.runs: list[UnderwritingRun] = []
        self.borrowers_seen: list[Borrower] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(self) -> None:
        await self.run_season()

    async def run_season(self) -> None:
        print("\n" + "=" * 70)
        print("SEASON MODE")
        print("=" * 70)
        print(
            f"Config: weeks={self.config.weeks}, cohort_size={self.config.cohort_size}, "
            f"months_per_week={self.config.months_per_week}, mix={self.config.season_mix}, "
            f"seed={self.config.seed}"
        )

        for week in range(1, self.config.weeks + 1):
            print("\n" + "#" * 70)
            print(f"WEEK {week}/{self.config.weeks}")
            print("#" * 70)

            self._resolve_week()
            week_lenders = self._build_week_lenders(week)
            cohort, mix_for_week = self._build_week_cohort(week)
            self.borrowers_seen.extend(cohort)

            print(
                f"Week mix: {mix_for_week} | cohort size: {len(cohort)} | "
                f"active loans entering week: {self._total_active_loans()}"
            )

            engine = SimulationEngine(
                borrowers=cohort,
                lenders=week_lenders,
                openrouter_api_key=self.openrouter_api_key,
                max_concurrent_per_lender=self.max_concurrent,
                mock=self.mock,
                data_mode=self.data_mode,
                use_los=self.use_los,
                los_url=self.los_url,
                los_provider=self.los_provider,
                los_mode=self.los_mode,
                underwrite_only=self.underwrite_only,
                los_model=self.los_model,
                economics=self.economics,
            )

            await engine.run_origination()
            engine.adjudicate_deals()
            engine.print_booked_ledger()

            self._ingest_week_results(engine)
            self._record_week_snapshots(week)
            self._print_week_summary(week)

        self._final_resolution()
        self._print_season_report()

    # ------------------------------------------------------------------
    # Initialization and state helpers
    # ------------------------------------------------------------------
    def _init_lender_states(self, lenders: list[LenderConfig]) -> None:
        for lender in lenders:
            baseline_deployed = sum(
                loan.remaining_balance for loan in lender.existing_portfolio
            )
            sector_exposure: dict[str, float] = {}
            for loan in lender.existing_portfolio:
                sector_exposure[loan.sector] = (
                    sector_exposure.get(loan.sector, 0.0) + loan.remaining_balance
                )

            available = max(0.0, lender.total_capital - baseline_deployed)
            self.lender_states[lender.id] = SeasonLenderState(
                lender_id=lender.id,
                lender_name=lender.name,
                model=lender.model,
                base_capital=lender.total_capital,
                baseline_deployed=baseline_deployed,
                deployed_capital=baseline_deployed,
                available_capital=available,
                sector_exposure=sector_exposure,
            )

    def _effective_capital(self, state: SeasonLenderState) -> float:
        cap = (
            state.base_capital
            + state.cumulative_interest
            + state.cumulative_fees
            - state.cumulative_losses
            - state.cumulative_workout_cost
        )
        return max(0.0, cap)

    def _recompute_capital(self, state: SeasonLenderState) -> None:
        state.deployed_capital = max(0.0, state.deployed_capital)
        state.available_capital = max(
            0.0,
            self._effective_capital(state) - state.deployed_capital,
        )

    def _state_net_pnl(self, state: SeasonLenderState) -> float:
        return (
            state.cumulative_interest
            + state.cumulative_fees
            - state.cumulative_losses
            - state.cumulative_workout_cost
        )

    def _total_active_loans(self) -> int:
        return sum(len(s.active_loans) for s in self.lender_states.values())

    def _reduce_exposure(self, state: SeasonLenderState, sector: str, amount: float) -> None:
        if amount <= 0:
            return
        current = state.sector_exposure.get(sector, 0.0)
        updated = max(0.0, current - amount)
        if updated <= 1e-6:
            state.sector_exposure.pop(sector, None)
        else:
            state.sector_exposure[sector] = updated

    def _extract_lender_id(self, run: UnderwritingRun) -> str:
        lid = (run.policy.params or {}).get("_lender_id", "")
        if lid:
            return str(lid)
        pid = run.policy.policy_id or ""
        parts = pid.split("_", 2)
        if len(parts) >= 2 and parts[0] == "p":
            return parts[1]
        return pid

    # ------------------------------------------------------------------
    # Cohort generation
    # ------------------------------------------------------------------
    def _mix_for_week(self, week: int) -> str:
        if self.config.season_mix != "escalating":
            return _BASE_MIX_MAP.get(self.config.season_mix, self.config.season_mix)

        first_cut = max(1, math.ceil(self.config.weeks * 0.30))
        second_cut = max(first_cut + 1, math.ceil(self.config.weeks * 0.60))
        if week <= first_cut:
            return "easy"
        if week <= second_cut:
            return "balanced"
        return "hard"

    def _draw_from_mix(self, mix: str, count: int) -> list[Borrower]:
        queue = self._mix_queues.setdefault(mix, [])
        selected: list[Borrower] = []

        while len(selected) < count:
            if not queue:
                refreshed = get_borrowers(mix=mix, seed=self._rng.randint(1, 10_000_000))
                queue.extend(refreshed)
                self._rng.shuffle(queue)
            selected.append(queue.pop())

        return selected

    def _build_week_cohort(self, week: int) -> tuple[list[Borrower], str]:
        mix_for_week = self._mix_for_week(week)
        base = self._draw_from_mix(mix_for_week, self.config.cohort_size)
        cohort: list[Borrower] = []
        for index, borrower in enumerate(base, start=1):
            cloned = copy.deepcopy(borrower)
            cloned.id = f"{borrower.id}-W{week:02d}-C{index:02d}"
            cohort.append(cloned)
        return cohort, mix_for_week

    # ------------------------------------------------------------------
    # Weekly lender construction
    # ------------------------------------------------------------------
    def _format_weekly_briefing(self, state: SeasonLenderState, week: int) -> str:
        events = state.recent_events or ["No material events since last week."]
        lines = [
            f"WEEKLY PORTFOLIO UPDATE (Week {week})",
            "================================",
            f"Available Capital: ${state.available_capital:,.0f}",
            f"Deployed Capital: ${state.deployed_capital:,.0f}",
            f"Active Loans: {len(state.active_loans)}",
            "",
            "EVENTS SINCE LAST WEEK:",
        ]
        lines.extend(f"  - {event}" for event in events[:8])

        lines.append("")
        lines.append("CURRENT SECTOR EXPOSURE:")
        effective = max(1.0, self._effective_capital(state))
        if state.sector_exposure:
            for sector, amount in sorted(
                state.sector_exposure.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                pct = amount / effective * 100
                lines.append(f"  - {sector}: ${amount:,.0f} ({pct:.1f}% of effective capital)")
        else:
            lines.append("  - None")

        lines.append("")
        lines.append("SEASON PERFORMANCE:")
        lines.append(f"  - Net P&L: ${self._state_net_pnl(state):,.0f}")
        lines.append(
            f"  - Deals Won: {state.deals_won} / Rejected: {state.deals_rejected} / Lost: {state.deals_lost}"
        )
        defaults = sum(1 for outcome in state.resolved_loans if outcome.defaulted)
        frauds = sum(1 for outcome in state.resolved_loans if outcome.was_fraud)
        lines.append(f"  - Defaults: {defaults} / Frauds Funded: {frauds}")
        return "\n".join(lines)

    def _build_week_lenders(self, week: int) -> list[LenderConfig]:
        week_lenders: list[LenderConfig] = []
        for base in self.base_lenders:
            state = self.lender_states[base.id]
            lender = copy.deepcopy(base)

            lender.total_capital = round(self._effective_capital(state), 2)
            dynamic_existing = [
                ExistingLoan(
                    borrower_name=loan.borrower_name,
                    sector=loan.sector,
                    original_amount=loan.original_principal,
                    remaining_balance=round(loan.principal_outstanding, 2),
                    interest_rate=loan.interest_rate_pct,
                    months_remaining=max(0, loan.term_months - loan.months_paid),
                )
                for loan in state.active_loans
            ]
            lender.existing_portfolio = copy.deepcopy(base.existing_portfolio) + dynamic_existing
            lender.persona = f"{base.persona}\n\n{self._format_weekly_briefing(state, week)}"
            week_lenders.append(lender)

        return week_lenders

    # ------------------------------------------------------------------
    # Rolling resolution
    # ------------------------------------------------------------------
    def _monthly_payment(self, principal: float, rate_pct: float, term_months: int) -> float:
        if term_months <= 0:
            return principal
        monthly_rate = rate_pct / 100.0 / 12.0
        if monthly_rate <= 0:
            return principal / term_months
        return principal * (
            monthly_rate * (1 + monthly_rate) ** term_months
        ) / ((1 + monthly_rate) ** term_months - 1)

    def _resolve_week(self) -> None:
        for state in self.lender_states.values():
            resolution_events: list[str] = []
            closed: list[ActiveLoanState] = []

            for loan in state.active_loans:
                outcome = self._advance_loan(
                    state=state,
                    loan=loan,
                    months=self.config.months_per_week,
                    events=resolution_events,
                )
                if outcome:
                    state.resolved_loans.append(outcome)
                    self.loan_outcomes.append(outcome)
                    closed.append(loan)

            if closed:
                state.active_loans = [loan for loan in state.active_loans if loan not in closed]

            state.recent_events = state.pending_events + resolution_events
            state.pending_events = []
            self._recompute_capital(state)

    def _advance_loan(
        self,
        state: SeasonLenderState,
        loan: ActiveLoanState,
        months: int,
        events: list[str],
    ) -> LoanOutcome | None:
        payment = self._monthly_payment(
            principal=loan.original_principal,
            rate_pct=loan.interest_rate_pct,
            term_months=max(1, loan.term_months),
        )
        monthly_rate = loan.interest_rate_pct / 100.0 / 12.0
        default_month = loan.months_before_default or min(6, max(1, loan.term_months))

        for _ in range(months):
            if loan.status != "active":
                break
            if loan.principal_outstanding <= 1e-6:
                break

            interest = loan.principal_outstanding * monthly_rate
            principal_paid = max(0.0, payment - interest)
            principal_paid = min(principal_paid, loan.principal_outstanding)

            loan.principal_outstanding -= principal_paid
            loan.months_paid += 1
            loan.total_interest_paid += interest
            loan.total_principal_paid += principal_paid

            state.cumulative_interest += interest
            state.deployed_capital -= principal_paid
            self._reduce_exposure(state, loan.sector, principal_paid)

            if loan.true_outcome == "bad" and loan.months_paid >= default_month:
                outstanding = max(0.0, loan.principal_outstanding)
                recovery = outstanding * self.economics.recovery_rate_bad
                workout_cost = outstanding * self.economics.workout_cost_rate
                principal_lost = max(0.0, outstanding - recovery)

                state.deployed_capital -= outstanding
                self._reduce_exposure(state, loan.sector, outstanding)
                state.cumulative_losses += principal_lost
                state.cumulative_workout_cost += workout_cost
                loan.status = "defaulted"

                outcome = LoanOutcome(
                    loan_id=loan.loan_id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    sector=loan.sector,
                    principal=loan.original_principal,
                    total_interest_paid=round(loan.total_interest_paid, 2),
                    principal_recovered=round(loan.total_principal_paid + recovery, 2),
                    principal_lost=round(principal_lost, 2),
                    defaulted=True,
                    was_fraud=False,
                    months_paid=loan.months_paid,
                    total_fees_paid=round(loan.total_fees_paid, 2),
                    recovery_amount=round(recovery, 2),
                    workout_cost=round(workout_cost, 2),
                    prepaid=False,
                )
                events.append(
                    f"{loan.loan_id} ({loan.borrower_name}): DEFAULT at month {loan.months_paid}; "
                    f"lost ${principal_lost:,.0f}"
                )
                self._recompute_capital(state)
                return outcome

            if loan.principal_outstanding <= 1e-6 or loan.months_paid >= loan.term_months:
                if loan.principal_outstanding > 0:
                    balloon = loan.principal_outstanding
                    loan.total_principal_paid += balloon
                    state.deployed_capital -= balloon
                    self._reduce_exposure(state, loan.sector, balloon)
                    loan.principal_outstanding = 0.0

                loan.status = "repaid"
                outcome = LoanOutcome(
                    loan_id=loan.loan_id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    sector=loan.sector,
                    principal=loan.original_principal,
                    total_interest_paid=round(loan.total_interest_paid, 2),
                    principal_recovered=round(loan.original_principal, 2),
                    principal_lost=0.0,
                    defaulted=False,
                    was_fraud=False,
                    months_paid=loan.months_paid,
                    total_fees_paid=round(loan.total_fees_paid, 2),
                    recovery_amount=0.0,
                    workout_cost=0.0,
                    prepaid=False,
                )
                events.append(
                    f"{loan.loan_id} ({loan.borrower_name}): Performing; "
                    f"${loan.total_interest_paid:,.0f} interest collected to date"
                )
                self._recompute_capital(state)
                return outcome

        self._recompute_capital(state)
        return None

    # ------------------------------------------------------------------
    # Ingest weekly simulation results
    # ------------------------------------------------------------------
    def _next_loan_id(self) -> str:
        self._loan_counter += 1
        return f"SEASON-LOAN-{self._loan_counter:05d}"

    def _book_season_loan(
        self,
        loan: BookedLoan,
    ) -> str:
        state = self.lender_states[loan.lender_id]
        season_loan_id = self._next_loan_id()
        orig_fee = max(0.0, loan.principal * self.economics.origination_fee_rate)

        season_loan = BookedLoan(
            id=season_loan_id,
            borrower_id=loan.borrower_id,
            lender_id=loan.lender_id,
            borrower_name=loan.borrower_name,
            sector=loan.sector,
            principal=loan.principal,
            interest_rate=loan.interest_rate,
            term_months=loan.term_months,
            true_outcome=loan.true_outcome,
            months_before_default=loan.months_before_default,
        )
        self.booked_loans.append(season_loan)

        state.deals_won += 1
        state.cumulative_fees += orig_fee
        state.deployed_capital += loan.principal
        state.sector_exposure[loan.sector] = (
            state.sector_exposure.get(loan.sector, 0.0) + loan.principal
        )

        if loan.true_outcome == "fraud":
            recovery = loan.principal * self.economics.recovery_rate_fraud
            workout_cost = loan.principal * self.economics.workout_cost_rate
            principal_lost = max(0.0, loan.principal - recovery)

            state.deployed_capital -= loan.principal
            self._reduce_exposure(state, loan.sector, loan.principal)
            state.cumulative_losses += principal_lost
            state.cumulative_workout_cost += workout_cost

            outcome = LoanOutcome(
                loan_id=season_loan_id,
                lender_id=loan.lender_id,
                borrower_name=loan.borrower_name,
                sector=loan.sector,
                principal=loan.principal,
                total_interest_paid=0.0,
                principal_recovered=round(recovery, 2),
                principal_lost=round(principal_lost, 2),
                defaulted=True,
                was_fraud=True,
                months_paid=0,
                total_fees_paid=round(orig_fee, 2),
                recovery_amount=round(recovery, 2),
                workout_cost=round(workout_cost, 2),
                prepaid=False,
            )
            state.resolved_loans.append(outcome)
            self.loan_outcomes.append(outcome)
            state.pending_events.append(
                f"{season_loan_id} ({loan.borrower_name}): FRAUD default on booking; "
                f"lost ${principal_lost:,.0f}"
            )
        else:
            active = ActiveLoanState(
                loan_id=season_loan_id,
                lender_id=loan.lender_id,
                borrower_id=loan.borrower_id,
                borrower_name=loan.borrower_name,
                sector=loan.sector,
                original_principal=loan.principal,
                principal_outstanding=loan.principal,
                interest_rate_pct=loan.interest_rate,
                term_months=loan.term_months,
                true_outcome=loan.true_outcome,
                months_before_default=loan.months_before_default,
                total_fees_paid=orig_fee,
            )
            state.active_loans.append(active)
            state.pending_events.append(
                f"{season_loan_id} ({loan.borrower_name}): booked ${loan.principal:,.0f} "
                f"@ {loan.interest_rate:.2f}% for {loan.term_months}mo"
            )

        self._recompute_capital(state)
        return season_loan_id

    def _ingest_week_results(self, engine: SimulationEngine) -> None:
        for lender_id, decisions in engine.all_decisions.items():
            state = self.lender_states[lender_id]
            self.all_decisions[lender_id].extend(decisions)
            state.total_evaluations += len(decisions)
            state.deals_rejected += sum(1 for d in decisions if d.decision == "REJECT")

            for decision in decisions:
                if decision.decision != "APPROVE":
                    continue
                result = engine.deal_results.get(decision.borrower_id, {})
                if result.get("outcome") == "booked" and result.get("winner") != lender_id:
                    state.deals_lost += 1

        for borrower_id, result in engine.deal_results.items():
            self.deal_results[borrower_id] = dict(result)

        self.runs.extend(engine.runs)
        for run in engine.runs:
            lender_id = self._extract_lender_id(run)
            if lender_id not in self.lender_states:
                continue
            tool_calls = sum(
                1 for step in run.trace.steps if getattr(step, "type", "") == "tool_call"
            )
            self.lender_states[lender_id].total_tool_calls += tool_calls

        for loan in engine.booked_loans:
            season_loan_id = self._book_season_loan(loan)
            result = self.deal_results.get(loan.borrower_id)
            if result and result.get("loan_id") == loan.id:
                result["loan_id"] = season_loan_id

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _record_week_snapshots(self, week: int) -> None:
        for state in self.lender_states.values():
            defaults = sum(1 for o in state.resolved_loans if o.defaulted)
            frauds = sum(1 for o in state.resolved_loans if o.was_fraud)
            state.weekly_snapshots.append(
                {
                    "week": week,
                    "effective_capital": round(self._effective_capital(state), 2),
                    "available_capital": round(state.available_capital, 2),
                    "deployed_capital": round(state.deployed_capital, 2),
                    "active_loans": len(state.active_loans),
                    "defaults_to_date": defaults,
                    "frauds_to_date": frauds,
                    "net_pnl": round(self._state_net_pnl(state), 2),
                }
            )

    def _print_week_summary(self, week: int) -> None:
        print("\n" + "-" * 70)
        print(f"WEEK {week} SUMMARY")
        print("-" * 70)
        for lender in self.base_lenders:
            state = self.lender_states[lender.id]
            defaults = sum(1 for o in state.resolved_loans if o.defaulted)
            print(
                f"{state.lender_name}: available=${state.available_capital:,.0f}, "
                f"deployed=${state.deployed_capital:,.0f}, active={len(state.active_loans)}, "
                f"net_pnl=${self._state_net_pnl(state):,.0f}, defaults={defaults}"
            )

    def _final_resolution(self) -> None:
        print("\n" + "=" * 70)
        print("FINAL RESOLUTION")
        print("=" * 70)

        rounds = 0
        while self._total_active_loans() > 0 and rounds < 240:
            rounds += 1
            self._resolve_week()
            print(f"  Fast-forward round {rounds}: active loans remaining={self._total_active_loans()}")

        if self._total_active_loans() > 0:
            print("  WARNING: active loans remain after max final-resolution rounds.")

    def _print_season_report(self) -> None:
        print("\n" + "=" * 70)
        print("SEASON REPORT")
        print("=" * 70)
        ranked = sorted(
            self.lender_states.values(),
            key=lambda state: self._state_net_pnl(state),
            reverse=True,
        )

        for rank, state in enumerate(ranked, start=1):
            defaults = sum(1 for outcome in state.resolved_loans if outcome.defaulted)
            frauds = sum(1 for outcome in state.resolved_loans if outcome.was_fraud)
            efficiency = (
                state.total_evaluations / state.total_tool_calls
                if state.total_tool_calls > 0
                else 0.0
            )
            print("\n" + "─" * 60)
            print(f"#{rank} {state.lender_name} ({state.model})")
            print("─" * 60)
            print(f"  Effective Capital: ${self._effective_capital(state):,.0f}")
            print(f"  Available Capital: ${state.available_capital:,.0f}")
            print(f"  Deployed Capital:  ${state.deployed_capital:,.0f}")
            print(f"  Deals Won/Lost/Rejected: {state.deals_won}/{state.deals_lost}/{state.deals_rejected}")
            print(f"  Active Loans: {len(state.active_loans)}")
            print(f"  Defaults: {defaults} | Frauds Funded: {frauds}")
            print(f"  Net P&L: ${self._state_net_pnl(state):,.0f}")
            if state.total_tool_calls > 0:
                print(
                    f"  Efficiency: {efficiency:.3f} eval/tool "
                    f"({state.total_evaluations} evals, {state.total_tool_calls} tool calls)"
                )

