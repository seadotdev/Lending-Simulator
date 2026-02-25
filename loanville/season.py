"""
Season mode: multi-week game where capital, exposures, and loan outcomes
carry forward across weeks.
"""

import copy
from collections import Counter

from .borrower_gen import generate_cohort
from .cost_tracking import print_season_cost_summary
from .custom_tools import LenderToolkit
from .engine import SimulationEngine, resolve_loan_period
from .models import (
    ActiveLoan,
    BookedLoan,
    EconomicsConfig,
    ExistingLoan,
    LenderConfig,
    LoanOutcome,
    SeasonConfig,
    SeasonLenderState,
    WeekResult,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_season_header(config: SeasonConfig) -> None:
    print("\n" + "=" * 70)
    print("  LOANVILLE — SEASON MODE")
    print("=" * 70)
    print(f"  Weeks: {config.weeks} | Cohort size: {config.cohort_size} | "
          f"Months/week: {config.months_per_week}")
    print(f"  Mix: {config.season_mix} | Seed: {config.seed}")
    flags = []
    if config.speed_scoring:
        flags.append("speed-scoring")
    if config.custom_tools:
        flags.append("custom-tools")
    if config.capital_adequacy_ratio > 0:
        flags.append(f"capital-adequacy({config.capital_adequacy_ratio:.0%})")
    if config.capital_decay_rate > 0:
        flags.append(f"capital-decay({config.capital_decay_rate:.1%}/wk)")
    if config.info_asymmetry != "none":
        flags.append(f"info-asymmetry({config.info_asymmetry})")
    if flags:
        print(f"  Features: {', '.join(flags)}")
    print("=" * 70)


def _print_week_header(week: int, total: int) -> None:
    print(f"\n{'#' * 70}")
    print(f"#  WEEK {week}/{total}")
    print(f"{'#' * 70}")


def _print_cohort_summary(week: int, cohort) -> None:
    outcomes = Counter(b.true_outcome for b in cohort)
    parts = []
    for k in ["good", "bad", "fraud"]:
        n = outcomes.get(k, 0)
        if n > 0:
            parts.append(f"{n} {k}")
    print(f"\n  Week {week} cohort: {len(cohort)} borrowers ({', '.join(parts)})")
    for b in cohort:
        print(f"    {b.id}: {b.dossier.company_name} ({b.dossier.sector}) "
              f"requesting ${b.dossier.loan_request_amount:,.0f}")


# ---------------------------------------------------------------------------
# SeasonEngine
# ---------------------------------------------------------------------------

class SeasonEngine:
    def __init__(
        self,
        config: SeasonConfig,
        lenders: list[LenderConfig],
        openrouter_api_key: str = "",
        mock: bool = False,
        data_mode: str = "full",
        use_los: bool = False,
        los_url: str = "http://localhost:3000",
        los_provider: str = "openrouter",
        los_mode: str = "rules_only",
        underwrite_only: bool = False,
        los_model: str | None = None,
    ):
        self.config = config
        self.base_lenders = lenders
        self.lender_states: dict[str, SeasonLenderState] = {}
        self.toolkits: dict[str, LenderToolkit] = {}
        self.week_results: list[WeekResult] = []
        self.used_static_ids: set[str] = set()

        # Accumulated data for leaderboard integration.
        # NOTE: these grow linearly with weeks*cohort_size. For large seasons
        # (many weeks, large cohorts) this could use significant memory.
        self.all_decisions: dict[str, list] = {}   # lender_id -> [LenderDecision, ...]
        self.all_borrowers: list = []               # all borrowers across all weeks
        self.all_deal_results: dict[str, dict] = {} # borrower_id -> deal result

        # Pass-through kwargs for SimulationEngine
        self.engine_kwargs = dict(
            openrouter_api_key=openrouter_api_key,
            mock=mock,
            data_mode=data_mode,
            use_los=use_los,
            los_url=los_url,
            los_provider=los_provider,
            los_mode=los_mode,
            underwrite_only=underwrite_only,
            los_model=los_model,
            economics=config.economics,
            info_asymmetry=config.info_asymmetry,
        )

        self._init_states(lenders)

    def _init_states(self, lenders: list[LenderConfig]) -> None:
        for lender in lenders:
            existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
            self.lender_states[lender.id] = SeasonLenderState(
                lender_id=lender.id,
                lender_name=lender.name,
                model=lender.model,
                total_capital=lender.total_capital,
                deployed_capital=existing_deployed,
                available_capital=lender.total_capital - existing_deployed,
            )
            self.toolkits[lender.id] = LenderToolkit(lender.id)

    def _effective_capital(self, state: SeasonLenderState) -> float:
        """Capital base adjusted for cumulative P&L — interest and fees grow it,
        losses shrink it, decay erodes idle capital."""
        return max(0.0,
                   state.total_capital
                   + state.cumulative_interest
                   + state.cumulative_fees
                   - state.cumulative_losses
                   - state.cumulative_decay)

    def _recompute_available(self, state: SeasonLenderState) -> None:
        """Recompute available capital from effective capital minus deployed."""
        state.available_capital = max(0.0,
                                      self._effective_capital(state)
                                      - state.deployed_capital)

    def _apply_capital_decay(self, week: int) -> list[str]:
        """Apply time-value decay to undeployed capital.

        Undeployed capital loses value each week, creating tension between
        deploying early and waiting for better opportunities (inspired by
        Skirmish's energy economy where idle resources lose value).
        """
        decay_rate = self.config.capital_decay_rate
        if decay_rate <= 0.0:
            return []

        events = []
        for lid, state in self.lender_states.items():
            if state.eliminated:
                continue
            idle_capital = max(0.0, state.available_capital)
            decay_amount = idle_capital * decay_rate
            if decay_amount > 0:
                state.cumulative_decay += decay_amount
                self._recompute_available(state)
                events.append(
                    f"  [{state.lender_name}] Capital decay: "
                    f"-${decay_amount:,.0f} on ${idle_capital:,.0f} idle "
                    f"(cumulative: ${state.cumulative_decay:,.0f})"
                )
        return events

    def _check_capital_adequacy(self, week: int) -> list[str]:
        """Eliminate lenders whose effective capital falls below the threshold.

        Mirrors Skirmish's spawn destruction mechanic and real banking
        capital adequacy requirements.  Eliminated lenders no longer
        participate in subsequent weeks.
        """
        threshold = self.config.capital_adequacy_ratio
        if threshold <= 0.0:
            return []

        events = []
        for lid, state in self.lender_states.items():
            if state.eliminated:
                continue
            eff = self._effective_capital(state)
            min_capital = state.total_capital * threshold
            if eff < min_capital:
                state.eliminated = True
                state.eliminated_week = week
                events.append(
                    f"  *** [{state.lender_name}] ELIMINATED in week {week} — "
                    f"effective capital ${eff:,.0f} < "
                    f"${min_capital:,.0f} ({threshold:.0%} of initial) ***"
                )
        return events

    @property
    def _active_lender_ids(self) -> set[str]:
        """IDs of lenders not yet eliminated."""
        return {
            lid for lid, state in self.lender_states.items()
            if not state.eliminated
        }

    # ------------------------------------------------------------------
    # Main season loop
    # ------------------------------------------------------------------

    async def run_season(self) -> None:
        _print_season_header(self.config)

        for week in range(1, self.config.weeks + 1):
            _print_week_header(week, self.config.weeks)

            # Check if all lenders eliminated (early termination)
            if not self._active_lender_ids:
                print("\n  All lenders eliminated — ending season early.")
                break

            # 1. Resolve aging loans
            events = self._resolve_week(week)

            # 1a. Apply capital time-value decay on idle capital
            decay_events = self._apply_capital_decay(week)
            events.extend(decay_events)
            for e in decay_events:
                print(e)

            # 1b. Check capital adequacy — eliminate bankrupt lenders
            elim_events = self._check_capital_adequacy(week)
            events.extend(elim_events)
            for e in elim_events:
                print(e)

            # 2. Build and print portfolio briefings (only for active lenders)
            briefings = self._build_briefings(week, events)
            for lid, briefing in briefings.items():
                print(briefing)

            # 3. Tooling phase (if enabled)
            if self.config.custom_tools:
                self._tooling_phase(week)

            # 4. Generate cohort
            cohort = generate_cohort(week, self.config, self.used_static_ids)
            _print_cohort_summary(week, cohort)

            # 5. Build week lenders (inject briefing + available capital)
            #    Only include active (non-eliminated) lenders
            week_lenders = self._build_week_lenders(briefings)
            active_ids = self._active_lender_ids
            week_lenders = [l for l in week_lenders if l.id in active_ids]

            if not week_lenders:
                print("\n  No active lenders remaining — skipping origination.")
                continue

            # 6. Run origination via SimulationEngine
            engine = SimulationEngine(cohort, week_lenders, **self.engine_kwargs)
            await engine.run_origination()

            # 7. Adjudicate with season capital + optional speed scoring
            capital = self._get_remaining_capital()
            tool_counts = self._extract_tool_counts(engine) if self.config.speed_scoring else None
            engine.adjudicate_deals(
                remaining_capital=capital,
                speed_scoring=self.config.speed_scoring,
                tool_call_counts=tool_counts,
            )

            # 8. Ingest results
            week_result = self._ingest_results(week, engine)
            week_result.events = events
            self.week_results.append(week_result)

            # 9. Snapshot utilization + weekly analytics
            self._snapshot_utilization()
            self._record_week_snapshots(week)

        # Final resolution — fast-forward remaining active loans
        self._final_resolution()
        self._print_season_report()

    # ------------------------------------------------------------------
    # Loan resolution
    # ------------------------------------------------------------------

    def _resolve_week(self, week: int) -> list[str]:
        """Advance all performing loans by months_per_week."""
        events = []
        eco = self.config.economics

        for state in self.lender_states.values():
            still_active = []
            for loan in state.active_loans:
                if loan.status != "performing":
                    still_active.append(loan)
                    continue

                result = resolve_loan_period(
                    principal=loan.principal,
                    interest_rate=loan.interest_rate,
                    term_months=loan.term_months,
                    true_outcome=loan.true_outcome,
                    months_before_default=loan.months_before_default,
                    months_already_elapsed=loan.months_elapsed,
                    months_to_advance=self.config.months_per_week,
                    economics=eco,
                )

                loan.months_elapsed += result.months_actually_advanced
                loan.total_interest_collected += result.interest
                loan.total_principal_repaid += result.principal_repaid
                loan.total_fees_collected += result.fees
                loan.remaining_balance = result.remaining_balance

                if result.defaulted:
                    loan.status = "defaulted"
                    state.cumulative_interest += result.interest
                    state.cumulative_losses += result.principal_lost
                    principal_resolved = (
                        result.principal_repaid
                        + result.recovery_amount
                        + result.principal_lost
                    )
                    state.deployed_capital = max(
                        0.0, state.deployed_capital - principal_resolved
                    )
                    state.cumulative_fees += result.fees
                    self._recompute_available(state)
                    events.append(
                        f"DEFAULT: {loan.borrower_name} ({loan.sector}) — "
                        f"${result.principal_lost:,.0f} lost, "
                        f"${result.recovery_amount:,.0f} recovered "
                        f"[{state.lender_name}]"
                    )
                    state.resolved_loans.append(LoanOutcome(
                        loan_id=loan.loan_id,
                        lender_id=loan.lender_id,
                        borrower_name=loan.borrower_name,
                        borrower_id=loan.borrower_id,
                        sector=loan.sector,
                        principal=loan.principal,
                        total_interest_paid=loan.total_interest_collected,
                        principal_recovered=round(loan.total_principal_repaid + result.recovery_amount, 2),
                        principal_lost=round(result.principal_lost, 2),
                        defaulted=True,
                        was_fraud=(loan.true_outcome == "fraud"),
                        months_paid=loan.months_elapsed,
                        total_fees_paid=round(loan.total_fees_collected, 2),
                        recovery_amount=result.recovery_amount,
                        workout_cost=result.workout_cost,
                    ))
                elif result.matured or result.prepaid:
                    loan.status = "repaid" if result.matured else "prepaid"
                    state.deployed_capital = max(
                        0.0, state.deployed_capital - result.principal_repaid
                    )
                    state.cumulative_interest += result.interest
                    state.cumulative_fees += result.fees
                    self._recompute_available(state)
                    label = "REPAID" if result.matured else "PREPAID"
                    events.append(
                        f"{label}: {loan.borrower_name} ({loan.sector}) — "
                        f"${loan.total_interest_collected:,.0f} total interest "
                        f"[{state.lender_name}]"
                    )
                    state.resolved_loans.append(LoanOutcome(
                        loan_id=loan.loan_id,
                        lender_id=loan.lender_id,
                        borrower_name=loan.borrower_name,
                        borrower_id=loan.borrower_id,
                        sector=loan.sector,
                        principal=loan.principal,
                        total_interest_paid=loan.total_interest_collected,
                        principal_recovered=round(loan.principal, 2),
                        principal_lost=0.0,
                        defaulted=False,
                        was_fraud=False,
                        months_paid=loan.months_elapsed,
                        total_fees_paid=round(loan.total_fees_collected, 2),
                        recovery_amount=0.0,
                        workout_cost=0.0,
                        prepaid=result.prepaid,
                    ))
                else:
                    # Still performing — collect interest + principal payments
                    state.cumulative_interest += result.interest
                    state.cumulative_fees += result.fees
                    state.deployed_capital = max(
                        0.0, state.deployed_capital - result.principal_repaid
                    )
                    self._recompute_available(state)
                    if result.interest > 0 or result.principal_repaid > 0:
                        events.append(
                            f"PAYMENT: {loan.borrower_name} — "
                            f"${result.interest:,.0f} interest, "
                            f"${result.principal_repaid:,.0f} principal "
                            f"[{state.lender_name}]"
                        )
                    still_active.append(loan)

            state.active_loans = still_active
            self._recompute_sector_exposure(state)

        return events

    def _recompute_sector_exposure(self, state: SeasonLenderState) -> None:
        exposure: dict[str, float] = {}
        lender = next((l for l in self.base_lenders if l.id == state.lender_id), None)
        if lender:
            for existing in lender.existing_portfolio:
                exposure[existing.sector] = (
                    exposure.get(existing.sector, 0.0) + existing.remaining_balance
                )
        for loan in state.active_loans:
            if loan.status == "performing":
                exposure[loan.sector] = exposure.get(loan.sector, 0.0) + loan.remaining_balance
        state.sector_exposure = exposure

    # ------------------------------------------------------------------
    # Briefings
    # ------------------------------------------------------------------

    def _build_briefings(self, week: int, events: list[str]) -> dict[str, str]:
        """Build weekly portfolio update text per lender."""
        briefings: dict[str, str] = {}

        for lid, state in self.lender_states.items():
            eff_cap = self._effective_capital(state)
            lines = [
                f"\n--- PORTFOLIO BRIEFING: Week {week} ---",
                f"Effective Capital: ${eff_cap:,.0f} (base ${state.total_capital:,.0f} + P&L)",
                f"Available Capital: ${state.available_capital:,.0f}",
                f"Deployed Capital: ${state.deployed_capital:,.0f}",
                f"Active Loans: {len(state.active_loans)}",
            ]

            if state.active_loans:
                lines.append("Current Portfolio:")
                for loan in state.active_loans:
                    lines.append(
                        f"  - {loan.borrower_name} ({loan.sector}): "
                        f"${loan.remaining_balance:,.0f} remaining, "
                        f"{loan.months_elapsed}/{loan.term_months} months"
                    )

            if state.sector_exposure:
                lines.append("Sector Exposure:")
                for sector, amt in sorted(state.sector_exposure.items(),
                                          key=lambda x: -x[1]):
                    pct = amt / state.total_capital * 100 if state.total_capital > 0 else 0
                    lines.append(f"  - {sector}: ${amt:,.0f} ({pct:.1f}%)")

            # Filter events for this lender
            lender_events = [e for e in events if state.lender_name in e]
            if lender_events:
                lines.append("Events This Week:")
                for e in lender_events:
                    lines.append(f"  {e}")

            # Season P&L summary
            total_pnl = (state.cumulative_interest + state.cumulative_fees
                         - state.cumulative_losses)
            lines.append(
                f"Season P&L: ${total_pnl:,.0f} "
                f"(interest: ${state.cumulative_interest:,.0f}, "
                f"fees: ${state.cumulative_fees:,.0f}, "
                f"losses: -${state.cumulative_losses:,.0f})"
            )
            lines.append("---")

            briefings[lid] = "\n".join(lines)

        return briefings

    # ------------------------------------------------------------------
    # Week lender construction
    # ------------------------------------------------------------------

    def _build_week_lenders(self, briefings: dict[str, str]) -> list[LenderConfig]:
        """Create LenderConfig copies with briefing prepended to persona and
        existing_portfolio reflecting active season loans."""
        week_lenders = []
        for lender in self.base_lenders:
            state = self.lender_states[lender.id]
            briefing = briefings.get(lender.id, "")

            # Build existing_portfolio from static base book + active season loans
            existing = copy.deepcopy(lender.existing_portfolio)
            for loan in state.active_loans:
                if loan.status == "performing":
                    existing.append(ExistingLoan(
                        borrower_name=loan.borrower_name,
                        sector=loan.sector,
                        original_amount=loan.principal,
                        remaining_balance=loan.remaining_balance,
                        interest_rate=loan.interest_rate,
                        months_remaining=loan.term_months - loan.months_elapsed,
                    ))

            week_lender = LenderConfig(
                id=lender.id,
                name=lender.name,
                persona=briefing + "\n\n" + lender.persona,
                model=lender.model,
                target_yield_pct=lender.target_yield_pct,
                max_single_loan=lender.max_single_loan,
                total_capital=state.total_capital,
                sector_limits=lender.sector_limits,
                existing_portfolio=existing,
            )
            week_lenders.append(week_lender)
        return week_lenders

    # ------------------------------------------------------------------
    # Capital + tool tracking
    # ------------------------------------------------------------------

    def _get_remaining_capital(self) -> dict[str, float]:
        return {lid: max(0.0, self._effective_capital(state) - state.deployed_capital)
                for lid, state in self.lender_states.items()}

    def _extract_tool_counts(self, engine: SimulationEngine) -> dict[tuple[str, str], int]:
        """Extract tool call counts per (lender_id, borrower_id) from engine runs."""
        counts: dict[tuple[str, str], int] = {}
        for run in engine.runs:
            lid = (run.policy.params or {}).get("_lender_id", "")
            bid = run.case.case_id
            if lid and bid:
                if run.trace and run.trace.steps:
                    tool_calls = sum(
                        1
                        for step in run.trace.steps
                        if getattr(step, "type", "") == "tool_call"
                    )
                else:
                    tool_calls = 0
                counts[(lid, bid)] = tool_calls
        return counts

    # ------------------------------------------------------------------
    # Tooling phase
    # ------------------------------------------------------------------

    def _tooling_phase(self, week: int) -> None:
        """Tooling phase between weeks. In mock mode, skip."""
        if self.engine_kwargs.get("mock"):
            return
        # For live mode, this would send a prompt to each lender's LLM
        # asking them to create/update tools. Placeholder for now.
        print(f"\n  [Tooling phase — week {week}] (not active in current mode)")

    # ------------------------------------------------------------------
    # Result ingestion
    # ------------------------------------------------------------------

    def _ingest_results(self, week: int, engine: SimulationEngine) -> WeekResult:
        """Convert booked loans to ActiveLoans, update state."""
        loans_booked = 0
        tool_counts = self._extract_tool_counts(engine)

        for loan in engine.booked_loans:
            active = ActiveLoan(
                loan_id=loan.id,
                borrower_id=loan.borrower_id,
                borrower_name=loan.borrower_name,
                lender_id=loan.lender_id,
                sector=loan.sector,
                principal=loan.principal,
                interest_rate=loan.interest_rate,
                term_months=loan.term_months,
                true_outcome=loan.true_outcome,
                months_before_default=loan.months_before_default,
                booked_week=week,
                remaining_balance=loan.principal,
            )
            state = self.lender_states[loan.lender_id]
            state.active_loans.append(active)
            state.deployed_capital += loan.principal
            state.deals_won += 1
            self._recompute_available(state)
            loans_booked += 1

        # Count rejections and losses for each lender
        booked_bids = {loan.borrower_id for loan in engine.booked_loans}
        for lender in self.base_lenders:
            state = self.lender_states[lender.id]
            decisions = engine.all_decisions.get(lender.id, [])
            for d in decisions:
                state.total_evaluations += 1
                state.total_tool_calls += tool_counts.get((lender.id, d.borrower_id), 0)
                if d.decision != "APPROVE":
                    state.deals_rejected += 1
                elif d.borrower_id in booked_bids:
                    # Check if this lender won or lost
                    winner = next(
                        (l for l in engine.booked_loans if l.borrower_id == d.borrower_id),
                        None,
                    )
                    if winner and winner.lender_id != lender.id:
                        state.deals_lost += 1

        if self.config.speed_scoring:
            for loan in engine.booked_loans:
                deal = engine.deal_results.get(loan.borrower_id, {})
                if deal.get("speed_bonus_decisive"):
                    self.lender_states[loan.lender_id].speed_wins += 1

        for state in self.lender_states.values():
            self._recompute_sector_exposure(state)

        # Accumulate data for leaderboard
        for lid, decisions in engine.all_decisions.items():
            self.all_decisions.setdefault(lid, []).extend(decisions)
        self.all_borrowers.extend(engine.borrowers)
        self.all_deal_results.update(engine.deal_results)

        return WeekResult(
            week=week,
            cohort_size=len(engine.borrowers),
            loans_booked=loans_booked,
            defaults_this_week=0,  # filled by caller from events
            repayments_this_week=0,
        )

    # ------------------------------------------------------------------
    # Utilization snapshot
    # ------------------------------------------------------------------

    def _snapshot_utilization(self) -> None:
        """Record utilization for each lender this week."""
        for state in self.lender_states.values():
            if state.total_capital > 0:
                util = state.deployed_capital / state.total_capital
            else:
                util = 0.0
            state.weekly_utilization.append(util)

            # Check concentration violations — only count if the season's own
            # lending pushed a sector over the limit. Inherited concentration from
            # the existing portfolio is not penalized.
            lender = next(
                (l for l in self.base_lenders if l.id == state.lender_id), None
            )
            if lender and state.total_capital > 0:
                baseline_exposure: dict[str, float] = {}
                for ex in lender.existing_portfolio:
                    baseline_exposure[ex.sector] = (
                        baseline_exposure.get(ex.sector, 0.0) + ex.remaining_balance
                    )
                violated = False
                for sector, exposure in state.sector_exposure.items():
                    limit = lender.sector_limits.get(sector, 1.0)
                    pct = exposure / state.total_capital
                    baseline_pct = baseline_exposure.get(sector, 0.0) / lender.total_capital
                    if pct > limit and pct > baseline_pct:
                        violated = True
                        break
                if violated:
                    state.weeks_with_concentration_violations += 1

    def _record_week_snapshots(self, week: int) -> None:
        """Save structured per-week snapshot for analytics."""
        for state in self.lender_states.values():
            defaults = sum(1 for o in state.resolved_loans if o.defaulted)
            frauds = sum(1 for o in state.resolved_loans if o.was_fraud)
            state.weekly_snapshots.append({
                "week": week,
                "effective_capital": round(self._effective_capital(state), 2),
                "available_capital": round(state.available_capital, 2),
                "deployed_capital": round(state.deployed_capital, 2),
                "active_loans": len(state.active_loans),
                "defaults_to_date": defaults,
                "frauds_to_date": frauds,
                "net_pnl": round(
                    state.cumulative_interest + state.cumulative_fees
                    - state.cumulative_losses, 2
                ),
            })

    # ------------------------------------------------------------------
    # Final resolution
    # ------------------------------------------------------------------

    def _final_resolution(self) -> None:
        """Fast-forward all remaining active loans to maturity."""
        print(f"\n{'=' * 70}")
        print("FINAL RESOLUTION: Fast-forwarding remaining loans")
        print(f"{'=' * 70}")

        eco = self.config.economics
        for state in self.lender_states.values():
            for loan in list(state.active_loans):
                if loan.status != "performing":
                    continue

                remaining_months = loan.term_months - loan.months_elapsed
                if remaining_months <= 0:
                    continue

                result = resolve_loan_period(
                    principal=loan.principal,
                    interest_rate=loan.interest_rate,
                    term_months=loan.term_months,
                    true_outcome=loan.true_outcome,
                    months_before_default=loan.months_before_default,
                    months_already_elapsed=loan.months_elapsed,
                    months_to_advance=remaining_months,
                    economics=eco,
                )

                loan.months_elapsed += result.months_actually_advanced
                loan.total_interest_collected += result.interest
                loan.total_principal_repaid += result.principal_repaid
                loan.total_fees_collected += result.fees
                loan.remaining_balance = result.remaining_balance

                if result.defaulted:
                    loan.status = "defaulted"
                    state.cumulative_losses += result.principal_lost
                    principal_resolved = (
                        result.principal_repaid
                        + result.recovery_amount
                        + result.principal_lost
                    )
                    state.deployed_capital = max(
                        0.0, state.deployed_capital - principal_resolved
                    )
                    was_fraud = loan.true_outcome == "fraud"
                    label = "FRAUD DEFAULT" if was_fraud else "DEFAULT"
                    print(f"  {loan.loan_id} ({loan.borrower_name}): {label} "
                          f"— ${result.principal_lost:,.0f} lost")
                else:
                    loan.status = "repaid" if result.matured else "prepaid"
                    state.deployed_capital = max(
                        0.0, state.deployed_capital - result.principal_repaid
                    )
                    print(f"  {loan.loan_id} ({loan.borrower_name}): REPAID "
                          f"— ${loan.total_interest_collected:,.0f} total interest")

                state.cumulative_interest += result.interest
                state.cumulative_fees += result.fees
                self._recompute_available(state)

                state.resolved_loans.append(LoanOutcome(
                    loan_id=loan.loan_id,
                    lender_id=loan.lender_id,
                    borrower_name=loan.borrower_name,
                    borrower_id=loan.borrower_id,
                    sector=loan.sector,
                    principal=loan.principal,
                    total_interest_paid=loan.total_interest_collected,
                    principal_recovered=round(
                        loan.total_principal_repaid + result.recovery_amount, 2
                    ),
                    principal_lost=round(result.principal_lost, 2),
                    defaulted=result.defaulted,
                    was_fraud=(loan.true_outcome == "fraud"),
                    months_paid=loan.months_elapsed,
                    total_fees_paid=round(loan.total_fees_collected, 2),
                    recovery_amount=result.recovery_amount,
                    workout_cost=result.workout_cost,
                    prepaid=result.prepaid,
                ))

            state.active_loans = [
                l for l in state.active_loans if l.status == "performing"
            ]
            self._recompute_sector_exposure(state)

    # ------------------------------------------------------------------
    # Season report
    # ------------------------------------------------------------------

    def _print_season_report(self) -> None:
        print(f"\n{'=' * 70}")
        print("  SEASON RESULTS")
        print(f"{'=' * 70}")

        for state in self.lender_states.values():
            total_pnl = (state.cumulative_interest + state.cumulative_fees
                         - state.cumulative_losses)
            avg_util = (
                sum(state.weekly_utilization) / len(state.weekly_utilization)
                if state.weekly_utilization else 0.0
            )

            print(f"\n  {state.lender_name} ({state.model})")
            print(f"    Capital: ${state.total_capital:,.0f}")
            print(f"    Deals Won: {state.deals_won} | Lost: {state.deals_lost} | "
                  f"Rejected: {state.deals_rejected}")
            print(f"    Interest Earned: ${state.cumulative_interest:,.0f}")
            print(f"    Fees Earned: ${state.cumulative_fees:,.0f}")
            print(f"    Losses: ${state.cumulative_losses:,.0f}")
            print(f"    Net P&L: ${total_pnl:,.0f}")
            print(f"    Avg Utilization: {avg_util:.1%}")
            print(f"    Concentration Violation Weeks: "
                  f"{state.weeks_with_concentration_violations}")

            # Loan breakdown
            resolved = state.resolved_loans
            defaults = sum(1 for o in resolved if o.defaulted)
            frauds = sum(1 for o in resolved if o.was_fraud)
            print(f"    Resolved Loans: {len(resolved)} "
                  f"(defaults: {defaults}, frauds funded: {frauds})")

        # Cost summary
        is_mock = self.engine_kwargs.get("mock", False)
        print_season_cost_summary(self.lender_states, self.config, mock=is_mock)
