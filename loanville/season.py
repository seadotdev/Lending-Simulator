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
from .los_adapter import bootstrap_los_tenant
from .market_pool import MarketPool
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

BANDWIDTH_LIMIT_REASON = (
    "[BANDWIDTH_LIMIT] Deferred after skim due to deep-underwrite slot cap."
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
    if config.scratchpad:
        flags.append("scratchpad")
    if config.consistency_samples > 0:
        flags.append(f"consistency(k={config.consistency_samples},"
                     f"{config.consistency_sample_pct:.0%})")
    if config.info_asymmetry != "none":
        flags.append(f"info-asymmetry({config.info_asymmetry})")
    if config.arrival_phases > 1:
        flags.append(f"arrival-phases({config.arrival_phases})")
    if config.deep_uw_slots_per_week > 0:
        flags.append(f"deep-uw-slots({config.deep_uw_slots_per_week}/wk)")
    if config.underwriting_cost_mode == "real":
        flags.append(f"real-uw-costs(×{config.uw_cost_multiplier:g})")
    if config.borrower_patience_weeks > 1:
        flags.append(f"borrower-patience({config.borrower_patience_weeks}w)")
    if config.offer_validity_weeks > 1:
        flags.append(f"offer-validity({config.offer_validity_weeks}w)")
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
        mock: bool = False,
        data_mode: str = "full",
        los_url: str = "http://localhost:3000",
        los_provider: str = "openrouter",
        los_mode: str = "rules_only",
        underwrite_only: bool = False,
        los_model: str | None = None,
        formal_los_only: bool = False,
    ):
        self.config = config
        self.base_lenders = lenders
        self.lender_states: dict[str, SeasonLenderState] = {}
        self.toolkits: dict[str, LenderToolkit] = {}
        self.week_results: list[WeekResult] = []
        self.used_static_ids: set[str] = set()
        self.week_details: list[dict] = []  # per-week JSON-serializable detail

        # Accumulated data for leaderboard integration.
        # NOTE: these grow linearly with weeks*cohort_size. For large seasons
        # (many weeks, large cohorts) this could use significant memory.
        self.all_decisions: dict[str, list] = {}   # lender_id -> [LenderDecision, ...]
        self.all_borrowers: list = []               # all borrowers across all weeks
        self.all_deal_results: dict[str, dict] = {} # borrower_id -> deal result
        self.weekly_match_data: list[dict] = []     # one entry per completed week
        self._seen_borrower_ids: set[str] = set()

        # Pass-through kwargs for SimulationEngine
        self.engine_kwargs = dict(
            mock=mock,
            data_mode=data_mode,
            los_url=los_url,
            los_provider=los_provider,
            los_mode=los_mode,
            underwrite_only=underwrite_only,
            los_model=los_model,
            formal_los_only=formal_los_only,
            economics=config.economics,
            info_asymmetry=config.info_asymmetry,
        )

        # Per-week competition stats for briefing feedback
        # lid -> {won, lost, rejected, avg_offered_rate, avg_winning_rate}
        self._last_week_stats: dict[str, dict] = {}
        self.deliberate_mode = (
            config.borrower_patience_weeks > 1 or config.offer_validity_weeks > 1
        )
        self.market_pool = MarketPool(
            borrower_patience_weeks=config.borrower_patience_weeks,
            offer_validity_weeks=config.offer_validity_weeks,
        ) if self.deliberate_mode else None
        self._market_loan_counter = 0

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
        losses/workout costs shrink it, decay erodes idle capital."""
        return max(0.0,
                   state.total_capital
                   + state.cumulative_interest
                   + state.cumulative_fees
                   - state.cumulative_losses
                   - state.cumulative_workout_cost
                   - state.cumulative_decay)

    def _net_pnl(self, state: SeasonLenderState) -> float:
        """Season-to-date net P&L including workout costs."""
        return (
            state.cumulative_interest
            + state.cumulative_fees
            - state.cumulative_losses
            - state.cumulative_workout_cost
        )

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

    def _apply_underwriting_costs(self, week: int) -> list[str]:
        """Charge real API costs to lender P&L when underwriting_cost_mode='real'.

        Instead of the flat simulated underwriting_cost_per_application_usd,
        each lender's actual API spend (× uw_cost_multiplier) is deducted from
        effective capital.  This makes expensive models genuinely more costly
        to operate — a model that "thinks harder" per loan eats into its own
        margins, creating the same tension a real lending ops team faces between
        thoroughness and efficiency.

        The cost flows through cumulative_uw_ops_cost and reduces effective
        capital via the same path as losses/workout costs.  Capital adequacy
        elimination can then trigger naturally if costs spiral.
        """
        if self.config.underwriting_cost_mode != "real":
            return []

        multiplier = self.config.uw_cost_multiplier
        events = []

        for lid, state in self.lender_states.items():
            if state.eliminated:
                continue

            # Compute this week's API cost delta
            prev_charged = state.cumulative_uw_ops_cost
            raw_api_cost = state.cumulative_cost_usd
            effective_cost = raw_api_cost * multiplier
            week_charge = effective_cost - prev_charged

            if week_charge > 0:
                state.cumulative_uw_ops_cost = effective_cost
                # Deduct from capital via workout_cost channel (existing P&L path)
                state.cumulative_workout_cost += week_charge
                self._recompute_available(state)
                events.append(
                    f"  [{state.lender_name}] Underwriting ops: "
                    f"-${week_charge:,.0f} this week "
                    f"(API: ${raw_api_cost:.4f} × {multiplier:g} = "
                    f"${effective_cost:,.0f} cumulative)"
                )
        return events

    def _format_uw_cost_briefing(self, state: SeasonLenderState) -> list[str]:
        """Format underwriting cost status for portfolio briefing."""
        if self.config.underwriting_cost_mode != "real":
            return []

        lines: list[str] = []
        multiplier = self.config.uw_cost_multiplier
        effective = state.cumulative_cost_usd * multiplier
        lines.append(
            f"Underwriting Ops Cost: ${effective:,.0f} "
            f"(${state.cumulative_cost_usd:.4f} API × {multiplier:g})"
        )

        # Per-eval average and projection
        if state.total_evaluations > 0:
            avg = effective / state.total_evaluations
            lines.append(
                f"  Avg ${avg:,.0f}/evaluation "
                f"({state.total_evaluations} evals to date)"
            )

        return lines

    @property
    def _active_lender_ids(self) -> set[str]:
        """IDs of lenders not yet eliminated."""
        return {
            lid for lid, state in self.lender_states.items()
            if not state.eliminated
        }

    def _assign_arrival_phases(self, cohort) -> dict[str, int]:
        """Map borrower_id -> phase index (1-based)."""
        if self.config.arrival_phases <= 1 or not cohort:
            return {b.id: 1 for b in cohort}

        phase_map: dict[str, int] = {}
        total = len(cohort)
        phases = min(self.config.arrival_phases, total)
        for idx, borrower in enumerate(cohort):
            phase = min(phases, (idx * phases // total) + 1)
            phase_map[borrower.id] = phase
        return phase_map

    def _print_arrival_schedule(self, cohort, phase_map: dict[str, int]) -> None:
        if self.config.arrival_phases <= 1:
            return
        print(f"\n  Intra-week arrival phases ({self.config.arrival_phases}):")
        for phase in range(1, self.config.arrival_phases + 1):
            ids = [b.id for b in cohort if phase_map.get(b.id, 1) == phase]
            if ids:
                print(f"    Phase {phase}: {', '.join(ids)}")

    def _enforce_deep_uw_slots(
        self,
        engine: SimulationEngine,
        phase_map: dict[str, int],
    ) -> dict[str, int]:
        """Convert excess approvals to rejects based on phase-priority slots."""
        slots = self.config.deep_uw_slots_per_week
        if slots <= 0:
            return {}

        borrower_order = {b.id: i for i, b in enumerate(engine.borrowers)}
        used_per_lender: dict[str, int] = {}

        print(f"\n  [Bandwidth] Deep-underwrite cap active: {slots} slot(s) per lender.")
        for lender in engine.lenders:
            decisions = engine.all_decisions.get(lender.id, [])
            approvals = [
                d for d in decisions if d.decision == "APPROVE" and d.term_sheet
            ]
            if len(approvals) <= slots:
                used_per_lender[lender.id] = len(approvals)
                print(
                    f"    {lender.name}: {len(approvals)} approval(s), "
                    "no deferrals."
                )
                continue

            ranked = sorted(
                approvals,
                key=lambda d: (
                    phase_map.get(d.borrower_id, 1),
                    borrower_order.get(d.borrower_id, 10**6),
                ),
            )
            keep_ids = {d.borrower_id for d in ranked[:slots]}
            deferred = 0
            for decision in decisions:
                if (
                    decision.decision == "APPROVE"
                    and decision.term_sheet
                    and decision.borrower_id not in keep_ids
                ):
                    decision.decision = "REJECT"
                    decision.term_sheet = None
                    decision.reasoning = BANDWIDTH_LIMIT_REASON
                    deferred += 1

            used_per_lender[lender.id] = slots
            print(
                f"    {lender.name}: deferred {deferred} approval(s) "
                f"to stay within {slots} slot(s)."
            )

        return used_per_lender

    def _inject_pipeline_context(self, week: int, lenders: list[LenderConfig]) -> None:
        if not self.deliberate_mode or not self.market_pool:
            return
        self.market_pool.set_current_week(week)
        for lender in lenders:
            state = self.lender_states[lender.id]
            self.market_pool.set_lender_capital_snapshot(
                lender.id,
                total_capital=state.total_capital,
                deployed_capital=state.deployed_capital,
                available_capital=state.available_capital,
                cost_of_capital=self.config.economics.funding_rate,
            )
            params = dict(lender.policy_params or {})
            params["offer_validity_weeks"] = self.config.offer_validity_weeks
            params["pipeline"] = self.market_pool.build_pipeline_context(lender.id)
            lender.policy_params = params

    def _apply_market_decisions(
        self,
        week: int,
        engine: SimulationEngine,
        events: list[str],
    ) -> None:
        if not self.deliberate_mode or not self.market_pool:
            return

        for lender_id, decisions in engine.all_decisions.items():
            state = self.lender_states.get(lender_id)
            if not state:
                continue

            for decision in decisions:
                mb = self.market_pool.get_market_borrower(decision.borrower_id)
                if not mb or mb.status != "shopping":
                    continue
                if lender_id in mb.rejected_by:
                    continue

                if decision.reasoning and decision.reasoning.startswith("[LLM_ERROR]"):
                    self.market_pool.record_pass(lender_id, decision.borrower_id, week)
                    decision.decision = "PASS"
                    decision.term_sheet = None
                    continue

                if decision.reasoning and decision.reasoning.startswith("[BANDWIDTH_LIMIT]"):
                    self.market_pool.record_pass(lender_id, decision.borrower_id, week)
                    decision.decision = "PASS"
                    decision.term_sheet = None
                    continue

                if decision.decision == "PASS":
                    self.market_pool.record_pass(lender_id, decision.borrower_id, week)
                    continue

                if decision.decision == "REJECT" or not decision.term_sheet:
                    self.market_pool.record_reject(lender_id, decision.borrower_id)
                    decision.term_sheet = None
                    decision.offer_valid_weeks = 1
                    continue

                existing_open = next(
                    (
                        o.term_sheet.loan_amount
                        for o in mb.open_offers
                        if o.status == "open" and o.lender_id == lender_id
                    ),
                    0.0,
                )
                reserved = self.market_pool.reserved_capital_for_lender(lender_id)
                available_for_new = max(
                    0.0,
                    state.available_capital - max(0.0, reserved - existing_open),
                )
                if decision.term_sheet.loan_amount > available_for_new:
                    decision.decision = "PASS"
                    decision.term_sheet = None
                    decision.reasoning = (
                        f"{decision.reasoning} "
                        "[SYSTEM: Offer skipped due to reserved-capital limit]"
                    ).strip()
                    self.market_pool.record_pass(lender_id, decision.borrower_id, week)
                    continue

                valid_weeks = max(
                    1,
                    int(decision.offer_valid_weeks or self.config.offer_validity_weeks),
                )
                offer = self.market_pool.add_offer(
                    lender_id=lender_id,
                    borrower_id=decision.borrower_id,
                    term_sheet=decision.term_sheet,
                    reasoning=decision.reasoning or "",
                    issued_week=week,
                    validity_weeks=valid_weeks,
                )
                if offer:
                    events.append(
                        f"  [Market] {state.lender_name} offered "
                        f"{mb.borrower.dossier.company_name}: "
                        f"${offer.term_sheet.loan_amount:,.0f} @ "
                        f"{offer.term_sheet.interest_rate:.1f}% "
                        f"({valid_weeks}w validity)"
                    )

    def _resolve_market_bookings(
        self,
        week: int,
        engine: SimulationEngine,
        events: list[str],
    ) -> None:
        if not self.deliberate_mode or not self.market_pool:
            return

        accepted, expired = self.market_pool.resolve_expired()
        if not accepted and not expired:
            return

        for mb, offer in accepted:
            self._market_loan_counter += 1
            loan = BookedLoan(
                id=f"LOAN-{self._market_loan_counter:03d}",
                borrower_id=mb.borrower.id,
                lender_id=offer.lender_id,
                borrower_name=mb.borrower.dossier.company_name,
                sector=mb.borrower.dossier.sector,
                principal=offer.term_sheet.loan_amount,
                interest_rate=offer.term_sheet.interest_rate,
                term_months=offer.term_sheet.term_months,
                true_outcome=mb.borrower.true_outcome,
                months_before_default=mb.borrower.months_before_default,
            )
            engine.booked_loans.append(loan)
            competing = [
                o for o in mb.open_offers
                if o.status in {"accepted", "open"} and o.lender_id != offer.lender_id
            ]
            engine.deal_results[mb.borrower.id] = {
                "outcome": "booked",
                "winner": offer.lender_id,
                "loan_id": loan.id,
                "competitive": bool(competing),
                "speed_bonus_decisive": False,
            }
            lender_name = self.lender_states[offer.lender_id].lender_name
            events.append(
                f"  [Market] BOOKED: {mb.borrower.dossier.company_name} accepted "
                f"{offer.offer_id} from {lender_name}"
            )

        for mb in expired:
            engine.deal_results[mb.borrower.id] = {"outcome": "no_takers", "winner": None}
            events.append(
                f"  [Market] EXITED: {mb.borrower.dossier.company_name} left market "
                "(no open offers)"
            )

    def _flush_market_pool_at_close(self) -> list[str]:
        if not self.deliberate_mode or not self.market_pool:
            return []

        accepted, expired = self.market_pool.force_resolve_all()
        events: list[str] = []
        for mb, offer in accepted:
            self._market_loan_counter += 1
            loan_id = f"LOAN-{self._market_loan_counter:03d}"
            active = ActiveLoan(
                loan_id=loan_id,
                borrower_id=mb.borrower.id,
                borrower_name=mb.borrower.dossier.company_name,
                lender_id=offer.lender_id,
                sector=mb.borrower.dossier.sector,
                principal=offer.term_sheet.loan_amount,
                interest_rate=offer.term_sheet.interest_rate,
                term_months=offer.term_sheet.term_months,
                true_outcome=mb.borrower.true_outcome,
                months_before_default=mb.borrower.months_before_default,
                booked_week=self.config.weeks + 1,
                remaining_balance=offer.term_sheet.loan_amount,
            )
            state = self.lender_states[offer.lender_id]
            state.active_loans.append(active)
            state.deployed_capital += active.principal
            state.deals_won += 1
            self._recompute_available(state)
            self.all_deal_results[mb.borrower.id] = {
                "outcome": "booked",
                "winner": offer.lender_id,
                "loan_id": loan_id,
                "competitive": False,
            }
            events.append(
                f"  [Market close] BOOKED: {mb.borrower.dossier.company_name} "
                f"with {state.lender_name}"
            )
        for mb in expired:
            self.all_deal_results[mb.borrower.id] = {"outcome": "no_takers", "winner": None}
            events.append(
                f"  [Market close] EXITED: {mb.borrower.dossier.company_name} "
                "(no open offers)"
            )
        for state in self.lender_states.values():
            self._recompute_sector_exposure(state)
        return events

    # ------------------------------------------------------------------
    # Main season loop
    # ------------------------------------------------------------------

    async def run_season(self) -> None:
        _print_season_header(self.config)

        # Bootstrap per-tenant LOS config if the scenario provides one
        los_config = self.config.los_config
        if los_config.disabled_guards or los_config.gate_policies:
            los_url = self.engine_kwargs.get("los_url", "http://localhost:3000")
            for lender in self.base_lenders:
                tenant_id = f"lender_{lender.id}"
                try:
                    await bootstrap_los_tenant(
                        tenant_id, los_config, los_url=los_url,
                    )
                except Exception as exc:
                    print(f"  [WARN] Failed to bootstrap LOS config for {tenant_id}: {exc}")

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

            # 3b. Deliberate market timers
            if self.deliberate_mode and self.market_pool:
                tick_events = self.market_pool.tick(week)
                events.extend(tick_events)
                for e in tick_events:
                    print(e)

            # 4. Generate cohort
            cohort = generate_cohort(week, self.config, self.used_static_ids)
            _print_cohort_summary(week, cohort)
            phase_map = self._assign_arrival_phases(cohort)
            self._print_arrival_schedule(cohort, phase_map)

            if self.deliberate_mode and self.market_pool:
                self.market_pool.add_cohort(cohort, week)
                origination_pool = self.market_pool.get_all_shopping_borrowers()
                print(
                    f"\n  [Market] Shopping pool this week: {len(origination_pool)} "
                    f"borrowers (new cohort: {len(cohort)})"
                )
            else:
                origination_pool = cohort

            # 5. Build week lenders (inject briefing + available capital)
            #    Only include active (non-eliminated) lenders
            week_lenders = self._build_week_lenders(briefings)
            active_ids = self._active_lender_ids
            week_lenders = [l for l in week_lenders if l.id in active_ids]

            if not week_lenders:
                print("\n  No active lenders remaining — skipping origination.")
                continue

            if self.deliberate_mode and self.market_pool:
                self._inject_pipeline_context(week, week_lenders)

            # 6. Run origination via SimulationEngine
            engine = SimulationEngine(origination_pool, week_lenders, **self.engine_kwargs)
            await engine.run_origination()
            slots_used = self._enforce_deep_uw_slots(engine, phase_map)

            # 7. Deal resolution
            if self.deliberate_mode:
                self._apply_market_decisions(week, engine, events)
                self._resolve_market_bookings(week, engine, events)
            else:
                capital = self._get_remaining_capital()
                tool_counts = self._extract_tool_counts(engine) if self.config.speed_scoring else None
                engine.adjudicate_deals(
                    remaining_capital=capital,
                    speed_scoring=self.config.speed_scoring,
                    tool_call_counts=tool_counts,
                )

            # 8. Ingest results
            week_result = self._ingest_results(week, engine, slots_used)
            week_result.events = events
            self.week_results.append(week_result)

            # 8a. Update scratchpads (after results, before next week)
            self._update_scratchpads(week, engine)

            # 8b. Consistency checks (pass@k)
            if self.config.consistency_samples > 0:
                await self._run_consistency_checks(week, engine)

            # 8c. Apply real underwriting costs to P&L (if enabled)
            uw_events = self._apply_underwriting_costs(week)
            events.extend(uw_events)
            for e in uw_events:
                print(e)

            # 8d. Capture per-week detail for JSON export
            detail_cohort = origination_pool if self.deliberate_mode else cohort
            self._capture_week_detail(week, detail_cohort, engine, events, phase_map, slots_used)
            self._capture_week_match_data(week, engine)

            # 9. Snapshot utilization + weekly analytics
            self._snapshot_utilization()
            self._record_week_snapshots(week)

        # Final market close for any still-shopping borrowers
        if self.deliberate_mode:
            market_close_events = self._flush_market_pool_at_close()
            for e in market_close_events:
                print(e)

        # Final resolution — fast-forward remaining active loans
        final_events = self._final_resolution()
        final_step = len(self.week_details) + 1
        self._record_week_snapshots(final_step)
        self._capture_final_resolution_detail(final_step, final_events)
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
                    state.cumulative_workout_cost += result.workout_cost
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
            if self.config.arrival_phases > 1:
                lines.append(
                    f"Pipeline this week: {self.config.arrival_phases} arrival phases."
                )
            if self.config.deep_uw_slots_per_week > 0:
                lines.append(
                    "Deep-underwrite capacity cap: "
                    f"{self.config.deep_uw_slots_per_week} approvals/week."
                )

            if state.active_loans:
                lines.append("Current Portfolio:")
                for loan in state.active_loans:
                    lines.append(
                        f"  - {loan.borrower_name} ({loan.sector}): "
                        f"${loan.remaining_balance:,.0f} remaining, "
                        f"{loan.months_elapsed}/{loan.term_months} months, "
                        f"${loan.total_interest_collected:,.0f} interest collected"
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

            # Decision attribution — link recent defaults/repayments to
            # original underwriting decisions so the model can learn from
            # its past choices (inspired by Terra Nova's credit assignment).
            recent_resolved = [
                lo for lo in state.resolved_loans
                if lo.defaulted  # focus on losses — most actionable
            ]
            if recent_resolved:
                lines.append("Decision Attribution (your past approvals that defaulted):")
                for lo in recent_resolved[-5:]:  # last 5 to control length
                    # Find the original active loan for booked_week context
                    booked_week = "?"
                    orig_rate = lo.total_interest_paid  # fallback
                    for al in state.active_loans:
                        if al.loan_id == lo.loan_id:
                            booked_week = str(al.booked_week)
                            orig_rate = al.interest_rate
                            break
                    # Check inactive loans too (already removed from active)
                    if booked_week == "?":
                        # Search all_decisions for the original rate
                        decisions = self.all_decisions.get(lid, [])
                        for d in decisions:
                            if d.borrower_id == lo.borrower_id and d.term_sheet:
                                orig_rate = d.term_sheet.interest_rate
                                break
                    fraud_tag = " [FRAUD]" if lo.was_fraud else ""
                    lines.append(
                        f"  - {lo.borrower_name} ({lo.sector}){fraud_tag}: "
                        f"approved ${lo.principal:,.0f} @ {orig_rate:.1f}%, "
                        f"lost ${lo.principal_lost:,.0f} after {lo.months_paid}mo"
                    )

            # Season P&L summary
            total_pnl = self._net_pnl(state)
            lines.append(
                f"Season P&L: ${total_pnl:,.0f} "
                f"(interest: ${state.cumulative_interest:,.0f}, "
                f"fees: ${state.cumulative_fees:,.0f}, "
                f"losses: -${state.cumulative_losses:,.0f}, "
                f"workout: -${state.cumulative_workout_cost:,.0f})"
            )

            # Underwriting cost feedback (real mode only)
            uw_lines = self._format_uw_cost_briefing(state)
            if uw_lines:
                lines.extend(uw_lines)

            # Competition feedback from last week
            stats = self._last_week_stats.get(lid)
            if stats:
                lines.append("Competition (last week):")
                lines.append(
                    f"  Your deals: {stats['won']} won, "
                    f"{stats['lost']} lost to cheaper offers, "
                    f"{stats['rejected']} rejected, "
                    f"{stats.get('passed', 0)} passed"
                )
                avg_off = stats["avg_offered_rate"]
                avg_win = stats["avg_winning_rate"]
                if avg_off > 0:
                    lines.append(
                        f"  Your avg offered rate: {avg_off:.1f}% | "
                        f"Avg winning rate: {avg_win:.1f}%"
                    )
                    gap = avg_off - avg_win
                    if gap > 1.0:
                        lines.append(
                            f"  Note: You are pricing {gap:.1f}% above winning rates. "
                            f"Consider lowering to win more deals."
                        )
                    elif gap < -0.5:
                        lines.append(
                            f"  Note: You are pricing {-gap:.1f}% below winning rates. "
                            f"You may be leaving margin on the table."
                        )

            # Persistent strategy scratchpad
            if self.config.scratchpad and state.scratchpad:
                lines.append("--- YOUR STRATEGY NOTES (from prior weeks) ---")
                lines.append(state.scratchpad)
                lines.append("--- END NOTES ---")
                lines.append(
                    "Update your notes after this week by including "
                    "SCRATCHPAD_UPDATE: <your notes> in your reasoning."
                )

            # Consistency score (if tracking)
            if state.consistency_checks:
                lines.append(
                    f"Decision Consistency: {state.consistency_agreement_rate:.0%} "
                    f"({len(state.consistency_checks)} checks)"
                )

            lines.append("---")

            briefings[lid] = "\n".join(lines)

        return briefings

    # ------------------------------------------------------------------
    # Week lender construction
    # ------------------------------------------------------------------

    def _build_tool_context(self, lender_id: str) -> str:
        """Render custom tool context to prepend to lender persona."""
        toolkit = self.toolkits.get(lender_id)
        if not toolkit or not toolkit.tools:
            return ""
        lines = ["Available custom underwriting tools (call by tool name when useful):"]
        for tool in toolkit.tools:
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def _build_week_lenders(self, briefings: dict[str, str]) -> list[LenderConfig]:
        """Create LenderConfig copies with briefing prepended to persona and
        existing_portfolio reflecting active season loans."""
        week_lenders = []
        for lender in self.base_lenders:
            state = self.lender_states[lender.id]
            briefing = briefings.get(lender.id, "")
            toolkit = self.toolkits[lender.id]
            tool_definitions = toolkit.get_tool_definitions()

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

            persona_parts = []
            if briefing:
                persona_parts.append(briefing)
            if toolkit.tools:
                tool_lines = [
                    "--- CUSTOM TOOLKIT ---",
                    "You have lender-specific custom tools from prior weeks.",
                    "Call these tools when relevant instead of repeating manual analysis:",
                ]
                for tool in toolkit.tools:
                    tool_lines.append(
                        f"- {tool.name}: {tool.description} "
                        f"(implementation hint: {tool.implementation})"
                    )
                tool_lines.append(
                    "If a tool directly answers the question, prefer using it before finalizing terms."
                )
                persona_parts.append("\n".join(tool_lines))
            persona_parts.append(lender.persona)

            week_lender = LenderConfig(
                id=lender.id,
                name=lender.name,
                persona="\n\n".join(p for p in persona_parts if p),
                model=lender.model,
                target_yield_pct=lender.target_yield_pct,
                max_single_loan=lender.max_single_loan,
                total_capital=state.total_capital,
                sector_limits=lender.sector_limits,
                existing_portfolio=existing,
                custom_tools=tool_definitions,
                policy_params=copy.deepcopy(lender.policy_params),
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
        """Tooling phase between weeks.

        In mock mode, auto-generate tools based on lender performance.
        In live mode, this would prompt each lender's LLM to create/update tools.
        Tools persist in `self.toolkits` and are available during evaluation.
        """
        print(f"\n  [Tooling phase — week {week}]")
        active_ids = self._active_lender_ids

        for lender in self.base_lenders:
            if lender.id not in active_ids:
                continue
            state = self.lender_states[lender.id]
            toolkit = self.toolkits[lender.id]

            if self.engine_kwargs.get("mock"):
                self._mock_tooling(toolkit, state, week)
            else:
                self._live_tooling(toolkit, state, lender, week)

            # Keep season scoring state in sync with the toolkit registry.
            state.custom_tools = list(toolkit.tools)

            if toolkit.tools:
                tool_names = [t.name for t in toolkit.tools]
                print(f"    {lender.name}: {len(toolkit.tools)} tool(s) "
                      f"[{', '.join(tool_names)}]")

    def _mock_tooling(
        self, toolkit: LenderToolkit, state: SeasonLenderState, week: int,
    ) -> None:
        """Auto-generate plausible tools based on lender performance patterns."""
        # Week 1: every lender creates a sector concentration checker
        if week == 1 and not toolkit.get_tool("sector_concentration_check"):
            toolkit.create_tool(
                name="sector_concentration_check",
                description="Flag if >40% of portfolio is in one sector",
                implementation="check sector_exposure > 0.4 * deployed_capital",
                week=week,
            )

        # After experiencing a default: create a cash-flow stress tool
        defaults = sum(1 for lo in state.resolved_loans if lo.defaulted)
        if (defaults > 0
                and not toolkit.get_tool("cashflow_stress_test")):
            toolkit.create_tool(
                name="cashflow_stress_test",
                description="Stress test borrower cash flows at -20% revenue",
                implementation="recalculate net_income with revenue * 0.8",
                week=week,
            )

        # After losing deals (bid too low): create a pricing optimizer
        if (state.deals_lost > 2
                and not toolkit.get_tool("competitive_pricer")):
            toolkit.create_tool(
                name="competitive_pricer",
                description="Suggest rate within market range to win deals",
                implementation="target rate = max(floor_rate, market_avg - 0.5%)",
                week=week,
            )

    def _live_tooling(
        self, toolkit: LenderToolkit, state: SeasonLenderState,
        lender: LenderConfig, week: int,
    ) -> None:
        """LLM-driven tool creation between season weeks.

        The lender's model sees its performance history and decides
        whether to create, update, or skip tool creation.
        Falls back to _mock_tooling if the LLM call fails.
        """
        import asyncio
        import json
        import os

        import httpx

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            # No API key — fall back to deterministic heuristics
            self._mock_tooling(toolkit, state, week)
            return

        # Build performance summary
        defaults = sum(1 for lo in state.resolved_loans if lo.defaulted)
        active_count = len(state.active_loans)
        deployed = sum(lo.remaining_balance for lo in state.active_loans)
        existing_tools = [t.name for t in toolkit.tools]

        perf_summary = (
            f"Week {week} performance: "
            f"{active_count} active loans, ${deployed:,.0f} deployed, "
            f"{defaults} defaults, {state.deals_lost} deals lost to competitors, "
            f"{state.total_rejections} rejections. "
            f"Current tools: {existing_tools or 'none'}."
        )

        prompt = f"""TOOLING PHASE (Week {week})

You are managing tools for a lending institution. Based on performance, decide whether to CREATE a new tool, UPDATE an existing one, or SKIP.

{perf_summary}

Rules:
- Maximum {toolkit.MAX_TOOLS} tools allowed
- Tools should help with underwriting analysis
- Each tool has a name, description, and bash implementation
- Creating a tool costs {toolkit.CREATION_COST} efficiency points, updating costs {toolkit.UPDATE_COST}

Respond with JSON:
- To create: {{"action": "create", "name": "tool_name", "description": "what it does", "implementation": "bash script"}}
- To update: {{"action": "update", "name": "existing_tool", "description": "new desc", "implementation": "new script"}}
- To skip: {{"action": "skip"}}
"""

        async def _call() -> dict | None:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": lender.model,
                "messages": [
                    {"role": "system", "content": "You are a tool creation assistant for a lending platform. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            }
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    if resp.status_code != 200:
                        return None
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    # Extract JSON from response
                    import re
                    m = re.search(r'\{.*\}', content, re.DOTALL)
                    if m:
                        return json.loads(m.group())
                    return None
            except Exception:
                return None

        try:
            tool_decision = asyncio.get_event_loop().run_until_complete(_call())
        except RuntimeError:
            # No event loop running — create one
            tool_decision = asyncio.run(_call())

        if tool_decision is None:
            # LLM call failed — fall back to deterministic
            self._mock_tooling(toolkit, state, week)
            return

        action = tool_decision.get("action", "skip")
        if action == "create":
            name = tool_decision.get("name", "")
            desc = tool_decision.get("description", "")
            impl = tool_decision.get("implementation", "echo 'not implemented'")
            if name and desc:
                toolkit.create_tool(name=name, description=desc, implementation=impl, week=week)
        elif action == "update":
            name = tool_decision.get("name", "")
            desc = tool_decision.get("description")
            impl = tool_decision.get("implementation")
            if name:
                toolkit.update_tool(name=name, description=desc, implementation=impl)

    # ------------------------------------------------------------------
    # Scratchpad update
    # ------------------------------------------------------------------

    def _update_scratchpads(self, week: int, engine: SimulationEngine) -> None:
        """Update each lender's persistent strategy scratchpad after a week.

        In mock mode, auto-generate notes based on performance patterns.
        In live mode, extract SCRATCHPAD_UPDATE directives from LLM reasoning.
        """
        if not self.config.scratchpad:
            return

        active_ids = self._active_lender_ids

        for lender in self.base_lenders:
            if lender.id not in active_ids:
                continue
            state = self.lender_states[lender.id]

            if self.engine_kwargs.get("mock"):
                self._mock_scratchpad_update(state, week, engine)
            else:
                self._live_scratchpad_update(state, week, engine)

    def _mock_scratchpad_update(
        self, state: SeasonLenderState, week: int, engine: SimulationEngine,
    ) -> None:
        """Auto-generate scratchpad notes from performance patterns."""
        notes: list[str] = []

        # Note defaults by sector
        default_sectors: dict[str, int] = {}
        for lo in state.resolved_loans:
            if lo.defaulted:
                default_sectors[lo.sector] = default_sectors.get(lo.sector, 0) + 1
        if default_sectors:
            worst = max(default_sectors, key=default_sectors.get)  # type: ignore[arg-type]
            notes.append(
                f"W{week}: {default_sectors[worst]} default(s) in {worst} sector — "
                "increase scrutiny on this sector."
            )

        # Note pricing competitiveness
        stats = self._last_week_stats.get(state.lender_id, {})
        if stats.get("lost", 0) > stats.get("won", 0) and stats.get("avg_offered_rate", 0) > 0:
            gap = stats["avg_offered_rate"] - stats["avg_winning_rate"]
            if gap > 0.5:
                notes.append(
                    f"W{week}: Lost {stats['lost']} deals — pricing "
                    f"{gap:.1f}% above market. Consider lowering rates."
                )
        elif stats.get("won", 0) > 0 and stats.get("avg_offered_rate", 0) > 0:
            gap = stats["avg_winning_rate"] - stats["avg_offered_rate"]
            if gap > 0.5:
                notes.append(
                    f"W{week}: Winning at {gap:.1f}% below market — "
                    "may be leaving margin on the table."
                )

        # Note concentration risk
        if state.total_capital > 0:
            for sector, exposure in state.sector_exposure.items():
                pct = exposure / state.total_capital
                if pct > 0.35:
                    notes.append(
                        f"W{week}: {sector} concentration at {pct:.0%} — "
                        "watch for breach."
                    )

        if notes:
            # Append new notes, keeping total under ~500 chars
            existing = state.scratchpad
            new_section = "\n".join(notes)
            combined = f"{existing}\n{new_section}".strip() if existing else new_section
            # Trim oldest lines if too long
            lines = combined.split("\n")
            while len("\n".join(lines)) > 500 and len(lines) > 3:
                lines.pop(0)
            state.scratchpad = "\n".join(lines)

    def _live_scratchpad_update(
        self, state: SeasonLenderState, week: int, engine: SimulationEngine,
    ) -> None:
        """Extract SCRATCHPAD_UPDATE directives from LLM reasoning."""
        import re

        decisions = engine.all_decisions.get(state.lender_id, [])
        updates: list[str] = []
        for decision in decisions:
            if not decision.reasoning:
                continue
            # Look for SCRATCHPAD_UPDATE: <text> pattern
            match = re.search(
                r"SCRATCHPAD_UPDATE:\s*(.+?)(?:\n|$)",
                decision.reasoning,
                re.IGNORECASE,
            )
            if match:
                updates.append(f"W{week}: {match.group(1).strip()}")

        if updates:
            existing = state.scratchpad
            new_section = "\n".join(updates)
            combined = f"{existing}\n{new_section}".strip() if existing else new_section
            lines = combined.split("\n")
            while len("\n".join(lines)) > 500 and len(lines) > 3:
                lines.pop(0)
            state.scratchpad = "\n".join(lines)

    # ------------------------------------------------------------------
    # Pass@k consistency checks
    # ------------------------------------------------------------------

    async def _run_consistency_checks(
        self, week: int, engine: SimulationEngine,
    ) -> None:
        """Re-evaluate a random sample of borrowers to measure decision stability.

        For each sampled borrower, runs `consistency_samples` additional evaluations
        and compares whether the decision matches the original.  Tracks per-lender
        agreement rate (inspired by InteractiveBench pass@k methodology).
        """
        k = self.config.consistency_samples
        if k <= 0:
            return

        import random as rng

        cohort = engine.borrowers
        sample_size = max(1, int(len(cohort) * self.config.consistency_sample_pct))
        sampled = rng.sample(cohort, min(sample_size, len(cohort)))

        print(f"\n  [Consistency] pass@{k} check on {len(sampled)} borrower(s)...")

        active_ids = self._active_lender_ids

        for lender in self.base_lenders:
            if lender.id not in active_ids:
                continue
            state = self.lender_states[lender.id]
            original_decisions = engine.all_decisions.get(lender.id, [])
            orig_map = {d.borrower_id: d.decision for d in original_decisions}

            week_lenders = self._build_week_lenders(
                self._build_briefings(week, [])
            )
            week_lender = next(
                (l for l in week_lenders if l.id == lender.id), None
            )
            if not week_lender:
                continue

            for borrower in sampled:
                original = orig_map.get(borrower.id)
                if original is None:
                    continue

                # Run k additional evaluations
                agreements = 0
                for _ in range(k):
                    re_engine = SimulationEngine(
                        [borrower], [week_lender], **self.engine_kwargs,
                    )
                    await re_engine.run_origination()
                    re_decisions = re_engine.all_decisions.get(lender.id, [])
                    if re_decisions and re_decisions[0].decision == original:
                        agreements += 1

                check = {
                    "week": week,
                    "borrower_id": borrower.id,
                    "original_decision": original,
                    "samples": k,
                    "agreements": agreements,
                    "agreement_rate": agreements / k,
                }
                state.consistency_checks.append(check)

            # Recompute running agreement rate
            if state.consistency_checks:
                total_agree = sum(c["agreements"] for c in state.consistency_checks)
                total_samples = sum(c["samples"] for c in state.consistency_checks)
                state.consistency_agreement_rate = (
                    total_agree / total_samples if total_samples > 0 else 0.0
                )

            print(
                f"    {state.lender_name}: "
                f"{state.consistency_agreement_rate:.0%} agreement "
                f"({len(state.consistency_checks)} total checks)"
            )

    # ------------------------------------------------------------------
    # Result ingestion
    # ------------------------------------------------------------------

    def _ingest_results(
        self,
        week: int,
        engine: SimulationEngine,
        slots_used: dict[str, int] | None = None,
    ) -> WeekResult:
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

        # Extract per-lender token/cost data from engine runs
        lender_costs: dict[str, dict] = {}
        for run in engine.runs:
            lid = (run.policy.params or {}).get("_lender_id", "")
            if not lid:
                pid = run.policy.policy_id or ""
                parts = pid.split("_", 2)
                lid = parts[1] if len(parts) >= 2 and parts[0] == "p" else pid
            if lid and run.trace and run.trace.cost:
                c = lender_costs.setdefault(lid, {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
                c["tokens_in"] += run.trace.cost.tokens_in
                c["tokens_out"] += run.trace.cost.tokens_out
                c["cost_usd"] += run.trace.cost.estimated_cost_usd

            # Reflect live tool-call traces back into per-lender toolkit state.
            toolkit = self.toolkits.get(lid)
            if toolkit and run.trace and run.trace.steps:
                for step in run.trace.steps:
                    if getattr(step, "type", "") != "tool_call":
                        continue
                    args = getattr(step, "args", {}) or {}
                    tool_name = (
                        getattr(step, "name", "")
                        or (args.get("tool") if isinstance(args, dict) else "")
                        or (args.get("name") if isinstance(args, dict) else "")
                    )
                    if tool_name:
                        toolkit.record_usage(tool_name)

        # Count rejections and losses for each lender
        booked_bids = {loan.borrower_id for loan in engine.booked_loans}
        for lender in self.base_lenders:
            state = self.lender_states[lender.id]
            decisions = engine.all_decisions.get(lender.id, [])
            approvals = sum(1 for d in decisions if d.decision == "APPROVE")
            used = approvals
            if slots_used is not None:
                used = slots_used.get(lender.id, approvals)
            state.weekly_deep_uw_used.append(used)
            # Accumulate token/cost stats
            lc = lender_costs.get(lender.id, {})
            state.cumulative_tokens_in += lc.get("tokens_in", 0)
            state.cumulative_tokens_out += lc.get("tokens_out", 0)
            state.cumulative_cost_usd += lc.get("cost_usd", 0.0)
            for d in decisions:
                state.total_evaluations += 1
                bandwidth_limited = (
                    d.reasoning and d.reasoning.startswith("[BANDWIDTH_LIMIT]")
                )
                if bandwidth_limited:
                    state.deep_uw_deferred += 1
                    decision_tool_calls = 0
                else:
                    decision_tool_calls = tool_counts.get((lender.id, d.borrower_id), 0)
                state.total_tool_calls += decision_tool_calls
                if d.reasoning and d.reasoning.startswith("[LLM_ERROR]"):
                    state.deals_errored += 1
                elif d.decision == "PASS":
                    state.deals_passed += 1
                elif d.decision != "APPROVE":
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
            bucket = self.all_decisions.setdefault(lid, [])
            if self.deliberate_mode:
                by_borrower = {d.borrower_id: idx for idx, d in enumerate(bucket)}
                for decision in decisions:
                    idx = by_borrower.get(decision.borrower_id)
                    if idx is None:
                        by_borrower[decision.borrower_id] = len(bucket)
                        bucket.append(decision)
                    else:
                        bucket[idx] = decision
            else:
                bucket.extend(decisions)
        for borrower in engine.borrowers:
            if borrower.id in self._seen_borrower_ids:
                continue
            self._seen_borrower_ids.add(borrower.id)
            self.all_borrowers.append(borrower)
        self.all_deal_results.update(engine.deal_results)

        # Compute per-week competition stats for briefing feedback
        winning_rates: list[float] = []
        for loan in engine.booked_loans:
            winning_rates.append(loan.interest_rate)

        self._last_week_stats.clear()
        for lender in self.base_lenders:
            lid = lender.id
            decisions = engine.all_decisions.get(lid, [])
            won = sum(1 for l in engine.booked_loans if l.lender_id == lid)
            rejected = sum(1 for d in decisions if d.decision == "REJECT")
            passed = sum(1 for d in decisions if d.decision == "PASS")
            approved = sum(1 for d in decisions if d.decision == "APPROVE")
            lost = approved - won
            if self.deliberate_mode and self.market_pool:
                pending_offers = len(self.market_pool.open_offers_for_lender(lid))
                lost = max(0, lost - pending_offers)
            offered_rates = [
                d.term_sheet.interest_rate for d in decisions
                if d.decision == "APPROVE" and d.term_sheet
            ]
            avg_offered = sum(offered_rates) / len(offered_rates) if offered_rates else 0.0
            avg_winning = sum(winning_rates) / len(winning_rates) if winning_rates else 0.0
            self._last_week_stats[lid] = {
                "won": won, "lost": lost, "rejected": rejected, "passed": passed,
                "avg_offered_rate": avg_offered,
                "avg_winning_rate": avg_winning,
            }

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
            # Compute per-week cost delta for budget tracking
            prev_cost = (
                state.weekly_snapshots[-1].get("cumulative_cost_usd", 0.0)
                if state.weekly_snapshots else 0.0
            )
            week_cost = state.cumulative_cost_usd - prev_cost

            state.weekly_snapshots.append({
                "week": week,
                "effective_capital": round(self._effective_capital(state), 2),
                "available_capital": round(state.available_capital, 2),
                "deployed_capital": round(state.deployed_capital, 2),
                "active_loans": len(state.active_loans),
                "defaults_to_date": defaults,
                "frauds_to_date": frauds,
                "net_pnl": round(
                    self._net_pnl(state), 2
                ),
                "cumulative_cost_usd": round(state.cumulative_cost_usd, 4),
                "week_cost_usd": round(week_cost, 4),
            })

    # ------------------------------------------------------------------
    # Final resolution
    # ------------------------------------------------------------------

    def _final_resolution(self) -> list[str]:
        """Fast-forward all remaining active loans to maturity."""
        print(f"\n{'=' * 70}")
        print("FINAL RESOLUTION: Fast-forwarding remaining loans")
        print(f"{'=' * 70}")

        eco = self.config.economics
        events: list[str] = []
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
                    state.cumulative_workout_cost += result.workout_cost
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
                    events.append(
                        f"{label}: {loan.borrower_name} ({loan.sector}) — "
                        f"${result.principal_lost:,.0f} lost, "
                        f"${result.recovery_amount:,.0f} recovered "
                        f"[{state.lender_name}]"
                    )
                else:
                    loan.status = "repaid" if result.matured else "prepaid"
                    state.deployed_capital = max(
                        0.0, state.deployed_capital - result.principal_repaid
                    )
                    label = "REPAID" if result.matured else "PREPAID"
                    print(f"  {loan.loan_id} ({loan.borrower_name}): {label} "
                          f"— ${loan.total_interest_collected:,.0f} total interest")
                    events.append(
                        f"{label}: {loan.borrower_name} ({loan.sector}) — "
                        f"${loan.total_interest_collected:,.0f} total interest "
                        f"[{state.lender_name}]"
                    )

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

        return events

    # ------------------------------------------------------------------
    # Per-week detail capture (for JSON export)
    # ------------------------------------------------------------------

    def _capture_week_match_data(self, week: int, engine: SimulationEngine) -> None:
        """Capture per-week raw artifacts for leaderboard match emission."""
        self.weekly_match_data.append({
            "week": week,
            "borrowers": copy.deepcopy(engine.borrowers),
            "all_decisions": copy.deepcopy(engine.all_decisions),
            "booked_loans": copy.deepcopy(engine.booked_loans),
            "deal_results": copy.deepcopy(engine.deal_results),
            "runs": copy.deepcopy(engine.runs),
        })

    def _capture_week_detail(
        self,
        week: int,
        cohort,
        engine,
        events: list[str],
        phase_map: dict[str, int] | None = None,
        slots_used: dict[str, int] | None = None,
    ) -> None:
        """Capture JSON-serializable per-week detail for the web viewer."""
        phase_map = phase_map or {}
        slots_used = slots_used or {}
        run_lookup: dict[tuple[str, str], dict] = {}

        def _lender_id_for_run(run) -> str:
            lid = (run.policy.params or {}).get("_lender_id", "")
            if lid:
                return lid
            pid = run.policy.policy_id or ""
            parts = pid.split("_", 2)
            if len(parts) >= 2 and parts[0] == "p":
                return parts[1]
            return pid

        for run in getattr(engine, "runs", []) or []:
            lid = _lender_id_for_run(run)
            bid = getattr(run.case, "case_id", "")
            if not lid or not bid:
                continue

            trace_steps = []
            for s in (getattr(run.trace, "steps", []) or []):
                trace_steps.append({
                    "t": getattr(s, "t", ""),
                    "type": getattr(s, "type", ""),
                    "name": getattr(s, "name", ""),
                    "args": getattr(s, "args", {}) or {},
                    "result": getattr(s, "result", {}) or {},
                    "content": getattr(s, "content", "") or "",
                })

            decision_obj = getattr(run, "decision", None)
            rationale = getattr(decision_obj, "rationale", None)
            terms = getattr(decision_obj, "terms", None)
            run_lookup[(lid, bid)] = {
                "inputs": {
                    "financials": {
                        "revenue_ttm": getattr(getattr(run.inputs, "financials", None), "revenue_ttm", 0.0),
                        "gross_margin": getattr(getattr(run.inputs, "financials", None), "gross_margin", 0.0),
                        "ebitda_ttm": getattr(getattr(run.inputs, "financials", None), "ebitda_ttm", 0.0),
                        "net_income": getattr(getattr(run.inputs, "financials", None), "net_income", 0.0),
                        "annual_expenses": getattr(getattr(run.inputs, "financials", None), "annual_expenses", 0.0),
                    },
                    "banking": {
                        "avg_daily_balance_90d": getattr(getattr(run.inputs, "banking", None), "avg_daily_balance_90d", 0.0),
                        "nsf_12m": getattr(getattr(run.inputs, "banking", None), "nsf_12m", 0),
                        "total_deposits_12m": getattr(getattr(run.inputs, "banking", None), "total_deposits_12m", 0.0),
                        "total_withdrawals_12m": getattr(getattr(run.inputs, "banking", None), "total_withdrawals_12m", 0.0),
                    },
                    "business": {
                        "industry": getattr(getattr(run.inputs, "business", None), "industry", ""),
                        "years_trading": getattr(getattr(run.inputs, "business", None), "years_trading", 0),
                        "employee_count": getattr(getattr(run.inputs, "business", None), "employee_count", 0),
                        "company_name": getattr(getattr(run.inputs, "business", None), "company_name", ""),
                    },
                    "raw_documents": list(getattr(run.inputs, "raw_documents", []) or []),
                    "missing_info": list(getattr(run.inputs, "missing_info", []) or []),
                },
                "trace": {
                    "latency_ms": getattr(getattr(run, "trace", None), "latency_ms", 0),
                    "steps": trace_steps,
                    "cost": {
                        "tokens_in": getattr(getattr(getattr(run, "trace", None), "cost", None), "tokens_in", 0),
                        "tokens_out": getattr(getattr(getattr(run, "trace", None), "cost", None), "tokens_out", 0),
                        "estimated_cost_usd": getattr(getattr(getattr(run, "trace", None), "cost", None), "estimated_cost_usd", 0.0),
                    },
                },
                "decision": {
                    "action": getattr(decision_obj, "action", ""),
                    "risk_grade": getattr(decision_obj, "risk_grade", ""),
                    "prob_default_12m": getattr(decision_obj, "prob_default_12m", 0.0),
                    "confidence": getattr(decision_obj, "confidence", 0.0),
                    "conditions": list(getattr(decision_obj, "conditions", []) or []),
                    "covenants": list(getattr(decision_obj, "covenants", []) or []),
                    "terms": {
                        "amount": getattr(terms, "amount", 0.0) if terms else 0.0,
                        "apr": getattr(terms, "apr", 0.0) if terms else 0.0,
                        "tenor_months": getattr(terms, "tenor_months", 0) if terms else 0,
                        "fees": getattr(terms, "fees", {}) if terms else {},
                    },
                    "rationale": {
                        "summary": getattr(rationale, "summary", "") if rationale else "",
                        "key_factors": list(getattr(rationale, "key_factors", []) or []) if rationale else [],
                        "what_would_change": list(getattr(rationale, "what_would_change", []) or []) if rationale else [],
                    },
                },
            }
        borrowers = []
        for b in cohort:
            borrowers.append({
                "id": b.id,
                "name": b.dossier.company_name,
                "sector": b.dossier.sector,
                "amount": b.dossier.loan_request_amount,
                "true_outcome": b.true_outcome,
                "arrival_phase": phase_map.get(b.id, 1),
            })

        decisions = []
        for lender_id, decs in engine.all_decisions.items():
            for d in decs:
                run_data = run_lookup.get((lender_id, d.borrower_id), {})
                trace = run_data.get("trace", {})
                trace_steps = trace.get("steps", []) or []
                tool_calls = sum(1 for s in trace_steps if s.get("type") == "tool_call")
                chain_parts = [
                    (s.get("content") or "").strip()
                    for s in trace_steps
                    if s.get("type") in ("reasoning", "note") and (s.get("content") or "").strip()
                ]
                decisions.append({
                    "lender_id": lender_id,
                    "borrower_id": d.borrower_id,
                    "decision": d.decision,
                    "offer_valid_weeks": getattr(d, "offer_valid_weeks", 1),
                    "bandwidth_limited": bool(d.reasoning and d.reasoning.startswith("[BANDWIDTH_LIMIT]")),
                    "reasoning": d.reasoning or "",
                    "term_sheet": {
                        "amount": d.term_sheet.loan_amount,
                        "rate": d.term_sheet.interest_rate,
                        "term_months": d.term_sheet.term_months,
                    } if d.term_sheet else None,
                    "tokens_in": trace.get("cost", {}).get("tokens_in", 0),
                    "tokens_out": trace.get("cost", {}).get("tokens_out", 0),
                    "cost_usd": trace.get("cost", {}).get("estimated_cost_usd", 0.0),
                    "tool_calls": tool_calls,
                    "chain_of_thought": "\n\n".join(chain_parts),
                    "los_detail": run_data or None,
                })

        booked = []
        for loan in engine.booked_loans:
            booked.append({
                "id": loan.id,
                "borrower_id": loan.borrower_id,
                "borrower_name": loan.borrower_name,
                "lender_id": loan.lender_id,
                "sector": loan.sector,
                "principal": loan.principal,
                "interest_rate": loan.interest_rate,
                "term_months": loan.term_months,
            })

        # Snapshot lender states at end of this week
        lender_snapshots = {}
        for lid, state in self.lender_states.items():
            total_pnl = self._net_pnl(state)
            lender_snapshots[lid] = {
                "name": state.lender_name,
                "model": state.model,
                "net_pnl": round(total_pnl, 2),
                "deployed": round(state.deployed_capital, 2),
                "available": round(state.available_capital, 2),
                "effective_capital": round(self._effective_capital(state), 2),
                "deals_won": state.deals_won,
                "deals_rejected": state.deals_rejected,
                "deals_passed": state.deals_passed,
                "deals_lost": state.deals_lost,
                "active_loans": len(state.active_loans),
                "active_loans_detail": [
                    {
                        "loan_id": l.loan_id,
                        "borrower_id": l.borrower_id,
                        "borrower_name": l.borrower_name,
                        "status": l.status,
                        "months_elapsed": l.months_elapsed,
                        "term_months": l.term_months,
                        "remaining_balance": round(l.remaining_balance, 2),
                        "total_interest_collected": round(l.total_interest_collected, 2),
                        "total_principal_repaid": round(l.total_principal_repaid, 2),
                    }
                    for l in state.active_loans
                ],
                "cumulative_interest": round(state.cumulative_interest, 2),
                "cumulative_losses": round(state.cumulative_losses, 2),
                "cumulative_fees": round(state.cumulative_fees, 2),
                "cumulative_workout_cost": round(state.cumulative_workout_cost, 2),
                "defaults": sum(1 for o in state.resolved_loans if o.defaulted),
                "frauds_funded": sum(1 for o in state.resolved_loans if o.was_fraud),
                "deep_uw_deferred": state.deep_uw_deferred,
                "deep_uw_used_this_week": slots_used.get(lid, 0),
                "tokens_in": state.cumulative_tokens_in,
                "tokens_out": state.cumulative_tokens_out,
                "cost_usd": round(state.cumulative_cost_usd, 4),
            }

        self.week_details.append({
            "week": week,
            "borrowers": borrowers,
            "decisions": decisions,
            "booked_loans": booked,
            "events": events,
            "bandwidth": {
                "arrival_phases": self.config.arrival_phases,
                "deep_uw_slots_per_week": self.config.deep_uw_slots_per_week,
                "slots_used": slots_used,
            },
            "lender_snapshots": lender_snapshots,
        })

    def _capture_final_resolution_detail(self, week: int, events: list[str]) -> None:
        """Append a final timeline step so post-season resolution is visible in exports."""
        lender_snapshots = {}
        for lid, state in self.lender_states.items():
            lender_snapshots[lid] = {
                "name": state.lender_name,
                "model": state.model,
                "net_pnl": round(self._net_pnl(state), 2),
                "deployed": round(state.deployed_capital, 2),
                "available": round(state.available_capital, 2),
                "effective_capital": round(self._effective_capital(state), 2),
                "deals_won": state.deals_won,
                "deals_rejected": state.deals_rejected,
                "deals_passed": state.deals_passed,
                "deals_lost": state.deals_lost,
                "active_loans": len(state.active_loans),
                "active_loans_detail": [
                    {
                        "loan_id": l.loan_id,
                        "borrower_id": l.borrower_id,
                        "borrower_name": l.borrower_name,
                        "status": l.status,
                        "months_elapsed": l.months_elapsed,
                        "term_months": l.term_months,
                        "remaining_balance": round(l.remaining_balance, 2),
                        "total_interest_collected": round(l.total_interest_collected, 2),
                        "total_principal_repaid": round(l.total_principal_repaid, 2),
                    }
                    for l in state.active_loans
                ],
                "cumulative_interest": round(state.cumulative_interest, 2),
                "cumulative_losses": round(state.cumulative_losses, 2),
                "cumulative_fees": round(state.cumulative_fees, 2),
                "cumulative_workout_cost": round(state.cumulative_workout_cost, 2),
                "defaults": sum(1 for o in state.resolved_loans if o.defaulted),
                "frauds_funded": sum(1 for o in state.resolved_loans if o.was_fraud),
                "deep_uw_deferred": state.deep_uw_deferred,
                "tokens_in": state.cumulative_tokens_in,
                "tokens_out": state.cumulative_tokens_out,
                "cost_usd": round(state.cumulative_cost_usd, 4),
            }

        self.week_details.append({
            "week": week,
            "final_resolution": True,
            "borrowers": [],
            "decisions": [],
            "booked_loans": [],
            "events": events,
            "lender_snapshots": lender_snapshots,
        })

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Export full season data as a JSON-serializable dict for the web viewer."""
        lenders = []
        for lender in self.base_lenders:
            state = self.lender_states[lender.id]
            total_pnl = self._net_pnl(state)
            lenders.append({
                "id": lender.id,
                "name": state.lender_name,
                "model": state.model,
                "total_capital": state.total_capital,
                "net_pnl": round(total_pnl, 2),
                "deployed": round(state.deployed_capital, 2),
                "deals_won": state.deals_won,
                "deals_rejected": state.deals_rejected,
                "deals_passed": state.deals_passed,
                "deals_lost": state.deals_lost,
                "cumulative_interest": round(state.cumulative_interest, 2),
                "cumulative_losses": round(state.cumulative_losses, 2),
                "cumulative_fees": round(state.cumulative_fees, 2),
                "cumulative_workout_cost": round(state.cumulative_workout_cost, 2),
                "defaults": sum(1 for o in state.resolved_loans if o.defaulted),
                "frauds_funded": sum(1 for o in state.resolved_loans if o.was_fraud),
                "weekly_utilization": [round(u, 4) for u in state.weekly_utilization],
                "weekly_snapshots": state.weekly_snapshots,
                "deep_uw_deferred": state.deep_uw_deferred,
                "weekly_deep_uw_used": state.weekly_deep_uw_used,
                "active_loans": [
                    {
                        "loan_id": l.loan_id,
                        "borrower_id": l.borrower_id,
                        "borrower_name": l.borrower_name,
                        "status": l.status,
                        "months_elapsed": l.months_elapsed,
                        "term_months": l.term_months,
                        "remaining_balance": round(l.remaining_balance, 2),
                        "total_interest_collected": round(l.total_interest_collected, 2),
                        "total_principal_repaid": round(l.total_principal_repaid, 2),
                    }
                    for l in state.active_loans
                ],
                "tokens_in": state.cumulative_tokens_in,
                "tokens_out": state.cumulative_tokens_out,
                "cost_usd": round(state.cumulative_cost_usd, 4),
                "scratchpad": state.scratchpad or None,
                "consistency_agreement_rate": (
                    round(state.consistency_agreement_rate, 4)
                    if state.consistency_checks else None
                ),
                "consistency_checks": len(state.consistency_checks),
                "uw_ops_cost": round(state.cumulative_uw_ops_cost, 2),
            })

        return {
            "type": "season",
            "config": {
                "weeks": self.config.weeks,
                "cohort_size": self.config.cohort_size,
                "months_per_week": self.config.months_per_week,
                "season_mix": self.config.season_mix,
                "seed": self.config.seed,
                "arrival_phases": self.config.arrival_phases,
                "deep_uw_slots_per_week": self.config.deep_uw_slots_per_week,
                "borrower_patience_weeks": self.config.borrower_patience_weeks,
                "offer_validity_weeks": self.config.offer_validity_weeks,
                "scratchpad": self.config.scratchpad,
                "underwriting_cost_mode": self.config.underwriting_cost_mode,
                "uw_cost_multiplier": self.config.uw_cost_multiplier,
                "consistency_samples": self.config.consistency_samples,
                "consistency_sample_pct": self.config.consistency_sample_pct,
            },
            "lenders": lenders,
            "weeks": self.week_details,
        }

    # ------------------------------------------------------------------
    # Season report
    # ------------------------------------------------------------------

    def _print_season_report(self) -> None:
        print(f"\n{'=' * 70}")
        print("  SEASON RESULTS")
        print(f"{'=' * 70}")

        for state in self.lender_states.values():
            total_pnl = self._net_pnl(state)
            avg_util = (
                sum(state.weekly_utilization) / len(state.weekly_utilization)
                if state.weekly_utilization else 0.0
            )

            print(f"\n  {state.lender_name} ({state.model})")
            print(f"    Capital: ${state.total_capital:,.0f}")
            print(f"    Deals Won: {state.deals_won} | Lost: {state.deals_lost} | "
                  f"Rejected: {state.deals_rejected} | Passed: {state.deals_passed}")
            print(f"    Interest Earned: ${state.cumulative_interest:,.0f}")
            print(f"    Fees Earned: ${state.cumulative_fees:,.0f}")
            print(f"    Losses: ${state.cumulative_losses:,.0f}")
            print(f"    Workout Costs: ${state.cumulative_workout_cost:,.0f}")
            print(f"    Net P&L: ${total_pnl:,.0f}")
            print(f"    Avg Utilization: {avg_util:.1%}")
            print(f"    Concentration Violation Weeks: "
                  f"{state.weeks_with_concentration_violations}")
            if self.config.deep_uw_slots_per_week > 0:
                avg_slots = (
                    sum(state.weekly_deep_uw_used) / len(state.weekly_deep_uw_used)
                    if state.weekly_deep_uw_used
                    else 0.0
                )
                print(
                    f"    Deep UW (avg/week): {avg_slots:.1f} "
                    f"| Deferred: {state.deep_uw_deferred}"
                )

            # Consistency
            if state.consistency_checks:
                print(f"    Decision Consistency: "
                      f"{state.consistency_agreement_rate:.0%} "
                      f"({len(state.consistency_checks)} checks)")

            # Underwriting ops cost (real mode)
            if state.cumulative_uw_ops_cost > 0:
                print(f"    UW Ops Cost: ${state.cumulative_uw_ops_cost:,.0f} "
                      f"(API: ${state.cumulative_cost_usd:.4f} × "
                      f"{self.config.uw_cost_multiplier:g})")

            # Scratchpad
            if state.scratchpad:
                pad_lines = state.scratchpad.count("\n") + 1
                print(f"    Scratchpad: {pad_lines} note(s)")

            # Loan breakdown
            resolved = state.resolved_loans
            defaults = sum(1 for o in resolved if o.defaulted)
            frauds = sum(1 for o in resolved if o.was_fraud)
            print(f"    Resolved Loans: {len(resolved)} "
                  f"(defaults: {defaults}, frauds funded: {frauds})")

        # Cost summary
        is_mock = self.engine_kwargs.get("mock", False)
        print_season_cost_summary(self.lender_states, self.config, mock=is_mock)
