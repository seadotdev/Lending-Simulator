"""
Scoring and final reporting for the lending simulation.

Score = Net P&L vs a risk-free benchmark.

A lender's job is to deploy capital profitably.  Rejecting everything is
safe but earns nothing; approving everything is reckless.  The scoring
system measures actual P&L (interest earned minus principal lost) against
what the lender *could* have earned at the risk-free rate on the same
capital, then applies penalties for fraud and concentration breaches.

Key mechanics:
  - Opportunity cost: undeployed available capital earns the risk-free rate
    as a benchmark.  A lender that deploys nothing scores 0 net P&L but
    "missed" the risk-free return, giving it a negative relative score.
  - Yield drag: loans priced below the lender's target yield incur a
    penalty proportional to the shortfall, penalizing giveaway rates.
  - Fraud penalty: 25% of the fraud loan's principal is deducted on top
    of the actual loss (regulatory / reputational cost).
  - Concentration penalty: 5% of excess exposure above the sector limit.
"""

from .models import (
    BookedLoan,
    LenderConfig,
    LenderDecision,
    LenderScore,
    LoanOutcome,
)

# Annualized risk-free rate used as opportunity cost benchmark
RISK_FREE_RATE = 0.05
# Simulation horizon in months (used to prorate risk-free earnings)
SIM_HORIZON_MONTHS = 24


def calculate_sector_exposure(
    lender: LenderConfig,
    booked_loans: list[BookedLoan],
) -> dict[str, float]:
    """Calculate total sector exposure including existing portfolio and new loans."""
    exposure: dict[str, float] = {}

    # Existing portfolio
    for loan in lender.existing_portfolio:
        exposure[loan.sector] = exposure.get(loan.sector, 0) + loan.remaining_balance

    # New booked loans
    for loan in booked_loans:
        if loan.lender_id == lender.id:
            exposure[loan.sector] = exposure.get(loan.sector, 0) + loan.principal

    return exposure


def find_concentration_violations(
    lender: LenderConfig,
    booked_loans: list[BookedLoan],
) -> list[str]:
    """Check which sector limits are breached."""
    exposure = calculate_sector_exposure(lender, booked_loans)
    total_capital = lender.total_capital
    violations = []

    for sector, amount in exposure.items():
        limit = lender.sector_limits.get(sector, 0.25)
        actual_pct = amount / total_capital
        if actual_pct > limit:
            violations.append(
                f"{sector}: {actual_pct*100:.1f}% (limit: {limit*100:.0f}%)"
            )

    return violations


def _concentration_penalty_dollars(
    lender: LenderConfig,
    booked_loans: list[BookedLoan],
) -> float:
    """Dollar penalty for concentration breaches: 5% of excess exposure."""
    exposure = calculate_sector_exposure(lender, booked_loans)
    total_capital = lender.total_capital
    penalty = 0.0
    for sector, amount in exposure.items():
        limit = lender.sector_limits.get(sector, 0.25)
        max_allowed = total_capital * limit
        if amount > max_allowed:
            excess = amount - max_allowed
            penalty += excess * 0.05
    return penalty


def score_lenders(
    lenders: list[LenderConfig],
    all_decisions: dict[str, list[LenderDecision]],
    booked_loans: list[BookedLoan],
    loan_outcomes: list[LoanOutcome],
    deal_results: dict[str, dict],
) -> list[LenderScore]:
    """Calculate final scores for all lenders."""
    scores = []

    for lender in lenders:
        lender_outcomes = [o for o in loan_outcomes if o.lender_id == lender.id]
        lender_loans = [l for l in booked_loans if l.lender_id == lender.id]
        lender_decisions = all_decisions.get(lender.id, [])

        # Basic counts
        deals_won = len(lender_loans)
        deals_rejected = sum(1 for d in lender_decisions if d.decision == "REJECT")

        # Count deals lost to competitors
        deals_lost = 0
        for d in lender_decisions:
            if d.decision == "APPROVE":
                result = deal_results.get(d.borrower_id, {})
                if result.get("outcome") == "booked" and result.get("winner") != lender.id:
                    deals_lost += 1

        # --- Financial metrics ---
        total_deployed = sum(o.principal for o in lender_outcomes)
        total_interest = sum(o.total_interest_paid for o in lender_outcomes)
        total_principal_lost = sum(o.principal_lost for o in lender_outcomes)
        net_pnl = total_interest - total_principal_lost

        # Fraud and default counts
        frauds_funded = sum(1 for o in lender_outcomes if o.was_fraud)
        defaults_count = sum(1 for o in lender_outcomes if o.defaulted)

        # --- Opportunity cost ---
        # Available capital that *could* have been deployed
        existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
        available_capital = lender.total_capital - existing_deployed
        # Risk-free benchmark on available capital over the sim horizon
        risk_free_earnings = available_capital * RISK_FREE_RATE * (SIM_HORIZON_MONTHS / 12)

        # --- Fraud penalty (extra cost beyond actual loss) ---
        fraud_penalty_dollars = 0.0
        for o in lender_outcomes:
            if o.was_fraud:
                fraud_penalty_dollars += o.principal * 0.25  # 25% regulatory/reputational

        # --- Concentration penalty ---
        violations = find_concentration_violations(lender, booked_loans)
        concentration_penalty_dollars = _concentration_penalty_dollars(lender, booked_loans)

        # --- Yield drag ---
        # Penalize loans priced below target yield
        yield_drag = 0.0
        for loan in lender_loans:
            if loan.interest_rate < lender.target_yield_pct:
                shortfall_pct = (lender.target_yield_pct - loan.interest_rate) / 100.0
                yield_drag += loan.principal * shortfall_pct * (loan.term_months / 12)

        # --- Final score ---
        # Score = (net P&L - penalties - yield drag) vs risk-free benchmark
        adjusted_pnl = net_pnl - fraud_penalty_dollars - concentration_penalty_dollars - yield_drag
        # Express as return on available capital, relative to risk-free
        if available_capital > 0:
            actual_return_pct = (adjusted_pnl / available_capital) * 100
            benchmark_pct = RISK_FREE_RATE * (SIM_HORIZON_MONTHS / 12) * 100
            final_score = actual_return_pct - benchmark_pct
        else:
            final_score = 0.0

        # Legacy percentage fields (kept for report compatibility)
        roi_pct = (net_pnl / total_deployed * 100) if total_deployed > 0 else 0.0
        fraud_penalty_pct = (fraud_penalty_dollars / available_capital * 100) if available_capital > 0 else 0.0
        concentration_penalty_pct = (concentration_penalty_dollars / available_capital * 100) if available_capital > 0 else 0.0

        scores.append(LenderScore(
            lender_id=lender.id,
            lender_name=lender.name,
            model=lender.model,
            total_deployed=total_deployed,
            total_interest_earned=total_interest,
            total_principal_lost=total_principal_lost,
            net_return=net_pnl,
            roi_pct=roi_pct,
            deals_won=deals_won,
            deals_lost=deals_lost,
            deals_rejected=deals_rejected,
            frauds_funded=frauds_funded,
            defaults_count=defaults_count,
            concentration_violations=violations,
            concentration_penalty_pct=concentration_penalty_pct,
            fraud_penalty_pct=fraud_penalty_pct,
            final_adjusted_score=final_score,
        ))

    return scores


def print_final_report(scores: list[LenderScore]) -> None:
    """Print the final scoring report and rankings."""
    print("\n" + "=" * 70)
    print("FINAL SCORECARD")
    print("=" * 70)

    # Sort by final adjusted score descending
    ranked = sorted(scores, key=lambda s: s.final_adjusted_score, reverse=True)

    benchmark_pct = RISK_FREE_RATE * (SIM_HORIZON_MONTHS / 12) * 100
    print(f"\n  Benchmark: {benchmark_pct:.1f}% risk-free return "
          f"({RISK_FREE_RATE*100:.0f}% annual over {SIM_HORIZON_MONTHS}mo)")

    for rank, s in enumerate(ranked, 1):
        print(f"\n{'─' * 60}")
        print(f"  #{rank}  {s.lender_name}")
        print(f"       Model: {s.model}")
        print(f"{'─' * 60}")
        print(f"  Portfolio Activity:")
        print(f"    Deals Won:      {s.deals_won}")
        print(f"    Deals Lost:     {s.deals_lost}")
        print(f"    Deals Rejected: {s.deals_rejected}")
        print(f"  Financial Performance:")
        print(f"    Capital Deployed:    ${s.total_deployed:>12,.2f}")
        print(f"    Interest Earned:     ${s.total_interest_earned:>12,.2f}")
        print(f"    Principal Lost:      ${s.total_principal_lost:>12,.2f}")
        print(f"    Net P&L:             ${s.net_return:>12,.2f}")
        if s.total_deployed > 0:
            print(f"    Raw ROI:             {s.roi_pct:>11.2f}%")
        else:
            print(f"    Raw ROI:                  N/A (nothing deployed)")
        print(f"  Risk Metrics:")
        print(f"    Frauds Funded:       {s.frauds_funded}")
        print(f"    Total Defaults:      {s.defaults_count}")
        if s.concentration_violations:
            print(f"    Concentration Violations:")
            for v in s.concentration_violations:
                print(f"      - {v}")
        else:
            print(f"    Concentration Violations: None")
        print(f"  Penalties:")
        print(f"    Fraud Penalty:       {s.fraud_penalty_pct:>5.1f}% of available capital")
        print(f"    Concentration Penalty: {s.concentration_penalty_pct:>5.1f}% of available capital")
        print(f"  {'='*40}")
        print(f"  SCORE vs BENCHMARK:    {s.final_adjusted_score:>+11.2f}%")

    # Winner announcement
    if ranked:
        print(f"\n{'*' * 70}")
        winner = ranked[0]
        if len(ranked) > 1 and winner.final_adjusted_score == ranked[1].final_adjusted_score:
            print(f"  TIE!")
        else:
            print(f"  WINNER: {winner.lender_name} ({winner.model})")
            print(f"  Score vs Benchmark: {winner.final_adjusted_score:+.2f}%")
        print(f"{'*' * 70}")
