"""
Scoring and final reporting for the lending simulation.

Calculates risk-adjusted returns, fraud penalties, and concentration compliance.
"""

from .models import (
    BookedLoan,
    LenderConfig,
    LenderDecision,
    LenderScore,
    LoanOutcome,
)


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

        # Financial metrics
        total_deployed = sum(o.principal for o in lender_outcomes)
        total_interest = sum(o.total_interest_paid for o in lender_outcomes)
        total_principal_lost = sum(o.principal_lost for o in lender_outcomes)
        net_return = total_interest - total_principal_lost

        roi_pct = (net_return / total_deployed * 100) if total_deployed > 0 else 0.0

        # Fraud and default counts
        frauds_funded = sum(1 for o in lender_outcomes if o.was_fraud)
        defaults_count = sum(1 for o in lender_outcomes if o.defaulted)

        # Concentration violations
        violations = find_concentration_violations(lender, booked_loans)

        # Penalties
        # Fraud: 20% score penalty per fraud funded
        fraud_penalty = min(1.0, frauds_funded * 0.20)

        # Concentration: 10% penalty per violation
        concentration_penalty = min(1.0, len(violations) * 0.10)

        # Final adjusted score
        if total_deployed > 0:
            base_score = roi_pct
            final_score = base_score * (1 - fraud_penalty) * (1 - concentration_penalty)
        else:
            final_score = 0.0

        scores.append(LenderScore(
            lender_id=lender.id,
            lender_name=lender.name,
            model=lender.model,
            total_deployed=total_deployed,
            total_interest_earned=total_interest,
            total_principal_lost=total_principal_lost,
            net_return=net_return,
            roi_pct=roi_pct,
            deals_won=deals_won,
            deals_lost=deals_lost,
            deals_rejected=deals_rejected,
            frauds_funded=frauds_funded,
            defaults_count=defaults_count,
            concentration_violations=violations,
            concentration_penalty_pct=concentration_penalty * 100,
            fraud_penalty_pct=fraud_penalty * 100,
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
        print(f"    Net Return:          ${s.net_return:>12,.2f}")
        print(f"    Raw ROI:             {s.roi_pct:>11.2f}%")
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
        print(f"    Fraud Penalty:       {s.fraud_penalty_pct:.0f}%")
        print(f"    Concentration Penalty: {s.concentration_penalty_pct:.0f}%")
        print(f"  {'='*40}")
        print(f"  FINAL ADJUSTED SCORE:  {s.final_adjusted_score:>11.2f}%")

    # Winner announcement
    if ranked:
        print(f"\n{'*' * 70}")
        winner = ranked[0]
        if len(ranked) > 1 and winner.final_adjusted_score == ranked[1].final_adjusted_score:
            print(f"  TIE!")
        else:
            print(f"  WINNER: {winner.lender_name} ({winner.model})")
            print(f"  Final Adjusted Score: {winner.final_adjusted_score:.2f}%")
        print(f"{'*' * 70}")
