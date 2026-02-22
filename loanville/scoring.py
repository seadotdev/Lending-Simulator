"""
Scoring and final reporting for the lending simulation.

Score = RAROC (Risk-Adjusted Return on Capital) vs a risk-free benchmark.

A lender's job is to deploy capital profitably.  Rejecting everything is
safe but earns nothing; approving everything is reckless.  The scoring
system measures actual P&L (interest earned minus principal lost minus
funding cost) against what the lender *could* have earned at the risk-free
rate on the same capital, then applies penalties for fraud, concentration
breaches, and risk (loss volatility).

Key mechanics:
  - Opportunity cost: undeployed available capital earns the risk-free rate
    as a benchmark.  A lender that deploys nothing scores 0 net P&L but
    "missed" the risk-free return, giving it a negative relative score.
  - Funding cost: deployed capital incurs a cost-of-funds charge
    (principal x funding_rate x term).  Lending must beat the cost of money.
  - RAROC: profit is penalized by loss volatility (lambda * sigma * n).
    A portfolio with highly variable per-loan outcomes is riskier and scores
    lower, even if the mean P&L is identical.  The penalty scales with per-loan
    sigma, not aggregate portfolio size — avoiding penalizing diversification.
  - Volume floor: models must deploy at least 20% of available capital.
    Under-deployment incurs a quadratic penalty, preventing gaming by
    declining everything to avoid risk.
  - Hard constraints: default rate caps (30%) and min ROE thresholds act as
    disqualifiers — breaching them incurs a quadratic penalty that scales
    with the severity of the violation.
  - Yield drag: loans priced below the lender's target yield incur a
    penalty proportional to the shortfall, penalizing giveaway rates.
  - Fraud penalty: 25% of the fraud loan's principal is deducted on top
    of the actual loss (regulatory / reputational cost).
  - Concentration penalty: 5% of excess exposure above the sector limit.
"""

import math
from typing import Optional

from .models import (
    BookedLoan,
    Borrower,
    LenderConfig,
    LenderDecision,
    LenderScore,
    LoanOutcome,
)

# Annualized risk-free rate used as opportunity cost benchmark
RISK_FREE_RATE = 0.05
# Simulation horizon in months (used to prorate risk-free earnings)
SIM_HORIZON_MONTHS = 24

# --- Funding cost ---
# Annualized cost of funds (what the lender pays to borrow/source capital)
FUNDING_RATE = 0.04

# --- RAROC risk penalty ---
# Coefficient for loss-volatility penalty: higher = more penalty for variance
RISK_LAMBDA = 0.5

# --- Volume floor ---
# Minimum deployment ratio (fraction of available capital).  Models that deploy
# less than this fraction are penalized proportionally — prevents gaming by
# declining everything to avoid risk penalties.
MIN_DEPLOYMENT_RATIO = 0.20  # must deploy >= 20% of available capital
VOLUME_PENALTY_LAMBDA = 0.5  # penalty = lambda * shortfall^2 * available_capital

# --- Hard constraint thresholds ---
# If default rate (defaults / deals_won) exceeds this, severe penalty
MAX_DEFAULT_RATE = 0.30
# If return on deployed capital is below this threshold, severe penalty
MIN_ROE_THRESHOLD = -0.10
# Hard constraint penalties scale quadratically with severity of violation
HARD_CONSTRAINT_BASE_PCT = 5.0  # base penalty in ppt for at-threshold violation


# ---------------------------------------------------------------------------
# Per-loan payoff computation (used by both scoring and per-applicant Elo)
# ---------------------------------------------------------------------------

def compute_loan_payoff(
    principal: float,
    interest_rate: float,       # annual percentage (e.g. 10.0 for 10%)
    term_months: int,
    true_outcome: str,          # "good", "bad", "fraud"
    months_before_default: Optional[int] = None,
    funding_rate: float = FUNDING_RATE,
) -> dict:
    """Compute realized P&L for a single loan given the borrower's true outcome.

    Returns a dict with:
      - interest_earned: total interest collected before default (if any)
      - principal_lost: unrecovered principal
      - funding_cost: cost of funds for the capital deployed
      - fraud_penalty: extra 25% regulatory/reputational charge on fraud
      - net_profit: interest - principal_lost - funding_cost - fraud_penalty
    """
    monthly_rate = interest_rate / 100.0 / 12.0

    if true_outcome == "fraud":
        # Immediate default — total principal loss, minimal funding period
        fc = principal * funding_rate * (1.0 / 12.0)  # ~1 month before discovery
        fp = principal * 0.25
        return {
            "interest_earned": 0.0,
            "principal_lost": principal,
            "funding_cost": fc,
            "fraud_penalty": fp,
            "net_profit": -principal - fc - fp,
        }

    if true_outcome == "bad":
        months_paid = min(months_before_default or 6, term_months)

        # Amortization schedule
        if monthly_rate > 0:
            payment = principal * (monthly_rate * (1 + monthly_rate) ** term_months) / \
                      ((1 + monthly_rate) ** term_months - 1)
        else:
            payment = principal / term_months

        remaining = principal
        total_interest = 0.0
        fc = 0.0
        for _ in range(months_paid):
            # Funding cost on outstanding balance this month
            fc += remaining * funding_rate / 12.0
            interest_portion = remaining * monthly_rate
            principal_portion = payment - interest_portion
            total_interest += interest_portion
            remaining -= principal_portion

        principal_lost = max(0.0, remaining)

        return {
            "interest_earned": total_interest,
            "principal_lost": principal_lost,
            "funding_cost": fc,
            "fraud_penalty": 0.0,
            "net_profit": total_interest - principal_lost - fc,
        }

    # Good loan — full repayment
    if monthly_rate > 0:
        payment = principal * (monthly_rate * (1 + monthly_rate) ** term_months) / \
                  ((1 + monthly_rate) ** term_months - 1)
        total_interest = payment * term_months - principal
    else:
        total_interest = 0.0
        payment = principal / term_months if term_months > 0 else 0.0

    # Funding cost on amortizing outstanding balance
    fc = 0.0
    remaining = principal
    for _ in range(term_months):
        fc += remaining * funding_rate / 12.0
        if monthly_rate > 0:
            principal_portion = payment - remaining * monthly_rate
        else:
            principal_portion = payment
        remaining -= principal_portion

    return {
        "interest_earned": total_interest,
        "principal_lost": 0.0,
        "funding_cost": fc,
        "fraud_penalty": 0.0,
        "net_profit": total_interest - fc,
    }


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


def calculate_perfect_score(
    lender: LenderConfig,
    borrowers: list[Borrower],
) -> float:
    """Calculate the theoretical best score if the lender had perfect foresight.

    Assumes the lender:
    - Approves all good borrowers (that fit within constraints)
    - Rejects all bad and fraud borrowers
    - Prices every loan at their target yield
    - Uses the standard sim horizon as term
    - Wins every deal (ignores competition — gives an upper bound)
    """
    existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
    available_capital = lender.total_capital - existing_deployed

    # Track sector exposure from existing portfolio
    sector_exposure: dict[str, float] = {}
    for loan in lender.existing_portfolio:
        sector_exposure[loan.sector] = sector_exposure.get(loan.sector, 0) + loan.remaining_balance

    # Greedily fund good borrowers (sorted by loan size descending for max deployment)
    good_borrowers = [b for b in borrowers if b.true_outcome == "good"]
    good_borrowers.sort(key=lambda b: b.dossier.loan_request_amount, reverse=True)

    total_interest = 0.0
    capital_remaining = available_capital
    rate = lender.target_yield_pct
    term = SIM_HORIZON_MONTHS
    monthly_rate = rate / 100.0 / 12.0

    for b in good_borrowers:
        principal = b.dossier.loan_request_amount

        # Check max single loan
        if principal > lender.max_single_loan:
            principal = lender.max_single_loan

        # Check available capital
        if principal > capital_remaining:
            continue

        # Check sector concentration limit
        sector = b.dossier.sector
        current = sector_exposure.get(sector, 0)
        limit = lender.sector_limits.get(sector, 0.25)
        max_allowed = lender.total_capital * limit
        if current + principal > max_allowed:
            # Can we do a partial? Skip for simplicity.
            continue

        # Fund this loan
        capital_remaining -= principal
        sector_exposure[sector] = current + principal

        # Calculate interest earned (full amortization at target yield)
        if monthly_rate > 0:
            payment = principal * (monthly_rate * (1 + monthly_rate) ** term) / \
                      ((1 + monthly_rate) ** term - 1)
            interest = payment * term - principal
        else:
            interest = 0.0
        total_interest += interest

    # Compute funding cost for perfect portfolio (amortizing)
    # For simplicity in the theoretical max, use average outstanding balance ≈ principal/2
    total_deployed = available_capital - capital_remaining
    total_funding_cost = (total_deployed / 2.0) * FUNDING_RATE * (SIM_HORIZON_MONTHS / 12)

    # Score using same formula as actual scoring (no losses, no penalties, no volatility)
    net_pnl = total_interest - total_funding_cost
    if available_capital > 0:
        actual_return_pct = (net_pnl / available_capital) * 100
        benchmark_pct = RISK_FREE_RATE * (SIM_HORIZON_MONTHS / 12) * 100
        return actual_return_pct - benchmark_pct
    return 0.0


def score_lenders(
    lenders: list[LenderConfig],
    all_decisions: dict[str, list[LenderDecision]],
    booked_loans: list[BookedLoan],
    loan_outcomes: list[LoanOutcome],
    deal_results: dict[str, dict],
    borrowers: list[Borrower] | None = None,
) -> list[LenderScore]:
    """Calculate final scores for all lenders.

    The score is RAROC-adjusted: net P&L minus funding cost minus penalties,
    with a volatility penalty and hard-constraint disqualifiers.
    """
    scores = []
    borrower_map = {b.id: b for b in borrowers} if borrowers else {}

    for lender in lenders:
        lender_outcomes = [o for o in loan_outcomes if o.lender_id == lender.id]
        lender_loans = [l for l in booked_loans if l.lender_id == lender.id]
        lender_decisions = all_decisions.get(lender.id, [])

        # Basic counts
        deals_won = len(lender_loans)
        deals_rejected = sum(1 for d in lender_decisions if d.decision == "REJECT")
        n_decisions = len(lender_decisions)
        approval_rate = (n_decisions - deals_rejected) / n_decisions if n_decisions > 0 else 0.0

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
        existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
        available_capital = lender.total_capital - existing_deployed

        # --- Funding cost (on amortizing outstanding balance) ---
        funding_cost_dollars = 0.0
        for loan in lender_loans:
            outcome = next((o for o in lender_outcomes if o.loan_id == loan.id), None)
            months_active = loan.term_months
            if outcome:
                months_active = outcome.months_paid if outcome.defaulted else loan.term_months

            monthly_rate = loan.interest_rate / 100.0 / 12.0
            if monthly_rate > 0:
                pmt = loan.principal * (monthly_rate * (1 + monthly_rate) ** loan.term_months) / \
                      ((1 + monthly_rate) ** loan.term_months - 1)
            else:
                pmt = loan.principal / loan.term_months if loan.term_months > 0 else 0.0

            remaining = loan.principal
            for _ in range(months_active):
                funding_cost_dollars += remaining * FUNDING_RATE / 12.0
                if monthly_rate > 0:
                    principal_portion = pmt - remaining * monthly_rate
                else:
                    principal_portion = pmt
                remaining -= principal_portion

        # --- Per-loan profit for volatility computation ---
        per_loan_profits: list[float] = []
        for loan in lender_loans:
            outcome = next((o for o in lender_outcomes if o.loan_id == loan.id), None)
            if outcome:
                months_active = outcome.months_paid if outcome.defaulted else loan.term_months
                # Amortizing funding cost for this loan
                loan_mr = loan.interest_rate / 100.0 / 12.0
                if loan_mr > 0:
                    loan_pmt = loan.principal * (loan_mr * (1 + loan_mr) ** loan.term_months) / \
                               ((1 + loan_mr) ** loan.term_months - 1)
                else:
                    loan_pmt = loan.principal / loan.term_months if loan.term_months > 0 else 0.0
                loan_fc = 0.0
                loan_rem = loan.principal
                for _ in range(months_active):
                    loan_fc += loan_rem * FUNDING_RATE / 12.0
                    if loan_mr > 0:
                        loan_pp = loan_pmt - loan_rem * loan_mr
                    else:
                        loan_pp = loan_pmt
                    loan_rem -= loan_pp

                loan_fp = loan.principal * 0.25 if outcome.was_fraud else 0.0
                loan_profit = outcome.total_interest_paid - outcome.principal_lost - loan_fc - loan_fp
                per_loan_profits.append(loan_profit)

        # --- Loss volatility (RAROC) ---
        loss_volatility = 0.0
        if len(per_loan_profits) > 1:
            mean_profit = sum(per_loan_profits) / len(per_loan_profits)
            variance = sum((p - mean_profit) ** 2 for p in per_loan_profits) / (len(per_loan_profits) - 1)
            loss_volatility = math.sqrt(variance)

        # Risk penalty: lambda * sigma (penalizes per-loan outcome variance,
        # not aggregate portfolio size — avoids penalizing diversification)
        n_loans = len(per_loan_profits)
        risk_penalty_dollars = RISK_LAMBDA * loss_volatility * n_loans if n_loans > 0 else 0.0

        # Volume penalty: quadratic penalty for deploying less than the floor
        deployment_ratio = total_deployed / available_capital if available_capital > 0 else 0.0
        if deployment_ratio < MIN_DEPLOYMENT_RATIO:
            shortfall = MIN_DEPLOYMENT_RATIO - deployment_ratio
            volume_penalty_dollars = VOLUME_PENALTY_LAMBDA * (shortfall ** 2) * available_capital
        else:
            volume_penalty_dollars = 0.0

        # --- Fraud penalty (extra cost beyond actual loss) ---
        fraud_penalty_dollars = 0.0
        for o in lender_outcomes:
            if o.was_fraud:
                fraud_penalty_dollars += o.principal * 0.25

        # --- Concentration penalty ---
        violations = find_concentration_violations(lender, booked_loans)
        concentration_penalty_dollars = _concentration_penalty_dollars(lender, booked_loans)

        # --- Yield drag ---
        yield_drag = 0.0
        for loan in lender_loans:
            if loan.interest_rate < lender.target_yield_pct:
                shortfall_pct = (lender.target_yield_pct - loan.interest_rate) / 100.0
                yield_drag += loan.principal * shortfall_pct * (loan.term_months / 12)

        # --- Hard constraints ---
        hard_violations: list[str] = []
        default_rate = defaults_count / deals_won if deals_won > 0 else 0.0
        roe_pct = (net_pnl / total_deployed * 100) if total_deployed > 0 else 0.0

        if deals_won > 0 and default_rate > MAX_DEFAULT_RATE:
            hard_violations.append(
                f"Default rate {default_rate*100:.0f}% exceeds cap {MAX_DEFAULT_RATE*100:.0f}%"
            )
        if total_deployed > 0 and roe_pct < MIN_ROE_THRESHOLD * 100:
            hard_violations.append(
                f"ROE {roe_pct:.1f}% below minimum {MIN_ROE_THRESHOLD*100:.0f}%"
            )

        # Quadratic hard-constraint penalty: scales with severity of violation.
        # "slightly off" is tolerable; "way off" is crushed.
        hard_constraint_penalty_dollars = 0.0
        if deals_won > 0 and default_rate > MAX_DEFAULT_RATE:
            overshoot = (default_rate - MAX_DEFAULT_RATE) / MAX_DEFAULT_RATE  # relative
            hard_constraint_penalty_dollars += HARD_CONSTRAINT_BASE_PCT * (1 + overshoot ** 2) / 100.0 * available_capital
        if total_deployed > 0 and roe_pct < MIN_ROE_THRESHOLD * 100:
            undershoot = abs(roe_pct - MIN_ROE_THRESHOLD * 100) / 100.0
            hard_constraint_penalty_dollars += HARD_CONSTRAINT_BASE_PCT * (1 + undershoot ** 2) / 100.0 * available_capital

        # --- Final RAROC score ---
        adjusted_pnl = (
            net_pnl
            - funding_cost_dollars
            - fraud_penalty_dollars
            - concentration_penalty_dollars
            - yield_drag
            - risk_penalty_dollars
            - volume_penalty_dollars
            - hard_constraint_penalty_dollars
        )

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
        risk_penalty_pct = (risk_penalty_dollars / available_capital * 100) if available_capital > 0 else 0.0
        volume_penalty_pct = (volume_penalty_dollars / available_capital * 100) if available_capital > 0 else 0.0
        hard_constraint_penalty_pct = (hard_constraint_penalty_dollars / available_capital * 100) if available_capital > 0 else 0.0

        # Perfect score (theoretical max with omniscient foresight)
        perfect = calculate_perfect_score(lender, borrowers) if borrowers else 0.0

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
            perfect_score=perfect,
            # New: RAROC
            funding_cost=funding_cost_dollars,
            loss_volatility=loss_volatility,
            risk_penalty_pct=risk_penalty_pct,
            raroc_score=final_score,
            # New: Hard constraints
            default_rate=default_rate,
            roe_pct=roe_pct,
            hard_constraint_violations=hard_violations,
            hard_constraint_penalty_pct=hard_constraint_penalty_pct,
            # New: Diagnostics
            approval_rate=approval_rate,
            deployment_ratio=deployment_ratio,
            volume_penalty_pct=volume_penalty_pct,
        ))

    return scores


def print_final_report(scores: list[LenderScore]) -> None:
    """Print the final scoring report and rankings."""
    print("\n" + "=" * 70)
    print("FINAL SCORECARD (RAROC)")
    print("=" * 70)

    # Sort by final adjusted score descending
    ranked = sorted(scores, key=lambda s: s.final_adjusted_score, reverse=True)

    benchmark_pct = RISK_FREE_RATE * (SIM_HORIZON_MONTHS / 12) * 100
    print(f"\n  Benchmark: {benchmark_pct:.1f}% risk-free return "
          f"({RISK_FREE_RATE*100:.0f}% annual over {SIM_HORIZON_MONTHS}mo)")
    print(f"  Funding rate: {FUNDING_RATE*100:.0f}% | "
          f"Risk lambda: {RISK_LAMBDA} | "
          f"Max default rate: {MAX_DEFAULT_RATE*100:.0f}% | "
          f"Min ROE: {MIN_ROE_THRESHOLD*100:.0f}% | "
          f"Min deploy: {MIN_DEPLOYMENT_RATIO*100:.0f}%")

    for rank, s in enumerate(ranked, 1):
        print(f"\n{'─' * 60}")
        print(f"  #{rank}  {s.lender_name}")
        print(f"       Model: {s.model}")
        print(f"{'─' * 60}")
        print(f"  Portfolio Activity:")
        print(f"    Deals Won:      {s.deals_won}")
        print(f"    Deals Lost:     {s.deals_lost}")
        print(f"    Deals Rejected: {s.deals_rejected}")
        print(f"    Approval Rate:  {s.approval_rate*100:.0f}%")
        print(f"  Financial Performance:")
        print(f"    Capital Deployed:    ${s.total_deployed:>12,.2f}")
        print(f"    Interest Earned:     ${s.total_interest_earned:>12,.2f}")
        print(f"    Principal Lost:      ${s.total_principal_lost:>12,.2f}")
        print(f"    Funding Cost:        ${s.funding_cost:>12,.2f}")
        print(f"    Net P&L:             ${s.net_return:>12,.2f}")
        if s.total_deployed > 0:
            print(f"    Raw ROI:             {s.roi_pct:>11.2f}%")
            print(f"    ROE:                 {s.roe_pct:>11.2f}%")
        else:
            print(f"    Raw ROI / ROE:            N/A (nothing deployed)")
        print(f"  Risk Metrics:")
        print(f"    Frauds Funded:       {s.frauds_funded}")
        print(f"    Total Defaults:      {s.defaults_count}")
        print(f"    Default Rate:        {s.default_rate*100:.0f}%")
        print(f"    Loss Volatility:     ${s.loss_volatility:>10,.0f}")
        if s.concentration_violations:
            print(f"    Concentration Violations:")
            for v in s.concentration_violations:
                print(f"      - {v}")
        else:
            print(f"    Concentration Violations: None")
        if s.hard_constraint_violations:
            print(f"    HARD CONSTRAINT VIOLATIONS:")
            for v in s.hard_constraint_violations:
                print(f"      !! {v}")
        print(f"    Deployment Ratio:    {s.deployment_ratio*100:>10.0f}%")
        print(f"  Penalties:")
        print(f"    Fraud Penalty:         {s.fraud_penalty_pct:>5.1f}%")
        print(f"    Concentration Penalty: {s.concentration_penalty_pct:>5.1f}%")
        print(f"    Risk Penalty (RAROC):  {s.risk_penalty_pct:>5.1f}%")
        if s.volume_penalty_pct > 0:
            print(f"    Volume Penalty:        {s.volume_penalty_pct:>5.1f}%")
        if s.hard_constraint_penalty_pct > 0:
            print(f"    Hard Constraint Pen:   {s.hard_constraint_penalty_pct:>5.1f}%")
        print(f"  {'='*40}")
        print(f"  RAROC SCORE vs BENCHMARK: {s.final_adjusted_score:>+8.2f}%")
        if s.perfect_score != 0.0:
            gap = s.final_adjusted_score - s.perfect_score
            print(f"  PERFECT SCORE:            {s.perfect_score:>+8.2f}%")
            print(f"  GAP TO PERFECT:           {gap:>+8.2f}%")

    # Winner announcement
    if ranked:
        print(f"\n{'*' * 70}")
        winner = ranked[0]
        if len(ranked) > 1 and winner.final_adjusted_score == ranked[1].final_adjusted_score:
            print(f"  TIE!")
        else:
            print(f"  WINNER: {winner.lender_name} ({winner.model})")
            print(f"  RAROC Score: {winner.final_adjusted_score:+.2f}%")
        print(f"{'*' * 70}")
