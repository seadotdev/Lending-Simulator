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
  - RAROC: profit is penalized by loss volatility (lambda * sigma * sqrt(n)).
    A portfolio with highly variable per-loan outcomes is riskier and scores
    lower, even if the mean P&L is identical.  The penalty uses sigma * sqrt(n)
    (portfolio standard deviation scaling) rather than sigma * n, which would
    double-count scale.  This is analogous to a portfolio VaR-style term.
  - Volume floor: models must deploy a minimum fraction of available capital.
    Under-deployment incurs a quadratic penalty, preventing gaming by
    declining everything to avoid risk.
  - Hard constraints: default rate caps and min ROE thresholds act as
    disqualifiers — breaching them incurs a quadratic penalty that scales
    with the severity of the violation.
  - Yield drag: loans priced below the lender's target yield incur a
    penalty proportional to the shortfall, penalizing giveaway rates.
  - Fraud penalty: a configurable fraction of the fraud loan's principal is
    deducted on top of the actual loss (regulatory / reputational cost).
  - Concentration penalty: a configurable fraction of excess exposure above
    the sector limit.

All economic parameters are configurable via EconomicsConfig with 3 presets:
  balanced (default), aggressive, and conservative.

"""

import math
from typing import Optional

from .models import (
    BookedLoan,
    Borrower,
    EconomicsConfig,
    ECONOMICS_PRESETS,
    LenderConfig,
    LenderDecision,
    LenderScore,
    LoanOutcome,
    SeasonConfig,
    SeasonLenderState,
    SeasonScore,
)
from .run_schema import UnderwritingRun

# Default config instance — used when no config is passed and for backward compat
_DEFAULT = EconomicsConfig()

# Backward-compatible module-level constants (reference the default config)
RISK_FREE_RATE = _DEFAULT.risk_free_rate
SIM_HORIZON_MONTHS = _DEFAULT.sim_horizon_months
FUNDING_RATE = _DEFAULT.funding_rate
RISK_LAMBDA = _DEFAULT.risk_lambda
MIN_DEPLOYMENT_RATIO = _DEFAULT.min_deployment_ratio
VOLUME_PENALTY_LAMBDA = _DEFAULT.volume_penalty_lambda
MAX_DEFAULT_RATE = _DEFAULT.max_default_rate
MIN_ROE_THRESHOLD = _DEFAULT.min_roe_threshold
HARD_CONSTRAINT_BASE_PCT = _DEFAULT.hard_constraint_base_pct


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
    economics: Optional[EconomicsConfig] = None,
) -> dict:
    """Compute realized P&L for a single loan given the borrower's true outcome.

    Returns a dict with:
      - interest_earned: total interest collected before default (if any)
      - fees_earned: origination / other fees (simplified)
      - principal_lost: unrecovered principal
      - recovery_amount: recovered principal after default (simplified)
      - funding_cost: cost of funds for the capital deployed
      - servicing_cost: ongoing non-interest expense on outstanding balance
      - workout_cost: collections/legal expense on defaulted balance
      - fraud_penalty: extra 25% regulatory/reputational charge on fraud
      - net_profit: (interest + fees) - principal_lost - funding_cost
                    - servicing_cost - workout_cost - fraud_penalty
    """
    eco = economics or _DEFAULT
    fraud_rate = eco.fraud_penalty_rate
    monthly_rate = interest_rate / 100.0 / 12.0
    fees = max(0.0, principal * eco.origination_fee_rate)
    servicing_rate = max(0.0, eco.servicing_cost_rate_annual)

    if true_outcome == "fraud":
        # Immediate default — no payments; minimal funding/servicing period.
        recovery = max(0.0, principal * eco.recovery_rate_fraud)
        workout = max(0.0, principal * eco.workout_cost_rate)
        principal_lost = max(0.0, principal - recovery)
        fc = principal * funding_rate * (1.0 / 12.0)  # ~1 month before discovery
        sc = principal * servicing_rate * (1.0 / 12.0)
        fp = principal * fraud_rate
        return {
            "interest_earned": 0.0,
            "fees_earned": fees,
            "principal_lost": principal_lost,
            "recovery_amount": recovery,
            "funding_cost": fc,
            "servicing_cost": sc,
            "workout_cost": workout,
            "fraud_penalty": fp,
            "net_profit": fees - principal_lost - fc - sc - workout - fp,
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
        sc = 0.0
        for _ in range(months_paid):
            # Funding cost on outstanding balance this month
            fc += remaining * funding_rate / 12.0
            sc += remaining * servicing_rate / 12.0
            interest_portion = remaining * monthly_rate
            principal_portion = payment - interest_portion
            total_interest += interest_portion
            remaining -= principal_portion

        recovery = max(0.0, remaining * eco.recovery_rate_bad)
        workout = max(0.0, remaining * eco.workout_cost_rate)
        principal_lost = max(0.0, remaining - recovery)

        return {
            "interest_earned": total_interest,
            "fees_earned": fees,
            "principal_lost": principal_lost,
            "recovery_amount": recovery,
            "funding_cost": fc,
            "servicing_cost": sc,
            "workout_cost": workout,
            "fraud_penalty": 0.0,
            "net_profit": total_interest + fees - principal_lost - fc - sc - workout,
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
    sc = 0.0
    remaining = principal
    for _ in range(term_months):
        fc += remaining * funding_rate / 12.0
        sc += remaining * servicing_rate / 12.0
        if monthly_rate > 0:
            principal_portion = payment - remaining * monthly_rate
        else:
            principal_portion = payment
        remaining -= principal_portion

    return {
        "interest_earned": total_interest,
        "fees_earned": fees,
        "principal_lost": 0.0,
        "recovery_amount": 0.0,
        "funding_cost": fc,
        "servicing_cost": sc,
        "workout_cost": 0.0,
        "fraud_penalty": 0.0,
        "net_profit": total_interest + fees - fc - sc,
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
    economics: Optional[EconomicsConfig] = None,
) -> float:
    """Dollar penalty for concentration breaches."""
    eco = economics or _DEFAULT
    exposure = calculate_sector_exposure(lender, booked_loans)
    total_capital = lender.total_capital
    penalty = 0.0
    for sector, amount in exposure.items():
        limit = lender.sector_limits.get(sector, 0.25)
        max_allowed = total_capital * limit
        if amount > max_allowed:
            excess = amount - max_allowed
            penalty += excess * eco.concentration_penalty_rate
    return penalty


def calculate_perfect_score(
    lender: LenderConfig,
    borrowers: list[Borrower],
    economics: Optional[EconomicsConfig] = None,
) -> float:
    """Calculate the theoretical best score if the lender had perfect foresight.

    Assumes the lender:
    - Approves all good borrowers (that fit within constraints)
    - Rejects all bad and fraud borrowers
    - Prices every loan at their target yield
    - Uses the standard sim horizon as term
    - Wins every deal (ignores competition — gives an upper bound)
    """
    eco = economics or _DEFAULT
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
    term = eco.sim_horizon_months
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
    total_funding_cost = (total_deployed / 2.0) * eco.funding_rate * (eco.sim_horizon_months / 12)
    total_servicing_cost = (total_deployed / 2.0) * eco.servicing_cost_rate_annual * (eco.sim_horizon_months / 12)
    total_fees = total_deployed * eco.origination_fee_rate

    # Score using same formula as actual scoring (no losses, no penalties, no volatility)
    net_pnl = total_interest + total_fees - total_funding_cost - total_servicing_cost
    if available_capital > 0:
        actual_return_pct = (net_pnl / available_capital) * 100
        benchmark_pct = eco.risk_free_rate * (eco.sim_horizon_months / 12) * 100
        return actual_return_pct - benchmark_pct
    return 0.0


def compute_heuristic_baseline(
    lender: LenderConfig,
    borrowers: list[Borrower],
    economics: Optional[EconomicsConfig] = None,
) -> float:
    """Calculate score using a simple DSCR + margin + leverage heuristic.

    This baseline shows the task is solvable by straightforward financial
    analysis — it doesn't require LLM-specific reasoning.

    Heuristic rules:
      - Reject if net margin < 10%
      - Reject if DSCR < 1.25 (annual net income / annual debt service)
      - Reject if loan amount > 1.5x annual net income
      - Approve everything else at target yield
    """
    eco = economics or _DEFAULT
    existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
    available_capital = lender.total_capital - existing_deployed

    sector_exposure: dict[str, float] = {}
    for loan in lender.existing_portfolio:
        sector_exposure[loan.sector] = sector_exposure.get(loan.sector, 0) + loan.remaining_balance

    rate = lender.target_yield_pct
    term = eco.sim_horizon_months
    monthly_rate = rate / 100.0 / 12.0

    # Compute annual debt service for the standard loan
    if monthly_rate > 0:
        std_payment = 1.0 * (monthly_rate * (1 + monthly_rate) ** term) / \
                      ((1 + monthly_rate) ** term - 1)
    else:
        std_payment = 1.0 / term if term > 0 else 0.0
    annual_debt_service_per_dollar = std_payment * 12

    total_interest = 0.0
    total_fees = 0.0
    total_principal_lost = 0.0
    total_funding_cost = 0.0
    total_servicing_cost = 0.0
    total_workout_cost = 0.0
    capital_remaining = available_capital

    for b in borrowers:
        principal = b.dossier.loan_request_amount
        net_margin = (b.dossier.net_income / b.dossier.annual_revenue * 100) if b.dossier.annual_revenue > 0 else 0
        annual_ds = principal * annual_debt_service_per_dollar
        dscr = b.dossier.net_income / annual_ds if annual_ds > 0 else 0
        leverage = principal / b.dossier.net_income if b.dossier.net_income > 0 else float('inf')

        # Heuristic: reject if margin, DSCR, or leverage fail
        if net_margin < 10 or dscr < 1.25 or leverage > 1.5:
            continue

        if principal > lender.max_single_loan:
            continue
        if principal > capital_remaining:
            continue

        sector = b.dossier.sector
        current = sector_exposure.get(sector, 0)
        limit = lender.sector_limits.get(sector, 0.25)
        max_allowed = lender.total_capital * limit
        if current + principal > max_allowed:
            continue

        capital_remaining -= principal
        sector_exposure[sector] = current + principal

        # Compute payoff
        result = compute_loan_payoff(
            principal=principal,
            interest_rate=rate,
            term_months=term,
            true_outcome=b.true_outcome,
            months_before_default=b.months_before_default,
        )
        total_interest += result["interest_earned"]
        total_fees += result.get("fees_earned", 0.0)
        total_principal_lost += result["principal_lost"]
        total_funding_cost += result["funding_cost"]
        total_servicing_cost += result.get("servicing_cost", 0.0)
        total_workout_cost += result.get("workout_cost", 0.0)

    net_pnl = (
        total_interest
        + total_fees
        - total_principal_lost
        - total_funding_cost
        - total_servicing_cost
        - total_workout_cost
    )
    if available_capital > 0:
        actual_return_pct = (net_pnl / available_capital) * 100
        benchmark_pct = eco.risk_free_rate * (eco.sim_horizon_months / 12) * 100
        return actual_return_pct - benchmark_pct
    return 0.0


def compute_confusion_matrix(
    decisions: list[LenderDecision],
    borrowers: list[Borrower],
) -> dict[str, dict[str, int]]:
    """Compute confusion matrix of approve/reject decisions by borrower ground truth.

    Returns:
        {
            "good":  {"approved": N, "rejected": N},
            "bad":   {"approved": N, "rejected": N},
            "fraud": {"approved": N, "rejected": N},
        }
    """
    borrower_map = {b.id: b for b in borrowers}
    matrix: dict[str, dict[str, int]] = {
        "good": {"approved": 0, "rejected": 0},
        "bad": {"approved": 0, "rejected": 0},
        "fraud": {"approved": 0, "rejected": 0},
    }
    for d in decisions:
        b = borrower_map.get(d.borrower_id)
        if not b:
            continue
        # Skip LLM errors — they are infrastructure failures, not credit decisions
        if d.reasoning and d.reasoning.startswith("[LLM_ERROR]"):
            continue
        outcome = b.true_outcome
        if d.decision == "APPROVE":
            matrix[outcome]["approved"] += 1
        else:
            matrix[outcome]["rejected"] += 1
    return matrix


def compute_penalty_decomposition(
    lender: LenderConfig,
    lender_outcomes: list[LoanOutcome],
    booked_loans: list[BookedLoan],
    all_booked_loans: list[BookedLoan],
    economics: Optional[EconomicsConfig] = None,
) -> dict[str, float]:
    """Compute per-penalty-type dollar amounts for a lender.

    Returns dict with keys: funding_cost, fraud_penalty, concentration_penalty,
    yield_drag, risk_penalty, volume_penalty, hard_constraint_penalty.
    """
    eco = economics or _DEFAULT
    lender_loans = [l for l in booked_loans if l.lender_id == lender.id]
    existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
    available_capital = lender.total_capital - existing_deployed

    # Funding cost
    funding_cost = 0.0
    servicing_cost = 0.0
    for loan in lender_loans:
        outcome = next((o for o in lender_outcomes if o.loan_id == loan.id), None)
        months_active = loan.term_months
        if outcome:
            if outcome.was_fraud:
                months_active = 1
            elif outcome.defaulted or outcome.prepaid:
                months_active = max(1, outcome.months_paid)
            else:
                months_active = loan.term_months
        monthly_rate = loan.interest_rate / 100.0 / 12.0
        if monthly_rate > 0 and loan.term_months > 0:
            pmt = loan.principal * (monthly_rate * (1 + monthly_rate) ** loan.term_months) / \
                  ((1 + monthly_rate) ** loan.term_months - 1)
        else:
            pmt = loan.principal / loan.term_months if loan.term_months > 0 else 0.0
        remaining = loan.principal
        for _ in range(months_active):
            funding_cost += remaining * eco.funding_rate / 12.0
            servicing_cost += remaining * eco.servicing_cost_rate_annual / 12.0
            if monthly_rate > 0:
                pp = pmt - remaining * monthly_rate
            else:
                pp = pmt
            remaining -= pp

    # Fraud penalty
    fraud_penalty = sum(o.principal * eco.fraud_penalty_rate for o in lender_outcomes if o.was_fraud)

    # Concentration penalty
    concentration_penalty = _concentration_penalty_dollars(lender, all_booked_loans, economics=eco)

    # Yield drag
    yield_drag = 0.0
    for loan in lender_loans:
        if loan.interest_rate < lender.target_yield_pct:
            shortfall_pct = (lender.target_yield_pct - loan.interest_rate) / 100.0
            yield_drag += loan.principal * shortfall_pct * (loan.term_months / 12)

    # Per-loan profits for risk penalty
    per_loan_profits: list[float] = []
    for loan in lender_loans:
        outcome = next((o for o in lender_outcomes if o.loan_id == loan.id), None)
        if outcome:
            months_active = loan.term_months
            if outcome.was_fraud:
                months_active = 1
            elif outcome.defaulted or outcome.prepaid:
                months_active = max(1, outcome.months_paid)
            loan_mr = loan.interest_rate / 100.0 / 12.0
            if loan_mr > 0 and loan.term_months > 0:
                loan_pmt = loan.principal * (loan_mr * (1 + loan_mr) ** loan.term_months) / \
                           ((1 + loan_mr) ** loan.term_months - 1)
            else:
                loan_pmt = loan.principal / loan.term_months if loan.term_months > 0 else 0.0
            loan_fc = 0.0
            loan_sc = 0.0
            loan_rem = loan.principal
            for _ in range(months_active):
                loan_fc += loan_rem * eco.funding_rate / 12.0
                loan_sc += loan_rem * eco.servicing_cost_rate_annual / 12.0
                if loan_mr > 0:
                    loan_pp = loan_pmt - loan_rem * loan_mr
                else:
                    loan_pp = loan_pmt
                loan_rem -= loan_pp
            loan_fp = loan.principal * eco.fraud_penalty_rate if outcome.was_fraud else 0.0
            per_loan_profits.append(
                outcome.total_interest_paid
                + outcome.total_fees_paid
                - outcome.principal_lost
                - outcome.workout_cost
                - loan_sc
                - loan_fc
                - loan_fp
            )

    # Risk penalty
    loss_vol = 0.0
    if len(per_loan_profits) > 1:
        mean_p = sum(per_loan_profits) / len(per_loan_profits)
        var = sum((p - mean_p) ** 2 for p in per_loan_profits) / (len(per_loan_profits) - 1)
        loss_vol = math.sqrt(var)
    n_loans = len(per_loan_profits)
    risk_penalty = eco.risk_lambda * loss_vol * math.sqrt(n_loans) if n_loans > 0 else 0.0

    # Volume penalty
    total_deployed = sum(o.principal for o in lender_outcomes)
    deployment_ratio = total_deployed / available_capital if available_capital > 0 else 0.0
    if deployment_ratio < eco.min_deployment_ratio:
        shortfall = eco.min_deployment_ratio - deployment_ratio
        volume_penalty = eco.volume_penalty_lambda * (shortfall ** 2) * available_capital
    else:
        volume_penalty = 0.0

    # Hard constraint penalty
    deals_won = len(lender_loans)
    defaults_count = sum(1 for o in lender_outcomes if o.defaulted)
    default_rate = defaults_count / deals_won if deals_won > 0 else 0.0
    net_pnl = (
        sum(
            o.total_interest_paid
            + o.total_fees_paid
            - o.principal_lost
            - o.workout_cost
            for o in lender_outcomes
        )
        - servicing_cost
    )
    roe_pct = (net_pnl / total_deployed * 100) if total_deployed > 0 else 0.0

    hard_constraint_penalty = 0.0
    if deals_won > 0 and default_rate > eco.max_default_rate:
        overshoot = (default_rate - eco.max_default_rate) / eco.max_default_rate
        hard_constraint_penalty += eco.hard_constraint_base_pct * (1 + overshoot ** 2) / 100.0 * available_capital
    if total_deployed > 0 and roe_pct < eco.min_roe_threshold * 100:
        undershoot = abs(roe_pct - eco.min_roe_threshold * 100) / 100.0
        hard_constraint_penalty += eco.hard_constraint_base_pct * (1 + undershoot ** 2) / 100.0 * available_capital

    return {
        "funding_cost": round(funding_cost, 2),
        "fraud_penalty": round(fraud_penalty, 2),
        "concentration_penalty": round(concentration_penalty, 2),
        "yield_drag": round(yield_drag, 2),
        "risk_penalty": round(risk_penalty, 2),
        "volume_penalty": round(volume_penalty, 2),
        "hard_constraint_penalty": round(hard_constraint_penalty, 2),
    }


def bootstrap_raroc_interval(
    lender_outcomes: list[LoanOutcome],
    available_capital: float,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    economics: Optional[EconomicsConfig] = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for RAROC score.

    Resamples loan outcomes with replacement and computes the RAROC
    distribution to estimate uncertainty.

    Returns (lower_bound, upper_bound) as percentages.
    """
    if not lender_outcomes or available_capital <= 0:
        return (0.0, 0.0)

    eco = economics or _DEFAULT
    import random as _rng
    benchmark_pct = eco.risk_free_rate * (eco.sim_horizon_months / 12) * 100
    scores: list[float] = []

    for _ in range(n_bootstrap):
        sample = _rng.choices(lender_outcomes, k=len(lender_outcomes))
        net_pnl = sum(
            o.total_interest_paid
            + o.total_fees_paid
            - o.principal_lost
            - o.workout_cost
            for o in sample
        )
        total_deployed = sum(o.principal for o in sample)
        # Simplified: just net P&L vs benchmark, no penalties
        if total_deployed > 0:
            ret_pct = (net_pnl / available_capital) * 100
            scores.append(ret_pct - benchmark_pct)
        else:
            scores.append(-benchmark_pct)

    scores.sort()
    alpha = (1 - confidence) / 2
    lo_idx = int(alpha * len(scores))
    hi_idx = int((1 - alpha) * len(scores))
    return (scores[lo_idx], scores[min(hi_idx, len(scores) - 1)])


def score_lenders(
    lenders: list[LenderConfig],
    all_decisions: dict[str, list[LenderDecision]],
    booked_loans: list[BookedLoan],
    loan_outcomes: list[LoanOutcome],
    deal_results: dict[str, dict],
    borrowers: list[Borrower] | None = None,
    economics: Optional[EconomicsConfig] = None,
    runs: list[UnderwritingRun] | None = None,
) -> list[LenderScore]:
    """Calculate final scores for all lenders.

    The score is RAROC-adjusted: net P&L minus funding cost minus penalties,
    with a volatility penalty and hard-constraint disqualifiers.
    """
    eco = economics or _DEFAULT
    scores: list[LenderScore] = []
    runs = runs or []

    # Aggregate run-derived ops metrics (primarily populated in LOS mode).
    doc_requests_by_lender: dict[str, int] = {}
    llm_cost_by_lender: dict[str, float] = {}
    for run in runs:
        lid = (run.policy.params or {}).get("_lender_id", "")
        if not lid:
            pid = run.policy.policy_id or ""
            # policy_id format: "p_{lender_id}_{model}"
            parts = pid.split("_", 2)
            if len(parts) >= 2 and parts[0] == "p":
                lid = parts[1]
        if not lid:
            continue

        doc_reqs = sum(1 for s in run.trace.steps if getattr(s, "type", "") == "doc_request")
        doc_requests_by_lender[lid] = doc_requests_by_lender.get(lid, 0) + doc_reqs
        llm_cost_by_lender[lid] = llm_cost_by_lender.get(lid, 0.0) + float(
            getattr(run.trace.cost, "estimated_cost_usd", 0.0) or 0.0
        )

    # Simple discounting: only applies to delayed recoveries (when enabled).
    discount_recovery = eco.discount_rate_annual > 0 and eco.recovery_lag_months > 0
    recovery_df = 1.0
    if discount_recovery:
        recovery_df = 1.0 / ((1.0 + eco.discount_rate_annual / 12.0) ** eco.recovery_lag_months)

    def _effective_principal_loss(o: LoanOutcome) -> float:
        if not discount_recovery or o.recovery_amount <= 0:
            return o.principal_lost
        # Outcome stores "undiscounted recovery"; discounting reduces its value.
        return o.principal_lost + o.recovery_amount * (1.0 - recovery_df)

    for lender in lenders:
        lender_outcomes = [o for o in loan_outcomes if o.lender_id == lender.id]
        lender_loans = [l for l in booked_loans if l.lender_id == lender.id]
        lender_decisions = all_decisions.get(lender.id, [])

        # Basic counts
        deals_won = len(lender_loans)
        deals_errored = sum(
            1 for d in lender_decisions
            if d.reasoning and d.reasoning.startswith("[LLM_ERROR]")
        )
        deals_rejected = sum(
            1 for d in lender_decisions
            if d.decision != "APPROVE"
            and not (d.reasoning and d.reasoning.startswith("[LLM_ERROR]"))
        )
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
        total_fees = sum(o.total_fees_paid for o in lender_outcomes)
        total_principal_lost = sum(o.principal_lost for o in lender_outcomes)
        total_workout_cost = sum(o.workout_cost for o in lender_outcomes)
        credit_loss_scoring = sum(_effective_principal_loss(o) for o in lender_outcomes)

        # Fraud and default counts
        frauds_funded = sum(1 for o in lender_outcomes if o.was_fraud)
        defaults_count = sum(1 for o in lender_outcomes if o.defaulted)

        # --- Opportunity cost ---
        existing_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
        available_capital = lender.total_capital - existing_deployed

        # --- Origination ops cost (applies to all decisions, not just booked loans) ---
        underwriting_cost = len(lender_decisions) * eco.underwriting_cost_per_application_usd
        doc_requests = doc_requests_by_lender.get(lender.id, 0)
        doc_request_cost = doc_requests * eco.doc_request_cost_usd
        llm_cost = llm_cost_by_lender.get(lender.id, 0.0)
        ops_cost = underwriting_cost + doc_request_cost + llm_cost

        # --- Funding & servicing costs (on amortizing outstanding balance) ---
        funding_cost_dollars = 0.0
        servicing_cost_dollars = 0.0

        # --- Per-loan profit for volatility computation ---
        per_loan_profits: list[float] = []
        for loan in lender_loans:
            outcome = next((o for o in lender_outcomes if o.loan_id == loan.id), None)
            months_active = loan.term_months
            if outcome:
                if outcome.was_fraud:
                    months_active = 1  # discovery lag
                elif outcome.defaulted or outcome.prepaid:
                    months_active = max(1, outcome.months_paid)

            monthly_rate = loan.interest_rate / 100.0 / 12.0
            if monthly_rate > 0 and loan.term_months > 0:
                pmt = loan.principal * (monthly_rate * (1 + monthly_rate) ** loan.term_months) / \
                      ((1 + monthly_rate) ** loan.term_months - 1)
            else:
                pmt = loan.principal / loan.term_months if loan.term_months > 0 else 0.0

            remaining = loan.principal
            loan_fc = 0.0
            loan_sc = 0.0
            for _ in range(months_active):
                loan_fc += remaining * eco.funding_rate / 12.0
                loan_sc += remaining * eco.servicing_cost_rate_annual / 12.0
                if monthly_rate > 0:
                    principal_portion = pmt - remaining * monthly_rate
                else:
                    principal_portion = pmt
                remaining -= principal_portion

            funding_cost_dollars += loan_fc
            servicing_cost_dollars += loan_sc

            if outcome:
                eff_loss = _effective_principal_loss(outcome)
                loan_fp = loan.principal * eco.fraud_penalty_rate if outcome.was_fraud else 0.0
                per_loan_profits.append(
                    outcome.total_interest_paid
                    + outcome.total_fees_paid
                    - eff_loss
                    - outcome.workout_cost
                    - loan_sc
                    - loan_fc
                    - loan_fp
                )

        # Net P&L before funding + scoring penalties
        net_pnl = (
            total_interest
            + total_fees
            - credit_loss_scoring
            - total_workout_cost
            - servicing_cost_dollars
            - ops_cost
        )

        # --- Loss volatility (RAROC) ---
        loss_volatility = 0.0
        if len(per_loan_profits) > 1:
            mean_profit = sum(per_loan_profits) / len(per_loan_profits)
            variance = sum((p - mean_profit) ** 2 for p in per_loan_profits) / (len(per_loan_profits) - 1)
            loss_volatility = math.sqrt(variance)

        # Risk penalty: lambda * sigma * sqrt(n) (portfolio volatility scaling).
        # Uses sqrt(n) rather than n to avoid double-counting scale — analogous
        # to how portfolio standard deviation scales with sqrt(n) assets.
        n_loans = len(per_loan_profits)
        risk_penalty_dollars = eco.risk_lambda * loss_volatility * math.sqrt(n_loans) if n_loans > 0 else 0.0

        # Volume penalty: quadratic penalty for deploying less than the floor
        deployment_ratio = total_deployed / available_capital if available_capital > 0 else 0.0
        if deployment_ratio < eco.min_deployment_ratio:
            shortfall = eco.min_deployment_ratio - deployment_ratio
            volume_penalty_dollars = eco.volume_penalty_lambda * (shortfall ** 2) * available_capital
        else:
            volume_penalty_dollars = 0.0

        # --- Fraud penalty (extra cost beyond actual loss) ---
        fraud_penalty_dollars = 0.0
        for o in lender_outcomes:
            if o.was_fraud:
                fraud_penalty_dollars += o.principal * eco.fraud_penalty_rate

        # --- Concentration penalty ---
        violations = find_concentration_violations(lender, booked_loans)
        concentration_penalty_dollars = _concentration_penalty_dollars(lender, booked_loans, economics=eco)

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

        if deals_won > 0 and default_rate > eco.max_default_rate:
            hard_violations.append(
                f"Default rate {default_rate*100:.0f}% exceeds cap {eco.max_default_rate*100:.0f}%"
            )
        if total_deployed > 0 and roe_pct < eco.min_roe_threshold * 100:
            hard_violations.append(
                f"ROE {roe_pct:.1f}% below minimum {eco.min_roe_threshold*100:.0f}%"
            )

        # Quadratic hard-constraint penalty: scales with severity of violation.
        # "slightly off" is tolerable; "way off" is crushed.
        hard_constraint_penalty_dollars = 0.0
        if deals_won > 0 and default_rate > eco.max_default_rate:
            overshoot = (default_rate - eco.max_default_rate) / eco.max_default_rate  # relative
            hard_constraint_penalty_dollars += eco.hard_constraint_base_pct * (1 + overshoot ** 2) / 100.0 * available_capital
        if total_deployed > 0 and roe_pct < eco.min_roe_threshold * 100:
            undershoot = abs(roe_pct - eco.min_roe_threshold * 100) / 100.0
            hard_constraint_penalty_dollars += eco.hard_constraint_base_pct * (1 + undershoot ** 2) / 100.0 * available_capital

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
            benchmark_pct = eco.risk_free_rate * (eco.sim_horizon_months / 12) * 100
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
        perfect = calculate_perfect_score(lender, borrowers, economics=eco) if borrowers else 0.0

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
            deals_errored=deals_errored,
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
            # New: Extended economics
            total_fees_earned=total_fees,
            total_workout_cost=total_workout_cost,
            total_servicing_cost=servicing_cost_dollars,
            underwriting_cost=underwriting_cost,
            doc_request_cost=doc_request_cost,
            llm_cost=llm_cost,
            ops_cost=ops_cost,
            adjusted_pnl_dollars=adjusted_pnl,
        ))

    return scores


def print_final_report(
    scores: list[LenderScore],
    economics: Optional[EconomicsConfig] = None,
) -> None:
    """Print the final scoring report and rankings."""
    eco = economics or _DEFAULT
    print("\n" + "=" * 70)
    print(f"FINAL SCORECARD (RAROC) — Economics: {eco.name}")
    print("=" * 70)

    # Sort by final adjusted score descending
    ranked = sorted(scores, key=lambda s: s.final_adjusted_score, reverse=True)

    benchmark_pct = eco.risk_free_rate * (eco.sim_horizon_months / 12) * 100
    print(f"\n  Benchmark: {benchmark_pct:.1f}% risk-free return "
          f"({eco.risk_free_rate*100:.0f}% annual over {eco.sim_horizon_months}mo)")
    print(f"  Funding rate: {eco.funding_rate*100:.0f}% | "
          f"Risk lambda: {eco.risk_lambda} | "
          f"Max default rate: {eco.max_default_rate*100:.0f}% | "
          f"Min ROE: {eco.min_roe_threshold*100:.0f}% | "
          f"Min deploy: {eco.min_deployment_ratio*100:.0f}%")
    print(f"  Fees/Costs: orig fee {eco.origination_fee_rate*100:.1f}% | "
          f"servicing {eco.servicing_cost_rate_annual*100:.2f}%/yr | "
          f"UW ${eco.underwriting_cost_per_application_usd:.0f}/app | "
          f"doc req ${eco.doc_request_cost_usd:.0f} | "
          f"recovery bad {eco.recovery_rate_bad*100:.0f}% fraud {eco.recovery_rate_fraud*100:.0f}% | "
          f"workout {eco.workout_cost_rate*100:.0f}%")
    if eco.discount_rate_annual > 0 and eco.recovery_lag_months > 0:
        print(f"  Discounting: {eco.discount_rate_annual*100:.1f}%/yr "
              f"(recoveries discounted over {eco.recovery_lag_months}mo lag)")

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
        if s.total_fees_earned != 0.0:
            print(f"    Fees Earned:         ${s.total_fees_earned:>12,.2f}")
        print(f"    Principal Lost:      ${s.total_principal_lost:>12,.2f}")
        if s.total_workout_cost != 0.0:
            print(f"    Workout Cost:        ${s.total_workout_cost:>12,.2f}")
        if s.total_servicing_cost != 0.0:
            print(f"    Servicing Cost:      ${s.total_servicing_cost:>12,.2f}")
        if s.ops_cost != 0.0:
            print(f"    UW Ops Cost:         ${s.ops_cost:>12,.2f} "
                  f"(apps=${s.underwriting_cost:,.0f}, docs=${s.doc_request_cost:,.0f}, llm=${s.llm_cost:,.2f})")
        print(f"    Funding Cost:        ${s.funding_cost:>12,.2f}")
        print(f"    Net P&L (pre-pen):   ${s.net_return:>12,.2f}")
        print(f"    Adj. P&L (RAROC):    ${s.adjusted_pnl_dollars:>12,.2f}")
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


# ---------------------------------------------------------------------------
# Season scoring
# ---------------------------------------------------------------------------

def _score_credit_quality(state: SeasonLenderState, economics: EconomicsConfig) -> float:
    """Score credit quality 0-100 based on resolved loan outcomes."""
    resolved = state.resolved_loans
    if not resolved:
        return 50.0  # neutral if no loans

    total_interest = sum(o.total_interest_paid for o in resolved)
    total_fees = sum(o.total_fees_paid for o in resolved)
    total_losses = sum(o.principal_lost for o in resolved)
    total_workout = sum(o.workout_cost for o in resolved)
    total_principal = sum(o.principal for o in resolved)

    if total_principal == 0:
        return 50.0

    # Net P&L as fraction of deployed principal
    net_pnl = total_interest + total_fees - total_losses - total_workout
    pnl_ratio = net_pnl / total_principal

    # Score: 50 at breakeven, +/- up to 50 based on pnl_ratio
    # A 10% return maps to ~80, a -10% loss maps to ~20
    score = 50.0 + pnl_ratio * 300.0  # scale factor

    # Fraud penalty: each fraud funded costs 5 points
    frauds = sum(1 for o in resolved if o.was_fraud)
    score -= frauds * 5.0

    # Default rate penalty
    defaults = sum(1 for o in resolved if o.defaulted)
    default_rate = defaults / len(resolved) if resolved else 0
    if default_rate > economics.max_default_rate:
        excess = default_rate - economics.max_default_rate
        score -= excess * 100.0

    return max(0.0, min(100.0, score))


def _score_portfolio_management(state: SeasonLenderState, season_length: int) -> float:
    """Score portfolio management 0-100.
    - Capital utilization (0-40): sweet spot 60-85%
    - Concentration discipline (0-30): fraction of weeks without violations
    - Adaptive behavior (0-30): rejection rate change after defaults
    """
    score = 0.0

    # Capital utilization (0-40)
    # Sweet spot: 60-85%. Symmetric penalties outside this range:
    #   below 60%: linear decrease to 0 at 30% (penalizes idle capital)
    #   above 85%: linear decrease to 0 at 115% (penalizes over-deployment)
    # Same slope both sides so conservative inactivity isn't rewarded.
    if state.weekly_utilization:
        avg_util = sum(state.weekly_utilization) / len(state.weekly_utilization)
        if 0.60 <= avg_util <= 0.85:
            util_score = 40.0
        elif avg_util < 0.60:
            util_score = 40.0 * max(0.0, (avg_util - 0.30) / 0.30)
        else:
            util_score = 40.0 * max(0.0, 1.0 - (avg_util - 0.85) / 0.30)
        score += util_score

    # Concentration discipline (0-30)
    weeks_clean = season_length - state.weeks_with_concentration_violations
    if season_length > 0:
        score += 30.0 * (weeks_clean / season_length)

    # Adaptive behavior (0-30)
    # Simple heuristic: if lender has defaults, check if rejection rate
    # increased afterward (adaptation_score tracks this externally)
    # For now, give credit based on not having too many defaults relative to wins
    if state.deals_won > 0:
        default_ratio = sum(1 for o in state.resolved_loans if o.defaulted) / state.deals_won
        if default_ratio <= 0.15:
            score += 30.0
        elif default_ratio <= 0.30:
            score += 30.0 * (1.0 - (default_ratio - 0.15) / 0.15)
        # Above 30% default ratio: 0 adaptation score

    return max(0.0, min(100.0, score))


def _score_efficiency(state: SeasonLenderState) -> float:
    """Score efficiency 0-100.
    - Tool call efficiency (0-40): evaluations / tool_calls ratio
    - Custom tool adoption (0-30): bonus for tools used after creation
    - Speed-to-offer win rate (0-30): % of competitive deals won via speed bonus
    """
    score = 0.0

    # Tool call efficiency (0-40)
    if state.total_evaluations > 0 and state.total_tool_calls > 0:
        avg_calls = state.total_tool_calls / state.total_evaluations
        if avg_calls <= 3:
            score += 40.0
        elif avg_calls <= 6:
            score += 40.0 * (1.0 - (avg_calls - 3) / 6)
        else:
            score += max(0.0, 40.0 * (1.0 - (avg_calls - 3) / 12))
    elif state.total_evaluations > 0:
        # No tool calls tracked — give neutral score
        score += 20.0

    # Custom tool adoption (0-30)
    if state.custom_tools:
        tools_used = sum(1 for t in state.custom_tools if hasattr(t, 'times_used') and t.times_used > 0)
        total_tools = len(state.custom_tools)
        if total_tools > 0:
            score += 30.0 * (tools_used / total_tools)

    # Speed-to-offer win rate (0-30)
    if state.deals_won > 0:
        speed_rate = state.speed_wins / state.deals_won
        score += 30.0 * min(1.0, speed_rate)

    return max(0.0, min(100.0, score))


def score_season(
    lender_states: dict[str, SeasonLenderState],
    config: SeasonConfig,
) -> list[SeasonScore]:
    """Compute season scores: Credit Quality 70% + Portfolio Mgmt 20% + Efficiency 10%."""
    scores: list[SeasonScore] = []

    for state in lender_states.values():
        credit = _score_credit_quality(state, config.economics)
        portfolio = _score_portfolio_management(state, config.weeks)
        efficiency = _score_efficiency(state)
        final = credit * 0.70 + portfolio * 0.20 + efficiency * 0.10

        net_pnl = (
            state.cumulative_interest
            + state.cumulative_fees
            - state.cumulative_losses
            - state.cumulative_workout_cost
        )

        avg_util = (
            sum(state.weekly_utilization) / len(state.weekly_utilization)
            if state.weekly_utilization else 0.0
        )

        conc_discipline = (
            (config.weeks - state.weeks_with_concentration_violations) / config.weeks
            if config.weeks > 0 else 1.0
        )

        tool_efficiency = (
            state.total_tool_calls / state.total_evaluations
            if state.total_evaluations > 0 and state.total_tool_calls > 0
            else 0.0
        )

        custom_adoption = 0.0
        if state.custom_tools:
            used = sum(1 for t in state.custom_tools if hasattr(t, 'times_used') and t.times_used > 0)
            custom_adoption = used / len(state.custom_tools) if state.custom_tools else 0.0

        speed_rate = state.speed_wins / state.deals_won if state.deals_won > 0 else 0.0

        defaults_count = sum(1 for o in state.resolved_loans if o.defaulted)
        frauds_funded = sum(1 for o in state.resolved_loans if o.was_fraud)

        scores.append(SeasonScore(
            lender_id=state.lender_id,
            lender_name=state.lender_name,
            model=state.model,
            credit_quality_score=round(credit, 2),
            net_pnl=round(net_pnl, 2),
            frauds_funded=frauds_funded,
            defaults_count=defaults_count,
            portfolio_mgmt_score=round(portfolio, 2),
            avg_utilization=round(avg_util, 4),
            concentration_discipline=round(conc_discipline, 4),
            adaptation_score=round(state.adaptation_score, 2),
            efficiency_score=round(efficiency, 2),
            tool_call_efficiency=round(tool_efficiency, 2),
            custom_tool_adoption=round(custom_adoption, 4),
            speed_win_rate=round(speed_rate, 4),
            final_score=round(final, 2),
            total_deployed=round(state.deployed_capital, 2),
            total_interest=round(state.cumulative_interest, 2),
            total_losses=round(state.cumulative_losses, 2),
            deals_won=state.deals_won,
            deals_rejected=state.deals_rejected,
            deals_errored=state.deals_errored,
        ))

    return sorted(scores, key=lambda s: -s.final_score)


def print_season_report(scores: list[SeasonScore]) -> None:
    """Print a formatted season scoring report."""
    print(f"\n{'=' * 70}")
    print("  SEASON SCORING")
    print(f"{'=' * 70}")

    for s in scores:
        print(f"\n  {s.lender_name} ({s.model})")
        print(f"  {'─' * 50}")
        print(f"    Credit Quality (70%):     {s.credit_quality_score:.1f}/100")
        print(f"      Net P&L: ${s.net_pnl:,.0f} | Defaults: {s.defaults_count} | "
              f"Frauds: {s.frauds_funded}")
        print(f"    Portfolio Mgmt (20%):     {s.portfolio_mgmt_score:.1f}/100")
        print(f"      Avg Util: {s.avg_utilization:.1%} | "
              f"Concentration: {s.concentration_discipline:.1%}")
        print(f"    Efficiency (10%):         {s.efficiency_score:.1f}/100")
        print(f"      Tool Calls/Eval: {s.tool_call_efficiency:.1f} | "
              f"Custom Tools: {s.custom_tool_adoption:.0%} | "
              f"Speed Wins: {s.speed_win_rate:.0%}")
        print(f"    {'─' * 46}")
        print(f"    FINAL SCORE: {s.final_score:.2f}/100")
        print(f"    Deals: {s.deals_won} won, {s.deals_rejected} rejected | "
              f"Interest: ${s.total_interest:,.0f} | Losses: ${s.total_losses:,.0f}")

    if scores:
        print(f"\n{'*' * 70}")
        winner = scores[0]
        if len(scores) > 1 and winner.final_score == scores[1].final_score:
            print(f"  TIE!")
        else:
            print(f"  SEASON WINNER: {winner.lender_name} ({winner.model})")
            print(f"  Final Score: {winner.final_score:.2f}/100")
        print(f"{'*' * 70}")
