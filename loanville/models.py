from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Transaction:
    date: str
    description: str
    amount: float


@dataclass
class MonthlyStatement:
    month: str
    opening_balance: float
    deposits: list[Transaction]
    withdrawals: list[Transaction]
    ending_balance: float

    @property
    def total_deposits(self) -> float:
        return sum(d.amount for d in self.deposits)

    @property
    def total_withdrawals(self) -> float:
        return sum(w.amount for w in self.withdrawals)


@dataclass
class QuarterlyIncome:
    """Pre-computed quarterly income statement for LLM consumption."""
    quarter: str  # e.g. "Q1 2025"
    revenue: float
    expenses: float
    gross_profit: float
    gross_margin_pct: float
    net_income: float
    net_margin_pct: float


@dataclass
class FinancialDossier:
    company_name: str
    sector: str
    years_in_business: int
    annual_revenue: float
    annual_expenses: float
    net_income: float
    employee_count: int
    bank_statements: list[MonthlyStatement]
    quarterly_income: list[QuarterlyIncome]
    narrative: str
    loan_request_amount: float
    loan_purpose: str


@dataclass
class Borrower:
    id: str
    dossier: FinancialDossier
    # Hidden from LLMs - ground truth
    true_outcome: str  # "good", "bad", "fraud"
    months_before_default: Optional[int] = None


@dataclass
class ExistingLoan:
    borrower_name: str
    sector: str
    original_amount: float
    remaining_balance: float
    interest_rate: float
    months_remaining: int


@dataclass
class LenderConfig:
    id: str
    name: str
    persona: str
    model: str
    target_yield_pct: float
    max_single_loan: float
    total_capital: float
    sector_limits: dict  # sector -> max fraction (0.0 to 1.0)
    existing_portfolio: list[ExistingLoan] = field(default_factory=list)
    # Optional per-lender custom tool definitions (season mode).
    custom_tools: list[dict] = field(default_factory=list)


@dataclass
class TermSheet:
    loan_amount: float
    interest_rate: float  # annual percentage
    term_months: int


@dataclass
class LenderDecision:
    lender_id: str
    borrower_id: str
    decision: str  # "APPROVE" or "REJECT"
    reasoning: str
    term_sheet: Optional[TermSheet] = None


@dataclass
class BookedLoan:
    id: str
    borrower_id: str
    lender_id: str
    borrower_name: str
    sector: str
    principal: float
    interest_rate: float
    term_months: int
    true_outcome: str
    months_before_default: Optional[int] = None


@dataclass
class LoanOutcome:
    loan_id: str
    lender_id: str
    borrower_name: str
    borrower_id: str
    sector: str
    principal: float
    total_interest_paid: float
    principal_recovered: float
    principal_lost: float
    defaulted: bool
    was_fraud: bool
    months_paid: int
    # Additional economics: fees, recoveries, and non-interest costs
    total_fees_paid: float = 0.0
    recovery_amount: float = 0.0
    workout_cost: float = 0.0
    prepaid: bool = False


@dataclass
class EconomicsConfig:
    """Tunable economic parameters for RAROC scoring."""
    name: str = "balanced"
    # Opportunity cost / benchmark
    risk_free_rate: float = 0.05          # annualized
    sim_horizon_months: int = 24
    # Funding
    funding_rate: float = 0.04            # annualized cost of funds
    discount_rate_annual: float = 0.0     # optional PV discounting (0 disables)
    # Risk penalty
    risk_lambda: float = 0.35             # loss-volatility coefficient
    # Volume floor
    min_deployment_ratio: float = 0.35    # must deploy >= 35%
    volume_penalty_lambda: float = 0.8    # quadratic penalty for under-deployment
    # Hard constraints
    max_default_rate: float = 0.25        # 25% cap
    min_roe_threshold: float = -0.08      # -8% floor
    hard_constraint_base_pct: float = 5.0
    # Fraud
    fraud_penalty_rate: float = 0.25      # 25% of principal
    recovery_rate_bad: float = 0.25       # fraction of defaulted balance recovered (non-fraud)
    recovery_rate_fraud: float = 0.02     # fraction recovered on fraud defaults (chargebacks, clawbacks)
    workout_cost_rate: float = 0.03       # collections/legal as fraction of defaulted balance
    recovery_lag_months: int = 6          # affects PV only (if discount_rate_annual > 0)
    # Concentration
    concentration_penalty_rate: float = 0.05  # 5% of excess
    # Fees & servicing
    origination_fee_rate: float = 0.01        # one-time fee on principal (revenue)
    servicing_cost_rate_annual: float = 0.003 # annual cost on outstanding balance (expense)
    prepayment_rate_annual: float = 0.0       # annualized prepay hazard for good loans (0 disables)
    prepayment_penalty_rate: float = 0.0      # fraction of remaining balance if prepaid
    # Origination ops / underwriting costs
    underwriting_cost_per_application_usd: float = 20.0
    doc_request_cost_usd: float = 50.0
    # Borrower friction / abandonment (applies at deal booking time)
    abandonment_base_rate: float = 0.0
    abandonment_per_doc_request: float = 0.02
    abandonment_per_second_latency: float = 0.0
    abandonment_cap: float = 0.30


ECONOMICS_PRESETS: dict[str, "EconomicsConfig"] = {
    "balanced": EconomicsConfig(name="balanced"),
    "aggressive": EconomicsConfig(
        name="aggressive",
        risk_free_rate=0.03,
        funding_rate=0.025,
        risk_lambda=0.2,
        min_deployment_ratio=0.50,
        volume_penalty_lambda=1.5,
        max_default_rate=0.40,
        min_roe_threshold=-0.15,
        fraud_penalty_rate=0.15,
        recovery_rate_bad=0.18,
        recovery_rate_fraud=0.01,
        workout_cost_rate=0.04,
        origination_fee_rate=0.008,
        servicing_cost_rate_annual=0.0025,
        underwriting_cost_per_application_usd=15.0,
        doc_request_cost_usd=35.0,
    ),
    "conservative": EconomicsConfig(
        name="conservative",
        risk_free_rate=0.07,
        funding_rate=0.06,
        risk_lambda=0.8,
        min_deployment_ratio=0.20,
        volume_penalty_lambda=0.3,
        max_default_rate=0.15,
        min_roe_threshold=-0.05,
        fraud_penalty_rate=0.40,
        concentration_penalty_rate=0.10,
        recovery_rate_bad=0.35,
        recovery_rate_fraud=0.03,
        workout_cost_rate=0.02,
        origination_fee_rate=0.0125,
        servicing_cost_rate_annual=0.0035,
        underwriting_cost_per_application_usd=30.0,
        doc_request_cost_usd=60.0,
        abandonment_per_doc_request=0.03,
    ),
}


@dataclass
class ActiveLoan:
    """A booked loan tracked across season weeks."""
    loan_id: str
    borrower_id: str
    borrower_name: str
    lender_id: str
    sector: str
    principal: float
    interest_rate: float
    term_months: int
    true_outcome: str
    months_before_default: Optional[int]
    booked_week: int
    months_elapsed: int = 0
    total_interest_collected: float = 0.0
    total_principal_repaid: float = 0.0
    total_fees_collected: float = 0.0
    remaining_balance: float = 0.0  # initialized to principal
    status: str = "performing"  # performing | defaulted | repaid | prepaid


@dataclass
class SeasonConfig:
    weeks: int = 10
    cohort_size: int = 5
    months_per_week: int = 2
    season_mix: str = "realistic"  # gentle | realistic | adversarial | stress | escalating
    seed: int = 42
    speed_scoring: bool = True
    custom_tools: bool = True
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)

    # Capital adequacy elimination — lenders below this fraction of initial
    # capital are eliminated from the season (inspired by Skirmish's spawn
    # destruction and real banking regulation).  0.0 = disabled.
    capital_adequacy_ratio: float = 0.20

    # Capital time-value decay — undeployed capital loses this fraction of
    # value per week, creating Skirmish-style tension between "deploy early"
    # and "wait for better opportunities."  0.0 = disabled.
    capital_decay_rate: float = 0.002

    # Information asymmetry — controls whether lenders see identical or
    # different subsets of borrower data.  "full" = all see everything.
    # "partial_statements" = each lender sees a random subset of bank
    # statement months.  "redacted" = some financial fields are hidden
    # per-lender.
    info_asymmetry: str = "none"  # none | partial_statements | redacted

    # Strategic pipeline pressure (post-v1 roadmap item): optional phase-based
    # arrival and a weekly cap on full deep-underwrite capacity.
    arrival_phases: int = 1
    deep_uw_slots_per_week: int = 0  # 0 disables cap (unlimited)

    def __post_init__(self) -> None:
        if self.weeks <= 0:
            raise ValueError("weeks must be > 0")
        if self.cohort_size <= 0:
            raise ValueError("cohort_size must be > 0")
        if self.months_per_week <= 0:
            raise ValueError("months_per_week must be > 0")
        valid_mixes = ("gentle", "realistic", "adversarial", "stress", "escalating")
        if self.season_mix not in valid_mixes:
            raise ValueError(
                f"season_mix must be one of {valid_mixes}, got '{self.season_mix}'"
            )
        valid_asymmetry = ("none", "partial_statements", "redacted")
        if self.info_asymmetry not in valid_asymmetry:
            raise ValueError(
                f"info_asymmetry must be one of {valid_asymmetry}, got '{self.info_asymmetry}'"
            )
        if self.arrival_phases <= 0:
            raise ValueError("arrival_phases must be > 0")
        if self.deep_uw_slots_per_week < 0:
            raise ValueError("deep_uw_slots_per_week must be >= 0")


@dataclass
class SeasonLenderState:
    lender_id: str
    lender_name: str
    model: str
    total_capital: float
    deployed_capital: float = 0.0
    available_capital: float = 0.0
    active_loans: list[ActiveLoan] = field(default_factory=list)
    resolved_loans: list[LoanOutcome] = field(default_factory=list)
    sector_exposure: dict[str, float] = field(default_factory=dict)
    cumulative_interest: float = 0.0
    cumulative_losses: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_workout_cost: float = 0.0
    total_tool_calls: int = 0
    total_evaluations: int = 0
    deals_won: int = 0
    deals_lost: int = 0
    deals_rejected: int = 0
    deals_errored: int = 0
    # Portfolio management tracking
    weekly_utilization: list[float] = field(default_factory=list)
    weeks_with_concentration_violations: int = 0
    adaptation_score: float = 0.0
    # Speed tracking
    speed_wins: int = 0
    # Bandwidth / pipeline tracking
    deep_uw_deferred: int = 0
    weekly_deep_uw_used: list[int] = field(default_factory=list)
    # Custom tools
    custom_tools: list = field(default_factory=list)
    # Weekly snapshots for analytics
    weekly_snapshots: list[dict] = field(default_factory=list)
    # Capital adequacy elimination tracking
    eliminated: bool = False
    eliminated_week: int = 0
    # Capital time-value decay accumulator
    cumulative_decay: float = 0.0
    # Token / cost tracking (from LOS traces)
    cumulative_tokens_in: int = 0
    cumulative_tokens_out: int = 0
    cumulative_cost_usd: float = 0.0


@dataclass
class WeekResult:
    week: int
    cohort_size: int
    loans_booked: int
    defaults_this_week: int
    repayments_this_week: int
    events: list[str] = field(default_factory=list)


@dataclass
class SeasonScore:
    lender_id: str
    lender_name: str
    model: str
    # Credit Quality (70%)
    credit_quality_score: float
    net_pnl: float
    frauds_funded: int
    defaults_count: int
    # Portfolio Management (20%)
    portfolio_mgmt_score: float
    avg_utilization: float
    concentration_discipline: float
    adaptation_score: float
    # Efficiency (10%)
    efficiency_score: float
    tool_call_efficiency: float
    custom_tool_adoption: float
    speed_win_rate: float
    # Final
    final_score: float
    # Raw totals
    total_deployed: float
    total_interest: float
    total_losses: float
    deals_won: int
    deals_rejected: int
    deals_errored: int = 0


@dataclass
class ResolveResult:
    """Result of advancing a loan by N months."""
    interest: float
    principal_repaid: float
    remaining_balance: float
    principal_lost: float
    recovery_amount: float
    workout_cost: float
    fees: float
    defaulted: bool
    matured: bool
    prepaid: bool
    months_actually_advanced: int


@dataclass
class LenderScore:
    lender_id: str
    lender_name: str
    model: str
    total_deployed: float
    total_interest_earned: float
    total_principal_lost: float
    net_return: float
    roi_pct: float
    deals_won: int
    deals_lost: int
    deals_rejected: int
    frauds_funded: int
    defaults_count: int
    concentration_violations: list[str]
    concentration_penalty_pct: float
    fraud_penalty_pct: float
    final_adjusted_score: float
    deals_errored: int = 0
    perfect_score: float = 0.0  # Theoretical max if lender had perfect foresight
    # --- RAROC / funding cost ---
    funding_cost: float = 0.0           # Cost of funds on deployed capital
    loss_volatility: float = 0.0        # Std dev of per-loan profits (dollars)
    risk_penalty_pct: float = 0.0       # Volatility penalty as % of available capital
    raroc_score: float = 0.0            # RAROC-adjusted final score
    # --- Hard constraints ---
    default_rate: float = 0.0           # Fraction of funded deals that defaulted
    roe_pct: float = 0.0                # Return on deployed capital (%)
    hard_constraint_violations: list[str] = field(default_factory=list)
    hard_constraint_penalty_pct: float = 0.0
    # --- Diagnostics ---
    approval_rate: float = 0.0          # Fraction of applications approved
    deployment_ratio: float = 0.0       # Fraction of available capital deployed
    volume_penalty_pct: float = 0.0     # Penalty for under-deployment (% of capital)
    # --- Extended economics breakdown ---
    total_fees_earned: float = 0.0
    total_workout_cost: float = 0.0
    total_servicing_cost: float = 0.0
    underwriting_cost: float = 0.0
    doc_request_cost: float = 0.0
    llm_cost: float = 0.0
    ops_cost: float = 0.0
    adjusted_pnl_dollars: float = 0.0
