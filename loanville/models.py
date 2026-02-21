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
    sector: str
    principal: float
    total_interest_paid: float
    principal_recovered: float
    principal_lost: float
    defaulted: bool
    was_fraud: bool
    months_paid: int


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
