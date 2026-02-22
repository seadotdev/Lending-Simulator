"""
Underwriting Run schema — the single contract that unifies
Benchmark, Simulator+Elo, and LOS into one flywheel.

Every underwriting evaluation produces a Run artifact.
- Benchmark produces case + labels.gold
- LOS produces inputs/trace/decision
- Simulator consumes decision + policy, writes back labels.outcome + scores

Runs are append-only, JSON-serializable, and replayable.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    Borrower,
    FinancialDossier,
    LenderConfig,
    LenderDecision,
    LoanOutcome,
    TermSheet,
)


# ---------------------------------------------------------------------------
# Run sub-schemas
# ---------------------------------------------------------------------------

@dataclass
class RunCase:
    """What's being underwritten."""
    case_id: str
    source: str  # "benchmark" | "simulator" | "production"
    segment: str  # e.g. "smb_term_loan"
    jurisdiction: str = "US"
    currency: str = "USD"
    requested_amount: float = 0.0
    requested_tenor_months: int = 24
    requested_purpose: str = ""

    @classmethod
    def from_borrower(cls, borrower: Borrower, source: str = "simulator") -> RunCase:
        d = borrower.dossier
        return cls(
            case_id=borrower.id,
            source=source,
            segment=f"{d.sector.lower().replace(' ', '_')}_term_loan",
            requested_amount=d.loan_request_amount,
            requested_tenor_months=24,
            requested_purpose=d.loan_purpose,
        )


@dataclass
class RunPolicy:
    """The underwriting policy/config used."""
    policy_id: str
    model: str
    prompt_hash: str = ""
    tools_version: str = ""
    params: dict = field(default_factory=dict)

    @classmethod
    def from_lender(cls, lender: LenderConfig, policy_id: str = "") -> RunPolicy:
        pid = policy_id or f"p_{lender.id}_{lender.model.replace('/', '_')}"
        return cls(
            policy_id=pid,
            model=lender.model,
            params={
                "target_yield_pct": lender.target_yield_pct,
                "max_single_loan": lender.max_single_loan,
                "total_capital": lender.total_capital,
                "sector_limits": lender.sector_limits,
                "persona": lender.persona,
            },
        )


@dataclass
class ExtractedFinancials:
    """Key financials extracted from documents."""
    revenue_ttm: float = 0.0
    gross_margin: float = 0.0
    ebitda_ttm: float = 0.0
    net_income: float = 0.0
    annual_expenses: float = 0.0


@dataclass
class ExtractedBanking:
    """Key banking metrics extracted."""
    avg_daily_balance_90d: float = 0.0
    nsf_12m: int = 0
    total_deposits_12m: float = 0.0
    total_withdrawals_12m: float = 0.0


@dataclass
class ExtractedBusiness:
    """Key business attributes."""
    industry: str = ""
    years_trading: int = 0
    employee_count: int = 0
    company_name: str = ""


@dataclass
class RunInputs:
    """What the underwriter saw."""
    raw_documents: list[dict] = field(default_factory=list)  # [{doc_id, type, bytes_sha}]
    financials: ExtractedFinancials = field(default_factory=ExtractedFinancials)
    banking: ExtractedBanking = field(default_factory=ExtractedBanking)
    business: ExtractedBusiness = field(default_factory=ExtractedBusiness)
    missing_info: list[str] = field(default_factory=list)

    @classmethod
    def from_dossier(cls, d: FinancialDossier) -> RunInputs:
        # Compute banking aggregates from statements
        total_deps = sum(s.total_deposits for s in d.bank_statements)
        total_wds = sum(s.total_withdrawals for s in d.bank_statements)
        # Average balance from last 3 months (90d proxy)
        last_3 = d.bank_statements[-3:] if len(d.bank_statements) >= 3 else d.bank_statements
        avg_bal = sum(s.ending_balance for s in last_3) / len(last_3) if last_3 else 0.0

        return cls(
            raw_documents=[
                {"doc_id": "bank_statements_12m", "type": "bank_statement",
                 "bytes_sha": hashlib.sha256(str(total_deps).encode()).hexdigest()[:16]},
                {"doc_id": "quarterly_income", "type": "pnl",
                 "bytes_sha": hashlib.sha256(str(d.annual_revenue).encode()).hexdigest()[:16]},
            ],
            financials=ExtractedFinancials(
                revenue_ttm=d.annual_revenue,
                gross_margin=(d.annual_revenue - d.annual_expenses) / d.annual_revenue
                if d.annual_revenue > 0 else 0.0,
                ebitda_ttm=d.net_income,  # simplified: net_income as EBITDA proxy
                net_income=d.net_income,
                annual_expenses=d.annual_expenses,
            ),
            banking=ExtractedBanking(
                avg_daily_balance_90d=round(avg_bal, 2),
                total_deposits_12m=round(total_deps, 2),
                total_withdrawals_12m=round(total_wds, 2),
            ),
            business=ExtractedBusiness(
                industry=d.sector,
                years_trading=d.years_in_business,
                employee_count=d.employee_count,
                company_name=d.company_name,
            ),
        )


@dataclass
class TraceStep:
    """A single step in the underwriting trace."""
    t: str  # ISO timestamp
    type: str  # "tool_call" | "note" | "reasoning" | "doc_request"
    name: str = ""
    args: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    content: str = ""


@dataclass
class TraceCost:
    """Token/cost metrics for the trace."""
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class RunTrace:
    """Full execution trace of the underwriting run."""
    steps: list[TraceStep] = field(default_factory=list)
    latency_ms: int = 0
    cost: TraceCost = field(default_factory=TraceCost)


@dataclass
class DecisionTerms:
    """Proposed loan terms."""
    amount: float = 0.0
    apr: float = 0.0
    tenor_months: int = 0
    fees: dict = field(default_factory=dict)


@dataclass
class DecisionRationale:
    """Structured rationale for the decision."""
    summary: str = ""
    key_factors: list[str] = field(default_factory=list)
    what_would_change: list[str] = field(default_factory=list)


@dataclass
class RunDecision:
    """The underwriting decision."""
    action: str  # "approve" | "decline" | "counter" | "refer"
    risk_grade: str = ""
    prob_default_12m: float = 0.0
    terms: DecisionTerms = field(default_factory=DecisionTerms)
    conditions: list[str] = field(default_factory=list)
    covenants: list[str] = field(default_factory=list)
    rationale: DecisionRationale = field(default_factory=DecisionRationale)
    confidence: float = 0.0

    @classmethod
    def from_lender_decision(cls, d: LenderDecision) -> RunDecision:
        action = "approve" if d.decision == "APPROVE" else "decline"
        terms = DecisionTerms()
        if d.term_sheet:
            terms = DecisionTerms(
                amount=d.term_sheet.loan_amount,
                apr=d.term_sheet.interest_rate / 100.0,
                tenor_months=d.term_sheet.term_months,
            )
        return cls(
            action=action,
            terms=terms,
            rationale=DecisionRationale(summary=d.reasoning or ""),
        )


@dataclass
class RunLabels:
    """Ground truth labels — filled by benchmark or by real outcome data."""
    available: bool = False
    gold: Optional[dict] = None       # Benchmark gold label
    outcome: Optional[dict] = None    # Real or simulated outcome


@dataclass
class RunScores:
    """Scorecard results — filled by the evaluation harness."""
    gates: Optional[dict] = None
    uw_quality: Optional[dict] = None
    business: Optional[dict] = None
    overall: Optional[dict] = None


# ---------------------------------------------------------------------------
# The Run itself
# ---------------------------------------------------------------------------

@dataclass
class UnderwritingRun:
    """The single artifact that all three flywheel components speak.

    - Benchmark produces: case + labels.gold
    - LOS produces: inputs + trace + decision
    - Simulator consumes: decision + policy → writes labels.outcome + scores
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    case: RunCase = field(default_factory=lambda: RunCase(case_id="", source="", segment=""))
    policy: RunPolicy = field(default_factory=lambda: RunPolicy(policy_id="", model=""))
    inputs: RunInputs = field(default_factory=RunInputs)
    trace: RunTrace = field(default_factory=RunTrace)
    decision: RunDecision = field(
        default_factory=lambda: RunDecision(action="pending")
    )
    labels: RunLabels = field(default_factory=RunLabels)
    scores: RunScores = field(default_factory=RunScores)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> UnderwritingRun:
        """Deserialize from a plain dict."""
        run = cls()
        run.run_id = data.get("run_id", run.run_id)
        run.timestamp_utc = data.get("timestamp_utc", run.timestamp_utc)

        if "case" in data:
            c = data["case"]
            run.case = RunCase(**{k: v for k, v in c.items()
                                  if k in RunCase.__dataclass_fields__})
        if "policy" in data:
            p = data["policy"]
            run.policy = RunPolicy(**{k: v for k, v in p.items()
                                      if k in RunPolicy.__dataclass_fields__})
        if "decision" in data:
            d = data["decision"]
            run.decision = RunDecision(
                action=d.get("action", "pending"),
                risk_grade=d.get("risk_grade", ""),
                prob_default_12m=d.get("prob_default_12m", 0.0),
                conditions=d.get("conditions", []),
                covenants=d.get("covenants", []),
                confidence=d.get("confidence", 0.0),
            )
            if "terms" in d:
                run.decision.terms = DecisionTerms(**{
                    k: v for k, v in d["terms"].items()
                    if k in DecisionTerms.__dataclass_fields__
                })
            if "rationale" in d:
                run.decision.rationale = DecisionRationale(**{
                    k: v for k, v in d["rationale"].items()
                    if k in DecisionRationale.__dataclass_fields__
                })
        if "labels" in data:
            lb = data["labels"]
            run.labels = RunLabels(
                available=lb.get("available", False),
                gold=lb.get("gold"),
                outcome=lb.get("outcome"),
            )
        if "scores" in data:
            sc = data["scores"]
            run.scores = RunScores(
                gates=sc.get("gates"),
                uw_quality=sc.get("uw_quality"),
                business=sc.get("business"),
                overall=sc.get("overall"),
            )
        return run

    @classmethod
    def from_json(cls, json_str: str) -> UnderwritingRun:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Factory: build a Run from existing Loanville objects
# ---------------------------------------------------------------------------

def build_run(
    borrower: Borrower,
    lender: LenderConfig,
    decision: LenderDecision,
    outcome: Optional[LoanOutcome] = None,
    trace_steps: Optional[list[dict]] = None,
    token_usage: Optional[dict] = None,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    source: str = "simulator",
    policy_id: str = "",
) -> UnderwritingRun:
    """Build an UnderwritingRun from existing Loanville simulation objects.

    This is the primary bridge between the current codebase and the flywheel
    contract. Every simulation evaluation can call this to emit a Run.
    """
    run = UnderwritingRun()
    run.case = RunCase.from_borrower(borrower, source=source)
    run.policy = RunPolicy.from_lender(lender, policy_id=policy_id)
    run.inputs = RunInputs.from_dossier(borrower.dossier)
    run.decision = RunDecision.from_lender_decision(decision)

    # Trace
    if trace_steps:
        run.trace.steps = [
            TraceStep(
                t=s.get("t", run.timestamp_utc),
                type=s.get("type", "tool_call"),
                name=s.get("name", ""),
                args=s.get("args", {}),
                result=s.get("result", {}),
                content=s.get("content", ""),
            )
            for s in trace_steps
        ]
    run.trace.latency_ms = latency_ms
    if token_usage:
        run.trace.cost = TraceCost(
            tokens_in=token_usage.get("prompt", 0),
            tokens_out=token_usage.get("completion", 0),
            estimated_cost_usd=cost_usd,
        )

    # Labels: ground truth from borrower (hidden in production)
    if borrower.true_outcome:
        run.labels = RunLabels(
            available=True,
            gold={
                "true_outcome": borrower.true_outcome,
                "months_before_default": borrower.months_before_default,
                "correct_action": "decline" if borrower.true_outcome in ("bad", "fraud") else "approve",
            },
        )
        if outcome:
            run.labels.outcome = {
                "defaulted": outcome.defaulted,
                "was_fraud": outcome.was_fraud,
                "months_paid": outcome.months_paid,
                "interest_paid": outcome.total_interest_paid,
                "principal_lost": outcome.principal_lost,
                "principal_recovered": outcome.principal_recovered,
            }

    return run
