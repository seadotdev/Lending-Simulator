"""
Scorecard — scores an UnderwritingRun on three layers.

Layer A: Hard gates (must-pass)
Layer B: Underwriting quality (benchmark-driven)
Layer C: Business outcomes (sim-driven)

The scorecard is the single evaluation contract consumed by both
the benchmark harness and the Elo tournament.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .contracts import MAX_APR_DECIMAL
from .models import EconomicsConfig
from .run_schema import RunScores, UnderwritingRun
from .scoring import (
    FUNDING_RATE,
    MAX_DEFAULT_RATE,
    MIN_DEPLOYMENT_RATIO,
    MIN_ROE_THRESHOLD,
    RISK_FREE_RATE,
    SIM_HORIZON_MONTHS,
    compute_loan_payoff,
)


# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

GATE_DEFINITIONS = [
    # (gate_id, description)
    ("compliance_action", "Decision is a valid action type"),
    ("sanity_apr", "APR within allowable bounds (0-55%)"),
    ("sanity_tenor", "Tenor within allowable bounds (1-360 months)"),
    ("sanity_amount", "Loan amount positive and within max limits"),
    ("trace_supports_decision", "Trace/rationale consistent with decision"),
    ("required_docs", "Required documents requested when missing"),
]


@dataclass
class GateResult:
    """Result of a single gate check."""
    gate_id: str
    passed: bool
    detail: str = ""


@dataclass
class GatesScore:
    """Layer A: hard gates."""
    passed: bool = True
    results: list[GateResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "results": [
                {"gate_id": r.gate_id, "passed": r.passed, "detail": r.detail}
                for r in self.results
            ],
        }


@dataclass
class UWQualityScore:
    """Layer B: underwriting quality (benchmark-driven)."""
    decision_correct: bool = False
    decision_acc: float = 0.0  # 1.0 if correct, 0.0 if wrong
    terms_mae: dict = field(default_factory=dict)  # {apr: float, amount: float}
    calibration: dict = field(default_factory=dict)  # {brier: float} if PD labels exist
    explainability: float = 0.0  # rubric score 0-1

    def to_dict(self) -> dict:
        return {
            "decision_correct": self.decision_correct,
            "decision_acc": self.decision_acc,
            "terms_mae": self.terms_mae,
            "calibration": self.calibration,
            "explainability": self.explainability,
        }


@dataclass
class BusinessScore:
    """Layer C: business outcomes (sim-driven)."""
    exp_profit: float = 0.0
    loss_rate: float = 0.0
    approval_rate: float = 0.0
    ops: dict = field(default_factory=dict)  # {latency_ms, doc_requests, tokens}

    def to_dict(self) -> dict:
        return {
            "exp_profit": round(self.exp_profit, 2),
            "loss_rate": round(self.loss_rate, 4),
            "approval_rate": round(self.approval_rate, 4),
            "ops": self.ops,
        }


@dataclass
class Scorecard:
    """Complete scorecard for an UnderwritingRun."""
    run_id: str = ""
    gates: GatesScore = field(default_factory=GatesScore)
    uw_quality: UWQualityScore = field(default_factory=UWQualityScore)
    business: BusinessScore = field(default_factory=BusinessScore)
    overall_score: float = 0.0
    weights: dict = field(default_factory=lambda: {
        "gates": "hard",
        "uw": 0.55,
        "biz": 0.35,
        "ops": 0.10,
    })

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "gates": self.gates.to_dict(),
            "uw_quality": self.uw_quality.to_dict(),
            "business": self.business.to_dict(),
            "overall": {
                "score": round(self.overall_score, 2),
                "weights": self.weights,
            },
        }


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def _check_compliance_action(run: UnderwritingRun) -> GateResult:
    valid = {"approve", "decline", "counter", "refer", "pending"}
    ok = run.decision.action in valid
    return GateResult(
        gate_id="compliance_action",
        passed=ok,
        detail=f"action='{run.decision.action}'" if not ok else "",
    )


def _check_sanity_apr(run: UnderwritingRun) -> GateResult:
    apr = run.decision.terms.apr
    # APR stored as decimal in Run schema (0.168 = 16.8%)
    # but also tolerate percentage-scale values (16.8)
    if apr > 1.0:
        # Likely percentage scale
        apr_pct = apr
    else:
        apr_pct = apr * 100

    if run.decision.action == "decline":
        return GateResult(gate_id="sanity_apr", passed=True, detail="declined, no terms")

    ok = 0.0 <= apr_pct <= MAX_APR_DECIMAL * 100
    return GateResult(
        gate_id="sanity_apr",
        passed=ok,
        detail=f"apr={apr_pct:.2f}%" if not ok else "",
    )


def _check_sanity_tenor(run: UnderwritingRun) -> GateResult:
    if run.decision.action == "decline":
        return GateResult(gate_id="sanity_tenor", passed=True, detail="declined, no terms")

    tenor = run.decision.terms.tenor_months
    ok = 1 <= tenor <= 360
    return GateResult(
        gate_id="sanity_tenor",
        passed=ok,
        detail=f"tenor={tenor}" if not ok else "",
    )


def _check_sanity_amount(run: UnderwritingRun) -> GateResult:
    if run.decision.action == "decline":
        return GateResult(gate_id="sanity_amount", passed=True, detail="declined, no terms")

    amt = run.decision.terms.amount
    ok = amt > 0
    return GateResult(
        gate_id="sanity_amount",
        passed=ok,
        detail=f"amount={amt}" if not ok else "",
    )


def _check_trace_supports_decision(run: UnderwritingRun) -> GateResult:
    """Basic check: if approved, rationale should exist; if declined, should have reason."""
    has_rationale = bool(
        run.decision.rationale.summary
        or run.trace.steps
    )
    ok = has_rationale
    return GateResult(
        gate_id="trace_supports_decision",
        passed=ok,
        detail="" if ok else "no rationale or trace steps",
    )


def _check_required_docs(run: UnderwritingRun) -> GateResult:
    """If inputs.missing_info is non-empty, trace should reference doc requests."""
    missing = run.inputs.missing_info
    if not missing:
        return GateResult(gate_id="required_docs", passed=True)

    # Check if any trace step mentions the missing info
    trace_text = " ".join(
        s.content + s.name + str(s.args) for s in run.trace.steps
    ).lower()
    requested = any(m.lower() in trace_text for m in missing)

    # Also pass if decision is decline (declined due to missing docs is fine)
    ok = requested or run.decision.action == "decline"
    return GateResult(
        gate_id="required_docs",
        passed=ok,
        detail="" if ok else f"missing {missing} not requested",
    )


def check_gates(run: UnderwritingRun) -> GatesScore:
    """Run all hard gates on a Run."""
    checks = [
        _check_compliance_action(run),
        _check_sanity_apr(run),
        _check_sanity_tenor(run),
        _check_sanity_amount(run),
        _check_trace_supports_decision(run),
        _check_required_docs(run),
    ]
    failures = [c.gate_id for c in checks if not c.passed]
    return GatesScore(
        passed=len(failures) == 0,
        results=checks,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# UW Quality scoring
# ---------------------------------------------------------------------------

def score_uw_quality(
    run: UnderwritingRun,
    gold: Optional[dict] = None,
) -> UWQualityScore:
    """Score underwriting quality against gold labels.

    Gold label format:
        {
            "correct_action": "approve" | "decline",
            "true_outcome": "good" | "bad" | "fraud",
            "terms": {"apr_range": [lo, hi], "max_amount": N},  # optional
        }
    """
    gold = gold or (run.labels.gold if run.labels.available else None)
    if not gold:
        return UWQualityScore()

    # Decision accuracy
    correct_action = gold.get("correct_action", "")
    decision_correct = run.decision.action == correct_action
    decision_acc = 1.0 if decision_correct else 0.0

    # Terms MAE (only if both approved and gold has terms)
    terms_mae = {}
    gold_terms = gold.get("terms", {})
    if run.decision.action == "approve" and gold_terms:
        apr_range = gold_terms.get("apr_range")
        if apr_range and run.decision.terms.apr > 0:
            apr_val = run.decision.terms.apr
            if apr_val < 1.0:
                apr_val *= 100  # normalize to percentage
            apr_mid = (apr_range[0] + apr_range[1]) / 2
            terms_mae["apr"] = abs(apr_val - apr_mid)

        max_amount = gold_terms.get("max_amount")
        if max_amount and run.decision.terms.amount > 0:
            terms_mae["amount"] = abs(run.decision.terms.amount - max_amount)

    # Calibration (Brier score if PD label exists)
    calibration = {}
    if "prob_default_actual" in gold and run.decision.prob_default_12m > 0:
        actual = gold["prob_default_actual"]
        pred = run.decision.prob_default_12m
        calibration["brier"] = (pred - actual) ** 2

    # Explainability (basic rubric: has summary + factors + what_would_change)
    expl_score = 0.0
    if run.decision.rationale.summary:
        expl_score += 0.4
    if run.decision.rationale.key_factors:
        expl_score += 0.3
    if run.decision.rationale.what_would_change:
        expl_score += 0.3

    return UWQualityScore(
        decision_correct=decision_correct,
        decision_acc=decision_acc,
        terms_mae=terms_mae,
        calibration=calibration,
        explainability=expl_score,
    )


# ---------------------------------------------------------------------------
# Business outcome scoring
# ---------------------------------------------------------------------------

def score_business(
    run: UnderwritingRun,
    outcome: Optional[dict] = None,
) -> BusinessScore:
    """Score business outcomes from simulation or real data.

    Uses the Run's decision + outcome labels to compute expected profit.
    """
    outcome = outcome or (run.labels.outcome if run.labels.available else None)

    ops = {
        "latency_ms": run.trace.latency_ms,
        "tokens_in": run.trace.cost.tokens_in,
        "tokens_out": run.trace.cost.tokens_out,
        "doc_requests": sum(1 for s in run.trace.steps if s.type == "doc_request"),
    }

    if run.decision.action == "decline":
        # Declined — no profit, no loss
        return BusinessScore(
            exp_profit=0.0,
            loss_rate=0.0,
            approval_rate=0.0,
            ops=ops,
        )

    # Approved — compute expected profit from outcome
    exp_profit = 0.0
    loss_rate = 0.0

    if outcome:
        interest_paid = outcome.get("interest_paid", 0.0)
        fees_paid = outcome.get("fees_paid", 0.0)
        principal_lost = outcome.get("principal_lost", 0.0)
        workout_cost = outcome.get("workout_cost", 0.0)
        exp_profit = interest_paid + fees_paid - principal_lost - workout_cost
        if run.decision.terms.amount > 0:
            loss_rate = principal_lost / run.decision.terms.amount
    elif run.labels.gold:
        # Estimate from gold labels using payoff model
        true_outcome = run.labels.gold.get("true_outcome", "good")
        months_default = run.labels.gold.get("months_before_default")

        apr_pct = run.decision.terms.apr
        if apr_pct < 1.0:
            apr_pct *= 100  # normalize

        result = compute_loan_payoff(
            principal=run.decision.terms.amount,
            interest_rate=apr_pct,
            term_months=run.decision.terms.tenor_months or SIM_HORIZON_MONTHS,
            true_outcome=true_outcome,
            months_before_default=months_default,
        )
        exp_profit = result["net_profit"]
        if run.decision.terms.amount > 0:
            loss_rate = result["principal_lost"] / run.decision.terms.amount

    return BusinessScore(
        exp_profit=exp_profit,
        loss_rate=loss_rate,
        approval_rate=1.0,  # This run approved
        ops=ops,
    )


# ---------------------------------------------------------------------------
# Full scorecard
# ---------------------------------------------------------------------------

def score_run(
    run: UnderwritingRun,
    gold: Optional[dict] = None,
    outcome: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> Scorecard:
    """Produce a complete Scorecard for a single UnderwritingRun.

    This is the primary evaluation function for the flywheel.
    """
    w = weights or {"gates": "hard", "uw": 0.55, "biz": 0.35, "ops": 0.10}

    gates = check_gates(run)
    uw = score_uw_quality(run, gold)
    biz = score_business(run, outcome)

    # Compute overall score
    # UW component: decision_acc * 0.6 + explainability * 0.4
    uw_component = uw.decision_acc * 0.6 + uw.explainability * 0.4

    # Business component: normalized profit (positive = good)
    # Normalize: $10K profit → 1.0, $0 → 0.5, -$10K → 0.0
    profit_norm = max(0.0, min(1.0, (biz.exp_profit + 10000) / 20000))
    loss_penalty = 1.0 - biz.loss_rate
    biz_component = profit_norm * 0.6 + loss_penalty * 0.4

    # Ops component: faster = better (normalize latency)
    # 60s → 1.0, 300s → 0.0
    latency_s = biz.ops.get("latency_ms", 0) / 1000
    ops_component = max(0.0, min(1.0, 1.0 - (latency_s - 60) / 240))

    # Weighted overall (0-100 scale)
    if gates.passed:
        overall = (
            w.get("uw", 0.55) * uw_component +
            w.get("biz", 0.35) * biz_component +
            w.get("ops", 0.10) * ops_component
        ) * 100
    else:
        # Failed gates → capped at 25
        overall = min(25.0, (
            w.get("uw", 0.55) * uw_component +
            w.get("biz", 0.35) * biz_component +
            w.get("ops", 0.10) * ops_component
        ) * 100 * 0.25)

    card = Scorecard(
        run_id=run.run_id,
        gates=gates,
        uw_quality=uw,
        business=biz,
        overall_score=overall,
        weights=w,
    )

    # Write scores back to the run
    d = card.to_dict()
    run.scores = RunScores(
        gates=d.get("gates"),
        uw_quality=d.get("uw_quality"),
        business=d.get("business"),
        overall=d.get("overall"),
    )

    return card


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_runs(
    runs: list[UnderwritingRun],
    weights: Optional[dict] = None,
) -> list[Scorecard]:
    """Score a batch of runs. Returns scorecards sorted by overall score desc."""
    cards = [score_run(r, weights=weights) for r in runs]
    cards.sort(key=lambda c: c.overall_score, reverse=True)
    return cards


def print_scorecard(card: Scorecard) -> None:
    """Pretty-print a scorecard to stdout."""
    print(f"\n{'─' * 60}")
    print(f"  SCORECARD — Run {card.run_id[:12]}...")
    print(f"{'─' * 60}")

    # Gates
    gate_status = "PASS" if card.gates.passed else "FAIL"
    print(f"  A) Hard Gates: [{gate_status}]")
    if card.gates.failures:
        for f in card.gates.failures:
            print(f"     !! FAILED: {f}")

    # UW Quality
    print(f"  B) UW Quality:")
    correct_str = "correct" if card.uw_quality.decision_correct else "WRONG"
    print(f"     Decision: {correct_str} (acc={card.uw_quality.decision_acc:.0f})")
    if card.uw_quality.terms_mae:
        for k, v in card.uw_quality.terms_mae.items():
            print(f"     Terms MAE {k}: {v:.2f}")
    if card.uw_quality.calibration:
        for k, v in card.uw_quality.calibration.items():
            print(f"     Calibration {k}: {v:.4f}")
    print(f"     Explainability: {card.uw_quality.explainability:.2f}")

    # Business
    print(f"  C) Business Outcomes:")
    print(f"     Expected Profit: ${card.business.exp_profit:,.2f}")
    print(f"     Loss Rate: {card.business.loss_rate:.2%}")
    if card.business.ops:
        lat = card.business.ops.get("latency_ms", 0)
        print(f"     Latency: {lat}ms")

    # Overall
    print(f"  {'=' * 40}")
    print(f"  OVERALL SCORE: {card.overall_score:.1f}/100")
    print(f"{'─' * 60}")
