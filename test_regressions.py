from loanville.data import get_borrowers, get_lenders
from loanville.engine import SimulationEngine
from loanville.los_adapter import run_to_lender_decision
from loanville.models import BookedLoan, LenderDecision
from loanville.run_schema import DecisionTerms, RunDecision, RunScores, TraceStep, UnderwritingRun, build_run
from loanville.scorecard import score_run


def test_underwriting_run_roundtrip_preserves_inputs_and_trace() -> None:
    borrower = get_borrowers("easy", seed=1, sample_size=1)[0]
    lender = get_lenders()[0]
    decision = LenderDecision(
        lender_id=lender.id,
        borrower_id=borrower.id,
        decision="REJECT",
        reasoning="insufficient support",
    )

    run = build_run(borrower, lender, decision)
    run.inputs.missing_info = ["bank statements"]
    run.trace.steps = [
        TraceStep(
            t="2026-02-24T00:00:00Z",
            type="doc_request",
            name="request_docs",
            args={"doc_type": "bank_statement"},
            content="Need latest bank statements",
        )
    ]
    run.trace.latency_ms = 1234

    restored = UnderwritingRun.from_json(run.to_json())
    assert restored.inputs.business.company_name == run.inputs.business.company_name
    assert restored.inputs.missing_info == ["bank statements"]
    assert len(restored.trace.steps) == 1
    assert restored.trace.steps[0].content == "Need latest bank statements"
    assert restored.trace.latency_ms == 1234


def test_score_run_preserves_run_scores_type() -> None:
    borrower = get_borrowers("easy", seed=1, sample_size=1)[0]
    lender = get_lenders()[0]
    decision = LenderDecision(
        lender_id=lender.id,
        borrower_id=borrower.id,
        decision="REJECT",
        reasoning="decline",
    )
    run = build_run(borrower, lender, decision)

    card = score_run(run)
    assert card.overall_score >= 0
    assert isinstance(run.scores, RunScores)
    assert run.scores.overall is not None
    assert isinstance(run.scores.overall.get("score"), float)


def test_los_decision_normalizes_invalid_tenor() -> None:
    lender = get_lenders()[0]
    borrower = get_borrowers("easy", seed=1, sample_size=1)[0]
    run = UnderwritingRun()
    run.decision = RunDecision(
        action="approve",
        terms=DecisionTerms(amount=10000.0, apr=0.1, tenor_months=0),
    )

    decision = run_to_lender_decision(run, lender, borrower)
    assert decision.decision == "APPROVE"
    assert decision.term_sheet is not None
    assert decision.term_sheet.term_months == 24


def test_engine_resolve_loans_handles_zero_tenor() -> None:
    engine = SimulationEngine(borrowers=[], lenders=[], mock=True)
    engine.booked_loans = [
        BookedLoan(
            id="LOAN-001",
            borrower_id="BRW-001",
            lender_id="LND-001",
            borrower_name="Test Borrower",
            sector="SaaS",
            principal=10000.0,
            interest_rate=10.0,
            term_months=0,
            true_outcome="good",
            months_before_default=None,
        )
    ]

    engine.resolve_loans()

    assert len(engine.loan_outcomes) == 1
    assert engine.loan_outcomes[0].months_paid == 24
