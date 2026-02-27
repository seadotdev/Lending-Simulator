import unittest

from loanville.los_adapter import run_to_decision
from loanville.run_schema import (
    DecisionRationale,
    DecisionTerms,
    RunCase,
    RunDecision,
    RunPolicy,
    RunTrace,
    TraceStep,
    UnderwritingRun,
)


def _approve_run(with_tool_call: bool) -> UnderwritingRun:
    steps = []
    if with_tool_call:
        steps.append(
            TraceStep(
                t="2026-02-26T00:00:00Z",
                type="tool_call",
                name="llm_evaluate",
                args={},
                result={},
            )
        )

    return UnderwritingRun(
        case=RunCase(
            case_id="BRW-001",
            source="los",
            segment="crm",
            requested_amount=150000,
            requested_tenor_months=24,
        ),
        policy=RunPolicy(
            policy_id="p_LND-001_model",
            model="model",
            params={"max_single_loan": 200000, "target_yield_pct": 11.0},
        ),
        decision=RunDecision(
            action="approve",
            terms=DecisionTerms(amount=140000, apr=0.11, tenor_months=24),
            rationale=DecisionRationale(summary="approve"),
        ),
        trace=RunTrace(steps=steps, latency_ms=1200),
    )


class LOSFormalityTests(unittest.TestCase):
    def test_approve_without_tool_call_is_not_formal_offer(self) -> None:
        run = _approve_run(with_tool_call=False)
        decision = run_to_decision(run, require_formal_offer_trace=True)
        self.assertEqual(decision.decision, "PASS")
        self.assertIsNone(decision.term_sheet)
        self.assertIn("[LOS_FORMALITY]", decision.reasoning)

    def test_approve_with_tool_call_remains_approve(self) -> None:
        run = _approve_run(with_tool_call=True)
        decision = run_to_decision(run, require_formal_offer_trace=True)
        self.assertEqual(decision.decision, "APPROVE")
        self.assertIsNotNone(decision.term_sheet)


if __name__ == "__main__":
    unittest.main()

