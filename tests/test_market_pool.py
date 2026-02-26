import unittest

from loanville.data import get_borrowers
from loanville.los_adapter import run_to_decision
from loanville.market_pool import MarketPool
from loanville.models import TermSheet
from loanville.run_schema import (
    DecisionRationale,
    DecisionTerms,
    RunCase,
    RunDecision,
    RunPolicy,
    UnderwritingRun,
)


class MarketPoolTests(unittest.TestCase):
    def test_borrower_and_offer_lifecycle(self) -> None:
        borrower = get_borrowers("easy", seed=42, sample_size=1)[0]
        pool = MarketPool(borrower_patience_weeks=3, offer_validity_weeks=2)

        pool.add_cohort([borrower], week=1)
        pool.add_offer(
            lender_id="LND-001",
            borrower_id=borrower.id,
            term_sheet=TermSheet(loan_amount=100000, interest_rate=9.5, term_months=24),
            reasoning="initial offer",
            issued_week=1,
            validity_weeks=2,
        )

        pool.tick(week=2)
        accepted, expired = pool.resolve_expired()
        self.assertEqual((len(accepted), len(expired)), (0, 0))
        self.assertEqual(pool.reserved_capital_for_lender("LND-001"), 100000)

        pool.tick(week=3)
        accepted, expired = pool.resolve_expired()
        self.assertEqual((len(accepted), len(expired)), (0, 0))
        self.assertEqual(pool.reserved_capital_for_lender("LND-001"), 0)

        pool.add_offer(
            lender_id="LND-001",
            borrower_id=borrower.id,
            term_sheet=TermSheet(loan_amount=120000, interest_rate=8.9, term_months=24),
            reasoning="renewed offer",
            issued_week=3,
            validity_weeks=2,
        )

        pool.tick(week=4)
        accepted, expired = pool.resolve_expired()
        self.assertEqual(len(expired), 0)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0][1].term_sheet.loan_amount, 120000)

    def test_run_to_decision_maps_pass_and_offer_validity(self) -> None:
        pass_run = UnderwritingRun(
            case=RunCase(case_id="BRW-001", source="simulator", segment="test"),
            policy=RunPolicy(policy_id="p_LND-001_model", model="model"),
            decision=RunDecision(
                action="refer",
                rationale=DecisionRationale(summary="[PASS] wait and monitor"),
            ),
        )
        pass_decision = run_to_decision(pass_run)
        self.assertEqual(pass_decision.decision, "PASS")
        self.assertIsNone(pass_decision.term_sheet)

        approve_run = UnderwritingRun(
            case=RunCase(
                case_id="BRW-002",
                source="simulator",
                segment="test",
                requested_amount=150000,
                requested_tenor_months=24,
            ),
            policy=RunPolicy(
                policy_id="p_LND-001_model",
                model="model",
                params={"max_single_loan": 200000, "target_yield_pct": 10.0},
            ),
            decision=RunDecision(
                action="approve",
                terms=DecisionTerms(amount=150000, apr=0.1, tenor_months=24),
                conditions=["offer_valid_weeks=3"],
                rationale=DecisionRationale(summary="approve"),
            ),
        )
        approve_decision = run_to_decision(approve_run)
        self.assertEqual(approve_decision.decision, "APPROVE")
        self.assertEqual(approve_decision.offer_valid_weeks, 3)


if __name__ == "__main__":
    unittest.main()
