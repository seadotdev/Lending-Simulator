import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from loanville.leaderboard.core import season_week_to_match_record, validate_match
from loanville.models import (
    BookedLoan,
    Borrower,
    EconomicsConfig,
    FinancialDossier,
    LenderConfig,
    LenderDecision,
    MonthlyStatement,
    QuarterlyIncome,
    TermSheet,
    Transaction,
)


ROOT = Path(__file__).resolve().parents[1]
ELO_BENCHMARK_DIR = ROOT / "legacy" / "benchmarks"
if str(ELO_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(ELO_BENCHMARK_DIR))
import elo_benchmark  # noqa: E402


def _build_borrower(
    borrower_id: str,
    company_name: str,
    true_outcome: str,
    months_before_default: int | None = None,
    request_amount: float = 100_000.0,
) -> Borrower:
    dossier = FinancialDossier(
        company_name=company_name,
        sector="Technology",
        years_in_business=6,
        annual_revenue=1_500_000.0,
        annual_expenses=1_200_000.0,
        net_income=300_000.0,
        employee_count=30,
        bank_statements=[
            MonthlyStatement(
                month="2025-12",
                opening_balance=250_000.0,
                deposits=[Transaction(date="2025-12-05", description="Revenue", amount=180_000.0)],
                withdrawals=[Transaction(date="2025-12-15", description="Payroll", amount=95_000.0)],
                ending_balance=335_000.0,
            )
        ],
        quarterly_income=[
            QuarterlyIncome(
                quarter="Q4 2025",
                revenue=380_000.0,
                expenses=300_000.0,
                gross_profit=80_000.0,
                gross_margin_pct=21.1,
                net_income=80_000.0,
                net_margin_pct=21.1,
            )
        ],
        narrative="Stable borrower profile.",
        loan_request_amount=request_amount,
        loan_purpose="Working capital",
    )
    return Borrower(
        id=borrower_id,
        dossier=dossier,
        true_outcome=true_outcome,
        months_before_default=months_before_default,
    )


def _approve(lender_id: str, borrower: Borrower, rate: float) -> LenderDecision:
    return LenderDecision(
        lender_id=lender_id,
        borrower_id=borrower.id,
        decision="APPROVE",
        reasoning="Approved after underwriting review.",
        term_sheet=TermSheet(
            loan_amount=borrower.dossier.loan_request_amount,
            interest_rate=rate,
            term_months=24,
        ),
    )


def _reject(lender_id: str, borrower: Borrower) -> LenderDecision:
    return LenderDecision(
        lender_id=lender_id,
        borrower_id=borrower.id,
        decision="REJECT",
        reasoning="Rejected due to risk profile.",
        term_sheet=None,
    )


class Tier2RegressionTests(unittest.TestCase):
    def test_season_week_record_is_valid_and_uses_unique_model_slot_ids(self) -> None:
        lenders = [
            LenderConfig(
                id="LND-001",
                name="Lender One",
                persona="Lender One persona",
                model="meta-llama/llama-3.3-70b-instruct",
                target_yield_pct=10.0,
                max_single_loan=500_000.0,
                total_capital=5_000_000.0,
                sector_limits={"Technology": 0.6},
            ),
            LenderConfig(
                id="LND-002",
                name="Lender Two",
                persona="Lender Two persona",
                model="meta-llama/llama-3.3-70b-instruct",
                target_yield_pct=10.0,
                max_single_loan=500_000.0,
                total_capital=5_000_000.0,
                sector_limits={"Technology": 0.6},
            ),
        ]

        borrowers = [
            _build_borrower("B-001", "Alpha", "good"),
            _build_borrower("B-002", "Bravo", "bad", months_before_default=6),
            _build_borrower("B-003", "Charlie", "fraud", months_before_default=1),
            _build_borrower("B-004", "Delta", "good"),
            _build_borrower("B-005", "Echo", "bad", months_before_default=8),
        ]

        decisions_l1 = [
            _approve("LND-001", borrowers[0], 11.0),
            _approve("LND-001", borrowers[1], 12.5),
            _reject("LND-001", borrowers[2]),
            _reject("LND-001", borrowers[3]),
            _reject("LND-001", borrowers[4]),
        ]
        decisions_l2 = [
            _reject("LND-002", borrowers[0]),
            _reject("LND-002", borrowers[1]),
            _reject("LND-002", borrowers[2]),
            _approve("LND-002", borrowers[3], 10.5),
            _approve("LND-002", borrowers[4], 13.0),
        ]

        booked_loans = [
            BookedLoan(
                id="LN-001",
                borrower_id="B-001",
                lender_id="LND-001",
                borrower_name="Alpha",
                sector="Technology",
                principal=100_000.0,
                interest_rate=11.0,
                term_months=24,
                true_outcome="good",
            ),
            BookedLoan(
                id="LN-002",
                borrower_id="B-002",
                lender_id="LND-001",
                borrower_name="Bravo",
                sector="Technology",
                principal=100_000.0,
                interest_rate=12.5,
                term_months=24,
                true_outcome="bad",
                months_before_default=6,
            ),
            BookedLoan(
                id="LN-003",
                borrower_id="B-004",
                lender_id="LND-002",
                borrower_name="Delta",
                sector="Technology",
                principal=100_000.0,
                interest_rate=10.5,
                term_months=24,
                true_outcome="good",
            ),
            BookedLoan(
                id="LN-004",
                borrower_id="B-005",
                lender_id="LND-002",
                borrower_name="Echo",
                sector="Technology",
                principal=100_000.0,
                interest_rate=13.0,
                term_months=24,
                true_outcome="bad",
                months_before_default=8,
            ),
        ]

        week_data = {
            "week": 1,
            "borrowers": borrowers,
            "all_decisions": {
                "LND-001": decisions_l1,
                "LND-002": decisions_l2,
            },
            "booked_loans": booked_loans,
            "deal_results": {
                "B-001": {"outcome": "booked", "winner": "LND-001"},
                "B-002": {"outcome": "booked", "winner": "LND-001"},
                "B-003": {"outcome": "no_takers", "winner": None},
                "B-004": {"outcome": "booked", "winner": "LND-002"},
                "B-005": {"outcome": "booked", "winner": "LND-002"},
            },
            "runs": [],
        }

        record = season_week_to_match_record(
            week_data=week_data,
            lenders=lenders,
            mix="gentle",
            economics=EconomicsConfig(),
        )

        model_ids = [m["model_id"] for m in record["models"]]
        self.assertEqual(len(model_ids), len(set(model_ids)))
        self.assertEqual(
            model_ids,
            [
                "meta-llama/llama-3.3-70b-instruct",
                "meta-llama/llama-3.3-70b-instruct::2",
            ],
        )
        self.assertEqual(record["season_week"], 1)
        self.assertEqual(record["season_match_type"], "weekly")

        validation = validate_match(record)
        self.assertTrue(validation["valid"], msg=f"validation errors: {validation['errors']}")

    def test_resume_tournament_seeds_inter_match_feedback(self) -> None:
        models = [("m1", "Model One"), ("m2", "Model Two"), ("m3", "Model Three")]

        resume_data = {
            "dealshare_ratings": {"m1": 1500.0, "m2": 1500.0, "m3": 1500.0},
            "profit_ratings": {"m1": 1500.0, "m2": 1500.0, "m3": 1500.0},
            "credit_ratings": {"m1": 1500.0, "m2": 1500.0, "m3": 1500.0},
            "match_log": [
                {
                    "match": 1,
                    "models": ["m1", "m2", "m3"],
                    "results": [
                        {"model": "m1", "score": 12.0, "deals_won": 2},
                        {"model": "m2", "score": 4.0, "deals_won": 1},
                        {"model": "m3", "score": -8.0, "deals_won": 0},
                    ],
                    "elapsed": 1.2,
                    "cost": 0.0,
                }
            ],
            "total_cost": 0.0,
        }

        captured: dict[str, dict | None] = {"feedback": None}

        def _fake_run_match(
            triplet,
            mix,
            api_key,
            sample_borrowers=None,
            rng=None,
            model_feedback=None,
        ):
            captured["feedback"] = model_feedback

            def _pap(state: str, utility: float, rate_offered):
                return {
                    "b1": 0.0,
                    "_decision_states": {"b1": state},
                    "_ground_truth": {"b1": "good"},
                    "_utility": {"b1": utility},
                    "_rates_offered": {"b1": rate_offered},
                }

            return [
                {
                    "model": "m1",
                    "display_name": "Model One",
                    "score": 10.0,
                    "deals_won": 1,
                    "approvals": 1,
                    "n_borrowers": 1,
                    "frauds_funded": 0,
                    "defaults": 0,
                    "deployed": 100_000.0,
                    "net_pnl": 2_000.0,
                    "cost": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "confusion_matrix": {
                        "good": {"approved": 1, "rejected": 0},
                        "bad": {"approved": 0, "rejected": 0},
                        "fraud": {"approved": 0, "rejected": 0},
                    },
                    "per_applicant_payoffs": _pap("won", 2_000.0, 10.0),
                },
                {
                    "model": "m2",
                    "display_name": "Model Two",
                    "score": 5.0,
                    "deals_won": 0,
                    "approvals": 1,
                    "n_borrowers": 1,
                    "frauds_funded": 0,
                    "defaults": 0,
                    "deployed": 0.0,
                    "net_pnl": 0.0,
                    "cost": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "confusion_matrix": {
                        "good": {"approved": 1, "rejected": 0},
                        "bad": {"approved": 0, "rejected": 0},
                        "fraud": {"approved": 0, "rejected": 0},
                    },
                    "per_applicant_payoffs": _pap("lost", 500.0, 11.0),
                },
                {
                    "model": "m3",
                    "display_name": "Model Three",
                    "score": -5.0,
                    "deals_won": 0,
                    "approvals": 0,
                    "n_borrowers": 1,
                    "frauds_funded": 0,
                    "defaults": 0,
                    "deployed": 0.0,
                    "net_pnl": 0.0,
                    "cost": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "confusion_matrix": {
                        "good": {"approved": 0, "rejected": 1},
                        "bad": {"approved": 0, "rejected": 0},
                        "fraud": {"approved": 0, "rejected": 0},
                    },
                    "per_applicant_payoffs": _pap("declined", 500.0, None),
                },
            ]

        with patch.object(elo_benchmark, "make_lender", return_value=object()), \
             patch.object(elo_benchmark, "get_borrowers", return_value=[]), \
             patch.object(elo_benchmark, "calculate_perfect_score", return_value=0.0), \
             patch.object(elo_benchmark, "compute_heuristic_baseline", return_value=0.0), \
             patch.object(elo_benchmark, "generate_matchups", return_value=[models]), \
             patch.object(elo_benchmark, "run_match", side_effect=_fake_run_match), \
             patch.object(elo_benchmark, "_save_results", return_value=None):
            elo_benchmark.run_tournament(
                models=models,
                n_matches=2,
                mix="easy",
                api_key="test-key",
                resume_data=resume_data,
                inter_match_feedback=True,
                leaderboard=False,
                output_file="unused.json",
            )

        feedback = captured["feedback"]
        self.assertIsInstance(feedback, dict)
        assert isinstance(feedback, dict)
        self.assertEqual(set(feedback.keys()), {"m1", "m2", "m3"})
        self.assertEqual(feedback["m1"]["score"], 12.0)
        self.assertEqual(feedback["m2"]["deals_won"], 1)
        self.assertEqual(feedback["m3"]["score"], -8.0)


if __name__ == "__main__":
    unittest.main()
