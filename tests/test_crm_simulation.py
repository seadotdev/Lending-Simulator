import unittest

from loanville.crm_sim import (
    CRMSimulationConfig,
    _build_crm_cases,
    _load_open_los_crm_catalog,
)


class CRMSimulationTests(unittest.TestCase):
    def test_build_cases_all_incomplete_generates_follow_up_pass(self) -> None:
        cfg = CRMSimulationConfig(cases=8, incomplete_ratio=1.0, seed=7)
        borrowers, expectations = _build_crm_cases(cfg)

        self.assertEqual(len(borrowers), 8)
        self.assertEqual(len(expectations), 8)
        self.assertTrue(all(exp.requires_follow_up for exp in expectations.values()))
        self.assertTrue(all(exp.expected_action == "PASS" for exp in expectations.values()))

    def test_build_cases_zero_incomplete_uses_formulaic_actions(self) -> None:
        cfg = CRMSimulationConfig(cases=8, incomplete_ratio=0.0, seed=9)
        _, expectations = _build_crm_cases(cfg)

        self.assertTrue(all(not exp.requires_follow_up for exp in expectations.values()))
        self.assertTrue(
            all(exp.expected_action in {"APPROVE", "REJECT"} for exp in expectations.values())
        )

    def test_open_los_catalog_loader_returns_shape(self) -> None:
        catalog = _load_open_los_crm_catalog()
        self.assertIn("available", catalog)
        self.assertIn("path", catalog)


if __name__ == "__main__":
    unittest.main()

