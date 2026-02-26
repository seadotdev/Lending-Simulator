import unittest

from loanville.data import get_lenders
from loanville.presets import (
    apply_lender_preset,
    list_lender_presets,
    list_scenario_presets,
    load_lender_preset,
    load_scenario_preset,
)


class PresetTests(unittest.TestCase):
    def test_builtin_preset_names_are_discoverable(self) -> None:
        self.assertIn("budget-league", list_lender_presets())
        self.assertIn("realistic-10w", list_scenario_presets())

    def test_budget_league_applies_models_and_selects_lenders(self) -> None:
        assignments, _ = load_lender_preset("budget-league")
        configured = apply_lender_preset(get_lenders(), assignments)

        self.assertEqual(
            [l.name for l in configured],
            ["Velocity Capital", "Heritage Trust Bank", "Meridian Partners"],
        )
        self.assertEqual(
            [l.model for l in configured],
            [
                "openai/gpt-4.1-nano",
                "mistralai/mistral-small-3.2-24b-instruct",
                "deepseek/deepseek-v3.2",
            ],
        )

    def test_scenario_preset_parses_expected_fields(self) -> None:
        scenario, _ = load_scenario_preset("stress-20w")

        self.assertEqual(scenario["weeks"], 20)
        self.assertEqual(scenario["cohort_size"], 20)
        self.assertEqual(scenario["season_mix"], "stress")
        self.assertEqual(scenario["arrival_phases"], 3)
        self.assertEqual(scenario["deep_uw_slots_per_week"], 3)
        self.assertTrue(scenario["speed_scoring"])
        self.assertTrue(scenario["custom_tools"])


if __name__ == "__main__":
    unittest.main()
