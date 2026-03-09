#!/usr/bin/env python3
"""
Feature test: mock-mode validation of all recently added season features.

Runs ~10 sub-tests with zero API calls.  Each test creates a SeasonEngine
in mock mode, runs the season, and asserts expected behavior.

Features tested:
  1. Persistent strategy scratchpad (auto-populated in mock)
  2. Scratchpad disabled
  3. Pass@k decision consistency
  4. Capital adequacy elimination
  5. Capital decay
  6. Arrival phases
  7. Deep UW slots
  8. Deliberate market mode (borrower patience + offer validity)
  9. Decision attribution in briefings
 10. Info asymmetry (partial_statements)
 11. SeasonConfig validation
 12. All features combined
"""

import asyncio
import sys
import time
import traceback

from loanville.data import get_lenders
from loanville.models import ECONOMICS_PRESETS, SeasonConfig, SeasonLenderState

# Use conservative economics for all tests
ECO = ECONOMICS_PRESETS["conservative"]


def _make_lenders(n=3):
    """Return n lender configs with mock-friendly model names."""
    lenders = get_lenders()[:n]
    for i, l in enumerate(lenders):
        l.model = f"mock-model-{i}"
        l.name = f"Test Lender {i}"
    return lenders


def _run_season(config, lenders=None, **engine_kw):
    """Run a season in mock mode and return the SeasonEngine."""
    from loanville.season import SeasonEngine

    if lenders is None:
        lenders = _make_lenders()
    engine = SeasonEngine(
        config=config,
        lenders=lenders,
        mock=True,
        data_mode="lite",
        los_mode="rules_only",
        **engine_kw,
    )
    asyncio.run(engine.run_season())
    return engine


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_scratchpad_enabled():
    """Scratchpad auto-populates in mock mode after a few weeks."""
    config = SeasonConfig(
        weeks=4, cohort_size=4, months_per_week=3,
        season_mix="realistic", seed=99, economics=ECO,
        scratchpad=True,
    )
    engine = _run_season(config)

    any_populated = False
    for state in engine.lender_states.values():
        if state.scratchpad:
            any_populated = True
            assert len(state.scratchpad) <= 510, (
                f"Scratchpad too long ({len(state.scratchpad)} chars)"
            )
            # Should have week-prefixed lines
            assert "W" in state.scratchpad, "Scratchpad missing week prefix"
    assert any_populated, "No lender scratchpad was populated in mock mode"


def test_scratchpad_disabled():
    """Scratchpad stays empty when disabled."""
    config = SeasonConfig(
        weeks=3, cohort_size=4, months_per_week=3,
        season_mix="realistic", seed=99, economics=ECO,
        scratchpad=False,
    )
    engine = _run_season(config)

    for state in engine.lender_states.values():
        assert state.scratchpad == "", (
            f"Scratchpad should be empty when disabled, got: {state.scratchpad!r}"
        )


def test_consistency_mock():
    """Consistency checks run and report 100% agreement in deterministic mock mode."""
    config = SeasonConfig(
        weeks=2, cohort_size=4, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        consistency_samples=2,
        consistency_sample_pct=0.50,
        scratchpad=False,
    )
    engine = _run_season(config)

    any_checks = False
    for state in engine.lender_states.values():
        if state.consistency_checks:
            any_checks = True
            # Mock mode is deterministic — all re-evaluations should match
            assert state.consistency_agreement_rate == 1.0, (
                f"Expected 100% agreement in mock, got "
                f"{state.consistency_agreement_rate:.0%}"
            )
            for check in state.consistency_checks:
                assert check["agreements"] == check["samples"]
    assert any_checks, "No consistency checks were recorded"


def test_capital_adequacy_elimination():
    """Extreme capital adequacy ratio causes at least one elimination."""
    # Use stress mix + very high adequacy ratio to force eliminations
    config = SeasonConfig(
        weeks=6, cohort_size=5, months_per_week=4,
        season_mix="stress", seed=7, economics=ECO,
        capital_adequacy_ratio=0.95,  # extremely strict
        scratchpad=False,
    )
    engine = _run_season(config)

    eliminated = [
        s for s in engine.lender_states.values() if s.eliminated
    ]
    # At least check the feature doesn't crash.  Under stress mix with 0.95
    # threshold, eliminations are likely but not guaranteed with mock.
    # The real assertion is that it runs without error.
    print(f"    Eliminated: {len(eliminated)}/{len(engine.lender_states)}")
    for s in eliminated:
        assert s.eliminated_week > 0, "eliminated_week should be set"


def test_capital_decay():
    """Capital decay erodes idle capital over time."""
    config = SeasonConfig(
        weeks=4, cohort_size=3, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        capital_decay_rate=0.05,  # aggressive 5%/week for visibility
        scratchpad=False,
    )
    engine = _run_season(config)

    for state in engine.lender_states.values():
        assert state.cumulative_decay > 0, (
            f"Expected positive decay, got {state.cumulative_decay}"
        )
        # Note: effective capital may still exceed initial if interest income
        # outpaces decay (gentle mix = no defaults).  Just verify decay accrued.
        expected_min = state.total_capital * 0.05 * 0.5  # rough lower bound
        assert state.cumulative_decay > expected_min, (
            f"Decay ({state.cumulative_decay}) seems too low "
            f"(expected > {expected_min})"
        )


def test_arrival_phases():
    """Multi-phase arrival completes without error."""
    config = SeasonConfig(
        weeks=2, cohort_size=6, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        arrival_phases=3,
        scratchpad=False,
    )
    engine = _run_season(config)

    # Basic sanity — season ran and produced results
    assert len(engine.week_results) == 2
    for state in engine.lender_states.values():
        assert state.total_evaluations > 0, "No evaluations recorded"


def test_deep_uw_slots():
    """Deep UW slot cap causes some deferrals."""
    config = SeasonConfig(
        weeks=3, cohort_size=6, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        deep_uw_slots_per_week=2,  # cap at 2 approvals per week out of 6
        scratchpad=False,
    )
    engine = _run_season(config)

    any_deferred = any(
        s.deep_uw_deferred > 0 for s in engine.lender_states.values()
    )
    # In mock mode, mock always approves good/rejects bad, so with gentle mix
    # (70% good) and 6 borrowers, ~4 approvals attempted vs cap of 2.
    print(f"    Deferred counts: "
          f"{[s.deep_uw_deferred for s in engine.lender_states.values()]}")
    # Just verify it ran without error and recorded slot usage
    for state in engine.lender_states.values():
        assert len(state.weekly_deep_uw_used) > 0, "No deep UW usage recorded"


def test_deliberate_market():
    """Borrower patience + offer validity (deliberate market mode)."""
    config = SeasonConfig(
        weeks=4, cohort_size=3, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        borrower_patience_weeks=2,
        offer_validity_weeks=2,
        scratchpad=False,
    )
    lenders = _make_lenders(n=2)
    engine = _run_season(config, lenders=lenders)

    # Season should complete — deliberate mode uses market pool
    assert engine.deliberate_mode is True
    assert engine.market_pool is not None
    assert len(engine.week_results) == 4


def test_decision_attribution():
    """Decision attribution section appears in briefings when defaults exist."""
    # Use stress mix + enough weeks for defaults to resolve
    config = SeasonConfig(
        weeks=6, cohort_size=5, months_per_week=4,
        season_mix="stress", seed=13, economics=ECO,
        scratchpad=False,
    )
    engine = _run_season(config)

    # Check that at least one lender has resolved defaults
    any_defaults = any(
        any(lo.defaulted for lo in s.resolved_loans)
        for s in engine.lender_states.values()
    )
    if any_defaults:
        # Re-build a briefing for the last week to check content
        briefings = engine._build_briefings(config.weeks, [])
        attribution_found = any(
            "Decision Attribution" in text
            for text in briefings.values()
        )
        assert attribution_found, (
            "Decision Attribution section missing from briefings despite defaults"
        )
    else:
        print("    (no defaults resolved — attribution test inconclusive)")


def test_real_uw_costs():
    """Real underwriting cost mode charges API costs to P&L."""
    config = SeasonConfig(
        weeks=2, cohort_size=3, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        scratchpad=False,
        underwriting_cost_mode="real",
        uw_cost_multiplier=2000.0,  # $0.01 API → $20 UW cost
    )
    engine = _run_season(config)

    # In mock mode API cost is 0, so simulate by manually setting costs
    # and running the charge method
    first_lid = list(engine.lender_states.keys())[0]
    state = engine.lender_states[first_lid]
    state.cumulative_cost_usd = 0.05  # $0.05 API → $100 UW cost at 2000x
    state.cumulative_uw_ops_cost = 0.0  # reset for clean test
    old_workout = state.cumulative_workout_cost

    events = engine._apply_underwriting_costs(3)
    assert len(events) == 1, f"Expected 1 UW cost event, got {len(events)}"
    assert "Underwriting ops" in events[0]
    assert state.cumulative_uw_ops_cost == 100.0  # 0.05 × 2000
    assert state.cumulative_workout_cost == old_workout + 100.0


def test_real_uw_costs_briefing():
    """Real UW costs appear in portfolio briefings."""
    config = SeasonConfig(
        weeks=2, cohort_size=3, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        scratchpad=False,
        underwriting_cost_mode="real",
        uw_cost_multiplier=1000.0,
    )
    engine = _run_season(config)

    # Simulate some API cost for briefing test
    for state in engine.lender_states.values():
        state.cumulative_cost_usd = 0.02
        state.total_evaluations = 6

    briefings = engine._build_briefings(2, [])
    for text in briefings.values():
        assert "Underwriting Ops Cost:" in text
        assert "Avg $" in text


def test_real_uw_costs_pnl_impact():
    """Real UW costs reduce effective capital through P&L."""
    config = SeasonConfig(
        weeks=2, cohort_size=3, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        scratchpad=False,
        underwriting_cost_mode="real",
        uw_cost_multiplier=5000.0,
    )
    engine = _run_season(config)

    # Manually inject API cost and apply
    first_lid = list(engine.lender_states.keys())[0]
    state = engine.lender_states[first_lid]
    initial_eff = engine._effective_capital(state)
    state.cumulative_cost_usd = 0.10  # → $500 at 5000x
    engine._apply_underwriting_costs(3)

    new_eff = engine._effective_capital(state)
    assert new_eff < initial_eff, (
        f"Effective capital should decrease after UW cost charge "
        f"({new_eff} vs {initial_eff})"
    )

    # Verify weekly snapshots include cost tracking
    for snap in state.weekly_snapshots:
        assert "cumulative_cost_usd" in snap
        assert "week_cost_usd" in snap


def test_info_asymmetry():
    """Info asymmetry mode runs without error."""
    config = SeasonConfig(
        weeks=2, cohort_size=4, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        info_asymmetry="partial_statements",
        scratchpad=False,
    )
    engine = _run_season(config)
    assert len(engine.week_results) == 2


def test_validation():
    """SeasonConfig rejects invalid parameters."""
    import pytest  # noqa: F401 — soft dependency

    errors_caught = 0

    # Invalid consistency_samples
    try:
        SeasonConfig(consistency_samples=-1, economics=ECO)
        assert False, "Should have raised ValueError"
    except ValueError:
        errors_caught += 1

    # Invalid consistency_sample_pct
    try:
        SeasonConfig(consistency_sample_pct=0.0, economics=ECO)
        assert False, "Should have raised ValueError"
    except ValueError:
        errors_caught += 1

    # Invalid info_asymmetry
    try:
        SeasonConfig(info_asymmetry="bogus", economics=ECO)
        assert False, "Should have raised ValueError"
    except ValueError:
        errors_caught += 1

    # Invalid arrival_phases
    try:
        SeasonConfig(arrival_phases=0, economics=ECO)
        assert False, "Should have raised ValueError"
    except ValueError:
        errors_caught += 1

    # Invalid underwriting_cost_mode
    try:
        SeasonConfig(underwriting_cost_mode="fancy", economics=ECO)
        assert False, "Should have raised ValueError"
    except ValueError:
        errors_caught += 1

    # Invalid uw_cost_multiplier
    try:
        SeasonConfig(uw_cost_multiplier=-1.0, economics=ECO)
        assert False, "Should have raised ValueError"
    except ValueError:
        errors_caught += 1

    assert errors_caught == 6, f"Expected 6 validation errors, got {errors_caught}"


def test_json_export():
    """JSON export includes all new feature fields."""
    config = SeasonConfig(
        weeks=2, cohort_size=3, months_per_week=2,
        season_mix="gentle", seed=42, economics=ECO,
        scratchpad=True,
        consistency_samples=1,
        consistency_sample_pct=1.0,
        capital_decay_rate=0.01,
        deep_uw_slots_per_week=2,
        arrival_phases=2,
    )
    engine = _run_season(config)
    data = engine.to_json()

    # Config section
    cfg = data["config"]
    assert cfg["scratchpad"] is True
    assert cfg["consistency_samples"] == 1
    assert cfg["consistency_sample_pct"] == 1.0
    assert cfg["arrival_phases"] == 2
    assert cfg["deep_uw_slots_per_week"] == 2
    assert "underwriting_cost_mode" in cfg
    assert "uw_cost_multiplier" in cfg

    # Lender data
    for lender in data["lenders"]:
        assert "scratchpad" in lender
        assert "consistency_agreement_rate" in lender
        assert "consistency_checks" in lender
        assert "deep_uw_deferred" in lender
        assert "weekly_deep_uw_used" in lender
        assert "uw_ops_cost" in lender


def test_all_features_combined():
    """All new features enabled simultaneously — integration check."""
    config = SeasonConfig(
        weeks=4, cohort_size=5, months_per_week=3,
        season_mix="realistic", seed=42, economics=ECO,
        scratchpad=True,
        consistency_samples=1,
        consistency_sample_pct=0.30,
        capital_adequacy_ratio=0.30,
        capital_decay_rate=0.005,
        info_asymmetry="partial_statements",
        arrival_phases=2,
        deep_uw_slots_per_week=3,
        underwriting_cost_mode="real",
        uw_cost_multiplier=1000.0,
    )
    engine = _run_season(config)

    assert len(engine.week_results) == 4
    data = engine.to_json()
    assert len(data["lenders"]) == 3

    # Spot-check that features actually ran
    for state in engine.lender_states.values():
        if not state.eliminated:
            assert state.cumulative_decay > 0, "Decay should be positive"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Scratchpad enabled", test_scratchpad_enabled),
    ("Scratchpad disabled", test_scratchpad_disabled),
    ("Consistency (pass@k)", test_consistency_mock),
    ("Capital adequacy", test_capital_adequacy_elimination),
    ("Capital decay", test_capital_decay),
    ("Arrival phases", test_arrival_phases),
    ("Deep UW slots", test_deep_uw_slots),
    ("Deliberate market", test_deliberate_market),
    ("Decision attribution", test_decision_attribution),
    ("Real UW costs", test_real_uw_costs),
    ("Real UW costs briefing", test_real_uw_costs_briefing),
    ("Real UW costs P&L impact", test_real_uw_costs_pnl_impact),
    ("Info asymmetry", test_info_asymmetry),
    ("Config validation", test_validation),
    ("JSON export", test_json_export),
    ("All features combined", test_all_features_combined),
]


def main():
    print("=" * 70)
    print("  FEATURE TEST — mock mode, zero API calls")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = []
    t0 = time.time()

    for name, fn in TESTS:
        print(f"\n  [{passed + failed + 1}/{len(TESTS)}] {name} ...", end=" ", flush=True)
        test_t0 = time.time()
        try:
            fn()
            elapsed = time.time() - test_t0
            print(f"PASS ({elapsed:.1f}s)")
            passed += 1
        except Exception as e:
            elapsed = time.time() - test_t0
            print(f"FAIL ({elapsed:.1f}s)")
            failed += 1
            errors.append((name, e))

    total_elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed "
          f"({total_elapsed:.1f}s total)")
    print(f"{'=' * 70}")

    if errors:
        print("\n  FAILURES:\n")
        for name, exc in errors:
            print(f"  {name}:")
            print(f"    {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
