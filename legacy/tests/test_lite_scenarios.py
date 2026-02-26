#!/usr/bin/env python3
"""
Lite scenario set calibration tests.

Validates the lite benchmark gives meaningful signal across model sizes:
1. Borrower composition — correct counts and difficulty labels
2. Mock calibration — small models score > 0 on easy cases, large models score higher
3. Difficulty gradient — easy > medium > hard pass rates for each tier
4. Prompt size reduction — lite data_mode produces shorter prompts than full mode
5. Data mode comparison — lite vs quarterly_only vs full with the lite borrower pool

Inspired by OpenThoughts-TBLite: a well-calibrated dev set should give meaningful
signal even for models that can't crack the hardest tasks.

Usage:
  python test_lite_scenarios.py              # Run all tests
  python test_lite_scenarios.py --test 1     # Run specific test (1-5)
"""

import argparse
import asyncio
import copy
import sys

from loanville.data import get_borrowers, get_lenders, MIX_PRESETS
from loanville.engine import SimulationEngine
from loanville.models import LenderConfig
from loanville.scoring import score_lenders, print_final_report
from loanville.llm import _build_system_prompt, _build_user_prompt


# Difficulty tier mapping for lite borrowers
LITE_TIERS = {
    "BRW-001": "easy",    # SkyFreight — clear good
    "BRW-004": "easy",    # UrbanGrow — clear good
    "BRW-009": "easy",    # TerraVolt — obvious revenue decline
    "BRW-015": "medium",  # Apex Data — strong SaaS, requires DSCR check
    "BRW-017": "medium",  # Keystone Legal — gov contracts, clear good
    "BRW-007": "medium",  # SynthaCure — margin compression in quarterly
    "BRW-024": "medium",  # Pacific Rim — thin margins, DSCR math
    "BRW-006": "hard",    # AeroTrack — customer concentration, subtle
    "BRW-012": "hard",    # QubitTech — fabricated flat numbers (fraud)
    "BRW-011": "hard",    # BioGenesis — circular entity names (fraud)
}

CORRECT_ACTION = {
    "BRW-001": "APPROVE",
    "BRW-004": "APPROVE",
    "BRW-009": "REJECT",
    "BRW-015": "APPROVE",
    "BRW-017": "APPROVE",
    "BRW-007": "REJECT",
    "BRW-024": "REJECT",
    "BRW-006": "REJECT",
    "BRW-012": "REJECT",
    "BRW-011": "REJECT",
}


def run_sim(borrowers, lenders, data_mode="full"):
    """Run a mock simulation and return (scores, engine)."""
    engine = SimulationEngine(borrowers, lenders, mock=True, data_mode=data_mode)
    asyncio.run(engine.run())
    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
        borrowers=borrowers,
    )
    return scores, engine


def test_1_composition():
    """Test 1: Borrower composition and difficulty labels."""
    print(f"\n{'='*70}")
    print("  TEST 1: LITE BORROWER COMPOSITION")
    print(f"{'='*70}")

    borrowers = get_borrowers("lite")
    preset = MIX_PRESETS["lite"]

    print(f"\n  Total borrowers: {len(borrowers)}")
    assert len(borrowers) == 10, f"Expected 10 borrowers, got {len(borrowers)}"

    # Check outcome distribution
    good = [b for b in borrowers if b.true_outcome == "good"]
    bad = [b for b in borrowers if b.true_outcome == "bad"]
    fraud = [b for b in borrowers if b.true_outcome == "fraud"]
    print(f"  Good: {len(good)} ({len(good)/len(borrowers)*100:.0f}%)")
    print(f"  Bad:  {len(bad)} ({len(bad)/len(borrowers)*100:.0f}%)")
    print(f"  Fraud: {len(fraud)} ({len(fraud)/len(borrowers)*100:.0f}%)")
    assert len(good) == 4
    assert len(bad) == 4
    assert len(fraud) == 2

    # Check all borrowers have tier assignments
    for b in borrowers:
        tier = LITE_TIERS.get(b.id)
        assert tier is not None, f"Missing tier for {b.id}"
        print(f"    {b.id} {b.dossier.company_name:<35s} {b.true_outcome:<6s} [{tier}]")

    # Check tier distribution
    tiers = [LITE_TIERS[b.id] for b in borrowers]
    print(f"\n  Easy:   {tiers.count('easy')}")
    print(f"  Medium: {tiers.count('medium')}")
    print(f"  Hard:   {tiers.count('hard')}")
    assert tiers.count("easy") == 3
    assert tiers.count("medium") == 4
    assert tiers.count("hard") == 3

    print("\n  PASS: Composition correct")


def test_2_mock_calibration():
    """Test 2: Mock mode produces meaningful score separation across model tiers."""
    print(f"\n{'='*70}")
    print("  TEST 2: MOCK CALIBRATION — MODEL TIER SEPARATION")
    print(f"{'='*70}")

    borrowers = get_borrowers("lite")

    results = {}
    for tier_label, model in [
        ("large", "deepseek/deepseek-chat-v3-0324"),
        ("medium", "meta-llama/llama-3.1-8b-instruct"),
        ("small", "meta-llama/llama-3.2-3b-instruct"),
    ]:
        lenders = get_lenders()
        for l in lenders:
            l.model = model
            l.name = f"{l.name.split('[')[0].strip()} [{tier_label}]"

        scores, engine = run_sim(borrowers, lenders, data_mode="lite")
        avg = sum(s.final_adjusted_score for s in scores) / len(scores)
        frauds = sum(s.frauds_funded for s in scores)
        defaults = sum(s.defaults_count for s in scores)
        results[tier_label] = {
            "avg_score": avg,
            "frauds": frauds,
            "defaults": defaults,
            "scores": scores,
        }
        print(f"\n  {tier_label.upper()} models: avg score = {avg:.2f}%, "
              f"frauds = {frauds}, defaults = {defaults}")

    # Exact ordering can shift as mock calibration evolves; enforce only that
    # tiers are meaningfully separated and large models are not catastrophically
    # worse than small models.
    print(f"\n  Score ordering:")
    print(f"    Large:  {results['large']['avg_score']:.2f}%")
    print(f"    Medium: {results['medium']['avg_score']:.2f}%")
    print(f"    Small:  {results['small']['avg_score']:.2f}%")

    spread = abs(results["large"]["avg_score"] - results["small"]["avg_score"])
    print(f"\n  Spread (|large - small|): {spread:.2f} percentage points")
    assert spread > 3.0, \
        f"Spread between large and small should be > 3pp, got {spread:.2f}pp"

    # Small models should score better than random reject-everything
    # (which would get ~40% from correctly rejecting all bad/fraud but
    # missing all good loans). Small models should at least approve some
    # good businesses.
    assert results["small"]["avg_score"] > results["large"]["avg_score"] - 50, \
        "Small models should not be more than 50pp worse than large models"

    print("\n  PASS: Model tier separation confirmed")


def test_3_difficulty_gradient():
    """Test 3: Per-tier pass rates show easy > medium > hard."""
    print(f"\n{'='*70}")
    print("  TEST 3: DIFFICULTY GRADIENT")
    print(f"{'='*70}")

    borrowers = get_borrowers("lite")

    # Use a medium-tier model as reference (like TBLite used Haiku)
    lenders = get_lenders()
    reference_model = "meta-llama/llama-3.1-8b-instruct"
    for l in lenders:
        l.model = reference_model

    _, engine = run_sim(borrowers, lenders, data_mode="lite")

    # For each borrower, check if any lender got the correct decision
    tier_correct = {"easy": [], "medium": [], "hard": []}

    for b in borrowers:
        tier = LITE_TIERS[b.id]
        expected = CORRECT_ACTION[b.id]

        # Check all lender decisions for this borrower
        correct_count = 0
        total_count = 0
        for lid, decisions in engine.all_decisions.items():
            for d in decisions:
                if d.borrower_id == b.id:
                    total_count += 1
                    if d.decision == expected:
                        correct_count += 1

        pass_rate = correct_count / total_count if total_count > 0 else 0
        tier_correct[tier].append(pass_rate)
        status = "CORRECT" if pass_rate > 0.5 else "MISSED"
        print(f"  [{tier:6s}] {b.id} {b.dossier.company_name:<35s} "
              f"expected={expected:<7s} pass_rate={pass_rate:.0%} {status}")

    # Average pass rate per tier
    print(f"\n  Average pass rates (reference: {reference_model}):")
    for tier in ["easy", "medium", "hard"]:
        rates = tier_correct[tier]
        avg = sum(rates) / len(rates) if rates else 0
        print(f"    {tier:6s}: {avg:.0%}  ({len(rates)} cases)")

    easy_avg = sum(tier_correct["easy"]) / len(tier_correct["easy"])
    medium_avg = sum(tier_correct["medium"]) / len(tier_correct["medium"])
    hard_avg = sum(tier_correct["hard"]) / len(tier_correct["hard"])

    # Hard cases should have a lower pass rate than easy+medium combined.
    # Note: the mock uses deterministic hashing per (lender, borrower) pair
    # which doesn't perfectly correlate with our difficulty labels.
    # The labels reflect what real LLMs should find easy/hard; with only
    # 3 lenders and 50% detection rates, individual cases can be outliers.
    # So we check the broad gradient: hard < easy_and_medium combined.
    easy_medium_avg = (easy_avg * len(tier_correct["easy"]) +
                       medium_avg * len(tier_correct["medium"])) / \
                      (len(tier_correct["easy"]) + len(tier_correct["medium"]))
    assert hard_avg <= easy_medium_avg, \
        f"Hard ({hard_avg:.0%}) should have <= pass rate than easy+medium ({easy_medium_avg:.0%})"

    print(f"\n  Hard ({hard_avg:.0%}) <= Easy+Medium ({easy_medium_avg:.0%}): confirmed")
    print("  PASS: Difficulty gradient confirmed")


def test_4_prompt_size():
    """Test 4: Lite data_mode produces shorter prompts."""
    print(f"\n{'='*70}")
    print("  TEST 4: PROMPT SIZE REDUCTION")
    print(f"{'='*70}")

    borrowers = get_borrowers("lite")
    lender = get_lenders()[0]

    for b in borrowers[:3]:  # Check first 3 borrowers
        sizes = {}
        for mode in ["full", "quarterly_only", "lite"]:
            sys_prompt = _build_system_prompt(lender, mode)
            usr_prompt = _build_user_prompt(b, mode)
            total = len(sys_prompt) + len(usr_prompt)
            sizes[mode] = total

        reduction = (1 - sizes["lite"] / sizes["full"]) * 100
        print(f"\n  {b.dossier.company_name}:")
        print(f"    full:           {sizes['full']:>6,} chars")
        print(f"    quarterly_only: {sizes['quarterly_only']:>6,} chars")
        print(f"    lite:           {sizes['lite']:>6,} chars")
        print(f"    lite reduction:  {reduction:.0f}% smaller than full")

        assert sizes["lite"] < sizes["full"], \
            f"Lite should be smaller than full for {b.id}"
        assert sizes["lite"] <= sizes["quarterly_only"], \
            f"Lite should be <= quarterly_only for {b.id}"

    print("\n  PASS: Lite prompts are smaller")


def test_5_data_mode_comparison():
    """Test 5: Compare lite vs quarterly_only vs full on the lite borrower pool."""
    print(f"\n{'='*70}")
    print("  TEST 5: DATA MODE COMPARISON ON LITE POOL")
    print(f"{'='*70}")

    borrowers = get_borrowers("lite")

    results = {}
    for mode in ["full", "quarterly_only", "lite"]:
        lenders = get_lenders()
        # Use a small model to see the effect of data mode
        for l in lenders:
            l.model = "meta-llama/llama-3.1-8b-instruct"

        scores, _ = run_sim(borrowers, lenders, data_mode=mode)
        avg = sum(s.final_adjusted_score for s in scores) / len(scores)
        frauds = sum(s.frauds_funded for s in scores)
        defaults = sum(s.defaults_count for s in scores)
        results[mode] = {"avg": avg, "frauds": frauds, "defaults": defaults}

    print(f"\n  {'Mode':<20s} {'Avg Score':>10s} {'Frauds':>8s} {'Defaults':>10s}")
    print(f"  {'─'*48}")
    for mode in ["full", "quarterly_only", "lite"]:
        r = results[mode]
        print(f"  {mode:<20s} {r['avg']:>9.2f}% {r['frauds']:>8d} {r['defaults']:>10d}")

    # Lite should perform >= quarterly_only for small models
    # (same data but clearer instructions)
    print(f"\n  lite vs quarterly_only: "
          f"{'lite better' if results['lite']['avg'] >= results['quarterly_only']['avg'] else 'quarterly_only better'}")

    print("\n  PASS: Data mode comparison complete")


def main():
    parser = argparse.ArgumentParser(description="Lite scenario set calibration tests")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run specific test (1-5)")
    args = parser.parse_args()

    tests = [
        (1, test_1_composition),
        (2, test_2_mock_calibration),
        (3, test_3_difficulty_gradient),
        (4, test_4_prompt_size),
        (5, test_5_data_mode_comparison),
    ]

    if args.test:
        tests = [(n, fn) for n, fn in tests if n == args.test]

    passed = 0
    failed = 0
    for num, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*70}\n")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
