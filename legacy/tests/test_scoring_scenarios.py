#!/usr/bin/env python3
"""
Scoring verification tests for the Loanville lending simulation.

Validates that "profitable sustainable lending" is the key scoring determinant:

1. Same model (large) for all lenders — scores differ by strategy/capital, not model
2. Same model (medium) for all lenders — worse detection → worse scores overall
3. Same model (small) for all lenders — worst detection → worst scores overall
4. Cloned lenders (3x Velocity Capital) — similar configs produce similar-range scores
5. Cloned lenders (3x Heritage Trust) — ditto for conservative profile
6. Mixed: 1 good model + 2 bad models — good model should dominate
7. Tier comparison summary — large > medium > small on average

Usage:
  python test_scoring_scenarios.py              # Run all scenarios
  python test_scoring_scenarios.py --scenario 1 # Run specific scenario (1-7)
"""

import argparse
import asyncio
import copy
import sys

from loanville.data import get_borrowers, get_lenders
from loanville.engine import SimulationEngine
from loanville.models import LenderConfig, ExistingLoan
from loanville.scoring import score_lenders, print_final_report


def run_sim(borrowers, lenders):
    """Run a mock simulation and return (scores, engine)."""
    engine = SimulationEngine(borrowers, lenders, mock=True)
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


def test_mock_scoring_pipeline_smoke():
    """Pytest smoke test: mock pipeline runs end-to-end on lite mix."""
    borrowers = get_borrowers("lite")
    lenders = get_lenders()
    for lender in lenders:
        lender.model = "meta-llama/llama-3.1-8b-instruct"
    scores, engine = run_sim(borrowers, lenders)
    assert len(scores) == len(lenders)
    assert len(engine.booked_loans) >= 0
    assert all(hasattr(s, "final_adjusted_score") for s in scores)


def print_scores(scores, title):
    """Print a concise score table."""
    ranked = sorted(scores, key=lambda x: x.final_adjusted_score, reverse=True)
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"\n  {'Lender':<40s} {'Score':>8s} {'Won':>4s} {'Lost':>5s} "
          f"{'Rej':>4s} {'Fraud':>6s} {'Dflt':>5s} {'Net P&L':>12s}")
    print(f"  {'─'*84}")
    for s in ranked:
        print(f"  {s.lender_name:<40s} {s.final_adjusted_score:>+7.2f}% "
              f"{s.deals_won:>4d} {s.deals_lost:>5d} "
              f"{s.deals_rejected:>4d} {s.frauds_funded:>6d} {s.defaults_count:>5d} "
              f"${s.net_return:>10,.0f}")
    print()
    if ranked:
        winner = ranked[0]
        print(f"  Winner: {winner.lender_name} (Score: {winner.final_adjusted_score:+.2f}%)")
    return ranked


def print_detailed(scores):
    """Print detailed breakdown for each lender."""
    for s in sorted(scores, key=lambda x: x.final_adjusted_score, reverse=True):
        print(f"\n  {s.lender_name} ({s.model})")
        print(f"    Deployed:    ${s.total_deployed:>12,.2f}")
        print(f"    Interest:    ${s.total_interest_earned:>12,.2f}")
        print(f"    Prin. Lost:  ${s.total_principal_lost:>12,.2f}")
        print(f"    Net P&L:     ${s.net_return:>12,.2f}")
        if s.total_deployed > 0:
            print(f"    Raw ROI:     {s.roi_pct:>11.2f}%")
        print(f"    Frauds:      {s.frauds_funded}")
        print(f"    Defaults:    {s.defaults_count}")
        print(f"    Conc.Viol:   {len(s.concentration_violations)}")
        print(f"    Fraud Pen:   {s.fraud_penalty_pct:.1f}%")
        print(f"    Conc. Pen:   {s.concentration_penalty_pct:.1f}%")
        print(f"    Score:       {s.final_adjusted_score:+.2f}%")
        print(f"    Perfect:     {s.perfect_score:+.2f}%")


def make_clone_lenders(base_lender, count, model):
    """Create N clones of a lender with different IDs but same config."""
    clones = []
    for i in range(count):
        clone = copy.deepcopy(base_lender)
        clone.id = f"CLONE-{i+1:03d}"
        clone.name = f"{base_lender.name} (Clone {i+1})"
        clone.model = model
        clones.append(clone)
    return clones


# ==========================================================================
# Test Scenarios
# ==========================================================================

def scenario_1_same_model_large(borrowers):
    """All lenders use the same large model."""
    model = "deepseek/deepseek-chat-v3-0324"
    lenders = get_lenders()
    for l in lenders:
        l.model = model

    scores, engine = run_sim(borrowers, lenders)
    ranked = print_scores(scores, "SCENARIO 1: Same Model (Large) — All DeepSeek V3")
    print_detailed(scores)

    # Validate: scores should differ due to strategy, not model
    print(f"\n  ANALYSIS:")
    print(f"  All lenders use the same model ({model}), so differences")
    print(f"  must come from capital allocation, target yields, sector limits,")
    print(f"  and competitive dynamics.")
    spread = ranked[0].final_adjusted_score - ranked[-1].final_adjusted_score
    print(f"  Score spread: {spread:.2f} percentage points")

    # Check that fraud/default counts are reasonable for large model (95%/85% detection)
    total_fraud = sum(s.frauds_funded for s in scores)
    total_defaults = sum(s.defaults_count for s in scores)
    print(f"  Total frauds funded (all lenders): {total_fraud}")
    print(f"  Total defaults (all lenders): {total_defaults}")

    return scores


def scenario_2_same_model_medium(borrowers):
    """All lenders use the same medium model."""
    model = "meta-llama/llama-3.1-8b-instruct"
    lenders = get_lenders()
    for l in lenders:
        l.model = model

    scores, engine = run_sim(borrowers, lenders)
    ranked = print_scores(scores, "SCENARIO 2: Same Model (Medium) — All Llama 8B")
    print_detailed(scores)

    total_fraud = sum(s.frauds_funded for s in scores)
    total_defaults = sum(s.defaults_count for s in scores)
    print(f"\n  ANALYSIS:")
    print(f"  Medium models: 50% fraud detection, 40% bad detection")
    print(f"  Total frauds funded: {total_fraud} (expect more than large)")
    print(f"  Total defaults: {total_defaults} (expect more than large)")

    return scores


def scenario_3_same_model_small(borrowers):
    """All lenders use the same small model."""
    model = "microsoft/phi-3-mini-128k-instruct"
    lenders = get_lenders()
    for l in lenders:
        l.model = model

    scores, engine = run_sim(borrowers, lenders)
    ranked = print_scores(scores, "SCENARIO 3: Same Model (Small) — All Phi-3 Mini")
    print_detailed(scores)

    total_fraud = sum(s.frauds_funded for s in scores)
    total_defaults = sum(s.defaults_count for s in scores)
    print(f"\n  ANALYSIS:")
    print(f"  Small models: 20% fraud detection, 15% bad detection")
    print(f"  Total frauds funded: {total_fraud} (expect most)")
    print(f"  Total defaults: {total_defaults} (expect most)")

    return scores


def scenario_4_clone_velocity(borrowers):
    """3 clones of Velocity Capital (aggressive lender)."""
    base = get_lenders()[0]  # Velocity Capital
    model = "deepseek/deepseek-chat-v3-0324"
    clones = make_clone_lenders(base, 3, model)

    scores, engine = run_sim(borrowers, clones)
    ranked = print_scores(scores, "SCENARIO 4: 3x Velocity Capital Clones (Large Model)")
    print_detailed(scores)

    print(f"\n  ANALYSIS:")
    print(f"  Three identical Velocity Capital clones compete against each other.")
    print(f"  Hash-based randomness (different IDs) creates slight variation in")
    print(f"  detection outcomes and rate offers.")
    score_vals = [s.final_adjusted_score for s in scores]
    avg = sum(score_vals) / len(score_vals)
    spread = max(score_vals) - min(score_vals)
    print(f"  Average score: {avg:+.2f}%")
    print(f"  Score spread: {spread:.2f}pp")
    print(f"  Deals split: {[s.deals_won for s in ranked]}")

    return scores


def scenario_5_clone_heritage(borrowers):
    """3 clones of Heritage Trust (conservative lender)."""
    base = get_lenders()[1]  # Heritage Trust Bank
    model = "deepseek/deepseek-chat-v3-0324"
    clones = make_clone_lenders(base, 3, model)

    scores, engine = run_sim(borrowers, clones)
    ranked = print_scores(scores, "SCENARIO 5: 3x Heritage Trust Clones (Large Model)")
    print_detailed(scores)

    print(f"\n  ANALYSIS:")
    print(f"  Three identical Heritage Trust clones (conservative) competing.")
    score_vals = [s.final_adjusted_score for s in scores]
    avg = sum(score_vals) / len(score_vals)
    spread = max(score_vals) - min(score_vals)
    print(f"  Average score: {avg:+.2f}%")
    print(f"  Score spread: {spread:.2f}pp")
    print(f"  Deals split: {[s.deals_won for s in ranked]}")

    return scores


def scenario_6_mixed_tiers(borrowers):
    """1 large model vs 2 small models — test that better model wins."""
    lenders = get_lenders()
    lenders[0].model = "deepseek/deepseek-chat-v3-0324"
    lenders[0].name = "Velocity Capital [DeepSeek V3 - LARGE]"
    lenders[1].model = "microsoft/phi-3-mini-128k-instruct"
    lenders[1].name = "Heritage Trust [Phi-3 - SMALL]"
    lenders[2].model = "microsoft/phi-3-mini-128k-instruct"
    lenders[2].name = "Meridian Partners [Phi-3 - SMALL]"

    scores, engine = run_sim(borrowers, lenders)
    ranked = print_scores(scores, "SCENARIO 6: 1 Large Model vs 2 Small Models")
    print_detailed(scores)

    print(f"\n  ANALYSIS:")
    large_score = next(s for s in scores if s.lender_id == "LND-001")
    small_scores = [s for s in scores if s.lender_id != "LND-001"]
    small_avg = sum(s.final_adjusted_score for s in small_scores) / len(small_scores)
    print(f"  Large model (Velocity): {large_score.final_adjusted_score:+.2f}%")
    print(f"  Small models avg: {small_avg:+.2f}%")
    print(f"  Large model advantage: {large_score.final_adjusted_score - small_avg:+.2f}pp")
    print(f"  Large model frauds: {large_score.frauds_funded} | "
          f"Small frauds: {sum(s.frauds_funded for s in small_scores)}")

    return scores


def scenario_7_tier_summary(all_results):
    """Cross-scenario summary comparing tier performance."""
    print(f"\n{'='*80}")
    print(f"  SCENARIO 7: TIER COMPARISON SUMMARY")
    print(f"{'='*80}")

    if len(all_results) < 3:
        print("  (Need scenarios 1-3 to compute tier summary)")
        return

    tiers = ["Large (Scenario 1)", "Medium (Scenario 2)", "Small (Scenario 3)"]
    for i, (tier_name, scores) in enumerate(zip(tiers, all_results[:3])):
        avg_score = sum(s.final_adjusted_score for s in scores) / len(scores)
        total_fraud = sum(s.frauds_funded for s in scores)
        total_defaults = sum(s.defaults_count for s in scores)
        total_net = sum(s.net_return for s in scores)
        total_deployed = sum(s.total_deployed for s in scores)
        print(f"\n  {tier_name}:")
        print(f"    Avg Score:      {avg_score:>+8.2f}%")
        print(f"    Total Frauds:   {total_fraud:>8d}")
        print(f"    Total Defaults: {total_defaults:>8d}")
        print(f"    Total Net P&L:  ${total_net:>12,.0f}")
        print(f"    Total Deployed: ${total_deployed:>12,.0f}")

    # Validate the key assertion
    large_avg = sum(s.final_adjusted_score for s in all_results[0]) / len(all_results[0])
    medium_avg = sum(s.final_adjusted_score for s in all_results[1]) / len(all_results[1])
    small_avg = sum(s.final_adjusted_score for s in all_results[2]) / len(all_results[2])

    print(f"\n  KEY ASSERTIONS:")
    large_gt_medium = large_avg > medium_avg
    medium_gt_small = medium_avg > small_avg
    print(f"  Large > Medium: {large_avg:+.2f}% vs {medium_avg:+.2f}% "
          f"=> {'PASS' if large_gt_medium else 'FAIL'}")
    print(f"  Medium > Small: {medium_avg:+.2f}% vs {small_avg:+.2f}% "
          f"=> {'PASS' if medium_gt_small else 'FAIL'}")

    large_fraud = sum(s.frauds_funded for s in all_results[0])
    medium_fraud = sum(s.frauds_funded for s in all_results[1])
    small_fraud = sum(s.frauds_funded for s in all_results[2])
    print(f"  Large fraud <= Medium fraud: {large_fraud} <= {medium_fraud} "
          f"=> {'PASS' if large_fraud <= medium_fraud else 'FAIL'}")
    print(f"  Medium fraud <= Small fraud: {medium_fraud} <= {small_fraud} "
          f"=> {'PASS' if medium_fraud <= small_fraud else 'FAIL'}")


def run_all_mixes():
    """Run scenarios across different mix presets for comprehensive testing."""
    for mix in ["easy", "hard"]:
        print(f"\n\n{'#'*80}")
        print(f"#  BORROWER MIX: {mix.upper()}")
        print(f"{'#'*80}")

        borrowers = get_borrowers(mix)
        from collections import Counter
        outcomes = Counter(b.true_outcome for b in borrowers)
        print(f"  Pool: {len(borrowers)} borrowers — "
              f"{outcomes.get('good',0)} good, "
              f"{outcomes.get('bad',0)} bad, "
              f"{outcomes.get('fraud',0)} fraud")

        all_results = []

        # Run tier scenarios
        r1 = scenario_1_same_model_large(borrowers)
        all_results.append(r1)

        r2 = scenario_2_same_model_medium(borrowers)
        all_results.append(r2)

        r3 = scenario_3_same_model_small(borrowers)
        all_results.append(r3)

        # Clone scenarios
        scenario_4_clone_velocity(borrowers)
        scenario_5_clone_heritage(borrowers)

        # Mixed tier
        scenario_6_mixed_tiers(borrowers)

        # Summary
        scenario_7_tier_summary(all_results)


def main():
    parser = argparse.ArgumentParser(description="Scoring scenario tests")
    parser.add_argument("--scenario", type=int, choices=range(1, 8),
                        help="Run a specific scenario (1-7)")
    parser.add_argument("--mix", default="easy", choices=["easy", "balanced", "hard", "all"],
                        help="Borrower mix (default: easy)")
    parser.add_argument("--all-mixes", action="store_true",
                        help="Run across easy + hard mixes")
    args = parser.parse_args()

    if args.all_mixes:
        run_all_mixes()
        return

    borrowers = get_borrowers(args.mix)
    from collections import Counter
    outcomes = Counter(b.true_outcome for b in borrowers)
    print(f"\nBorrower mix: {args.mix} — {len(borrowers)} borrowers")
    print(f"  {outcomes.get('good',0)} good, {outcomes.get('bad',0)} bad, "
          f"{outcomes.get('fraud',0)} fraud")

    if args.scenario:
        scenarios = {
            1: scenario_1_same_model_large,
            2: scenario_2_same_model_medium,
            3: scenario_3_same_model_small,
            4: scenario_4_clone_velocity,
            5: scenario_5_clone_heritage,
            6: scenario_6_mixed_tiers,
        }
        if args.scenario == 7:
            print("  (Scenario 7 requires running 1-3 first, use no --scenario flag)")
            return
        scenarios[args.scenario](borrowers)
    else:
        # Run all scenarios
        all_results = []
        for i, fn in [(1, scenario_1_same_model_large),
                       (2, scenario_2_same_model_medium),
                       (3, scenario_3_same_model_small),
                       (4, scenario_4_clone_velocity),
                       (5, scenario_5_clone_heritage),
                       (6, scenario_6_mixed_tiers)]:
            result = fn(borrowers)
            if i <= 3:
                all_results.append(result)

        scenario_7_tier_summary(all_results)

    print(f"\n{'='*80}")
    print(f"  ALL SCENARIOS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
