"""
Entry point for: python -m loanville

Usage:
  python -m loanville              # Live mode (requires OPENROUTER_API_KEY)
  python -m loanville --mock       # Mock mode (no API key needed)
  python -m loanville --compare    # Compare big vs small models (mock)
"""

import argparse
import asyncio
import copy
import os
import sys

from dotenv import load_dotenv

from .data import get_borrowers, get_lenders
from .engine import SimulationEngine
from .scoring import print_final_report, score_lenders


def _run_single(borrowers, lenders, api_key="", mock=False):
    """Run a single simulation and return scores."""
    engine = SimulationEngine(borrowers, lenders, api_key, mock=mock)
    asyncio.run(engine.run())

    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
    )
    print_final_report(scores)
    return scores


def run_compare():
    """Run two simulations: big models vs small models, then compare."""
    borrowers = get_borrowers()

    # --- Round 1: Big Models ---
    big_lenders = get_lenders()
    big_lenders[0].model = "openai/gpt-4o"
    big_lenders[0].name = "Velocity Capital [GPT-4o]"
    big_lenders[1].model = "anthropic/claude-sonnet-4"
    big_lenders[1].name = "Heritage Trust [Claude Sonnet]"
    big_lenders[2].model = "meta-llama/llama-3.1-70b-instruct"
    big_lenders[2].name = "Meridian Partners [Llama-70B]"

    print("\n" + "#" * 70)
    print("#  ROUND 1: BIG MODELS")
    print("#" * 70)
    big_scores = _run_single(borrowers, big_lenders, mock=True)

    # --- Round 2: Small Models ---
    small_lenders = get_lenders()
    small_lenders[0].model = "meta-llama/llama-3.2-3b-instruct"
    small_lenders[0].name = "Velocity Capital [Llama-3B]"
    small_lenders[1].model = "google/gemma-2-2b-it"
    small_lenders[1].name = "Heritage Trust [Gemma-2B]"
    small_lenders[2].model = "qwen/qwen-2.5-3b-instruct"
    small_lenders[2].name = "Meridian Partners [Qwen-3B]"

    print("\n\n" + "#" * 70)
    print("#  ROUND 2: SMALL MODELS")
    print("#" * 70)
    small_scores = _run_single(borrowers, small_lenders, mock=True)

    # --- Comparison ---
    print("\n\n" + "=" * 70)
    print("  HEAD-TO-HEAD: BIG MODELS vs SMALL MODELS")
    print("=" * 70)

    print(f"\n  {'Metric':<30s} {'Big Models':>15s} {'Small Models':>15s}")
    print(f"  {'─'*60}")

    for label, big_list, small_list in [
        ("Avg Final Score (%)", big_scores, small_scores),
    ]:
        big_avg = sum(s.final_adjusted_score for s in big_list) / len(big_list)
        small_avg = sum(s.final_adjusted_score for s in small_list) / len(small_list)
        print(f"  {label:<30s} {big_avg:>14.2f}% {small_avg:>14.2f}%")

    big_fraud = sum(s.frauds_funded for s in big_scores)
    small_fraud = sum(s.frauds_funded for s in small_scores)
    print(f"  {'Total Frauds Funded':<30s} {big_fraud:>15d} {small_fraud:>15d}")

    big_defaults = sum(s.defaults_count for s in big_scores)
    small_defaults = sum(s.defaults_count for s in small_scores)
    print(f"  {'Total Defaults':<30s} {big_defaults:>15d} {small_defaults:>15d}")

    big_return = sum(s.net_return for s in big_scores)
    small_return = sum(s.net_return for s in small_scores)
    print(f"  {'Total Net Return ($)':<30s} ${big_return:>13,.0f} ${small_return:>13,.0f}")

    big_deployed = sum(s.total_deployed for s in big_scores)
    small_deployed = sum(s.total_deployed for s in small_scores)
    print(f"  {'Total Capital Deployed ($)':<30s} ${big_deployed:>13,.0f} ${small_deployed:>13,.0f}")

    big_lost = sum(s.total_principal_lost for s in big_scores)
    small_lost = sum(s.total_principal_lost for s in small_scores)
    print(f"  {'Total Principal Lost ($)':<30s} ${big_lost:>13,.0f} ${small_lost:>13,.0f}")

    big_violations = sum(len(s.concentration_violations) for s in big_scores)
    small_violations = sum(len(s.concentration_violations) for s in small_scores)
    print(f"  {'Concentration Violations':<30s} {big_violations:>15d} {small_violations:>15d}")

    print(f"\n  {'─'*60}")
    print(f"\n  Per-Lender Breakdown:")
    print(f"\n  {'Lender Slot':<18s} {'Big Model Score':>15s} {'Small Model Score':>17s} {'Delta':>10s}")
    print(f"  {'─'*60}")
    for bs, ss in zip(big_scores, small_scores):
        delta = bs.final_adjusted_score - ss.final_adjusted_score
        sign = "+" if delta >= 0 else ""
        print(f"  {bs.lender_name.split('[')[0].strip():<18s} "
              f"{bs.final_adjusted_score:>14.2f}% "
              f"{ss.final_adjusted_score:>16.2f}% "
              f"{sign}{delta:>8.2f}%")

    print(f"\n{'*'*70}")
    big_avg = sum(s.final_adjusted_score for s in big_scores) / len(big_scores)
    small_avg = sum(s.final_adjusted_score for s in small_scores) / len(small_scores)
    if big_avg > small_avg:
        print(f"  BIG MODELS WIN by {big_avg - small_avg:.2f} percentage points")
    elif small_avg > big_avg:
        print(f"  SMALL MODELS WIN by {small_avg - big_avg:.2f} percentage points")
    else:
        print(f"  TIE!")
    print(f"  Big: {big_avg:.2f}% avg | Small: {small_avg:.2f}% avg")
    print(f"  Big frauds funded: {big_fraud} | Small frauds funded: {small_fraud}")
    print(f"{'*'*70}\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Loanville — The LLM Lending Simulator")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM responses (no API key needed)")
    parser.add_argument("--compare", action="store_true",
                        help="Run big-vs-small model comparison (uses mock mode)")
    args = parser.parse_args()

    if args.compare:
        print("=" * 70)
        print("  LOANVILLE — MODEL SIZE COMPARISON")
        print("=" * 70)
        run_compare()
        print("Comparison complete.\n")
        return

    mock = args.mock
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not mock and not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        print("Set it in a .env file or export it directly:")
        print("  export OPENROUTER_API_KEY=your-key-here")
        print("\nOr run with --mock for offline simulation:")
        print("  python -m loanville --mock")
        sys.exit(1)

    print("=" * 70)
    print("  LOANVILLE — THE LLM LENDING SIMULATOR")
    if mock:
        print("  [MOCK MODE]")
    print("=" * 70)

    borrowers = get_borrowers()
    lenders = get_lenders()

    print(f"\nLoaded {len(borrowers)} borrower applications")
    print(f"Loaded {len(lenders)} competing lenders:\n")
    for l in lenders:
        deployed = sum(x.remaining_balance for x in l.existing_portfolio)
        print(f"  {l.name} ({l.model})")
        print(f"    Capital: ${l.total_capital:,.0f} | "
              f"Deployed: ${deployed:,.0f} | "
              f"Target Yield: {l.target_yield_pct}%")

    _run_single(borrowers, lenders, api_key, mock=mock)

    print("\nSimulation complete.\n")


if __name__ == "__main__":
    main()
