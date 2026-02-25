"""
Entry point for: python -m loanville

Usage:
  python -m loanville                    # LOS mode, easy mix (default)
  python -m loanville --mock             # Mock mode (no API key needed)
  python -m loanville --mix hard         # Adversarial stress test
  python -m loanville --compare          # Compare big vs small models (mock)
"""

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from .data import get_borrowers, get_lenders, MIX_PRESETS
from .engine import SimulationEngine
from .models import EconomicsConfig, ECONOMICS_PRESETS, SeasonConfig
from .scoring import print_final_report, print_season_report, score_lenders, score_season
from .season import SeasonEngine


def _run_single(borrowers, lenders, mock=False, data_mode="full",
                 los_url="http://localhost:3000",
                 los_provider="openrouter", los_mode="rules_only",
                 underwrite_only=False, los_model=None,
                 economics=None):
    """Run a single simulation and return (scores, engine)."""
    engine = SimulationEngine(
        borrowers, lenders, mock=mock, data_mode=data_mode,
        los_url=los_url,
        los_provider=los_provider, los_mode=los_mode,
        underwrite_only=underwrite_only, los_model=los_model,
        economics=economics,
    )
    asyncio.run(engine.run())

    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
        borrowers=borrowers,
        economics=economics,
        runs=engine.runs,
    )
    print_final_report(scores, economics=economics)
    return scores, engine


def _print_mix_info(mix: str, borrowers) -> None:
    """Print the borrower population breakdown."""
    from collections import Counter
    outcomes = Counter(b.true_outcome for b in borrowers)
    total = len(borrowers)
    parts = []
    for k in ["good", "bad", "fraud"]:
        n = outcomes.get(k, 0)
        parts.append(f"{n} {k} ({n/total*100:.0f}%)")
    print(f"  Pipeline mix: {mix} — {', '.join(parts)}")


def run_compare(mix: str):
    """Run two simulations: big models vs small models, then compare."""
    borrowers = get_borrowers(mix)

    # --- Round 1: Big frontier models (tool-use capable) ---
    big_lenders = get_lenders()
    big_lenders[0].model = "deepseek/deepseek-chat-v3-0324"
    big_lenders[0].name = "Velocity Capital [DeepSeek V3]"
    big_lenders[1].model = "qwen/qwen3-235b-a22b"
    big_lenders[1].name = "Heritage Trust [Qwen3-235B]"
    big_lenders[2].model = "meta-llama/llama-3.3-70b-instruct"
    big_lenders[2].name = "Meridian Partners [Llama-3.3-70B]"

    print("\n" + "#" * 70)
    print("#  ROUND 1: FRONTIER MODELS (tool-use capable)")
    print("#" * 70)
    big_scores, _ = _run_single(borrowers, big_lenders, mock=True, data_mode="full")

    # --- Round 2: Small / mid-tier models ---
    small_lenders = get_lenders()
    small_lenders[0].model = "meta-llama/llama-3.1-8b-instruct"
    small_lenders[0].name = "Velocity Capital [Llama-8B]"
    small_lenders[1].model = "qwen/qwen-2.5-7b-instruct"
    small_lenders[1].name = "Heritage Trust [Qwen-7B]"
    small_lenders[2].model = "qwen/qwen3-30b-a3b"
    small_lenders[2].name = "Meridian Partners [Qwen3-30B]"

    print("\n\n" + "#" * 70)
    print("#  ROUND 2: SMALL MODELS")
    print("#" * 70)
    small_scores, _ = _run_single(borrowers, small_lenders, mock=True, data_mode="full")

    # --- Comparison ---
    print("\n\n" + "=" * 70)
    print("  HEAD-TO-HEAD: FRONTIER vs SMALL MODELS")
    print("=" * 70)

    print(f"\n  {'Metric':<30s} {'Frontier':>15s} {'Small':>15s}")
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
    print(f"\n  {'Lender Slot':<18s} {'Frontier Score':>15s} {'Small Score':>17s} {'Delta':>10s}")
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
        print(f"  FRONTIER MODELS WIN by {big_avg - small_avg:.2f} percentage points")
    elif small_avg > big_avg:
        print(f"  SMALL MODELS WIN by {small_avg - big_avg:.2f} percentage points")
    else:
        print(f"  TIE!")
    print(f"  Frontier: {big_avg:.2f}% avg | Small: {small_avg:.2f}% avg")
    print(f"  Frontier frauds funded: {big_fraud} | Small frauds funded: {small_fraud}")
    print(f"{'*'*70}\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Loanville — The LLM Lending Simulator")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM responses (no API key needed)")
    parser.add_argument("--compare", action="store_true",
                        help="Run frontier-vs-small model comparison (uses mock mode)")
    parser.add_argument("--mix", choices=list(MIX_PRESETS.keys()), default="easy",
                        help="Borrower population mix (default: easy)")
    parser.add_argument("--data-mode",
                        choices=["full", "quarterly_only", "aggregate_only", "statements_inline", "lite"],
                        default="full",
                        help="Financial data presentation mode (default: full). "
                             "'lite' uses compact prompts optimized for small models (3B-30B).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deterministic borrower ordering")
    parser.add_argument("--los-url", default="http://localhost:3000",
                        help="Open LOS API URL (default: http://localhost:3000)")
    parser.add_argument("--los-provider", default="anthropic",
                        choices=["openrouter", "anthropic"],
                        help="LLM provider for LOS evaluation (default: anthropic)")
    parser.add_argument("--los-mode", default="rules_only",
                        choices=["full", "rules_only"],
                        help="LOS evaluation mode: 'full' uses LLM agent, "
                             "'rules_only' uses deterministic rules (default: rules_only)")
    parser.add_argument("--underwrite-only", action="store_true",
                        help="Skip LOS pipeline ceremony (entity/deal/docs/spread/stages), "
                             "just POST dossier to /v1/underwrite for standalone LLM evaluation")
    parser.add_argument("--los-model", default=None,
                        help="Override the LLM model used by the LOS for underwriting "
                             "(default: uses each lender's model)")
    parser.add_argument("--economics",
                        choices=list(ECONOMICS_PRESETS.keys()),
                        default="balanced",
                        help="Economics preset: balanced (default), aggressive, conservative")
    # Season mode arguments
    parser.add_argument("--season", action="store_true",
                        help="Run multi-week season mode")
    parser.add_argument("--weeks", type=int, default=10,
                        help="Number of weeks in season (default: 10)")
    parser.add_argument("--cohort-size", type=int, default=5,
                        help="Borrowers per week in season mode (default: 5)")
    parser.add_argument("--season-mix",
                        choices=["gentle", "realistic", "adversarial", "escalating"],
                        default="realistic",
                        help="Season borrower mix (default: realistic)")
    parser.add_argument("--speed-scoring", action="store_true",
                        help="Enable speed-to-offer scoring in season mode")
    parser.add_argument("--custom-tools", action="store_true",
                        help="Enable custom tool creation in season mode")
    parser.add_argument("--months-per-week", type=int, default=2,
                        help="Months of loan aging per season week (default: 2)")
    # Leaderboard
    parser.add_argument("--leaderboard", action="store_true",
                        help="Emit match record to leaderboard after scoring")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable detailed logging (LOS calls, per-borrower progress)")
    parser.add_argument("--json", type=str, default=None, metavar="FILE",
                        help="Export season results to JSON file (for web viewer)")
    args = parser.parse_args()

    if args.season and args.compare:
        parser.error("--season and --compare cannot be used together.")

    # Configure logging — errors always shown, -v adds per-call progress
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("loanville.los_adapter").setLevel(
        logging.INFO if args.verbose else logging.ERROR
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if args.compare:
        print("=" * 70)
        print("  LOANVILLE — MODEL SIZE COMPARISON")
        print("=" * 70)
        run_compare(args.mix)
        print("Comparison complete.\n")
        return

    mock = args.mock

    economics = ECONOMICS_PRESETS[args.economics]

    if args.season:
        season_config = SeasonConfig(
            weeks=args.weeks,
            cohort_size=args.cohort_size,
            months_per_week=args.months_per_week,
            season_mix=args.season_mix,
            seed=args.seed or 42,
            speed_scoring=args.speed_scoring,
            custom_tools=args.custom_tools,
            economics=economics,
        )
        lenders = get_lenders()
        season = SeasonEngine(
            config=season_config,
            lenders=lenders,
            mock=mock,
            data_mode=args.data_mode,
            los_url=args.los_url,
            los_provider=args.los_provider,
            los_mode=args.los_mode,
            underwrite_only=args.underwrite_only,
            los_model=args.los_model,
        )
        asyncio.run(season.run_season())

        # Score and report
        season_scores = score_season(season.lender_states, season_config)
        print_season_report(season_scores)

        # Leaderboard integration for season mode
        if args.leaderboard:
            from .leaderboard import emit_match_record_from_season, emit_and_update
            record = emit_match_record_from_season(
                season_engine=season,
                season_scores=season_scores,
                lenders=lenders,
                mix=args.season_mix,
            )
            match_path, lb_path = emit_and_update(record)
            print(f"\n  Leaderboard: match -> {match_path.name}")
            print(f"  Leaderboard: standings -> {lb_path.name}")

        # JSON export for web viewer
        import json
        from datetime import datetime
        season_json = season.to_json()

        # Save to web/seasons/ with timestamped name and rebuild index
        web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
        seasons_dir = os.path.join(web_dir, "seasons")
        if os.path.isdir(web_dir):
            os.makedirs(seasons_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            lender_names = "-".join(l["name"].split()[0] for l in season_json.get("lenders", [])[:3])
            filename = f"{ts}_{lender_names}.json"
            filepath = os.path.join(seasons_dir, filename)
            with open(filepath, "w") as f:
                json.dump(season_json, f, indent=2)
            # Rebuild index: list all .json files sorted newest first
            all_files = sorted(
                [f for f in os.listdir(seasons_dir) if f.endswith(".json") and f != "index.json"],
                reverse=True,
            )
            with open(os.path.join(seasons_dir, "index.json"), "w") as f:
                json.dump(all_files, f, indent=2)
            print(f"\nSeason data saved to: seasons/{filename}")

        if args.json:
            with open(args.json, "w") as f:
                json.dump(season_json, f, indent=2)
            print(f"Season data also exported to: {args.json}")

        print("\nSeason complete.\n")
        return

    print("=" * 70)
    print("  LOANVILLE — THE LLM LENDING SIMULATOR")
    if mock:
        print("  [MOCK MODE]")
    else:
        uw_flag = " underwrite-only" if args.underwrite_only else ""
        model_flag = f" model={args.los_model}" if args.los_model else ""
        print(f"  [LOS MODE{uw_flag}] → {args.los_url} (mode={args.los_mode}{model_flag})")
    print(f"  Economics: {economics.name}")
    print("=" * 70)

    borrowers = get_borrowers(args.mix, seed=args.seed)
    lenders = get_lenders()

    print(f"\nLoaded {len(borrowers)} borrower applications")
    _print_mix_info(args.mix, borrowers)
    print(f"Loaded {len(lenders)} competing lenders:\n")
    for l in lenders:
        deployed = sum(x.remaining_balance for x in l.existing_portfolio)
        print(f"  {l.name} ({l.model})")
        print(f"    Capital: ${l.total_capital:,.0f} | "
              f"Deployed: ${deployed:,.0f} | "
              f"Target Yield: {l.target_yield_pct}%")

    scores, engine = _run_single(
        borrowers, lenders, mock=mock, data_mode=args.data_mode,
        los_url=args.los_url,
        los_provider=args.los_provider, los_mode=args.los_mode,
        underwrite_only=args.underwrite_only, los_model=args.los_model,
        economics=economics,
    )

    # Emit match record to leaderboard
    if args.leaderboard:
        from .leaderboard import emit_match_record_from_sim, emit_and_update

        models_info = [
            {"model_id": l.model, "display_name": l.name}
            for l in lenders
        ]
        record = emit_match_record_from_sim(
            lenders=lenders,
            models_info=models_info,
            engine=engine,
            scores=scores,
            borrowers=borrowers,
            mix=args.mix,
        )
        match_path, lb_path = emit_and_update(record)
        print(f"\n  Leaderboard: match → {match_path.name}")
        print(f"  Leaderboard: standings → {lb_path.name}")

    print("\nSimulation complete.\n")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
