"""
Entry point for: python -m loanville

Usage:
  python -m loanville                    # Live mode, easy mix (default)
  python -m loanville --mock             # Mock mode (no API key needed)
  python -m loanville --underwriting-backend los --los-base-url http://localhost:3000
  python -m loanville --mix hard         # Adversarial stress test
  python -m loanville --mock --seed 42   # Deterministic borrower ordering
  python -m loanville --compare          # Compare big vs small models (mock)
  python -m loanville --rotate           # Rotate models across lender roles (live)
"""

import argparse
import asyncio
import copy
import os
import sys

from dotenv import load_dotenv

from .data import get_borrowers, get_lenders, MIX_PRESETS
from .engine import SimulationEngine
from .scoring import print_final_report, score_lenders


# ---------------------------------------------------------------------------
# Model rotation pools — cheap frontier models with strong tool-use support
# ---------------------------------------------------------------------------

# All models verified: tool-use support on OpenRouter
ROTATION_POOL = [
    "deepseek/deepseek-chat-v3-0324",
    "qwen/qwen3-235b-a22b",
    "meta-llama/llama-3.3-70b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "qwen/qwen3-30b-a3b",
    "meta-llama/llama-3.1-8b-instruct",
]


def _print_cost_summary() -> None:
    """Print token usage and estimated cost per model."""
    from .llm import get_cost_summary, get_token_usage

    usage = get_token_usage()
    costs = get_cost_summary()
    if not usage:
        return

    print(f"\n{'─'*60}")
    print(f"  API COST ESTIMATE")
    print(f"{'─'*60}")
    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    for model in sorted(usage.keys()):
        u = usage[model]
        c = costs.get(model, 0.0)
        total_cost += c
        total_prompt += u["prompt"]
        total_completion += u["completion"]
        print(f"  {model}")
        print(f"    Tokens: {u['prompt']:>10,} prompt + {u['completion']:>10,} completion")
        print(f"    Est. cost: ${c:.4f}")
    print(f"  {'─'*56}")
    print(f"  TOTAL: {total_prompt:>10,} prompt + {total_completion:>10,} completion")
    print(f"  TOTAL COST: ${total_cost:.4f}")

    # Project cost for larger pools
    n_borrowers = sum(1 for _ in usage)  # rough proxy
    per_call = total_cost / max(1, total_prompt + total_completion) * (total_prompt + total_completion)
    calls_made = sum(u["prompt"] > 0 for u in usage.values())
    if total_cost > 0:
        avg_per_eval = total_cost / max(1, sum(u["prompt"] > 0 for u in usage.values()))
        print(f"\n  Projections (at current avg cost per evaluation):")
        for pool_size in [25, 50, 100, 200]:
            for n_lenders in [3, 5]:
                projected = avg_per_eval * pool_size * n_lenders
                print(f"    {pool_size} borrowers x {n_lenders} lenders = "
                      f"~${projected:.2f}")


def _run_single(
    borrowers,
    lenders,
    api_key="",
    mock=False,
    data_mode="full",
    underwriting_backend="openrouter",
    los_base_url="http://localhost:3000",
    los_timeout_s=20.0,
    los_tenant_id="loanville-sim",
    los_provider="openrouter",
    los_mode="rules_only",
):
    """Run a single simulation and return scores."""
    engine = SimulationEngine(
        borrowers=borrowers,
        lenders=lenders,
        openrouter_api_key=api_key,
        mock=mock,
        data_mode=data_mode,
        underwriting_backend=underwriting_backend,
        los_base_url=los_base_url,
        los_timeout_s=los_timeout_s,
        los_tenant_id=los_tenant_id,
        los_provider=los_provider,
        los_mode=los_mode,
    )
    asyncio.run(engine.run())

    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
        borrowers=borrowers,
    )
    print_final_report(scores)
    return scores


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
    big_scores = _run_single(borrowers, big_lenders, mock=True, data_mode="full")

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
    small_scores = _run_single(borrowers, small_lenders, mock=True, data_mode="full")

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


def run_rotate(api_key: str, mix: str, rounds: int = 3):
    """Run multiple rounds, rotating which model plays which lender role."""
    borrowers = get_borrowers(mix)
    n_models = len(ROTATION_POOL)
    all_round_scores = []

    for r in range(rounds):
        print(f"\n{'#'*70}")
        print(f"#  ROTATION ROUND {r+1}/{rounds}")
        print(f"{'#'*70}")

        lenders = get_lenders()
        for i, lender in enumerate(lenders):
            model_idx = (i + r) % n_models
            model = ROTATION_POOL[model_idx]
            lender.model = model
            short_name = model.split("/")[-1]
            base_name = lender.name.split("[")[0].strip()
            lender.name = f"{base_name} [{short_name}]"
            print(f"  {base_name} -> {model}")

        scores = _run_single(borrowers, lenders, api_key, mock=False, data_mode="full")
        all_round_scores.append(scores)

    # Summary across rounds
    print(f"\n\n{'='*70}")
    print(f"  ROTATION SUMMARY ({rounds} rounds)")
    print(f"{'='*70}")

    model_stats: dict[str, list[float]] = {}
    model_frauds: dict[str, int] = {}
    model_defaults: dict[str, int] = {}
    for round_scores in all_round_scores:
        for s in round_scores:
            model = s.model
            model_stats.setdefault(model, []).append(s.final_adjusted_score)
            model_frauds[model] = model_frauds.get(model, 0) + s.frauds_funded
            model_defaults[model] = model_defaults.get(model, 0) + s.defaults_count

    print(f"\n  {'Model':<40s} {'Avg Score':>10s} {'Frauds':>8s} {'Defaults':>10s} {'Rounds':>8s}")
    print(f"  {'─'*76}")
    for model in sorted(model_stats.keys()):
        scores_list = model_stats[model]
        avg = sum(scores_list) / len(scores_list)
        print(f"  {model:<40s} {avg:>9.2f}% {model_frauds[model]:>8d} "
              f"{model_defaults[model]:>10d} {len(scores_list):>8d}")

    print(f"\n{'*'*70}")
    best_model = max(model_stats.keys(), key=lambda m: sum(model_stats[m]) / len(model_stats[m]))
    best_avg = sum(model_stats[best_model]) / len(model_stats[best_model])
    print(f"  BEST MODEL: {best_model}")
    print(f"  Average Score: {best_avg:.2f}%")
    print(f"{'*'*70}")

    _print_cost_summary()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Loanville — The LLM Lending Simulator")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM responses (no API key needed)")
    parser.add_argument("--compare", action="store_true",
                        help="Run frontier-vs-small model comparison (uses mock mode)")
    parser.add_argument("--rotate", action="store_true",
                        help="Run rotation: different models in different roles (live)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of rotation rounds (default: 3)")
    parser.add_argument("--mix", choices=list(MIX_PRESETS.keys()), default="easy",
                        help="Borrower population mix (default: easy)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Deterministic seed for borrower ordering/sampling")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Optional borrower sample size after deterministic ordering")
    parser.add_argument("--data-mode",
                        choices=["full", "quarterly_only", "aggregate_only", "statements_inline", "lite"],
                        default="full",
                        help="Financial data presentation mode (default: full). "
                             "'lite' uses compact prompts optimized for small models (3B-30B).")
    parser.add_argument(
        "--underwriting-backend",
        choices=["openrouter", "los"],
        default=os.environ.get("LOANVILLE_UNDERWRITING_BACKEND", "openrouter"),
        help="Underwriting backend in live mode (default: openrouter).",
    )
    parser.add_argument(
        "--los-base-url",
        default=os.environ.get("LOS_BASE_URL", "http://localhost:3000"),
        help="Base URL for LOS API when --underwriting-backend los is selected.",
    )
    parser.add_argument(
        "--los-timeout-s",
        type=float,
        default=float(os.environ.get("LOS_TIMEOUT_S", "20")),
        help="Request timeout in seconds for LOS API calls.",
    )
    parser.add_argument(
        "--los-tenant-id",
        default=os.environ.get("LOS_TENANT_ID", "loanville-sim"),
        help="Tenant ID header for LOS API calls.",
    )
    parser.add_argument(
        "--los-provider",
        default=os.environ.get("LOS_PROVIDER", "openrouter"),
        choices=["openrouter", "anthropic"],
        help="Provider field forwarded to LOS /evaluate.",
    )
    parser.add_argument(
        "--los-mode",
        default=os.environ.get("LOS_MODE", "rules_only"),
        choices=["full", "rules_only"],
        help="Evaluation mode forwarded to LOS /evaluate.",
    )
    args = parser.parse_args()

    if args.compare:
        print("=" * 70)
        print("  LOANVILLE — MODEL SIZE COMPARISON")
        print("=" * 70)
        run_compare(args.mix)
        print("Comparison complete.\n")
        return

    mock = args.mock
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    backend = args.underwriting_backend

    if not mock and backend == "openrouter" and not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        print("Set it in a .env file or export it directly:")
        print("  export OPENROUTER_API_KEY=your-key-here")
        print("\nOr run with --mock for offline simulation:")
        print("  python -m loanville --mock")
        sys.exit(1)

    if args.rotate:
        if backend != "openrouter":
            print("ERROR: --rotate only supports --underwriting-backend openrouter.")
            sys.exit(1)
        print("=" * 70)
        print("  LOANVILLE — MODEL ROTATION TOURNAMENT")
        print("=" * 70)
        run_rotate(api_key, args.mix, rounds=args.rounds)
        print("\nRotation complete.\n")
        return

    print("=" * 70)
    print("  LOANVILLE — THE LLM LENDING SIMULATOR")
    if mock:
        print("  [MOCK MODE]")
    elif backend == "los":
        print(f"  [LOS BACKEND] {args.los_base_url}")
        print(f"  [LOS EVALUATE] provider={args.los_provider}, mode={args.los_mode}")
    else:
        print("  [OPENROUTER BACKEND]")
    print("=" * 70)

    borrowers = get_borrowers(args.mix, seed=args.seed, sample_size=args.sample_size)
    lenders = get_lenders()

    print(f"\nLoaded {len(borrowers)} borrower applications")
    if args.seed is not None:
        print(f"Deterministic seed: {args.seed}")
    if args.sample_size is not None:
        print(f"Sample size: {args.sample_size}")
    _print_mix_info(args.mix, borrowers)
    print(f"Loaded {len(lenders)} competing lenders:\n")
    for l in lenders:
        deployed = sum(x.remaining_balance for x in l.existing_portfolio)
        print(f"  {l.name} ({l.model})")
        print(f"    Capital: ${l.total_capital:,.0f} | "
              f"Deployed: ${deployed:,.0f} | "
              f"Target Yield: {l.target_yield_pct}%")

    if not mock and backend == "openrouter":
        from .llm import clear_usage  # Imported lazily to preserve mock/offline mode.
        clear_usage()
    _run_single(
        borrowers,
        lenders,
        api_key,
        mock=mock,
        data_mode=args.data_mode,
        underwriting_backend=backend,
        los_base_url=args.los_base_url,
        los_timeout_s=args.los_timeout_s,
        los_tenant_id=args.los_tenant_id,
        los_provider=args.los_provider,
        los_mode=args.los_mode,
    )
    if not mock and backend == "openrouter":
        _print_cost_summary()

    print("\nSimulation complete.\n")


if __name__ == "__main__":
    main()
