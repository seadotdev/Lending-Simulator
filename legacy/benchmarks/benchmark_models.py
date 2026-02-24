#!/usr/bin/env python3
"""
Model benchmark: run the Loanville lending simulation across all OpenRouter
models up to Claude 3.5 Haiku cost, then produce a Pareto cost-vs-score curve.

Usage:
  python benchmark_models.py                     # Run full benchmark
  python benchmark_models.py --quick             # Subset of ~10 diverse models
  python benchmark_models.py --resume results.json  # Resume from partial results
  python benchmark_models.py --plot results.json    # Just plot existing results
  python benchmark_models.py --list              # List all benchmark models + pricing
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from loanville.data import get_borrowers, get_lenders
from loanville.engine import SimulationEngine
from loanville.llm import (
    MODEL_PRICING,
    clear_usage,
    get_cost_summary,
    get_token_usage,
)
from loanville.scoring import score_lenders

# ---------------------------------------------------------------------------
# Benchmark model list — curated for tool-use support on OpenRouter
# Sorted roughly by expected capability (expensive → cheap)
# ---------------------------------------------------------------------------

BENCHMARK_MODELS = [
    # --- Tier 1: Near-Haiku cost, strong models ---
    ("anthropic/claude-3.5-haiku",                   "Claude 3.5 Haiku"),
    ("google/gemini-3-flash-preview",                "Gemini 3 Flash"),
    ("google/gemini-2.5-flash",                      "Gemini 2.5 Flash"),
    ("deepseek/deepseek-r1",                         "DeepSeek R1"),
    ("moonshotai/kimi-k2",                           "Kimi K2"),
    ("mistralai/mistral-medium-3.1",                 "Mistral Medium 3.1"),
    ("openai/gpt-5-mini",                            "GPT-5 Mini"),
    ("qwen/qwen3-235b-a22b",                         "Qwen3-235B"),
    ("deepseek/deepseek-r1-0528",                    "DeepSeek R1 0528"),
    ("z-ai/glm-4.7",                                 "GLM-4.7"),
    ("openai/gpt-4.1-mini",                          "GPT-4.1 Mini"),
    # --- Tier 2: Mid-range ---
    ("minimax/minimax-m2.5",                          "MiniMax M2.5"),
    ("qwen/qwen3.5-397b-a17b",                       "Qwen3.5-397B"),
    ("inception/mercury",                             "Mercury"),
    ("qwen/qwen3-coder",                              "Qwen3 Coder"),
    ("deepseek/deepseek-chat-v3-0324",               "DeepSeek V3 0324"),
    ("deepseek/deepseek-chat-v3.1",                  "DeepSeek V3.1"),
    # --- Tier 3: Budget ---
    ("openai/gpt-4o-mini",                           "GPT-4o Mini"),
    ("meta-llama/llama-4-maverick",                  "Llama 4 Maverick"),
    ("x-ai/grok-4-fast",                              "Grok 4 Fast"),
    ("qwen/qwq-32b",                                  "QwQ-32B"),
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5",    "Nemotron Super 49B"),
    ("google/gemini-2.0-flash-001",                  "Gemini 2.0 Flash"),
    ("openai/gpt-4.1-nano",                          "GPT-4.1 Nano"),
    ("openai/gpt-5-nano",                            "GPT-5 Nano"),
    ("meta-llama/llama-3.3-70b-instruct",            "Llama 3.3 70B"),
    ("meta-llama/llama-4-scout",                     "Llama 4 Scout"),
    ("qwen/qwen3-30b-a3b",                           "Qwen3-30B"),
    # --- Tier 4: Cheapest ---
    ("mistralai/mistral-small-3.2-24b-instruct",    "Mistral Small 3.2"),
    ("google/gemma-3-27b-it",                        "Gemma 3 27B"),
    ("amazon/nova-micro-v1",                          "Nova Micro"),
    ("qwen/qwen-2.5-7b-instruct",                   "Qwen 2.5 7B"),
    ("mistralai/mistral-nemo",                       "Mistral Nemo"),
    ("meta-llama/llama-3.1-8b-instruct",             "Llama 3.1 8B"),
    ("meta-llama/llama-3-8b-instruct",               "Llama 3 8B"),
]

QUICK_MODELS = [
    # A fast subset spanning the full price range
    ("anthropic/claude-3.5-haiku",                   "Claude 3.5 Haiku"),
    ("google/gemini-2.5-flash",                      "Gemini 2.5 Flash"),
    ("openai/gpt-4.1-mini",                          "GPT-4.1 Mini"),
    ("deepseek/deepseek-chat-v3-0324",               "DeepSeek V3 0324"),
    ("qwen/qwen3.5-397b-a17b",                       "Qwen3.5-397B"),
    ("openai/gpt-4o-mini",                           "GPT-4o Mini"),
    ("meta-llama/llama-3.3-70b-instruct",            "Llama 3.3 70B"),
    ("qwen/qwen3-30b-a3b",                           "Qwen3-30B"),
    ("google/gemma-3-27b-it",                        "Gemma 3 27B"),
    ("meta-llama/llama-3.1-8b-instruct",             "Llama 3.1 8B"),
]

SMALL_MODELS = [
    # Reference (strong + cheap) for calibration
    ("deepseek/deepseek-chat-v3-0324",               "DeepSeek V3 0324"),
    ("google/gemini-2.5-flash",                      "Gemini 2.5 Flash"),
    # Mid-range
    ("openai/gpt-4o-mini",                           "GPT-4o Mini"),
    ("openai/gpt-4.1-nano",                          "GPT-4.1 Nano"),
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5",    "Nemotron Super 49B"),
    ("meta-llama/llama-3.3-70b-instruct",            "Llama 3.3 70B"),
    ("qwen/qwq-32b",                                 "QwQ-32B"),
    ("qwen/qwen3-30b-a3b",                           "Qwen3-30B"),
    # Small / budget
    ("mistralai/mistral-small-3.2-24b-instruct",    "Mistral Small 3.2"),
    ("google/gemma-3-27b-it",                        "Gemma 3 27B"),
    ("mistralai/mistral-nemo",                       "Mistral Nemo"),
    ("meta-llama/llama-3.1-8b-instruct",             "Llama 3.1 8B"),
    ("qwen/qwen-2.5-7b-instruct",                   "Qwen 2.5 7B"),
    ("meta-llama/llama-3-8b-instruct",               "Llama 3 8B"),
]

MIX = "easy"


def _estimate_cost_per_eval(model_id: str) -> float:
    """Estimate cost for one borrower evaluation based on pricing.

    Assumes ~4K prompt tokens + ~500 completion tokens per evaluation
    (based on observed usage from previous runs).
    """
    in_price, out_price = MODEL_PRICING.get(model_id, (1.0, 3.0))
    prompt_tokens = 4000
    completion_tokens = 500
    return (prompt_tokens / 1_000_000 * in_price +
            completion_tokens / 1_000_000 * out_price)


def run_model(model_id: str, api_key: str) -> dict:
    """Run a single simulation with the given model in the Meridian Partners slot."""
    borrowers = get_borrowers(MIX)
    lenders = get_lenders()

    # Override Meridian Partners (slot 2) with the test model
    lenders[2].model = model_id
    short = model_id.split("/")[-1]
    lenders[2].name = f"Meridian Partners [{short}]"

    clear_usage()
    engine = SimulationEngine(borrowers, lenders, api_key, mock=False)
    asyncio.run(engine.run())

    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
        borrowers=borrowers,
    )

    # Extract Meridian Partners score
    mp = next(s for s in scores if s.lender_id == lenders[2].id)
    costs = get_cost_summary()
    model_cost = costs.get(model_id, 0.0)
    usage = get_token_usage().get(model_id, {"prompt": 0, "completion": 0})

    # Count decisions
    mp_decisions = engine.all_decisions.get(lenders[2].id, [])
    approvals = sum(1 for d in mp_decisions if d.decision == "APPROVE")
    rejections = sum(1 for d in mp_decisions if d.decision == "REJECT")

    # Compute estimated cost-per-eval from actual usage
    n_evals = len(mp_decisions) or 1
    actual_cost_per_eval = model_cost / n_evals

    return {
        "model": model_id,
        "score": mp.final_adjusted_score,
        "perfect": mp.perfect_score,
        "approvals": approvals,
        "rejections": rejections,
        "deals_won": mp.deals_won,
        "frauds_funded": mp.frauds_funded,
        "defaults": mp.defaults_count,
        "deployed": mp.total_deployed,
        "net_pnl": mp.net_return,
        "total_cost": model_cost,
        "cost_per_eval": actual_cost_per_eval,
        "prompt_tokens": usage["prompt"],
        "completion_tokens": usage["completion"],
        "pricing": list(MODEL_PRICING.get(model_id, (1.0, 3.0))),
    }


def run_benchmark(models: list[tuple[str, str]], api_key: str,
                   resume_from: dict | None = None) -> list[dict]:
    """Run the full benchmark across all models."""
    results = list(resume_from.get("results", [])) if resume_from else []
    done_models = {r["model"] for r in results}

    remaining = [(mid, name) for mid, name in models if mid not in done_models]

    if done_models:
        print(f"\nResuming: {len(done_models)} already done, {len(remaining)} remaining\n")

    for i, (model_id, display_name) in enumerate(remaining, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(remaining)}] {display_name}")
        print(f"  Model: {model_id}")
        est = _estimate_cost_per_eval(model_id)
        n_borrowers = len(get_borrowers(MIX))
        print(f"  Est. cost/eval: ${est:.6f} (x{n_borrowers} borrowers = ${est*n_borrowers:.4f})")
        print(f"{'='*70}")

        try:
            t0 = time.time()
            r = run_model(model_id, api_key)
            elapsed = time.time() - t0
            r["elapsed_seconds"] = round(elapsed, 1)
            r["display_name"] = display_name
            results.append(r)

            print(f"\n  Score={r['score']:+.2f}% | "
                  f"Approved={r['approvals']} | Won={r['deals_won']} | "
                  f"Frauds={r['frauds_funded']} | Defaults={r['defaults']} | "
                  f"Cost=${r['total_cost']:.4f} | "
                  f"Time={elapsed:.0f}s")

        except Exception as e:
            print(f"\n  FAILED: {type(e).__name__}: {e}")
            results.append({
                "model": model_id,
                "display_name": display_name,
                "score": None,
                "error": str(e),
            })

        # Save incremental results after each model
        _save_results(results)

    return results


def _save_results(results: list[dict], filename: str = "benchmark_results.json"):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "mix": MIX,
        "n_models": len(results),
        "results": results,
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [saved to {filename}]")


# ---------------------------------------------------------------------------
# Pareto analysis and visualization
# ---------------------------------------------------------------------------

def compute_pareto_frontier(results: list[dict]) -> list[dict]:
    """Compute the Pareto frontier: models not dominated on cost vs score.

    A model is Pareto-optimal if no other model is both cheaper AND
    higher-scoring.
    """
    # Filter to successful results only
    valid = [r for r in results if r.get("score") is not None and r.get("total_cost") is not None]
    if not valid:
        return []

    # Sort by cost ascending
    valid.sort(key=lambda r: r["total_cost"])

    frontier = []
    best_score = float("-inf")

    for r in valid:
        if r["score"] >= best_score:
            frontier.append(r)
            best_score = r["score"]

    return frontier


def print_results_table(results: list[dict]):
    """Print a formatted results table sorted by score."""
    valid = [r for r in results if r.get("score") is not None]
    valid.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*110}")
    print(f"  BENCHMARK RESULTS — {len(valid)} models on '{MIX}' mix")
    print(f"{'='*110}")
    print(f"\n  {'#':>3s}  {'Model':<40s} {'Score':>8s} {'Appr':>5s} "
          f"{'Won':>4s} {'Fraud':>6s} {'Dflt':>5s} {'Net P&L':>12s} "
          f"{'Cost':>8s} {'$/eval':>8s}")
    print(f"  {'─'*106}")

    for i, r in enumerate(valid, 1):
        score_str = f"{r['score']:+.2f}%"
        cost_str = f"${r['total_cost']:.4f}" if r.get('total_cost') else "N/A"
        cpe_str = f"${r['cost_per_eval']:.5f}" if r.get('cost_per_eval') else "N/A"
        pnl_str = f"${r.get('net_pnl', 0):>10,.0f}"
        print(f"  {i:>3d}  {r.get('display_name', r['model']):<40s} "
              f"{score_str:>8s} "
              f"{r.get('approvals', 0):>5d} {r.get('deals_won', 0):>4d} "
              f"{r.get('frauds_funded', 0):>6d} {r.get('defaults', 0):>5d} "
              f"{pnl_str:>12s} {cost_str:>8s} {cpe_str:>8s}")

    # Show errors
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n  Failed models:")
        for r in errors:
            print(f"    {r['model']}: {r['error']}")

    # Perfect score reference
    perfects = [r for r in valid if r.get("perfect") is not None]
    if perfects:
        print(f"\n  Perfect score (theoretical max): {perfects[0]['perfect']:+.2f}%")

    print(f"\n{'='*110}")


def print_pareto_frontier(results: list[dict]):
    """Print the Pareto frontier analysis."""
    frontier = compute_pareto_frontier(results)
    if not frontier:
        print("\n  No valid results for Pareto analysis.")
        return

    valid = [r for r in results if r.get("score") is not None and r.get("total_cost") is not None]

    print(f"\n{'='*90}")
    print(f"  PARETO COST-PERFORMANCE FRONTIER")
    print(f"  (Models where no other model is both cheaper AND higher-scoring)")
    print(f"{'='*90}")
    print(f"\n  {'Model':<40s} {'Score':>8s} {'Cost':>8s} {'$/eval':>8s} {'Fraud':>6s}")
    print(f"  {'─'*74}")

    for r in frontier:
        cost_str = f"${r['total_cost']:.4f}"
        cpe_str = f"${r.get('cost_per_eval', 0):.5f}"
        print(f"  {r.get('display_name', r['model']):<40s} "
              f"{r['score']:>+7.2f}% {cost_str:>8s} {cpe_str:>8s} "
              f"{r.get('frauds_funded', '?'):>6}")

    # Dominated models
    frontier_ids = {r["model"] for r in frontier}
    dominated = [r for r in valid if r["model"] not in frontier_ids]
    if dominated:
        print(f"\n  Dominated models ({len(dominated)}):")
        dominated.sort(key=lambda r: r["total_cost"])
        for r in dominated:
            # Find which frontier model dominates it
            dominators = [f for f in frontier
                         if f["total_cost"] <= r["total_cost"] and f["score"] >= r["score"]]
            dom_names = ", ".join(f.get("display_name", f["model"]) for f in dominators[:2])
            print(f"    {r.get('display_name', r['model']):<35s} "
                  f"score={r['score']:+.2f}% cost=${r['total_cost']:.4f}"
                  f"  (dominated by: {dom_names})")

    print(f"\n{'='*90}")


def plot_pareto(results: list[dict], output_file: str = "benchmark_pareto.png"):
    """Generate a Pareto cost-vs-score scatter plot with frontier line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping plot generation")
        return

    valid = [r for r in results if r.get("score") is not None and r.get("total_cost") is not None]
    if not valid:
        print("  No valid results to plot.")
        return

    frontier = compute_pareto_frontier(results)
    frontier_ids = {r["model"] for r in frontier}

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Plot all points
    costs = [r["total_cost"] for r in valid]
    scores = [r["score"] for r in valid]
    names = [r.get("display_name", r["model"].split("/")[-1]) for r in valid]
    is_frontier = [r["model"] in frontier_ids for r in valid]

    # Color by frontier membership
    colors = ["#2196F3" if f else "#BDBDBD" for f in is_frontier]
    sizes = [120 if f else 60 for f in is_frontier]
    zorders = [10 if f else 5 for f in is_frontier]

    for cost, score, name, color, size, zorder, on_frontier in zip(
        costs, scores, names, colors, sizes, zorders, is_frontier
    ):
        ax.scatter(cost, score, c=color, s=size, zorder=zorder,
                  edgecolors="white", linewidth=0.5)
        # Label all points, bold for frontier
        weight = "bold" if on_frontier else "normal"
        fontsize = 8 if on_frontier else 6.5
        ax.annotate(name, (cost, score),
                   textcoords="offset points", xytext=(6, 4),
                   fontsize=fontsize, fontweight=weight,
                   color="#333" if on_frontier else "#888")

    # Draw Pareto frontier line
    if len(frontier) > 1:
        f_costs = [r["total_cost"] for r in frontier]
        f_scores = [r["score"] for r in frontier]
        ax.plot(f_costs, f_scores, 'b--', alpha=0.5, linewidth=1.5,
               label="Pareto Frontier", zorder=8)

    # Formatting
    ax.set_xlabel("Total API Cost ($)", fontsize=12)
    ax.set_ylabel("Loanville Score (%)", fontsize=12)
    ax.set_title("Loanville Benchmark: Cost vs. Performance\n"
                 f"({len(valid)} models, '{MIX}' borrower mix, Meridian Partners slot)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add pricing tier shading
    max_cost = max(costs) * 1.1
    ax.axhline(y=0, color='red', linestyle=':', alpha=0.3, label='Break-even')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {output_file}")

    # Also save a log-scale version
    log_file = output_file.replace(".png", "_log.png")
    ax.set_xscale("log")
    ax.set_xlabel("Total API Cost ($, log scale)", fontsize=12)
    plt.savefig(log_file, dpi=150, bbox_inches="tight")
    print(f"  Log-scale plot saved to {log_file}")
    plt.close()


def plot_cost_per_eval_pareto(results: list[dict],
                               output_file: str = "benchmark_cost_per_eval.png"):
    """Generate a Pareto plot using cost-per-evaluation (normalized)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid = [r for r in results
             if r.get("score") is not None and r.get("cost_per_eval") is not None]
    if not valid:
        return

    # Compute Pareto on cost_per_eval
    valid.sort(key=lambda r: r["cost_per_eval"])
    frontier = []
    best_score = float("-inf")
    for r in valid:
        if r["score"] >= best_score:
            frontier.append(r)
            best_score = r["score"]
    frontier_ids = {r["model"] for r in frontier}

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    for r in valid:
        on_frontier = r["model"] in frontier_ids
        color = "#2196F3" if on_frontier else "#BDBDBD"
        size = 120 if on_frontier else 60
        name = r.get("display_name", r["model"].split("/")[-1])

        ax.scatter(r["cost_per_eval"] * 1000, r["score"],
                  c=color, s=size, edgecolors="white", linewidth=0.5,
                  zorder=10 if on_frontier else 5)
        weight = "bold" if on_frontier else "normal"
        fontsize = 8 if on_frontier else 6.5
        ax.annotate(name, (r["cost_per_eval"] * 1000, r["score"]),
                   textcoords="offset points", xytext=(6, 4),
                   fontsize=fontsize, fontweight=weight,
                   color="#333" if on_frontier else "#888")

    if len(frontier) > 1:
        f_x = [r["cost_per_eval"] * 1000 for r in frontier]
        f_y = [r["score"] for r in frontier]
        ax.plot(f_x, f_y, 'b--', alpha=0.5, linewidth=1.5, label="Pareto Frontier")

    ax.set_xlabel("Cost per Evaluation ($ x 1000 = milli-cents)", fontsize=12)
    ax.set_ylabel("Loanville Score (%)", fontsize=12)
    ax.set_title("Loanville Benchmark: Cost/Eval vs. Performance\n"
                 f"({len(valid)} models, '{MIX}' mix)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Cost-per-eval plot saved to {output_file}")
    plt.close()


def list_models():
    """Print all benchmark models with their pricing."""
    print(f"\n{'='*90}")
    print(f"  BENCHMARK MODEL LIST ({len(BENCHMARK_MODELS)} models)")
    print(f"{'='*90}")
    print(f"\n  {'#':>3s}  {'Model ID':<50s} {'Name':<25s} {'In$/M':>7s} {'Out$/M':>7s} {'Est$/eval':>10s}")
    print(f"  {'─'*106}")

    total_est = 0
    for i, (model_id, name) in enumerate(BENCHMARK_MODELS, 1):
        inp, out = MODEL_PRICING.get(model_id, (1.0, 3.0))
        est = _estimate_cost_per_eval(model_id)
        total_est += est * 12  # 12 borrowers per model
        print(f"  {i:>3d}  {model_id:<50s} {name:<25s} ${inp:>5.2f} ${out:>5.2f} ${est:>.6f}")

    print(f"\n  Estimated total benchmark cost: ${total_est:.2f}")
    print(f"  (12 borrower evaluations per model)")
    print(f"\n{'='*90}")


def main():
    parser = argparse.ArgumentParser(
        description="Loanville multi-model benchmark with Pareto analysis"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run quick subset (~10 models)")
    parser.add_argument("--small", action="store_true",
                        help="Run small/budget model subset (~14 models)")
    parser.add_argument("--mix", type=str, default="easy",
                        help="Borrower mix preset (easy, balanced, hard, all, analyst)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous results JSON file")
    parser.add_argument("--plot", type=str, default=None,
                        help="Just plot existing results (no new runs)")
    parser.add_argument("--list", action="store_true",
                        help="List all benchmark models and exit")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for results (default: benchmark_<mix>_results.json)")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # Set mix (module-level MIX is used by run_model for borrower loading)
    global MIX
    MIX = args.mix

    # Default output filename based on mix
    if args.output is None:
        args.output = f"benchmark_{MIX}_results.json"

    # Plot-only mode
    if args.plot:
        with open(args.plot) as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        print_results_table(results)
        print_pareto_frontier(results)
        plot_pareto(results)
        plot_cost_per_eval_pareto(results)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    if args.small:
        models = SMALL_MODELS
        mode_str = "small"
    elif args.quick:
        models = QUICK_MODELS
        mode_str = "quick"
    else:
        models = BENCHMARK_MODELS
        mode_str = "full"

    # Validate mix
    from loanville.data import MIX_PRESETS
    if MIX not in MIX_PRESETS:
        print(f"ERROR: Unknown mix '{MIX}'. Choose from: {list(MIX_PRESETS)}")
        sys.exit(1)

    n_borrowers = sum(len(v) for v in MIX_PRESETS[MIX].values())

    # Resume support
    resume_data = None
    if args.resume:
        with open(args.resume) as f:
            resume_data = json.load(f)
        print(f"Loaded {len(resume_data.get('results', []))} previous results from {args.resume}")

    print("=" * 70)
    print("  LOANVILLE MODEL BENCHMARK")
    print(f"  Models: {len(models)} | Mix: {MIX} ({n_borrowers} borrowers) | Mode: {mode_str}")
    print("=" * 70)

    total_est = sum(_estimate_cost_per_eval(m) * n_borrowers for m, _ in models)
    already = len(resume_data.get("results", [])) if resume_data else 0
    print(f"\n  Estimated total cost: ~${total_est:.2f} (full run)")
    if already:
        print(f"  Already completed: {already} models")
    print()

    results = run_benchmark(models, api_key, resume_from=resume_data)

    # Final output
    _save_results(results, args.output)
    print_results_table(results)
    print_pareto_frontier(results)
    plot_pareto(results, args.output.replace(".json", "_pareto.png"))
    plot_cost_per_eval_pareto(results, args.output.replace(".json", "_cost_per_eval.png"))

    print(f"\nBenchmark complete. {len(results)} models tested.")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
