#!/usr/bin/env python3
"""
Run the lite scenario set against real small models via OpenRouter.

Tests models from 3B to 70B+ to find the difficulty frontier:
- Which cases do even 3B models get right?
- Where does the transition from "can't solve" to "can solve" happen?
- What should we simplify to get smallest models working?

Usage:
  python run_lite_eval.py                    # Run all models
  python run_lite_eval.py --models 3         # Run first 3 models only
  python run_lite_eval.py --model qwen/qwen3-8b  # Run a specific model
"""

import argparse
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

from loanville.data import get_borrowers, get_lenders
from loanville.engine import SimulationEngine
from loanville.llm import get_cost_summary, get_token_usage, clear_usage
from loanville.scoring import score_lenders

load_dotenv()

# Models to test, ordered by approximate size (smallest first)
EVAL_MODELS = [
    # ~3B class
    "meta-llama/llama-3.2-3b-instruct",
    # ~7-9B class
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-8b",
    # ~14B class
    "qwen/qwen3-14b",
    # ~24-32B class
    "google/gemma-3-27b-it",
    "qwen/qwen3-32b",
    "qwen/qwq-32b",
    # ~49-70B class
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "meta-llama/llama-3.3-70b-instruct",
    # Reference: frontier model
    "deepseek/deepseek-chat-v3-0324",
]

# Ground truth for the lite set
CORRECT_ACTION = {
    "BRW-001": "APPROVE",  # SkyFreight - clear good [easy]
    "BRW-004": "APPROVE",  # UrbanGrow - clear good [easy]
    "BRW-009": "REJECT",   # TerraVolt - revenue cliff [easy]
    "BRW-015": "APPROVE",  # Apex Data - strong SaaS [medium]
    "BRW-017": "APPROVE",  # Keystone Legal - gov contracts [medium]
    "BRW-007": "REJECT",   # SynthaCure - margin compression [medium]
    "BRW-024": "REJECT",   # Pacific Rim - thin margins [medium]
    "BRW-006": "REJECT",   # AeroTrack - concentration [hard]
    "BRW-012": "REJECT",   # QubitTech - fabricated [hard]
    "BRW-011": "REJECT",   # BioGenesis - circular [hard]
}

DIFFICULTY = {
    "BRW-001": "easy", "BRW-004": "easy", "BRW-009": "easy",
    "BRW-015": "medium", "BRW-017": "medium", "BRW-007": "medium", "BRW-024": "medium",
    "BRW-006": "hard", "BRW-012": "hard", "BRW-011": "hard",
}

BORROWER_NAMES = {
    "BRW-001": "SkyFreight", "BRW-004": "UrbanGrow", "BRW-009": "TerraVolt",
    "BRW-015": "Apex Data", "BRW-017": "Keystone", "BRW-007": "SynthaCure",
    "BRW-024": "Pacific Rim", "BRW-006": "AeroTrack", "BRW-012": "QubitTech",
    "BRW-011": "BioGenesis",
}


def run_model_eval(model: str, borrowers, api_key: str, data_mode: str = "lite"):
    """Run the lite set with a single model as all 3 lenders."""
    lenders = get_lenders()
    for l in lenders:
        l.model = model

    clear_usage()
    engine = SimulationEngine(
        borrowers, lenders, api_key,
        max_concurrent_per_lender=3,
        data_mode=data_mode,
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

    # Collect per-borrower decisions across all lenders
    per_borrower = {}
    for b in borrowers:
        decisions = []
        for lid, dec_list in engine.all_decisions.items():
            for d in dec_list:
                if d.borrower_id == b.id:
                    decisions.append({
                        "lender": lid,
                        "decision": d.decision,
                        "reasoning": d.reasoning[:200],
                    })
        expected = CORRECT_ACTION[b.id]
        correct_count = sum(1 for d in decisions if d["decision"] == expected)
        per_borrower[b.id] = {
            "name": b.dossier.company_name,
            "expected": expected,
            "outcome": b.true_outcome,
            "difficulty": DIFFICULTY[b.id],
            "correct_count": correct_count,
            "total_count": len(decisions),
            "pass_rate": correct_count / len(decisions) if decisions else 0,
            "decisions": decisions,
        }

    cost = get_cost_summary()
    usage = get_token_usage()

    return {
        "model": model,
        "avg_score": sum(s.final_adjusted_score for s in scores) / len(scores),
        "frauds_funded": sum(s.frauds_funded for s in scores),
        "defaults": sum(s.defaults_count for s in scores),
        "per_borrower": per_borrower,
        "cost": cost,
        "usage": usage,
    }


def print_model_result(result):
    """Print a concise summary for one model."""
    model = result["model"]
    short = model.split("/")[-1]
    print(f"\n  {'─'*66}")
    print(f"  {short}")
    print(f"  {'─'*66}")
    print(f"  Avg Score: {result['avg_score']:.2f}%  |  "
          f"Frauds: {result['frauds_funded']}  |  Defaults: {result['defaults']}")

    # Per-borrower results
    print(f"\n  {'Borrower':<16s} {'Tier':<8s} {'Expected':<9s} {'Pass Rate':>10s} {'Status':<10s}")
    print(f"  {'─'*53}")
    for bid in ["BRW-001", "BRW-004", "BRW-009",
                 "BRW-015", "BRW-017", "BRW-007", "BRW-024",
                 "BRW-006", "BRW-012", "BRW-011"]:
        pb = result["per_borrower"][bid]
        status = "CORRECT" if pb["pass_rate"] > 0.5 else "MISSED"
        marker = "x" if pb["pass_rate"] > 0.5 else " "
        print(f"  [{marker}] {BORROWER_NAMES[bid]:<13s} {pb['difficulty']:<8s} "
              f"{pb['expected']:<9s} {pb['pass_rate']:>9.0%}  {status}")

    # Tier summary
    for tier in ["easy", "medium", "hard"]:
        cases = [pb for pb in result["per_borrower"].values() if pb["difficulty"] == tier]
        avg_pass = sum(c["pass_rate"] for c in cases) / len(cases) if cases else 0
        correct = sum(1 for c in cases if c["pass_rate"] > 0.5)
        print(f"  {tier:>8s}: {correct}/{len(cases)} correct ({avg_pass:.0%} avg pass rate)")

    # Cost
    total_cost = sum(result["cost"].values()) if result["cost"] else 0
    print(f"  Cost: ${total_cost:.4f}")


def print_summary_table(all_results):
    """Print the final comparison table across all models."""
    print(f"\n\n{'='*90}")
    print(f"  LITE BENCHMARK RESULTS — REAL MODEL EVALUATION")
    print(f"{'='*90}")

    # Header
    borrower_ids = ["BRW-001", "BRW-004", "BRW-009",
                    "BRW-015", "BRW-017", "BRW-007", "BRW-024",
                    "BRW-006", "BRW-012", "BRW-011"]
    short_names = [BORROWER_NAMES[b][:6] for b in borrower_ids]

    print(f"\n  {'Model':<35s} {'Score':>6s} ", end="")
    for sn in short_names:
        print(f" {sn:>6s}", end="")
    print(f" {'Easy':>5s} {'Med':>5s} {'Hard':>5s} {'Cost':>7s}")

    print(f"  {'─'*35} {'─'*6} ", end="")
    for _ in short_names:
        print(f" {'─'*6}", end="")
    print(f" {'─'*5} {'─'*5} {'─'*5} {'─'*7}")

    for r in all_results:
        short = r["model"].split("/")[-1][:33]
        print(f"  {short:<35s} {r['avg_score']:>5.1f}%", end=" ")

        for bid in borrower_ids:
            pb = r["per_borrower"][bid]
            if pb["pass_rate"] > 0.5:
                print(f"    {'OK':>4s}", end="")
            else:
                print(f"  {'MISS':>4s}", end="")

        # Tier averages
        for tier in ["easy", "medium", "hard"]:
            cases = [pb for pb in r["per_borrower"].values() if pb["difficulty"] == tier]
            correct = sum(1 for c in cases if c["pass_rate"] > 0.5)
            print(f" {correct}/{len(cases)}", end="")

        total_cost = sum(r["cost"].values()) if r["cost"] else 0
        print(f" ${total_cost:.3f}")

    # Difficulty analysis
    print(f"\n\n  DIFFICULTY ANALYSIS — Cases to simplify for small models:")
    print(f"  {'─'*70}")

    for bid in borrower_ids:
        name = BORROWER_NAMES[bid]
        tier = DIFFICULTY[bid]
        expected = CORRECT_ACTION[bid]

        # Count how many models got it right
        model_correct = 0
        model_total = len(all_results)
        for r in all_results:
            if r["per_borrower"][bid]["pass_rate"] > 0.5:
                model_correct += 1

        pct = model_correct / model_total * 100 if model_total else 0

        if pct < 30:
            suggestion = "<-- SIMPLIFY: most models miss this"
        elif pct < 60:
            suggestion = "<-- borderline"
        else:
            suggestion = ""

        print(f"  {name:<16s} [{tier:6s}] {expected:<8s} "
              f"{model_correct}/{model_total} models correct ({pct:.0f}%) {suggestion}")


def main():
    parser = argparse.ArgumentParser(description="Run lite eval against real models")
    parser.add_argument("--models", type=int, default=0,
                        help="Number of models to test (0 = all)")
    parser.add_argument("--model", type=str, default="",
                        help="Run a specific model")
    parser.add_argument("--data-mode", default="lite",
                        choices=["full", "quarterly_only", "lite"],
                        help="Data mode (default: lite)")
    parser.add_argument("--output", default="lite_eval_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    borrowers = get_borrowers("lite")

    if args.model:
        models = [args.model]
    elif args.models > 0:
        models = EVAL_MODELS[:args.models]
    else:
        models = EVAL_MODELS

    print(f"{'='*70}")
    print(f"  LOANVILLE LITE — REAL MODEL EVALUATION")
    print(f"  Models: {len(models)} | Borrowers: {len(borrowers)} | Mode: {args.data_mode}")
    print(f"{'='*70}")

    all_results = []
    for i, model in enumerate(models):
        print(f"\n{'#'*70}")
        print(f"#  MODEL {i+1}/{len(models)}: {model}")
        print(f"{'#'*70}")

        try:
            start = time.time()
            result = run_model_eval(model, borrowers, api_key, args.data_mode)
            elapsed = time.time() - start
            result["elapsed_seconds"] = round(elapsed, 1)
            print_model_result(result)
            all_results.append(result)

            # Save incrementally
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        except Exception as e:
            print(f"\n  ERROR: {type(e).__name__}: {e}")
            all_results.append({
                "model": model,
                "error": f"{type(e).__name__}: {e}",
            })

    if len(all_results) > 1:
        valid = [r for r in all_results if "error" not in r]
        if valid:
            print_summary_table(valid)

    print(f"\n  Results saved to {args.output}")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
