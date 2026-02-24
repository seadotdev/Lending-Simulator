#!/usr/bin/env python3
"""
Quick capability test: run progressively smaller models as the Meridian
Partners lender and see where tool-use / financial reasoning collapses.

Usage:
  python test_models.py                  # Test all models in the list
  python test_models.py model/id         # Test a single model
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from loanville.data import get_borrowers, get_lenders
from loanville.engine import SimulationEngine
from loanville.llm import clear_usage, get_cost_summary, get_token_usage
from loanville.scoring import score_lenders

# Models to test, from largest to smallest.
# All verified: tool-use support on OpenRouter.
TEST_MODELS = [
    "deepseek/deepseek-chat-v3-0324",                # ~685B MoE — known good
    "meta-llama/llama-3.3-70b-instruct",              # 70B dense
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",       # 49B dense
    "qwen/qwen3-30b-a3b",                            # 30B MoE (3B active)
    "meta-llama/llama-3.1-8b-instruct",               # 8B dense
    "qwen/qwen-2.5-7b-instruct",                     # 7B dense
]

MIX = "easy"
api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)


def run_model(model_id: str) -> dict:
    """Run a single sim with the given model in the Meridian Partners slot."""
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

    # Count approvals/rejections from decisions
    mp_decisions = engine.all_decisions.get(lenders[2].id, [])
    approvals = sum(1 for d in mp_decisions if d.decision == "APPROVE")
    rejections = sum(1 for d in mp_decisions if d.decision == "REJECT")

    return {
        "model": model_id,
        "score": mp.final_adjusted_score,
        "perfect": mp.perfect_score,
        "approvals": approvals,
        "rejections": rejections,
        "deals_won": mp.deals_won,
        "frauds": mp.frauds_funded,
        "defaults": mp.defaults_count,
        "deployed": mp.total_deployed,
        "net_pnl": mp.net_return,
        "cost": model_cost,
    }


def main():
    models = TEST_MODELS
    if len(sys.argv) > 1:
        models = [sys.argv[1]]

    results = []
    for model in models:
        print(f"\n{'='*70}")
        print(f"  TESTING: {model}")
        print(f"{'='*70}")
        try:
            r = run_model(model)
            results.append(r)
            print(f"\n  Result: Score={r['score']:+.2f}% | "
                  f"Approved={r['approvals']} | Won={r['deals_won']} | "
                  f"Frauds={r['frauds']} | Defaults={r['defaults']} | "
                  f"Cost=${r['cost']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "model": model, "score": None, "perfect": None,
                "approvals": 0, "rejections": 0, "deals_won": 0,
                "frauds": 0, "defaults": 0, "deployed": 0,
                "net_pnl": 0, "cost": 0, "error": str(e),
            })

    # Summary table
    if len(results) > 1:
        print(f"\n\n{'='*90}")
        print(f"  MODEL CAPABILITY SUMMARY")
        print(f"{'='*90}")
        print(f"\n  {'Model':<45s} {'Score':>8s} {'Appr':>5s} {'Won':>4s} "
              f"{'Fraud':>6s} {'Dflt':>5s} {'Net P&L':>12s} {'Cost':>8s}")
        print(f"  {'─'*86}")

        for r in results:
            if r.get("error"):
                print(f"  {r['model']:<45s} {'ERROR':>8s}")
                continue
            score_str = f"{r['score']:+.2f}%" if r['score'] is not None else "N/A"
            print(f"  {r['model']:<45s} {score_str:>8s} "
                  f"{r['approvals']:>5d} {r['deals_won']:>4d} "
                  f"{r['frauds']:>6d} {r['defaults']:>5d} "
                  f"${r['net_pnl']:>10,.0f} ${r['cost']:>.4f}")

        # Perfect score reference
        if results and results[0].get("perfect") is not None:
            print(f"\n  Perfect score (theoretical max): {results[0]['perfect']:+.2f}%")

        print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
