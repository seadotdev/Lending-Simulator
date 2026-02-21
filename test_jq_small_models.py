#!/usr/bin/env python3
"""
Test small models' ability to use jq for bank statement analysis.

Specifically measures:
  1. Whether models invoke the run_bash tool at all
  2. What jq queries they generate (valid syntax? useful queries?)
  3. Whether jq-based analysis improves fraud detection vs no-tool mode
  4. JSON response reliability (can the model still produce valid output?)

Tests each model against the 3 fraud borrowers (where jq analysis matters most)
and a sample of good/bad borrowers for false-positive calibration.

Usage:
  python test_jq_small_models.py                  # All models, fraud borrowers
  python test_jq_small_models.py --model <id>     # Single model
  python test_jq_small_models.py --full-pool      # All borrowers (slower)
"""

import argparse
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from loanville.data import get_borrowers, get_lenders
from loanville.llm import (
    evaluate_borrower,
    clear_usage,
    get_cost_summary,
    get_token_usage,
    get_call_traces,
    clear_call_traces,
)
from openai import AsyncOpenAI


# Models to test, from large (reference) down to small
TEST_MODELS = [
    # Reference: known-good large model
    ("meta-llama/llama-3.3-70b-instruct",              "large",  "70B dense — reference"),
    # Mid-tier
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5",       "large",  "49B dense"),
    ("qwen/qwen3-30b-a3b-04-28",                       "medium", "30B MoE (3B active)"),
    # Small
    ("meta-llama/llama-3.1-8b-instruct",               "small",  "8B dense"),
    ("qwen/qwen-2.5-7b-instruct",                      "small",  "7B dense"),
]

# Borrower IDs: fraud cases are the primary test targets
FRAUD_IDS = ["BRW-010", "BRW-011", "BRW-012", "BRW-018"]
BAD_IDS = ["BRW-006", "BRW-007"]
GOOD_IDS = ["BRW-001", "BRW-002"]


def get_test_borrowers(full_pool: bool):
    """Get the subset of borrowers to test against."""
    all_borrowers = get_borrowers("all")
    if full_pool:
        return all_borrowers
    # Default: fraud + a few bad + a few good for calibration
    target_ids = set(FRAUD_IDS + BAD_IDS + GOOD_IDS)
    return [b for b in all_borrowers if b.id in target_ids]


async def test_single_evaluation(client, model_id, borrower, lender, data_mode, semaphore):
    """Run a single model evaluation and capture detailed results."""
    clear_call_traces()

    start = time.time()
    decision = await evaluate_borrower(client, lender, borrower, semaphore, data_mode)
    elapsed = time.time() - start

    traces = get_call_traces()
    trace = traces[0] if traces else {}

    tool_calls = trace.get("tool_calls", [])
    tool_rounds = trace.get("tool_rounds", 0)
    raw_response = trace.get("raw_response", "")
    parsed_json = trace.get("parsed_json")
    error = trace.get("error")

    # Analyse jq usage
    jq_queries = []
    bash_calls = []
    for tc in tool_calls:
        if tc.startswith("run_bash:"):
            cmd = tc[len("run_bash:"):].strip()
            bash_calls.append(cmd)
            if "jq" in cmd:
                jq_queries.append(cmd)

    return {
        "model": model_id,
        "borrower_id": borrower.id,
        "borrower_name": borrower.dossier.company_name,
        "true_outcome": borrower.true_outcome,
        "data_mode": data_mode,
        "decision": decision.decision,
        "reasoning": decision.reasoning,
        "tool_calls_count": len(tool_calls),
        "tool_rounds": tool_rounds,
        "bash_calls": bash_calls,
        "jq_queries": jq_queries,
        "jq_count": len(jq_queries),
        "used_tool": len(tool_calls) > 0,
        "used_jq": len(jq_queries) > 0,
        "json_valid": parsed_json is not None,
        "error": error,
        "elapsed_s": round(elapsed, 1),
        "raw_response_len": len(raw_response) if raw_response else 0,
    }


async def run_model_test(client, model_id, tier, desc, borrowers, semaphore):
    """Test a single model against all test borrowers in both data modes."""
    lenders = get_lenders()
    lender = lenders[0]  # Use Velocity Capital as the test harness
    lender.model = model_id
    short = model_id.split("/")[-1]
    lender.name = f"Test Harness [{short}]"

    results = []

    # Test in full mode (jq available)
    for borrower in borrowers:
        clear_call_traces()
        r = await test_single_evaluation(
            client, model_id, borrower, lender, "full", semaphore,
        )
        results.append(r)
        status = r["decision"]
        jq_info = f"jq={r['jq_count']}" if r["used_jq"] else "no-jq"
        tool_info = f"tools={r['tool_calls_count']}" if r["used_tool"] else "no-tool"
        err_info = " ERROR" if r["error"] else ""
        json_info = "" if r["json_valid"] else " BAD-JSON"
        print(f"    {borrower.dossier.company_name:<28s} [{borrower.true_outcome:>5s}] "
              f"full: {status:<7s} {tool_info} {jq_info} "
              f"({r['elapsed_s']}s){err_info}{json_info}")

    # Test in quarterly_only mode (no jq) for comparison
    for borrower in borrowers:
        clear_call_traces()
        r = await test_single_evaluation(
            client, model_id, borrower, lender, "quarterly_only", semaphore,
        )
        results.append(r)
        status = r["decision"]
        err_info = " ERROR" if r["error"] else ""
        json_info = "" if r["json_valid"] else " BAD-JSON"
        print(f"    {borrower.dossier.company_name:<28s} [{borrower.true_outcome:>5s}] "
              f"q_only: {status:<7s} ({r['elapsed_s']}s){err_info}{json_info}")

    return results


def print_summary(all_results, borrowers):
    """Print a comprehensive summary of JQ capability across models."""
    models = list(dict.fromkeys(r["model"] for r in all_results))
    borrower_map = {b.id: b for b in borrowers}

    # ── Tool Use Summary ──
    print(f"\n{'='*100}")
    print(f"  JQ TOOL-USE CAPABILITY SUMMARY")
    print(f"{'='*100}")

    print(f"\n  {'Model':<45s} {'Tool Use':>9s} {'JQ Use':>8s} "
          f"{'Avg JQ/eval':>12s} {'JSON OK':>8s} {'Errors':>7s}")
    print(f"  {'─'*92}")

    for model in models:
        full_results = [r for r in all_results if r["model"] == model and r["data_mode"] == "full"]
        n = len(full_results)
        tool_use = sum(1 for r in full_results if r["used_tool"])
        jq_use = sum(1 for r in full_results if r["used_jq"])
        total_jq = sum(r["jq_count"] for r in full_results)
        avg_jq = total_jq / n if n else 0
        json_ok = sum(1 for r in full_results if r["json_valid"])
        errors = sum(1 for r in full_results if r["error"])

        print(f"  {model:<45s} {tool_use:>4d}/{n:<4d} {jq_use:>3d}/{n:<4d} "
              f"{avg_jq:>11.1f} {json_ok:>3d}/{n:<4d} {errors:>7d}")

    # ── Fraud Detection Comparison ──
    print(f"\n{'='*100}")
    print(f"  FRAUD DETECTION: full (with jq) vs quarterly_only (no jq)")
    print(f"{'='*100}")

    fraud_borrowers = [b for b in borrowers if b.true_outcome == "fraud"]
    bad_borrowers = [b for b in borrowers if b.true_outcome == "bad"]
    good_borrowers = [b for b in borrowers if b.true_outcome == "good"]

    print(f"\n  {'Model':<45s} {'Fraud Catch':>12s} {'Fraud Catch':>12s} "
          f"{'Bad Catch':>10s} {'Bad Catch':>10s} {'Good FP':>8s} {'Good FP':>8s}")
    print(f"  {'':45s} {'(full)':>12s} {'(q_only)':>12s} "
          f"{'(full)':>10s} {'(q_only)':>10s} {'(full)':>8s} {'(q_only)':>8s}")
    print(f"  {'─'*100}")

    for model in models:
        # Fraud detection in full mode
        full_fraud = [r for r in all_results
                      if r["model"] == model and r["data_mode"] == "full"
                      and r["true_outcome"] == "fraud"]
        full_fraud_caught = sum(1 for r in full_fraud if r["decision"] == "REJECT")
        n_fraud = len(full_fraud)

        # Fraud detection in quarterly_only
        qonly_fraud = [r for r in all_results
                       if r["model"] == model and r["data_mode"] == "quarterly_only"
                       and r["true_outcome"] == "fraud"]
        qonly_fraud_caught = sum(1 for r in qonly_fraud if r["decision"] == "REJECT")

        # Bad detection
        full_bad = [r for r in all_results
                    if r["model"] == model and r["data_mode"] == "full"
                    and r["true_outcome"] == "bad"]
        full_bad_caught = sum(1 for r in full_bad if r["decision"] == "REJECT")
        n_bad = len(full_bad)

        qonly_bad = [r for r in all_results
                     if r["model"] == model and r["data_mode"] == "quarterly_only"
                     and r["true_outcome"] == "bad"]
        qonly_bad_caught = sum(1 for r in qonly_bad if r["decision"] == "REJECT")

        # False positives on good borrowers
        full_good = [r for r in all_results
                     if r["model"] == model and r["data_mode"] == "full"
                     and r["true_outcome"] == "good"]
        full_fp = sum(1 for r in full_good if r["decision"] == "REJECT")
        n_good = len(full_good)

        qonly_good = [r for r in all_results
                      if r["model"] == model and r["data_mode"] == "quarterly_only"
                      and r["true_outcome"] == "good"]
        qonly_fp = sum(1 for r in qonly_good if r["decision"] == "REJECT")

        fraud_full_str = f"{full_fraud_caught}/{n_fraud}" if n_fraud else "N/A"
        fraud_qonly_str = f"{qonly_fraud_caught}/{n_fraud}" if n_fraud else "N/A"
        bad_full_str = f"{full_bad_caught}/{n_bad}" if n_bad else "N/A"
        bad_qonly_str = f"{qonly_bad_caught}/{n_bad}" if n_bad else "N/A"
        good_full_str = f"{full_fp}/{n_good}" if n_good else "N/A"
        good_qonly_str = f"{qonly_fp}/{n_good}" if n_good else "N/A"

        print(f"  {model:<45s} {fraud_full_str:>12s} {fraud_qonly_str:>12s} "
              f"{bad_full_str:>10s} {bad_qonly_str:>10s} "
              f"{good_full_str:>8s} {good_qonly_str:>8s}")

    # ── Per-Borrower Detail ──
    print(f"\n{'='*100}")
    print(f"  PER-BORROWER DECISIONS (full mode with jq)")
    print(f"{'='*100}")

    header = f"  {'Borrower':<25s} {'Truth':>6s}"
    for model in models:
        short = model.split("/")[-1][:15]
        header += f" {short:>16s}"
    print(header)
    print(f"  {'─'*25}{'─'*6}" + f"{'─'*16}" * len(models))

    test_borrower_ids = list(dict.fromkeys(r["borrower_id"] for r in all_results))
    for bid in test_borrower_ids:
        b = borrower_map.get(bid)
        if not b:
            continue
        truth = b.true_outcome.upper()
        row = f"  {b.dossier.company_name[:24]:<25s} {truth:>6s}"
        for model in models:
            r = next((r for r in all_results
                      if r["model"] == model and r["borrower_id"] == bid
                      and r["data_mode"] == "full"), None)
            if r:
                d = r["decision"][0]  # A or R
                jq_mark = f"+jq({r['jq_count']})" if r["used_jq"] else "-jq"
                cell = f"{d} {jq_mark}"
                row += f" {cell:>16s}"
            else:
                row += f" {'?':>16s}"
        print(row)

    # ── JQ Query Samples ──
    print(f"\n{'='*100}")
    print(f"  JQ QUERY SAMPLES BY MODEL")
    print(f"{'='*100}")

    for model in models:
        model_results = [r for r in all_results
                         if r["model"] == model and r["data_mode"] == "full"]
        all_jq = []
        for r in model_results:
            for q in r["jq_queries"]:
                all_jq.append((r["borrower_name"], q))

        short = model.split("/")[-1]
        print(f"\n  {short} ({len(all_jq)} jq queries total):")
        if not all_jq:
            print(f"    (no jq queries executed)")
        else:
            # Show up to 6 sample queries
            for borrower_name, query in all_jq[:6]:
                q_display = query[:90] + "..." if len(query) > 90 else query
                print(f"    [{borrower_name[:20]}] {q_display}")
            if len(all_jq) > 6:
                print(f"    ... and {len(all_jq) - 6} more")

    # ── Key Findings ──
    print(f"\n{'='*100}")
    print(f"  KEY FINDINGS")
    print(f"{'='*100}")

    for model in models:
        short = model.split("/")[-1]
        full_results = [r for r in all_results if r["model"] == model and r["data_mode"] == "full"]
        qonly_results = [r for r in all_results if r["model"] == model and r["data_mode"] == "quarterly_only"]

        n = len(full_results)
        tool_pct = sum(1 for r in full_results if r["used_tool"]) / n * 100 if n else 0
        jq_pct = sum(1 for r in full_results if r["used_jq"]) / n * 100 if n else 0
        json_pct = sum(1 for r in full_results if r["json_valid"]) / n * 100 if n else 0
        error_pct = sum(1 for r in full_results if r["error"]) / n * 100 if n else 0

        # Fraud detection improvement
        full_fraud = [r for r in full_results if r["true_outcome"] == "fraud"]
        qonly_fraud = [r for r in qonly_results if r["true_outcome"] == "fraud"]
        full_catch = sum(1 for r in full_fraud if r["decision"] == "REJECT")
        qonly_catch = sum(1 for r in qonly_fraud if r["decision"] == "REJECT")

        print(f"\n  {short}:")
        print(f"    Tool use rate: {tool_pct:.0f}% | JQ use rate: {jq_pct:.0f}% | "
              f"JSON valid: {json_pct:.0f}% | Errors: {error_pct:.0f}%")
        print(f"    Fraud detection: {full_catch}/{len(full_fraud)} (full) vs "
              f"{qonly_catch}/{len(qonly_fraud)} (quarterly_only)")

        if jq_pct == 0:
            print(f"    VERDICT: Model does NOT use jq tool — no benefit from bank statement access")
        elif jq_pct < 50:
            print(f"    VERDICT: Inconsistent jq usage — unreliable for bank statement analysis")
        elif full_catch > qonly_catch:
            print(f"    VERDICT: JQ usage IMPROVES fraud detection (+{full_catch - qonly_catch} catches)")
        elif full_catch == qonly_catch:
            print(f"    VERDICT: JQ usage does not improve fraud detection despite tool use")
        else:
            print(f"    VERDICT: JQ usage HURTS detection — model gets confused by tool output")


def main():
    parser = argparse.ArgumentParser(description="Test small models' jq capabilities")
    parser.add_argument("--model", type=str, default=None,
                        help="Test a single model instead of all")
    parser.add_argument("--full-pool", action="store_true",
                        help="Test against all 12 borrowers (slower, more expensive)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent API calls (default: 3)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    borrowers = get_test_borrowers(args.full_pool)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Filter models
    if args.model:
        models = [(args.model, "unknown", "user-specified")]
    else:
        models = TEST_MODELS

    print("=" * 100)
    print("  JQ CAPABILITY TEST FOR SMALL MODELS")
    print("=" * 100)
    print(f"\n  Testing {len(models)} models against {len(borrowers)} borrowers")
    print(f"  Borrowers: {', '.join(b.dossier.company_name for b in borrowers)}")
    print(f"  Each model tested in 'full' mode (jq available) and 'quarterly_only' (no jq)")
    print(f"  Total evaluations: {len(models) * len(borrowers) * 2}")

    all_results = []
    clear_usage()

    for model_id, tier, desc in models:
        print(f"\n{'─'*100}")
        print(f"  MODEL: {model_id} ({tier}, {desc})")
        print(f"{'─'*100}")

        try:
            results = asyncio.run(
                run_model_test(client, model_id, tier, desc, borrowers, semaphore)
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Print summary
    print_summary(all_results, borrowers)

    # Cost summary
    costs = get_cost_summary()
    if costs:
        total = sum(costs.values())
        print(f"\n{'='*100}")
        print(f"  COST SUMMARY")
        print(f"{'='*100}")
        for model, cost in sorted(costs.items()):
            print(f"  {model:<45s} ${cost:.4f}")
        print(f"  {'─'*55}")
        print(f"  {'TOTAL':<45s} ${total:.4f}")

    # Save raw results
    output_path = "jq_test_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Raw results saved to {output_path}")

    print(f"\n{'='*100}")
    print(f"  TEST COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
