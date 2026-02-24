#!/usr/bin/env python3
"""
Test whether bank statement analysis adds signal or noise.

Runs the same lender configuration with different data_modes and compares
fraud/bad detection, false positive rates, and overall scores.

Modes compared:
  - full:            quarterly income + bank statement tool (baseline)
  - quarterly_only:  quarterly income only, no raw bank data
  - statements_inline: raw bank statements in prompt, no quarterly summary
  - aggregate_only:  just annual totals + narrative

Usage:
  python test_bank_statements.py                    # Mock mode, all comparisons
  python test_bank_statements.py --live             # Live mode with API
  python test_bank_statements.py --live --model meta-llama/llama-3.3-70b-instruct
"""

import argparse
import asyncio
import copy
import os
import sys
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from loanville.data import get_borrowers, get_lenders
from loanville.engine import SimulationEngine
from loanville.llm import clear_usage, get_cost_summary, get_token_usage, get_call_traces, clear_call_traces
from loanville.scoring import score_lenders


DATA_MODES = ["full", "quarterly_only", "statements_inline", "aggregate_only"]

# Sweet-spot models for live testing
SWEET_SPOT_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
]


def run_sim(borrowers, lenders, api_key="", mock=False, data_mode="full"):
    """Run a simulation and return (scores, engine, decisions_detail)."""
    engine = SimulationEngine(borrowers, lenders, api_key, mock=mock, data_mode=data_mode)
    asyncio.run(engine.run())
    scores = score_lenders(
        lenders, engine.all_decisions, engine.booked_loans,
        engine.loan_outcomes, engine.deal_results, borrowers=borrowers,
    )

    # Build per-borrower decision map for comparison
    borrower_map = {b.id: b for b in borrowers}
    decisions_detail = {}  # borrower_id -> {lender_id: decision_str}
    for lender_id, decisions in engine.all_decisions.items():
        for d in decisions:
            if d.borrower_id not in decisions_detail:
                decisions_detail[d.borrower_id] = {}
            decisions_detail[d.borrower_id][lender_id] = {
                "decision": d.decision,
                "reasoning": d.reasoning[:120] if d.reasoning else "",
                "rate": d.term_sheet.interest_rate if d.term_sheet else None,
            }

    return scores, engine, decisions_detail


def compare_modes(borrowers, lenders, api_key="", mock=False, modes=None):
    """Run simulations across data modes and return structured results."""
    if modes is None:
        modes = DATA_MODES

    results = {}
    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"#  DATA MODE: {mode.upper()}")
        print(f"{'#'*70}")

        # Deep copy lenders so each run starts fresh
        run_lenders = copy.deepcopy(lenders)
        scores, engine, details = run_sim(
            borrowers, run_lenders, api_key, mock=mock, data_mode=mode,
        )
        results[mode] = {
            "scores": scores,
            "details": details,
            "engine": engine,
        }

    return results


def print_comparison(results, borrowers, lenders):
    """Print a detailed comparison across data modes."""
    modes = list(results.keys())
    borrower_map = {b.id: b for b in borrowers}
    lender_map = {l.id: l.name for l in lenders}

    # --- Score comparison ---
    print(f"\n{'='*90}")
    print(f"  SCORE COMPARISON ACROSS DATA MODES")
    print(f"{'='*90}")

    header = f"  {'Lender':<30s}"
    for mode in modes:
        header += f" {mode:>18s}"
    print(header)
    print(f"  {'─'*30}" + f"{'─'*18}" * len(modes))

    lender_ids = [l.id for l in lenders]
    for lid in lender_ids:
        lname = lender_map[lid]
        row = f"  {lname:<30s}"
        for mode in modes:
            s = next((s for s in results[mode]["scores"] if s.lender_id == lid), None)
            if s:
                row += f" {s.final_adjusted_score:>+16.2f}%"
            else:
                row += f" {'N/A':>17s}"
        print(row)

    # Averages
    row = f"  {'AVERAGE':<30s}"
    for mode in modes:
        avg = sum(s.final_adjusted_score for s in results[mode]["scores"]) / len(results[mode]["scores"])
        row += f" {avg:>+16.2f}%"
    print(f"  {'─'*30}" + f"{'─'*18}" * len(modes))
    print(row)

    # --- Fraud & default comparison ---
    print(f"\n{'='*90}")
    print(f"  RISK METRICS ACROSS DATA MODES")
    print(f"{'='*90}")

    header = f"  {'Metric':<30s}"
    for mode in modes:
        header += f" {mode:>18s}"
    print(header)
    print(f"  {'─'*30}" + f"{'─'*18}" * len(modes))

    for label, attr in [
        ("Total Frauds Funded", "frauds_funded"),
        ("Total Defaults", "defaults_count"),
        ("Total Net P&L ($)", "net_return"),
        ("Total Deployed ($)", "total_deployed"),
        ("Total Principal Lost ($)", "total_principal_lost"),
    ]:
        row = f"  {label:<30s}"
        for mode in modes:
            val = sum(getattr(s, attr) for s in results[mode]["scores"])
            if "($)" in label:
                row += f" ${val:>15,.0f}"
            else:
                row += f" {val:>17d}"
        print(row)

    # --- Per-borrower decision comparison ---
    print(f"\n{'='*90}")
    print(f"  PER-BORROWER DECISIONS (first lender)")
    print(f"{'='*90}")

    # Use first lender for per-borrower comparison
    first_lid = lender_ids[0]
    header = f"  {'Borrower':<25s} {'True':>6s}"
    for mode in modes:
        header += f" {mode:>14s}"
    print(header)
    print(f"  {'─'*25}{'─'*6}" + f"{'─'*14}" * len(modes))

    for b in borrowers:
        outcome_mark = {"good": "GOOD", "bad": "BAD", "fraud": "FRAUD"}[b.true_outcome]
        row = f"  {b.dossier.company_name[:24]:<25s} {outcome_mark:>6s}"
        for mode in modes:
            d = results[mode]["details"].get(b.id, {}).get(first_lid, {})
            decision = d.get("decision", "?")
            marker = "A" if decision == "APPROVE" else "R"
            # Color code: correct rejection of bad/fraud = good, false rejection of good = bad
            if b.true_outcome in ("bad", "fraud") and decision == "REJECT":
                marker += " (catch)"
            elif b.true_outcome in ("bad", "fraud") and decision == "APPROVE":
                marker += " (miss!)"
            elif b.true_outcome == "good" and decision == "REJECT":
                marker += " (FP!)"
            row += f" {marker:>14s}"
        print(row)

    # --- Signal analysis ---
    print(f"\n{'='*90}")
    print(f"  BANK STATEMENT VALUE ANALYSIS")
    print(f"{'='*90}")

    for lid in lender_ids:
        lname = lender_map[lid]
        print(f"\n  {lname}:")

        for mode in modes:
            details = results[mode]["details"]
            correct_catches = 0
            missed = 0
            false_positives = 0
            true_approvals = 0

            for b in borrowers:
                d = details.get(b.id, {}).get(lid, {})
                decision = d.get("decision", "?")

                if b.true_outcome in ("bad", "fraud"):
                    if decision == "REJECT":
                        correct_catches += 1
                    else:
                        missed += 1
                else:  # good
                    if decision == "REJECT":
                        false_positives += 1
                    else:
                        true_approvals += 1

            total_bad = sum(1 for b in borrowers if b.true_outcome in ("bad", "fraud"))
            total_good = sum(1 for b in borrowers if b.true_outcome == "good")
            catch_rate = correct_catches / total_bad * 100 if total_bad else 0
            fp_rate = false_positives / total_good * 100 if total_good else 0

            print(f"    {mode:<20s}: "
                  f"Catch={correct_catches}/{total_bad} ({catch_rate:.0f}%) | "
                  f"FP={false_positives}/{total_good} ({fp_rate:.0f}%) | "
                  f"Miss={missed}")

    # --- Key finding ---
    if "full" in results and "quarterly_only" in results:
        full_fraud = sum(s.frauds_funded for s in results["full"]["scores"])
        qonly_fraud = sum(s.frauds_funded for s in results["quarterly_only"]["scores"])
        full_avg = sum(s.final_adjusted_score for s in results["full"]["scores"]) / len(results["full"]["scores"])
        qonly_avg = sum(s.final_adjusted_score for s in results["quarterly_only"]["scores"]) / len(results["quarterly_only"]["scores"])

        print(f"\n  KEY FINDING:")
        print(f"    Full mode avg score:      {full_avg:+.2f}%  (frauds funded: {full_fraud})")
        print(f"    Quarterly-only avg score:  {qonly_avg:+.2f}%  (frauds funded: {qonly_fraud})")
        delta = full_avg - qonly_avg
        if delta > 1.0:
            print(f"    Bank statements ADD {delta:.1f}pp of value — meaningful signal")
        elif delta > 0:
            print(f"    Bank statements add {delta:.1f}pp — marginal value")
        elif delta < -1.0:
            print(f"    Bank statements HURT by {-delta:.1f}pp — adding noise")
        else:
            print(f"    Bank statements have negligible impact ({delta:+.1f}pp)")
        print(f"    Fraud detection delta: {full_fraud} vs {qonly_fraud} "
              f"({'better with statements' if full_fraud < qonly_fraud else 'no difference' if full_fraud == qonly_fraud else 'worse with statements'})")


def print_live_traces(results):
    """Print tool-use analysis for live runs."""
    traces = get_call_traces()
    if not traces:
        return

    print(f"\n{'='*90}")
    print(f"  TOOL USE ANALYSIS (LIVE MODE)")
    print(f"{'='*90}")

    by_model = {}
    for t in traces:
        model = t.get("model", "?")
        if model not in by_model:
            by_model[model] = {"total": 0, "used_tool": 0, "json_fail": 0}
        by_model[model]["total"] += 1
        if t.get("tool_calls"):
            by_model[model]["used_tool"] += 1
        if t.get("parsed_json") is None and not t.get("error"):
            by_model[model]["json_fail"] += 1

    for model, stats in by_model.items():
        pct = stats["used_tool"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {model}: {stats['total']} evals, "
              f"{stats['used_tool']} used tool ({pct:.0f}%), "
              f"{stats['json_fail']} JSON failures")


def main():
    parser = argparse.ArgumentParser(description="Bank statement value test")
    parser.add_argument("--live", action="store_true",
                        help="Use live API instead of mock")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use for all lenders (live mode)")
    parser.add_argument("--mix", default="hard",
                        choices=["easy", "balanced", "hard", "all"],
                        help="Borrower mix (default: hard)")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=DATA_MODES,
                        help="Data modes to compare (default: all)")
    args = parser.parse_args()

    mock = not args.live
    api_key = ""
    if args.live:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set for live mode")
            sys.exit(1)

    borrowers = get_borrowers(args.mix)
    lenders = get_lenders()
    outcomes = Counter(b.true_outcome for b in borrowers)

    # Override model if specified
    if args.model:
        for l in lenders:
            l.model = args.model
            short = args.model.split("/")[-1]
            base = l.name.split("[")[0].strip()
            l.name = f"{base} [{short}]"

    modes = args.modes or DATA_MODES

    print("=" * 90)
    print("  BANK STATEMENT VALUE TEST")
    print(f"  Mode: {'LIVE' if args.live else 'MOCK'}")
    if args.model:
        print(f"  Model: {args.model}")
    print(f"  Mix: {args.mix} — {len(borrowers)} borrowers "
          f"({outcomes['good']} good, {outcomes.get('bad',0)} bad, "
          f"{outcomes.get('fraud',0)} fraud)")
    print(f"  Data modes: {', '.join(modes)}")
    print("=" * 90)

    clear_usage()
    clear_call_traces()

    results = compare_modes(borrowers, lenders, api_key, mock=mock, modes=modes)
    print_comparison(results, borrowers, lenders)

    if args.live:
        print_live_traces(results)
        # Print cost
        costs = get_cost_summary()
        usage = get_token_usage()
        if costs:
            total = sum(costs.values())
            print(f"\n  Total API cost: ${total:.4f}")

    print(f"\n{'='*90}")
    print(f"  TEST COMPLETE")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
