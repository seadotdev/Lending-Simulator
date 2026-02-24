#!/usr/bin/env python3
"""
Compare underwriting outcomes: bank statements only vs financials only,
across 4 models.

Runs each model in two data modes:
  - statements_inline: raw 12-month bank statements in prompt (no quarterly summaries)
  - quarterly_only:    quarterly income statements only (no raw bank data)

This isolates whether a model's underwriting quality comes from analyzing
raw transaction data vs structured financial summaries.

Usage:
  python test_underwriting_modes.py                    # Mock mode (fast, deterministic)
  python test_underwriting_modes.py --live             # Live mode with OpenRouter API
  python test_underwriting_modes.py --live --mix hard  # Live mode, hard borrower mix
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from loanville.data import get_borrowers, get_lenders
from loanville.engine import SimulationEngine
from loanville.llm import clear_usage, get_cost_summary, get_token_usage, get_call_traces, clear_call_traces
from loanville.scoring import score_lenders


# The two data modes to compare
DATA_MODES = ["statements_inline", "quarterly_only"]
MODE_LABELS = {
    "statements_inline": "Bank Stmts Only",
    "quarterly_only":    "Financials Only",
}

# 4 models spanning different architectures and sizes
TEST_MODELS = [
    "deepseek/deepseek-chat-v3-0324",                # ~685B MoE — frontier
    "meta-llama/llama-3.3-70b-instruct",              # 70B dense — strong open-source
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",       # 49B dense — mid-tier
    "qwen/qwen3-30b-a3b",                            # 30B MoE (3B active) — small
]


def run_sim(borrowers, lenders, api_key="", mock=False, data_mode="full"):
    """Run a simulation and return (scores, engine, decisions_detail)."""
    engine = SimulationEngine(borrowers, lenders, api_key, mock=mock, data_mode=data_mode)
    asyncio.run(engine.run())
    scores = score_lenders(
        lenders, engine.all_decisions, engine.booked_loans,
        engine.loan_outcomes, engine.deal_results, borrowers=borrowers,
    )

    # Build per-borrower decision map
    decisions_detail = {}
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


def run_model_comparison(borrowers, base_lenders, api_key="", mock=False, models=None):
    """Run all models x modes and return structured results.

    Returns: {model_id: {mode: {scores, details, engine}}}
    """
    if models is None:
        models = TEST_MODELS

    results = {}
    for model in models:
        short = model.split("/")[-1]
        results[model] = {}
        for mode in DATA_MODES:
            print(f"\n{'#'*70}")
            print(f"#  MODEL: {short}  |  MODE: {mode.upper()}")
            print(f"{'#'*70}")

            # Deep copy lenders and override all to use this model
            lenders = copy.deepcopy(base_lenders)
            for l in lenders:
                l.model = model
                base_name = l.name.split("[")[0].strip()
                l.name = f"{base_name} [{short}]"

            scores, engine, details = run_sim(
                borrowers, lenders, api_key, mock=mock, data_mode=mode,
            )
            results[model][mode] = {
                "scores": scores,
                "details": details,
                "engine": engine,
            }

    return results


def _get_confusion(details, borrowers, lender_ids):
    """Compute aggregate confusion stats across all lenders for a single run."""
    correct_catches = 0  # bad/fraud correctly rejected
    missed = 0           # bad/fraud incorrectly approved
    true_approvals = 0   # good correctly approved
    false_positives = 0  # good incorrectly rejected

    for b in borrowers:
        for lid in lender_ids:
            d = details.get(b.id, {}).get(lid, {})
            decision = d.get("decision", "?")
            if b.true_outcome in ("bad", "fraud"):
                if decision == "REJECT":
                    correct_catches += 1
                else:
                    missed += 1
            else:
                if decision == "REJECT":
                    false_positives += 1
                else:
                    true_approvals += 1

    return correct_catches, missed, true_approvals, false_positives


def print_results(results, borrowers, base_lenders):
    """Print a comprehensive comparison."""
    models = list(results.keys())
    lender_ids = [l.id for l in base_lenders]
    n_lenders = len(lender_ids)

    # ── Section 1: Score Matrix ──
    print(f"\n{'='*100}")
    print(f"  SCORE COMPARISON: BANK STATEMENTS vs FINANCIALS ACROSS MODELS")
    print(f"{'='*100}")

    header = f"  {'Model':<45s}"
    for mode in DATA_MODES:
        header += f" {MODE_LABELS[mode]:>18s}"
    header += f" {'Delta':>10s}"
    print(header)
    print(f"  {'─'*45}" + f"{'─'*18}" * len(DATA_MODES) + f"{'─'*10}")

    for model in models:
        short = model.split("/")[-1]
        row = f"  {short:<45s}"
        mode_avgs = {}
        for mode in DATA_MODES:
            scores = results[model][mode]["scores"]
            avg = sum(s.final_adjusted_score for s in scores) / len(scores)
            mode_avgs[mode] = avg
            row += f" {avg:>+16.2f}%"
        delta = mode_avgs["statements_inline"] - mode_avgs["quarterly_only"]
        row += f" {delta:>+8.2f}%"
        print(row)

    # ── Section 2: Risk Metrics Matrix ──
    print(f"\n{'='*100}")
    print(f"  RISK METRICS: BANK STATEMENTS vs FINANCIALS")
    print(f"{'='*100}")

    for metric_label, attr in [
        ("Frauds Funded", "frauds_funded"),
        ("Total Defaults", "defaults_count"),
        ("Net P&L ($)", "net_return"),
        ("Total Deployed ($)", "total_deployed"),
        ("Principal Lost ($)", "total_principal_lost"),
    ]:
        print(f"\n  {metric_label}:")
        header = f"    {'Model':<43s}"
        for mode in DATA_MODES:
            header += f" {MODE_LABELS[mode]:>18s}"
        print(header)
        print(f"    {'─'*43}" + f"{'─'*18}" * len(DATA_MODES))

        for model in models:
            short = model.split("/")[-1]
            row = f"    {short:<43s}"
            for mode in DATA_MODES:
                val = sum(getattr(s, attr) for s in results[model][mode]["scores"])
                if "($)" in metric_label:
                    row += f" ${val:>15,.0f}"
                else:
                    row += f" {val:>17d}"
            print(row)

    # ── Section 3: Fraud & Bad Detection Rates ──
    print(f"\n{'='*100}")
    print(f"  DETECTION RATES BY MODEL AND MODE")
    print(f"{'='*100}")

    total_bad_fraud = sum(1 for b in borrowers if b.true_outcome in ("bad", "fraud"))
    total_good = sum(1 for b in borrowers if b.true_outcome == "good")
    total_fraud = sum(1 for b in borrowers if b.true_outcome == "fraud")
    total_bad = sum(1 for b in borrowers if b.true_outcome == "bad")

    # Per-lender detection (averaged across lenders)
    print(f"\n  Pool: {len(borrowers)} borrowers ({total_good} good, "
          f"{total_bad} bad, {total_fraud} fraud) × {n_lenders} lenders")

    print(f"\n  {'Model':<35s} {'Mode':<18s} "
          f"{'Catch%':>7s} {'FP%':>7s} {'Catches':>8s} {'Misses':>7s} {'FPs':>5s}")
    print(f"  {'─'*35} {'─'*18} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*5}")

    for model in models:
        short = model.split("/")[-1]
        for mode in DATA_MODES:
            details = results[model][mode]["details"]
            catches, misses, ta, fps = _get_confusion(details, borrowers, lender_ids)
            total_bad_decisions = catches + misses
            total_good_decisions = ta + fps
            catch_rate = catches / total_bad_decisions * 100 if total_bad_decisions else 0
            fp_rate = fps / total_good_decisions * 100 if total_good_decisions else 0

            print(f"  {short:<35s} {MODE_LABELS[mode]:<18s} "
                  f"{catch_rate:>6.1f}% {fp_rate:>6.1f}% "
                  f"{catches:>8d} {misses:>7d} {fps:>5d}")
        print()

    # ── Section 4: Per-Borrower Decision Grid ──
    print(f"\n{'='*100}")
    print(f"  PER-BORROWER DECISIONS (first lender)")
    print(f"{'='*100}")

    first_lid = lender_ids[0]

    # Build column headers: model_short × mode
    col_labels = []
    for model in models:
        short = model.split("/")[-1][:12]
        for mode in DATA_MODES:
            abbr = "BS" if mode == "statements_inline" else "FN"
            col_labels.append(f"{short[:8]}:{abbr}")

    header = f"  {'Borrower':<22s} {'True':>5s}"
    for cl in col_labels:
        header += f" {cl:>13s}"
    print(header)
    print(f"  {'─'*22}{'─'*5}" + f"{'─'*13}" * len(col_labels))

    for b in borrowers:
        outcome_mark = {"good": "GOOD", "bad": "BAD", "fraud": "FRAUD"}[b.true_outcome]
        row = f"  {b.dossier.company_name[:21]:<22s} {outcome_mark:>5s}"
        for model in models:
            for mode in DATA_MODES:
                d = results[model][mode]["details"].get(b.id, {}).get(first_lid, {})
                decision = d.get("decision", "?")
                marker = "A" if decision == "APPROVE" else "R"
                if b.true_outcome in ("bad", "fraud") and decision == "REJECT":
                    marker += " catch"
                elif b.true_outcome in ("bad", "fraud") and decision == "APPROVE":
                    marker += " MISS"
                elif b.true_outcome == "good" and decision == "REJECT":
                    marker += " FP"
                row += f" {marker:>13s}"
        print(row)

    # ── Section 5: Key Findings ──
    print(f"\n{'='*100}")
    print(f"  KEY FINDINGS")
    print(f"{'='*100}")

    for model in models:
        short = model.split("/")[-1]
        bs_scores = results[model]["statements_inline"]["scores"]
        fn_scores = results[model]["quarterly_only"]["scores"]
        bs_avg = sum(s.final_adjusted_score for s in bs_scores) / len(bs_scores)
        fn_avg = sum(s.final_adjusted_score for s in fn_scores) / len(fn_scores)
        bs_fraud = sum(s.frauds_funded for s in bs_scores)
        fn_fraud = sum(s.frauds_funded for s in fn_scores)
        bs_defaults = sum(s.defaults_count for s in bs_scores)
        fn_defaults = sum(s.defaults_count for s in fn_scores)

        delta = bs_avg - fn_avg
        print(f"\n  {short}:")
        print(f"    Bank Statements:  score={bs_avg:+.2f}%  frauds={bs_fraud}  defaults={bs_defaults}")
        print(f"    Financials:       score={fn_avg:+.2f}%  frauds={fn_fraud}  defaults={fn_defaults}")

        if abs(delta) < 0.5:
            print(f"    --> Negligible difference ({delta:+.2f}pp)")
        elif delta > 0:
            print(f"    --> Bank statements BETTER by {delta:.1f}pp")
            if bs_fraud < fn_fraud:
                print(f"        Catches {fn_fraud - bs_fraud} more fraud(s) with raw transaction data")
        else:
            print(f"    --> Financials BETTER by {-delta:.1f}pp")
            if fn_fraud < bs_fraud:
                print(f"        Better fraud detection with structured financials")
            if fn_defaults < bs_defaults:
                print(f"        Better at catching bad businesses with quarterly trends")

    # ── Section 6: Model Ranking by Mode ──
    print(f"\n{'='*100}")
    print(f"  MODEL RANKINGS BY MODE")
    print(f"{'='*100}")

    for mode in DATA_MODES:
        print(f"\n  {MODE_LABELS[mode]} (ranked by avg score):")
        model_avgs = []
        for model in models:
            scores = results[model][mode]["scores"]
            avg = sum(s.final_adjusted_score for s in scores) / len(scores)
            model_avgs.append((model.split("/")[-1], avg))
        model_avgs.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, avg) in enumerate(model_avgs, 1):
            print(f"    #{rank}  {name:<45s} {avg:>+8.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare underwriting: bank statements only vs financials only across 4 models"
    )
    parser.add_argument("--live", action="store_true",
                        help="Use live API instead of mock")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model list (space-separated model IDs)")
    parser.add_argument("--mix", default="balanced",
                        choices=["easy", "balanced", "hard", "all", "analyst", "fraud", "stress"],
                        help="Borrower mix preset (default: balanced)")
    args = parser.parse_args()

    mock = not args.live
    api_key = ""
    if args.live:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set for live mode")
            sys.exit(1)

    models = args.models or TEST_MODELS
    borrowers = get_borrowers(args.mix)
    base_lenders = get_lenders()
    outcomes = Counter(b.true_outcome for b in borrowers)

    print("=" * 100)
    print("  UNDERWRITING MODE COMPARISON TEST")
    print(f"  Bank Statements Only vs Financials Only × {len(models)} Models")
    print(f"  Mode: {'LIVE' if args.live else 'MOCK'}")
    print(f"  Mix: {args.mix} — {len(borrowers)} borrowers "
          f"({outcomes['good']} good, {outcomes.get('bad', 0)} bad, "
          f"{outcomes.get('fraud', 0)} fraud)")
    print(f"  Models: {', '.join(m.split('/')[-1] for m in models)}")
    print(f"  Data modes: {', '.join(DATA_MODES)}")
    print("=" * 100)

    clear_usage()
    clear_call_traces()

    results = run_model_comparison(borrowers, base_lenders, api_key, mock=mock, models=models)
    print_results(results, borrowers, base_lenders)

    if args.live:
        costs = get_cost_summary()
        usage = get_token_usage()
        if costs:
            print(f"\n{'='*100}")
            print(f"  API COSTS")
            print(f"{'='*100}")
            for model, cost in sorted(costs.items()):
                tokens = usage.get(model, {})
                print(f"  {model}: ${cost:.4f} "
                      f"({tokens.get('prompt', 0):,} in / {tokens.get('completion', 0):,} out)")
            print(f"  Total: ${sum(costs.values()):.4f}")

    # Save results to JSON for further analysis
    output = {}
    for model in models:
        short = model.split("/")[-1]
        output[short] = {}
        for mode in DATA_MODES:
            scores = results[model][mode]["scores"]
            output[short][mode] = {
                "avg_score": sum(s.final_adjusted_score for s in scores) / len(scores),
                "frauds_funded": sum(s.frauds_funded for s in scores),
                "defaults": sum(s.defaults_count for s in scores),
                "net_pnl": sum(s.net_return for s in scores),
                "total_deployed": sum(s.total_deployed for s in scores),
                "principal_lost": sum(s.total_principal_lost for s in scores),
            }

    with open("underwriting_mode_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to underwriting_mode_results.json")

    print(f"\n{'='*100}")
    print(f"  TEST COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
