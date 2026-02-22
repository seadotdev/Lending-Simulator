#!/usr/bin/env python3
"""
Nemotron model family generalization test.

The original underwriting mode comparison found that Nemotron Super 49B
bucked the trend: it was *better* at spotting bad businesses from raw
bank statements (1/4 missed) than from quarterly financials (2/4 missed),
while three other model families showed the opposite pattern.

This experiment tests whether that characteristic generalizes across the
full Nemotron model family — from 9B to 253B — and how small the models
can go before they stop producing useful underwriting decisions.

Usage:
  python test_nemotron_generalization.py                    # Mock mode
  python test_nemotron_generalization.py --live             # Live API
  python test_nemotron_generalization.py --live --mix hard  # Harder borrower mix
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from collections import Counter
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
    get_call_traces,
    clear_call_traces,
)
from loanville.scoring import score_lenders


# The two data modes — same as the original experiment
DATA_MODES = ["statements_inline", "quarterly_only"]
MODE_LABELS = {
    "statements_inline": "Bank Stmts",
    "quarterly_only":    "Financials",
}

# Full Nemotron model lineup, largest to smallest
NEMOTRON_MODELS = [
    ("nvidia/llama-3.1-nemotron-ultra-253b-v1",        "Nemotron Ultra 253B",  253),
    ("nvidia/llama-3.1-nemotron-70b-instruct",          "Nemotron 70B",          70),
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5",        "Nemotron Super 49B",    49),
    ("nvidia/nemotron-3-nano-30b-a3b",                   "Nemotron Nano 30B",     30),
    ("nvidia/nemotron-nano-12b-v2-vl",                   "Nemotron Nano 12B VL",  12),
    ("nvidia/nemotron-nano-9b-v2",                       "Nemotron Nano 9B",       9),
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
                "reasoning": d.reasoning[:200] if d.reasoning else "",
                "rate": d.term_sheet.interest_rate if d.term_sheet else None,
            }

    return scores, engine, decisions_detail


def run_nemotron_comparison(borrowers, base_lenders, api_key="", mock=False,
                             models=None, resume_data=None):
    """Run all Nemotron models × modes and return structured results.

    Returns: {model_id: {mode: {scores, details, engine}}}
    """
    if models is None:
        models = NEMOTRON_MODELS

    done_models = set()
    if resume_data:
        done_models = set(resume_data.keys())

    results = {}
    for model_id, display_name, params_b in models:
        short = model_id.split("/")[-1]

        if short in done_models:
            print(f"\n  [SKIP] {display_name} — already in resume data")
            continue

        results[model_id] = {}
        for mode in DATA_MODES:
            print(f"\n{'#'*70}")
            print(f"#  {display_name} ({params_b}B params)  |  MODE: {mode.upper()}")
            print(f"{'#'*70}")

            # Deep copy lenders and override all to use this model
            lenders = copy.deepcopy(base_lenders)
            for l in lenders:
                l.model = model_id
                base_name = l.name.split("[")[0].strip()
                l.name = f"{base_name} [{short}]"

            try:
                scores, engine, details = run_sim(
                    borrowers, lenders, api_key, mock=mock, data_mode=mode,
                )
                results[model_id][mode] = {
                    "scores": scores,
                    "details": details,
                    "engine": engine,
                }
            except Exception as e:
                print(f"\n  FAILED: {type(e).__name__}: {e}")
                results[model_id][mode] = {
                    "scores": [],
                    "details": {},
                    "engine": None,
                    "error": str(e),
                }

    return results


def _get_confusion(details, borrowers, lender_ids, outcome_filter=None):
    """Compute confusion stats, optionally filtering by true outcome type.

    outcome_filter: None = all bad+fraud, "bad" = bad only, "fraud" = fraud only
    """
    correct_catches = 0
    missed = 0
    true_approvals = 0
    false_positives = 0

    for b in borrowers:
        for lid in lender_ids:
            d = details.get(b.id, {}).get(lid, {})
            decision = d.get("decision", "?")
            if b.true_outcome in ("bad", "fraud"):
                if outcome_filter and b.true_outcome != outcome_filter:
                    continue
                if decision == "REJECT":
                    correct_catches += 1
                else:
                    missed += 1
            else:
                if outcome_filter:
                    continue
                if decision == "REJECT":
                    false_positives += 1
                else:
                    true_approvals += 1

    return correct_catches, missed, true_approvals, false_positives


def print_nemotron_results(results, borrowers, base_lenders, models=None):
    """Print Nemotron-specific analysis focused on the generalization question."""
    if models is None:
        models = NEMOTRON_MODELS
    lender_ids = [l.id for l in base_lenders]

    total_bad = sum(1 for b in borrowers if b.true_outcome == "bad")
    total_fraud = sum(1 for b in borrowers if b.true_outcome == "fraud")
    total_good = sum(1 for b in borrowers if b.true_outcome == "good")

    # Filter to models that actually ran
    active_models = [(mid, name, p) for mid, name, p in models if mid in results]

    # ── Section 1: The Core Question ──
    print(f"\n{'='*100}")
    print(f"  NEMOTRON FAMILY GENERALIZATION TEST")
    print(f"  Does the 49B's raw-statement advantage persist across model sizes?")
    print(f"{'='*100}")
    print(f"\n  Pool: {len(borrowers)} borrowers ({total_good} good, "
          f"{total_bad} bad, {total_fraud} fraud)")

    # ── Section 2: Bad Business Detection (the key metric) ──
    print(f"\n{'='*100}")
    print(f"  BAD BUSINESS DETECTION: Bank Statements vs Financials")
    print(f"  ({total_bad} bad businesses in pool, across {len(lender_ids)} lenders "
          f"= {total_bad * len(lender_ids)} decisions)")
    print(f"{'='*100}")

    print(f"\n  {'Model':<30s} {'Params':>7s} "
          f"{'Stmts Catch':>12s} {'Fin. Catch':>11s} {'Delta':>8s} {'Winner':>12s}")
    print(f"  {'─'*30} {'─'*7} {'─'*12} {'─'*11} {'─'*8} {'─'*12}")

    statements_advantage_count = 0
    for model_id, display_name, params_b in active_models:
        r = results[model_id]
        if not r.get("statements_inline", {}).get("scores"):
            print(f"  {display_name:<30s} {params_b:>5d}B   [FAILED]")
            continue

        # Bad-only detection rates
        bs_catch, bs_miss, _, _ = _get_confusion(
            r["statements_inline"]["details"], borrowers, lender_ids, "bad")
        fn_catch, fn_miss, _, _ = _get_confusion(
            r["quarterly_only"]["details"], borrowers, lender_ids, "bad")

        bs_total = bs_catch + bs_miss
        fn_total = fn_catch + fn_miss
        bs_rate = bs_catch / bs_total * 100 if bs_total else 0
        fn_rate = fn_catch / fn_total * 100 if fn_total else 0
        delta = bs_rate - fn_rate

        if delta > 0:
            winner = "STATEMENTS"
            statements_advantage_count += 1
        elif delta < 0:
            winner = "FINANCIALS"
        else:
            winner = "TIE"

        print(f"  {display_name:<30s} {params_b:>5d}B "
              f"  {bs_catch}/{bs_total} ({bs_rate:4.0f}%) "
              f" {fn_catch}/{fn_total} ({fn_rate:4.0f}%) "
              f" {delta:>+6.0f}pp  {winner:>12s}")

    print(f"\n  Models where statements beat financials: "
          f"{statements_advantage_count}/{len(active_models)}")

    # ── Section 3: Fraud Detection ──
    print(f"\n{'='*100}")
    print(f"  FRAUD DETECTION: Bank Statements vs Financials")
    print(f"  ({total_fraud} frauds in pool)")
    print(f"{'='*100}")

    print(f"\n  {'Model':<30s} {'Params':>7s} "
          f"{'Stmts Catch':>12s} {'Fin. Catch':>11s} {'Delta':>8s}")
    print(f"  {'─'*30} {'─'*7} {'─'*12} {'─'*11} {'─'*8}")

    for model_id, display_name, params_b in active_models:
        r = results[model_id]
        if not r.get("statements_inline", {}).get("scores"):
            continue

        bs_catch, bs_miss, _, _ = _get_confusion(
            r["statements_inline"]["details"], borrowers, lender_ids, "fraud")
        fn_catch, fn_miss, _, _ = _get_confusion(
            r["quarterly_only"]["details"], borrowers, lender_ids, "fraud")

        bs_total = bs_catch + bs_miss
        fn_total = fn_catch + fn_miss
        bs_rate = bs_catch / bs_total * 100 if bs_total else 0
        fn_rate = fn_catch / fn_total * 100 if fn_total else 0
        delta = bs_rate - fn_rate

        print(f"  {display_name:<30s} {params_b:>5d}B "
              f"  {bs_catch}/{bs_total} ({bs_rate:4.0f}%) "
              f" {fn_catch}/{fn_total} ({fn_rate:4.0f}%) "
              f" {delta:>+6.0f}pp")

    # ── Section 4: Overall Score Comparison ──
    print(f"\n{'='*100}")
    print(f"  OVERALL SCORE: Bank Statements vs Financials")
    print(f"{'='*100}")

    print(f"\n  {'Model':<30s} {'Params':>7s} "
          f"{'Stmts Score':>12s} {'Fin. Score':>11s} {'Delta':>8s}")
    print(f"  {'─'*30} {'─'*7} {'─'*12} {'─'*11} {'─'*8}")

    for model_id, display_name, params_b in active_models:
        r = results[model_id]
        if not r.get("statements_inline", {}).get("scores"):
            continue

        bs_avg = sum(s.final_adjusted_score for s in r["statements_inline"]["scores"]) / len(r["statements_inline"]["scores"])
        fn_avg = sum(s.final_adjusted_score for s in r["quarterly_only"]["scores"]) / len(r["quarterly_only"]["scores"])
        delta = bs_avg - fn_avg

        print(f"  {display_name:<30s} {params_b:>5d}B "
              f"  {bs_avg:>+10.2f}% "
              f" {fn_avg:>+9.2f}% "
              f" {delta:>+6.1f}pp")

    # ── Section 5: Financial Impact ──
    print(f"\n{'='*100}")
    print(f"  FINANCIAL IMPACT BY MODEL AND MODE")
    print(f"{'='*100}")

    print(f"\n  {'Model':<30s} {'Mode':<12s} {'Deployed':>12s} "
          f"{'Princ Lost':>12s} {'Net P&L':>12s} {'Defaults':>9s} {'Frauds':>7s}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*9} {'─'*7}")

    for model_id, display_name, params_b in active_models:
        r = results[model_id]
        for mode in DATA_MODES:
            if not r.get(mode, {}).get("scores"):
                continue
            scores = r[mode]["scores"]
            deployed = sum(s.total_deployed for s in scores)
            lost = sum(s.total_principal_lost for s in scores)
            pnl = sum(s.net_return for s in scores)
            defaults = sum(s.defaults_count for s in scores)
            frauds = sum(s.frauds_funded for s in scores)

            print(f"  {display_name:<30s} {MODE_LABELS[mode]:<12s} "
                  f"${deployed:>10,.0f} ${lost:>10,.0f} ${pnl:>10,.0f} "
                  f"{defaults:>9d} {frauds:>7d}")
        print()

    # ── Section 6: Size Floor Analysis ──
    print(f"\n{'='*100}")
    print(f"  SIZE FLOOR: How Small Can Nemotron Go?")
    print(f"{'='*100}")

    print(f"\n  A model is 'useful' if it scores above the reject-all baseline (~-10%)")
    print(f"  and funds fewer bad/fraud businesses than it catches.\n")

    print(f"  {'Model':<30s} {'Params':>7s} "
          f"{'Best Score':>11s} {'Catch Rate':>11s} {'FP Rate':>8s} {'Useful?':>8s}")
    print(f"  {'─'*30} {'─'*7} {'─'*11} {'─'*11} {'─'*8} {'─'*8}")

    for model_id, display_name, params_b in active_models:
        r = results[model_id]
        best_score = None
        best_catch = 0
        best_fp = 100

        for mode in DATA_MODES:
            if not r.get(mode, {}).get("scores"):
                continue
            scores = r[mode]["scores"]
            avg = sum(s.final_adjusted_score for s in scores) / len(scores)
            catches, misses, ta, fps = _get_confusion(
                r[mode]["details"], borrowers, lender_ids)
            catch_rate = catches / (catches + misses) * 100 if (catches + misses) else 0
            fp_rate = fps / (ta + fps) * 100 if (ta + fps) else 0

            if best_score is None or avg > best_score:
                best_score = avg
                best_catch = catch_rate
                best_fp = fp_rate

        useful = "YES" if (best_score is not None and best_score > -10
                           and best_catch > 50) else "NO"
        if best_score is not None:
            print(f"  {display_name:<30s} {params_b:>5d}B "
                  f"  {best_score:>+9.2f}% "
                  f"   {best_catch:>8.1f}% "
                  f" {best_fp:>6.1f}% "
                  f" {useful:>8s}")
        else:
            print(f"  {display_name:<30s} {params_b:>5d}B   [NO DATA]")

    # ── Section 7: Per-Borrower Decision Grid ──
    print(f"\n{'='*100}")
    print(f"  PER-BORROWER DECISIONS (first lender, bad businesses only)")
    print(f"{'='*100}")

    first_lid = lender_ids[0]
    bad_borrowers = [b for b in borrowers if b.true_outcome in ("bad", "fraud")]

    col_labels = []
    for model_id, display_name, params_b in active_models:
        short = f"{params_b}B"
        for mode in DATA_MODES:
            abbr = "BS" if mode == "statements_inline" else "FN"
            col_labels.append(f"{short}:{abbr}")

    header = f"  {'Borrower':<22s} {'Type':>5s}"
    for cl in col_labels:
        header += f" {cl:>8s}"
    print(header)
    print(f"  {'─'*22}{'─'*5}" + f"{'─'*8}" * len(col_labels))

    for b in bad_borrowers:
        outcome_mark = {"bad": "BAD", "fraud": "FRAUD"}[b.true_outcome]
        row = f"  {b.dossier.company_name[:21]:<22s} {outcome_mark:>5s}"
        for model_id, display_name, params_b in active_models:
            r = results[model_id]
            for mode in DATA_MODES:
                if not r.get(mode, {}).get("details"):
                    row += f" {'?':>8s}"
                    continue
                d = r[mode]["details"].get(b.id, {}).get(first_lid, {})
                decision = d.get("decision", "?")
                if decision == "REJECT":
                    row += f" {'CATCH':>8s}"
                elif decision == "APPROVE":
                    row += f" {'MISS':>8s}"
                else:
                    row += f" {'?':>8s}"
        print(row)


def save_results(results, borrowers, base_lenders, models, mix_name, filename):
    """Save structured results to JSON."""
    lender_ids = [l.id for l in base_lenders]
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "nemotron_generalization",
        "mix": mix_name,
        "n_borrowers": len(borrowers),
        "borrower_pool": {
            "good": sum(1 for b in borrowers if b.true_outcome == "good"),
            "bad": sum(1 for b in borrowers if b.true_outcome == "bad"),
            "fraud": sum(1 for b in borrowers if b.true_outcome == "fraud"),
        },
        "models": {},
    }

    for model_id, display_name, params_b in models:
        if model_id not in results:
            continue
        r = results[model_id]
        short = model_id.split("/")[-1]
        model_output = {
            "display_name": display_name,
            "params_b": params_b,
            "model_id": model_id,
        }

        for mode in DATA_MODES:
            if not r.get(mode, {}).get("scores"):
                model_output[mode] = {"error": r.get(mode, {}).get("error", "unknown")}
                continue

            scores = r[mode]["scores"]

            # Bad-only detection
            bad_catch, bad_miss, _, _ = _get_confusion(
                r[mode]["details"], borrowers, lender_ids, "bad")
            # Fraud-only detection
            fraud_catch, fraud_miss, _, _ = _get_confusion(
                r[mode]["details"], borrowers, lender_ids, "fraud")
            # Overall detection
            all_catch, all_miss, true_app, fp = _get_confusion(
                r[mode]["details"], borrowers, lender_ids)

            model_output[mode] = {
                "avg_score": round(sum(s.final_adjusted_score for s in scores) / len(scores), 4),
                "frauds_funded": sum(s.frauds_funded for s in scores),
                "defaults": sum(s.defaults_count for s in scores),
                "net_pnl": round(sum(s.net_return for s in scores), 2),
                "total_deployed": round(sum(s.total_deployed for s in scores), 2),
                "principal_lost": round(sum(s.total_principal_lost for s in scores), 2),
                "bad_catch_rate": round(bad_catch / (bad_catch + bad_miss) * 100, 1) if (bad_catch + bad_miss) else None,
                "bad_caught": bad_catch,
                "bad_missed": bad_miss,
                "fraud_catch_rate": round(fraud_catch / (fraud_catch + fraud_miss) * 100, 1) if (fraud_catch + fraud_miss) else None,
                "fraud_caught": fraud_catch,
                "fraud_missed": fraud_miss,
                "overall_catch_rate": round(all_catch / (all_catch + all_miss) * 100, 1) if (all_catch + all_miss) else None,
                "false_positive_rate": round(fp / (true_app + fp) * 100, 1) if (true_app + fp) else None,
            }

        output["models"][short] = model_output

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Nemotron model family generalization on bank statement analysis"
    )
    parser.add_argument("--live", action="store_true",
                        help="Use live API instead of mock")
    parser.add_argument("--mix", default="balanced",
                        choices=["easy", "balanced", "hard", "all", "analyst", "fraud", "stress"],
                        help="Borrower mix preset (default: balanced)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous results JSON file")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Model sizes to skip (e.g., --skip 253 70)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run these model IDs (overrides default list)")
    parser.add_argument("--free", action="store_true",
                        help="Use :free variants where available")
    args = parser.parse_args()

    mock = not args.live
    api_key = ""
    if args.live:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set for live mode")
            sys.exit(1)

    # Filter/override models
    models = NEMOTRON_MODELS
    if args.only:
        # Build custom model list from --only IDs
        models = []
        for mid in args.only:
            # Try to match against known models
            matched = False
            for known_id, name, params in NEMOTRON_MODELS:
                if mid == known_id or mid.rstrip(":free") == known_id:
                    models.append((mid, name + (" (free)" if ":free" in mid else ""), params))
                    matched = True
                    break
            if not matched:
                short = mid.split("/")[-1]
                models.append((mid, short, 0))
    elif args.free:
        # Use :free variants where available
        FREE_VARIANTS = {
            "nvidia/nemotron-3-nano-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b:free",
            "nvidia/nemotron-nano-12b-v2-vl": "nvidia/nemotron-nano-12b-v2-vl:free",
            "nvidia/nemotron-nano-9b-v2": "nvidia/nemotron-nano-9b-v2:free",
        }
        models = [
            (FREE_VARIANTS.get(m, m), n + (" (free)" if m in FREE_VARIANTS else ""), p)
            for m, n, p in models
        ]
    if args.skip:
        skip_sizes = {int(s) for s in args.skip}
        models = [(m, n, p) for m, n, p in models if p not in skip_sizes]

    borrowers = get_borrowers(args.mix)
    base_lenders = get_lenders()
    outcomes = Counter(b.true_outcome for b in borrowers)

    # Estimate cost
    total_est = 0
    n_evals = len(borrowers) * len(base_lenders) * len(DATA_MODES)
    for model_id, _, _ in models:
        inp, out = MODEL_PRICING.get(model_id, (1.0, 3.0))
        est_per_eval = 4000 / 1_000_000 * inp + 500 / 1_000_000 * out
        total_est += est_per_eval * n_evals

    # Resume support
    resume_data = None
    if args.resume:
        with open(args.resume) as f:
            resume_data = json.load(f).get("models", {})
        print(f"  Loaded {len(resume_data)} models from resume file")

    print("=" * 100)
    print("  NEMOTRON MODEL FAMILY GENERALIZATION TEST")
    print(f"  Testing {len(models)} Nemotron models × 2 data modes")
    print(f"  Mode: {'LIVE' if args.live else 'MOCK'}")
    print(f"  Mix: {args.mix} — {len(borrowers)} borrowers "
          f"({outcomes['good']} good, {outcomes.get('bad', 0)} bad, "
          f"{outcomes.get('fraud', 0)} fraud)")
    print(f"  Models: {', '.join(n for _, n, _ in models)}")
    print(f"  Estimated cost: ~${total_est:.2f}")
    print("=" * 100)

    clear_usage()
    clear_call_traces()

    results = run_nemotron_comparison(
        borrowers, base_lenders, api_key, mock=mock, models=models,
        resume_data=resume_data,
    )
    print_nemotron_results(results, borrowers, base_lenders, models=models)

    # Save results
    output_file = f"nemotron_generalization_{args.mix}_results.json"
    save_results(results, borrowers, base_lenders, models, args.mix, output_file)

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

    print(f"\n{'='*100}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
