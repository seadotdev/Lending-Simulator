#!/usr/bin/env python3
"""
Smoke test: 1 week, 3 borrowers per model — catalogue failure modes.

Tests each model individually (1 lender, no competition) to isolate
whether failures are model issues or competitive dynamics.
"""

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from loanville.data import get_lenders, get_borrowers
from loanville.models import ECONOMICS_PRESETS, LenderConfig
from loanville.engine import SimulationEngine

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set. Add it to ~/.env")
    sys.exit(1)

# Models to smoke test — mix of known-good, known-bad, and untested
MODELS = [
    # Known good
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5", "Nemotron-49B"),
    ("qwen/qwen3-next-80b-a3b-instruct",         "Qwen3-Next-80B"),
    ("z-ai/glm-4.7-flash",                        "GLM-4.7-Flash"),
    ("openai/gpt-4.1-nano",                        "GPT-4.1-Nano"),
    # Known failures / issues
    ("minimax/minimax-01",                         "MiniMax-01"),
    ("z-ai/glm-4-32b",                            "GLM-4-32B"),
    ("google/gemini-2.5-flash",                    "Gemini-Flash"),
    # Untested cheap models
    ("minimax/minimax-m2",                         "MiniMax-M2"),
    ("qwen/qwen3-30b-a3b",                        "Qwen3-30B"),
    ("qwen/qwen3-32b",                            "Qwen3-32B"),
    ("moonshotai/kimi-k2-0905",                   "Kimi-K2"),
]

economics = ECONOMICS_PRESETS["conservative"]

# Use 3 known borrowers: 1 good, 1 bad, 1 fraud
all_borrowers = get_borrowers()
# Pick specific borrowers by outcome
good_b = next(b for b in all_borrowers if b.true_outcome == "good")
bad_b = next(b for b in all_borrowers if b.true_outcome == "bad")
fraud_b = next(b for b in all_borrowers if b.true_outcome == "fraud")
test_borrowers = [good_b, bad_b, fraud_b]

print("=" * 80)
print("  SMOKE TEST — 1 lender per model, 3 borrowers (good/bad/fraud)")
print("=" * 80)
print(f"\n  Borrowers:")
for b in test_borrowers:
    print(f"    {b.id}: {b.dossier.company_name} ({b.dossier.sector}) — {b.true_outcome}")
print()

# Use the first lender config as template (Velocity Capital, 14% target)
base_lender = get_lenders()[0]

results = []

for model_id, display_name in MODELS:
    print(f"\n{'─'*80}")
    print(f"  Testing: {display_name} ({model_id})")
    print(f"{'─'*80}")

    # Create a single-lender config
    lender = LenderConfig(
        id=base_lender.id,
        name=f"Test [{display_name}]",
        persona=base_lender.persona,
        model=model_id,
        target_yield_pct=base_lender.target_yield_pct,
        max_single_loan=base_lender.max_single_loan,
        total_capital=base_lender.total_capital,
        sector_limits=base_lender.sector_limits,
        existing_portfolio=base_lender.existing_portfolio,
    )

    t0 = time.time()
    error_msg = None
    decisions = []

    try:
        engine = SimulationEngine(
            borrowers=test_borrowers,
            lenders=[lender],
            openrouter_api_key=API_KEY,
            mock=False,
            data_mode="lite",
            economics=economics,
        )
        asyncio.run(engine.run_origination())
        decisions = engine.all_decisions.get(lender.id, [])
    except Exception as e:
        error_msg = str(e)
        print(f"  ERROR: {error_msg}")

    elapsed = time.time() - t0

    # Analyse decisions
    approves = 0
    rejects = 0
    errors = 0
    rate_list = []
    decision_details = []

    for d in decisions:
        borrower = next((b for b in test_borrowers if b.id == d.borrower_id), None)
        truth = borrower.true_outcome if borrower else "?"
        bname = borrower.dossier.company_name if borrower else d.borrower_id

        if d.decision == "APPROVE" and d.term_sheet:
            approves += 1
            rate_list.append(d.term_sheet.interest_rate)
            detail = f"APPROVE @ {d.term_sheet.interest_rate:.1f}% {d.term_sheet.term_months}mo"
        elif d.decision == "REJECT":
            rejects += 1
            detail = "REJECT"
        else:
            errors += 1
            detail = f"ERROR/INVALID ({d.decision})"

        reasoning = (d.reasoning or "")[:80]
        decision_details.append({
            "borrower": bname,
            "truth": truth,
            "decision": detail,
            "reasoning": reasoning,
        })
        print(f"    {bname} ({truth}): {detail}")
        if reasoning:
            print(f"      → {reasoning}")

    # Classify failure mode
    if error_msg:
        failure_mode = "CRASH"
    elif len(decisions) == 0:
        failure_mode = "NO_OUTPUT"
    elif errors > 0:
        failure_mode = f"PARSE_ERROR ({errors}/{len(decisions)})"
    elif rejects == len(decisions):
        failure_mode = "ALL_REJECT"
    elif approves == len(decisions):
        # Check if they approved fraud
        fraud_approved = any(
            d.decision == "APPROVE"
            for d in decisions
            for b in test_borrowers
            if b.id == d.borrower_id and b.true_outcome == "fraud"
        )
        failure_mode = "ALL_APPROVE" + (" (FRAUD_FUNDED!)" if fraud_approved else "")
    else:
        failure_mode = "OK"

    rates_str = "/".join(f"{r:.1f}" for r in rate_list) if rate_list else "—"

    result = {
        "model": display_name,
        "model_id": model_id,
        "approves": approves,
        "rejects": rejects,
        "errors": errors,
        "rates": rates_str,
        "failure_mode": failure_mode,
        "elapsed": elapsed,
        "details": decision_details,
    }
    results.append(result)

    print(f"  → {failure_mode} | {approves}A/{rejects}R/{errors}E | rates: {rates_str} | {elapsed:.1f}s")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("  SMOKE TEST SUMMARY")
print("=" * 80)

print(f"\n  {'Model':<25s} {'Status':<25s} {'A/R/E':>7} {'Rates':>15} {'Time':>6}")
print(f"  {'─'*78}")
for r in results:
    status_icon = "✓" if r["failure_mode"] == "OK" else "✗"
    print(f"  {r['model']:<25s} {status_icon} {r['failure_mode']:<23s} "
          f"{r['approves']}/{r['rejects']}/{r['errors']:>3} {r['rates']:>15} {r['elapsed']:>5.1f}s")

# Catalogue failures
failures = [r for r in results if r["failure_mode"] not in ("OK",)]
if failures:
    print(f"\n  FAILURE CATALOGUE ({len(failures)} models):")
    for r in failures:
        print(f"\n    {r['model']} ({r['model_id']})")
        print(f"    Mode: {r['failure_mode']}")
        for d in r["details"]:
            print(f"      {d['borrower']} ({d['truth']}): {d['decision']}")
            if d["reasoning"]:
                print(f"        → {d['reasoning']}")

print()
