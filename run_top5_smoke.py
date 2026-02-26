#!/usr/bin/env python3
"""
Smoke test: top 5 leaderboard models, 1 lender each, 3 borrowers (good/bad/fraud).
Tests each model individually (no competition) to verify they work correctly.
"""

import asyncio
import os
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from loanville.data import get_lenders, get_borrowers
from loanville.models import ECONOMICS_PRESETS, LenderConfig
from loanville.engine import SimulationEngine

# Top 5 from leaderboard by composite Elo
MODELS = [
    ("deepseek/deepseek-chat-v3-0324",              "DeepSeek-V3"),
    ("meta-llama/llama-3.3-70b-instruct",            "Llama-3.3-70B"),
    ("openai/gpt-4.1-nano",                          "GPT-4.1-Nano"),
    ("qwen/qwen3-next-80b-a3b-instruct",             "Qwen3-Next-80B"),
    ("deepseek/deepseek-r1",                          "DeepSeek-R1"),
]

economics = ECONOMICS_PRESETS["conservative"]

# Use 3 known borrowers: 1 good, 1 bad, 1 fraud
all_borrowers = get_borrowers()
good_b = next(b for b in all_borrowers if b.true_outcome == "good")
bad_b = next(b for b in all_borrowers if b.true_outcome == "bad")
fraud_b = next(b for b in all_borrowers if b.true_outcome == "fraud")
test_borrowers = [good_b, bad_b, fraud_b]

print("=" * 80)
print("  SMOKE TEST — Top 5 Leaderboard Models")
print("=" * 80)
print(f"\n  Borrowers:")
for b in test_borrowers:
    print(f"    {b.id}: {b.dossier.company_name} ({b.dossier.sector}) — {b.true_outcome}")
print()

base_lender = get_lenders()[0]

results = []

for model_id, display_name in MODELS:
    print(f"\n{'─'*80}")
    print(f"  Testing: {display_name} ({model_id})")
    print(f"{'─'*80}")

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
            mock=False,
            data_mode="lite",
            los_mode="full",
            economics=economics,
        )
        asyncio.run(engine.run_origination())
        decisions = engine.all_decisions.get(lender.id, [])
    except Exception as e:
        error_msg = str(e)
        print(f"  ERROR: {error_msg}")

    elapsed = time.time() - t0

    approves = 0
    rejects = 0
    errors = 0
    rate_list = []

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
        print(f"    {bname} ({truth}): {detail}")
        if reasoning:
            print(f"      → {reasoning}")

    if error_msg:
        failure_mode = "CRASH"
    elif len(decisions) == 0:
        failure_mode = "NO_OUTPUT"
    elif errors > 0:
        failure_mode = f"PARSE_ERROR ({errors}/{len(decisions)})"
    elif rejects == len(decisions):
        failure_mode = "ALL_REJECT"
    elif approves == len(decisions):
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

    results.append({
        "model": display_name,
        "model_id": model_id,
        "approves": approves,
        "rejects": rejects,
        "errors": errors,
        "rates": rates_str,
        "failure_mode": failure_mode,
        "elapsed": elapsed,
    })

    print(f"  → {failure_mode} | {approves}A/{rejects}R/{errors}E | rates: {rates_str} | {elapsed:.1f}s")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("  SMOKE TEST SUMMARY — Top 5 Leaderboard")
print("=" * 80)

print(f"\n  {'Model':<25s} {'Status':<25s} {'A/R/E':>7} {'Rates':>15} {'Time':>6}")
print(f"  {'─'*78}")
for r in results:
    status_icon = "✓" if r["failure_mode"] == "OK" else "✗"
    print(f"  {r['model']:<25s} {status_icon} {r['failure_mode']:<23s} "
          f"{r['approves']}/{r['rejects']}/{r['errors']:>3} {r['rates']:>15} {r['elapsed']:>5.1f}s")
print()
