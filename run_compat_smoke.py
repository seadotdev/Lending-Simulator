#!/usr/bin/env python3
"""
Compatibility smoke test: 18 untested models across risk tiers.

Tests each model with 3 borrowers (good/bad/fraud) to map which models
work with our forced tool_choice LOS, and which fail.

Groups:
  1. Llama 4 — did Meta fix tool calling?
  2. Open-weight — is it a Llama/DeepInfra bug or all open-weight?
  3. Reasoning — does think-stripping generalize?
  4. Native API budget — confirm they just work
  5. Chinese/new vendors — unknown territory
"""

import asyncio
import os
import time
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from loanville.data import get_lenders, get_borrowers
from loanville.models import ECONOMICS_PRESETS, LenderConfig
from loanville.engine import SimulationEngine

# ── Models to test, organized by hypothesis ──────────────────────────────
GROUPS = [
    ("Llama 4 — tool calling fixed?", [
        ("meta-llama/llama-4-scout",                    "Llama-4-Scout"),
        ("meta-llama/llama-4-maverick",                 "Llama-4-Maverick"),
        ("meta-llama/llama-3.1-8b-instruct",            "Llama-3.1-8B"),
    ]),
    ("Open-weight on 3rd-party inference", [
        ("google/gemma-3-27b-it",                       "Gemma-3-27B"),
        ("nvidia/llama-3.3-nemotron-super-49b-v1.5",    "Nemotron-Super-49B"),
        ("nvidia/nemotron-nano-9b-v2",                  "Nemotron-Nano-9B-v2"),
    ]),
    ("Reasoning models — think-stripping", [
        ("qwen/qwq-32b",                               "QwQ-32B"),
        ("deepseek/deepseek-r1-0528",                   "DeepSeek-R1-0528"),
        ("x-ai/grok-3-mini",                            "Grok-3-Mini"),
    ]),
    ("Native API budget — should just work", [
        ("openai/gpt-5-nano",                           "GPT-5-Nano"),
        ("google/gemini-2.5-flash",                     "Gemini-2.5-Flash"),
        ("mistralai/mistral-small-3.2-24b-instruct",    "Mistral-Small-3.2"),
    ]),
    ("Chinese/new vendors — unknown", [
        ("z-ai/glm-4.7-flash",                         "GLM-4.7-Flash"),
        ("minimax/minimax-m2.5",                        "MiniMax-M2.5"),
        ("bytedance-seed/seed-1.6-flash",               "Seed-1.6-Flash"),
        ("moonshotai/kimi-k2",                          "Kimi-K2"),
        ("xiaomi/mimo-v2-flash",                        "MiMo-v2-Flash"),
        ("baidu/ernie-4.5-21b-a3b",                     "ERNIE-4.5"),
    ]),
]

economics = ECONOMICS_PRESETS["conservative"]

# Use 3 known borrowers: 1 good, 1 bad, 1 fraud
all_borrowers = get_borrowers()
good_b = next(b for b in all_borrowers if b.true_outcome == "good")
bad_b = next(b for b in all_borrowers if b.true_outcome == "bad")
fraud_b = next(b for b in all_borrowers if b.true_outcome == "fraud")
test_borrowers = [good_b, bad_b, fraud_b]

print("=" * 90)
print("  COMPATIBILITY SMOKE TEST — 18 Models Across Risk Tiers")
print("=" * 90)
print(f"\n  Borrowers:")
for b in test_borrowers:
    print(f"    {b.id}: {b.dossier.company_name} ({b.dossier.sector}) — {b.true_outcome}")
print()

base_lender = get_lenders()[0]

all_results = []

for group_name, models in GROUPS:
    print(f"\n{'='*90}")
    print(f"  GROUP: {group_name}")
    print(f"{'='*90}")

    for model_id, display_name in models:
        print(f"\n{'─'*90}")
        print(f"  Testing: {display_name} ({model_id})")
        print(f"{'─'*90}")

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
            print(f"  ERROR: {error_msg[:200]}")

        elapsed = time.time() - t0

        approves = 0
        rejects = 0
        llm_errors = 0
        other_errors = 0
        rate_list = []

        for d in decisions:
            borrower = next((b for b in test_borrowers if b.id == d.borrower_id), None)
            truth = borrower.true_outcome if borrower else "?"
            bname = borrower.dossier.company_name if borrower else d.borrower_id

            is_llm_error = d.reasoning and d.reasoning.startswith("[LLM_ERROR]")

            if is_llm_error:
                llm_errors += 1
                detail = "LLM_ERROR"
                reasoning = (d.reasoning or "")[12:92]
            elif d.decision == "APPROVE" and d.term_sheet:
                approves += 1
                rate_list.append(d.term_sheet.interest_rate)
                detail = f"APPROVE @ {d.term_sheet.interest_rate:.1f}% {d.term_sheet.term_months}mo"
                reasoning = (d.reasoning or "")[:80]
            elif d.decision == "REJECT":
                rejects += 1
                detail = "REJECT"
                reasoning = (d.reasoning or "")[:80]
            else:
                other_errors += 1
                detail = f"OTHER ({d.decision})"
                reasoning = (d.reasoning or "")[:80]

            print(f"    {bname} ({truth}): {detail}")
            if reasoning:
                print(f"      > {reasoning}")

        # Classify result
        total = len(decisions)
        if error_msg:
            tier = "CRASH"
        elif total == 0:
            tier = "NO_OUTPUT"
        elif llm_errors == total:
            tier = "ALL_LLM_ERROR"
        elif llm_errors > 0:
            tier = f"PARTIAL_LLM_ERROR ({llm_errors}/{total})"
        elif other_errors > 0:
            tier = f"PARSE_ERROR ({other_errors}/{total})"
        elif rejects == total:
            tier = "ALL_REJECT"
        elif approves == total:
            tier = "ALL_APPROVE"
        else:
            tier = "OK"

        rates_str = "/".join(f"{r:.1f}" for r in rate_list) if rate_list else "—"

        result = {
            "group": group_name,
            "model": display_name,
            "model_id": model_id,
            "approves": approves,
            "rejects": rejects,
            "llm_errors": llm_errors,
            "other_errors": other_errors,
            "rates": rates_str,
            "tier": tier,
            "elapsed": elapsed,
            "crash": error_msg,
        }
        all_results.append(result)

        icon = "✓" if tier == "OK" else ("~" if "REJECT" in tier or "APPROVE" in tier else "✗")
        print(f"  => {icon} {tier} | {approves}A/{rejects}R/{llm_errors}E | rates: {rates_str} | {elapsed:.1f}s")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 90)
print("  COMPATIBILITY SMOKE TEST RESULTS")
print("=" * 90)

print(f"\n  {'Model':<25s} {'Group':<20s} {'Status':<25s} {'A/R/E':>7} {'Time':>6}")
print(f"  {'─'*90}")

for r in all_results:
    icon = "✓" if r["tier"] == "OK" else ("~" if "REJECT" in r["tier"] or "APPROVE" in r["tier"] else "✗")
    group_short = r["group"].split("—")[0].strip()[:18]
    print(f"  {r['model']:<25s} {group_short:<20s} {icon} {r['tier']:<23s} "
          f"{r['approves']}/{r['rejects']}/{r['llm_errors']:>3} {r['elapsed']:>5.1f}s")

# ── Risk tier summary ────────────────────────────────────────────────────
print(f"\n  {'─'*90}")
ok = sum(1 for r in all_results if r["tier"] == "OK")
partial = sum(1 for r in all_results if "PARTIAL" in r["tier"])
broken = sum(1 for r in all_results if r["tier"] in ("ALL_LLM_ERROR", "CRASH", "NO_OUTPUT"))
edge = sum(1 for r in all_results if r["tier"] in ("ALL_REJECT", "ALL_APPROVE", "PARSE_ERROR"))
total_models = len(all_results)

print(f"\n  Working (OK):          {ok}/{total_models}")
print(f"  Partial errors:        {partial}/{total_models}")
print(f"  Fully broken:          {broken}/{total_models}")
print(f"  Edge cases:            {edge}/{total_models}")

# ── Save JSON results ────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
out_path = f"compat_smoke_{ts}.json"
with open(out_path, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "borrowers": [b.id for b in test_borrowers],
        "results": all_results,
    }, f, indent=2, default=str)
print(f"\n  Results saved to: {out_path}")
print()
