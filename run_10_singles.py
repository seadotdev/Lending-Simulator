#!/usr/bin/env python3
"""Run 10 single 4-way mock matches for Elo convergence testing."""

import asyncio
import os
import random
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from loanville.data import get_borrowers, get_lenders
from loanville.models import ECONOMICS_PRESETS
from loanville.engine import SimulationEngine
from loanville.scoring import score_lenders
from loanville.leaderboard import emit_match_record_from_sim, emit_and_update

MODELS = [
    ("openai/gpt-4.1-nano", "GPT-4.1-Nano"),
    ("google/gemini-2.5-flash", "Gemini-Flash"),
    ("deepseek/deepseek-chat-v3-0324", "DeepSeek-V3"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama-70B"),
]

economics = ECONOMICS_PRESETS["balanced"]

for i in range(10):
    seed = 200 + i
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  Match {i+1}/10: 4-way (seed={seed})")
    print(f"  {' vs '.join(m[1] for m in MODELS)}")
    print(f"{'='*60}")

    borrowers = get_borrowers("realistic")
    lenders = get_lenders()[:4]
    for j, (model_id, display_name) in enumerate(MODELS):
        lenders[j].model = model_id
        base = lenders[j].name.split('[')[0].strip() if '[' in lenders[j].name else lenders[j].name
        lenders[j].name = f"{base} [{display_name}]"

    t0 = time.time()

    engine = SimulationEngine(
        borrowers, lenders, mock=True, data_mode="lite",
        economics=economics,
    )
    asyncio.run(engine.run())

    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
        borrowers=borrowers,
        economics=economics,
        runs=engine.runs,
    )

    seen = {}
    models_info = []
    for l in lenders:
        raw = l.model
        seen[raw] = seen.get(raw, 0) + 1
        mid = raw if seen[raw] == 1 else f"{raw}::{seen[raw]}"
        models_info.append({"model_id": mid, "display_name": l.name})

    record = emit_match_record_from_sim(
        lenders=lenders,
        models_info=models_info,
        engine=engine,
        scores=scores,
        borrowers=borrowers,
        mix="realistic",
    )
    match_path, lb_path = emit_and_update(record)

    elapsed = time.time() - t0
    valid = record["validation"]["valid"]
    print(f"  -> {'VALID' if valid else 'INVALID'} | {match_path.name} ({elapsed:.1f}s)")
    for r in record["results"]:
        pnl = r["net_pnl"]
        print(f"     {r['model_id'].split('/')[-1]:.<30s} won={r['deals_won']:>2} rej={r['deals_rejected']:>2} pnl=${pnl:>+10,.0f}")

# Final summary
import json
lb = json.load(open(lb_path))
print(f"\n{'='*60}")
print(f"  FINAL LEADERBOARD ({lb['n_matches']} matches, {len(lb['standings'])} models)")
print(f"{'='*60}")
print(f"\n  {'#':<3} {'Model':<35} {'Elo':>7} {'Matches':>8} {'Avg P&L':>10}")
print(f"  {'─'*65}")
for i, s in enumerate(lb["standings"]):
    print(f"  {i+1:<3} {s['display_name']:<35} {s['composite_elo']:>7.1f} {s['matches_played']:>8} ${s['avg_net_pnl']:>+9,.0f}")
print()
