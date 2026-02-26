#!/usr/bin/env python3
"""
Season test: 5-way competition with cheap models.

  - nvidia/llama-3.3-nemotron-super-49b-v1.5  ($0.20/M, proven strong)
  - qwen/qwen3-next-80b-a3b-instruct          ($0.09/M, new Qwen MoE)
  - z-ai/glm-4.7-flash                        ($0.06/M, Zhipu GLM)
  - z-ai/glm-4-32b                             ($0.10/M, Zhipu GLM 32B)
  - openai/gpt-4.1-nano                        ($0.10/M, compact baseline)

All under $0.20/M. 5 lenders, 5 weeks, 5 borrowers/week = 125 evaluations.
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from loanville.data import get_lenders
from loanville.models import ECONOMICS_PRESETS, SeasonConfig
from loanville.scoring import score_season, print_season_report
from loanville.season import SeasonEngine

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set. Add it to ~/.env")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────
MODELS = [
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5", "Nemotron-49B"),
    ("qwen/qwen3-next-80b-a3b-instruct",         "Qwen3-Next-80B"),
    ("z-ai/glm-4.7-flash",                        "GLM-4.7-Flash"),
    ("z-ai/glm-4-32b",                            "GLM-4-32B"),
    ("openai/gpt-4.1-nano",                        "GPT-4.1-Nano"),
]

WEEKS = 5
COHORT_SIZE = 5
MONTHS_PER_WEEK = 2
SEASON_MIX = "gentle"
ECONOMICS = "conservative"
SEED = 42

# ── Setup ─────────────────────────────────────────────────────────────────
lenders = get_lenders()
assert len(lenders) >= len(MODELS), f"Need {len(MODELS)} lenders, got {len(lenders)}"

for i, (model_id, display_name) in enumerate(MODELS):
    lenders[i].model = model_id
    lenders[i].name = f"{lenders[i].name.split('[')[0].strip()} [{display_name}]" \
        if "[" in lenders[i].name else f"{lenders[i].name} [{display_name}]"

# Only use the lenders we need
lenders = lenders[:len(MODELS)]

economics = ECONOMICS_PRESETS[ECONOMICS]

config = SeasonConfig(
    weeks=WEEKS,
    cohort_size=COHORT_SIZE,
    months_per_week=MONTHS_PER_WEEK,
    season_mix=SEASON_MIX,
    seed=SEED,
    speed_scoring=True,
    custom_tools=False,
    economics=economics,
)

# ── Banner ────────────────────────────────────────────────────────────────
print("=" * 70)
print("  LOANVILLE SEASON — 5-WAY COMPETITION")
print(f"  {WEEKS} weeks | {COHORT_SIZE}/week | {SEASON_MIX} mix | {ECONOMICS} economics")
print("=" * 70)
for i, (_, name) in enumerate(MODELS):
    print(f"  {i+1}. {lenders[i].name} (target {lenders[i].target_yield_pct}%, ${lenders[i].total_capital/1e6:.1f}M)")
print(f"\n  Mix: 70% good, 20% bad, 10% fraud")
print(f"  Total evaluations: {WEEKS * COHORT_SIZE * len(MODELS)}")
print()

# ── Run ───────────────────────────────────────────────────────────────────
t0 = time.time()

season = SeasonEngine(
    config=config,
    lenders=lenders,
    openrouter_api_key=API_KEY,
    mock=False,
    data_mode="lite",
)
asyncio.run(season.run_season())

elapsed = time.time() - t0

# ── Score ─────────────────────────────────────────────────────────────────
season_scores = score_season(season.lender_states, config)
print_season_report(season_scores)

# ── Leaderboard ──────────────────────────────────────────────────────────
from loanville.leaderboard import emit_match_record_from_season, emit_and_update
record = emit_match_record_from_season(
    season_engine=season,
    season_scores=season_scores,
    lenders=lenders,
    mix=SEASON_MIX,
)
match_path, lb_path = emit_and_update(record)
print(f"\n  Leaderboard: match -> {match_path.name}")
print(f"  Leaderboard: standings -> {lb_path.name}")

# ── Weekly trajectory ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  WEEKLY TRAJECTORY")
print("=" * 70)

print(f"\n  {'Wk':<4}", end="")
for l in lenders:
    short = l.name.split("[")[1].rstrip("]") if "[" in l.name else l.model.split("/")[-1]
    print(f" {short[:12]:>12s} {'D%':>4} {'Df':>3}", end="")
print()
print(f"  {'─'*(4 + len(lenders)*21)}")

for week in range(1, WEEKS + 1):
    print(f"  {week:<4}", end="")
    for state in season.lender_states.values():
        snap = None
        for s in state.weekly_snapshots:
            if s["week"] == week:
                snap = s
                break
        if snap:
            eff = snap["effective_capital"]
            depl = snap["deployed_capital"]
            ratio = depl / eff * 100 if eff > 0 else 0
            defaults = snap["defaults_to_date"]
            print(f" ${eff/1000:>8.0f}k {ratio:>3.0f}% {defaults:>3}", end="")
        else:
            print(f" {'—':>12} {'—':>4} {'—':>3}", end="")
    print()

# ── Head-to-head ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HEAD-TO-HEAD (sorted by score)")
print("=" * 70)

print(f"\n  {'Model':<30s} {'Persona':<12s} {'Score':>7} {'Won':>5} {'Rej':>5} {'Dflt':>5} {'Net P&L':>12}")
print(f"  {'─'*76}")
for state, score in sorted(
    zip(season.lender_states.values(), season_scores),
    key=lambda x: -x[1].final_score,
):
    frauds = sum(1 for loan in state.resolved_loans if loan.was_fraud and loan.defaulted)
    defaults = sum(1 for loan in state.resolved_loans if loan.defaulted)
    net = (
        state.cumulative_interest
        + state.cumulative_fees
        - state.cumulative_losses
        - state.cumulative_workout_cost
    )
    short = state.model.split("/")[-1][:28]
    persona = state.lender_name.split("[")[0].strip() if "[" in state.lender_name else ""
    print(f"  {short:<30s} {persona:<12s} {score.final_score:>6.1f} {state.deals_won:>5} "
          f"{state.deals_rejected:>5} {defaults:>5} ${net:>+10,.0f}")

scores_sorted = sorted(season_scores, key=lambda s: -s.final_score)
print(f"\n  Winner: {scores_sorted[0].lender_name} ({scores_sorted[0].final_score:.1f}/100)")
if len(scores_sorted) > 1:
    gap = scores_sorted[0].final_score - scores_sorted[1].final_score
    print(f"  Gap to 2nd: {gap:.1f} points")

print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/WEEKS:.0f}s per week)")
print(f"  Total evaluations: {WEEKS * COHORT_SIZE * len(lenders)}")
print()
