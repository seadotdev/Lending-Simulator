#!/usr/bin/env python3
"""
4x4 Leaderboard Test: 4 models, 4 lenders, 4 weeks.

Tests the full leaderboard pipeline with models that have proven reliable
in prior runs (most matches played, consistent behavior):

  1. openai/gpt-4.1-nano         (5 matches, highest credit Elo, cheap)
  2. nvidia/llama-3.3-nemotron-super-49b-v1.5  (5 matches, strong analytical)
  3. google/gemini-2.5-flash      (7 matches, highest deal share)
  4. deepseek/deepseek-chat-v3-0324  (4 matches, highest profit Elo)

Each model gets a distinct lender persona (Velocity, Heritage, Meridian, Pinnacle).
4 weeks x 5 borrowers/week = 80 total evaluations.
Realistic mix (55% good, 30% bad, 15% fraud) for meaningful scoring.

Purpose: validate end-to-end leaderboard ingestion from a season run.
"""

import asyncio
import os
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from loanville.data import get_lenders
from loanville.models import ECONOMICS_PRESETS, SeasonConfig
from loanville.scoring import score_season, print_season_report
from loanville.season import SeasonEngine

# ── Config ────────────────────────────────────────────────────────────────
MODELS = [
    ("openai/gpt-4.1-nano",                        "GPT-4.1-Nano"),
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5",   "Nemotron-49B"),
    ("google/gemini-2.5-flash",                     "Gemini-Flash"),
    ("deepseek/deepseek-chat-v3-0324",              "DeepSeek-V3"),
]

WEEKS = 4
COHORT_SIZE = 5          # 5 borrowers/week = 20 total per lender
MONTHS_PER_WEEK = 2      # 8 months of loan aging total
SEASON_MIX = "realistic"
ECONOMICS = "balanced"
SEED = 42

# ── Setup ─────────────────────────────────────────────────────────────────
lenders = get_lenders()
assert len(lenders) >= len(MODELS), f"Need {len(MODELS)} lenders, got {len(lenders)}"

for i, (model_id, display_name) in enumerate(MODELS):
    lenders[i].model = model_id
    lenders[i].name = f"{lenders[i].name.split('[')[0].strip()} [{display_name}]" \
        if "[" in lenders[i].name else f"{lenders[i].name} [{display_name}]"

# Only use the 4 lenders we need
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
print("  LOANVILLE SEASON — 4x4 LEADERBOARD TEST")
print(f"  {WEEKS} weeks | {COHORT_SIZE}/week | {SEASON_MIX} mix | {ECONOMICS} economics")
print("=" * 70)
for i, (_, name) in enumerate(MODELS):
    print(f"  {i+1}. {lenders[i].name}")
    print(f"     target {lenders[i].target_yield_pct}% | "
          f"${lenders[i].total_capital/1e6:.1f}M capital | "
          f"${lenders[i].max_single_loan/1e3:.0f}k max loan")
print(f"\n  Season mix: {SEASON_MIX}")
print(f"  Total evaluations: {WEEKS * COHORT_SIZE * len(MODELS)}")
print()

# ── Run ───────────────────────────────────────────────────────────────────
t0 = time.time()

season = SeasonEngine(
    config=config,
    lenders=lenders,
    mock=False,
    data_mode="lite",
    los_mode="full",
)
asyncio.run(season.run_season())

elapsed = time.time() - t0

# ── Score ─────────────────────────────────────────────────────────────────
season_scores = score_season(season.lender_states, config)
print_season_report(season_scores)
scores_by_id = {s.lender_id: s for s in season_scores}

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
for score in season_scores:
    state = season.lender_states.get(score.lender_id)
    if state is None:
        continue
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

# ── Leaderboard verification ────────────────────────────────────────────
print("\n" + "=" * 70)
print("  LEADERBOARD VERIFICATION")
print("=" * 70)

import json
with open(lb_path) as f:
    standings = json.load(f)

# Show updated Elo for the 4 models in this match
print(f"\n  Models updated: {len(record['models'])}")
print(f"  Match valid: {record['validation']['valid']}")
if record['validation']['errors']:
    print(f"  Validation errors: {record['validation']['errors']}")

print(f"\n  {'Model':<40s} {'Profit':>8} {'Credit':>8} {'Deal%':>8} {'Matches':>8}")
print(f"  {'─'*72}")
for entry in standings["standings"]:
    mid = entry["model_id"]
    # Highlight models from this match
    in_match = any(m["model_id"] == mid for m in record["models"])
    marker = " *" if in_match else "  "
    print(f" {marker}{mid:<39s} {entry['profit_elo']:>7.1f} {entry['credit_elo']:>7.1f} "
          f"{entry['dealshare_elo']:>7.1f} {entry['matches_played']:>8}")

print(f"\n  Total models on leaderboard: {len(standings['standings'])}")
print(f"  Total matches recorded: {standings['n_matches']}")
print(f"  * = updated in this match")

print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/WEEKS:.0f}s per week)")
print(f"  Total evaluations: {WEEKS * COHORT_SIZE * len(lenders)}")
print()
