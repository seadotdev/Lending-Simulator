#!/usr/bin/env python3
"""
Season test: conservative scenario, 3 competitive models.

Gentle mix: 70% good, 20% bad, 10% fraud — rewards good underwriting.
Conservative economics: higher fraud penalties, tighter default thresholds.

Three models that showed close outcomes in legacy ELO tournaments:
  - nvidia/llama-3.3-nemotron-super-49b-v1.5  (strong analytical, close to Gemini)
  - google/gemini-2.5-flash                    (strong all-around)
  - openai/gpt-4.1-nano                        (compact but sharp)

Each model gets one lender slot — pure 3-way head-to-head.

Notes:
  - LLM non-determinism causes score swings across runs despite fixed seed=42
    (seed controls borrower generation, not LLM outputs).
  - Results are emitted to the leaderboard (leaderboard/matches/) for Elo tracking.
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
MODEL_A = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
MODEL_B = "google/gemini-2.5-flash"
MODEL_C = "openai/gpt-4.1-nano"
NAME_A = "Nemotron-49B"
NAME_B = "Gemini-Flash"
NAME_C = "GPT-4.1-Nano"

WEEKS = 6
COHORT_SIZE = 5          # 5 borrowers/week = 30 total evaluations per lender
MONTHS_PER_WEEK = 2      # 12 months of loan aging total
SEASON_MIX = "gentle"
ECONOMICS = "conservative"
SEED = 42

# ── Setup ─────────────────────────────────────────────────────────────────
lenders = get_lenders()
lenders[0].model = MODEL_A
lenders[0].name = f"Velocity [{NAME_A}]"
lenders[1].model = MODEL_B
lenders[1].name = f"Heritage [{NAME_B}]"
lenders[2].model = MODEL_C
lenders[2].name = f"Meridian [{NAME_C}]"

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
print("  LOANVILLE SEASON — CONSERVATIVE UNDERWRITING TEST")
print(f"  {WEEKS} weeks | {COHORT_SIZE}/week | {SEASON_MIX} mix | {ECONOMICS} economics")
print(f"  {NAME_A} vs {NAME_B} vs {NAME_C}")
print("=" * 70)
print(f"\n  Mix: 70% good, 20% bad, 10% fraud")
print(f"  Conservative: high fraud penalty, tight default threshold")
print(f"  Best underwriting judgment wins — not just volume.")
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

# ── Deep dive: per-week trajectory ────────────────────────────────────────
print("\n" + "=" * 70)
print("  WEEKLY TRAJECTORY")
print("=" * 70)

print(f"\n  {'Week':<6}", end="")
for l in lenders:
    short = l.name.split("[")[1].rstrip("]") if "[" in l.name else l.model.split("/")[-1]
    print(f"  {short:>12s} {'Depl%':>6} {'Dflt':>5}", end="")
print()
print(f"  {'─'*78}")

for week in range(1, WEEKS + 1):
    print(f"  {week:<6}", end="")
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
            print(f"  ${eff/1000:>8.0f}k {ratio:>5.0f}% {defaults:>5}", end="")
        else:
            print(f"  {'—':>12} {'—':>6} {'—':>5}", end="")
    print()

# ── Per-lender detail ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PER-LENDER DETAIL")
print("=" * 70)

for state, score in zip(season.lender_states.values(), season_scores):
    net = state.cumulative_interest + state.cumulative_fees - state.cumulative_losses
    frauds = sum(1 for loan in state.resolved_loans if loan.was_fraud and loan.defaulted)
    defaults = sum(1 for loan in state.resolved_loans if loan.defaulted and not loan.was_fraud)
    good_repaid = sum(1 for loan in state.resolved_loans if not loan.defaulted)
    print(f"\n  {state.lender_name}")
    print(f"    Score: {score.final_score:.1f}/100 (Credit: {score.credit_quality_score:.1f}, "
          f"Portfolio: {score.portfolio_mgmt_score:.1f}, Efficiency: {score.efficiency_score:.1f})")
    print(f"    Deals: {state.deals_won} won, {state.deals_lost} lost, {state.deals_rejected} rejected")
    print(f"    Outcomes: {good_repaid} repaid, {defaults} defaults, {frauds} fraud defaults")
    print(f"    P&L: ${net:>+,.0f} (interest: ${state.cumulative_interest:,.0f}, "
          f"fees: ${state.cumulative_fees:,.0f}, losses: -${state.cumulative_losses:,.0f})")

# ── Model comparison summary ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HEAD-TO-HEAD")
print("=" * 70)

print(f"\n  {'Model':<35s} {'Score':>8} {'Won':>5} {'Rej':>5} {'Dflt':>5} {'Fraud':>6} {'Net P&L':>12}")
print(f"  {'─'*76}")
for state, score in sorted(
    zip(season.lender_states.values(), season_scores),
    key=lambda x: -x[1].final_score,
):
    frauds = sum(1 for loan in state.resolved_loans if loan.was_fraud and loan.defaulted)
    defaults = sum(1 for loan in state.resolved_loans if loan.defaulted)
    net = state.cumulative_interest + state.cumulative_fees - state.cumulative_losses
    short = state.model.split("/")[-1]
    print(f"  {short:<35s} {score.final_score:>7.1f} {state.deals_won:>5} "
          f"{state.deals_rejected:>5} {defaults:>5} {frauds:>6} ${net:>+10,.0f}")

scores_sorted = sorted(season_scores, key=lambda s: -s.final_score)
print(f"\n  Winner: {scores_sorted[0].lender_name} ({scores_sorted[0].final_score:.1f}/100)")
if len(scores_sorted) > 1:
    gap = scores_sorted[0].final_score - scores_sorted[1].final_score
    print(f"  Gap to 2nd: {gap:.1f} points")

print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/WEEKS:.0f}s per week)")
print(f"  Total evaluations: {WEEKS * COHORT_SIZE * len(lenders)}")
print()
