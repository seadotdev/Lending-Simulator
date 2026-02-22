#!/usr/bin/env python3
"""
Elo-rated tournament benchmark for Loanville lending models.

Instead of testing each model against fixed competitors, models compete
head-to-head in randomly matched triplets.  All three lenders get identical
parameters (capital, limits, persona) so the only variable is the model.

Elo ratings are computed per-applicant: for each borrower, every pair of
models is compared on their hypothetical profit (approve at their terms vs
reject).  This gives N_borrowers x C(3,2) pairwise signals per match
instead of a single aggregate comparison — dramatically faster convergence.

Usage:
  python elo_benchmark.py --mix analyst --matches 50
  python elo_benchmark.py --mix analyst --matches 80 --resume elo_results.json
  python elo_benchmark.py --standings elo_results.json
"""

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from loanville.data import get_borrowers, MIX_PRESETS
from loanville.engine import SimulationEngine
from loanville.llm import MODEL_PRICING, clear_usage, get_cost_summary, get_token_usage
from loanville.models import Borrower, LenderConfig
from loanville.scoring import compute_loan_payoff, score_lenders

# Import model lists from benchmark_models
from benchmark_models import SMALL_MODELS, BENCHMARK_MODELS

DEFAULT_K = 32
INITIAL_ELO = 1500


# ---------------------------------------------------------------------------
# Lender factory — identical config for all three slots
# ---------------------------------------------------------------------------

def make_lender(slot: int, model_id: str, display_name: str) -> LenderConfig:
    """Create a lender config for the Elo tournament.

    All lenders get identical parameters so the only differentiator is the
    model's analytical and pricing ability.
    """
    return LenderConfig(
        id=f"ELO-{slot:03d}",
        name=f"Lender {slot} [{display_name}]",
        persona=(
            "You are a middle-market commercial lender evaluating loan applications. "
            "Your goal is to maximize risk-adjusted returns by approving creditworthy "
            "borrowers at appropriate interest rates while rejecting borrowers who "
            "are unlikely to repay. Carefully analyze each borrower's financial "
            "statements, cash flow, debt service capacity, and business fundamentals. "
            "You offer terms between 12-30 months with competitive interest rates."
        ),
        model=model_id,
        target_yield_pct=11.0,
        max_single_loan=700000,
        total_capital=3500000,
        sector_limits={},  # defaults to 0.25 for all sectors
        existing_portfolio=[],  # clean book — no legacy position bias
    )


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def _compute_per_applicant_payoffs(
    borrowers: list[Borrower],
    lenders: list[LenderConfig],
    models: list[tuple[str, str]],
    engine: "SimulationEngine",
) -> dict[str, dict[str, float]]:
    """Compute competitive per-borrower profit for each model.

    Implements a per-borrower allocation rule (winner-takes-deal):
      1. Each model submits (approve/decline, APR).
      2. If nobody approves → all get 0 (no deal).
      3. If one or more approve → borrower takes the lowest APR offer.
      4. Only the winner gets realized payoff; losers get 0.
      5. Decline = 0 (distinct from losing a deal you bid on).

    This makes per-applicant Elo a "market share × profitability" competition,
    not just parallel-universe underwriting.

    Returns:
      model_id -> {borrower_id -> payoff}
      Also includes a "decision" key per borrower for tie-rule logic:
        "declined", "won", "lost"
    """
    borrower_map = {b.id: b for b in borrowers}

    # First pass: collect all decisions indexed by borrower
    # model_id -> lender decision for that borrower
    model_decisions: dict[str, dict[str, "LenderDecision"]] = {}
    model_id_for_lender: dict[str, str] = {}
    for lender, (model_id, _) in zip(lenders, models):
        model_id_for_lender[lender.id] = model_id
        decisions = engine.all_decisions.get(lender.id, [])
        dec_map = {}
        for d in decisions:
            dec_map[d.borrower_id] = d
        model_decisions[model_id] = dec_map

    payoffs: dict[str, dict[str, float]] = {mid: {} for mid, _ in models}
    # Also track decision states for Elo tie rules
    decision_states: dict[str, dict[str, str]] = {mid: {} for mid, _ in models}

    for b in borrowers:
        bid = b.id

        # Collect approvals with their rates
        approvals = []  # (model_id, decision, rate)
        for model_id, _ in models:
            d = model_decisions[model_id].get(bid)
            if d and d.decision == "APPROVE" and d.term_sheet:
                approvals.append((model_id, d, d.term_sheet.interest_rate))

        if not approvals:
            # Nobody approved — all declined, all get 0
            for model_id, _ in models:
                payoffs[model_id][bid] = 0.0
                decision_states[model_id][bid] = "declined"
            continue

        # Winner = lowest APR (ties broken by higher loan amount, matching engine.py)
        approvals.sort(key=lambda x: (x[2], -x[1].term_sheet.loan_amount))
        winner_model_id, winner_decision, _ = approvals[0]

        # Compute winner's realized payoff
        result = compute_loan_payoff(
            principal=winner_decision.term_sheet.loan_amount,
            interest_rate=winner_decision.term_sheet.interest_rate,
            term_months=winner_decision.term_sheet.term_months,
            true_outcome=b.true_outcome,
            months_before_default=b.months_before_default,
        )

        for model_id, _ in models:
            d = model_decisions[model_id].get(bid)
            if model_id == winner_model_id:
                payoffs[model_id][bid] = result["net_profit"]
                decision_states[model_id][bid] = "won"
            elif d and d.decision == "APPROVE" and d.term_sheet:
                # Approved but lost the deal — bid cost = 0, but distinct state
                payoffs[model_id][bid] = 0.0
                decision_states[model_id][bid] = "lost"
            else:
                # Declined
                payoffs[model_id][bid] = 0.0
                decision_states[model_id][bid] = "declined"

    # Attach decision_states alongside payoffs for Elo tie-rule logic
    for model_id, _ in models:
        payoffs[model_id]["_decision_states"] = decision_states[model_id]

    return payoffs


def run_match(
    models: list[tuple[str, str]],
    mix: str,
    api_key: str,
) -> list[dict]:
    """Run a single 3-way match.  Returns per-model results sorted by score."""
    borrowers = get_borrowers(mix)

    lenders = [
        make_lender(i + 1, model_id, name)
        for i, (model_id, name) in enumerate(models)
    ]

    clear_usage()
    engine = SimulationEngine(borrowers, lenders, api_key, mock=False)

    # Suppress the engine's verbose phase-by-phase output
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(engine.run())

    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
        borrowers=borrowers,
    )

    costs = get_cost_summary()

    # Compute per-applicant payoffs for pairwise Elo
    per_applicant = _compute_per_applicant_payoffs(
        borrowers, lenders, models, engine,
    )

    results = []
    for lender, (model_id, display_name) in zip(lenders, models):
        sc = next(s for s in scores if s.lender_id == lender.id)
        decisions = engine.all_decisions.get(lender.id, [])
        approvals = sum(1 for d in decisions if d.decision == "APPROVE")

        results.append({
            "model": model_id,
            "display_name": display_name,
            "score": sc.final_adjusted_score,
            "raroc_score": sc.raroc_score,
            "deals_won": sc.deals_won,
            "frauds_funded": sc.frauds_funded,
            "defaults": sc.defaults_count,
            "deployed": sc.total_deployed,
            "net_pnl": sc.net_return,
            "approvals": approvals,
            "cost": costs.get(model_id, 0.0),
            "per_applicant_payoffs": per_applicant.get(model_id, {}),
        })

    return results


# ---------------------------------------------------------------------------
# Elo math
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo_batch(
    ratings: dict[str, float],
    match_results: list[dict],
    k: float = DEFAULT_K,
) -> dict[str, float]:
    """Update Elo ratings using aggregate score (legacy batch mode).

    Every pair of players in the match generates a pairwise Elo update.
    K is scaled by 1/(n-1) so the total adjustment per match stays bounded.
    """
    new_ratings = dict(ratings)
    n = len(match_results)
    pair_k = k / (n - 1)

    for i in range(n):
        for j in range(i + 1, n):
            mi = match_results[i]["model"]
            mj = match_results[j]["model"]

            si = match_results[i]["score"]
            sj = match_results[j]["score"]
            if si > sj:
                actual_i, actual_j = 1.0, 0.0
            elif si == sj:
                actual_i, actual_j = 0.5, 0.5
            else:
                actual_i, actual_j = 0.0, 1.0

            exp_i = expected_score(ratings[mi], ratings[mj])
            exp_j = 1.0 - exp_i

            new_ratings[mi] += pair_k * (actual_i - exp_i)
            new_ratings[mj] += pair_k * (actual_j - exp_j)

    return new_ratings


def update_elo(
    ratings: dict[str, float],
    match_results: list[dict],
    k: float = DEFAULT_K,
) -> dict[str, float]:
    """Update Elo ratings using per-applicant pairwise comparisons.

    For each borrower, every pair of models generates an Elo signal based
    on competitive outcomes (not hypothetical parallel-universe payoffs).

    Tie rules (per borrower, for models A vs B):
      - Both declined          → tie (0.5 / 0.5) — neither acted
      - One won, other declined → winner wins, decliner loses
      - One won, other lost    → winner wins, loser loses
      - Both lost (third model won) → tie — neither earned anything
      - One lost, other declined → tie — neither earned, but losing
        a deal you bid on is not worse than not bidding

    K is scaled so total Elo movement per match stays bounded:
      pair_k = K / ((n-1) * n_borrowers)

    Falls back to batch mode if per_applicant_payoffs are not available.
    """
    # Check if per-applicant data is available
    if not match_results or "per_applicant_payoffs" not in match_results[0]:
        return update_elo_batch(ratings, match_results, k=k)

    # Collect all borrower IDs (union across all models, excluding metadata keys)
    all_borrower_ids: set[str] = set()
    for r in match_results:
        for key in r.get("per_applicant_payoffs", {}):
            if not key.startswith("_"):
                all_borrower_ids.add(key)

    if not all_borrower_ids:
        return update_elo_batch(ratings, match_results, k=k)

    new_ratings = dict(ratings)
    n = len(match_results)
    n_borrowers = len(all_borrower_ids)

    # Scale K: total Elo movement per match ≈ K (same as batch)
    pair_k = k / ((n - 1) * n_borrowers)

    for bid in all_borrower_ids:
        for i in range(n):
            for j in range(i + 1, n):
                mi = match_results[i]["model"]
                mj = match_results[j]["model"]

                pi = match_results[i]["per_applicant_payoffs"].get(bid, 0.0)
                pj = match_results[j]["per_applicant_payoffs"].get(bid, 0.0)

                # Get decision states for tie-rule logic
                states_i = match_results[i]["per_applicant_payoffs"].get("_decision_states", {})
                states_j = match_results[j]["per_applicant_payoffs"].get("_decision_states", {})
                si = states_i.get(bid, "declined")
                sj = states_j.get(bid, "declined")

                # Apply tie rules
                if si == "declined" and sj == "declined":
                    # Both declined → tie
                    actual_i, actual_j = 0.5, 0.5
                elif si == "won" and sj == "won":
                    # Shouldn't happen (only one winner), but defensive
                    actual_i, actual_j = 0.5, 0.5
                elif si == "won":
                    # i won the deal: compare on realized payoff vs 0
                    # Winner always "wins" the Elo matchup (even if payoff is
                    # negative — winning a bad deal is the model's fault)
                    actual_i, actual_j = 1.0, 0.0
                elif sj == "won":
                    actual_i, actual_j = 0.0, 1.0
                elif si == "lost" and sj == "lost":
                    # Both approved but a third model won → tie
                    actual_i, actual_j = 0.5, 0.5
                elif si == "lost" and sj == "declined":
                    # i bid and lost, j didn't bid → tie
                    # (not bidding is not better or worse than losing)
                    actual_i, actual_j = 0.5, 0.5
                elif si == "declined" and sj == "lost":
                    actual_i, actual_j = 0.5, 0.5
                else:
                    # Fallback: compare on payoff
                    if pi > pj:
                        actual_i, actual_j = 1.0, 0.0
                    elif pi == pj:
                        actual_i, actual_j = 0.5, 0.5
                    else:
                        actual_i, actual_j = 0.0, 1.0

                exp_i = expected_score(ratings[mi], ratings[mj])
                exp_j = 1.0 - exp_i

                new_ratings[mi] += pair_k * (actual_i - exp_i)
                new_ratings[mj] += pair_k * (actual_j - exp_j)

    return new_ratings


# ---------------------------------------------------------------------------
# Matchup generation
# ---------------------------------------------------------------------------

def generate_matchups(
    models: list[tuple[str, str]],
    n_matches: int,
    seed: int = 42,
) -> list[list[tuple[str, str]]]:
    """Generate random triplet matchups with balanced participation.

    Models with fewer scheduled matches get priority, ensuring every model
    appears roughly the same number of times.
    """
    rng = random.Random(seed)
    matchups = []
    play_count = {m[0]: 0 for m in models}

    for _ in range(n_matches):
        candidates = list(models)
        # Shuffle, then stable-sort by play count so least-played come first
        rng.shuffle(candidates)
        candidates.sort(key=lambda m: play_count[m[0]])
        selected = candidates[:3]
        rng.shuffle(selected)
        matchups.append(selected)
        for m in selected:
            play_count[m[0]] += 1

    return matchups


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    models: list[tuple[str, str]],
    n_matches: int,
    mix: str,
    api_key: str,
    k: float = DEFAULT_K,
    output_file: str = "elo_results.json",
    resume_data: dict | None = None,
) -> dict:
    """Run the full Elo tournament."""
    ratings = {m[0]: float(INITIAL_ELO) for m in models}
    match_log: list[dict] = []
    total_cost = 0.0
    completed = 0

    if resume_data:
        ratings = {k_: float(v) for k_, v in resume_data.get("ratings", {}).items()}
        match_log = list(resume_data.get("match_log", []))
        total_cost = resume_data.get("total_cost", 0.0)
        completed = len(match_log)

    remaining = n_matches - completed
    if remaining <= 0:
        print(f"Already completed {completed} matches.  Nothing to do.")
        return _build_output(ratings, match_log, models, mix, total_cost)

    matchups = generate_matchups(models, remaining, seed=42 + completed)
    display_map = {m[0]: m[1] for m in models}

    print(f"\n{'='*70}")
    print(f"  LOANVILLE ELO TOURNAMENT")
    print(f"  Models: {len(models)} | Mix: {mix} | Matches: {n_matches} | K={k}")
    if completed:
        print(f"  Resuming from match {completed + 1}")
    print(f"{'='*70}\n")

    for idx, triplet in enumerate(matchups):
        match_num = completed + idx + 1
        names = [m[1] for m in triplet]
        print(f"  Match {match_num}/{n_matches}: "
              f"{names[0]} vs {names[1]} vs {names[2]}  ", end="", flush=True)

        try:
            t0 = time.time()
            results = run_match(triplet, mix, api_key)
            elapsed = time.time() - t0

            old_ratings = dict(ratings)
            ratings = update_elo(ratings, results, k=k)

            match_cost = sum(r["cost"] for r in results)
            total_cost += match_cost

            ranked = sorted(results, key=lambda r: r["score"], reverse=True)
            winner = ranked[0]
            delta_w = ratings[winner["model"]] - old_ratings[winner["model"]]

            print(f"[{elapsed:.0f}s ${match_cost:.3f}]")
            for r in ranked:
                d = ratings[r["model"]] - old_ratings[r["model"]]
                print(f"    {r['display_name']:<28s} "
                      f"score={r['score']:+7.2f}%  won={r['deals_won']}  "
                      f"Elo {old_ratings[r['model']]:.0f}→{ratings[r['model']]:.0f} ({d:+.1f})")

            # Strip per_applicant_payoffs from logged results (too large for JSON)
            logged_results = [
                {k: v for k, v in r.items() if k != "per_applicant_payoffs"}
                for r in results
            ]
            match_log.append({
                "match": match_num,
                "models": [m[0] for m in triplet],
                "results": logged_results,
                "elapsed": round(elapsed, 1),
                "cost": match_cost,
            })

        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            match_log.append({
                "match": match_num,
                "models": [m[0] for m in triplet],
                "error": str(e),
            })

        # Incremental save
        output = _build_output(ratings, match_log, models, mix, total_cost)
        _save_results(output, output_file)

        # Print standings every 10 matches
        if match_num % 10 == 0:
            print_standings(ratings, models, match_log)

    return _build_output(ratings, match_log, models, mix, total_cost)


def _build_output(ratings, match_log, models, mix, total_cost):
    return {
        "timestamp": datetime.now().isoformat(),
        "mix": mix,
        "n_models": len(models),
        "n_matches": len(match_log),
        "total_cost": round(total_cost, 4),
        "ratings": {k: round(v, 1) for k, v in ratings.items()},
        "display_names": {m[0]: m[1] for m in models},
        "match_log": match_log,
    }


def _save_results(output, filename="elo_results.json"):
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_standings(ratings, models, match_log):
    """Print current Elo standings table."""
    display_map = {m[0]: m[1] for m in models}

    match_counts: dict[str, int] = {m[0]: 0 for m in models}
    win_counts: dict[str, float] = {m[0]: 0.0 for m in models}
    total_scores: dict[str, float] = {m[0]: 0.0 for m in models}

    for entry in match_log:
        results = entry.get("results")
        if not results:
            continue
        for r in results:
            match_counts[r["model"]] += 1
            total_scores[r["model"]] += r["score"]

        best_score = max(r["score"] for r in results)
        winners = [r for r in results if r["score"] == best_score]
        for w in winners:
            win_counts[w["model"]] += 1.0 / len(winners)

    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*80}")
    print(f"  ELO STANDINGS — {len(match_log)} matches played")
    print(f"{'='*80}")
    print(f"  {'#':>3s}  {'Model':<28s} {'Elo':>6s}  {'Matches':>7s}  "
          f"{'Wins':>5s}  {'Win%':>5s}  {'AvgScore':>9s}")
    print(f"  {'─'*72}")

    for rank, (model_id, elo) in enumerate(ranked, 1):
        name = display_map.get(model_id, model_id.split("/")[-1])
        matches = match_counts.get(model_id, 0)
        wins = win_counts.get(model_id, 0)
        win_pct = (wins / matches * 100) if matches > 0 else 0
        avg_score = (total_scores.get(model_id, 0) / matches) if matches > 0 else 0
        print(f"  {rank:>3d}  {name:<28s} {elo:>6.0f}  {matches:>7d}  "
              f"{wins:>5.1f}  {win_pct:>4.1f}%  {avg_score:>+8.2f}%")

    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Loanville Elo tournament benchmark"
    )
    parser.add_argument("--mix", type=str, default="analyst",
                        help="Borrower mix preset (default: analyst)")
    parser.add_argument("--matches", type=int, default=50,
                        help="Number of matches to run (default: 50)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from previous results JSON")
    parser.add_argument("--output", type=str, default="elo_results.json",
                        help="Output filename (default: elo_results.json)")
    parser.add_argument("--k", type=float, default=DEFAULT_K,
                        help=f"Elo K-factor (default: {DEFAULT_K})")
    parser.add_argument("--standings", type=str, default=None,
                        help="Just print standings from existing results")
    parser.add_argument("--full", action="store_true",
                        help="Use full 35-model set instead of small models")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific model IDs to include (overrides --full)")
    args = parser.parse_args()

    if args.standings:
        with open(args.standings) as f:
            data = json.load(f)
        models = list(data["display_names"].items())
        print_standings(data["ratings"], models, data["match_log"])
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    if args.mix not in MIX_PRESETS:
        print(f"ERROR: Unknown mix '{args.mix}'. Choose from: {list(MIX_PRESETS)}")
        sys.exit(1)

    all_models = BENCHMARK_MODELS if args.full else SMALL_MODELS
    if args.models:
        model_lookup = {m[0]: m for m in all_models + BENCHMARK_MODELS}
        models = []
        for mid in args.models:
            if mid in model_lookup:
                models.append(model_lookup[mid])
            else:
                print(f"ERROR: Unknown model '{mid}'")
                print(f"Available: {[m[0] for m in all_models]}")
                sys.exit(1)
    else:
        models = all_models

    resume_data = None
    if args.resume:
        with open(args.resume) as f:
            resume_data = json.load(f)
        print(f"Loaded {len(resume_data.get('match_log', []))} previous matches")

    output = run_tournament(
        models, args.matches, args.mix, api_key,
        k=args.k, output_file=args.output,
        resume_data=resume_data,
    )

    _save_results(output, args.output)
    print_standings(output["ratings"], models, output["match_log"])

    print(f"\nTournament complete. {output['n_matches']} matches played.")
    print(f"Total API cost: ${output['total_cost']:.2f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
