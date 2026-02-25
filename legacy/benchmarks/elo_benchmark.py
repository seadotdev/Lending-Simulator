#!/usr/bin/env python3
"""
Elo-rated tournament benchmark for Loanville lending models.

Instead of testing each model against fixed competitors, models compete
head-to-head in randomly matched triplets.  All three lenders get identical
parameters (capital, limits, persona) so the only variable is the model.

THREE SEPARATE ELO RATINGS are tracked per model:

  1. DealShare Elo  — "who wins deals?"  Measures market participation and
     bid aggressiveness.  Winning a deal = 1.0, regardless of whether the
     loan is profitable.  This is the original Elo formulation.

  2. Profit Elo     — "who earns more per-borrower?"  Compares realized
     utility (net profit if deal won, risk-free benchmark return if declined
     or outbid).  Aligns Elo rankings with RAROC economics.

  3. Credit Elo     — "who makes correct approve/reject decisions?"  Uses
     borrower ground truth: correctly declining a bad borrower beats
     approving it; correctly approving a good borrower beats declining it.
     Measures pure underwriting judgment.

All three use per-applicant pairwise signals for fast convergence.

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
import math
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
from loanville.scoring import (
    RISK_FREE_RATE, SIM_HORIZON_MONTHS,
    calculate_perfect_score, compute_confusion_matrix,
    compute_heuristic_baseline, compute_loan_payoff,
    compute_penalty_decomposition, bootstrap_raroc_interval, score_lenders,
)

# Import model lists from benchmark_models
from benchmark_models import SMALL_MODELS, BENCHMARK_MODELS

DEFAULT_K = 32
INITIAL_ELO = 1500

# Epsilon for utility comparison ties (Profit Elo)
UTILITY_EPSILON = 500.0  # $500 — within this range counts as a tie


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
) -> dict[str, dict]:
    """Compute competitive per-borrower data for each model.

    Implements a per-borrower allocation rule (winner-takes-deal):
      1. Each model submits (approve/decline, APR).
      2. If nobody approves -> all get benchmark utility.
      3. If one or more approve -> borrower takes the lowest APR offer.
      4. Only the winner gets realized payoff; losers get benchmark utility.
      5. Decline = benchmark return on the capital that would have been deployed.

    Returns:
      model_id -> {
          borrower_id -> payoff (realized net profit or 0),
          "_decision_states" -> {borrower_id -> "declined"/"won"/"lost"},
          "_ground_truth"    -> {borrower_id -> "good"/"bad"/"fraud"},
          "_utility"         -> {borrower_id -> utility for Profit Elo},
          "_rates_offered"   -> {borrower_id -> rate or None},
      }
    """
    # Benchmark return per dollar deployed (risk-free over standard term)
    benchmark_rate_per_dollar = RISK_FREE_RATE * (SIM_HORIZON_MONTHS / 12.0)

    # Collect all decisions indexed by borrower
    model_decisions: dict[str, dict[str, object]] = {}
    for lender, (model_id, _) in zip(lenders, models):
        decisions = engine.all_decisions.get(lender.id, [])
        dec_map = {}
        for d in decisions:
            dec_map[d.borrower_id] = d
        model_decisions[model_id] = dec_map

    payoffs: dict[str, dict] = {mid: {} for mid, _ in models}
    decision_states: dict[str, dict[str, str]] = {mid: {} for mid, _ in models}
    ground_truth: dict[str, dict[str, str]] = {mid: {} for mid, _ in models}
    utility: dict[str, dict[str, float]] = {mid: {} for mid, _ in models}
    rates_offered: dict[str, dict] = {mid: {} for mid, _ in models}

    for b in borrowers:
        bid = b.id
        notional = b.dossier.loan_request_amount
        benchmark_return = notional * benchmark_rate_per_dollar

        # Collect approvals with their rates
        approvals = []  # (model_id, decision, rate)
        for model_id, _ in models:
            d = model_decisions[model_id].get(bid)
            if d and d.decision == "APPROVE" and d.term_sheet:
                approvals.append((model_id, d, d.term_sheet.interest_rate))
                rates_offered[model_id][bid] = d.term_sheet.interest_rate
            else:
                rates_offered[model_id][bid] = None

        if not approvals:
            # Nobody approved — all declined
            for model_id, _ in models:
                payoffs[model_id][bid] = 0.0
                decision_states[model_id][bid] = "declined"
                ground_truth[model_id][bid] = b.true_outcome
                utility[model_id][bid] = benchmark_return
            continue

        # Winner = lowest APR (ties broken by higher loan amount)
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
            ground_truth[model_id][bid] = b.true_outcome

            if model_id == winner_model_id:
                payoffs[model_id][bid] = result["net_profit"]
                decision_states[model_id][bid] = "won"
                utility[model_id][bid] = result["net_profit"]
            elif d and d.decision == "APPROVE" and d.term_sheet:
                payoffs[model_id][bid] = 0.0
                decision_states[model_id][bid] = "lost"
                lost_notional = d.term_sheet.loan_amount
                utility[model_id][bid] = lost_notional * benchmark_rate_per_dollar
            else:
                payoffs[model_id][bid] = 0.0
                decision_states[model_id][bid] = "declined"
                utility[model_id][bid] = benchmark_return

    # Attach metadata for Elo logic
    for model_id, _ in models:
        payoffs[model_id]["_decision_states"] = decision_states[model_id]
        payoffs[model_id]["_ground_truth"] = ground_truth[model_id]
        payoffs[model_id]["_utility"] = utility[model_id]
        payoffs[model_id]["_rates_offered"] = rates_offered[model_id]

    return payoffs


def run_match(
    models: list[tuple[str, str]],
    mix: str,
    api_key: str,
    sample_borrowers: int | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """Run a single 3-way match.  Returns per-model results sorted by score.

    If sample_borrowers is set, randomly samples that many borrowers from the
    pool each match (stratified: maintains good/bad/fraud ratio).
    """
    borrowers = get_borrowers(mix)

    # Optional borrower sampling per match
    if sample_borrowers and sample_borrowers < len(borrowers):
        if rng is None:
            rng = random.Random()
        # Stratified sampling: maintain category ratios
        good = [b for b in borrowers if b.true_outcome == "good"]
        bad = [b for b in borrowers if b.true_outcome == "bad"]
        fraud = [b for b in borrowers if b.true_outcome == "fraud"]
        total = len(borrowers)
        n_good = max(1, round(len(good) / total * sample_borrowers))
        n_bad = max(0, round(len(bad) / total * sample_borrowers))
        n_fraud = max(0, sample_borrowers - n_good - n_bad)
        # Clamp to available
        n_good = min(n_good, len(good))
        n_bad = min(n_bad, len(bad))
        n_fraud = min(n_fraud, len(fraud))
        sampled = (
            rng.sample(good, n_good) +
            rng.sample(bad, n_bad) +
            (rng.sample(fraud, n_fraud) if fraud else [])
        )
        borrowers = sampled

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

    # Compute confusion matrices per model
    confusion_matrices = {}
    for lender, (model_id, _) in zip(lenders, models):
        decisions = engine.all_decisions.get(lender.id, [])
        confusion_matrices[model_id] = compute_confusion_matrix(decisions, borrowers)

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
            "confusion_matrix": confusion_matrices.get(model_id, {}),
            "n_borrowers": len(borrowers),
        })

    return results


# ---------------------------------------------------------------------------
# Elo math — three rating systems
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _get_borrower_ids(match_results: list[dict]) -> list[str]:
    """Extract borrower IDs in deterministic order (excluding metadata keys)."""
    ids: set[str] = set()
    for r in match_results:
        for key in r.get("per_applicant_payoffs", {}):
            if not key.startswith("_"):
                ids.add(key)
    return sorted(ids)


def _apply_elo_update(
    ratings: dict[str, float],
    mi: str, mj: str,
    actual_i: float, actual_j: float,
    pair_k: float,
) -> None:
    """Apply a single pairwise Elo update in-place."""
    exp_i = expected_score(ratings[mi], ratings[mj])
    exp_j = 1.0 - exp_i
    ratings[mi] += pair_k * (actual_i - exp_i)
    ratings[mj] += pair_k * (actual_j - exp_j)


def update_dealshare_elo(
    ratings: dict[str, float],
    match_results: list[dict],
    k: float = DEFAULT_K,
) -> dict[str, float]:
    """DealShare Elo: rewards winning deals (market participation).

    Per borrower, for models A vs B:
      - Both declined          -> tie (0.5 / 0.5)
      - One won, other anything -> winner wins (1.0 / 0.0)
      - Both lost (third won)  -> tie (0.5 / 0.5)
      - One lost, other declined -> tie (0.5 / 0.5)

    This measures who captures market share, regardless of profitability.
    """
    all_bids = _get_borrower_ids(match_results)
    if not all_bids:
        return dict(ratings)

    new_ratings = dict(ratings)
    n = len(match_results)
    pair_k = k / ((n - 1) * len(all_bids))

    for bid in all_bids:
        for i in range(n):
            for j in range(i + 1, n):
                mi = match_results[i]["model"]
                mj = match_results[j]["model"]

                states_i = match_results[i]["per_applicant_payoffs"].get("_decision_states", {})
                states_j = match_results[j]["per_applicant_payoffs"].get("_decision_states", {})
                si = states_i.get(bid, "declined")
                sj = states_j.get(bid, "declined")

                if si == "won" and sj != "won":
                    actual_i, actual_j = 1.0, 0.0
                elif sj == "won" and si != "won":
                    actual_i, actual_j = 0.0, 1.0
                else:
                    # Both declined, both lost, or both won (impossible) -> tie
                    actual_i, actual_j = 0.5, 0.5

                _apply_elo_update(new_ratings, mi, mj, actual_i, actual_j, pair_k)

    return new_ratings


def update_profit_elo(
    ratings: dict[str, float],
    match_results: list[dict],
    k: float = DEFAULT_K,
    epsilon: float = UTILITY_EPSILON,
) -> dict[str, float]:
    """Profit Elo: rewards economic utility per borrower.

    Per borrower, for models A vs B:
      - Compare Utility(A) vs Utility(B)
      - Utility if won deal = realized net profit (can be negative for bad loans)
      - Utility if declined or lost = risk-free benchmark return on notional capital
      - A wins if Utility(A) > Utility(B) + epsilon
      - Tie if |Utility(A) - Utility(B)| <= epsilon
      - B wins otherwise

    This aligns Elo with RAROC: correctly declining a bad loan earns benchmark
    return, which beats the negative profit from funding a defaulting borrower.
    """
    all_bids = _get_borrower_ids(match_results)
    if not all_bids:
        return dict(ratings)

    new_ratings = dict(ratings)
    n = len(match_results)
    pair_k = k / ((n - 1) * len(all_bids))

    for bid in all_bids:
        for i in range(n):
            for j in range(i + 1, n):
                mi = match_results[i]["model"]
                mj = match_results[j]["model"]

                util_i = match_results[i]["per_applicant_payoffs"].get("_utility", {}).get(bid, 0.0)
                util_j = match_results[j]["per_applicant_payoffs"].get("_utility", {}).get(bid, 0.0)

                diff = util_i - util_j
                if diff > epsilon:
                    actual_i, actual_j = 1.0, 0.0
                elif diff < -epsilon:
                    actual_i, actual_j = 0.0, 1.0
                else:
                    actual_i, actual_j = 0.5, 0.5

                _apply_elo_update(new_ratings, mi, mj, actual_i, actual_j, pair_k)

    return new_ratings


def update_credit_elo(
    ratings: dict[str, float],
    match_results: list[dict],
    k: float = DEFAULT_K,
) -> dict[str, float]:
    """Credit Elo: rewards correct approve/reject decisions vs ground truth.

    Per borrower, for models A vs B:
      - Correct decision:
          * Borrower is "good" and model approves (with non-usurious rate) -> correct
          * Borrower is "bad" or "fraud" and model declines -> correct
      - Incorrect decision:
          * Borrower is "good" and model declines -> incorrect
          * Borrower is "bad"/"fraud" and model approves -> incorrect

      - Model with correct decision wins vs model with incorrect decision
      - Both correct or both incorrect -> tie

    This measures pure underwriting judgment, independent of pricing or
    competitive dynamics.
    """
    all_bids = _get_borrower_ids(match_results)
    if not all_bids:
        return dict(ratings)

    new_ratings = dict(ratings)
    n = len(match_results)
    pair_k = k / ((n - 1) * len(all_bids))

    for bid in all_bids:
        for i in range(n):
            for j in range(i + 1, n):
                mi = match_results[i]["model"]
                mj = match_results[j]["model"]

                gt_i = match_results[i]["per_applicant_payoffs"].get("_ground_truth", {}).get(bid, "good")
                states_i = match_results[i]["per_applicant_payoffs"].get("_decision_states", {})
                states_j = match_results[j]["per_applicant_payoffs"].get("_decision_states", {})
                si = states_i.get(bid, "declined")
                sj = states_j.get(bid, "declined")

                # Ground truth is same for both models on same borrower
                gt = gt_i

                # Determine correctness
                def is_correct(state: str, ground_truth: str) -> bool:
                    approved = state in ("won", "lost")  # both mean model said APPROVE
                    if ground_truth == "good":
                        return approved
                    else:  # "bad" or "fraud"
                        return not approved

                correct_i = is_correct(si, gt)
                correct_j = is_correct(sj, gt)

                if correct_i and not correct_j:
                    actual_i, actual_j = 1.0, 0.0
                elif correct_j and not correct_i:
                    actual_i, actual_j = 0.0, 1.0
                else:
                    actual_i, actual_j = 0.5, 0.5

                _apply_elo_update(new_ratings, mi, mj, actual_i, actual_j, pair_k)

    return new_ratings


# Legacy compatibility
def update_elo_batch(
    ratings: dict[str, float],
    match_results: list[dict],
    k: float = DEFAULT_K,
) -> dict[str, float]:
    """Update Elo ratings using aggregate score (legacy batch mode)."""
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
            _apply_elo_update(new_ratings, mi, mj, actual_i, actual_j, pair_k)
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
    sample_borrowers: int | None = None,
    leaderboard: bool = False,
) -> dict:
    """Run the full Elo tournament with three rating systems."""
    # Compute baselines once (they only depend on the mix + standard lender config)
    baseline_lender = make_lender(1, "baseline", "baseline")
    all_borrowers = get_borrowers(mix)
    oracle_score = calculate_perfect_score(baseline_lender, all_borrowers)
    heuristic_score = compute_heuristic_baseline(baseline_lender, all_borrowers)

    init = {m[0]: float(INITIAL_ELO) for m in models}
    dealshare_ratings = dict(init)
    profit_ratings = dict(init)
    credit_ratings = dict(init)
    match_log: list[dict] = []
    total_cost = 0.0
    completed = 0

    # Accumulate rate stats across matches (per_applicant_payoffs are stripped from log)
    # model_id -> {"good_rates": [floats], "bad_rates": [floats]}
    rate_stats: dict[str, dict[str, list[float]]] = {
        m[0]: {"good_rates": [], "bad_rates": []} for m in models
    }

    if resume_data:
        # Load all three rating types (fall back to legacy "ratings" key)
        legacy = resume_data.get("ratings", {})
        dealshare_ratings = {k_: float(v) for k_, v in resume_data.get("dealshare_ratings", legacy).items()}
        profit_ratings = {k_: float(v) for k_, v in resume_data.get("profit_ratings", legacy).items()}
        credit_ratings = {k_: float(v) for k_, v in resume_data.get("credit_ratings", legacy).items()}
        match_log = list(resume_data.get("match_log", []))
        total_cost = resume_data.get("total_cost", 0.0)
        completed = len(match_log)
        # Ensure new models get initial ratings
        for m_id, _ in models:
            dealshare_ratings.setdefault(m_id, float(INITIAL_ELO))
            profit_ratings.setdefault(m_id, float(INITIAL_ELO))
            credit_ratings.setdefault(m_id, float(INITIAL_ELO))

    remaining = n_matches - completed
    if remaining <= 0:
        print(f"Already completed {completed} matches.  Nothing to do.")
        return _build_output(
            dealshare_ratings, profit_ratings, credit_ratings,
            match_log, models, mix, total_cost,
            oracle_score=oracle_score, heuristic_score=heuristic_score,
            rate_stats=rate_stats,
        )

    matchups = generate_matchups(models, remaining, seed=42 + completed)
    match_rng = random.Random(1337 + completed)

    print(f"\n{'='*70}")
    print(f"  LOANVILLE ELO TOURNAMENT (3-Rating System)")
    print(f"  Models: {len(models)} | Mix: {mix} | Matches: {n_matches} | K={k}")
    if sample_borrowers:
        print(f"  Borrower sampling: {sample_borrowers} per match")
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
            results = run_match(
                triplet, mix, api_key,
                sample_borrowers=sample_borrowers,
                rng=match_rng,
            )
            elapsed = time.time() - t0

            # Update all three rating systems
            old_ds = dict(dealshare_ratings)
            old_pr = dict(profit_ratings)
            old_cr = dict(credit_ratings)

            dealshare_ratings = update_dealshare_elo(dealshare_ratings, results, k=k)
            profit_ratings = update_profit_elo(profit_ratings, results, k=k)
            credit_ratings = update_credit_elo(credit_ratings, results, k=k)

            match_cost = sum(r["cost"] for r in results)
            total_cost += match_cost

            ranked = sorted(results, key=lambda r: r["score"], reverse=True)

            print(f"[{elapsed:.0f}s ${match_cost:.3f}]")
            for r in ranked:
                mid = r["model"]
                d_ds = dealshare_ratings[mid] - old_ds[mid]
                d_pr = profit_ratings[mid] - old_pr[mid]
                d_cr = credit_ratings[mid] - old_cr[mid]
                print(f"    {r['display_name']:<28s} "
                      f"RAROC={r['score']:+7.2f}%  won={r['deals_won']}  "
                      f"DS:{dealshare_ratings[mid]:.0f}({d_ds:+.1f}) "
                      f"PR:{profit_ratings[mid]:.0f}({d_pr:+.1f}) "
                      f"CR:{credit_ratings[mid]:.0f}({d_cr:+.1f})")

            # Accumulate rate stats before stripping per_applicant_payoffs
            for r in results:
                mid = r["model"]
                pap = r.get("per_applicant_payoffs", {})
                rates = pap.get("_rates_offered", {})
                gt = pap.get("_ground_truth", {})
                for bid, rate in rates.items():
                    if rate is None:
                        continue  # model declined — no rate offered
                    outcome = gt.get(bid, "good")
                    if mid not in rate_stats:
                        rate_stats[mid] = {"good_rates": [], "bad_rates": []}
                    if outcome == "good":
                        rate_stats[mid]["good_rates"].append(rate)
                    else:  # bad or fraud
                        rate_stats[mid]["bad_rates"].append(rate)

            # Emit leaderboard match record (before stripping per_applicant_payoffs)
            if leaderboard:
                from loanville.leaderboard import emit_match_record_from_elo, emit_and_update
                n_b = results[0].get("n_borrowers", len(get_borrowers(mix))) if results else 0
                lb_record = emit_match_record_from_elo(
                    triplet=triplet,
                    match_results=results,
                    mix=mix,
                    n_borrowers=n_b,
                )
                match_path, lb_path = emit_and_update(lb_record)
                print(f"    → leaderboard: {match_path.name}")

            # Strip per_applicant_payoffs from logged results (too large for JSON)
            logged_results = []
            for r in results:
                logged = {k_: v for k_, v in r.items()
                          if k_ != "per_applicant_payoffs"}
                logged_results.append(logged)

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
        output = _build_output(
            dealshare_ratings, profit_ratings, credit_ratings,
            match_log, models, mix, total_cost,
            oracle_score=oracle_score, heuristic_score=heuristic_score,
            rate_stats=rate_stats,
        )
        _save_results(output, output_file)

        # Print standings every 10 matches
        if match_num % 10 == 0:
            print_standings(
                dealshare_ratings, profit_ratings, credit_ratings,
                models, match_log,
                baselines=(oracle_score, heuristic_score),
                rate_stats=rate_stats,
            )

    return _build_output(
        dealshare_ratings, profit_ratings, credit_ratings,
        match_log, models, mix, total_cost,
        oracle_score=oracle_score, heuristic_score=heuristic_score,
        rate_stats=rate_stats,
    )


def _build_output(dealshare_ratings, profit_ratings, credit_ratings,
                  match_log, models, mix, total_cost,
                  oracle_score=None, heuristic_score=None, rate_stats=None):
    out = {
        "timestamp": datetime.now().isoformat(),
        "mix": mix,
        "n_models": len(models),
        "n_matches": len(match_log),
        "total_cost": round(total_cost, 4),
        # Three rating systems
        "dealshare_ratings": {k: round(v, 1) for k, v in dealshare_ratings.items()},
        "profit_ratings": {k: round(v, 1) for k, v in profit_ratings.items()},
        "credit_ratings": {k: round(v, 1) for k, v in credit_ratings.items()},
        # Legacy compatibility: "ratings" points to profit_ratings (the recommended default)
        "ratings": {k: round(v, 1) for k, v in profit_ratings.items()},
        "display_names": {m[0]: m[1] for m in models},
        "match_log": match_log,
    }
    if oracle_score is not None:
        out["baselines"] = {
            "oracle_raroc": round(oracle_score, 2),
            "heuristic_raroc": round(heuristic_score, 2),
        }
    if rate_stats:
        # Summarize rate stats for JSON output
        out["rate_analysis"] = {}
        for mid, stats in rate_stats.items():
            good = stats["good_rates"]
            bad = stats["bad_rates"]
            out["rate_analysis"][mid] = {
                "avg_rate_good": round(sum(good) / len(good), 2) if good else None,
                "avg_rate_bad": round(sum(bad) / len(bad), 2) if bad else None,
                "n_good_offers": len(good),
                "n_bad_offers": len(bad),
            }
    return out


def _save_results(output, filename="elo_results.json"):
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_standings(dealshare_ratings, profit_ratings, credit_ratings,
                    models, match_log, baselines=None, rate_stats=None):
    """Print current standings with all three Elo systems, baselines, and rate analysis."""
    display_map = {m[0]: m[1] for m in models}

    match_counts: dict[str, int] = {m[0]: 0 for m in models}
    win_counts: dict[str, float] = {m[0]: 0.0 for m in models}
    total_scores: dict[str, float] = {m[0]: 0.0 for m in models}
    # Aggregate confusion matrices
    agg_cm: dict[str, dict[str, dict[str, int]]] = {}
    for m_id, _ in models:
        agg_cm[m_id] = {
            "good": {"approved": 0, "rejected": 0},
            "bad": {"approved": 0, "rejected": 0},
            "fraud": {"approved": 0, "rejected": 0},
        }

    for entry in match_log:
        results = entry.get("results")
        if not results:
            continue
        for r in results:
            mid = r["model"]
            match_counts[mid] += 1
            total_scores[mid] += r["score"]
            # Aggregate confusion matrices
            cm = r.get("confusion_matrix", {})
            for category in ("good", "bad", "fraud"):
                cat_data = cm.get(category, {})
                agg_cm[mid][category]["approved"] += cat_data.get("approved", 0)
                agg_cm[mid][category]["rejected"] += cat_data.get("rejected", 0)

        best_score = max(r["score"] for r in results)
        winners = [r for r in results if r["score"] == best_score]
        for w in winners:
            win_counts[w["model"]] += 1.0 / len(winners)

    # Sort by Profit Elo (the recommended ranking)
    ranked = sorted(profit_ratings.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*110}")
    print(f"  ELO STANDINGS — {len(match_log)} matches played")
    print(f"  Sorted by Profit Elo (recommended ranking)")
    print(f"{'='*110}")
    print(f"  {'#':>3s}  {'Model':<24s} {'Profit':>7s} {'Credit':>7s} {'DealSh':>7s}  "
          f"{'Matches':>7s}  {'Win%':>5s}  {'AvgRAROC':>9s}  "
          f"{'Good✓':>6s} {'Bad✓':>6s}")
    print(f"  {'─'*100}")

    for rank, (model_id, profit_elo) in enumerate(ranked, 1):
        name = display_map.get(model_id, model_id.split("/")[-1])
        ds_elo = dealshare_ratings.get(model_id, INITIAL_ELO)
        cr_elo = credit_ratings.get(model_id, INITIAL_ELO)
        matches = match_counts.get(model_id, 0)
        wins = win_counts.get(model_id, 0)
        win_pct = (wins / matches * 100) if matches > 0 else 0
        avg_score = (total_scores.get(model_id, 0) / matches) if matches > 0 else 0

        # Confusion matrix summary
        cm = agg_cm.get(model_id, {})
        good_total = cm["good"]["approved"] + cm["good"]["rejected"]
        good_correct = cm["good"]["approved"]  # approving good = correct
        good_pct = f"{good_correct}/{good_total}" if good_total > 0 else "—"

        bad_total = cm["bad"]["approved"] + cm["bad"]["rejected"]
        bad_correct = cm["bad"]["rejected"]  # rejecting bad = correct
        bad_pct = f"{bad_correct}/{bad_total}" if bad_total > 0 else "—"

        print(f"  {rank:>3d}  {name:<24s} {profit_elo:>7.0f} {cr_elo:>7.0f} {ds_elo:>7.0f}  "
              f"{matches:>7d}  {win_pct:>4.1f}%  {avg_score:>+8.2f}%  "
              f"{good_pct:>6s} {bad_pct:>6s}")

    # Baselines section
    if baselines:
        oracle_score, heuristic_score = baselines
        print(f"  {'─'*100}")
        print(f"  {'':>3s}  {'Oracle (perfect info)':24s} {'':>7s} {'':>7s} {'':>7s}  "
              f"{'':>7s}  {'':>5s}  {oracle_score:>+8.2f}%  {'':>6s} {'':>6s}")
        print(f"  {'':>3s}  {'Heuristic (DSCR rules)':24s} {'':>7s} {'':>7s} {'':>7s}  "
              f"{'':>7s}  {'':>5s}  {heuristic_score:>+8.2f}%  {'':>6s} {'':>6s}")

    print(f"{'='*110}")
    print(f"  Legend: Profit=Profit Elo, Credit=Credit Elo, DealSh=DealShare Elo")
    print(f"  Good✓=good borrowers correctly approved, Bad✓=bad/fraud correctly rejected")

    # Rate analysis section
    if rate_stats and any(s["good_rates"] or s["bad_rates"] for s in rate_stats.values()):
        print(f"\n  PRICING ANALYSIS — avg rate offered by borrower quality")
        print(f"  {'─'*80}")
        print(f"  {'Model':<24s}  {'Rate→Good':>10s} {'(n)':>5s}  "
              f"{'Rate→Bad':>10s} {'(n)':>5s}  {'Spread':>8s}  {'Signal?':>8s}")
        print(f"  {'─'*80}")

        for model_id, _ in sorted(ranked, key=lambda x: x[1], reverse=True):
            name = display_map.get(model_id, model_id.split("/")[-1])
            stats = rate_stats.get(model_id, {"good_rates": [], "bad_rates": []})
            good = stats["good_rates"]
            bad = stats["bad_rates"]
            avg_good = sum(good) / len(good) if good else 0
            avg_bad = sum(bad) / len(bad) if bad else 0
            n_good = len(good)
            n_bad = len(bad)

            if good and bad:
                spread = avg_bad - avg_good
                # Signal = model charges more for bad credits (positive spread = good)
                signal = "YES" if spread > 0.5 else ("weak" if spread > 0 else "NO")
                print(f"  {name:<24s}  {avg_good:>9.1f}% {n_good:>5d}  "
                      f"{avg_bad:>9.1f}% {n_bad:>5d}  {spread:>+7.1f}%  {signal:>8s}")
            elif good:
                print(f"  {name:<24s}  {avg_good:>9.1f}% {n_good:>5d}  "
                      f"{'—':>10s} {n_bad:>5d}  {'—':>8s}  {'decl all':>8s}")
            else:
                print(f"  {name:<24s}  {'—':>10s} {n_good:>5d}  "
                      f"{'—':>10s} {n_bad:>5d}  {'—':>8s}  {'no data':>8s}")

        print(f"  {'─'*80}")
        print(f"  Spread = (avg rate on bad) − (avg rate on good). Positive = risk-aware pricing.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Loanville Elo tournament benchmark (3-rating system)"
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
    parser.add_argument("--sample-borrowers", type=int, default=12,
                        help="Sample N borrowers per match from the pool (default: 12, 0=use all)")
    parser.add_argument("--no-sample", action="store_true",
                        help="Disable borrower sampling (use full pool every match)")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Emit match records to leaderboard after each match")
    args = parser.parse_args()

    if args.standings:
        with open(args.standings) as f:
            data = json.load(f)
        models = list(data["display_names"].items())
        ds = data.get("dealshare_ratings", data.get("ratings", {}))
        pr = data.get("profit_ratings", data.get("ratings", {}))
        cr = data.get("credit_ratings", data.get("ratings", {}))
        # Load baselines from saved results (if available), otherwise compute
        bl = data.get("baselines")
        baselines = None
        if bl:
            baselines = (bl["oracle_raroc"], bl["heuristic_raroc"])
        else:
            # Recompute from the mix
            mix_name = data.get("mix", "analyst")
            if mix_name in MIX_PRESETS:
                bl_lender = make_lender(1, "baseline", "baseline")
                bl_borrowers = get_borrowers(mix_name)
                baselines = (
                    calculate_perfect_score(bl_lender, bl_borrowers),
                    compute_heuristic_baseline(bl_lender, bl_borrowers),
                )
        # Load rate analysis from saved results (if available)
        ra = data.get("rate_analysis")
        loaded_rate_stats = None
        if ra:
            # Convert JSON summary back to the format print_standings expects
            loaded_rate_stats = {}
            for mid, info in ra.items():
                avg_g = info.get("avg_rate_good")
                avg_b = info.get("avg_rate_bad")
                n_g = info.get("n_good_offers", 0)
                n_b = info.get("n_bad_offers", 0)
                loaded_rate_stats[mid] = {
                    "good_rates": [avg_g] * n_g if avg_g is not None else [],
                    "bad_rates": [avg_b] * n_b if avg_b is not None else [],
                }
        print_standings(ds, pr, cr, models, data["match_log"],
                        baselines=baselines, rate_stats=loaded_rate_stats)
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

    # Resolve sampling: --no-sample disables, --sample-borrowers 0 disables,
    # otherwise default is 12
    sample_n = None if args.no_sample or args.sample_borrowers == 0 else args.sample_borrowers

    output = run_tournament(
        models, args.matches, args.mix, api_key,
        k=args.k, output_file=args.output,
        resume_data=resume_data,
        sample_borrowers=sample_n,
        leaderboard=args.leaderboard,
    )

    _save_results(output, args.output)
    bl = output.get("baselines")
    baselines = (bl["oracle_raroc"], bl["heuristic_raroc"]) if bl else None
    # Reconstruct rate_stats from output for final display
    ra = output.get("rate_analysis")
    final_rate_stats = None
    if ra:
        final_rate_stats = {}
        for mid, info in ra.items():
            avg_g = info.get("avg_rate_good")
            avg_b = info.get("avg_rate_bad")
            n_g = info.get("n_good_offers", 0)
            n_b = info.get("n_bad_offers", 0)
            final_rate_stats[mid] = {
                "good_rates": [avg_g] * n_g if avg_g is not None else [],
                "bad_rates": [avg_b] * n_b if avg_b is not None else [],
            }
    print_standings(
        output["dealshare_ratings"],
        output["profit_ratings"],
        output["credit_ratings"],
        models, output["match_log"],
        baselines=baselines, rate_stats=final_rate_stats,
    )

    print(f"\nTournament complete. {output['n_matches']} matches played.")
    print(f"Total API cost: ${output['total_cost']:.2f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
