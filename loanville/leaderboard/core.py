"""
Core leaderboard logic: validate, emit, compute, adapt.

Validates sim runs, stores match records as JSON in leaderboard/matches/,
computes Elo rankings from match history, and exports standings.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Elo math reused from elo_benchmark (moved to legacy/benchmarks/ after PR #20)
import sys
_legacy_benchmarks = str(Path(__file__).resolve().parent.parent.parent / "legacy" / "benchmarks")
if _legacy_benchmarks not in sys.path:
    sys.path.insert(0, _legacy_benchmarks)

from elo_benchmark import (
    DEFAULT_K,
    INITIAL_ELO,
    UTILITY_EPSILON,
    ELO_MATCH_CAP,
    COMPOSITE_WEIGHTS,
    update_dealshare_elo,
    update_profit_elo,
    update_credit_elo,
    compute_composite_elo,
)

LEADERBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "leaderboard"
MATCHES_DIR = LEADERBOARD_DIR / "matches"
LEADERBOARD_FILE = LEADERBOARD_DIR / "leaderboard.json"
CONFIG_FILE = LEADERBOARD_DIR / "config.json"


def load_config() -> dict:
    """Load leaderboard config (Elo constants, validation thresholds)."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {
        "k": DEFAULT_K,
        "initial_elo": INITIAL_ELO,
        "utility_epsilon": UTILITY_EPSILON,
        "min_borrowers": 6,
        "max_error_rate": 0.25,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_match(match_data: dict, config: dict | None = None) -> dict:
    """Validate a match record for data quality.

    Returns {"valid": bool, "errors": list[str]}
    """
    if config is None:
        config = load_config()

    errors = []
    min_borrowers = config.get("min_borrowers", 6)
    max_error_rate = config.get("max_error_rate", 0.25)

    # Check basic structure
    if "models" not in match_data or "results" not in match_data:
        return {"valid": False, "errors": ["Missing 'models' or 'results' fields"]}

    models = match_data.get("models", [])
    results = match_data.get("results", [])
    n_borrowers = match_data.get("n_borrowers", 0)

    # 0. Basic model/result consistency
    model_ids = [m.get("model_id", "") for m in models if isinstance(m, dict)]
    if not model_ids:
        errors.append("No models provided")
    dup_model_ids = sorted({mid for mid in model_ids if model_ids.count(mid) > 1})
    if dup_model_ids:
        errors.append(f"Duplicate model_id entries in models: {', '.join(dup_model_ids)}")

    result_model_ids = [r.get("model_id", "") for r in results if isinstance(r, dict)]
    if not result_model_ids:
        errors.append("No results provided")
    dup_result_ids = sorted({mid for mid in result_model_ids if result_model_ids.count(mid) > 1})
    if dup_result_ids:
        errors.append(f"Duplicate model_id entries in results: {', '.join(dup_result_ids)}")
    for mid in result_model_ids:
        if mid and mid not in model_ids:
            errors.append(f"Result references unknown model_id: {mid}")

    # 1. Min borrowers
    if n_borrowers < min_borrowers:
        errors.append(f"Only {n_borrowers} borrowers (minimum {min_borrowers})")

    # 2. Max error rate per model — count valid models
    valid_models: set[str] = set()
    for r in results:
        mid = r.get("model_id", "")
        total = r.get("deals_won", 0) + r.get("deals_rejected", 0) + r.get("deals_errored", 0)
        errored = r.get("deals_errored", 0)
        if total > 0 and errored / total > max_error_rate:
            errors.append(
                f"Model {mid}: error rate {errored}/{total} "
                f"({errored/total:.0%}) exceeds {max_error_rate:.0%}"
            )
        else:
            if mid:
                valid_models.add(mid)

    # 3. Labels present — check per_borrower ground truth
    for r in results:
        mid = r.get("model_id", "")
        per_b = r.get("per_borrower", {})
        for bid, bdata in per_b.items():
            if "ground_truth" not in bdata:
                errors.append(f"Model {mid}, borrower {bid}: missing ground_truth")
                break  # one error per model is enough

    # 4. At least 2 valid models
    if len(valid_models) < 2:
        errors.append(f"Only {len(valid_models)} valid unique models (need at least 2)")

    return {"valid": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Match record construction
# ---------------------------------------------------------------------------

def _generate_match_id(timestamp_utc: str) -> str:
    """Generate a unique match ID from timestamp + hash."""
    ts_safe = timestamp_utc.replace(":", "-")[:19]
    h = hashlib.sha256(
        f"{timestamp_utc}-{os.getpid()}-{id(timestamp_utc)}".encode()
    ).hexdigest()[:6]
    return f"{ts_safe}_{h}"


def build_match_record(
    models: list[dict],
    results: list[dict],
    mix: str,
    n_borrowers: int,
    timestamp_utc: str | None = None,
) -> dict:
    """Build a match record from sim results.

    Args:
        models: [{"model_id": str, "display_name": str}, ...]
        results: Per-model result dicts with keys:
            model_id, raroc_score, deals_won, deals_rejected, deals_errored,
            frauds_funded, defaults, deployed, net_pnl,
            confusion_matrix, per_borrower
        mix: Borrower mix name
        n_borrowers: Total borrowers in the match
        timestamp_utc: Optional ISO timestamp (defaults to now)
    """
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    match_id = _generate_match_id(timestamp_utc)

    record = {
        "match_id": match_id,
        "timestamp_utc": timestamp_utc,
        "mix": mix,
        "n_borrowers": n_borrowers,
        "models": models,
        "results": results,
        "validation": {"valid": True, "errors": []},
    }

    # Run validation
    validation = validate_match(record)
    record["validation"] = validation

    return record


# ---------------------------------------------------------------------------
# Adapter: match record → elo_benchmark.py format
# ---------------------------------------------------------------------------

def match_to_elo_results(match: dict) -> list[dict]:
    """Convert leaderboard match record to elo_benchmark.py format.

    The Elo update functions expect:
    [{"model": model_id, "score": raroc, "per_applicant_payoffs": {
        "_decision_states": {bid: "won"/"lost"/"declined"},
        "_ground_truth": {bid: "good"/"bad"/"fraud"},
        "_utility": {bid: float},
        "_rates_offered": {bid: float|None},
        bid: 0.0,  # placeholder payoff entries for _get_borrower_ids()
    }}]
    """
    elo_results = []
    for r in match.get("results", []):
        per_b = r.get("per_borrower", {})

        per_applicant_payoffs = {
            "_decision_states": {},
            "_ground_truth": {},
            "_utility": {},
            "_rates_offered": {},
        }

        for bid, bdata in per_b.items():
            per_applicant_payoffs[bid] = 0.0  # placeholder for _get_borrower_ids()
            per_applicant_payoffs["_decision_states"][bid] = bdata.get("decision_state", "declined")
            per_applicant_payoffs["_ground_truth"][bid] = bdata.get("ground_truth", "good")
            per_applicant_payoffs["_utility"][bid] = bdata.get("utility", 0.0)
            per_applicant_payoffs["_rates_offered"][bid] = bdata.get("rate_offered")

        elo_results.append({
            "model": r["model_id"],
            "score": r.get("raroc_score", 0.0),
            "per_applicant_payoffs": per_applicant_payoffs,
        })

    return elo_results


# ---------------------------------------------------------------------------
# Emit match record from sim data
# ---------------------------------------------------------------------------

def emit_match_record_from_sim(
    lenders,
    models_info: list[dict],
    engine,
    scores,
    borrowers,
    mix: str,
    per_applicant_payoffs: dict | None = None,
) -> dict:
    """Build and write a match record from a completed sim run.

    Args:
        lenders: List of LenderConfig
        models_info: [{"model_id": str, "display_name": str}, ...]
        engine: SimulationEngine (has all_decisions, booked_loans, etc.)
        scores: List of LenderScore from score_lenders()
        borrowers: List of Borrower
        mix: Borrower mix name
        per_applicant_payoffs: Optional pre-computed payoffs dict
            (model_id -> {bid -> payoff, "_decision_states" -> ..., etc.})
            If None, will be computed from engine data.
    """
    from ..scoring import compute_confusion_matrix

    # Build per_applicant_payoffs if not provided
    if per_applicant_payoffs is None:
        from elo_benchmark import _compute_per_applicant_payoffs
        models_tuples = [(m["model_id"], m["display_name"]) for m in models_info]
        per_applicant_payoffs = _compute_per_applicant_payoffs(
            borrowers, lenders, models_tuples, engine,
        )

    results = []
    for lender, minfo in zip(lenders, models_info):
        model_id = minfo["model_id"]
        sc = next((s for s in scores if s.lender_id == lender.id), None)
        decisions = engine.all_decisions.get(lender.id, [])

        # Confusion matrix
        cm = compute_confusion_matrix(decisions, borrowers)

        # Per-borrower data
        pap = per_applicant_payoffs.get(model_id, {})
        per_borrower = {}
        for b in borrowers:
            bid = b.id
            states = pap.get("_decision_states", {})
            gt = pap.get("_ground_truth", {})
            util = pap.get("_utility", {})
            rates = pap.get("_rates_offered", {})

            per_borrower[bid] = {
                "decision_state": states.get(bid, "declined"),
                "ground_truth": gt.get(bid, b.true_outcome),
                "utility": util.get(bid, 0.0),
                "rate_offered": rates.get(bid),
            }

        # Extract token/cost from engine runs for this lender
        tokens_in = 0
        tokens_out = 0
        cost_usd = 0.0
        for run in engine.runs:
            lid = (run.policy.params or {}).get("_lender_id", "")
            if not lid:
                pid = run.policy.policy_id or ""
                parts = pid.split("_", 2)
                lid = parts[1] if len(parts) >= 2 and parts[0] == "p" else pid
            if lid == lender.id and run.trace and run.trace.cost:
                tokens_in += run.trace.cost.tokens_in
                tokens_out += run.trace.cost.tokens_out
                cost_usd += run.trace.cost.estimated_cost_usd

        result = {
            "model_id": model_id,
            "raroc_score": sc.raroc_score if sc else 0.0,
            "deals_won": sc.deals_won if sc else 0,
            "deals_rejected": sc.deals_rejected if sc else 0,
            "deals_errored": getattr(sc, 'deals_errored', 0) if sc else 0,
            "frauds_funded": sc.frauds_funded if sc else 0,
            "defaults": sc.defaults_count if sc else 0,
            "deployed": sc.total_deployed if sc else 0,
            "net_pnl": sc.net_return if sc else 0,
            "confusion_matrix": cm,
            "per_borrower": per_borrower,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 4),
        }
        results.append(result)

    record = build_match_record(
        models=models_info,
        results=results,
        mix=mix,
        n_borrowers=len(borrowers),
    )

    return record


def emit_match_record_from_season(
    season_engine,
    season_scores,
    lenders,
    mix: str,
) -> dict:
    """Build a match record from a completed season run.

    Season mode accumulates per-borrower decisions across weeks.
    We reconstruct the per_borrower data from the season engine's
    accumulated all_decisions, all_borrowers, and all_deal_results.

    Note: raroc_score is set to SeasonScore.final_score (0-100 composite)
    rather than actual RAROC %. The leaderboard displays avg_net_pnl instead
    of avg_raroc to avoid scale confusion across match types.
    """
    from ..scoring import compute_confusion_matrix, compute_loan_payoff

    models_info = [
        {"model_id": l.model, "display_name": l.name}
        for l in lenders
    ]

    eco = season_engine.config.economics
    benchmark_rate_per_dollar = eco.risk_free_rate * (eco.sim_horizon_months / 12.0)

    # Build deal winner lookup: borrower_id -> winning lender_id
    deal_winners = {}
    for bid, deal in season_engine.all_deal_results.items():
        if deal.get("outcome") == "booked":
            deal_winners[bid] = deal.get("winner")

    results = []
    for lender in lenders:
        state = season_engine.lender_states[lender.id]
        score = next((s for s in season_scores if s.lender_id == lender.id), None)

        decisions = season_engine.all_decisions.get(lender.id, [])
        cm = compute_confusion_matrix(decisions, season_engine.all_borrowers)

        # Build per_borrower data
        per_borrower = {}
        decision_map = {d.borrower_id: d for d in decisions}

        for b in season_engine.all_borrowers:
            bid = b.id
            d = decision_map.get(bid)
            if d is None:
                continue

            # Determine decision state
            if d.decision != "APPROVE":
                decision_state = "declined"
            elif deal_winners.get(bid) == lender.id:
                decision_state = "won"
            else:
                decision_state = "lost"

            rate_offered = None
            if d.decision == "APPROVE" and d.term_sheet:
                rate_offered = d.term_sheet.interest_rate

            # Align utility semantics with elo_benchmark._compute_per_applicant_payoffs():
            # won = realized net profit, lost/declined = risk-free benchmark return.
            notional = b.dossier.loan_request_amount
            utility = notional * benchmark_rate_per_dollar
            if decision_state == "won" and d.term_sheet:
                payoff = compute_loan_payoff(
                    principal=d.term_sheet.loan_amount,
                    interest_rate=d.term_sheet.interest_rate,
                    term_months=d.term_sheet.term_months,
                    true_outcome=b.true_outcome,
                    months_before_default=b.months_before_default,
                    economics=eco,
                )
                utility = payoff["net_profit"]
            elif decision_state == "lost" and d.term_sheet:
                utility = d.term_sheet.loan_amount * benchmark_rate_per_dollar

            per_borrower[bid] = {
                "decision_state": decision_state,
                "ground_truth": b.true_outcome,
                "utility": utility,
                "rate_offered": rate_offered,
            }

        # Aggregate stats
        frauds_funded = sum(1 for lo in state.resolved_loans if lo.was_fraud)
        defaults = sum(1 for lo in state.resolved_loans if lo.defaulted)
        net_pnl = state.cumulative_interest + state.cumulative_fees - state.cumulative_losses

        result = {
            "model_id": lender.model,
            "raroc_score": score.final_score if score else 0.0,
            "deals_won": state.deals_won,
            "deals_rejected": state.deals_rejected,
            "deals_errored": 0,
            "frauds_funded": frauds_funded,
            "defaults": defaults,
            "deployed": state.deployed_capital,
            "net_pnl": net_pnl,
            "confusion_matrix": cm,
            "per_borrower": per_borrower,
            "tokens_in": state.cumulative_tokens_in,
            "tokens_out": state.cumulative_tokens_out,
            "cost_usd": round(state.cumulative_cost_usd, 4),
        }
        results.append(result)

    record = build_match_record(
        models=models_info,
        results=results,
        mix=f"season-{mix}",
        n_borrowers=len(season_engine.all_borrowers),
    )

    return record


def emit_match_record_from_elo(
    triplet: list[tuple[str, str]],
    match_results: list[dict],
    mix: str,
    n_borrowers: int,
) -> dict:
    """Build a match record from elo_benchmark.py tournament match results.

    Args:
        triplet: [(model_id, display_name), ...]
        match_results: Results from run_match() with per_applicant_payoffs
        mix: Borrower mix name
        n_borrowers: Total borrowers
    """
    models_info = [
        {"model_id": mid, "display_name": dname}
        for mid, dname in triplet
    ]

    results = []
    for r in match_results:
        model_id = r["model"]
        pap = r.get("per_applicant_payoffs", {})

        # Build per_borrower from per_applicant_payoffs
        per_borrower = {}
        states = pap.get("_decision_states", {})
        gt = pap.get("_ground_truth", {})
        util = pap.get("_utility", {})
        rates = pap.get("_rates_offered", {})

        for bid in states.keys():
            per_borrower[bid] = {
                "decision_state": states.get(bid, "declined"),
                "ground_truth": gt.get(bid, "good"),
                "utility": util.get(bid, 0.0),
                "rate_offered": rates.get(bid),
            }

        result = {
            "model_id": model_id,
            "raroc_score": r.get("raroc_score", r.get("score", 0.0)),
            "deals_won": r.get("deals_won", 0),
            "deals_rejected": max(
                0,
                r.get("n_borrowers", n_borrowers) - r.get("approvals", r.get("deals_won", 0)),
            ),
            "deals_errored": 0,
            "frauds_funded": r.get("frauds_funded", 0),
            "defaults": r.get("defaults", 0),
            "deployed": r.get("deployed", 0),
            "net_pnl": r.get("net_pnl", 0),
            "confusion_matrix": r.get("confusion_matrix", {}),
            "per_borrower": per_borrower,
            "tokens_in": r.get("tokens_in", 0),
            "tokens_out": r.get("tokens_out", 0),
            "cost_usd": r.get("cost_usd", 0.0),
        }
        results.append(result)

    return build_match_record(
        models=models_info,
        results=results,
        mix=mix,
        n_borrowers=n_borrowers,
    )


# ---------------------------------------------------------------------------
# Write / Read match records
# ---------------------------------------------------------------------------

def write_match_record(record: dict) -> Path:
    """Write a match record to leaderboard/matches/."""
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)

    match_id = record["match_id"]
    filename = f"{match_id}.json"
    filepath = MATCHES_DIR / filename

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return filepath


def load_all_matches() -> list[dict]:
    """Load all match records, sorted by timestamp."""
    matches = []
    if not MATCHES_DIR.exists():
        return matches

    for f in sorted(MATCHES_DIR.glob("*.json")):
        with open(f) as fh:
            try:
                matches.append(json.load(fh))
            except json.JSONDecodeError:
                continue

    # Sort by timestamp
    matches.sort(key=lambda m: m.get("timestamp_utc", ""))
    return matches


# ---------------------------------------------------------------------------
# Leaderboard computation
# ---------------------------------------------------------------------------

def compute_leaderboard(matches: list[dict] | None = None, config: dict | None = None) -> dict:
    """Compute full leaderboard by replaying all matches chronologically.

    Returns leaderboard state dict ready to write to leaderboard.json.
    """
    if matches is None:
        matches = load_all_matches()
    if config is None:
        config = load_config()

    k = config.get("k", DEFAULT_K)
    initial_elo = config.get("initial_elo", INITIAL_ELO)

    # Re-validate using current rules, not cached validation flags in match files.
    valid_matches = []
    for match in matches:
        validation = validate_match(match, config)
        if validation["valid"]:
            valid_matches.append(match)

    # Collect all model IDs seen — use model short name for display
    # (persona names like "Heritage Trust Bank" vary across match types)
    all_models = {}  # model_id -> display_name
    for match in valid_matches:
        for m in match.get("models", []):
            mid = m["model_id"]
            if mid not in all_models:
                all_models[mid] = mid.split("/")[-1]

    # Init ratings
    profit_ratings = {mid: float(initial_elo) for mid in all_models}
    credit_ratings = {mid: float(initial_elo) for mid in all_models}
    dealshare_ratings = {mid: float(initial_elo) for mid in all_models}

    # Track per-model aggregates
    match_counts = {mid: 0 for mid in all_models}
    total_net_pnl = {mid: 0.0 for mid in all_models}
    total_tokens_in = {mid: 0 for mid in all_models}
    total_tokens_out = {mid: 0 for mid in all_models}
    total_cost_usd = {mid: 0.0 for mid in all_models}
    total_decisions = {mid: 0 for mid in all_models}
    agg_confusion = {}
    for mid in all_models:
        agg_confusion[mid] = {
            "good": {"approved": 0, "rejected": 0},
            "bad": {"approved": 0, "rejected": 0},
            "fraud": {"approved": 0, "rejected": 0},
        }
    rate_stats = {mid: {"good_rates": [], "bad_rates": []} for mid in all_models}

    # Track Elo history for sparklines
    elo_history = {mid: [] for mid in all_models}
    match_ids = []

    # Replay each match
    for match in valid_matches:
        match_ids.append(match["match_id"])
        elo_results = match_to_elo_results(match)

        if len(elo_results) < 2:
            continue

        # Ensure all models in this match have ratings
        for r in elo_results:
            mid = r["model"]
            profit_ratings.setdefault(mid, float(initial_elo))
            credit_ratings.setdefault(mid, float(initial_elo))
            dealshare_ratings.setdefault(mid, float(initial_elo))
            match_counts.setdefault(mid, 0)
            total_net_pnl.setdefault(mid, 0.0)

        # Update all three Elo systems
        dealshare_ratings = update_dealshare_elo(dealshare_ratings, elo_results, k=k)
        profit_ratings = update_profit_elo(profit_ratings, elo_results, k=k)
        credit_ratings = update_credit_elo(credit_ratings, elo_results, k=k)

        # Aggregate stats
        for r in match.get("results", []):
            mid = r["model_id"]
            match_counts[mid] = match_counts.get(mid, 0) + 1
            total_net_pnl[mid] = total_net_pnl.get(mid, 0.0) + r.get("net_pnl", 0.0)
            total_tokens_in[mid] = total_tokens_in.get(mid, 0) + r.get("tokens_in", 0)
            total_tokens_out[mid] = total_tokens_out.get(mid, 0) + r.get("tokens_out", 0)
            total_cost_usd[mid] = total_cost_usd.get(mid, 0.0) + r.get("cost_usd", 0.0)
            n_decisions = r.get("deals_won", 0) + r.get("deals_rejected", 0) + r.get("deals_errored", 0)
            total_decisions[mid] = total_decisions.get(mid, 0) + n_decisions

            # Confusion matrix
            cm = r.get("confusion_matrix", {})
            if mid not in agg_confusion:
                agg_confusion[mid] = {
                    "good": {"approved": 0, "rejected": 0},
                    "bad": {"approved": 0, "rejected": 0},
                    "fraud": {"approved": 0, "rejected": 0},
                }
            for category in ("good", "bad", "fraud"):
                for action in ("approved", "rejected"):
                    agg_confusion[mid][category][action] += cm.get(category, {}).get(action, 0)

            # Rate stats from per_borrower
            per_b = r.get("per_borrower", {})
            if mid not in rate_stats:
                rate_stats[mid] = {"good_rates": [], "bad_rates": []}
            for bid, bdata in per_b.items():
                rate = bdata.get("rate_offered")
                if rate is None:
                    continue
                gt = bdata.get("ground_truth", "good")
                if gt == "good":
                    rate_stats[mid]["good_rates"].append(rate)
                else:
                    rate_stats[mid]["bad_rates"].append(rate)

        # Record Elo snapshot
        for mid in all_models:
            elo_history[mid].append({
                "match_id": match["match_id"],
                "profit_elo": round(profit_ratings.get(mid, initial_elo), 1),
                "credit_elo": round(credit_ratings.get(mid, initial_elo), 1),
                "dealshare_elo": round(dealshare_ratings.get(mid, initial_elo), 1),
            })

    # Compute composite Elo for single-number leaderboard ranking
    composite = compute_composite_elo(profit_ratings, credit_ratings, dealshare_ratings)

    # Build standings
    standings = []
    for mid in all_models:
        n = match_counts.get(mid, 0)
        good_rates = rate_stats.get(mid, {}).get("good_rates", [])
        bad_rates = rate_stats.get(mid, {}).get("bad_rates", [])

        standings.append({
            "model_id": mid,
            "display_name": all_models[mid],
            "composite_elo": round(composite.get(mid, initial_elo), 1),
            "profit_elo": round(profit_ratings.get(mid, initial_elo), 1),
            "credit_elo": round(credit_ratings.get(mid, initial_elo), 1),
            "dealshare_elo": round(dealshare_ratings.get(mid, initial_elo), 1),
            "matches_played": n,
            "avg_net_pnl": round(total_net_pnl.get(mid, 0.0) / n, 2) if n > 0 else 0.0,
            "confusion_agg": agg_confusion.get(mid, {}),
            "rate_analysis": {
                "avg_rate_good": round(sum(good_rates) / len(good_rates), 2) if good_rates else None,
                "avg_rate_bad": round(sum(bad_rates) / len(bad_rates), 2) if bad_rates else None,
                "n_good_offers": len(good_rates),
                "n_bad_offers": len(bad_rates),
            },
            "total_tokens_in": total_tokens_in.get(mid, 0),
            "total_tokens_out": total_tokens_out.get(mid, 0),
            "total_cost_usd": round(total_cost_usd.get(mid, 0.0), 4),
            "avg_cost_per_decision": (
                round(total_cost_usd.get(mid, 0.0) / total_decisions[mid], 4)
                if total_decisions.get(mid, 0) > 0 else 0.0
            ),
        })

    # Sort by Composite Elo descending (single-number ranking)
    standings.sort(key=lambda s: s["composite_elo"], reverse=True)

    return {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_matches": len(match_ids),
        "config": {
            "k": k,
            "initial_elo": initial_elo,
            "utility_epsilon": config.get("utility_epsilon", UTILITY_EPSILON),
            "elo_match_cap": ELO_MATCH_CAP,
            "composite_weights": COMPOSITE_WEIGHTS,
        },
        "standings": standings,
        "match_ids": match_ids,
        "elo_history": elo_history,
    }


def write_leaderboard(leaderboard: dict | None = None) -> Path:
    """Compute (if needed) and write leaderboard.json."""
    if leaderboard is None:
        leaderboard = compute_leaderboard()

    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard, f, indent=2, default=str)

    return LEADERBOARD_FILE


def load_leaderboard() -> dict | None:
    """Load the cached leaderboard.json."""
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# High-level: emit + update
# ---------------------------------------------------------------------------

def emit_and_update(record: dict) -> tuple[Path, Path]:
    """Write a match record and update the leaderboard.

    Returns (match_file_path, leaderboard_file_path).
    """
    match_path = write_match_record(record)
    leaderboard = compute_leaderboard()
    lb_path = write_leaderboard(leaderboard)
    return match_path, lb_path
