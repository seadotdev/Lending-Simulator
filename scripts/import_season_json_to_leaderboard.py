#!/usr/bin/env python3
"""Import exported season JSON files into the git-native leaderboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loanville.leaderboard import (
    build_match_record,
    compute_leaderboard,
    load_config,
    write_leaderboard,
    write_match_record,
)
from loanville.models import ECONOMICS_PRESETS
from loanville.scoring import compute_loan_payoff


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _is_error(decision: dict) -> bool:
    reasoning = str(decision.get("reasoning", "") or "")
    return reasoning.startswith("[LLM_ERROR]")


def _import_one(
    path: Path,
    economics_name: str,
    drop_high_error: bool,
    max_error_rate: float,
) -> tuple[Path, dict]:
    data = _load_json(path)
    eco = ECONOMICS_PRESETS[economics_name]
    benchmark_rate_per_dollar = eco.risk_free_rate * (eco.sim_horizon_months / 12.0)

    lenders = data.get("lenders", [])
    lender_by_id = {l["id"]: l for l in lenders}
    model_rows = [
        {"model_id": f"{l['model']}::{l['id']}", "display_name": l["name"]}
        for l in lenders
    ]

    borrower_truth: dict[str, str] = {}
    borrower_amount: dict[str, float] = {}
    winners: dict[str, str] = {}
    all_decisions: dict[str, list[dict]] = {l["id"]: [] for l in lenders}
    deployed_by_lender: dict[str, float] = {l["id"]: 0.0 for l in lenders}

    for week in data.get("weeks", []):
        borrowers = week.get("borrowers", []) or []
        for b in borrowers:
            bid = b["id"]
            borrower_truth[bid] = b.get("true_outcome", "good")
            borrower_amount[bid] = float(b.get("amount", 0.0) or 0.0)

        for d in (week.get("decisions", []) or []):
            lid = d.get("lender_id")
            if lid in all_decisions:
                all_decisions[lid].append(d)

        for loan in (week.get("booked_loans", []) or []):
            bid = loan["borrower_id"]
            lid = loan["lender_id"]
            winners[bid] = lid
            if lid in deployed_by_lender:
                deployed_by_lender[lid] += float(loan.get("principal", 0.0) or 0.0)

    n_borrowers = len(borrower_truth)
    season_mix = str((data.get("config", {}) or {}).get("season_mix", "realistic"))
    mix = f"season-{season_mix}"

    results = []
    model_rows_filtered = []
    for lender in lenders:
        lid = lender["id"]
        decisions = all_decisions.get(lid, [])

        cm = {
            "good": {"approved": 0, "rejected": 0},
            "bad": {"approved": 0, "rejected": 0},
            "fraud": {"approved": 0, "rejected": 0},
        }
        per_borrower: dict[str, dict] = {}

        deals_errored = 0
        deals_won = 0
        total_evals = len(decisions)

        for d in decisions:
            bid = d.get("borrower_id")
            if not bid:
                continue
            truth = borrower_truth.get(bid, "good")
            errored = _is_error(d)
            decision = str(d.get("decision", "REJECT"))
            ts = d.get("term_sheet")
            rate_offered = None
            loan_amount = borrower_amount.get(bid, 0.0)
            term_months = 24
            if isinstance(ts, dict):
                rate_offered = ts.get("interest_rate")
                loan_amount = float(ts.get("loan_amount", loan_amount) or loan_amount)
                term_months = int(ts.get("term_months", term_months) or term_months)

            if errored:
                deals_errored += 1
            else:
                if decision == "APPROVE":
                    cm[truth]["approved"] += 1
                else:
                    cm[truth]["rejected"] += 1

            if decision != "APPROVE":
                decision_state = "declined"
                utility = loan_amount * benchmark_rate_per_dollar
            elif winners.get(bid) == lid:
                decision_state = "won"
                deals_won += 1
                payoff = compute_loan_payoff(
                    principal=loan_amount,
                    interest_rate=float(rate_offered or 0.0),
                    term_months=term_months,
                    true_outcome=truth,
                    months_before_default=None,
                    economics=eco,
                )
                utility = payoff["net_profit"]
            else:
                decision_state = "lost"
                utility = loan_amount * benchmark_rate_per_dollar

            per_borrower[bid] = {
                "decision_state": decision_state,
                "ground_truth": truth,
                "utility": utility,
                "rate_offered": rate_offered,
            }

        deals_rejected = max(0, total_evals - deals_won - deals_errored)
        lender_totals = lender_by_id[lid]
        net_pnl = float(lender_totals.get("net_pnl", 0.0) or 0.0)
        total_capital = float(lender_totals.get("total_capital", 0.0) or 0.0)
        raroc_like = (net_pnl / total_capital * 100.0) if total_capital > 0 else 0.0

        result = (
            {
                "model_id": f"{lender['model']}::{lid}",
                "raroc_score": raroc_like,
                "deals_won": deals_won,
                "deals_rejected": deals_rejected,
                "deals_errored": deals_errored,
                "frauds_funded": int(lender_totals.get("frauds_funded", 0) or 0),
                "defaults": int(lender_totals.get("defaults", 0) or 0),
                "deployed": round(deployed_by_lender.get(lid, 0.0), 2),
                "net_pnl": net_pnl,
                "confusion_matrix": cm,
                "per_borrower": per_borrower,
                "tokens_in": int(lender_totals.get("tokens_in", 0) or 0),
                "tokens_out": int(lender_totals.get("tokens_out", 0) or 0),
                "cost_usd": round(float(lender_totals.get("cost_usd", 0.0) or 0.0), 4),
            }
        )
        error_rate = (deals_errored / total_evals) if total_evals else 0.0
        if drop_high_error and error_rate > max_error_rate:
            continue
        results.append(result)
        model_rows_filtered.append(
            {"model_id": f"{lender['model']}::{lid}", "display_name": lender["name"]}
        )

    record = build_match_record(
        models=model_rows_filtered if drop_high_error else model_rows,
        results=results,
        mix=mix,
        n_borrowers=n_borrowers,
    )
    match_path = write_match_record(record)
    return match_path, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Import season JSON exports into leaderboard.")
    parser.add_argument("season_json", nargs="+", help="Season JSON files to import.")
    parser.add_argument(
        "--economics",
        default="balanced",
        choices=list(ECONOMICS_PRESETS.keys()),
        help="Economics preset used to compute per-borrower utility for won loans.",
    )
    parser.add_argument(
        "--drop-high-error",
        action="store_true",
        help="Exclude models whose error rate exceeds leaderboard max_error_rate.",
    )
    args = parser.parse_args()

    cfg = load_config()
    max_error_rate = float(cfg.get("max_error_rate", 0.25))

    imported = []
    for raw in args.season_json:
        path = Path(raw)
        match_path, record = _import_one(
            path=path,
            economics_name=args.economics,
            drop_high_error=args.drop_high_error,
            max_error_rate=max_error_rate,
        )
        imported.append((match_path, record))
        print(f"Imported {path} -> {match_path.name} | valid={record['validation']['valid']}")
        if record["validation"]["errors"]:
            for e in record["validation"]["errors"]:
                print(f"  - {e}")

    lb = compute_leaderboard()
    lb_path = write_leaderboard(lb)
    print(f"Leaderboard updated: {lb_path}")


if __name__ == "__main__":
    main()
