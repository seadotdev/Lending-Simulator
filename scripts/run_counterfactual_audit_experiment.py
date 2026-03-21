from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from loanville.data import get_borrowers, get_lenders
from loanville.models import Borrower, ExistingLoan, LenderConfig


DEFAULT_MATCH = Path("leaderboard/matches/2026-02-25T07-44-55_c002ce.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/pre_submission_experiments")


RISK_PROFILES = {
    "Velocity Capital": {
        "margin_min_pct": 5.0,
        "dscr_min": 1.00,
        "leverage_max": 2.00,
        "max_deficit_months": 1,
    },
    "Heritage Trust Bank": {
        "margin_min_pct": 8.0,
        "dscr_min": 1.25,
        "leverage_max": 1.50,
        "max_deficit_months": 0,
    },
    "Meridian Partners": {
        "margin_min_pct": 6.0,
        "dscr_min": 1.15,
        "leverage_max": 1.80,
        "max_deficit_months": 1,
    },
}


CURATED_CASES = [
    {
        "model_id": "google/gemini-2.5-flash",
        "borrower_id": "BRW-011",
        "protocol_state": {
            "fraud_screened": False,
            "dscr_checked": False,
            "leverage_checked": False,
            "cashflow_reviewed": True,
            "concentration_checked": True,
        },
    },
    {
        "model_id": "google/gemini-2.5-flash",
        "borrower_id": "BRW-025",
        "protocol_state": {
            "fraud_screened": True,
            "cashflow_reviewed": False,
            "concentration_checked": True,
        },
    },
    {
        "model_id": "google/gemini-2.5-flash",
        "borrower_id": "BRW-001",
        "protocol_state": {
            "fraud_screened": True,
            "cashflow_reviewed": True,
            "concentration_checked": True,
        },
    },
    {
        "model_id": "deepseek/deepseek-chat-v3-0324",
        "borrower_id": "BRW-025",
        "protocol_state": {
            "fraud_screened": True,
            "cashflow_reviewed": True,
            "concentration_checked": True,
        },
    },
    {
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "borrower_id": "BRW-001",
        "protocol_state": {
            "fraud_screened": True,
            "cashflow_reviewed": True,
            "concentration_checked": True,
        },
    },
]


@dataclass
class ClauseResult:
    name: str
    status: str
    detail: str


def amortized_monthly_payment(principal: float, annual_rate_pct: float, term_months: int = 24) -> float:
    rate = annual_rate_pct / 100.0 / 12.0
    if principal <= 0 or term_months <= 0:
        return 0.0
    if rate == 0:
        return principal / term_months
    factor = (rate * (1 + rate) ** term_months) / ((1 + rate) ** term_months - 1)
    return principal * factor


def annual_debt_service(principal: float, annual_rate_pct: float, term_months: int = 24) -> float:
    return amortized_monthly_payment(principal, annual_rate_pct, term_months=term_months) * 12.0


def detect_related_party_transfer_flag(borrower: Borrower) -> bool:
    descriptions = []
    for stmt in borrower.dossier.bank_statements:
        descriptions.extend(d.description for d in stmt.deposits)
    return any("Transfer from" in desc for desc in descriptions)


def cashflow_deficit_months(borrower: Borrower, annual_rate_pct: float, term_months: int = 24) -> int:
    monthly_payment = amortized_monthly_payment(
        borrower.dossier.loan_request_amount,
        annual_rate_pct,
        term_months=term_months,
    )
    deficits = 0
    for stmt in borrower.dossier.bank_statements:
        monthly_surplus = stmt.total_deposits - stmt.total_withdrawals - monthly_payment
        if monthly_surplus < 0:
            deficits += 1
    return deficits


def sector_exposure(lender: LenderConfig, sector: str) -> float:
    return sum(loan.remaining_balance for loan in lender.existing_portfolio if loan.sector == sector)


def get_thresholds(lender: LenderConfig) -> dict[str, float]:
    return RISK_PROFILES.get(
        lender.name,
        {
            "margin_min_pct": 7.0,
            "dscr_min": 1.15,
            "leverage_max": 1.75,
            "max_deficit_months": 1,
        },
    )


def evaluate_policy(
    borrower: Borrower,
    lender: LenderConfig,
    *,
    annual_rate_pct: float | None = None,
    protocol_state: dict | None = None,
) -> dict:
    protocol_state = protocol_state or {}
    thresholds = get_thresholds(lender)
    rate = annual_rate_pct if annual_rate_pct is not None else lender.target_yield_pct
    principal = borrower.dossier.loan_request_amount
    revenue = borrower.dossier.annual_revenue
    net_income = borrower.dossier.net_income
    margin_pct = (net_income / revenue * 100.0) if revenue > 0 else 0.0
    debt_service = annual_debt_service(principal, rate)
    dscr = (net_income / debt_service) if debt_service > 0 else math.inf
    leverage = (principal / net_income) if net_income > 0 else math.inf
    deficits = cashflow_deficit_months(borrower, rate)
    fraud_flag = detect_related_party_transfer_flag(borrower)

    clauses: list[ClauseResult] = []

    clauses.append(
        ClauseResult(
            name="max_single_loan",
            status="pass" if principal <= lender.max_single_loan else "fail",
            detail=f"request={principal:,.0f}, max={lender.max_single_loan:,.0f}",
        )
    )
    clauses.append(
        ClauseResult(
            name="margin_threshold",
            status="pass" if margin_pct >= thresholds["margin_min_pct"] else "fail",
            detail=f"margin={margin_pct:.2f}%, min={thresholds['margin_min_pct']:.2f}%",
        )
    )
    clauses.append(
        ClauseResult(
            name="dscr_threshold",
            status=(
                "pass"
                if (not protocol_state.get("dscr_checked", True) or dscr >= thresholds["dscr_min"])
                else "fail"
            ),
            detail=f"dscr={dscr:.2f}, min={thresholds['dscr_min']:.2f}",
        )
    )
    clauses.append(
        ClauseResult(
            name="leverage_threshold",
            status=(
                "pass"
                if (not protocol_state.get("leverage_checked", True) or leverage <= thresholds["leverage_max"])
                else "fail"
            ),
            detail=f"leverage={leverage:.2f}x, max={thresholds['leverage_max']:.2f}x",
        )
    )

    if protocol_state.get("fraud_screened", True):
        clauses.append(
            ClauseResult(
                name="fraud_related_party_flag",
                status="fail" if fraud_flag else "pass",
                detail=f"related_party_transfers={'yes' if fraud_flag else 'no'}",
            )
        )

    if protocol_state.get("cashflow_reviewed", True):
        clauses.append(
            ClauseResult(
                name="cashflow_stress",
                status="pass" if deficits <= thresholds["max_deficit_months"] else "fail",
                detail=f"deficit_months={deficits}, max={thresholds['max_deficit_months']}",
            )
        )

    if protocol_state.get("concentration_checked", True):
        current_sector = sector_exposure(lender, borrower.dossier.sector)
        sector_limit = lender.sector_limits.get(borrower.dossier.sector, 0.25)
        max_allowed = lender.total_capital * sector_limit
        projected = current_sector + principal
        clauses.append(
            ClauseResult(
                name="sector_concentration",
                status="pass" if projected <= max_allowed else "fail",
                detail=f"projected={projected:,.0f}, max={max_allowed:,.0f}",
            )
        )

    failed = [c for c in clauses if c.status == "fail"]
    action = "conditional_offer" if not failed else "refuse"
    return {
        "action": action,
        "rate_pct": rate,
        "metrics": {
            "margin_pct": round(margin_pct, 2),
            "dscr": round(dscr, 3),
            "leverage": round(leverage, 3),
            "cashflow_deficit_months": deficits,
            "fraud_related_party_flag": fraud_flag,
            "sector_exposure": sector_exposure(lender, borrower.dossier.sector),
        },
        "clauses": [asdict(c) for c in clauses],
        "failed_clauses": [c.name for c in failed],
    }


def infer_observed_decision(state: str) -> str:
    return "conditional_offer" if state in {"won", "lost"} else "refuse"


def clone_case(borrower: Borrower, lender: LenderConfig) -> tuple[Borrower, LenderConfig]:
    return copy.deepcopy(borrower), copy.deepcopy(lender)


def set_sector_exposure(lender: LenderConfig, sector: str, target_exposure: float) -> None:
    others = [loan for loan in lender.existing_portfolio if loan.sector != sector]
    current = sector_exposure(lender, sector)
    if current == 0:
        if target_exposure > 0:
            others.append(
                ExistingLoan(
                    borrower_name=f"Synthetic {sector} exposure",
                    sector=sector,
                    original_amount=target_exposure,
                    remaining_balance=target_exposure,
                    interest_rate=9.0,
                    months_remaining=18,
                )
            )
        lender.existing_portfolio = others
        return

    remaining = max(0.0, target_exposure)
    sector_loans = [loan for loan in lender.existing_portfolio if loan.sector == sector]
    for loan in sector_loans:
        if remaining <= 0:
            loan.remaining_balance = 0.0
        else:
            new_balance = min(loan.remaining_balance, remaining)
            remaining -= new_balance
            loan.remaining_balance = new_balance
    lender.existing_portfolio = others + [loan for loan in sector_loans if loan.remaining_balance > 0]


def binary_search_requested_amount(
    borrower: Borrower,
    lender: LenderConfig,
    protocol_state: dict,
    target_action: str,
    rate_pct: float,
) -> float:
    original = borrower.dossier.loan_request_amount
    lo, hi = 0.0, original
    best: float | None = None
    for _ in range(32):
        mid = (lo + hi) / 2.0
        borrower.dossier.loan_request_amount = mid
        action = evaluate_policy(
            borrower,
            lender,
            annual_rate_pct=rate_pct,
            protocol_state=protocol_state,
        )["action"]
        if action == target_action:
            best = mid
            lo = mid
        else:
            hi = mid
    borrower.dossier.loan_request_amount = original
    chosen = best if best is not None else original
    return round(chosen * 0.995, 2)


def binary_search_net_income(
    borrower: Borrower,
    lender: LenderConfig,
    protocol_state: dict,
    target_action: str,
    rate_pct: float,
    direction: str,
) -> float:
    original = borrower.dossier.net_income
    if direction == "down":
        lo, hi = 0.0, original
        best: float | None = None
        for _ in range(32):
            mid = (lo + hi) / 2.0
            borrower.dossier.net_income = mid
            action = evaluate_policy(
                borrower,
                lender,
                annual_rate_pct=rate_pct,
                protocol_state=protocol_state,
            )["action"]
            if action == target_action:
                best = mid
                lo = mid
            else:
                hi = mid
    else:
        lo, hi = original, original * 4.0
        best = hi
        for _ in range(32):
            mid = (lo + hi) / 2.0
            borrower.dossier.net_income = mid
            action = evaluate_policy(
                borrower,
                lender,
                annual_rate_pct=rate_pct,
                protocol_state=protocol_state,
            )["action"]
            if action == target_action:
                best = mid
                hi = mid
            else:
                lo = mid
    borrower.dossier.net_income = original
    chosen = best if best is not None else original
    if direction == "down":
        chosen *= 0.995
    else:
        chosen *= 1.005
    return round(chosen, 2)


def deficit_month_indices(borrower: Borrower, annual_rate_pct: float) -> list[int]:
    monthly_payment = amortized_monthly_payment(
        borrower.dossier.loan_request_amount,
        annual_rate_pct,
        term_months=24,
    )
    out: list[int] = []
    for idx, stmt in enumerate(borrower.dossier.bank_statements):
        if stmt.total_deposits - stmt.total_withdrawals - monthly_payment < 0:
            out.append(idx)
    return out


def binary_search_cashflow_buffer(
    borrower: Borrower,
    lender: LenderConfig,
    protocol_state: dict,
    rate_pct: float,
) -> dict:
    months = deficit_month_indices(borrower, rate_pct)
    lo, hi = 0.0, 100_000.0
    best: float | None = None
    for _ in range(32):
        mid = (lo + hi) / 2.0
        b2, l2 = clone_case(borrower, lender)
        for idx in months:
            if b2.dossier.bank_statements[idx].deposits:
                b2.dossier.bank_statements[idx].deposits[0].amount += mid
        action = evaluate_policy(
            b2,
            l2,
            annual_rate_pct=rate_pct,
            protocol_state=protocol_state,
        )["action"]
        if action == "conditional_offer":
            best = mid
            hi = mid
        else:
            lo = mid
    return {
        "kind": "banking_fact",
        "field": "deficit_month_revenue_buffer",
        "months": months,
        "from": 0.0,
        "to": round((best if best is not None else hi) * 1.005, 2),
        "summary": "increase verified deposits in deficit months until seasonal cashflow passes policy",
    }


def build_intervention(
    borrower: Borrower,
    lender: LenderConfig,
    observed_action: str,
    baseline_eval: dict,
    protocol_state: dict,
) -> dict:
    failed = baseline_eval["failed_clauses"]
    rate_pct = baseline_eval["rate_pct"]

    if observed_action == "conditional_offer":
        if not protocol_state.get("fraud_screened", True) and detect_related_party_transfer_flag(borrower):
            return {
                "kind": "protocol_state",
                "field": "fraud_screened",
                "from": False,
                "to": True,
                "summary": "enable related-party transfer screening",
            }
        if not protocol_state.get("dscr_checked", True):
            return {
                "kind": "protocol_state",
                "field": "dscr_checked",
                "from": False,
                "to": True,
                "summary": "enforce DSCR clause checking before approval",
            }
        if not protocol_state.get("leverage_checked", True):
            return {
                "kind": "protocol_state",
                "field": "leverage_checked",
                "from": False,
                "to": True,
                "summary": "enforce leverage clause checking before approval",
            }
        if not protocol_state.get("cashflow_reviewed", True):
            return {
                "kind": "protocol_state",
                "field": "cashflow_reviewed",
                "from": False,
                "to": True,
                "summary": "require seasonal cashflow review before approval",
            }
        target_net_income = binary_search_net_income(
            borrower,
            lender,
            protocol_state,
            "refuse",
            rate_pct,
            direction="down",
        )
        return {
            "kind": "fact",
            "field": "net_income",
            "from": borrower.dossier.net_income,
            "to": target_net_income,
            "summary": "reduce verified net income until the approval fails policy",
        }

    if "sector_concentration" in failed:
        sector_limit = lender.sector_limits.get(borrower.dossier.sector, 0.25)
        max_allowed = lender.total_capital * sector_limit
        target_exposure = max(0.0, max_allowed - borrower.dossier.loan_request_amount)
        return {
            "kind": "lender_state",
            "field": "sector_exposure",
            "sector": borrower.dossier.sector,
            "from": sector_exposure(lender, borrower.dossier.sector),
            "to": round(target_exposure, 2),
            "summary": "reduce existing sector exposure so the deal fits the policy cap",
        }

    if "cashflow_stress" in failed or "dscr_threshold" in failed or "leverage_threshold" in failed:
        target_amount = binary_search_requested_amount(
            borrower,
            lender,
            protocol_state,
            "conditional_offer",
            rate_pct,
        )
        borrower_probe, lender_probe = clone_case(borrower, lender)
        borrower_probe.dossier.loan_request_amount = target_amount
        amount_eval = evaluate_policy(
            borrower_probe,
            lender_probe,
            annual_rate_pct=rate_pct,
            protocol_state=protocol_state,
        )
        if amount_eval["action"] != "conditional_offer":
            return binary_search_cashflow_buffer(
                borrower,
                lender,
                protocol_state,
                rate_pct,
            )
        return {
            "kind": "fact",
            "field": "loan_request_amount",
            "from": borrower.dossier.loan_request_amount,
            "to": target_amount,
            "summary": "reduce requested amount until cashflow and coverage clear policy",
        }

    if "margin_threshold" in failed:
        target_net_income = binary_search_net_income(
            borrower,
            lender,
            protocol_state,
            "conditional_offer",
            rate_pct,
            direction="up",
        )
        return {
            "kind": "fact",
            "field": "net_income",
            "from": borrower.dossier.net_income,
            "to": target_net_income,
            "summary": "raise verified net income until the policy would permit an offer",
        }

    return {
        "kind": "noop",
        "summary": "no single-variable intervention found",
    }


def apply_intervention(
    borrower: Borrower,
    lender: LenderConfig,
    protocol_state: dict,
    intervention: dict,
) -> tuple[Borrower, LenderConfig, dict]:
    borrower2, lender2 = clone_case(borrower, lender)
    protocol2 = dict(protocol_state)
    kind = intervention.get("kind")
    if kind == "protocol_state":
        protocol2[intervention["field"]] = intervention["to"]
    elif kind == "fact":
        setattr(borrower2.dossier, intervention["field"], intervention["to"])
    elif kind == "lender_state":
        set_sector_exposure(lender2, intervention["sector"], intervention["to"])
    elif kind == "banking_fact":
        for idx in intervention.get("months", []):
            if borrower2.dossier.bank_statements[idx].deposits:
                borrower2.dossier.bank_statements[idx].deposits[0].amount += intervention["to"]
    return borrower2, lender2, protocol2


def estimate_robustness(
    borrower: Borrower,
    lender: LenderConfig,
    protocol_state: dict,
    intervention: dict,
    expected_action: str,
    trials: int = 50,
    seed: int = 42,
) -> float:
    rng = random.Random(seed)
    successes = 0
    for _ in range(trials):
        b2, l2, p2 = apply_intervention(borrower, lender, protocol_state, intervention)
        if intervention.get("field") != "net_income":
            b2.dossier.net_income *= rng.uniform(0.97, 1.03)
        if intervention.get("field") != "loan_request_amount":
            b2.dossier.loan_request_amount *= rng.uniform(0.98, 1.02)
        action = evaluate_policy(b2, l2, protocol_state=p2)["action"]
        if action == expected_action:
            successes += 1
    return round(successes / trials, 2)


def intervention_minimality(
    borrower: Borrower,
    lender: LenderConfig,
    protocol_state: dict,
    intervention: dict,
    flipped_action: str,
) -> bool:
    if intervention.get("kind") == "protocol_state":
        return True
    b2, l2, p2 = apply_intervention(borrower, lender, protocol_state, intervention)
    current_value = intervention.get("to")
    if intervention["field"] == "loan_request_amount":
        probe = current_value * 1.02
        b2.dossier.loan_request_amount = probe
    elif intervention["field"] == "net_income":
        if current_value <= borrower.dossier.net_income:
            probe = current_value * 1.02
        else:
            probe = current_value * 0.98
        b2.dossier.net_income = probe
    elif intervention["field"] == "sector_exposure":
        b2, l2 = clone_case(borrower, lender)
        p2 = dict(protocol_state)
        set_sector_exposure(l2, intervention["sector"], current_value * 1.02)
    else:
        return True
    replay = evaluate_policy(b2, l2, protocol_state=p2)["action"]
    return replay != flipped_action


def describe_counterfactual(intervention: dict, flipped_action: str) -> str:
    field = intervention.get("field", "")
    if intervention.get("kind") == "protocol_state":
        return f"with protocol state `{field}=true`, outcome changes to {flipped_action}"
    if field == "loan_request_amount":
        return (
            f"with requested amount reduced from {intervention['from']:,.0f} "
            f"to {intervention['to']:,.0f}, outcome changes to {flipped_action}"
        )
    if field == "net_income":
        return (
            f"with verified net income changed from {intervention['from']:,.0f} "
            f"to {intervention['to']:,.0f}, outcome changes to {flipped_action}"
        )
    if field == "deficit_month_revenue_buffer":
        return (
            f"with an additional verified {intervention['to']:,.0f} per deficit month "
            f"across months {intervention['months']}, outcome changes to {flipped_action}"
        )
    if field == "sector_exposure":
        return (
            f"with existing {intervention['sector']} exposure reduced from {intervention['from']:,.0f} "
            f"to {intervention['to']:,.0f}, outcome changes to {flipped_action}"
        )
    return intervention.get("summary", "")


def build_case_index(match_data: dict) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    model_to_display = {
        entry["model_id"]: entry["display_name"]
        for entry in match_data.get("models", [])
    }
    for result in match_data.get("results", []):
        model_id = result["model_id"]
        for borrower_id, payload in result.get("per_borrower", {}).items():
            rows[(model_id, borrower_id)] = {
                "display_name": model_to_display.get(model_id, ""),
                "per_borrower": payload,
            }
    return rows


def run_experiment(match_path: Path) -> dict:
    match_data = json.loads(match_path.read_text())
    borrowers = {b.id: b for b in get_borrowers("all")}
    lenders = {l.name: l for l in get_lenders()}
    indexed = build_case_index(match_data)

    cases = []
    replay_valid_count = 0
    minimal_count = 0
    robustness_values: list[float] = []
    policy_violation_count = 0

    for spec in CURATED_CASES:
        row = indexed[(spec["model_id"], spec["borrower_id"])]
        borrower = borrowers[spec["borrower_id"]]
        lender = lenders[row["display_name"]]
        observed = row["per_borrower"]
        observed_action = infer_observed_decision(observed["decision_state"])
        rate_pct = observed.get("rate_offered") or lender.target_yield_pct
        protocol_state = dict(spec["protocol_state"])

        baseline_eval = evaluate_policy(
            borrower,
            lender,
            annual_rate_pct=rate_pct,
            protocol_state=protocol_state,
        )
        if baseline_eval["action"] != observed_action:
            raise RuntimeError(
                f"SCM baseline mismatch for {spec['model_id']} {spec['borrower_id']}: "
                f"observed={observed_action} baseline={baseline_eval['action']}"
            )

        intervention = build_intervention(
            borrower,
            lender,
            observed_action,
            baseline_eval,
            protocol_state,
        )
        flipped_action = "refuse" if observed_action == "conditional_offer" else "conditional_offer"
        borrower2, lender2, protocol2 = apply_intervention(borrower, lender, protocol_state, intervention)
        counterfactual_eval = evaluate_policy(
            borrower2,
            lender2,
            annual_rate_pct=rate_pct,
            protocol_state=protocol2,
        )
        replay_valid = counterfactual_eval["action"] == flipped_action
        minimal = intervention_minimality(
            borrower,
            lender,
            protocol_state,
            intervention,
            flipped_action,
        )
        robustness = estimate_robustness(
            borrower,
            lender,
            protocol_state,
            intervention,
            flipped_action,
        )

        replay_valid_count += int(replay_valid)
        minimal_count += int(minimal)
        robustness_values.append(robustness)
        full_protocol_eval = evaluate_policy(
            borrower,
            lender,
            annual_rate_pct=rate_pct,
            protocol_state={
                "fraud_screened": True,
                "dscr_checked": True,
                "leverage_checked": True,
                "cashflow_reviewed": True,
                "concentration_checked": True,
            },
        )
        if observed_action == "conditional_offer" and full_protocol_eval["failed_clauses"]:
            policy_violation_count += 1

        violated_clause = full_protocol_eval["failed_clauses"][0] if full_protocol_eval["failed_clauses"] else (baseline_eval["failed_clauses"][0] if baseline_eval["failed_clauses"] else None)
        if intervention.get("field") == "fraud_screened":
            violated_clause = "fraud_related_party_flag"
        if intervention.get("field") == "cashflow_reviewed":
            violated_clause = "cashflow_stress"

        cases.append(
            {
                "model_id": spec["model_id"],
                "lender_name": lender.name,
                "borrower_id": borrower.id,
                "borrower_name": borrower.dossier.company_name,
                "ground_truth": borrower.true_outcome,
                "observed_decision": observed_action,
                "observed_rate_pct": observed.get("rate_offered"),
                "baseline_protocol_state": protocol_state,
                "baseline_evaluation": baseline_eval,
                "full_protocol_evaluation": full_protocol_eval,
                "violated_clause": violated_clause,
                "critical_missing_fact": intervention.get("field"),
                "intervention": intervention,
                "counterfactual": describe_counterfactual(intervention, flipped_action),
                "counterfactual_evaluation": counterfactual_eval,
                "replay_valid": replay_valid,
                "minimal": minimal,
                "robustness": robustness,
            }
        )

    summary = {
        "match_id": match_data["match_id"],
        "n_decisions": len(cases),
        "policy_violation_rate": round(policy_violation_count / len(cases), 2),
        "replay_validity": round(replay_valid_count / len(cases), 2),
        "minimality_rate": round(minimal_count / len(cases), 2),
        "average_robustness": round(sum(robustness_values) / len(robustness_values), 2),
    }
    return {
        "summary": summary,
        "cases": cases,
    }


def write_markdown(result: dict, out_path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Counterfactual Audit Experiment",
        "",
        f"- Match: `{summary['match_id']}`",
        f"- Decisions audited: `{summary['n_decisions']}`",
        f"- SCM replay validity: `{summary['replay_validity']:.0%}`",
        f"- Minimality rate: `{summary['minimality_rate']:.0%}`",
        f"- Average robustness: `{summary['average_robustness']:.2f}`",
        f"- Observed policy violation rate in the audited sample: `{summary['policy_violation_rate']:.0%}`",
        "",
        "## Audit Statements",
        "",
    ]
    for case in result["cases"]:
        lines.extend(
            [
                f"### {case['model_id']} on {case['borrower_id']} ({case['borrower_name']})",
                f"- Decision: `{case['observed_decision']}`",
                f"- Ground truth: `{case['ground_truth']}`",
                f"- Violated clause: `{case['violated_clause']}`",
                f"- Counterfactual: {case['counterfactual']}",
                f"- Robustness: `{case['robustness']}`",
                "",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-file", type=Path, default=DEFAULT_MATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = run_experiment(args.match_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "experiment2_counterfactual_audit.json"
    md_path = args.output_dir / "experiment2_counterfactual_audit.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)

    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
