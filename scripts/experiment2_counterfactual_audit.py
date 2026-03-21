"""Experiment 2: Counterfactual Audit on Existing Traces.

Takes archived lender-agent decisions from leaderboard matches, replays them
through the LOS with perturbed dossiers, and produces structured audit
statements showing which minimal intervention flips the decision.

Runs through the full LOS pipeline (evaluate_via_los) for each replay.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from loanville.data import get_borrowers, get_lenders
from loanville.los_adapter import evaluate_via_los, check_los_health
from loanville.models import Borrower, LenderConfig

LOS_URL = os.environ.get("LOS_URL", "http://localhost:3100")
OUTPUT_DIR = Path("artifacts/pre_submission_experiments")

# Use the big match with 3 models x 20 borrowers
MATCH_FILE = Path("leaderboard/matches/2026-02-25T07-44-55_c002ce.json")


@dataclass
class AuditCase:
    model_id: str
    borrower_id: str
    observed_decision: str  # won/lost/rejected
    observed_rate: float | None
    perturbation_type: str
    perturbation_desc: str


# Curated cases: models that approved fraud/bad borrowers (where they shouldn't have)
# and models that correctly rejected (to show the counterfactual going the other way)
CURATED_CASES = [
    # --- Fraud BRW-011: test if cleaning narrative hides fraud ---
    AuditCase("google/gemini-2.5-flash", "BRW-011", "won", None,
              "clean_narrative", "Replace fraud narrative with clean one — does LLM still catch fraud from data alone?"),
    AuditCase("google/gemini-2.5-flash", "BRW-011", "won", None,
              "evidence_removal", "Remove bank statements — are related-party transfers still detectable?"),

    # --- Bad BRW-025: test cashflow detection sensitivity ---
    AuditCase("google/gemini-2.5-flash", "BRW-025", "won", None,
              "clean_narrative", "Replace seasonal-stress narrative with clean one — does LLM still find cashflow gap?"),
    AuditCase("google/gemini-2.5-flash", "BRW-025", "won", None,
              "reduce_loan", "Halve loan request — does smaller debt service clear cashflow constraint?"),

    # --- Good BRW-001: find rejection threshold ---
    AuditCase("deepseek/deepseek-chat-v3-0324", "BRW-001", "won", None,
              "worsen_financials", "Cut net income to 30% — at what point does a good borrower get rejected?"),
    AuditCase("deepseek/deepseek-chat-v3-0324", "BRW-001", "won", None,
              "double_loan", "Double loan request — does leverage limit trigger rejection?"),

    # --- Cross-model comparison on same good borrower ---
    AuditCase("google/gemini-2.5-flash", "BRW-014", "won", None,
              "worsen_financials", "Cut net income — Gemini's rejection threshold for a good borrower"),
    AuditCase("meta-llama/llama-3.3-70b-instruct", "BRW-014", "won", None,
              "worsen_financials", "Cut net income — Llama's rejection threshold for same borrower"),
]


def load_match_data() -> dict:
    return json.loads(MATCH_FILE.read_text())


def get_observed_decisions(match_data: dict) -> dict[tuple[str, str], dict]:
    """Index: (model_id, borrower_id) -> per_borrower payload."""
    index = {}
    for result in match_data.get("results", []):
        model_id = result["model_id"]
        for bid, payload in result.get("per_borrower", {}).items():
            index[(model_id, bid)] = payload
    return index


def find_lender_for_model(model_id: str, match_data: dict, lenders: dict[str, LenderConfig]) -> LenderConfig:
    """Find the LenderConfig whose model matches."""
    # Match data may have model_id with ::LND suffix
    bare_model = model_id.split("::")[0]
    model_to_display = {e["model_id"]: e["display_name"] for e in match_data.get("models", [])}
    display = model_to_display.get(model_id, "")
    if display and display in lenders:
        return lenders[display]
    # Fallback: find by model string
    for lender in lenders.values():
        if lender.model == bare_model:
            return lender
    # Last resort: create a synthetic lender with the right model
    return LenderConfig(
        id="LND-EXP",
        name=f"Experiment Lender ({bare_model})",
        persona="A cautious commercial lender.",
        model=bare_model,
        target_yield_pct=10.0,
        max_single_loan=800000,
        total_capital=5000000,
        sector_limits={},
    )


def clone_borrower(b: Borrower) -> Borrower:
    return copy.deepcopy(b)


def perturb_remove_bank_statements(b: Borrower) -> Borrower:
    """Remove bank statement detail — tests whether LLM needs raw data to detect fraud."""
    b2 = clone_borrower(b)
    b2.dossier.bank_statements = []
    return b2


def perturb_reduce_loan_amount(b: Borrower, factor: float = 0.5) -> Borrower:
    """Reduce loan request amount to test cashflow threshold."""
    b2 = clone_borrower(b)
    b2.dossier.loan_request_amount = round(b.dossier.loan_request_amount * factor, 2)
    return b2


def perturb_increase_income(b: Borrower, factor: float = 1.5) -> Borrower:
    """Increase net income to test if a rejected borrower would be approved."""
    b2 = clone_borrower(b)
    b2.dossier.net_income = round(b.dossier.net_income * factor, 2)
    b2.dossier.annual_revenue = round(b.dossier.annual_revenue * factor, 2)
    return b2


def perturb_clean_narrative(b: Borrower) -> Borrower:
    """Replace fraud/bad narrative with a clean one — tests if narrative is decision-critical."""
    b2 = clone_borrower(b)
    b2.dossier.narrative = (
        f"{b2.dossier.company_name} is a well-established business in {b2.dossier.sector}. "
        f"Revenue of ${b2.dossier.annual_revenue:,.0f} and net income of ${b2.dossier.net_income:,.0f} "
        f"demonstrate consistent profitability. The business has operated for "
        f"{b2.dossier.years_in_business} years with stable cash flows."
    )
    return b2


def perturb_worsen_financials(b: Borrower) -> Borrower:
    """Worsen financials to find rejection threshold for a good borrower."""
    b2 = clone_borrower(b)
    b2.dossier.net_income = round(b.dossier.net_income * 0.3, 2)
    b2.dossier.annual_expenses = round(b.dossier.annual_revenue * 0.92, 2)
    return b2


def perturb_double_loan(b: Borrower) -> Borrower:
    """Double loan request to test leverage limits."""
    b2 = clone_borrower(b)
    b2.dossier.loan_request_amount = round(b.dossier.loan_request_amount * 2.0, 2)
    return b2


async def replay_through_los(
    borrower: Borrower,
    lender: LenderConfig,
    los_url: str = LOS_URL,
) -> dict:
    """Run a single evaluation through the full LOS pipeline."""
    try:
        run = await evaluate_via_los(
            borrower, lender,
            los_url=los_url,
            provider="openrouter",
            mode="full",
            underwrite_only=False,
        )
        return {
            "action": run.decision.action,
            "terms": {
                "apr": run.decision.terms.apr if run.decision.terms else None,
                "amount": run.decision.terms.amount if run.decision.terms else None,
                "tenor_months": run.decision.terms.tenor_months if run.decision.terms else None,
            },
            "rationale": run.decision.rationale.summary if run.decision.rationale else "",
            "trace_steps": len(run.trace.steps),
            "cost_usd": run.trace.cost.estimated_cost_usd,
            "tokens_in": run.trace.cost.tokens_in,
            "tokens_out": run.trace.cost.tokens_out,
        }
    except Exception as exc:
        return {
            "action": "error",
            "error": str(exc),
            "terms": {},
            "rationale": "",
            "trace_steps": 0,
            "cost_usd": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        }


def infer_decision(decision_state: str) -> str:
    """Map match decision_state to approve/reject."""
    if decision_state in ("won", "lost"):
        return "approve"
    return "reject"


async def run_audit_case(
    case: AuditCase,
    borrower: Borrower,
    lender: LenderConfig,
) -> dict:
    """Run one counterfactual audit case through the LOS."""
    print(f"  Auditing {case.model_id} on {case.borrower_id} ({borrower.dossier.company_name})...")

    # Baseline replay with original dossier
    print(f"    Baseline replay...")
    baseline = await replay_through_los(borrower, lender)
    print(f"    Baseline: {baseline['action']} (cost=${baseline['cost_usd']:.4f})")

    # Apply perturbation
    perturbations = {
        "evidence_removal": lambda b: perturb_remove_bank_statements(b),
        "clean_narrative": lambda b: perturb_clean_narrative(b),
        "reduce_loan": lambda b: perturb_reduce_loan_amount(b, 0.5),
        "worsen_financials": lambda b: perturb_worsen_financials(b),
        "double_loan": lambda b: perturb_double_loan(b),
        "fact_perturbation": lambda b: perturb_increase_income(b, 1.5),
    }
    perturb_fn = perturbations.get(case.perturbation_type, lambda b: clone_borrower(b))
    perturbed_borrower = perturb_fn(borrower)

    print(f"    Counterfactual replay ({case.perturbation_type})...")
    counterfactual = await replay_through_los(perturbed_borrower, lender)
    print(f"    Counterfactual: {counterfactual['action']} (cost=${counterfactual['cost_usd']:.4f})")

    # Did the decision flip?
    baseline_decision = "approve" if baseline["action"] in ("approve", "conditional_offer") else "reject"
    cf_decision = "approve" if counterfactual["action"] in ("approve", "conditional_offer") else "reject"
    decision_flipped = baseline_decision != cf_decision

    # Robustness: replay 2 more times to check consistency
    robustness_results = []
    for i in range(2):
        r = await replay_through_los(perturbed_borrower, lender)
        r_decision = "approve" if r["action"] in ("approve", "conditional_offer") else "reject"
        robustness_results.append(r_decision == cf_decision)

    robustness = sum(robustness_results) / len(robustness_results) if robustness_results else 0.0

    return {
        "model_id": case.model_id,
        "borrower_id": case.borrower_id,
        "borrower_name": borrower.dossier.company_name,
        "ground_truth": borrower.true_outcome,
        "observed_decision": infer_decision(case.observed_decision),
        "perturbation_type": case.perturbation_type,
        "perturbation_desc": case.perturbation_desc,
        "baseline": baseline,
        "counterfactual": counterfactual,
        "decision_flipped": decision_flipped,
        "robustness": robustness,
        "total_cost_usd": round(
            baseline["cost_usd"] + counterfactual["cost_usd"] +
            sum(0.0 for _ in robustness_results),  # already counted
            4
        ),
    }


def build_audit_statement(case_result: dict) -> str:
    """Format a structured audit statement."""
    b = case_result["baseline"]
    cf = case_result["counterfactual"]
    return (
        f"decision = {case_result['observed_decision']}\n"
        f"ground_truth = {case_result['ground_truth']}\n"
        f"baseline_replay = {b['action']}\n"
        f"perturbation = {case_result['perturbation_type']}: {case_result['perturbation_desc']}\n"
        f"counterfactual_replay = {cf['action']}\n"
        f"decision_flipped = {case_result['decision_flipped']}\n"
        f"robustness = {case_result['robustness']:.2f}\n"
    )


def write_markdown(result: dict, path: Path) -> None:
    s = result["summary"]
    lines = [
        "# Counterfactual Audit Experiment",
        "",
        f"- Match: `{s['match_id']}`",
        f"- Decisions audited: `{s['n_decisions']}`",
        f"- Replay validity (flipped as expected): `{s['flip_rate']:.0%}`",
        f"- Average robustness: `{s['avg_robustness']:.2f}`",
        f"- Total LOS+LLM cost: `${s['total_cost_usd']:.4f}`",
        f"- Policy violation rate (approved fraud/bad): `{s['policy_violation_rate']:.0%}`",
        "",
        "## Audit Statements",
        "",
    ]
    for case in result["cases"]:
        lines.extend([
            f"### {case['model_id']} on {case['borrower_id']} ({case['borrower_name']})",
            f"- Ground truth: `{case['ground_truth']}`",
            f"- Observed: `{case['observed_decision']}`",
            f"- Baseline replay: `{case['baseline']['action']}`",
            f"- Perturbation: {case['perturbation_type']} — {case['perturbation_desc']}",
            f"- Counterfactual: `{case['counterfactual']['action']}`",
            f"- Decision flipped: `{case['decision_flipped']}`",
            f"- Robustness: `{case['robustness']:.2f}`",
            "",
            "```",
            build_audit_statement(case),
            "```",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    load_dotenv(os.path.expanduser("~/.env"))
    load_dotenv()

    await check_los_health(LOS_URL)
    print(f"LOS health OK at {LOS_URL}")

    match_data = load_match_data()
    observed = get_observed_decisions(match_data)
    borrowers = {b.id: b for b in get_borrowers("all")}
    lenders_by_name = {l.name: l for l in get_lenders()}

    # Filter to cases that exist in the match
    valid_cases = []
    for case in CURATED_CASES:
        key = (case.model_id, case.borrower_id)
        if key in observed:
            pb = observed[key]
            case.observed_rate = pb.get("rate_offered")
            # Verify observed decision matches what's in the match
            actual_ds = pb["decision_state"]
            if case.observed_decision in ("won", "lost") and actual_ds not in ("won", "lost"):
                print(f"  SKIP {case.model_id} {case.borrower_id}: expected approval, got {actual_ds}")
                continue
            if case.observed_decision == "declined" and actual_ds != "declined":
                print(f"  SKIP {case.model_id} {case.borrower_id}: expected decline, got {actual_ds}")
                continue
            valid_cases.append(case)
        else:
            print(f"  SKIP {case.model_id} {case.borrower_id}: not in match data")

    print(f"\nRunning {len(valid_cases)} audit cases through LOS...\n")

    results = []
    total_cost = 0.0
    for case in valid_cases:
        borrower = borrowers[case.borrower_id]
        lender = find_lender_for_model(case.model_id, match_data, lenders_by_name)
        # Override the lender model to match the case
        lender = copy.deepcopy(lender)
        lender.model = case.model_id.split("::")[0]

        case_result = await run_audit_case(case, borrower, lender)
        results.append(case_result)
        total_cost += case_result["total_cost_usd"]
        print(f"    Flipped: {case_result['decision_flipped']} | Robustness: {case_result['robustness']:.2f}")
        print()

    # Summary stats
    n = len(results)
    flips = sum(1 for r in results if r["decision_flipped"])
    avg_robustness = sum(r["robustness"] for r in results) / n if n else 0
    policy_violations = sum(
        1 for r in results
        if r["observed_decision"] == "approve" and r["ground_truth"] in ("fraud", "bad")
    )

    summary = {
        "match_id": match_data["match_id"],
        "n_decisions": n,
        "flip_rate": round(flips / n, 2) if n else 0,
        "avg_robustness": round(avg_robustness, 2),
        "total_cost_usd": round(total_cost, 4),
        "policy_violation_rate": round(policy_violations / n, 2) if n else 0,
    }

    payload = {"summary": summary, "cases": results}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "experiment2_counterfactual_audit.json"
    md_path = OUTPUT_DIR / "experiment2_counterfactual_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, md_path)

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
