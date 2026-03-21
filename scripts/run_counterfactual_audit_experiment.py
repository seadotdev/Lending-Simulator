from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_preflight import preflight
from loanville.data import get_borrowers, get_lenders
from loanville.los_adapter import evaluate_via_los
from loanville.models import Borrower, ExistingLoan, LenderConfig


OUTPUT_DIR = Path("artifacts/pre_submission_experiments")
DEFAULT_LOS_URL = os.getenv("LOS_URL", "http://localhost:3200")
MODELS: list[tuple[str, str]] = [
    ("google/gemini-2.5-flash", "LND-002"),
    ("deepseek/deepseek-chat-v3-0324", "LND-003"),
    ("meta-llama/llama-3.3-70b-instruct", "LND-005"),
]
BORROWER_IDS = ["BRW-001", "BRW-014", "BRW-025", "BRW-011"]
MAX_CASES = 5


@dataclass
class Intervention:
    kind: str
    summary: str
    factor: float | None = None
    from_value: float | None = None
    to_value: float | None = None
    sector: str | None = None
    extra: dict[str, Any] | None = None


def load_env() -> None:
    load_dotenv(os.path.expanduser("~/.env"))
    load_dotenv()


def set_sector_exposure(lender: LenderConfig, sector: str, target_exposure: float) -> None:
    others = [loan for loan in lender.existing_portfolio if loan.sector != sector]
    remaining = max(0.0, target_exposure)
    sector_loans = [copy.deepcopy(loan) for loan in lender.existing_portfolio if loan.sector == sector]
    for loan in sector_loans:
        if remaining <= 0:
            loan.remaining_balance = 0.0
        else:
            new_balance = min(loan.remaining_balance, remaining)
            remaining -= new_balance
            loan.remaining_balance = new_balance
    sector_loans = [loan for loan in sector_loans if loan.remaining_balance > 0]
    if remaining > 0:
        sector_loans.append(
            ExistingLoan(
                borrower_name=f"Synthetic {sector} exposure",
                sector=sector,
                original_amount=remaining,
                remaining_balance=remaining,
                interest_rate=9.0,
                months_remaining=18,
            )
        )
    lender.existing_portfolio = others + sector_loans


def sector_exposure(lender: LenderConfig, sector: str) -> float:
    return sum(loan.remaining_balance for loan in lender.existing_portfolio if loan.sector == sector)


def clone_subjects(borrower: Borrower, lender: LenderConfig) -> tuple[Borrower, LenderConfig]:
    return copy.deepcopy(borrower), copy.deepcopy(lender)


def make_unique_lender(lender: LenderConfig, tag: str) -> LenderConfig:
    lender = copy.deepcopy(lender)
    lender.id = f"{lender.id}-{tag}"
    return lender


async def run_full_mode(
    borrower: Borrower,
    lender: LenderConfig,
    *,
    los_url: str,
    provider: str,
    tag: str,
) -> dict[str, Any]:
    run = await evaluate_via_los(
        borrower,
        make_unique_lender(lender, tag),
        los_url=los_url,
        provider=provider,
        mode="full",
        timeout=180.0,
        underwrite_only=False,
        los_model=lender.model,
    )
    return {
        "run_id": run.run_id,
        "action": run.decision.action,
        "summary": run.decision.rationale.summary,
        "apr": run.decision.terms.apr,
        "amount": run.decision.terms.amount,
        "tenor_months": run.decision.terms.tenor_months,
        "trace_steps": len(run.trace.steps),
        "trace_types": [step.type for step in run.trace.steps],
        "estimated_cost_usd": run.trace.cost.estimated_cost_usd,
        "tokens_in": run.trace.cost.tokens_in,
        "tokens_out": run.trace.cost.tokens_out,
    }


def candidate_interventions(borrower: Borrower, lender: LenderConfig, baseline_action: str) -> list[Intervention]:
    current_sector = sector_exposure(lender, borrower.dossier.sector)
    sector_limit = lender.sector_limits.get(borrower.dossier.sector, 0.25)
    sector_cap = lender.total_capital * sector_limit
    interventions: list[Intervention] = []

    if baseline_action == "approve":
        interventions.extend(
            [
                Intervention(
                    kind="increase_sector_exposure_to_limit",
                    summary="increase same-sector portfolio exposure until the new loan breaches concentration policy",
                    sector=borrower.dossier.sector,
                    from_value=current_sector,
                    to_value=sector_cap,
                ),
                Intervention(
                    kind="remove_bank_statements",
                    summary="remove verified bank statement evidence from the dossier",
                    from_value=float(len(borrower.dossier.bank_statements)),
                    to_value=0.0,
                ),
                Intervention(
                    kind="increase_loan_amount",
                    summary="increase requested amount by 60%",
                    factor=1.6,
                    from_value=borrower.dossier.loan_request_amount,
                    to_value=round(borrower.dossier.loan_request_amount * 1.6, 2),
                ),
            ]
        )
    else:
        interventions.extend(
            [
                Intervention(
                    kind="clear_sector_exposure",
                    summary="clear same-sector portfolio exposure before evaluation",
                    sector=borrower.dossier.sector,
                    from_value=current_sector,
                    to_value=0.0,
                ),
                Intervention(
                    kind="reduce_loan_amount",
                    summary="reduce requested amount by 40%",
                    factor=0.6,
                    from_value=borrower.dossier.loan_request_amount,
                    to_value=round(borrower.dossier.loan_request_amount * 0.6, 2),
                ),
                Intervention(
                    kind="boost_net_income",
                    summary="increase verified net income by 35%",
                    factor=1.35,
                    from_value=borrower.dossier.net_income,
                    to_value=round(borrower.dossier.net_income * 1.35, 2),
                ),
            ]
        )

    return interventions


def apply_intervention(borrower: Borrower, lender: LenderConfig, intervention: Intervention) -> tuple[Borrower, LenderConfig]:
    b2, l2 = clone_subjects(borrower, lender)
    if intervention.kind == "increase_sector_exposure_to_limit":
        set_sector_exposure(l2, intervention.sector or b2.dossier.sector, float(intervention.to_value or 0.0))
    elif intervention.kind == "clear_sector_exposure":
        set_sector_exposure(l2, intervention.sector or b2.dossier.sector, 0.0)
    elif intervention.kind == "remove_bank_statements":
        b2.dossier.bank_statements = []
    elif intervention.kind == "increase_loan_amount":
        b2.dossier.loan_request_amount = round(b2.dossier.loan_request_amount * float(intervention.factor or 1.0), 2)
    elif intervention.kind == "reduce_loan_amount":
        b2.dossier.loan_request_amount = round(b2.dossier.loan_request_amount * float(intervention.factor or 1.0), 2)
    elif intervention.kind == "boost_net_income":
        b2.dossier.net_income = round(b2.dossier.net_income * float(intervention.factor or 1.0), 2)
        scale = float(intervention.factor or 1.0)
        for quarter in b2.dossier.quarterly_income:
            quarter.net_income = round(quarter.net_income * scale, 2)
    return b2, l2


def apply_smaller_variant(
    borrower: Borrower,
    lender: LenderConfig,
    intervention: Intervention,
) -> tuple[Borrower, LenderConfig] | None:
    b2, l2 = clone_subjects(borrower, lender)
    if intervention.kind == "increase_sector_exposure_to_limit":
        target = float(intervention.from_value or 0.0) + (float(intervention.to_value or 0.0) - float(intervention.from_value or 0.0)) * 0.5
        set_sector_exposure(l2, intervention.sector or b2.dossier.sector, target)
        return b2, l2
    if intervention.kind == "clear_sector_exposure":
        target = float(intervention.from_value or 0.0) * 0.5
        set_sector_exposure(l2, intervention.sector or b2.dossier.sector, target)
        return b2, l2
    if intervention.kind == "remove_bank_statements":
        keep = max(1, len(b2.dossier.bank_statements) // 2)
        b2.dossier.bank_statements = b2.dossier.bank_statements[-keep:]
        return b2, l2
    if intervention.kind == "increase_loan_amount":
        b2.dossier.loan_request_amount = round(float(intervention.from_value or b2.dossier.loan_request_amount) * 1.3, 2)
        return b2, l2
    if intervention.kind == "reduce_loan_amount":
        b2.dossier.loan_request_amount = round(float(intervention.from_value or b2.dossier.loan_request_amount) * 0.8, 2)
        return b2, l2
    if intervention.kind == "boost_net_income":
        scale = 1.15
        b2.dossier.net_income = round(b2.dossier.net_income * scale, 2)
        for quarter in b2.dossier.quarterly_income:
            quarter.net_income = round(quarter.net_income * scale, 2)
        return b2, l2
    return None


async def robustness_score(
    borrower: Borrower,
    lender: LenderConfig,
    intervention: Intervention,
    expected_action: str,
    *,
    los_url: str,
    provider: str,
    tag_prefix: str,
    repeats: int = 3,
) -> float:
    matches = 0
    for idx in range(repeats):
        b2, l2 = apply_intervention(borrower, lender, intervention)
        result = await run_full_mode(
            b2,
            l2,
            los_url=los_url,
            provider=provider,
            tag=f"{tag_prefix}-robust-{idx}",
        )
        if result["action"] == expected_action:
            matches += 1
    return round(matches / repeats, 2)


async def find_counterfactual_cases(*, los_url: str, provider: str) -> dict[str, Any]:
    borrowers = {b.id: b for b in get_borrowers("all")}
    lenders = {l.id: l for l in get_lenders()}
    cases: list[dict[str, Any]] = []
    total_cost = 0.0

    for model, lender_id in MODELS:
        lender_template = copy.deepcopy(lenders[lender_id])
        lender_template.model = model

        for borrower_id in BORROWER_IDS:
            borrower = copy.deepcopy(borrowers[borrower_id])
            lender = copy.deepcopy(lender_template)
            base_tag = f"{lender_id}-{model.split('/')[-1]}-{borrower_id}"
            baseline = await run_full_mode(
                borrower,
                lender,
                los_url=los_url,
                provider=provider,
                tag=f"{base_tag}-baseline",
            )
            total_cost += float(baseline["estimated_cost_usd"])
            baseline_action = baseline["action"]
            expected_action = "decline" if baseline_action == "approve" else "approve"

            for intervention in candidate_interventions(borrower, lender, baseline_action):
                b2, l2 = apply_intervention(borrower, lender, intervention)
                counterfactual = await run_full_mode(
                    b2,
                    l2,
                    los_url=los_url,
                    provider=provider,
                    tag=f"{base_tag}-{intervention.kind}",
                )
                total_cost += float(counterfactual["estimated_cost_usd"])
                if counterfactual["action"] != expected_action:
                    continue

                smaller = apply_smaller_variant(borrower, lender, intervention)
                minimality = None
                smaller_result = None
                if smaller is not None:
                    smaller_result = await run_full_mode(
                        smaller[0],
                        smaller[1],
                        los_url=los_url,
                        provider=provider,
                        tag=f"{base_tag}-{intervention.kind}-smaller",
                    )
                    total_cost += float(smaller_result["estimated_cost_usd"])
                    minimality = smaller_result["action"] == baseline_action

                robustness = await robustness_score(
                    borrower,
                    lender,
                    intervention,
                    expected_action,
                    los_url=los_url,
                    provider=provider,
                    tag_prefix=base_tag,
                )

                case = {
                    "model": model,
                    "lender_policy": lender.name,
                    "borrower_id": borrower.id,
                    "borrower_name": borrower.dossier.company_name,
                    "ground_truth": borrower.true_outcome,
                    "baseline": baseline,
                    "intervention": asdict(intervention),
                    "counterfactual": counterfactual,
                    "smaller_probe": smaller_result,
                    "minimality": minimality,
                    "robustness": robustness,
                    "replay_valid": counterfactual["action"] == expected_action,
                }
                cases.append(case)
                break

            if len(cases) >= MAX_CASES:
                break
        if len(cases) >= MAX_CASES:
            break

    replay_validity = round(sum(1 for c in cases if c["replay_valid"]) / len(cases), 2) if cases else 0.0
    minimality_rate = round(
        sum(1 for c in cases if c["minimality"] is True) / max(1, sum(1 for c in cases if c["minimality"] is not None)),
        2,
    )
    avg_robustness = round(sum(c["robustness"] for c in cases) / len(cases), 2) if cases else 0.0

    return {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "los_url": los_url,
        "mode": "full",
        "provider": provider,
        "cases": cases,
        "summary": {
            "n_decisions": len(cases),
            "replay_validity": replay_validity,
            "minimality_rate": minimality_rate,
            "average_robustness": avg_robustness,
            "estimated_cost_usd": round(total_cost, 4),
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Counterfactual Audit Experiment",
        "",
        "- Underwriting path: `Open LOS /v1/deals/:dealId/evaluate` in `mode=full`",
        f"- LOS URL: `{payload['los_url']}`",
        f"- Decisions audited: `{summary['n_decisions']}`",
        f"- SCM replay validity: `{summary['replay_validity']:.0%}`",
        f"- Minimality rate: `{summary['minimality_rate']:.0%}`",
        f"- Average robustness: `{summary['average_robustness']}`",
        f"- Estimated OpenRouter cost: `${summary['estimated_cost_usd']:.4f}`",
        "",
    ]
    for case in payload["cases"]:
        baseline = case["baseline"]
        intervention = case["intervention"]
        counter = case["counterfactual"]
        lines.extend(
            [
                f"## {case['model']} :: {case['borrower_id']} :: {case['borrower_name']}",
                f"- Lender policy: `{case['lender_policy']}`",
                f"- Ground truth: `{case['ground_truth']}`",
                f"- Baseline decision: `{baseline['action']}`",
                f"- Baseline summary: {baseline['summary']}",
                f"- Intervention: `{intervention['kind']}` — {intervention['summary']}",
                f"- Counterfactual decision: `{counter['action']}`",
                f"- Counterfactual summary: {counter['summary']}",
                f"- Replay valid: `{case['replay_valid']}`",
                f"- Minimality: `{case['minimality']}`",
                f"- Robustness: `{case['robustness']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    load_env()
    admin_key = os.getenv("OR_ADMIN_KEY", "")
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    if admin_key and or_key:
        preflight(
            admin_key=admin_key,
            or_key=or_key,
            models=[model for model, _ in MODELS],
            budget_per_model=1.0,
            checks=["balance", "math", "models"],
            site_name="loanville-counterfactual-audit-full",
        )

    payload = await find_counterfactual_cases(los_url=DEFAULT_LOS_URL, provider="openrouter")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "experiment2_counterfactual_audit.json"
    md_path = OUTPUT_DIR / "experiment2_counterfactual_audit.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
