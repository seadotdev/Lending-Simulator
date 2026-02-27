"""
CRM Simulation Runner

High-volume LOS simulation mode focused on:
  - speed / throughput
  - decision correctness against formulaic rules
  - responsiveness to incomplete applications
  - formal LOS compliance for offers
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import math
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .borrower_gen import generate_cohort
from .data import get_lenders
from .los_adapter import check_los_health, evaluate_all_via_los
from .models import Borrower, EconomicsConfig, LenderConfig, SeasonConfig
from .run_schema import UnderwritingRun

_OPEN_LOS_CRM_SCENARIOS = (
    Path(__file__).resolve().parent.parent
    / "open-los"
    / "packages"
    / "simulation"
    / "src"
    / "crm-test"
    / "scenarios.ts"
)


@dataclass
class CRMSimulationConfig:
    cases: int = 120
    mix: str = "realistic"
    seed: int = 42
    incomplete_ratio: float = 0.35
    max_concurrent_per_lender: int = 12
    apr_tolerance_pct: float = 1.5
    los_url: str = "http://localhost:3000"
    los_provider: str = "openrouter"
    los_model: str | None = None
    require_formal_offer_trace: bool = True

    def __post_init__(self) -> None:
        if self.cases <= 0:
            raise ValueError("cases must be > 0")
        if not (0.0 <= self.incomplete_ratio <= 1.0):
            raise ValueError("incomplete_ratio must be between 0.0 and 1.0")
        if self.max_concurrent_per_lender <= 0:
            raise ValueError("max_concurrent_per_lender must be > 0")
        if self.apr_tolerance_pct < 0:
            raise ValueError("apr_tolerance_pct must be >= 0")


@dataclass
class _CRMExpectation:
    borrower_id: str
    expected_action: str  # APPROVE | REJECT | PASS
    expected_apr_pct: float | None
    requires_follow_up: bool


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _formulaic_expectation(borrower: Borrower) -> tuple[str, float]:
    """Simple deterministic underwriting heuristic for correctness scoring."""
    d = borrower.dossier
    revenue = max(1.0, float(d.annual_revenue))
    margin = float(d.net_income) / revenue
    request_ratio = float(d.loan_request_amount) / revenue
    years = float(d.years_in_business)
    employees = float(d.employee_count)

    risk = 0.0
    risk += max(0.0, 0.14 - margin) * 5.0
    risk += max(0.0, request_ratio - 0.30) * 3.5
    risk += 0.20 if years < 3 else 0.0
    risk += 0.15 if employees < 10 else 0.0

    action = "REJECT" if risk >= 1.2 else "APPROVE"
    apr = _clamp(8.0 + risk * 5.5 + request_ratio * 4.0, 7.5, 24.0)
    return action, apr


def _build_crm_cases(config: CRMSimulationConfig) -> tuple[list[Borrower], dict[str, _CRMExpectation]]:
    season_cfg = SeasonConfig(
        weeks=4,
        cohort_size=config.cases,
        months_per_week=2,
        season_mix=config.mix,
        seed=config.seed,
        speed_scoring=False,
        custom_tools=False,
        economics=EconomicsConfig(),
        borrower_patience_weeks=1,
        offer_validity_weeks=1,
        capital_adequacy_ratio=0.0,
        capital_decay_rate=0.0,
    )
    base = generate_cohort(week=3, config=season_cfg, used_static_ids=set())
    rng = random.Random(config.seed)

    out: list[Borrower] = []
    expectations: dict[str, _CRMExpectation] = {}

    for borrower in base:
        b = copy.deepcopy(borrower)
        needs_follow_up = rng.random() < config.incomplete_ratio
        if needs_follow_up:
            if len(b.dossier.bank_statements) > 3:
                b.dossier.bank_statements = b.dossier.bank_statements[-3:]
            if len(b.dossier.quarterly_income) > 2:
                b.dossier.quarterly_income = b.dossier.quarterly_income[-2:]
            b.dossier.narrative = (
                f"{b.dossier.narrative} "
                "[CRM_CONTEXT] Prior statements unavailable at intake; request additional docs."
            ).strip()
            expected_action = "PASS"
            expected_apr = None
        else:
            expected_action, expected_apr = _formulaic_expectation(b)

        out.append(b)
        expectations[b.id] = _CRMExpectation(
            borrower_id=b.id,
            expected_action=expected_action,
            expected_apr_pct=expected_apr,
            requires_follow_up=needs_follow_up,
        )

    return out, expectations


def _has_tool_call(run: UnderwritingRun) -> bool:
    return any((s.type or "").lower() == "tool_call" for s in run.trace.steps)


def _has_doc_request(run: UnderwritingRun) -> bool:
    for step in run.trace.steps:
        stype = (step.type or "").lower()
        name = (step.name or "").lower()
        if stype == "doc_request":
            return True
        if "doc" in name and ("request" in name or "missing" in name):
            return True
    return False


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil((p / 100.0) * len(ordered))) - 1
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _summarize_lender(
    lender: LenderConfig,
    decisions,
    runs: list[UnderwritingRun],
    expectations: dict[str, _CRMExpectation],
    elapsed_s: float,
    apr_tolerance_pct: float,
) -> dict:
    run_map = {r.case.case_id: r for r in runs}
    latencies = [float(max(0, int(r.trace.latency_ms or 0))) for r in runs]
    tool_calls_per_case = [
        sum(1 for s in r.trace.steps if (s.type or "").lower() == "tool_call")
        for r in runs
    ]

    correct = 0
    pricing_hits = 0
    pricing_targets = 0
    follow_hits = 0
    follow_targets = 0
    formal_offer_hits = 0
    approvals = 0

    for decision in decisions:
        exp = expectations.get(decision.borrower_id)
        if exp is None:
            continue
        run = run_map.get(decision.borrower_id)

        if decision.decision == exp.expected_action:
            correct += 1

        if exp.expected_action == "APPROVE":
            pricing_targets += 1
            if (
                decision.decision == "APPROVE"
                and decision.term_sheet is not None
                and exp.expected_apr_pct is not None
                and abs(float(decision.term_sheet.interest_rate) - exp.expected_apr_pct)
                <= apr_tolerance_pct
            ):
                pricing_hits += 1

        if exp.requires_follow_up:
            follow_targets += 1
            if run and _has_doc_request(run):
                follow_hits += 1

        if decision.decision == "APPROVE":
            approvals += 1
            if run and _has_tool_call(run):
                formal_offer_hits += 1

    throughput = 0.0
    if elapsed_s > 0:
        throughput = round(len(decisions) / (elapsed_s / 60.0), 2)

    return {
        "lender_id": lender.id,
        "lender_name": lender.name,
        "model": lender.model,
        "evaluations": len(decisions),
        "throughput_eval_per_min": throughput,
        "decision_accuracy_pct": _pct(correct, len(decisions)),
        "pricing_accuracy_pct": _pct(pricing_hits, pricing_targets),
        "follow_up_responsiveness_pct": _pct(follow_hits, follow_targets),
        "formal_offer_compliance_pct": _pct(formal_offer_hits, approvals),
        "approvals": approvals,
        "avg_latency_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 95.0), 1),
        "avg_tool_calls": round(statistics.fmean(tool_calls_per_case), 2)
        if tool_calls_per_case
        else 0.0,
    }


def _load_open_los_crm_catalog() -> dict:
    path = _OPEN_LOS_CRM_SCENARIOS
    if not path.exists():
        return {"available": False, "path": str(path)}

    text = path.read_text(encoding="utf-8")
    ids = re.findall(r'\bid:\s*"([^"]+)"', text)
    categories = re.findall(r'\bcategory:\s*"([^"]+)"', text)
    category_counts = dict(sorted(Counter(categories).items()))
    return {
        "available": True,
        "path": str(path),
        "task_count": len(ids),
        "category_count": len(category_counts),
        "categories": category_counts,
    }


def _aggregate_report(per_lender: list[dict], elapsed_s: float) -> dict:
    if not per_lender:
        return {}

    total_evals = sum(int(m["evaluations"]) for m in per_lender)

    def wavg(key: str) -> float:
        if total_evals <= 0:
            return 0.0
        return round(
            sum(float(m[key]) * int(m["evaluations"]) for m in per_lender) / total_evals,
            2,
        )

    throughput = 0.0
    if elapsed_s > 0:
        throughput = round(total_evals / (elapsed_s / 60.0), 2)

    return {
        "evaluations": total_evals,
        "throughput_eval_per_min": throughput,
        "decision_accuracy_pct": wavg("decision_accuracy_pct"),
        "pricing_accuracy_pct": wavg("pricing_accuracy_pct"),
        "follow_up_responsiveness_pct": wavg("follow_up_responsiveness_pct"),
        "formal_offer_compliance_pct": wavg("formal_offer_compliance_pct"),
        "avg_latency_ms": wavg("avg_latency_ms"),
        "p95_latency_ms": round(
            max(float(m.get("p95_latency_ms", 0.0)) for m in per_lender), 1
        ),
    }


async def run_crm_simulation(
    config: CRMSimulationConfig,
    lenders: list[LenderConfig] | None = None,
) -> dict:
    lenders = lenders or get_lenders()
    await check_los_health(config.los_url)

    borrowers, expectations = _build_crm_cases(config)
    start = time.perf_counter()

    tasks = [
        evaluate_all_via_los(
            lender=lender,
            borrowers=borrowers,
            los_url=config.los_url,
            max_concurrent=config.max_concurrent_per_lender,
            provider=config.los_provider,
            mode="full",
            underwrite_only=False,
            los_model=config.los_model,
            require_formal_offer_trace=config.require_formal_offer_trace,
        )
        for lender in lenders
    ]
    results = await asyncio.gather(*tasks)
    elapsed_s = time.perf_counter() - start

    per_lender = [
        _summarize_lender(
            lender=lender,
            decisions=decisions,
            runs=runs,
            expectations=expectations,
            elapsed_s=elapsed_s,
            apr_tolerance_pct=config.apr_tolerance_pct,
        )
        for lender, (decisions, runs) in zip(lenders, results)
    ]

    expected_counts = Counter(exp.expected_action for exp in expectations.values())
    follow_up_cases = sum(1 for exp in expectations.values() if exp.requires_follow_up)

    return {
        "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": asdict(config),
        "open_los_crm_catalog": _load_open_los_crm_catalog(),
        "inputs": {
            "lenders": [
                {"id": l.id, "name": l.name, "model": l.model}
                for l in lenders
            ],
            "cases": len(borrowers),
            "expected_actions": dict(sorted(expected_counts.items())),
            "follow_up_cases": follow_up_cases,
        },
        "elapsed_seconds": round(elapsed_s, 3),
        "per_lender": per_lender,
        "aggregate": _aggregate_report(per_lender, elapsed_s),
    }


def print_crm_simulation_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("CRM SIMULATION REPORT (LOS HIGH-VOLUME)")
    print("=" * 70)

    catalog = report.get("open_los_crm_catalog", {})
    if catalog.get("available"):
        print(
            "Open LOS CRM corpus: "
            f"{catalog.get('task_count', 0)} tasks / "
            f"{catalog.get('category_count', 0)} categories"
        )
    else:
        print("Open LOS CRM corpus: not found in this workspace")

    inputs = report.get("inputs", {})
    agg = report.get("aggregate", {})
    print(
        f"Cases: {inputs.get('cases', 0)} | "
        f"Follow-up cases: {inputs.get('follow_up_cases', 0)} | "
        f"Elapsed: {report.get('elapsed_seconds', 0)}s"
    )
    print(
        "Aggregate: "
        f"throughput={agg.get('throughput_eval_per_min', 0)} eval/min, "
        f"decision_acc={agg.get('decision_accuracy_pct', 0)}%, "
        f"pricing_acc={agg.get('pricing_accuracy_pct', 0)}%, "
        f"follow_up={agg.get('follow_up_responsiveness_pct', 0)}%, "
        f"formal_offer={agg.get('formal_offer_compliance_pct', 0)}%"
    )

    print("\nPer-lender:")
    for row in report.get("per_lender", []):
        print(
            f"  {row['lender_name']} [{row['model']}]\n"
            f"    evals={row['evaluations']} | "
            f"throughput={row['throughput_eval_per_min']} eval/min | "
            f"latency(avg/p95)={row['avg_latency_ms']}ms/{row['p95_latency_ms']}ms\n"
            f"    decision_acc={row['decision_accuracy_pct']}% | "
            f"pricing_acc={row['pricing_accuracy_pct']}% | "
            f"follow_up={row['follow_up_responsiveness_pct']}% | "
            f"formal_offer={row['formal_offer_compliance_pct']}%"
        )

    print("=" * 70 + "\n")
