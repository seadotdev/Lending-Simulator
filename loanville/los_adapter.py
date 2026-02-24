"""LOS API adapter for the Loanville simulation engine.

This adapter delegates underwriting decisions to LOS through:
  - POST /v1/entities
  - POST /v1/deals
  - POST /v1/deals/{dealId}/documents
  - POST /v1/deals/{dealId}/spread
  - POST /v1/deals/{dealId}/stage-transitions
  - POST /v1/deals/{dealId}/evaluate
  - GET  /v1/deals/{dealId}/audit
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Any

import httpx

from .contracts import apr_within_sanity_limit, normalize_apr_to_decimal
from .models import Borrower, LenderConfig, LenderDecision, TermSheet
from .run_schema import (
    DecisionRationale,
    DecisionTerms,
    RunCase,
    RunDecision,
    RunInputs,
    RunLabels,
    RunPolicy,
    RunTrace,
    TraceCost,
    TraceStep,
    UnderwritingRun,
)


DEFAULT_LOS_URL = "http://localhost:3000"


class LOSAdapterError(RuntimeError):
    """Raised when the LOS adapter cannot complete an evaluation."""


def _policy_payload(lender: LenderConfig) -> dict[str, Any]:
    return {
        "policy_id": f"p_{lender.id}_{lender.model.replace('/', '_')}",
        "model": lender.model,
        "persona": lender.persona,
        "target_yield_pct": lender.target_yield_pct,
        "max_single_loan": lender.max_single_loan,
        "total_capital": lender.total_capital,
        "sector_limits": lender.sector_limits,
    }


def _map_trace_steps(raw_steps: Any) -> list[TraceStep]:
    steps: list[TraceStep] = []
    if not isinstance(raw_steps, list):
        return steps

    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type", "note"))
        if step_type not in {"tool_call", "note", "reasoning", "doc_request"}:
            step_type = "note"
        steps.append(
            TraceStep(
                t=str(step.get("t", "")),
                type=step_type,
                name=str(step.get("name", "")),
                args=step.get("args", {}) if isinstance(step.get("args"), dict) else {},
                result=step.get("result", {}) if isinstance(step.get("result"), dict) else {},
                content=str(step.get("content", "")),
            )
        )
    return steps


def _to_float(raw: Any, default: float = 0.0) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    return default


def _to_int(raw: Any, default: int = 0) -> int:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return default


def _normalize_tenor_months(raw: Any, default: int = 24) -> int:
    tenor = _to_int(raw, default)
    if tenor < 1 or tenor > 360:
        return default
    return tenor


def _to_list_of_strings(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


async def evaluate_borrower_with_los(
    borrower: Borrower,
    lender: LenderConfig,
    base_url: str,
    timeout_s: float = 30.0,
    tenant_id: str = "loanville-sim",
    provider: str = "openrouter",
    mode: str = "rules_only",
) -> UnderwritingRun:
    """Evaluate one borrower by invoking LOS evaluate flow."""
    started = time.perf_counter()
    base = base_url.rstrip("/")
    actor = f"loanville:{lender.id}"
    policy = _policy_payload(lender)

    headers = {
        "Content-Type": "application/json",
        "X-Actor": actor,
        "X-Tenant-Id": tenant_id,
    }

    async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
        # Step 1: Create borrower entity.
        entity_resp = await client.post(
            f"{base}/v1/entities",
            json={
                "type": "company",
                "name": borrower.dossier.company_name,
                "identifiers": [{"scheme": "sim_borrower_id", "value": borrower.id}],
            },
        )
        entity_resp.raise_for_status()
        entity_json = entity_resp.json()
        entity_id = str(entity_json.get("id", ""))
        if not entity_id:
            raise LOSAdapterError("LOS create entity response did not include id")

        # Step 2: Create deal using LOS amount convention (minor units).
        requested_amount_minor = int(borrower.dossier.loan_request_amount * 100)
        deal_resp = await client.post(
            f"{base}/v1/deals",
            json={
                "borrower_name": borrower.dossier.company_name,
                "requested_amount": requested_amount_minor,
                "purpose": borrower.dossier.loan_purpose,
                "jurisdiction": "US",
                "primary_entity_id": entity_id,
                "custom_fields": {
                    "borrower_id": borrower.id,
                    "sim_borrower_id": borrower.id,
                    "lender_id": lender.id,
                    "sector": borrower.dossier.sector,
                    "years_in_business": borrower.dossier.years_in_business,
                    "employee_count": borrower.dossier.employee_count,
                    "annual_revenue": borrower.dossier.annual_revenue,
                },
            },
        )
        deal_resp.raise_for_status()
        deal_json = deal_resp.json()
        deal_id = str(deal_json.get("id", ""))
        if not deal_id:
            raise LOSAdapterError("LOS create deal response did not include id")

        # Step 3: Upload source documents (best-effort).
        try:
            bank_payload = [
                {
                    "month": s.month,
                    "opening_balance": s.opening_balance,
                    "ending_balance": s.ending_balance,
                    "total_deposits": s.total_deposits,
                    "total_withdrawals": s.total_withdrawals,
                }
                for s in borrower.dossier.bank_statements
            ]
            bank_json = json.dumps(bank_payload).encode("utf-8")
            await client.post(
                f"{base}/v1/deals/{deal_id}/documents",
                json={
                    "doc_type": "bank_statement",
                    "phase": "underwriting",
                    "filename": "bank_statements_12m.json",
                    "content_base64": base64.b64encode(bank_json).decode("utf-8"),
                },
            )

            pnl_payload = [
                {
                    "quarter": q.quarter,
                    "revenue": q.revenue,
                    "expenses": q.expenses,
                    "gross_profit": q.gross_profit,
                    "gross_margin_pct": q.gross_margin_pct,
                    "net_income": q.net_income,
                    "net_margin_pct": q.net_margin_pct,
                }
                for q in borrower.dossier.quarterly_income
            ]
            pnl_json = json.dumps(pnl_payload).encode("utf-8")
            await client.post(
                f"{base}/v1/deals/{deal_id}/documents",
                json={
                    "doc_type": "pnl",
                    "phase": "underwriting",
                    "filename": "quarterly_income.json",
                    "content_base64": base64.b64encode(pnl_json).decode("utf-8"),
                },
            )
        except Exception:
            pass

        # Step 4: Create spread from dossier aggregates.
        dossier = borrower.dossier
        spread_resp = await client.post(
            f"{base}/v1/deals/{deal_id}/spread",
            json={
                "entity_id": entity_id,
                "period": "TTM",
                "line_items": [
                    {
                        "category": "revenue",
                        "label": "Annual Revenue",
                        "amount": int(dossier.annual_revenue * 100),
                    },
                    {
                        "category": "expense",
                        "label": "Annual Expenses",
                        "amount": int(dossier.annual_expenses * 100),
                    },
                    {
                        "category": "income",
                        "label": "Net Income",
                        "amount": int(dossier.net_income * 100),
                    },
                ],
            },
        )
        spread_resp.raise_for_status()

        # Step 5: Progress deal stage (best-effort).
        for stage in ("origination", "underwriting"):
            try:
                await client.post(
                    f"{base}/v1/deals/{deal_id}/stage-transitions",
                    json={"target_stage": stage},
                )
            except Exception:
                pass

        # Step 6: Trigger LOS evaluation.
        eval_resp = await client.post(
            f"{base}/v1/deals/{deal_id}/evaluate",
            json={"policy": policy, "provider": provider, "mode": mode},
        )
        eval_resp.raise_for_status()
        eval_json = eval_resp.json()

        # Step 7: Optional audit pull for richer trace.
        audit_count = 0
        try:
            audit_resp = await client.get(f"{base}/v1/deals/{deal_id}/audit")
            if audit_resp.status_code == 200:
                audit_json = audit_resp.json()
                if isinstance(audit_json, dict) and isinstance(audit_json.get("events"), list):
                    audit_count = len(audit_json["events"])
        except Exception:
            audit_count = 0

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    run = UnderwritingRun()
    run.run_id = str(eval_json.get("run_id", uuid.uuid4()))
    run.timestamp_utc = str(eval_json.get("timestamp_utc", run.timestamp_utc))
    run.case = RunCase.from_borrower(borrower, source="los")
    run.policy = RunPolicy.from_lender(lender, policy_id=policy["policy_id"])
    run.inputs = RunInputs.from_dossier(borrower.dossier)

    decision_json = eval_json.get("decision", {}) if isinstance(eval_json, dict) else {}
    terms_json = decision_json.get("terms", {}) if isinstance(decision_json, dict) else {}
    rationale_json = (
        decision_json.get("rationale", {}) if isinstance(decision_json, dict) else {}
    )
    if not isinstance(terms_json, dict):
        terms_json = {}
    if not isinstance(rationale_json, dict):
        rationale_json = {}

    apr = normalize_apr_to_decimal(_to_float(terms_json.get("apr"), 0.0))
    if not apr_within_sanity_limit(apr):
        apr = normalize_apr_to_decimal(lender.target_yield_pct)

    action = str(decision_json.get("action", "pending")).lower()
    run.decision = RunDecision(
        action=action,
        risk_grade=str(decision_json.get("risk_grade", "")),
        prob_default_12m=_to_float(decision_json.get("prob_default_12m"), 0.0),
        terms=DecisionTerms(
            amount=_to_float(terms_json.get("amount"), 0.0),
            apr=apr,
            tenor_months=_normalize_tenor_months(terms_json.get("tenor_months"), 24),
            fees=terms_json.get("fees", {}) if isinstance(terms_json.get("fees"), dict) else {},
        ),
        conditions=_to_list_of_strings(decision_json.get("conditions")),
        covenants=_to_list_of_strings(decision_json.get("covenants")),
        rationale=DecisionRationale(
            summary=str(rationale_json.get("summary", "")),
            key_factors=_to_list_of_strings(rationale_json.get("key_factors")),
            what_would_change=_to_list_of_strings(rationale_json.get("what_would_change")),
        ),
        confidence=_to_float(decision_json.get("confidence"), 0.0),
    )

    trace_json = eval_json.get("trace", {}) if isinstance(eval_json, dict) else {}
    if not isinstance(trace_json, dict):
        trace_json = {}
    steps = _map_trace_steps(trace_json.get("steps"))
    if audit_count > 0:
        steps.append(
            TraceStep(
                t=run.timestamp_utc,
                type="note",
                content=f"Pulled {audit_count} LOS audit events for evaluation context",
            )
        )

    cost_json = trace_json.get("cost", {})
    if not isinstance(cost_json, dict):
        cost_json = {}
    run.trace = RunTrace(
        steps=steps,
        latency_ms=_to_int(trace_json.get("latency_ms"), elapsed_ms),
        cost=TraceCost(
            tokens_in=_to_int(cost_json.get("tokens_in"), 0),
            tokens_out=_to_int(cost_json.get("tokens_out"), 0),
            estimated_cost_usd=_to_float(cost_json.get("estimated_cost_usd"), 0.0),
        ),
    )

    if borrower.true_outcome:
        run.labels = RunLabels(
            available=True,
            gold={
                "true_outcome": borrower.true_outcome,
                "months_before_default": borrower.months_before_default,
                "correct_action": (
                    "decline" if borrower.true_outcome in ("bad", "fraud") else "approve"
                ),
            },
        )

    return run


def run_to_lender_decision(
    run: UnderwritingRun,
    lender: LenderConfig,
    borrower: Borrower,
) -> LenderDecision:
    approved = run.decision.action == "approve"
    term_sheet = None

    if approved and run.decision.terms.amount > 0:
        apr_decimal = normalize_apr_to_decimal(run.decision.terms.apr)
        if not apr_within_sanity_limit(apr_decimal):
            apr_decimal = normalize_apr_to_decimal(lender.target_yield_pct)
        term_sheet = TermSheet(
            loan_amount=run.decision.terms.amount,
            interest_rate=round(apr_decimal * 100, 4),
            term_months=_normalize_tenor_months(run.decision.terms.tenor_months, 24),
        )

    return LenderDecision(
        lender_id=lender.id,
        borrower_id=borrower.id,
        decision="APPROVE" if approved else "REJECT",
        reasoning=run.decision.rationale.summary,
        term_sheet=term_sheet,
    )


async def run_los_evaluations(
    lender: LenderConfig,
    borrowers: list[Borrower],
    base_url: str,
    max_concurrent: int = 5,
    timeout_s: float = 20.0,
    tenant_id: str = "loanville-sim",
    provider: str = "openrouter",
    mode: str = "rules_only",
) -> tuple[list[LenderDecision], list[UnderwritingRun]]:
    """Evaluate all borrowers for one lender via LOS."""
    semaphore = asyncio.Semaphore(max(1, max_concurrent))

    async def _evaluate_one(borrower: Borrower) -> tuple[LenderDecision, UnderwritingRun]:
        async with semaphore:
            run = await evaluate_borrower_with_los(
                borrower=borrower,
                lender=lender,
                base_url=base_url,
                timeout_s=timeout_s,
                tenant_id=tenant_id,
                provider=provider,
                mode=mode,
            )
            return run_to_lender_decision(run, lender, borrower), run

    results = await asyncio.gather(*[_evaluate_one(b) for b in borrowers])
    decisions = [decision for decision, _ in results]
    runs = [run for _, run in results]
    return decisions, runs
