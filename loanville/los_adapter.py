"""
LOS Adapter — calls the Open LOS REST API to evaluate borrowers.

The adapter translates SIM Borrower + LenderConfig into LOS API calls
and maps the UnderwritingRun response back into the SIM's data model.
This is the default evaluation path for all non-mock simulations.

Two modes:
  (default)              Full LOS pipeline (entity/deal/docs/spread/stages/evaluate)
  --underwrite-only      Just POST /v1/underwrite with dossier inline (no LOS ceremony)

APR convention (per los-cutover.md section 5):
  UnderwritingRun.DecisionTerms.apr uses DECIMAL (0.095 = 9.5%)
  TermSheet.interest_rate uses PERCENTAGE (9.5 = 9.5%)
  The adapter converts: term_sheet.interest_rate = run.decision.terms.apr * 100
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

from .models import (
    Borrower,
    LenderConfig,
    LenderDecision,
    TermSheet,
)
from .run_schema import (
    DecisionRationale,
    DecisionTerms,
    RunCase,
    RunDecision,
    RunInputs,
    RunPolicy,
    RunTrace,
    TraceCost,
    TraceStep,
    UnderwritingRun,
    build_run,
)


DEFAULT_LOS_URL = "http://localhost:3000"


async def check_los_health(los_url: str = DEFAULT_LOS_URL, timeout: float = 5.0) -> None:
    """Verify the LOS is reachable before starting a run. Raises on failure."""
    base = los_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base}/health")
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise RuntimeError(
            f"Cannot connect to Open LOS at {los_url}. "
            f"Start it with: cd open-los/packages/api && npm run start"
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Open LOS at {los_url} returned {exc.response.status_code} on health check."
        )


def serialize_dossier(borrower: Borrower) -> dict:
    """Serialize a Borrower's dossier into the LOS FinancialDossier format."""
    d = borrower.dossier
    dossier: dict = {
        "company_name": d.company_name,
        "sector": d.sector,
        "years_in_business": d.years_in_business,
        "employee_count": d.employee_count,
        "narrative": d.narrative,
        "loan_request_amount": d.loan_request_amount,
        "loan_purpose": d.loan_purpose,
        "annual_revenue": d.annual_revenue,
        "annual_expenses": d.annual_expenses,
        "net_income": d.net_income,
    }

    # Quarterly income
    if d.quarterly_income:
        dossier["quarterly_income"] = [{
            "quarter": q.quarter,
            "revenue": q.revenue,
            "expenses": q.expenses,
            "gross_profit": getattr(q, "gross_profit", None),
            "gross_margin_pct": getattr(q, "gross_margin_pct", None),
            "net_income": q.net_income,
            "net_margin_pct": q.net_margin_pct,
        } for q in d.quarterly_income]

    # Bank statements (full detail with individual transactions)
    if d.bank_statements:
        dossier["bank_statements"] = [{
            "month": s.month,
            "opening_balance": s.opening_balance,
            "ending_balance": s.ending_balance,
            "total_deposits": s.total_deposits,
            "total_withdrawals": s.total_withdrawals,
            "deposits": [
                {"date": t.date, "description": t.description, "amount": t.amount}
                for t in s.deposits
            ],
            "withdrawals": [
                {"date": t.date, "description": t.description, "amount": t.amount}
                for t in s.withdrawals
            ],
        } for s in d.bank_statements]

    return dossier


def serialize_policy(lender: LenderConfig) -> dict:
    """Serialize a LenderConfig into the LOS UnderwritePolicy format."""
    policy: dict = {
        "policy_id": f"p_{lender.id}_{lender.model.replace('/', '_')}",
        "model": lender.model,
        "persona": lender.persona,
        "target_yield_pct": lender.target_yield_pct,
        "max_single_loan": lender.max_single_loan,
        "total_capital": lender.total_capital,
        "sector_limits": lender.sector_limits,
    }

    # Include existing portfolio for portfolio-fit analysis
    if lender.existing_portfolio:
        policy["existing_portfolio"] = [{
            "borrower_name": loan.borrower_name,
            "sector": loan.sector,
            "remaining_balance": loan.remaining_balance,
            "interest_rate": loan.interest_rate,
        } for loan in lender.existing_portfolio]

    return policy


def build_model_config(lender: LenderConfig, cli_override: str | None = None) -> dict:
    """Build model configuration from lender + CLI overrides."""
    return {
        "default": cli_override or lender.model,
    }


async def evaluate_standalone(
    borrower: Borrower,
    lender: LenderConfig,
    los_url: str = DEFAULT_LOS_URL,
    provider: str = "openrouter",
    timeout: float = 120.0,
    los_model: str | None = None,
) -> UnderwritingRun:
    """Evaluate a borrower via the standalone /v1/underwrite endpoint.

    This is the "underwrite only" path — no entity/deal/docs/spread/stage
    ceremony. Just dossier + policy → LLM → decision.
    """
    start_time = time.time()
    tenant_id = f"lender_{lender.id}"
    actor = f"sim:{lender.id}"
    base = los_url.rstrip("/")

    headers = {
        "Content-Type": "application/json",
        "X-Actor": actor,
        "X-Tenant-Id": tenant_id,
    }

    dossier = serialize_dossier(borrower)
    policy = serialize_policy(lender)
    models = build_model_config(lender, los_model)

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        eval_resp = await client.post(f"{base}/v1/underwrite", json={
            "dossier": dossier,
            "policy": policy,
            "provider": provider,
            "models": models,
        })
        eval_resp.raise_for_status()
        eval_result = eval_resp.json()

    return _map_los_response(eval_result, borrower, lender, start_time)


async def evaluate_via_los(
    borrower: Borrower,
    lender: LenderConfig,
    los_url: str = DEFAULT_LOS_URL,
    provider: str = "openrouter",
    mode: str = "rules_only",
    timeout: float = 60.0,
    underwrite_only: bool = False,
    los_model: str | None = None,
) -> UnderwritingRun:
    """Evaluate a borrower through the Open LOS REST API.

    When underwrite_only=True, skips the LOS pipeline and calls
    /v1/underwrite directly with the full dossier.

    Returns an UnderwritingRun artifact with the LOS decision, trace, and inputs.
    """
    if underwrite_only:
        return await evaluate_standalone(
            borrower, lender, los_url,
            provider=provider, timeout=timeout,
            los_model=los_model,
        )

    start_time = time.time()
    tenant_id = f"lender_{lender.id}"
    actor = f"sim:{lender.id}"
    base = los_url.rstrip("/")

    headers = {
        "Content-Type": "application/json",
        "X-Actor": actor,
        "X-Tenant-Id": tenant_id,
    }

    # Serialize full dossier + policy for the evaluate call
    dossier = serialize_dossier(borrower)
    policy_dict = serialize_policy(lender)
    models = build_model_config(lender, los_model)

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        # Step 1: Create entity
        entity_resp = await client.post(f"{base}/v1/entities", json={
            "type": "company",
            "name": borrower.dossier.company_name,
            "identifiers": [
                {"scheme": "sim_borrower_id", "value": borrower.id},
            ],
        })
        entity_resp.raise_for_status()
        entity = entity_resp.json()
        entity_id = entity["id"]

        # Step 2: Create deal
        # Amounts in minor units (cents) as per LOS convention
        requested_amount_minor = int(borrower.dossier.loan_request_amount * 100)
        deal_resp = await client.post(f"{base}/v1/deals", json={
            "borrower_name": borrower.dossier.company_name,
            "requested_amount": requested_amount_minor,
            "purpose": borrower.dossier.loan_purpose,
            "jurisdiction": "US",
            "primary_entity_id": entity_id,
            "custom_fields": {
                "sim_borrower_id": borrower.id,
                "sector": borrower.dossier.sector,
                "years_in_business": borrower.dossier.years_in_business,
                "employee_count": borrower.dossier.employee_count,
                "annual_revenue": borrower.dossier.annual_revenue,
            },
        })
        deal_resp.raise_for_status()
        deal = deal_resp.json()
        deal_id = deal["id"]

        # Step 3: Upload documents (as JSON, base64-encoded)
        # Document uploads are non-fatal — evaluate works without them
        import base64 as b64mod
        try:
            # 3a: Bank statements
            bank_stmts_json = json.dumps([{
                "month": s.month,
                "opening_balance": s.opening_balance,
                "ending_balance": s.ending_balance,
                "total_deposits": s.total_deposits,
                "total_withdrawals": s.total_withdrawals,
            } for s in borrower.dossier.bank_statements])

            await client.post(f"{base}/v1/deals/{deal_id}/documents", json={
                "doc_type": "bank_statement",
                "phase": "underwriting",
                "filename": "bank_statements_12m.json",
                "content_base64": b64mod.b64encode(bank_stmts_json.encode()).decode(),
            })

            # 3b: Quarterly income
            quarterly_json = json.dumps([{
                "quarter": q.quarter,
                "revenue": q.revenue,
                "expenses": q.expenses,
                "gross_profit": q.gross_profit,
                "gross_margin_pct": q.gross_margin_pct,
                "net_income": q.net_income,
                "net_margin_pct": q.net_margin_pct,
            } for q in borrower.dossier.quarterly_income])

            await client.post(f"{base}/v1/deals/{deal_id}/documents", json={
                "doc_type": "pnl",
                "phase": "underwriting",
                "filename": "quarterly_income.json",
                "content_base64": b64mod.b64encode(quarterly_json.encode()).decode(),
            })
        except Exception as exc:
            logger.warning("Document upload failed for deal %s: %s", deal_id, exc)

        # Step 4: Create spread from dossier
        d = borrower.dossier
        line_items = [
            {"category": "revenue", "label": "Annual Revenue", "amount": int(d.annual_revenue * 100)},
            # LOS ratio logic expects expense buckets like cogs / operating_expense.
            # We only have aggregate annual expenses here, so map to cogs.
            {"category": "cogs", "label": "Annual Expenses", "amount": int(d.annual_expenses * 100)},
        ]

        spread_resp = await client.post(f"{base}/v1/deals/{deal_id}/spread", json={
            "entity_id": entity_id,
            "period": "TTM",
            "line_items": line_items,
        })
        spread_resp.raise_for_status()

        # Step 5: Mark origination outcome so underwriting stage guard can pass
        deal_patch_resp = await client.patch(
            f"{base}/v1/deals/{deal_id}",
            json={"origination_outcome": "proceed"},
        )
        deal_patch_resp.raise_for_status()

        # Step 6: Stage transitions (broker → origination → underwriting)
        # These may fail due to approval gates or other guards — that's OK,
        # the evaluate endpoint works regardless of deal stage.
        stage_headers = {
            "X-Actor": "system",
            "X-Tenant-Id": tenant_id,
            "Content-Type": "application/json",
        }
        for to_stage in ["origination", "underwriting"]:
            try:
                transition_resp = await client.post(
                    f"{base}/v1/deals/{deal_id}/stage-transitions",
                    json={"to_stage": to_stage, "rationale": "SIM pipeline progression"},
                    headers=stage_headers,
                )
                transition_resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Stage transition to %s failed for deal %s: status=%s body=%s",
                    to_stage,
                    deal_id,
                    exc.response.status_code,
                    exc.response.text,
                )
            except Exception as exc:
                logger.warning("Stage transition to %s failed for deal %s: %s", to_stage, deal_id, exc)

        # Step 7: Evaluate — always send dossier inline for parity with direct sim mode.
        eval_body: dict = {
            "policy": policy_dict,
            "provider": provider,
            "mode": mode,
            "dossier": dossier,
        }
        # Full mode also receives explicit model config.
        if mode == "full":
            eval_body["models"] = models

        eval_resp = await client.post(f"{base}/v1/deals/{deal_id}/evaluate", json=eval_body)
        eval_resp.raise_for_status()
        eval_result = eval_resp.json()

        # Step 7: Get audit events for trace
        try:
            audit_resp = await client.get(f"{base}/v1/deals/{deal_id}/audit")
            audit_events = audit_resp.json() if audit_resp.status_code == 200 else {}
        except Exception:
            audit_events = {}

    return _map_los_response(eval_result, borrower, lender, start_time)


def _map_los_response(
    eval_result: dict,
    borrower: Borrower,
    lender: LenderConfig,
    start_time: float,
) -> UnderwritingRun:
    """Map a LOS response dict → UnderwritingRun."""
    latency_ms = int((time.time() - start_time) * 1000)

    run = UnderwritingRun()
    run.run_id = eval_result.get("run_id", str(uuid.uuid4()))
    run.timestamp_utc = eval_result.get("timestamp_utc", "")

    # Case
    run.case = RunCase.from_borrower(borrower, source="los")

    # Policy
    run.policy = RunPolicy.from_lender(lender)

    # Inputs
    run.inputs = RunInputs.from_dossier(borrower.dossier)

    # Decision — map from LOS response
    los_decision = eval_result.get("decision", {})
    action = los_decision.get("action", "pending")
    terms = los_decision.get("terms", {})
    rationale_data = los_decision.get("rationale", eval_result.get("rationale", {}))

    run.decision = RunDecision(
        action=action,
        risk_grade=los_decision.get("risk_grade", ""),
        prob_default_12m=los_decision.get("prob_default_12m", 0.0),
        terms=DecisionTerms(
            amount=terms.get("amount", 0.0),
            apr=terms.get("apr", 0.0),
            tenor_months=terms.get("tenor_months", 0),
            fees=terms.get("fees", {}),
        ),
        conditions=los_decision.get("conditions", []),
        covenants=los_decision.get("covenants", []),
        rationale=DecisionRationale(
            summary=rationale_data.get("summary", ""),
            key_factors=rationale_data.get("key_factors", []),
            what_would_change=rationale_data.get("what_would_change", []),
        ),
        confidence=los_decision.get("confidence", 0.0),
    )

    # Trace — from LOS response
    los_trace = eval_result.get("trace", {})
    trace_steps = []
    for step in los_trace.get("steps", []):
        trace_steps.append(TraceStep(
            t=step.get("t", ""),
            type=step.get("type", "note"),
            name=step.get("name", ""),
            args=step.get("args", {}),
            result=step.get("result", {}),
            content=step.get("content", ""),
        ))

    run.trace = RunTrace(
        steps=trace_steps,
        latency_ms=latency_ms,
        cost=TraceCost(
            tokens_in=los_trace.get("cost", {}).get("tokens_in", 0),
            tokens_out=los_trace.get("cost", {}).get("tokens_out", 0),
            estimated_cost_usd=los_trace.get("cost", {}).get("estimated_cost_usd", 0.0),
        ),
    )

    # Labels: ground truth from borrower (hidden in production)
    if borrower.true_outcome:
        from .run_schema import RunLabels
        run.labels = RunLabels(
            available=True,
            gold={
                "true_outcome": borrower.true_outcome,
                "months_before_default": borrower.months_before_default,
                "correct_action": "decline" if borrower.true_outcome in ("bad", "fraud") else "approve",
            },
        )

    return run


LLM_FAILURE_PREFIX = "LLM evaluation failed:"


def run_to_decision(run: UnderwritingRun) -> LenderDecision:
    """Convert an UnderwritingRun back to a LenderDecision for adjudication.

    APR conversion: Run uses decimal (0.095), TermSheet uses percentage (9.5).

    When the LOS returns a decline due to an LLM failure (model 404, no tool
    call, etc.) rather than a genuine underwriting decision, the reasoning is
    prefixed with "[LLM_ERROR]" so callers can distinguish infrastructure
    failures from real rejections.
    """
    decision_str = "APPROVE" if run.decision.action == "approve" else "REJECT"
    reasoning = run.decision.rationale.summary or ""

    # Detect LLM infrastructure failures masquerading as declines
    if reasoning.startswith(LLM_FAILURE_PREFIX):
        logger.warning(
            "LLM failure for %s/%s: %s",
            run.policy.policy_id, run.case.case_id, reasoning,
        )
        reasoning = f"[LLM_ERROR] {reasoning}"

    term_sheet = None

    if run.decision.action == "approve":
        params = run.policy.params or {}
        requested_amount = float(getattr(run.case, "requested_amount", 0.0) or 0.0)
        requested_tenor = int(getattr(run.case, "requested_tenor_months", 24) or 24)
        max_single_loan = float(params.get("max_single_loan", requested_amount) or requested_amount)
        target_yield_pct = float(params.get("target_yield_pct", 10.0) or 10.0)

        # APR: decimal -> percentage
        apr_raw = run.decision.terms.apr
        rate_pct_raw = apr_raw * 100 if apr_raw <= 1.0 else apr_raw
        amount_raw = float(run.decision.terms.amount or 0.0)
        term_raw = int(run.decision.terms.tenor_months or requested_tenor or 24)

        if math.isfinite(amount_raw) and math.isfinite(rate_pct_raw):
            max_amount = min(max_single_loan, requested_amount) if requested_amount > 0 else max_single_loan
            norm_amount = max(0.0, min(amount_raw, max_amount))
            raw_or_target_rate = float(rate_pct_raw) if float(rate_pct_raw) > 0 else target_yield_pct
            norm_rate = max(0.1, min(raw_or_target_rate, 60.0))
            norm_term = max(1, min(term_raw, 120))

            if norm_amount > 0:
                term_sheet = TermSheet(
                    loan_amount=norm_amount,
                    interest_rate=norm_rate,
                    term_months=norm_term,
                )
                normalized = (
                    not math.isclose(norm_amount, amount_raw)
                    or not math.isclose(norm_rate, float(rate_pct_raw))
                    or norm_term != term_raw
                )
                if normalized:
                    reasoning = (reasoning + " [SYSTEM: Term sheet normalized to policy bounds]").strip()
            else:
                decision_str = "REJECT"
                reasoning = (reasoning + " [SYSTEM: Non-positive loan amount after policy clamp]").strip()
        else:
            decision_str = "REJECT"
            reasoning = (reasoning + " [SYSTEM: Invalid term sheet values]").strip()

    if term_sheet is None:
        decision_str = "REJECT"

    # Extract lender_id from policy
    policy_id = run.policy.policy_id
    # policy_id format: "p_{lender_id}_{model}"
    parts = policy_id.split("_", 2)
    lender_id = parts[1] if len(parts) >= 2 else policy_id

    return LenderDecision(
        lender_id=lender_id,
        borrower_id=run.case.case_id,
        decision=decision_str,
        reasoning=reasoning,
        term_sheet=term_sheet,
    )


async def evaluate_all_via_los(
    lender: LenderConfig,
    borrowers: list[Borrower],
    los_url: str = DEFAULT_LOS_URL,
    max_concurrent: int = 5,
    provider: str = "openrouter",
    mode: str = "rules_only",
    underwrite_only: bool = False,
    los_model: str | None = None,
    timeout: float | None = None,
) -> tuple[list[LenderDecision], list[UnderwritingRun]]:
    """Evaluate all borrowers for a single lender via LOS.

    Returns (decisions, runs) for compatibility with the engine.
    """
    # Default timeout: 180s for underwrite-only (LLM calls), 60s for rules
    if timeout is None:
        timeout = 180.0 if underwrite_only else 60.0

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _eval(borrower: Borrower) -> UnderwritingRun:
        async with semaphore:
            return await evaluate_via_los(
                borrower, lender, los_url,
                provider=provider, mode=mode, timeout=timeout,
                underwrite_only=underwrite_only,
                los_model=los_model,
            )

    runs = await asyncio.gather(*[_eval(b) for b in borrowers])
    decisions = [run_to_decision(r) for r in runs]
    return list(decisions), list(runs)
