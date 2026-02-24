"""
LOS Adapter — calls the Open LOS REST API to evaluate borrowers.

Replaces llm.py when running with --los flag. The adapter translates
SIM Borrower + LenderConfig into LOS API calls and maps the
UnderwritingRun response back into the SIM's data model.

API call sequence for one (borrower, lender) evaluation:
  1. POST /v1/entities         — create company entity
  2. POST /v1/deals            — create deal
  3. POST /v1/deals/{id}/documents (x2) — upload bank stmts + income JSON
  4. POST /v1/deals/{id}/spread — create financial spread
  5. POST /v1/deals/{id}/stage-transitions (x2) — broker→origination→underwriting
  6. POST /v1/deals/{id}/evaluate — trigger evaluation
  7. GET  /v1/deals/{id}/audit-events — get trace

APR convention (per los-cutover.md section 5):
  UnderwritingRun.DecisionTerms.apr uses DECIMAL (0.095 = 9.5%)
  TermSheet.interest_rate uses PERCENTAGE (9.5 = 9.5%)
  The adapter converts: term_sheet.interest_rate = run.decision.terms.apr * 100
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import httpx

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


async def evaluate_via_los(
    borrower: Borrower,
    lender: LenderConfig,
    los_url: str = DEFAULT_LOS_URL,
    provider: str = "openrouter",
    mode: str = "rules_only",
    timeout: float = 60.0,
) -> UnderwritingRun:
    """Evaluate a borrower through the Open LOS REST API.

    Returns an UnderwritingRun artifact with the LOS decision, trace, and inputs.
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
        except Exception:
            pass  # Document uploads are non-fatal

        # Step 4: Create spread from dossier
        d = borrower.dossier
        line_items = [
            {"category": "revenue", "label": "Annual Revenue", "amount": int(d.annual_revenue * 100)},
            {"category": "expense", "label": "Annual Expenses", "amount": int(d.annual_expenses * 100)},
            {"category": "income", "label": "Net Income", "amount": int(d.net_income * 100)},
        ]

        await client.post(f"{base}/v1/deals/{deal_id}/spread", json={
            "entity_id": entity_id,
            "period": "TTM",
            "line_items": line_items,
        })

        # Step 5: Stage transitions (broker → origination → underwriting)
        # These may fail due to approval gates or other guards — that's OK,
        # the evaluate endpoint works regardless of deal stage.
        for target_stage in ["origination", "underwriting"]:
            try:
                transition_resp = await client.post(
                    f"{base}/v1/deals/{deal_id}/stage-transitions",
                    json={"target_stage": target_stage},
                )
            except Exception:
                pass  # Stage transition failures are non-fatal

        # Step 6: Evaluate
        eval_resp = await client.post(f"{base}/v1/deals/{deal_id}/evaluate", json={
            "policy": {
                "policy_id": f"p_{lender.id}_{lender.model.replace('/', '_')}",
                "model": lender.model,
                "persona": lender.persona,
                "target_yield_pct": lender.target_yield_pct,
                "max_single_loan": lender.max_single_loan,
                "total_capital": lender.total_capital,
                "sector_limits": lender.sector_limits,
            },
            "provider": provider,
            "mode": mode,
        })
        eval_resp.raise_for_status()
        eval_result = eval_resp.json()

        # Step 7: Get audit events for trace
        try:
            audit_resp = await client.get(f"{base}/v1/deals/{deal_id}/audit")
            audit_events = audit_resp.json() if audit_resp.status_code == 200 else {}
        except Exception:
            audit_events = {}

    latency_ms = int((time.time() - start_time) * 1000)

    # Map LOS response → UnderwritingRun
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

    # Trace — from LOS response + audit events
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


def run_to_decision(run: UnderwritingRun) -> LenderDecision:
    """Convert an UnderwritingRun back to a LenderDecision for adjudication.

    APR conversion: Run uses decimal (0.095), TermSheet uses percentage (9.5).
    """
    decision_str = "APPROVE" if run.decision.action == "approve" else "REJECT"

    term_sheet = None
    if run.decision.action == "approve" and run.decision.terms.amount > 0:
        # APR: decimal → percentage
        apr_decimal = run.decision.terms.apr
        interest_rate_pct = apr_decimal * 100 if apr_decimal <= 1.0 else apr_decimal

        term_sheet = TermSheet(
            loan_amount=run.decision.terms.amount,
            interest_rate=interest_rate_pct,
            term_months=run.decision.terms.tenor_months or 24,
        )

    # Extract lender_id from policy
    policy_id = run.policy.policy_id
    # policy_id format: "p_{lender_id}_{model}"
    parts = policy_id.split("_", 2)
    lender_id = parts[1] if len(parts) >= 2 else policy_id

    return LenderDecision(
        lender_id=lender_id,
        borrower_id=run.case.case_id,
        decision=decision_str,
        reasoning=run.decision.rationale.summary or "",
        term_sheet=term_sheet,
    )


async def evaluate_all_via_los(
    lender: LenderConfig,
    borrowers: list[Borrower],
    los_url: str = DEFAULT_LOS_URL,
    max_concurrent: int = 5,
    provider: str = "openrouter",
    mode: str = "rules_only",
) -> tuple[list[LenderDecision], list[UnderwritingRun]]:
    """Evaluate all borrowers for a single lender via LOS.

    Returns (decisions, runs) for compatibility with the engine.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _eval(borrower: Borrower) -> UnderwritingRun:
        async with semaphore:
            return await evaluate_via_los(
                borrower, lender, los_url,
                provider=provider, mode=mode,
            )

    runs = await asyncio.gather(*[_eval(b) for b in borrowers])
    decisions = [run_to_decision(r) for r in runs]
    return list(decisions), list(runs)
