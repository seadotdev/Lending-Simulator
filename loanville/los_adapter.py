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
import re
import time
import uuid
from typing import Optional

import os

import httpx

logger = logging.getLogger(__name__)

from .models import (
    Borrower,
    LenderConfig,
    LenderDecision,
    LosConfig,
    TermSheet,
)
from .llm import _record_call_trace, _record_usage
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


async def preflight_season(
    lenders: list[LenderConfig],
    los_url: str = DEFAULT_LOS_URL,
    provider: str = "openrouter",
    timeout: float = 30.0,
    total_evaluations: int = 0,
) -> None:
    """Run preflight checks before a season starts.

    Validates LOS health, OpenRouter credits, and that every unique model
    across lenders actually works (correct ID, supports tool_choice, etc.).
    Raises RuntimeError with an actionable summary on any failure.
    """
    print("\n  PREFLIGHT CHECK")
    failures: list[str] = []

    # --- Check 1: LOS health ---
    try:
        await check_los_health(los_url, timeout=5.0)
        print(f"    LOS health .................. OK")
    except RuntimeError as exc:
        msg = str(exc)
        print(f"    LOS health .................. FAIL: {msg}")
        failures.append(f"LOS health: {msg}")
        # Can't continue without LOS
        raise RuntimeError(
            f"PREFLIGHT FAILED: LOS not reachable.\n  {msg}"
        )

    # --- Check 2: Credit / API key check ---
    if provider == "openrouter":
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if or_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "https://openrouter.ai/api/v1/credits",
                        headers={"Authorization": f"Bearer {or_key}"},
                    )
                    resp.raise_for_status()
                    data = resp.json().get("data", {})
                    total_credits = data.get("total_credits", 0.0)
                    total_usage = data.get("total_usage", 0.0)
                    balance = total_credits - total_usage

                    # Also fetch rate limit info
                    rl_resp = await client.get(
                        "https://openrouter.ai/api/v1/auth/key",
                        headers={"Authorization": f"Bearer {or_key}"},
                    )
                    rl_resp.raise_for_status()
                    rl_data = rl_resp.json().get("data", {})
                    rate_remaining = rl_data.get("limit_remaining")

                    # Estimate season cost if we know total evaluations
                    est_cost = 0.0
                    if total_evaluations > 0:
                        from .cost_tracking import _estimate_cost_per_eval
                        # Weight by number of lenders using each model
                        model_counts: dict[str, int] = {}
                        for l in lenders:
                            model_counts[l.model] = model_counts.get(l.model, 0) + 1
                        evals_per_lender = total_evaluations / len(lenders) if lenders else 0
                        for model, count in model_counts.items():
                            est_cost += _estimate_cost_per_eval(model) * evals_per_lender * count

                    est_tag = f", ~${est_cost:.2f} estimated" if est_cost > 0 else ""
                    status = f"${balance:.2f} balance, ${rate_remaining:.2f} rate limit{est_tag}"

                    if balance <= 0:
                        print(f"    OpenRouter credits .......... FAIL: {status}")
                        failures.append(f"OpenRouter credits: no balance remaining ({status})")
                    elif rate_remaining is not None and rate_remaining <= 0:
                        print(f"    OpenRouter credits .......... FAIL: {status}")
                        failures.append(f"OpenRouter credits: rate limit exhausted ({status})")
                    elif est_cost > 0 and balance < est_cost:
                        print(f"    OpenRouter credits .......... WARN: {status} — may not complete")
                    elif balance < 1.0 or (rate_remaining is not None and rate_remaining < 1.0):
                        print(f"    OpenRouter credits .......... WARN: {status}")
                    else:
                        print(f"    OpenRouter credits .......... {status}")
            except Exception as exc:
                print(f"    OpenRouter credits .......... WARN: could not check ({exc})")
        else:
            print(f"    OpenRouter credits .......... SKIP (no OPENROUTER_API_KEY)")
    elif provider == "anthropic":
        ak = os.environ.get("ANTHROPIC_API_KEY", "")
        if ak:
            print(f"    Anthropic API key ........... present ({ak[:12]}...)")
        else:
            print(f"    Anthropic API key ........... FAIL: no ANTHROPIC_API_KEY")
            failures.append("Anthropic API key: ANTHROPIC_API_KEY not set")
    elif provider == "openai":
        ok = os.environ.get("OPENAI_API_KEY", "")
        if ok:
            print(f"    OpenAI API key .............. present ({ok[:12]}...)")
        else:
            print(f"    OpenAI API key .............. FAIL: no OPENAI_API_KEY")
            failures.append("OpenAI API key: OPENAI_API_KEY not set")
    else:
        print(f"    API credentials ............. SKIP (no check for provider={provider})")

    # --- Check 3: Per-model smoke test via LOS /v1/underwrite ---
    unique_models: dict[str, list[str]] = {}  # model -> [lender names]
    for lender in lenders:
        model = lender.model
        unique_models.setdefault(model, []).append(lender.name)

    smoke_dossier = {
        "company_name": "Preflight Test",
        "sector": "Technology",
        "years_in_business": 5,
        "employee_count": 10,
        "narrative": "Smoke test for preflight validation.",
        "loan_request_amount": 50000,
        "loan_purpose": "Smoke test",
        "annual_revenue": 500000,
        "annual_expenses": 400000,
        "net_income": 100000,
    }
    smoke_policy = {
        "policy_id": "p_preflight",
        "model": "",  # filled per-model
        "persona": "You are a loan underwriter. Evaluate this application.",
        "target_yield_pct": 10.0,
        "max_single_loan": 50000,
        "total_capital": 1000000,
        "sector_limits": {},
    }

    base = los_url.rstrip("/")
    model_failures = 0
    total_models = len(unique_models)

    for model_id, lender_names in unique_models.items():
        label = model_id
        # Pad with dots for alignment
        dots = "." * max(2, 36 - len(label))
        try:
            t0 = time.time()
            policy = {**smoke_policy, "model": model_id}
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{base}/v1/underwrite",
                    json={
                        "dossier": smoke_dossier,
                        "policy": policy,
                        "provider": provider,
                        "models": {"default": model_id},
                    },
                    headers={
                        "Content-Type": "application/json",
                        "X-Actor": "sim:preflight",
                        "X-Tenant-Id": "preflight",
                    },
                )
            elapsed = time.time() - t0

            if resp.status_code >= 400:
                body = resp.text[:200]
                print(f"    {label} {dots} FAIL: {resp.status_code} {body}")
                failures.append(f"{model_id}: HTTP {resp.status_code} — {body}")
                model_failures += 1
            else:
                result = resp.json()
                decision = result.get("decision", {})
                action = decision.get("action", "")
                rationale = decision.get("rationale", {})
                summary = rationale.get("summary", "") if isinstance(rationale, dict) else str(rationale)

                if summary.startswith("LLM evaluation failed:"):
                    # LOS caught an LLM error and returned a synthetic decline
                    error_detail = summary[len("LLM evaluation failed:"):].strip()
                    print(f"    {label} {dots} FAIL: {error_detail[:120]}")
                    failures.append(f"{model_id}: {error_detail[:200]}")
                    model_failures += 1
                elif not action:
                    print(f"    {label} {dots} WARN: no action in response ({elapsed:.1f}s)")
                else:
                    print(f"    {label} {dots} OK ({elapsed:.1f}s)")

        except httpx.TimeoutException:
            print(f"    {label} {dots} FAIL: timeout after {timeout:.0f}s")
            failures.append(f"{model_id}: timeout after {timeout:.0f}s")
            model_failures += 1
        except Exception as exc:
            print(f"    {label} {dots} FAIL: {exc}")
            failures.append(f"{model_id}: {exc}")
            model_failures += 1

    # --- Summary ---
    if failures:
        summary_lines = [f"\n  PREFLIGHT FAILED: {len(failures)} issue(s) found."]
        for f in failures:
            summary_lines.append(f"    - {f}")
        summary_lines.append("  Fix these before running the season.")
        print("\n".join(summary_lines))
        raise RuntimeError(f"Preflight failed: {len(failures)} issue(s). See above.")
    else:
        print(f"    All {total_models} model(s) passed. Ready to go.\n")


async def bootstrap_los_tenant(
    tenant_id: str,
    los_config: LosConfig,
    los_url: str = DEFAULT_LOS_URL,
    timeout: float = 10.0,
) -> None:
    """Bootstrap LOS tenant configuration at season start.

    Sends disabled_guards via PUT /v1/settings and creates gate policies
    via POST /v1/gates/policies for the given tenant.
    """
    base = los_url.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "X-Tenant-Id": tenant_id,
        "X-Actor": "sim:bootstrap",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        # 1. Set disabled guards
        if los_config.disabled_guards:
            resp = await client.put(
                f"{base}/v1/settings",
                json={"disabled_guards": los_config.disabled_guards},
            )
            resp.raise_for_status()
            logger.info(
                "Bootstrapped tenant %s: disabled_guards=%s",
                tenant_id,
                los_config.disabled_guards,
            )

        # 2. Create gate policies
        for policy in los_config.gate_policies:
            resp = await client.post(
                f"{base}/v1/gates/policies",
                json=policy,
            )
            resp.raise_for_status()
            logger.info(
                "Bootstrapped gate policy for tenant %s: action=%s mode=%s",
                tenant_id,
                policy.get("action", "?"),
                policy.get("mode", "?"),
            )


def serialize_dossier(borrower: Borrower, borrower_view: Borrower | None = None) -> dict:
    """Serialize a Borrower's dossier into the LOS FinancialDossier format.

    If borrower_view is provided, serialize that lender-specific view instead
    (used by info_asymmetry modes where each lender sees filtered data).
    """
    source = borrower_view or borrower
    d = source.dossier
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
    persona = lender.persona
    if lender.custom_tools:
        tool_lines = [
            "Custom tools available for this lender (use when relevant):",
        ]
        for t in lender.custom_tools:
            fn = (t or {}).get("function", {}) if isinstance(t, dict) else {}
            name = fn.get("name", "unnamed_tool")
            desc = fn.get("description", "")
            if desc:
                tool_lines.append(f"- {name}: {desc}")
            else:
                tool_lines.append(f"- {name}")
        persona = f"{persona}\n\n" + "\n".join(tool_lines)

    policy: dict = {
        "policy_id": f"p_{lender.id}_{lender.model.replace('/', '_')}",
        "model": lender.model,
        "persona": persona,
        "target_yield_pct": lender.target_yield_pct,
        "max_single_loan": lender.max_single_loan,
        "total_capital": lender.total_capital,
        "sector_limits": lender.sector_limits,
    }
    if lender.policy_params:
        policy["params"] = dict(lender.policy_params)

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
            "allow_rules_fallback": False,
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

    # Backward-compatible usage/trace counters for legacy diagnostics.
    _record_usage(
        lender.model,
        tokens_in=run.trace.cost.tokens_in,
        tokens_out=run.trace.cost.tokens_out,
    )
    _record_call_trace(
        {
            "model": lender.model,
            "borrower_id": borrower.id,
            "policy_id": run.policy.policy_id,
            "decision": run.decision.action,
            "trace": {
                "steps": [
                    {
                        "type": s.type,
                        "name": s.name,
                        "args": s.args,
                        "result": s.result,
                    }
                    for s in run.trace.steps
                ],
                "cost": {
                    "tokens_in": run.trace.cost.tokens_in,
                    "tokens_out": run.trace.cost.tokens_out,
                    "estimated_cost_usd": run.trace.cost.estimated_cost_usd,
                },
            },
        }
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
PASS_PREFIX = "[PASS]"
_OFFER_VALID_RE = re.compile(r"(?:offer_valid_weeks|OFFER_VALID_WEEKS)\s*[:=]\s*(\d+)")


def _has_formal_offer_trace(run: UnderwritingRun) -> bool:
    """Whether the LOS trace shows at least one actionable tool call."""
    for step in getattr(run.trace, "steps", []) or []:
        if (step.type or "").lower() == "tool_call":
            return True
    return False


def _extract_offer_valid_weeks(run: UnderwritingRun, fallback: int = 1) -> int:
    """Best-effort extraction from LOS decision metadata."""
    conditions = getattr(run.decision, "conditions", None) or []
    for cond in conditions:
        if not isinstance(cond, str):
            continue
        m = _OFFER_VALID_RE.search(cond)
        if m:
            return max(1, int(m.group(1)))

    summary = getattr(getattr(run.decision, "rationale", None), "summary", "") or ""
    m = _OFFER_VALID_RE.search(summary)
    if m:
        return max(1, int(m.group(1)))

    return max(1, int(fallback))


def run_to_decision(
    run: UnderwritingRun,
    require_formal_offer_trace: bool = False,
) -> LenderDecision:
    """Convert an UnderwritingRun back to a LenderDecision for adjudication.

    APR conversion: Run uses decimal (0.095), TermSheet uses percentage (9.5).

    When the LOS returns a decline due to an LLM failure (model 404, no tool
    call, etc.) rather than a genuine underwriting decision, the reasoning is
    prefixed with "[LLM_ERROR]" so callers can distinguish infrastructure
    failures from real rejections.
    """
    action = (run.decision.action or "").lower()
    decision_str = "APPROVE" if action == "approve" else "REJECT"
    reasoning = run.decision.rationale.summary or ""
    offer_valid_weeks = 1

    # Detect LLM infrastructure failures masquerading as declines
    if reasoning.startswith(LLM_FAILURE_PREFIX):
        logger.warning(
            "LLM failure for %s/%s: %s",
            run.policy.policy_id, run.case.case_id, reasoning,
        )
        reasoning = f"[LLM_ERROR] {reasoning}"

    term_sheet = None

    is_pass = action == "refer" and reasoning.strip().upper().startswith(PASS_PREFIX)
    if is_pass:
        decision_str = "PASS"

    if action == "approve":
        if require_formal_offer_trace and not _has_formal_offer_trace(run):
            action = "refer"
            decision_str = "PASS"
            reasoning = (
                f"{reasoning} "
                "[LOS_FORMALITY] Approval ignored: no LOS tool_call trace for formal offer."
            ).strip()

    if action == "approve":
        params = run.policy.params or {}
        requested_amount = float(getattr(run.case, "requested_amount", 0.0) or 0.0)
        requested_tenor = int(getattr(run.case, "requested_tenor_months", 24) or 24)
        max_single_loan = float(params.get("max_single_loan", requested_amount) or requested_amount)
        target_yield_pct = float(params.get("target_yield_pct", 10.0) or 10.0)
        offer_valid_weeks = _extract_offer_valid_weeks(
            run,
            fallback=int(params.get("offer_validity_weeks", 1) or 1),
        )

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

    if term_sheet is None and decision_str == "APPROVE":
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
        offer_valid_weeks=offer_valid_weeks,
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
    require_formal_offer_trace: bool = False,
) -> tuple[list[LenderDecision], list[UnderwritingRun]]:
    """Evaluate all borrowers for a single lender via LOS.

    Returns (decisions, runs) for compatibility with the engine.
    """
    # Default timeout: 180s for LLM calls (underwrite-only or full+LLM mode), 60s for rules-only.
    # Reasoning models (DeepSeek-R1, QwQ, etc.) can take 90-180s per evaluation
    # due to internal chain-of-thought, so 60s is too aggressive for any LLM path.
    if timeout is None:
        if underwrite_only or mode != "rules_only":
            timeout = 180.0
        else:
            timeout = 60.0

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
    decisions = [
        run_to_decision(r, require_formal_offer_trace=require_formal_offer_trace)
        for r in runs
    ]
    return list(decisions), list(runs)
