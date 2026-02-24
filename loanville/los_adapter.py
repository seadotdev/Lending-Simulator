"""LOS API adapter for the Loanville simulation engine.

This module integrates Loanville with Open LOS using existing endpoints:
  - POST /v1/deals
  - POST /v1/deals/{dealId}/spread
  - GET  /v1/deals/{dealId}/ratios

The adapter converts Loanville Borrower/Lender objects into LOS requests,
then maps LOS ratio outputs into a deterministic underwriting decision.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

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


class LOSAdapterError(RuntimeError):
    """Raised when the LOS adapter cannot complete an evaluation."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round2(value: float) -> float:
    return round(float(value), 2)


def _load_json(
    method: str,
    base_url: str,
    path: str,
    timeout_s: float,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url=url,
        data=body,
        method=method.upper(),
        headers={
            "Accept": "application/json",
            **headers,
            **({"Content-Type": "application/json"} if payload is not None else {}),
        },
    )

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8").strip()
            return json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        preview = details[:300] if details else exc.reason
        raise LOSAdapterError(
            f"LOS request failed: {method.upper()} {path} -> {exc.code} ({preview})"
        ) from exc
    except error.URLError as exc:
        raise LOSAdapterError(f"LOS request failed: {method.upper()} {path} -> {exc.reason}") from exc
    except TimeoutError as exc:
        raise LOSAdapterError(f"LOS request timeout: {method.upper()} {path}") from exc
    except json.JSONDecodeError as exc:
        raise LOSAdapterError(f"LOS response was not valid JSON for {method.upper()} {path}") from exc


def _collect_bank_features(borrower: Borrower) -> dict[str, float]:
    statements = borrower.dossier.bank_statements
    if not statements:
        return {
            "avg_monthly_deposits": 0.0,
            "avg_monthly_withdrawals": 0.0,
            "avg_ending_balance": 0.0,
            "latest_ending_balance": 0.0,
        }

    avg_deps = sum(s.total_deposits for s in statements) / len(statements)
    avg_wds = sum(s.total_withdrawals for s in statements) / len(statements)
    avg_end = sum(s.ending_balance for s in statements) / len(statements)
    latest_end = statements[-1].ending_balance
    return {
        "avg_monthly_deposits": avg_deps,
        "avg_monthly_withdrawals": avg_wds,
        "avg_ending_balance": avg_end,
        "latest_ending_balance": latest_end,
    }


def _fraud_signals(borrower: Borrower) -> tuple[float, list[str]]:
    def _normalize_counterparty(description: str) -> str:
        text = description.lower().strip()
        for prefix in (
            "transfer from ",
            "transfer to ",
            "payment from ",
            "payment to ",
            "incoming ",
            "outgoing ",
        ):
            if text.startswith(prefix):
                text = text[len(prefix):]
        return re.sub(r"\s+", " ", text)

    deposits = []
    withdrawals = []
    for stmt in borrower.dossier.bank_statements:
        deposits.extend(stmt.deposits)
        withdrawals.extend(stmt.withdrawals)

    if not deposits:
        return 0.0, []

    round_hits = 0
    for txn in deposits:
        mod = abs(txn.amount) % 1000.0
        if abs(txn.amount) >= 1000 and (math.isclose(mod, 0.0, abs_tol=1e-3) or math.isclose(mod, 1000.0, abs_tol=1e-3)):
            round_hits += 1
    round_ratio = round_hits / len(deposits)

    withdrawal_descriptions = {_normalize_counterparty(w.description) for w in withdrawals}
    circular_hits = sum(
        1
        for d in deposits
        if _normalize_counterparty(d.description) in withdrawal_descriptions
    )
    circular_ratio = circular_hits / len(deposits)

    normalized_amounts = [round(d.amount, 2) for d in deposits]
    duplicate_count = len(normalized_amounts) - len(set(normalized_amounts))
    duplicate_ratio = duplicate_count / len(deposits)

    mean_deposit = sum(abs(d.amount) for d in deposits) / len(deposits)
    cv = statistics.pstdev([abs(d.amount) for d in deposits]) / mean_deposit if mean_deposit > 0 else 0.0
    low_variance_signal = 1.0 if cv < 0.08 and len(deposits) >= 24 else 0.0

    months = max(1, len(borrower.dossier.bank_statements))
    deposits_per_month = len(deposits) / months
    structuring_signal = 1.0 if deposits_per_month > 10.0 and mean_deposit < 15000.0 else 0.0

    score = _clamp(
        0.35 * round_ratio
        + 0.30 * circular_ratio
        + 0.20 * low_variance_signal
        + 0.25 * structuring_signal
        + 0.10 * duplicate_ratio,
        0.0,
        1.0,
    )

    signals: list[str] = []
    if round_ratio >= 0.45:
        signals.append("High concentration of round-number deposits")
    if circular_ratio >= 0.20:
        signals.append("Deposit/withdrawal counterparties overlap materially")
    if low_variance_signal > 0:
        signals.append("Deposit values are unnaturally stable month-over-month")
    if structuring_signal > 0:
        signals.append("High-frequency low-ticket deposit structuring detected")
    if duplicate_ratio >= 0.30:
        signals.append("Unusually repetitive deposit amounts")

    return score, signals


def _build_metrics_payload(borrower: Borrower, lender: LenderConfig) -> dict[str, float]:
    dossier = borrower.dossier
    bank = _collect_bank_features(borrower)

    revenue = max(0.0, dossier.annual_revenue)
    expenses = max(0.0, dossier.annual_expenses)
    net_income = dossier.net_income
    requested = max(0.0, dossier.loan_request_amount)

    cogs = expenses * 0.58
    operating_expense = max(0.0, expenses - cogs)
    interest_expense = max(1.0, requested * (lender.target_yield_pct / 100.0) * 0.45)
    tax = max(0.0, net_income * 0.20) if net_income > 0 else 0.0
    depreciation = max(0.0, revenue * 0.02)

    ebitda = max(0.0, net_income + interest_expense + tax + depreciation)
    total_debt = min(requested, lender.max_single_loan)
    debt_service = max(1.0, total_debt * ((lender.target_yield_pct / 100.0) + 0.04))

    current_assets = max(1.0, bank["avg_ending_balance"] + bank["avg_monthly_deposits"] * 1.1)
    current_liabilities = max(1.0, bank["avg_monthly_withdrawals"] * 0.9)
    total_equity = max(1.0, (net_income * 4.0) if net_income > 0 else revenue * 0.08)

    return {
        "revenue": _round2(revenue),
        "cogs": _round2(cogs),
        "operating_expense": _round2(operating_expense),
        "interest_expense": _round2(interest_expense),
        "tax": _round2(tax),
        "depreciation": _round2(depreciation),
        "net_income": _round2(net_income),
        "ebitda": _round2(ebitda),
        "total_debt": _round2(total_debt),
        "total_equity": _round2(total_equity),
        "current_assets": _round2(current_assets),
        "current_liabilities": _round2(current_liabilities),
        "debt_service": _round2(debt_service),
        "cash_and_equivalents": _round2(bank["latest_ending_balance"]),
    }


def _risk_profile(lender: LenderConfig) -> dict[str, float]:
    # Higher target yield implies more risk appetite.
    tilt = _clamp((lender.target_yield_pct - 11.0) / 6.0, -0.7, 0.7)
    return {
        "dscr_floor": 1.25 - 0.20 * tilt,
        "current_ratio_floor": 1.15 - 0.15 * tilt,
        "max_leverage": 4.50 + 1.00 * tilt,
        "min_net_margin": 0.05 - 0.015 * tilt,
        "max_debt_to_equity": 2.50 + 0.80 * tilt,
    }


def _numeric_ratio(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _extract_latest_ratios(
    spread_response: dict[str, Any],
    ratios_response: dict[str, Any],
) -> dict[str, float | None]:
    ratios = {}
    if isinstance(ratios_response.get("ratios"), list) and ratios_response["ratios"]:
        latest = ratios_response["ratios"][-1]
        if isinstance(latest, dict):
            ratios = latest
    if not ratios and isinstance(spread_response.get("ratios"), dict):
        ratios = spread_response["ratios"]

    return {
        "current_ratio": _numeric_ratio(ratios.get("current_ratio")),
        "debt_to_equity": _numeric_ratio(ratios.get("debt_to_equity")),
        "dscr": _numeric_ratio(ratios.get("dscr")),
        "gross_margin": _numeric_ratio(ratios.get("gross_margin")),
        "net_margin": _numeric_ratio(ratios.get("net_margin")),
        "leverage": _numeric_ratio(ratios.get("leverage")),
    }


def _risk_grade_from_score(score: float) -> str:
    if score >= 88:
        return "A"
    if score >= 76:
        return "B"
    if score >= 64:
        return "C"
    if score >= 52:
        return "D"
    if score >= 40:
        return "E"
    return "F"


def _build_decision(
    borrower: Borrower,
    lender: LenderConfig,
    ratios: dict[str, float | None],
    fraud_score: float,
    fraud_flags: list[str],
) -> RunDecision:
    profile = _risk_profile(lender)
    dscr = ratios["dscr"]
    current_ratio = ratios["current_ratio"]
    leverage = ratios["leverage"]
    net_margin = ratios["net_margin"]
    debt_to_equity = ratios["debt_to_equity"]

    hard_fail_reasons: list[str] = []
    soft_flags: list[str] = []

    if dscr is None or dscr < profile["dscr_floor"]:
        hard_fail_reasons.append(
            f"DSCR below floor ({dscr if dscr is not None else 'n/a'} < {profile['dscr_floor']:.2f})"
        )
    if current_ratio is None or current_ratio < profile["current_ratio_floor"]:
        hard_fail_reasons.append(
            f"Current ratio below floor ({current_ratio if current_ratio is not None else 'n/a'} < {profile['current_ratio_floor']:.2f})"
        )
    if net_margin is None or net_margin < profile["min_net_margin"]:
        hard_fail_reasons.append(
            f"Net margin below floor ({net_margin if net_margin is not None else 'n/a'} < {profile['min_net_margin']:.3f})"
        )
    if leverage is not None and leverage > profile["max_leverage"]:
        hard_fail_reasons.append(
            f"Leverage above cap ({leverage:.2f} > {profile['max_leverage']:.2f})"
        )
    if debt_to_equity is not None and debt_to_equity > profile["max_debt_to_equity"]:
        hard_fail_reasons.append(
            f"Debt-to-equity above cap ({debt_to_equity:.2f} > {profile['max_debt_to_equity']:.2f})"
        )

    fraud_decline_threshold = 0.20 if lender.target_yield_pct <= 11.0 else 0.25
    fraud_soft_threshold = max(0.12, fraud_decline_threshold - 0.07)

    if fraud_score >= fraud_decline_threshold:
        hard_fail_reasons.append(f"Fraud pattern score too high ({fraud_score:.2f})")
    elif fraud_score >= fraud_soft_threshold:
        soft_flags.append(f"Fraud pattern score elevated ({fraud_score:.2f})")

    soft_flags.extend(fraud_flags)

    action = "decline" if hard_fail_reasons else "approve"
    requested = borrower.dossier.loan_request_amount
    max_amount = min(requested, lender.max_single_loan)

    terms = DecisionTerms()
    covenants: list[str] = []
    conditions: list[str] = []

    if action == "approve":
        haircut = 1.0
        if dscr is not None and dscr < profile["dscr_floor"] + 0.25:
            haircut -= 0.15
        if net_margin is not None and net_margin < 0.08:
            haircut -= 0.10
        if fraud_score > 0.20:
            haircut -= 0.10
        amount = _round2(max_amount * _clamp(haircut, 0.55, 1.0))

        premium = 0.0
        if dscr is not None:
            premium += max(0.0, profile["dscr_floor"] - dscr) * 5.0
            if dscr > 2.0:
                premium -= 0.75
        if leverage is not None:
            premium += max(0.0, leverage - 3.0) * 0.60
        if debt_to_equity is not None:
            premium += max(0.0, debt_to_equity - 1.5) * 0.50
        if net_margin is not None and net_margin < 0.07:
            premium += 0.75
        if fraud_score > 0.22:
            premium += 2.50

        apr_pct = _clamp(
            lender.target_yield_pct + premium,
            max(4.5, lender.target_yield_pct - 2.0),
            35.0,
        )
        apr = normalize_apr_to_decimal(apr_pct)
        if not apr_within_sanity_limit(apr):
            apr = 0.50

        tenor = 30 if lender.target_yield_pct <= 8.5 else 24 if lender.target_yield_pct <= 11.5 else 18
        if dscr is not None and dscr > 2.0:
            tenor += 6
        if dscr is not None and dscr < 1.3:
            tenor -= 6
        tenor = int(_clamp(float(tenor), 12.0, 36.0))

        terms = DecisionTerms(amount=amount, apr=apr, tenor_months=tenor)

        if dscr is not None and dscr < 1.50:
            covenants.append("min_dscr_1_25")
        if current_ratio is not None and current_ratio < 1.30:
            covenants.append("min_current_ratio_1_20")
        if leverage is not None and leverage > 3.50:
            covenants.append("max_leverage_4_50")
        if fraud_score > 0.15:
            conditions.append("Provide 12 months of transaction-level statement exports")
    else:
        terms = DecisionTerms(amount=0.0, apr=0.0, tenor_months=0)
        conditions.extend([
            "Resubmit with stronger debt-service coverage and liquidity support",
            "Provide management accounts that reconcile monthly cash flow variance",
        ])

    risk_score = 100.0
    risk_score -= 14.0 * len(hard_fail_reasons)
    risk_score -= 5.0 * len(soft_flags)
    risk_score -= fraud_score * 20.0
    risk_score = _clamp(risk_score, 0.0, 100.0)

    prob_default = _clamp(
        0.02 + 0.07 * len(soft_flags) + 0.10 * len(hard_fail_reasons) + 0.22 * fraud_score,
        0.01,
        0.95,
    )

    key_factors = [
        f"DSCR={dscr:.2f}" if dscr is not None else "DSCR unavailable",
        f"Current ratio={current_ratio:.2f}" if current_ratio is not None else "Current ratio unavailable",
        f"Net margin={net_margin:.2%}" if net_margin is not None else "Net margin unavailable",
        f"Leverage={leverage:.2f}" if leverage is not None else "Leverage unavailable",
        f"Fraud score={fraud_score:.2f}",
    ]

    what_would_change = []
    if hard_fail_reasons:
        what_would_change.extend(
            [
                "Raise DSCR above lender-specific floor via lower requested debt service",
                "Demonstrate stronger short-term liquidity in operating cash data",
            ]
        )
    if fraud_score >= 0.35:
        what_would_change.append("Provide source-level support for atypical transaction patterns")

    if action == "approve":
        summary = (
            f"Approved using LOS ratio output with controlled risk adjustments; "
            f"DSCR {(dscr if dscr is not None else 0):.2f}, "
            f"current ratio {(current_ratio if current_ratio is not None else 0):.2f}, "
            f"fraud score {fraud_score:.2f}."
        )
    else:
        summary = (
            "Declined based on LOS ratio and risk checks: "
            + "; ".join(hard_fail_reasons[:3])
            + "."
        )

    confidence = _clamp(0.92 - 0.12 * len(soft_flags) - 0.18 * len(hard_fail_reasons), 0.20, 0.95)

    return RunDecision(
        action=action,
        risk_grade=_risk_grade_from_score(risk_score),
        prob_default_12m=round(prob_default, 4),
        terms=terms,
        conditions=conditions,
        covenants=covenants,
        rationale=DecisionRationale(
            summary=summary,
            key_factors=key_factors,
            what_would_change=what_would_change,
        ),
        confidence=round(confidence, 4),
    )


def _build_labels(borrower: Borrower) -> RunLabels:
    if not borrower.true_outcome:
        return RunLabels(available=False)
    return RunLabels(
        available=True,
        gold={
            "true_outcome": borrower.true_outcome,
            "months_before_default": borrower.months_before_default,
            "correct_action": "decline" if borrower.true_outcome in ("bad", "fraud") else "approve",
        },
    )


def run_to_lender_decision(
    run: UnderwritingRun,
    lender: LenderConfig,
    borrower: Borrower,
) -> LenderDecision:
    approved = run.decision.action == "approve"
    terms = None
    if approved:
        terms = TermSheet(
            loan_amount=run.decision.terms.amount,
            interest_rate=round(run.decision.terms.apr * 100.0, 4),
            term_months=run.decision.terms.tenor_months,
        )
    return LenderDecision(
        lender_id=lender.id,
        borrower_id=borrower.id,
        decision="APPROVE" if approved else "REJECT",
        reasoning=run.decision.rationale.summary,
        term_sheet=terms,
    )


async def evaluate_borrower_with_los(
    borrower: Borrower,
    lender: LenderConfig,
    base_url: str,
    timeout_s: float = 20.0,
    tenant_id: str = "loanville-sim",
) -> UnderwritingRun:
    """Evaluate one borrower by calling LOS API endpoints."""
    started = time.perf_counter()
    actor = f"loanville:{lender.id}"
    headers = {"X-Actor": actor, "X-Tenant-Id": tenant_id}

    deal_payload = {
        "borrower_name": borrower.dossier.company_name,
        "jurisdiction": "US",
        "requested_amount": borrower.dossier.loan_request_amount,
        "purpose": borrower.dossier.loan_purpose,
        "custom_fields": {
            "borrower_id": borrower.id,
            "lender_id": lender.id,
            "sim_source": "loanville",
            "sim_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    deal = await asyncio.to_thread(
        _load_json,
        "POST",
        base_url,
        "/v1/deals",
        timeout_s,
        headers,
        deal_payload,
    )
    deal_id = str(deal.get("id", "")).strip()
    if not deal_id:
        raise LOSAdapterError("LOS create deal response did not contain 'id'")

    metrics = _build_metrics_payload(borrower, lender)
    spread_payload = {"period": "TTM", "metrics": metrics}
    spread = await asyncio.to_thread(
        _load_json,
        "POST",
        base_url,
        f"/v1/deals/{deal_id}/spread",
        timeout_s,
        headers,
        spread_payload,
    )

    ratios_resp = await asyncio.to_thread(
        _load_json,
        "GET",
        base_url,
        f"/v1/deals/{deal_id}/ratios",
        timeout_s,
        headers,
        None,
    )
    ratios = _extract_latest_ratios(spread, ratios_resp)
    fraud_score, fraud_flags = _fraud_signals(borrower)
    run_decision = _build_decision(borrower, lender, ratios, fraud_score, fraud_flags)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    run = UnderwritingRun()
    run.case = RunCase.from_borrower(borrower, source="los")
    run.policy = RunPolicy.from_lender(lender, policy_id=f"los_{lender.id}_v1")
    run.inputs = RunInputs.from_dossier(borrower.dossier)
    run.decision = run_decision
    run.labels = _build_labels(borrower)
    run.trace = RunTrace(
        steps=[
            TraceStep(
                t=run.timestamp_utc,
                type="tool_call",
                name="los.create_deal",
                args={"path": "/v1/deals"},
                result={"deal_id": deal_id},
            ),
            TraceStep(
                t=run.timestamp_utc,
                type="tool_call",
                name="los.create_spread",
                args={"path": f"/v1/deals/{deal_id}/spread"},
                result={"spread_id": spread.get("spread_id"), "ratios": spread.get("ratios", {})},
            ),
            TraceStep(
                t=run.timestamp_utc,
                type="tool_call",
                name="los.get_ratios",
                args={"path": f"/v1/deals/{deal_id}/ratios"},
                result={"ratios": ratios},
            ),
            TraceStep(
                t=run.timestamp_utc,
                type="reasoning",
                content=run_decision.rationale.summary,
            ),
        ],
        latency_ms=elapsed_ms,
        cost=TraceCost(tokens_in=0, tokens_out=0, estimated_cost_usd=0.0),
    )
    return run


async def run_los_evaluations(
    lender: LenderConfig,
    borrowers: list[Borrower],
    base_url: str,
    max_concurrent: int = 5,
    timeout_s: float = 20.0,
    tenant_id: str = "loanville-sim",
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
            )
            return run_to_lender_decision(run, lender, borrower), run

    results = await asyncio.gather(*[_evaluate_one(b) for b in borrowers])
    decisions = [decision for decision, _ in results]
    runs = [run for _, run in results]
    return decisions, runs
