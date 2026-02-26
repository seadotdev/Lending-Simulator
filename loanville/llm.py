"""
Compatibility helpers for legacy test scripts that still import `loanville.llm`.

The runtime underwriting path has moved to `los_adapter` (live) and `mock_llm`
(mock). This module keeps lightweight prompt helpers plus token/trace counters
used by old diagnostics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from .models import Borrower, LenderConfig
from .mock_llm import _evaluate_mock

_token_usage: dict[str, dict[str, int]] = {}
_call_traces: list[dict] = []

# Conservative default pricing fallback ($ / 1M tokens)
_default_pricing = (0.25, 1.0)
_model_pricing: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-chat-v3-0324": (0.19, 0.87),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.10, 0.40),
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
}
MODEL_PRICING = dict(_model_pricing)


def _record_usage(model: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
    """Internal: accumulate token counters for compatibility reporting."""
    usage = _token_usage.setdefault(model, {"prompt": 0, "completion": 0})
    usage["prompt"] += max(0, int(tokens_in))
    usage["completion"] += max(0, int(tokens_out))


def _record_call_trace(trace: dict) -> None:
    """Internal: append one trace payload for compatibility reporting."""
    _call_traces.append(trace)


def clear_usage() -> None:
    _token_usage.clear()


def get_token_usage() -> dict[str, dict[str, int]]:
    return {k: dict(v) for k, v in _token_usage.items()}


def get_cost_summary() -> dict[str, float]:
    out: dict[str, float] = {}
    for model, usage in _token_usage.items():
        in_price, out_price = _model_pricing.get(model, _default_pricing)
        est = (usage["prompt"] / 1_000_000.0) * in_price + (
            usage["completion"] / 1_000_000.0
        ) * out_price
        out[model] = round(est, 4)
    return out


def clear_call_traces() -> None:
    _call_traces.clear()


def get_call_traces() -> list[dict]:
    return list(_call_traces)


def _format_quarterlies(borrower: Borrower) -> str:
    rows = []
    for q in borrower.dossier.quarterly_income:
        rows.append(
            f"- {q.quarter}: rev=${q.revenue:,.0f}, exp=${q.expenses:,.0f}, "
            f"net=${q.net_income:,.0f}, margin={q.net_margin_pct:.1f}%"
        )
    return "\n".join(rows) if rows else "- (none)"


def _format_statement_totals(borrower: Borrower) -> str:
    rows = []
    for s in borrower.dossier.bank_statements:
        rows.append(
            f"- {s.month}: opening=${s.opening_balance:,.0f}, deposits=${s.total_deposits:,.0f}, "
            f"withdrawals=${s.total_withdrawals:,.0f}, ending=${s.ending_balance:,.0f}"
        )
    return "\n".join(rows) if rows else "- (none)"


def _format_statements_inline(borrower: Borrower, months: Iterable) -> str:
    lines: list[str] = []
    for s in months:
        lines.append(f"- {s.month}")
        dep_preview = ", ".join(
            f"{d.description}:{d.amount:,.0f}" for d in s.deposits[:4]
        )
        wdr_preview = ", ".join(
            f"{w.description}:{w.amount:,.0f}" for w in s.withdrawals[:4]
        )
        lines.append(f"  deposits: {dep_preview or '(none)'}")
        lines.append(f"  withdrawals: {wdr_preview or '(none)'}")
    return "\n".join(lines) if lines else "- (none)"


def _build_system_prompt(
    lender: LenderConfig,
    data_mode: str = "full",
    tool_mode: str = "legacy",
) -> str:
    mode_note = {
        "full": (
            "You receive annual financials, quarterly trends, and bank statement data. "
            "Use all of it before deciding."
        ),
        "quarterly_only": (
            "You receive annual and quarterly financials only. No transaction-level bank data."
        ),
        "lite": (
            "You receive a compact dossier. Prioritize trend direction, debt-service ability, "
            "and obvious fraud signals."
        ),
        "statements_inline": (
            "You receive transaction-heavy statement snippets and annual context. "
            "Infer seasonality and suspicious cash movement."
        ),
        "aggregate_only": (
            "You only receive high-level aggregates. Be conservative where evidence is thin."
        ),
    }.get(data_mode, "Use provided borrower evidence to underwrite the loan.")
    tool_note = (
        "Tooling policy: use `analyse_bank_statements`/`run_bash` only when needed."
        if tool_mode == "legacy"
        else "Tooling policy: keep analysis concise."
    )
    return (
        "You are a commercial credit underwriter.\n"
        f"Lender profile: {lender.name} ({lender.model}).\n"
        f"Target yield: {lender.target_yield_pct:.1f}% | "
        f"Max single loan: ${lender.max_single_loan:,.0f}.\n"
        "Approve only if repayment risk is acceptable and pricing covers risk.\n"
        f"{mode_note}\n"
        f"{tool_note}\n"
        "Return a clear APPROVE/REJECT decision and concise rationale."
    )


def _build_user_prompt(borrower: Borrower, data_mode: str = "full") -> str:
    d = borrower.dossier
    header = (
        f"Borrower {borrower.id}: {d.company_name}\n"
        f"Sector: {d.sector} | Years in business: {d.years_in_business} | "
        f"Employees: {d.employee_count}\n"
        f"Loan request: ${d.loan_request_amount:,.0f} for {d.loan_purpose}\n"
        f"Annual revenue: ${d.annual_revenue:,.0f} | "
        f"Annual expenses: ${d.annual_expenses:,.0f} | "
        f"Net income: ${d.net_income:,.0f}\n"
        f"Narrative: {d.narrative}\n"
    )

    if data_mode == "aggregate_only":
        return (
            header
            + "Data mode: aggregate_only.\n"
            + "No quarterly or transaction-level details are available."
        )
    if data_mode == "quarterly_only":
        return (
            header
            + "Data mode: quarterly_only.\nQuarterly income:\n"
            + _format_quarterlies(borrower)
        )
    if data_mode == "lite":
        quarterlies = d.quarterly_income[-2:] if len(d.quarterly_income) > 2 else d.quarterly_income
        if quarterlies:
            first_q = quarterlies[0]
            last_q = quarterlies[-1]
            rev_delta_pct = (
                ((last_q.revenue - first_q.revenue) / first_q.revenue) * 100.0
                if first_q.revenue
                else 0.0
            )
            margin_delta = last_q.net_margin_pct - first_q.net_margin_pct
            quarterly_summary = (
                f"Revenue trend: {first_q.quarter}->{last_q.quarter} {rev_delta_pct:+.1f}%; "
                f"net margin change {margin_delta:+.1f}pp."
            )
        else:
            quarterly_summary = "Revenue trend: unavailable."

        if d.bank_statements:
            last_stmt = d.bank_statements[-1]
            cash_summary = (
                f"Latest cash snapshot ({last_stmt.month}): deposits=${last_stmt.total_deposits:,.0f}, "
                f"withdrawals=${last_stmt.total_withdrawals:,.0f}, ending=${last_stmt.ending_balance:,.0f}."
            )
        else:
            cash_summary = "Latest cash snapshot: unavailable."
        return (
            header
            + "Data mode: lite.\n"
            + quarterly_summary
            + "\n"
            + cash_summary
        )
    if data_mode == "statements_inline":
        return (
            header
            + "Data mode: statements_inline.\n"
            + "Quarterly income:\n"
            + _format_quarterlies(borrower)
            + "\nStatement transactions (preview):\n"
            + _format_statements_inline(borrower, d.bank_statements[-4:])
        )

    # full (default)
    return (
        header
        + "Data mode: full.\n"
        + "Quarterly income:\n"
        + _format_quarterlies(borrower)
        + "\nBank statement monthly totals:\n"
        + _format_statement_totals(borrower)
    )


async def evaluate_borrower(
    client,
    lender: LenderConfig,
    borrower: Borrower,
    semaphore,
    data_mode: str = "full",
    funding_rate_pct: float = 4.0,
):
    """Legacy-compatible async borrower evaluation.

    This compatibility implementation uses deterministic mock underwriting so
    legacy diagnostics still run even after the LOS cutover.
    """
    del client  # Compatibility only: evaluation now uses the mock harness.
    del funding_rate_pct

    system_prompt = _build_system_prompt(lender, data_mode=data_mode)
    user_prompt = _build_user_prompt(borrower, data_mode=data_mode)

    async with semaphore:
        decision = _evaluate_mock(lender, borrower, data_mode=data_mode)

    parsed = {
        "decision": decision.decision,
        "reasoning": decision.reasoning,
        "term_sheet": {
            "loan_amount": decision.term_sheet.loan_amount,
            "interest_rate": decision.term_sheet.interest_rate,
            "term_months": decision.term_sheet.term_months,
        }
        if decision.term_sheet
        else None,
    }
    _record_usage(
        lender.model,
        tokens_in=max(1, (len(system_prompt) + len(user_prompt)) // 4),
        tokens_out=max(1, len(decision.reasoning) // 4),
    )
    _record_call_trace(
        {
            "lender_id": lender.id,
            "lender_name": lender.name,
            "model": lender.model,
            "borrower_id": borrower.id,
            "borrower_name": borrower.dossier.company_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "tool_calls": [],
            "tool_rounds": 0,
            "raw_response": json.dumps(parsed),
            "parsed_json": parsed,
            "decision": decision.decision,
            "reasoning": decision.reasoning,
            "error": None,
        }
    )
    return decision
