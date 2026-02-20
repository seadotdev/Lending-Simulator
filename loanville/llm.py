"""
OpenRouter LLM client for lender agents.

Handles prompt construction, API calls, and JSON response parsing.
"""

import asyncio
import json
import re
from openai import AsyncOpenAI
from .models import (
    Borrower,
    LenderConfig,
    LenderDecision,
    TermSheet,
)


def _format_dossier(borrower: Borrower) -> str:
    """Format a borrower's financial dossier into a readable text block."""
    d = borrower.dossier
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"LOAN APPLICATION: {d.company_name}")
    lines.append(f"{'='*60}")
    lines.append(f"Application ID: {borrower.id}")
    lines.append(f"Sector: {d.sector}")
    lines.append(f"Years in Business: {d.years_in_business}")
    lines.append(f"Employee Count: {d.employee_count}")
    lines.append(f"Loan Requested: ${d.loan_request_amount:,.2f}")
    lines.append(f"Loan Purpose: {d.loan_purpose}")
    lines.append("")
    lines.append("--- COMPANY NARRATIVE ---")
    lines.append(d.narrative)
    lines.append("")
    lines.append("--- ANNUAL FINANCIAL SUMMARY ---")
    lines.append(f"Annual Revenue:  ${d.annual_revenue:,.2f}")
    lines.append(f"Annual Expenses: ${d.annual_expenses:,.2f}")
    lines.append(f"Net Income:      ${d.net_income:,.2f}")
    lines.append(f"Net Margin:      {d.net_income / d.annual_revenue * 100:.1f}%")
    lines.append("")
    lines.append("--- 12-MONTH BANK STATEMENT HISTORY ---")

    for stmt in d.bank_statements:
        lines.append(f"\n  === {stmt.month} ===")
        lines.append(f"  Opening Balance: ${stmt.opening_balance:,.2f}")
        lines.append("  DEPOSITS:")
        for t in stmt.deposits:
            lines.append(f"    {t.date}  {t.description:<40s} ${t.amount:>12,.2f}")
        lines.append(f"    {'Total Deposits:':<44s} ${stmt.total_deposits:>12,.2f}")
        lines.append("  WITHDRAWALS:")
        for t in stmt.withdrawals:
            lines.append(f"    {t.date}  {t.description:<40s} ${t.amount:>12,.2f}")
        lines.append(f"    {'Total Withdrawals:':<44s} ${stmt.total_withdrawals:>12,.2f}")
        lines.append(f"  Ending Balance: ${stmt.ending_balance:,.2f}")

    return "\n".join(lines)


def _format_portfolio_summary(lender: LenderConfig) -> str:
    """Format the lender's existing portfolio for the system prompt."""
    lines = []
    if not lender.existing_portfolio:
        lines.append("  (No existing loans)")
        return "\n".join(lines)

    total_deployed = sum(l.remaining_balance for l in lender.existing_portfolio)
    sector_exposure = {}
    for loan in lender.existing_portfolio:
        sector_exposure[loan.sector] = sector_exposure.get(loan.sector, 0) + loan.remaining_balance

    lines.append(f"  Total Capital: ${lender.total_capital:,.0f}")
    lines.append(f"  Currently Deployed: ${total_deployed:,.0f}")
    lines.append(f"  Available Capital: ${lender.total_capital - total_deployed:,.0f}")
    lines.append("")
    lines.append("  Existing Loans:")
    for loan in lender.existing_portfolio:
        lines.append(
            f"    - {loan.borrower_name} ({loan.sector}): "
            f"${loan.remaining_balance:,.0f} remaining at {loan.interest_rate}%"
        )
    lines.append("")
    lines.append("  Current Sector Exposure:")
    for sector, amount in sorted(sector_exposure.items()):
        pct = amount / lender.total_capital * 100
        limit = lender.sector_limits.get(sector, 0.25) * 100
        lines.append(f"    - {sector}: ${amount:,.0f} ({pct:.1f}% of capital, limit: {limit:.0f}%)")

    return "\n".join(lines)


def _build_system_prompt(lender: LenderConfig) -> str:
    """Build the system prompt that defines the lender's persona and guidelines."""
    sector_limits_str = "\n".join(
        f"    - {sector}: max {pct*100:.0f}% of total capital"
        for sector, pct in sorted(lender.sector_limits.items())
    )

    return f"""{lender.persona}

YOUR LENDING GUIDELINES:
- Target Portfolio Yield: {lender.target_yield_pct}% annual
- Maximum Single Loan Amount: ${lender.max_single_loan:,.0f}
- Sector Concentration Limits:
{sector_limits_str}

YOUR CURRENT PORTFOLIO:
{_format_portfolio_summary(lender)}

INSTRUCTIONS:
Evaluate the loan application below. You must analyze:

1. FRAUD DETECTION: Are the financial statements legitimate? Look for red flags such as:
   - Suspiciously round deposit amounts
   - Circular transfers between related entities
   - Unnaturally consistent figures month-over-month
   - Revenue claims that don't match bank deposit patterns

2. CREDITWORTHINESS: Can this business service the debt from free cash flow?
   - Calculate approximate monthly free cash flow
   - Assess revenue trends and stability
   - Check customer/revenue concentration risk
   - Evaluate expense trends vs revenue trends

3. PORTFOLIO FIT: Would this loan breach your sector concentration limits?
   - Consider your existing exposure to this sector
   - Factor in the new loan amount when checking limits

You MUST respond with ONLY a valid JSON object in exactly this format:
{{
  "decision": "APPROVE" or "REJECT",
  "reasoning": "Your 2-4 sentence analysis summary",
  "term_sheet": {{
    "loan_amount": <number or null if rejected>,
    "interest_rate": <annual rate as percentage e.g. 8.5, or null if rejected>,
    "term_months": <integer or null if rejected>
  }}
}}

Respond with ONLY the JSON. No other text before or after."""


def _build_user_prompt(borrower: Borrower) -> str:
    """Build the user prompt containing the borrower's dossier."""
    return f"Please evaluate the following loan application:\n\n{_format_dossier(borrower)}"


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM response text, handling markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{.*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def _parse_decision(lender_id: str, borrower_id: str, raw: dict | None) -> LenderDecision:
    """Parse a raw JSON dict into a LenderDecision, with fallback for bad data."""
    if raw is None:
        return LenderDecision(
            lender_id=lender_id,
            borrower_id=borrower_id,
            decision="REJECT",
            reasoning="[SYSTEM: Failed to parse LLM response as valid JSON]",
        )

    decision = str(raw.get("decision", "REJECT")).upper()
    if decision not in ("APPROVE", "REJECT"):
        decision = "REJECT"

    reasoning = str(raw.get("reasoning", "No reasoning provided"))

    term_sheet = None
    if decision == "APPROVE":
        ts = raw.get("term_sheet", {})
        if isinstance(ts, dict) and ts.get("loan_amount") is not None:
            try:
                term_sheet = TermSheet(
                    loan_amount=float(ts["loan_amount"]),
                    interest_rate=float(ts.get("interest_rate", 10.0)),
                    term_months=int(ts.get("term_months", 24)),
                )
            except (ValueError, TypeError):
                decision = "REJECT"
                reasoning += " [SYSTEM: Invalid term sheet values]"

    return LenderDecision(
        lender_id=lender_id,
        borrower_id=borrower_id,
        decision=decision,
        reasoning=reasoning,
        term_sheet=term_sheet,
    )


async def evaluate_borrower(
    client: AsyncOpenAI,
    lender: LenderConfig,
    borrower: Borrower,
    semaphore: asyncio.Semaphore,
) -> LenderDecision:
    """Have a lender LLM evaluate a single borrower application."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=lender.model,
                messages=[
                    {"role": "system", "content": _build_system_prompt(lender)},
                    {"role": "user", "content": _build_user_prompt(borrower)},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""
            raw = _extract_json(content)
            return _parse_decision(lender.id, borrower.id, raw)
        except Exception as e:
            return LenderDecision(
                lender_id=lender.id,
                borrower_id=borrower.id,
                decision="REJECT",
                reasoning=f"[SYSTEM ERROR: {type(e).__name__}: {e}]",
            )


async def run_lender_evaluations(
    client: AsyncOpenAI,
    lender: LenderConfig,
    borrowers: list[Borrower],
    max_concurrent: int = 5,
) -> list[LenderDecision]:
    """Run all borrower evaluations for a single lender concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        evaluate_borrower(client, lender, borrower, semaphore)
        for borrower in borrowers
    ]
    return await asyncio.gather(*tasks)
