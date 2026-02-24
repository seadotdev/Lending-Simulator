"""
OpenRouter LLM client for lender agents.

Handles prompt construction, API calls (with tool use), and JSON response parsing.
Models receive quarterly income statements upfront and can optionally use a
sandboxed bash environment (via just-bash) to run jq/awk/grep queries against
the borrower's 12-month bank statement data.
"""

import asyncio
import json
import re
from typing import Any

try:
    from just_bash import Bash as JustBash
    from just_bash.types import ExecutionLimits
    JUST_BASH_AVAILABLE = True
except ModuleNotFoundError:
    JustBash = Any  # type: ignore[assignment]
    ExecutionLimits = None
    JUST_BASH_AVAILABLE = False

from openai import AsyncOpenAI
from .models import (
    Borrower,
    LenderConfig,
    LenderDecision,
    TermSheet,
)


# ---------------------------------------------------------------------------
# Trace logging — captures full input/output of every LLM call
# ---------------------------------------------------------------------------
_call_traces: list[dict] = []

# ---------------------------------------------------------------------------
# Token / cost tracking
# ---------------------------------------------------------------------------
_token_usage: dict[str, dict[str, int]] = {}  # model -> {prompt, completion}

# Approximate $/1M-token pricing from OpenRouter (as of Feb 2026)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input $/M tokens, output $/M tokens)
    # --- Frontier / expensive (near Haiku) ---
    "anthropic/claude-3.5-haiku":       (0.80, 4.00),
    "google/gemini-3-flash-preview":    (0.50, 3.00),
    "google/gemini-2.5-flash":          (0.30, 2.50),
    "deepseek/deepseek-r1":             (0.70, 2.50),
    "moonshotai/kimi-k2":               (0.50, 2.40),
    "qwen/qwen3.5-plus-02-15":         (0.40, 2.40),
    "minimax/minimax-m1":               (0.40, 2.20),
    "mistralai/mistral-medium-3.1":     (0.40, 2.00),
    "mistralai/devstral-medium":        (0.40, 2.00),
    "bytedance-seed/seed-1.6":          (0.25, 2.00),
    "openai/gpt-5-mini":               (0.25, 2.00),
    "qwen/qwen3-235b-a22b":            (0.46, 1.82),
    "deepseek/deepseek-r1-0528":        (0.40, 1.75),
    "z-ai/glm-4.7":                     (0.38, 1.70),
    "z-ai/glm-4.6":                     (0.35, 1.71),
    "openai/gpt-4.1-mini":             (0.40, 1.60),
    "mistralai/mistral-large-2512":     (0.50, 1.50),
    "qwen/qwen3-coder-flash":          (0.30, 1.50),
    "x-ai/grok-code-fast-1":           (0.20, 1.50),
    # --- Mid-range ---
    "anthropic/claude-3-haiku":         (0.25, 1.25),
    "qwen/qwen-plus":                   (0.40, 1.20),
    "minimax/minimax-m2.5":             (0.30, 1.10),
    "prime-intellect/intellect-3":      (0.20, 1.10),
    "qwen/qwen3.5-397b-a17b":          (0.15, 1.00),
    "inception/mercury":                (0.25, 1.00),
    "qwen/qwen3-coder":                (0.22, 1.00),
    "minimax/minimax-m2.1":             (0.27, 0.95),
    "deepseek/deepseek-chat":           (0.32, 0.89),
    "deepseek/deepseek-chat-v3-0324":   (0.19, 0.87),
    "deepseek/deepseek-chat-v3.1":      (0.15, 0.75),
    # --- Budget ---
    "openai/gpt-4o-mini":              (0.15, 0.60),
    "meta-llama/llama-4-maverick":      (0.15, 0.60),
    "cohere/command-r-08-2024":         (0.15, 0.60),
    "x-ai/grok-3-mini":                (0.30, 0.50),
    "x-ai/grok-4-fast":                (0.20, 0.50),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.10, 0.40),
    "google/gemini-2.5-flash-lite":     (0.10, 0.40),
    "openai/gpt-4.1-nano":             (0.10, 0.40),
    "google/gemini-2.0-flash-001":      (0.10, 0.40),
    "z-ai/glm-4.7-flash":              (0.06, 0.40),
    "openai/gpt-5-nano":               (0.05, 0.40),
    "qwen/qwen3-8b":                    (0.05, 0.40),
    "qwen/qwq-32b":                     (0.15, 0.40),
    "meta-llama/llama-3.1-70b-instruct": (0.40, 0.40),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "meta-llama/llama-4-scout":         (0.08, 0.30),
    "mistralai/devstral-small":         (0.10, 0.30),
    "stepfun/step-3.5-flash":          (0.10, 0.30),
    "google/gemini-2.0-flash-lite-001": (0.07, 0.30),
    "bytedance-seed/seed-1.6-flash":    (0.07, 0.30),
    "qwen/qwen3-30b-a3b":              (0.08, 0.28),
    "qwen/qwen3-32b":                   (0.08, 0.24),
    "qwen/qwen3-14b":                   (0.06, 0.24),
    "amazon/nova-lite-v1":              (0.06, 0.24),
    # --- Cheapest ---
    "nvidia/nemotron-3-nano-30b-a3b":   (0.05, 0.20),
    "qwen/qwen-turbo":                  (0.05, 0.20),
    "mistralai/ministral-14b-2512":     (0.20, 0.20),
    "mistralai/mistral-small-3.2-24b-instruct": (0.06, 0.18),
    "google/gemma-3-27b-it":            (0.04, 0.15),
    "nvidia/nemotron-nano-9b-v2":       (0.04, 0.16),
    "amazon/nova-micro-v1":             (0.04, 0.14),
    "z-ai/glm-4-32b":                  (0.10, 0.10),
    "qwen/qwen3-235b-a22b-2507":       (0.07, 0.10),
    "qwen/qwen-2.5-7b-instruct":       (0.04, 0.10),
    "mistralai/mistral-small-24b-instruct-2501": (0.05, 0.08),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),
    "mistralai/mistral-nemo":           (0.02, 0.04),
    "meta-llama/llama-3-8b-instruct":   (0.03, 0.04),
    "mistralai/mistral-7b-instruct":    (0.20, 0.20),
}


def _track_usage(model: str, response) -> None:
    """Accumulate token usage from an API response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    if model not in _token_usage:
        _token_usage[model] = {"prompt": 0, "completion": 0}
    _token_usage[model]["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
    _token_usage[model]["completion"] += getattr(usage, "completion_tokens", 0) or 0


def get_token_usage() -> dict[str, dict[str, int]]:
    """Return accumulated token usage per model."""
    return dict(_token_usage)


def get_cost_summary() -> dict[str, float]:
    """Estimate dollar cost per model based on token usage."""
    costs: dict[str, float] = {}
    for model, tokens in _token_usage.items():
        in_price, out_price = MODEL_PRICING.get(model, (1.0, 3.0))
        cost = (tokens["prompt"] / 1_000_000 * in_price +
                tokens["completion"] / 1_000_000 * out_price)
        costs[model] = round(cost, 4)
    return costs


def clear_usage() -> None:
    """Clear accumulated token/cost tracking."""
    _token_usage.clear()


def get_call_traces() -> list[dict]:
    """Return accumulated LLM call traces."""
    return list(_call_traces)


def clear_call_traces() -> None:
    """Clear accumulated traces."""
    _call_traces.clear()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOL_ANALYSE_BANK_STATEMENTS = {
    "type": "function",
    "function": {
        "name": "analyse_bank_statements",
        "description": (
            "Retrieve the full 12-month bank statement history for the current "
            "loan applicant. Returns a JSON array of monthly statements, each "
            "containing individual deposit and withdrawal transactions with "
            "dates, descriptions (customer/vendor names), and amounts. Use this "
            "to examine raw transaction data for any anomalies not visible in "
            "the quarterly income summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional focus area: 'deposits', 'withdrawals', or 'all'. "
                        "Defaults to 'all'."
                    ),
                    "enum": ["deposits", "withdrawals", "all"],
                },
            },
            "required": [],
        },
    },
}

TOOL_RUN_BASH = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": (
            "Execute a bash command in a sandboxed environment to analyse the "
            "loan applicant's 12-month bank statement data. The bank statements "
            "are pre-loaded at /data/bank_statements.json as a JSON array of "
            "monthly statements with deposits and withdrawals. Use jq, awk, "
            "grep, sort, uniq, wc, etc. to compute statistics and detect "
            "anomalies. Examples:\n"
            "  jq 'length' /data/bank_statements.json\n"
            "  jq '.[0] | keys' /data/bank_statements.json\n"
            "  jq '.[0].deposits[0]' /data/bank_statements.json"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The bash command to execute. The bank statement JSON is "
                        "at /data/bank_statements.json. Use jq for JSON queries, "
                        "or pipe through awk/grep/sort for further processing."
                    ),
                },
            },
            "required": ["command"],
        },
    },
}

TOOLS_LEGACY = [TOOL_ANALYSE_BANK_STATEMENTS]
TOOLS_SANDBOX = [TOOL_RUN_BASH]


def _bank_statements_to_json(borrower: Borrower, focus: str = "all") -> str:
    """Serialize bank statements to a compact JSON string for tool responses."""
    months = []
    for stmt in borrower.dossier.bank_statements:
        entry: dict = {
            "month": stmt.month,
            "opening_balance": stmt.opening_balance,
            "ending_balance": stmt.ending_balance,
        }
        if focus in ("all", "deposits"):
            entry["deposits"] = [
                {"date": t.date, "description": t.description, "amount": t.amount}
                for t in stmt.deposits
            ]
            entry["total_deposits"] = round(stmt.total_deposits, 2)
        if focus in ("all", "withdrawals"):
            entry["withdrawals"] = [
                {"date": t.date, "description": t.description, "amount": t.amount}
                for t in stmt.withdrawals
            ]
            entry["total_withdrawals"] = round(stmt.total_withdrawals, 2)
        months.append(entry)
    return json.dumps(months, indent=2)


# ---------------------------------------------------------------------------
# Dossier formatting (quarterly income statements)
# ---------------------------------------------------------------------------

def _format_dossier(borrower: Borrower, data_mode: str = "full") -> str:
    """Format a borrower's financial dossier for LLM consumption.

    data_mode controls what financial evidence is presented:
    - "full": quarterly income upfront + bank statement tool available (default)
    - "quarterly_only": quarterly income only, no raw bank data
    - "aggregate_only": just annual totals + narrative
    - "statements_inline": raw 12-month bank statements embedded in prompt
    - "lite": compact prompt with quarterly income, optimized for small models (3B-30B).
              Shorter instructions, no tool use, clearer structure.
    """
    d = borrower.dossier
    lines = []

    # --- Identity (always present) ---
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

    # --- Quarterly income (full, quarterly_only, lite) ---
    if data_mode in ("full", "quarterly_only", "lite"):
        lines.append("--- QUARTERLY INCOME STATEMENTS ---")
        lines.append("")
        header = f"  {'':20s}"
        for q in d.quarterly_income:
            header += f"{q.quarter:>14s}"
        lines.append(header)
        lines.append(f"  {'─'*20}" + f"{'─'*14}" * len(d.quarterly_income))

        row_rev = f"  {'Revenue':<20s}"
        row_exp = f"  {'Expenses':<20s}"
        row_ni  = f"  {'Net Income':<20s}"
        row_nm  = f"  {'Net Margin':<20s}"
        for q in d.quarterly_income:
            row_rev += f"{'${:>,.0f}'.format(q.revenue):>14s}"
            row_exp += f"{'(${:>,.0f})'.format(q.expenses):>14s}"
            row_ni  += f"{'${:>,.0f}'.format(q.net_income):>14s}"
            row_nm  += f"{q.net_margin_pct:>13.1f}%"
        lines.append(row_rev)
        lines.append(row_exp)
        lines.append(row_ni)
        lines.append(row_nm)
        lines.append("")

    # --- Annual totals (all modes except statements_inline) ---
    if data_mode != "statements_inline":
        lines.append("--- ANNUAL TOTALS ---")
        lines.append(f"Annual Revenue:  ${d.annual_revenue:,.0f}")
        lines.append(f"Annual Expenses: ${d.annual_expenses:,.0f}")
        lines.append(f"Net Income:      ${d.net_income:,.0f}")
        lines.append(f"Net Margin:      {d.net_income / d.annual_revenue * 100:.1f}%")
        lines.append("")

    # --- Raw bank statements inline (statements_inline only) ---
    if data_mode == "statements_inline":
        lines.append("--- 12-MONTH BANK STATEMENTS ---")
        lines.append(_bank_statements_to_json(borrower))
        lines.append("")

    # --- Data availability note ---
    if data_mode == "full":
        lines.append("NOTE: You have access to a sandboxed bash environment via the run_bash tool.")
        lines.append("The applicant's 12-month bank statements are at /data/bank_statements.json.")
        lines.append("Use jq queries to analyse deposit patterns, customer names, and anomalies.")
    elif data_mode == "quarterly_only":
        lines.append("NOTE: Your evaluation is based solely on the quarterly income data above.")
        lines.append("No raw bank statement data is available for this application.")
    elif data_mode == "aggregate_only":
        lines.append("NOTE: Only aggregate annual financial data is available.")
        lines.append("No quarterly breakdown or raw bank statements are available.")
    elif data_mode == "statements_inline":
        lines.append("NOTE: The raw 12-month bank statements are provided above for analysis.")
        lines.append("No pre-computed summaries are available — derive insights from the transactions.")
    elif data_mode == "lite":
        lines.append("Evaluate based on the quarterly income data and annual totals above.")

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


def _analysis_instructions(data_mode: str) -> str:
    """Generate mode-specific analysis instructions for the system prompt."""
    parts = []

    # Lite mode: compact instructions optimized for small models
    if data_mode == "lite":
        parts.append("INSTRUCTIONS:")
        parts.append("Evaluate this loan application. Check three things:")
        parts.append("")
        parts.append(
            "1. REVENUE TREND: Is quarterly revenue growing, flat, or declining?\n"
            "   Declining revenue = high risk."
        )
        parts.append(
            "\n2. DEBT SERVICE: Calculate annual loan payment.\n"
            "   If annual payment > net income, the business cannot service the debt. REJECT."
        )
        parts.append(
            "\n3. RED FLAGS: Look for anything suspicious:\n"
            "   - Revenue numbers that are unnaturally identical across quarters\n"
            "   - Company names in the narrative that overlap with related entities\n"
            "   - Very thin margins (net margin < 8%) combined with large loan requests\n"
            "   - Over-reliance on a single customer or revenue source"
        )
        return "\n".join(parts)

    parts.append("INSTRUCTIONS:")
    parts.append("Evaluate the loan application below. You must analyze:")
    parts.append("")

    # 1. CREDITWORTHINESS
    if data_mode in ("full", "quarterly_only"):
        parts.append(
            "1. CREDITWORTHINESS: Review the quarterly income statements carefully.\n"
            "   - Look at revenue trends across quarters — is revenue growing, flat, or declining?\n"
            "   - Look at margin trends — are margins stable, expanding, or compressing?\n"
            "   - Can this business service the debt from free cash flow?"
        )
    elif data_mode == "aggregate_only":
        parts.append(
            "1. CREDITWORTHINESS: Review the annual financial summary.\n"
            "   - Are revenue and margins healthy for this sector and business size?\n"
            "   - Can this business service the debt from free cash flow?\n"
            "   - Is the requested loan amount reasonable relative to revenue and income?"
        )
    else:  # statements_inline
        parts.append(
            "1. CREDITWORTHINESS: Analyze the bank statement transactions.\n"
            "   - Calculate approximate monthly revenue from deposits and expenses from withdrawals\n"
            "   - Look for revenue trends — are monthly deposits growing, flat, or declining?\n"
            "   - Can this business service the debt from available cash flow?"
        )

    # 2. FRAUD DETECTION
    if data_mode == "full":
        parts.append(
            "\n2. FRAUD DETECTION: Use the run_bash tool to query the bank statement data at\n"
            "   /data/bank_statements.json with jq. Examine the raw transactions for any\n"
            "   patterns or anomalies that could indicate fabrication, misrepresentation,\n"
            "   or financial manipulation. Consider what normal business transactions look\n"
            "   like and flag anything that strikes you as unusual."
        )
    elif data_mode == "quarterly_only":
        parts.append(
            "\n2. FRAUD DETECTION: Examine the quarterly financial data for any signs of\n"
            "   fabrication or misrepresentation. Consider whether the numbers and trends\n"
            "   are consistent with what you'd expect from a real operating business."
        )
    elif data_mode == "aggregate_only":
        parts.append(
            "\n2. FRAUD DETECTION: With only aggregate data available, assess whether the\n"
            "   stated financials are plausible and internally consistent for this type\n"
            "   of business."
        )
    else:  # statements_inline
        parts.append(
            "\n2. FRAUD DETECTION: Carefully examine the raw bank statement transactions\n"
            "   for any patterns or anomalies that could indicate fabrication,\n"
            "   misrepresentation, or financial manipulation. Consider what normal\n"
            "   business transactions look like and flag anything unusual."
        )

    # 3. PORTFOLIO FIT (always the same)
    parts.append(
        "\n3. PORTFOLIO FIT: Would this loan breach your sector concentration limits?\n"
        "   - Consider your existing exposure to this sector\n"
        "   - Factor in the new loan amount when checking limits"
    )

    # Tool usage note (only for full mode)
    if data_mode == "full":
        parts.append(
            "\nIMPORTANT: You SHOULD use the run_bash tool to query /data/bank_statements.json\n"
            "before making your decision. The file is a JSON array of monthly statements,\n"
            "each with deposits (date, description, amount) and withdrawals (date,\n"
            "description, amount), plus total_deposits and total_withdrawals per month.\n"
            "Use jq to compute summary statistics and look for anomalies.\n"
            "You can run multiple queries. Each call is independent."
        )

    return "\n".join(parts)


def _build_system_prompt(lender: LenderConfig, data_mode: str = "full") -> str:
    """Build the system prompt that defines the lender's persona and guidelines."""

    # Lite mode: compact system prompt for small models
    if data_mode == "lite":
        return f"""You are a loan underwriter. Evaluate loan applications and decide APPROVE or REJECT.

YOUR GUIDELINES:
- Target yield: {lender.target_yield_pct}% annual
- Max single loan: ${lender.max_single_loan:,.0f}
- Available capital: ${lender.total_capital:,.0f}

{_analysis_instructions(data_mode)}

Respond with ONLY a JSON object:
{{{{"decision": "APPROVE" or "REJECT", "reasoning": "Brief explanation", "term_sheet": {{{{"loan_amount": <number or null>, "interest_rate": <percent or null>, "term_months": <integer or null>}}}}}}}}"""

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

{_analysis_instructions(data_mode)}

When you are ready to give your final decision, respond with ONLY a valid JSON
object in exactly this format:
{{{{
  "decision": "APPROVE" or "REJECT",
  "reasoning": "Your 2-4 sentence analysis summary",
  "term_sheet": {{{{
    "loan_amount": <number or null if rejected>,
    "interest_rate": <annual rate as percentage e.g. 8.5, or null if rejected>,
    "term_months": <integer or null if rejected>
  }}}}
}}}}

Respond with ONLY the JSON when giving your final answer. No other text."""


def _build_user_prompt(borrower: Borrower, data_mode: str = "full") -> str:
    """Build the user prompt containing the borrower's dossier."""
    return f"Please evaluate the following loan application:\n\n{_format_dossier(borrower, data_mode)}"


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


def _extract_decision_from_text(text: str, loan_amount: float) -> dict | None:
    """Fallback: extract APPROVE/REJECT from free-form text when JSON parsing fails.

    Small models (3B-7B) often can't produce valid JSON but DO write coherent
    analysis with a clear approve/reject signal. This function extracts that
    signal from natural language output so the model gets credit for correct
    reasoning even without JSON compliance.

    Returns a synthetic JSON dict compatible with _parse_decision, or None.
    """
    text_lower = text.lower()

    # Look for explicit decision keywords
    approve_signals = [
        r"\bapprove\b", r"\bapproved\b", r"\brecommend(?:ed)?\s+(?:for\s+)?approval\b",
        r"\baccept\b", r"\bloan\s+is\s+approved\b",
    ]
    reject_signals = [
        r"\breject\b", r"\brejected\b", r"\bdeny\b", r"\bdenied\b",
        r"\bdecline\b", r"\bdeclined\b", r"\brecommend(?:ed)?\s+(?:for\s+)?rejection\b",
        r"\bcannot\s+(?:approve|recommend)\b", r"\bdo\s+not\s+(?:approve|recommend)\b",
    ]

    approve_count = sum(1 for p in approve_signals if re.search(p, text_lower))
    reject_count = sum(1 for p in reject_signals if re.search(p, text_lower))

    if approve_count == 0 and reject_count == 0:
        return None  # Can't determine decision

    # Use the stronger signal
    if approve_count > reject_count:
        # Extract a reasoning snippet (first ~200 chars of substantive text)
        reasoning = text.strip()[:200].replace("\n", " ").strip()
        return {
            "decision": "APPROVE",
            "reasoning": f"[extracted from text] {reasoning}",
            "term_sheet": {
                "loan_amount": loan_amount,
                "interest_rate": 10.0,  # Default rate
                "term_months": 24,      # Default term
            },
        }
    else:
        reasoning = text.strip()[:200].replace("\n", " ").strip()
        return {
            "decision": "REJECT",
            "reasoning": f"[extracted from text] {reasoning}",
        }


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


# ---------------------------------------------------------------------------
# Tool-use conversation loop
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 3   # Max tool-call round-trips before forcing a final answer
MAX_CALLS_PER_ROUND = 5  # Max parallel tool calls processed per round


def _create_sandbox(borrower: Borrower) -> JustBash | None:
    """Create a just-bash sandbox with the borrower's bank statements pre-loaded."""
    if not JUST_BASH_AVAILABLE or ExecutionLimits is None:
        return None

    bank_json = _bank_statements_to_json(borrower, focus="all")
    return JustBash(
        files={"/data/bank_statements.json": bank_json},
        limits=ExecutionLimits(
            max_command_count=500,
            max_loop_iterations=1000,
            max_awk_iterations=1000,
        ),
    )


async def _exec_sandbox(sandbox: JustBash, command: str) -> str:
    """Execute a command in the sandbox, returning formatted output."""
    result = await sandbox.exec(command)
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[stderr] {result.stderr}")
    if result.exit_code != 0:
        parts.append(f"[exit code: {result.exit_code}]")
    return "\n".join(parts) if parts else "(no output)"


async def evaluate_borrower(
    client: AsyncOpenAI,
    lender: LenderConfig,
    borrower: Borrower,
    semaphore: asyncio.Semaphore,
    data_mode: str = "full",
) -> LenderDecision:
    """Have a lender LLM evaluate a borrower, with tool-use support.

    data_mode controls what financial data the model sees:
    - "full": quarterly income + sandboxed bash tool for bank statements (default)
    - "quarterly_only": quarterly income only, no tool
    - "aggregate_only": annual totals only, no tool
    - "statements_inline": raw bank statements in prompt, no tool
    - "lite": compact prompt with quarterly income, no tool (optimized for small models)
    """
    system_prompt = _build_system_prompt(lender, data_mode)
    user_prompt = _build_user_prompt(borrower, data_mode)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tool_calls_made: list[str] = []  # Track tool usage for tracing

    # Create a per-evaluation sandbox if in full mode
    sandbox: JustBash | None = None
    if data_mode == "full":
        sandbox = _create_sandbox(borrower)

    async with semaphore:
        try:
            for round_num in range(MAX_TOOL_ROUNDS + 1):
                # On the last round, drop tools to force a final text answer
                kwargs: dict = {
                    "model": lender.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                }
                if round_num < MAX_TOOL_ROUNDS and data_mode == "full" and sandbox is not None:
                    kwargs["tools"] = TOOLS_SANDBOX
                else:
                    # No tools: either not full mode, or final round
                    pass

                response = await client.chat.completions.create(**kwargs)
                _track_usage(lender.model, response)
                msg = response.choices[0].message

                # Check if model wants to call tools
                if msg.tool_calls:
                    # Cap parallel tool calls per round
                    tool_calls_this_round = msg.tool_calls[:MAX_CALLS_PER_ROUND]

                    # Append assistant message with only the calls we'll process
                    assistant_msg = msg.model_dump()
                    assistant_msg["tool_calls"] = assistant_msg["tool_calls"][:MAX_CALLS_PER_ROUND]
                    messages.append(assistant_msg)

                    for tc in tool_calls_this_round:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                        if fn_name == "run_bash" and sandbox is not None:
                            command = fn_args.get("command", "echo 'no command'")
                            result = await _exec_sandbox(sandbox, command)
                            tool_calls_made.append(f"run_bash: {command}")
                        elif fn_name == "analyse_bank_statements":
                            # Legacy fallback: model called old tool name
                            focus = fn_args.get("focus", "all")
                            result = _bank_statements_to_json(borrower, focus)
                            tool_calls_made.append(f"analyse_bank_statements(focus={focus})")
                        else:
                            result = json.dumps({"error": f"Unknown tool: {fn_name}"})
                            tool_calls_made.append(f"UNKNOWN:{fn_name}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                    continue  # Loop back to get model's next response

                # No tool calls — this should be the final decision
                content = msg.content or ""
                raw = _extract_json(content)

                # Lite mode fallback: if JSON parsing failed, try extracting
                # the decision from free-form text.  Small models (3B-7B)
                # often write correct analysis but can't format JSON.
                if raw is None and data_mode == "lite":
                    raw = _extract_decision_from_text(
                        content, borrower.dossier.loan_request_amount,
                    )

                decision = _parse_decision(lender.id, borrower.id, raw)

                # Log the full trace
                _call_traces.append({
                    "lender_id": lender.id,
                    "lender_name": lender.name,
                    "model": lender.model,
                    "borrower_id": borrower.id,
                    "borrower_name": borrower.dossier.company_name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "tool_calls": tool_calls_made,
                    "tool_rounds": round_num,
                    "raw_response": content,
                    "parsed_json": raw,
                    "decision": decision.decision,
                    "reasoning": decision.reasoning,
                })

                return decision

            # Should not reach here, but safety fallback
            return LenderDecision(
                lender_id=lender.id,
                borrower_id=borrower.id,
                decision="REJECT",
                reasoning="[SYSTEM: Exceeded max tool rounds without final decision]",
            )

        except Exception as e:
            _call_traces.append({
                "lender_id": lender.id,
                "lender_name": lender.name,
                "model": lender.model,
                "borrower_id": borrower.id,
                "borrower_name": borrower.dossier.company_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "tool_calls": tool_calls_made,
                "raw_response": None,
                "error": f"{type(e).__name__}: {e}",
                "decision": "REJECT",
                "reasoning": f"[SYSTEM ERROR: {type(e).__name__}: {e}]",
            })
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
    data_mode: str = "full",
) -> list[LenderDecision]:
    """Run all borrower evaluations for a single lender concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        evaluate_borrower(client, lender, borrower, semaphore, data_mode)
        for borrower in borrowers
    ]
    return await asyncio.gather(*tasks)
