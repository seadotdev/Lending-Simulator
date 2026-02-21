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
from just_bash import Bash as JustBash
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
    "deepseek/deepseek-chat-v3-0324":   (0.50, 1.50),
    "deepseek/deepseek-chat-v3.1":      (0.50, 1.50),
    "deepseek/deepseek-v3.2-20251201":  (0.50, 1.50),
    "qwen/qwen3-235b-a22b-07-25":      (0.70, 2.80),
    "qwen/qwen3-235b-a22b":            (0.70, 2.80),
    "qwen/qwen3-30b-a3b-04-28":        (0.14, 0.14),
    "google/gemini-2.0-flash-001":     (0.10, 0.40),
    "google/gemini-2.5-flash-preview": (0.15, 0.60),
    "z-ai/glm-4.7":                    (0.50, 0.50),
    "anthropic/claude-3.5-haiku":       (0.80, 4.00),
    "anthropic/claude-3.5-sonnet":      (3.00, 15.00),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.10, 0.40),
    "meta-llama/llama-3.1-8b-instruct": (0.05, 0.05),
    "qwen/qwen-2.5-7b-instruct":       (0.05, 0.05),
    "mistralai/mistral-7b-instruct":    (0.05, 0.05),
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

    # --- Quarterly income (full, quarterly_only) ---
    if data_mode in ("full", "quarterly_only"):
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


def _create_sandbox(borrower: Borrower) -> JustBash:
    """Create a just-bash sandbox with the borrower's bank statements pre-loaded."""
    from just_bash.types import ExecutionLimits

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
                if round_num < MAX_TOOL_ROUNDS and data_mode == "full":
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
