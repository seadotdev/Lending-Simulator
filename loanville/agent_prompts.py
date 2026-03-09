"""
System prompts for the agentic LOS simulation.

Three mode variants:
  - tool_call: tools self-describe, model uses function calling
  - cli: full CLI reference in system prompt, model emits `los` commands
  - repl: same as CLI but persistent session context
"""

from __future__ import annotations

from .los_adapter import serialize_dossier, serialize_policy
from .models import Borrower, LenderConfig


CLI_REFERENCE = """## LOS CLI Reference

All commands use the `los` prefix. Output is JSON.

### Deals
  los deal create -b <borrower> [-j <jurisdiction>] [-a <amount>] [-p <purpose>] [--custom <json>]
  los deal list [-s <stage>] [-l <limit>]
  los deal get <id>
  los deal update <id> [-b <borrower>] [-a <amount>] [--outcome <reject|need_info|proceed|refer>] [--primary-entity <id>]
  los deal advance <id> -t <stage> [-r <rationale>] [--override true --override-rationale <text>]
  los deal check-guards <id> -t <stage>     # check what's needed before advancing
  los deal evaluate <id> [--mode full|rules_only]
  los deal history <id>

### Entities
  los entity create -t <company|person> -n <name> [--legal-name <name>] [--reg-number <num>] [-j <jurisdiction>]
  los entity list [-t <type>] [-l <limit>]
  los entity get <id>
  los entity update <id> [-n <name>] [--legal-name <name>]
  los entity delete <id>

### Relationships
  los relationship create --from <id> --to <id> --type <owns|guarantees|directs> [--ownership-pct <pct>]

### Documents
  los doc upload <deal_id> --type <type> --file <path>    # from file
  los doc upload <deal_id> --type <type> --data '<json>'  # inline JSON
  los doc list <deal_id>

### Spreads
  los spread create <deal_id> [--entity <id>] [--period <period>] [--items <json>]

### Covenants
  los covenant create <deal_id> --name <name> --type <financial|reporting|information> [--metric <metric>] [--operator <op>] [--threshold <val>]
  los covenant list <deal_id>
  los covenant test <deal_id>

### Facilities & Loans
  los facility create <deal_id> --type <term_loan|revolver|letter_of_credit> --amount <amount> [--rate <rate>] [--term <months>]
  los facility list <deal_id>
  los loan create --deal <id> --amount <amount> [--rate <rate>] [--term <months>]
  los loan get <id>
  los loan balance <id>
  los loan schedule <id>
  los loan transact <id> --type <APPROVAL|DISBURSEMENT|REPAYMENT> [--amount <amount>]

### Evaluation
  los deal evaluate <deal_id> [--mode full|rules_only]
  los underwrite --dossier '<json>'

### Monitoring & Audit
  los monitoring ingest <deal_id> --source <type> --data '<json>'
  los monitoring status <deal_id>
  los audit list <deal_id> [--type <type>] [--actor <actor>]

### Completion
When done, output your decision as:
```json
{"decision": "approve|decline|counter|refer", "reasoning": "...", "apr": 0.095, "amount": 50000, "term_months": 24}
```
"""


def build_system_prompt(
    mode: str,
    lender: LenderConfig,
    portfolio_summary: str = "",
    custom_tools_info: str = "",
) -> str:
    """Build the system prompt for a given interaction mode."""
    policy = serialize_policy(lender)
    persona = policy.get("persona", "a commercial lender")
    target_yield = policy.get("target_yield_pct", 10.0)
    max_loan = policy.get("max_single_loan", 500_000)
    sector_limits = policy.get("sector_limits", {})

    base = f"""You are an autonomous loan officer agent for a commercial lending institution.

## Your Role
{persona}

## Lending Parameters
- Target yield: {target_yield}%
- Maximum single loan: ${max_loan:,.0f}
- Sector limits: {sector_limits or 'none specified'}

## Portfolio Context
{portfolio_summary or 'No existing portfolio.'}

## Your Task
You are connected to a Loan Origination System (LOS). Use it to process the borrower application.
Complete the following steps to process the application:

1. Create an entity for the borrower
2. Create a deal with the loan request
3. Upload financial documents (bank statements, P&L)
4. Create a financial spread using `los_spread_create` with figures from the borrower's
   financials (revenue, cogs, operating_expense, net_income, total_debt, total_equity,
   current_assets, current_liabilities, etc.)
5. Advance the deal through stages (origination → underwriting)
6. Review the spread ratios and make your underwriting decision

## Decision Output
APR must be in DECIMAL form (0.095 = 9.5%). Maximum APR is 0.55.
When you've made your decision, signal completion clearly."""

    if mode == "tool_call":
        mode_section = """
## Interaction Mode: Tool Calling
Use the provided function-calling tools to interact with the LOS.
When done, call the `agent_done` tool with your decision."""
    elif mode == "cli":
        mode_section = f"""
## Interaction Mode: CLI
Execute LOS commands by outputting them. Each command should be on its own line starting with `los`.

{CLI_REFERENCE}"""
    elif mode == "repl":
        mode_section = f"""
## Interaction Mode: REPL Session
You have a persistent LOS session. Execute commands and the session state persists.

{CLI_REFERENCE}"""
    else:
        mode_section = ""

    custom_section = ""
    if custom_tools_info:
        custom_section = f"""
## Custom Analysis Tools
You also have custom analysis tools available. Use them when relevant for your evaluation:
{custom_tools_info}"""

    return base + mode_section + custom_section


def build_task_prompt(
    borrower: Borrower,
    task_description: str = "",
) -> str:
    """Build the user message with borrower data and task."""
    dossier = serialize_dossier(borrower)
    d = borrower.dossier

    task = task_description or "Process this borrower's loan application through the LOS and make an underwriting decision."

    prompt = f"""## Task
{task}

## Borrower Application
- Company: {d.company_name}
- Sector: {d.sector}
- Years in business: {d.years_in_business}
- Employees: {d.employee_count}

## Financial Summary
- Annual Revenue: ${d.annual_revenue:,.0f}
- Annual Expenses: ${d.annual_expenses:,.0f}
- Net Income: ${d.net_income:,.0f}
- Loan Request: ${d.loan_request_amount:,.0f} for {d.loan_purpose}

## Narrative
{d.narrative}
"""

    if d.quarterly_income:
        prompt += "\n## Quarterly Income\n"
        for q in d.quarterly_income:
            prompt += (
                f"- {q.quarter}: rev=${q.revenue:,.0f}, exp=${q.expenses:,.0f}, "
                f"net=${q.net_income:,.0f}, margin={q.net_margin_pct:.1f}%\n"
            )

    if d.bank_statements:
        prompt += "\n## Bank Statements (Summary)\n"
        for s in d.bank_statements[:6]:
            prompt += (
                f"- {s.month}: balance ${s.ending_balance:,.0f}, "
                f"deposits ${s.total_deposits:,.0f}, "
                f"withdrawals ${s.total_withdrawals:,.0f}\n"
            )

    prompt += f"\n## Full Dossier (JSON)\n```json\n{__import__('json').dumps(dossier, indent=2, default=str)}\n```"

    return prompt
