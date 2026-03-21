# Loan Underwriting Task

You are an AI loan underwriter. Process this borrower's application through a
Loan Origination System (LOS) and make a credit decision.

## Environment

- LOS API: `{LOS_URL}`
- Tenant: `{TENANT_ID}` (pass as `--tenant` or `-T` flag to all `los` commands)
- Budget file: `/workspace/output/budget.txt` — check periodically

## LOS CLI

The `los` command-line tool is your interface to the LOS. Start by exploring it:

```
los --help
los deal --help
los doc --help
los spread --help
```

Set your defaults once so you don't repeat them:

```
export LOS_API_URL={LOS_URL}
export LOS_TENANT_ID={TENANT_ID}
export LOS_FORMAT=json
```

## Workflow

Process the application through these stages: **broker → origination → underwriting → closing**

1. **Create entity & deal** — register the borrower and their loan request
2. **Upload documents** — attach financial data (use `--data` for inline JSON)
3. **Create spread** — enter financial metrics for ratio analysis
4. **Set outcome** — mark origination outcome as `proceed` (or `reject`/`refer`)
5. **Advance stages** — move the deal forward; use `check-guards` to see what's needed
6. **Evaluate** — run the underwriting evaluation
7. **Decide** — write your decision to `/workspace/output/decision.json`

### Checking requirements

Before advancing a stage, check what guards are satisfied:

```
los deal check-guards <deal_id> --to underwriting
```

This shows exactly which requirements are met and which are missing.

### Handling errors

If a stage advance fails, the error message includes fix hints showing the
exact command to resolve each unsatisfied guard.

## Borrower Information

{BORROWER_DOSSIER}

## Task

{TASK_PROMPT}

## Output

Write your final decision to `/workspace/output/decision.json`:

```json
{
  "decision": "approve|decline|counter|refer",
  "reasoning": "your analysis and rationale",
  "risk_grade": "A|B|C|D",
  "conditions": ["any conditions for approval"],
  "apr": 0.095,
  "amount": 50000,
  "term_months": 24
}
```
