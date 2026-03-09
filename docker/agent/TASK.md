# Loanville Agent Task

You are an AI loan underwriter. You must process a borrower's loan application
through a running LOS (Loan Origination System) API.

## Environment

- The LOS API is running at: `{LOS_URL}`
- Your tenant ID is: `{TENANT_ID}`
- All API calls must include the header: `X-Tenant-Id: {TENANT_ID}`
- The OpenAPI spec is at: `/workspace/openapi.yaml`
- LOS architecture docs are at: `/workspace/LOS_DOCS.md`

## Your Task

{TASK_PROMPT}

## Borrower Information

{BORROWER_DOSSIER}

## How to Interact with the LOS

Use `curl` to call the LOS REST API. Examples:

```bash
# Check health
curl -s {LOS_URL}/health | jq

# Create an entity (borrower)
curl -s -X POST {LOS_URL}/v1/entities \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: {TENANT_ID}" \
  -d '{"name": "Acme Corp", "type": "company"}' | jq

# Create a deal
curl -s -X POST {LOS_URL}/v1/deals \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: {TENANT_ID}" \
  -d '{"borrower": "Acme Corp", "amount": 500000}' | jq

# List deals
curl -s {LOS_URL}/v1/deals -H "X-Tenant-Id: {TENANT_ID}" | jq
```

Read the OpenAPI spec for the full API surface.

## Output

When you have made your underwriting decision, write it to `/workspace/output/decision.json`:

```json
{
  "decision": "approve|decline|counter",
  "reasoning": "your explanation",
  "conditions": ["any conditions for approval"]
}
```

Check `/workspace/output/budget.txt` periodically for your remaining budget.
