# Loanville Agent Task

You are an AI loan underwriter. You must process a borrower's loan application
through a running LOS (Loan Origination System) API.

## Environment

- The LOS API is running at: `http://localhost:3000`
- Your tenant ID is: `rdocker-20260307-210856_LND-003_BRW-025`
- All API calls must include the header: `X-Tenant-Id: rdocker-20260307-210856_LND-003_BRW-025`
- The OpenAPI spec is at: `/workspace/openapi.yaml`
- LOS architecture docs are at: `/workspace/LOS_DOCS.md`

## Your Task

Process this borrower's loan application through the LOS. Create the entity, deal, upload financial documents, create a spread, advance through stages, and make your underwriting decision. You should approve, decline, or counter the application based on the financials.

## Borrower Information

Company: Summit Event Catering
Sector: Hospitality Services
Loan Amount Requested: $400,000
Purpose: Second commercial prep kitchen buildout
Years in Business: 8
Annual Revenue: $2,157,000

## How to Interact with the LOS

Use `curl` to call the LOS REST API. Examples:

```bash
# Check health
curl -s http://localhost:3000/health | jq

# Create an entity (borrower)
curl -s -X POST http://localhost:3000/v1/entities \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: rdocker-20260307-210856_LND-003_BRW-025" \
  -d '{"name": "Acme Corp", "type": "company"}' | jq

# Create a deal
curl -s -X POST http://localhost:3000/v1/deals \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: rdocker-20260307-210856_LND-003_BRW-025" \
  -d '{"borrower": "Acme Corp", "amount": 500000}' | jq

# List deals
curl -s http://localhost:3000/v1/deals -H "X-Tenant-Id: rdocker-20260307-210856_LND-003_BRW-025" | jq
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
