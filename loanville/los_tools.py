"""
LOS tool definitions for agentic mode.

Ports the 33 CLI_TOOLS from crm-test/tools.ts to Python dicts (OpenAI format),
plus adds los_deal_evaluate and los_underwrite tools.
"""

from __future__ import annotations


def _tool(name: str, description: str, properties: dict | None = None,
          required: list[str] | None = None) -> dict:
    params: dict = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


LOS_TOOLS: list[dict] = [
    _tool("los_health", "Check the API health status"),
    _tool("los_deal_create", "Create a new deal for a borrower", {
        "borrower": {"type": "string", "description": "Borrower name"},
        "jurisdiction": {"type": "string", "description": "Jurisdiction code (e.g., UK, US)"},
        "amount": {"type": "string", "description": "Requested amount (supports k/m/b suffixes)"},
        "purpose": {"type": "string", "description": "Loan purpose"},
        "custom": {"type": "string", "description": "Custom fields as JSON"},
    }, ["borrower"]),
    _tool("los_deal_list", "List deals, optionally filtered by stage", {
        "stage": {"type": "string", "description": "Filter by stage"},
        "limit": {"type": "string", "description": "Limit results (default: 20)"},
        "cursor": {"type": "string", "description": "Pagination cursor"},
        "format": {"type": "string", "description": "Output format", "enum": ["json", "table", "compact"]},
    }),
    _tool("los_deal_get", "Get details for a specific deal", {
        "id": {"type": "string", "description": "Deal ID"},
    }, ["id"]),
    _tool("los_deal_update", "Update deal fields", {
        "id": {"type": "string", "description": "Deal ID"},
        "borrower": {"type": "string", "description": "Borrower name"},
        "jurisdiction": {"type": "string", "description": "Jurisdiction"},
        "amount": {"type": "string", "description": "Requested amount"},
        "purpose": {"type": "string", "description": "Loan purpose"},
        "assigned_to": {"type": "string", "description": "Assign to user"},
        "outcome": {"type": "string", "description": "Origination outcome", "enum": ["reject", "need_info", "proceed", "refer"]},
        "primary_entity": {"type": "string", "description": "Primary entity ID"},
        "custom": {"type": "string", "description": "Custom fields as JSON"},
    }, ["id"]),
    _tool("los_deal_advance", "Advance deal to next stage", {
        "id": {"type": "string", "description": "Deal ID"},
        "to": {"type": "string", "description": "Target stage", "enum": ["origination", "underwriting", "closing", "monitoring"]},
        "rationale": {"type": "string", "description": "Rationale for transition"},
        "override": {"type": "string", "description": "Set to 'true' to override failed guards", "enum": ["true", "false"]},
        "override_rationale": {"type": "string", "description": "Rationale for override"},
    }, ["id", "to"]),
    _tool("los_deal_history", "Show stage transition history for a deal", {
        "id": {"type": "string", "description": "Deal ID"},
    }, ["id"]),
    _tool("los_entity_create", "Create a new entity (company or person)", {
        "type": {"type": "string", "description": "Entity type", "enum": ["company", "person"]},
        "name": {"type": "string", "description": "Entity name"},
        "legal_name": {"type": "string", "description": "Legal name"},
        "reg_number": {"type": "string", "description": "Registration number"},
        "jurisdiction": {"type": "string", "description": "Jurisdiction code"},
        "lei": {"type": "string", "description": "Legal Entity Identifier"},
    }, ["type", "name"]),
    _tool("los_entity_list", "List entities", {
        "type": {"type": "string", "description": "Filter by type", "enum": ["company", "person"]},
        "limit": {"type": "string", "description": "Limit results (default: 20)"},
    }),
    _tool("los_entity_get", "Get entity details", {
        "id": {"type": "string", "description": "Entity ID"},
    }, ["id"]),
    _tool("los_entity_update", "Update entity fields", {
        "id": {"type": "string", "description": "Entity ID"},
        "name": {"type": "string", "description": "Entity name"},
        "legal_name": {"type": "string", "description": "Legal name"},
        "reg_number": {"type": "string", "description": "Registration number"},
        "jurisdiction": {"type": "string", "description": "Jurisdiction code"},
        "lei": {"type": "string", "description": "Legal Entity Identifier"},
    }, ["id"]),
    _tool("los_entity_delete", "Delete an entity", {
        "id": {"type": "string", "description": "Entity ID"},
    }, ["id"]),
    _tool("los_relationship_create", "Create relationship between entities", {
        "from": {"type": "string", "description": "From entity ID"},
        "to": {"type": "string", "description": "To entity ID"},
        "type": {"type": "string", "description": "Relationship type", "enum": ["owns", "guarantees", "directs"]},
        "ownership_pct": {"type": "string", "description": "Ownership percentage"},
    }, ["from", "to", "type"]),
    _tool("los_doc_upload", "Upload a document to a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
        "type": {"type": "string", "description": "Document type"},
        "filename": {"type": "string", "description": "Filename"},
        "content_base64": {"type": "string", "description": "Base64-encoded content"},
    }, ["deal_id", "type"]),
    _tool("los_doc_list", "List documents for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
    }, ["deal_id"]),
    _tool("los_doc_get_content", "Get document content by ID", {
        "doc_id": {"type": "string", "description": "Document ID"},
    }, ["doc_id"]),
    _tool("los_covenant_create", "Create a covenant for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
        "name": {"type": "string", "description": "Covenant name"},
        "type": {"type": "string", "description": "Covenant type", "enum": ["financial", "reporting", "information"]},
        "metric": {"type": "string", "description": "Metric (e.g., debt_to_ebitda)"},
        "operator": {"type": "string", "description": "Comparison operator", "enum": [">=", "<=", ">", "<", "=="]},
        "threshold": {"type": "string", "description": "Threshold value"},
        "frequency": {"type": "string", "description": "Testing frequency", "enum": ["monthly", "quarterly", "annually"]},
        "grace_period": {"type": "string", "description": "Grace period in days"},
    }, ["deal_id", "name", "type"]),
    _tool("los_covenant_list", "List covenants for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
    }, ["deal_id"]),
    _tool("los_covenant_test", "Test covenant compliance for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
    }, ["deal_id"]),
    _tool("los_facility_create", "Create a loan facility for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
        "type": {"type": "string", "description": "Facility type", "enum": ["term_loan", "revolver", "letter_of_credit"]},
        "amount": {"type": "string", "description": "Facility amount"},
        "currency": {"type": "string", "description": "Currency (default: USD)"},
        "rate_type": {"type": "string", "description": "Interest rate type", "enum": ["fixed", "floating"]},
        "rate": {"type": "string", "description": "Interest rate value"},
        "term": {"type": "string", "description": "Term in months"},
    }, ["deal_id", "type", "amount"]),
    _tool("los_facility_list", "List facilities for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
    }, ["deal_id"]),
    _tool("los_loan_create", "Create a loan account", {
        "deal": {"type": "string", "description": "Deal ID"},
        "facility": {"type": "string", "description": "Facility ID"},
        "amount": {"type": "string", "description": "Loan amount"},
        "rate": {"type": "string", "description": "Interest rate"},
        "term": {"type": "string", "description": "Term in months"},
        "holder": {"type": "string", "description": "Account holder entity ID"},
    }, ["deal", "amount"]),
    _tool("los_loan_get", "Get loan details", {
        "id": {"type": "string", "description": "Loan ID"},
    }, ["id"]),
    _tool("los_loan_balance", "Get loan balance", {
        "id": {"type": "string", "description": "Loan ID"},
    }, ["id"]),
    _tool("los_loan_schedule", "Get repayment schedule", {
        "id": {"type": "string", "description": "Loan ID"},
    }, ["id"]),
    _tool("los_loan_transact", "Record a loan transaction", {
        "id": {"type": "string", "description": "Loan ID"},
        "type": {"type": "string", "description": "Transaction type", "enum": ["APPROVAL", "DISBURSEMENT", "REPAYMENT"]},
        "amount": {"type": "string", "description": "Transaction amount"},
        "date": {"type": "string", "description": "Value date (YYYY-MM-DD)"},
        "notes": {"type": "string", "description": "Notes"},
    }, ["id", "type"]),
    _tool("los_spread_create", "Create a financial spread for ratio analysis (DSCR, current ratio, debt-to-equity, margins, leverage). "
          "Use EITHER 'metrics' (simpler — pass key financial figures directly) OR 'items' (detailed line items). "
          "The spread computes ratios automatically and stores them on the deal.", {
        "deal_id": {"type": "string", "description": "Deal ID"},
        "entity": {"type": "string", "description": "Entity ID"},
        "period": {"type": "string", "description": "Period (e.g., FY2025)"},
        "metrics": {"type": "string", "description": "JSON object with financial figures. Accepted keys: "
                    "revenue, cogs, operating_expense, interest_expense, tax, depreciation, "
                    "net_income, ebitda, total_debt, total_equity, current_assets, current_liabilities, "
                    "debt_service, cash. Example: {\"revenue\": 850000, \"net_income\": 50000, \"total_debt\": 200000}"},
        "items": {"type": "string", "description": "Alternative to metrics: JSON array of line items, each with "
                  "{\"category\": \"<key>\", \"label\": \"<description>\", \"amount\": <number>}. "
                  "Categories: revenue, cogs, operating_expense, interest_expense, tax, depreciation, "
                  "current_assets, current_liabilities, total_debt, total_equity, cash"},
    }, ["deal_id"]),
    _tool("los_monitoring_ingest", "Ingest monitoring data for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
        "source": {"type": "string", "description": "Source type (e.g., bank_transactions)"},
        "data": {"type": "string", "description": "Transaction data as JSON"},
    }, ["deal_id", "source"]),
    _tool("los_monitoring_status", "Check monitoring status for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
    }, ["deal_id"]),
    _tool("los_audit_list", "List audit events for a deal", {
        "deal_id": {"type": "string", "description": "Deal ID"},
        "type": {"type": "string", "description": "Filter by event type"},
        "actor": {"type": "string", "description": "Filter by actor"},
    }, ["deal_id"]),
    _tool("los_deposit_create", "Create a deposit account", {
        "type": {"type": "string", "description": "Deposit type", "enum": ["demand_deposit", "time_deposit", "certificate_of_deposit"]},
        "holder": {"type": "string", "description": "Account holder name"},
        "currency": {"type": "string", "description": "Currency (default: USD)"},
    }, ["type", "holder"]),
    _tool("los_deposit_list", "List deposit accounts"),
    # --- Agent-specific evaluation tools ---
    _tool("los_deal_evaluate", "Request automated underwriting evaluation for a deal. Returns a structured decision with risk grade, terms, and rationale.", {
        "deal_id": {"type": "string", "description": "Deal ID to evaluate"},
        "mode": {"type": "string", "description": "Evaluation mode", "enum": ["full", "rules_only"]},
    }, ["deal_id"]),
    _tool("los_underwrite", "Standalone underwriting — submit a dossier for evaluation without going through LOS pipeline. Returns structured decision.", {
        "dossier_json": {"type": "string", "description": "Financial dossier as JSON string"},
    }, ["dossier_json"]),
    # --- Completion signal ---
    _tool("agent_done", "Signal that you have completed the task. Include your final decision.", {
        "decision": {"type": "string", "description": "Final decision: approve, decline, counter, refer", "enum": ["approve", "decline", "counter", "refer"]},
        "reasoning": {"type": "string", "description": "Brief explanation of your decision"},
        "apr": {"type": "number", "description": "Proposed APR as decimal (e.g., 0.095 for 9.5%)"},
        "amount": {"type": "number", "description": "Approved loan amount in dollars"},
        "term_months": {"type": "integer", "description": "Loan term in months"},
    }, ["decision", "reasoning"]),
]


# Map tool names to REST API endpoints and methods for the tool-call executor.
# Format: (method, url_template) where {arg_name} gets substituted from tool args.
TOOL_ROUTES: dict[str, tuple[str, str]] = {
    "los_health": ("GET", "/health"),
    "los_deal_create": ("POST", "/v1/deals"),
    "los_deal_list": ("GET", "/v1/deals"),
    "los_deal_get": ("GET", "/v1/deals/{id}"),
    "los_deal_update": ("PATCH", "/v1/deals/{id}"),
    "los_deal_advance": ("POST", "/v1/deals/{id}/stage-transitions"),
    "los_deal_history": ("GET", "/v1/deals/{id}/stage-transitions"),
    "los_entity_create": ("POST", "/v1/entities"),
    "los_entity_list": ("GET", "/v1/entities"),
    "los_entity_get": ("GET", "/v1/entities/{id}"),
    "los_entity_update": ("PATCH", "/v1/entities/{id}"),
    "los_entity_delete": ("DELETE", "/v1/entities/{id}"),
    "los_relationship_create": ("POST", "/v1/relationships"),
    "los_doc_upload": ("POST", "/v1/deals/{deal_id}/documents"),
    "los_doc_list": ("GET", "/v1/deals/{deal_id}/documents"),
    "los_doc_get_content": ("GET", "/v1/documents/{doc_id}/content"),
    "los_covenant_create": ("POST", "/v1/deals/{deal_id}/covenants"),
    "los_covenant_list": ("GET", "/v1/deals/{deal_id}/covenants"),
    "los_covenant_test": ("POST", "/v1/deals/{deal_id}/covenants/test"),
    "los_facility_create": ("POST", "/v1/deals/{deal_id}/facilities"),
    "los_facility_list": ("GET", "/v1/deals/{deal_id}/facilities"),
    "los_loan_create": ("POST", "/v1/loans"),
    "los_loan_get": ("GET", "/v1/loans/{id}"),
    "los_loan_balance": ("GET", "/v1/loans/{id}/balance"),
    "los_loan_schedule": ("GET", "/v1/loans/{id}/schedule"),
    "los_loan_transact": ("POST", "/v1/loans/{id}/transactions"),
    "los_spread_create": ("POST", "/v1/deals/{deal_id}/spread"),
    "los_monitoring_ingest": ("POST", "/v1/deals/{deal_id}/monitoring"),
    "los_monitoring_status": ("GET", "/v1/deals/{deal_id}/monitoring"),
    "los_audit_list": ("GET", "/v1/deals/{deal_id}/audit"),
    "los_deposit_create": ("POST", "/v1/deposits"),
    "los_deposit_list": ("GET", "/v1/deposits"),
    "los_deal_evaluate": ("POST", "/v1/deals/{deal_id}/evaluate"),
    "los_underwrite": ("POST", "/v1/underwrite"),
}

# Arguments that go into the URL path (not the request body/query)
PATH_ARGS: dict[str, list[str]] = {
    "los_deal_get": ["id"],
    "los_deal_update": ["id"],
    "los_deal_advance": ["id"],
    "los_deal_history": ["id"],
    "los_entity_get": ["id"],
    "los_entity_update": ["id"],
    "los_entity_delete": ["id"],
    "los_doc_upload": ["deal_id"],
    "los_doc_list": ["deal_id"],
    "los_doc_get_content": ["doc_id"],
    "los_covenant_create": ["deal_id"],
    "los_covenant_list": ["deal_id"],
    "los_covenant_test": ["deal_id"],
    "los_facility_create": ["deal_id"],
    "los_facility_list": ["deal_id"],
    "los_loan_get": ["id"],
    "los_loan_balance": ["id"],
    "los_loan_schedule": ["id"],
    "los_loan_transact": ["id"],
    "los_spread_create": ["deal_id"],
    "los_monitoring_ingest": ["deal_id"],
    "los_monitoring_status": ["deal_id"],
    "los_audit_list": ["deal_id"],
    "los_deal_evaluate": ["deal_id"],
}

# For advance, remap args to match the API's expected body shape
BODY_REMAP: dict[str, dict[str, str]] = {
    "los_deal_advance": {"to": "to_stage"},
    "los_deal_create": {"borrower": "borrower_name", "amount": "requested_amount"},
    "los_deal_update": {"borrower": "borrower_name", "amount": "requested_amount", "outcome": "origination_outcome", "primary_entity": "primary_entity_id"},
    "los_doc_upload": {"type": "doc_type"},
    "los_spread_create": {"entity": "entity_id", "items": "line_items"},
    "los_monitoring_ingest": {"source": "source_type"},
    "los_loan_create": {"deal": "deal_id"},
    "los_loan_transact": {"type": "transaction_type"},
}
