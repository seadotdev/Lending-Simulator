# Structured Disclosure Protocol Demo

- Episodes: `3`
- Conformant after mediation: `2`
- Escalations required: `1`
- Rejected non-conformant messages: `2`
- Forwarded boundary violations: `0`

## clean_exchange
- Final status: `conformant_exchange`
- enforcer::accept_request {"actor": "enforcer", "requested_fields": ["bank_feed_12m", "tax_filing_latest"], "type": "accept_request"}
- enforcer::forward_response {"actor": "enforcer", "attempt": 1, "evidence_source": "verified_feed", "forwarded_keys": ["bank_feed_12m", "tax_filing_latest"], "type": "forward_response"}

## boundary_violation_caught
- Final status: `conformant_exchange`
- enforcer::accept_request {"actor": "enforcer", "requested_fields": ["bank_feed_12m", "tax_filing_latest"], "type": "accept_request"}
- enforcer::reject_response {"actor": "enforcer", "attempt": 1, "blocked_payload_keys": ["bank_feed_12m", "client_names", "tax_filing_latest"], "errors": ["field_whitelist_violation:['client_names']", "boundary_violation:client_names"], "type": "reject_response"}
- enforcer::forward_response {"actor": "enforcer", "attempt": 2, "evidence_source": "verified_feed", "forwarded_keys": ["bank_feed_12m", "tax_filing_latest"], "type": "forward_response"}

## missing_evidence_escalation
- Final status: `escalation_required`
- enforcer::accept_request {"actor": "enforcer", "requested_fields": ["bank_feed_12m", "tax_filing_latest"], "type": "accept_request"}
- enforcer::reject_response {"actor": "enforcer", "attempt": 1, "blocked_payload_keys": ["bank_feed_12m"], "errors": ["evidence_source_nonconformant"], "type": "reject_response"}

