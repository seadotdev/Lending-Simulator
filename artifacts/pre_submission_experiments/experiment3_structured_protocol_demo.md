# Structured Disclosure Protocol Demo

- Date: `2026-03-21`
- Episodes: `3`
- Conformant after mediation: `2`
- Escalations required: `1`
- Rejected non-conformant messages: `2`
- Forwarded boundary violations: `0`

## clean_exchange
- Final status: `conformant_exchange`
- enforcer::accept_request
- enforcer::forward_response
  - Forwarded: ['bank_feed_12m', 'tax_filing_latest']

## boundary_violation_caught
- Final status: `conformant_exchange`
- enforcer::accept_request
- enforcer::reject_response
  - Errors: ["field_whitelist_violation:['client_names']", 'boundary_violation:client_names']
- enforcer::forward_response
  - Forwarded: ['bank_feed_12m', 'tax_filing_latest']

## missing_evidence_escalation
- Final status: `escalation_required`
- enforcer::accept_request
- enforcer::reject_response
  - Errors: ['evidence_source_nonconformant']

