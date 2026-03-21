# Counterfactual Audit Experiment

- Match: `2026-02-25T07-44-55_c002ce`
- Decisions audited: `5`
- SCM replay validity: `100%`
- Minimality rate: `100%`
- Average robustness: `0.81`
- Observed policy violation rate in the audited sample: `40%`

## Audit Statements

### google/gemini-2.5-flash on BRW-011 (BioGenesis Research)
- Decision: `conditional_offer`
- Ground truth: `fraud`
- Violated clause: `fraud_related_party_flag`
- Counterfactual: with protocol state `fraud_screened=true`, outcome changes to refuse
- Robustness: `1.0`

### google/gemini-2.5-flash on BRW-025 (Summit Event Catering)
- Decision: `conditional_offer`
- Ground truth: `bad`
- Violated clause: `cashflow_stress`
- Counterfactual: with protocol state `cashflow_reviewed=true`, outcome changes to refuse
- Robustness: `1.0`

### google/gemini-2.5-flash on BRW-001 (SkyFreight Solutions)
- Decision: `refuse`
- Ground truth: `good`
- Violated clause: `sector_concentration`
- Counterfactual: with existing Aero-Logistics exposure reduced from 1,100,000 to 500,000, outcome changes to conditional_offer
- Robustness: `0.54`

### deepseek/deepseek-chat-v3-0324 on BRW-025 (Summit Event Catering)
- Decision: `refuse`
- Ground truth: `bad`
- Violated clause: `cashflow_stress`
- Counterfactual: with an additional verified 56,486 per deficit month across months [0, 1, 2, 9, 10, 11], outcome changes to conditional_offer
- Robustness: `0.92`

### meta-llama/llama-3.3-70b-instruct on BRW-001 (SkyFreight Solutions)
- Decision: `conditional_offer`
- Ground truth: `good`
- Violated clause: `None`
- Counterfactual: with verified net income changed from 659,000 to 319,222, outcome changes to refuse
- Robustness: `0.6`

