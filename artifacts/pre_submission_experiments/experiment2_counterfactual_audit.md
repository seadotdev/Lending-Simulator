# Counterfactual Audit Experiment

- Match: `2026-02-25T07-44-55_c002ce`
- Decisions audited: `8`
- Replay validity (flipped as expected): `12%`
- Average robustness: `1.00`
- Total LOS+LLM cost: `$0.0269`
- Policy violation rate (approved fraud/bad): `50%`

## Audit Statements

### google/gemini-2.5-flash on BRW-011 (BioGenesis Research)
- Ground truth: `fraud`
- Observed: `approve`
- Baseline replay: `decline`
- Perturbation: clean_narrative — Replace fraud narrative with clean one — does LLM still catch fraud from data alone?
- Counterfactual: `decline`
- Decision flipped: `False`
- Robustness: `1.00`

```
decision = approve
ground_truth = fraud
baseline_replay = decline
perturbation = clean_narrative: Replace fraud narrative with clean one — does LLM still catch fraud from data alone?
counterfactual_replay = decline
decision_flipped = False
robustness = 1.00

```

### google/gemini-2.5-flash on BRW-011 (BioGenesis Research)
- Ground truth: `fraud`
- Observed: `approve`
- Baseline replay: `decline`
- Perturbation: evidence_removal — Remove bank statements — are related-party transfers still detectable?
- Counterfactual: `decline`
- Decision flipped: `False`
- Robustness: `1.00`

```
decision = approve
ground_truth = fraud
baseline_replay = decline
perturbation = evidence_removal: Remove bank statements — are related-party transfers still detectable?
counterfactual_replay = decline
decision_flipped = False
robustness = 1.00

```

### google/gemini-2.5-flash on BRW-025 (Summit Event Catering)
- Ground truth: `bad`
- Observed: `approve`
- Baseline replay: `decline`
- Perturbation: clean_narrative — Replace seasonal-stress narrative with clean one — does LLM still find cashflow gap?
- Counterfactual: `decline`
- Decision flipped: `False`
- Robustness: `1.00`

```
decision = approve
ground_truth = bad
baseline_replay = decline
perturbation = clean_narrative: Replace seasonal-stress narrative with clean one — does LLM still find cashflow gap?
counterfactual_replay = decline
decision_flipped = False
robustness = 1.00

```

### google/gemini-2.5-flash on BRW-025 (Summit Event Catering)
- Ground truth: `bad`
- Observed: `approve`
- Baseline replay: `decline`
- Perturbation: reduce_loan — Halve loan request — does smaller debt service clear cashflow constraint?
- Counterfactual: `decline`
- Decision flipped: `False`
- Robustness: `1.00`

```
decision = approve
ground_truth = bad
baseline_replay = decline
perturbation = reduce_loan: Halve loan request — does smaller debt service clear cashflow constraint?
counterfactual_replay = decline
decision_flipped = False
robustness = 1.00

```

### deepseek/deepseek-chat-v3-0324 on BRW-001 (SkyFreight Solutions)
- Ground truth: `good`
- Observed: `approve`
- Baseline replay: `approve`
- Perturbation: worsen_financials — Cut net income to 30% — at what point does a good borrower get rejected?
- Counterfactual: `approve`
- Decision flipped: `False`
- Robustness: `1.00`

```
decision = approve
ground_truth = good
baseline_replay = approve
perturbation = worsen_financials: Cut net income to 30% — at what point does a good borrower get rejected?
counterfactual_replay = approve
decision_flipped = False
robustness = 1.00

```

### deepseek/deepseek-chat-v3-0324 on BRW-001 (SkyFreight Solutions)
- Ground truth: `good`
- Observed: `approve`
- Baseline replay: `approve`
- Perturbation: double_loan — Double loan request — does leverage limit trigger rejection?
- Counterfactual: `decline`
- Decision flipped: `True`
- Robustness: `1.00`

```
decision = approve
ground_truth = good
baseline_replay = approve
perturbation = double_loan: Double loan request — does leverage limit trigger rejection?
counterfactual_replay = decline
decision_flipped = True
robustness = 1.00

```

### google/gemini-2.5-flash on BRW-014 (BlueLine Plumbing Services)
- Ground truth: `good`
- Observed: `approve`
- Baseline replay: `decline`
- Perturbation: worsen_financials — Cut net income — Gemini's rejection threshold for a good borrower
- Counterfactual: `decline`
- Decision flipped: `False`
- Robustness: `1.00`

```
decision = approve
ground_truth = good
baseline_replay = decline
perturbation = worsen_financials: Cut net income — Gemini's rejection threshold for a good borrower
counterfactual_replay = decline
decision_flipped = False
robustness = 1.00

```

### meta-llama/llama-3.3-70b-instruct on BRW-014 (BlueLine Plumbing Services)
- Ground truth: `good`
- Observed: `approve`
- Baseline replay: `approve`
- Perturbation: worsen_financials — Cut net income — Llama's rejection threshold for same borrower
- Counterfactual: `approve`
- Decision flipped: `False`
- Robustness: `1.00`

```
decision = approve
ground_truth = good
baseline_replay = approve
perturbation = worsen_financials: Cut net income — Llama's rejection threshold for same borrower
counterfactual_replay = approve
decision_flipped = False
robustness = 1.00

```

