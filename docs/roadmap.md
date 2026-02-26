# Loanville Roadmap

## Tier 3 Strategic Improvements

| Improvement | Status | Assessment | Action |
|---|---|---|---|
| Phased pipeline arrival | Implemented | Week cohorts now support configurable intra-week phases for pipeline pressure. | Added `arrival_phases` to season config/CLI and phase metadata to exports. |
| Deep-underwrite bandwidth limits | Implemented | Previously, lenders could effectively deep-underwrite every borrower each week. | Added `deep_uw_slots_per_week` cap and automatic deferral of excess approvals. |
| Borrower maturation visibility | Improved | Core payment progression existed, but exports/briefings had limited visibility into cumulative loan progress. | Added richer active-loan history and payment-progress details in reports/JSON. |
| Competitive refinancing | Not implemented | No dedicated refinance round exists yet in the season loop. | Next step: add post-origination refinance round with eligibility and transfer rules. |

