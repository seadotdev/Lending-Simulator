# Counterfactual Audit Experiment

- Underwriting path: `Open LOS /v1/deals/:dealId/evaluate` in `mode=full`
- LOS URL: `http://localhost:3200`
- Decisions audited: `5`
- SCM replay validity: `100%`
- Minimality rate: `60%`
- Average robustness: `0.87`
- Estimated OpenRouter cost: `$0.0539`

## google/gemini-2.5-flash :: BRW-001 :: SkyFreight Solutions
- Lender policy: `Heritage Trust Bank`
- Ground truth: `good`
- Baseline decision: `decline`
- Baseline summary: The company's sector, Aero-Logistics, is already over Heritage Trust Bank's concentration limit. Approving this loan would further exacerbate the bank's exposure to this sector. While the company shows good financial performance, the bank's strict portfolio guidelines prevent approval.
- Intervention: `clear_sector_exposure` — clear same-sector portfolio exposure before evaluation
- Counterfactual decision: `approve`
- Counterfactual summary: SkyFreight Solutions demonstrates consistent revenue growth and expanding net margins, indicating strong financial health. The company has a solid cash position and the requested loan amount is well within our limits. The Aero-Logistics sector exposure remains acceptable after this loan.
- Replay valid: `True`
- Minimality: `False`
- Robustness: `1.0`

## deepseek/deepseek-chat-v3-0324 :: BRW-014 :: BlueLine Plumbing Services
- Lender policy: `Meridian Partners`
- Ground truth: `good`
- Baseline decision: `approve`
- Baseline summary: BlueLine Plumbing Services demonstrates strong financial health with consistent revenue growth and stable margins above 30%. The company has a proven track record of 15 years, and the loan purpose aligns with expansion for a municipal contract, which is low-risk. The bank statements show no anomalies, and the business can comfortably service the debt from free cash flow. The loan fits within the Construction Services sector limit (25% of capital).
- Intervention: `increase_sector_exposure_to_limit` — increase same-sector portfolio exposure until the new loan breaches concentration policy
- Counterfactual decision: `decline`
- Counterfactual summary: BlueLine Plumbing Services operates in the Construction Services sector, where our current exposure is already at the 25% limit. Adding this loan would breach our sector concentration limits. Additionally, while the company demonstrates stable revenue and margins, the seasonal nature of the business introduces cash flow variability that could impact debt servicing.
- Replay valid: `True`
- Minimality: `True`
- Robustness: `0.67`

## deepseek/deepseek-chat-v3-0324 :: BRW-011 :: BioGenesis Research
- Lender policy: `Meridian Partners`
- Ground truth: `fraud`
- Baseline decision: `approve`
- Baseline summary: BioGenesis Research demonstrates consistent revenue and stable net margins (~10.8%) across quarters, indicating strong cash flow fundamentals. The bank statements show regular deposits from reputable clients like Genova Pharmaceuticals and LifeScience Direct, with no red flags for fraud. The loan fits within the Bio-Synthetics sector limit (current exposure: 18%, new loan would bring it to 31.8%, still under the 30% limit).
- Intervention: `increase_sector_exposure_to_limit` — increase same-sector portfolio exposure until the new loan breaches concentration policy
- Counterfactual decision: `decline`
- Counterfactual summary: The company's financials show stable revenue and margins, but the bank statements reveal heavy reliance on transfers from affiliated entities (BioGenesis Holdings LLC and BGH Capital Partners) rather than organic revenue. This raises concerns about the sustainability of cash flow and potential circular financing. Additionally, the requested loan would exceed the Bio-Synthetics sector concentration limit (30%).
- Replay valid: `True`
- Minimality: `False`
- Robustness: `0.67`

## meta-llama/llama-3.3-70b-instruct :: BRW-001 :: SkyFreight Solutions
- Lender policy: `Atlas Credit`
- Ground truth: `good`
- Baseline decision: `approve`
- Baseline summary: SkyFreight Solutions demonstrates a strong credit profile with growing revenue, stable margins, and a solid cash position, indicating its ability to service the debt from free cash flow. The loan amount of $500,000 is within our maximum single loan amount limit and does not breach our sector concentration limits for Aero-Logistics. We are offering a competitive interest rate of 9% and a term of 36 months.
- Intervention: `increase_sector_exposure_to_limit` — increase same-sector portfolio exposure until the new loan breaches concentration policy
- Counterfactual decision: `decline`
- Counterfactual summary: The loan application from SkyFreight Solutions would breach the sector concentration limit for Aero-Logistics, which is currently at its maximum of 25% of total capital. Approving this loan would exceed that limit.
- Replay valid: `True`
- Minimality: `True`
- Robustness: `1.0`

## meta-llama/llama-3.3-70b-instruct :: BRW-014 :: BlueLine Plumbing Services
- Lender policy: `Atlas Credit`
- Ground truth: `good`
- Baseline decision: `approve`
- Baseline summary: BlueLine Plumbing Services has a strong financial track record with consistent revenue growth and stable margins. The company's ability to service debt from free cash flow is evident, and the loan amount of $200,000 is reasonable for fleet expansion and inventory purposes. The sector concentration limit for Construction Services will not be breached with this loan. Therefore, the loan is approved with a competitive interest rate and term.
- Intervention: `increase_sector_exposure_to_limit` — increase same-sector portfolio exposure until the new loan breaches concentration policy
- Counterfactual decision: `decline`
- Counterfactual summary: The loan application from BlueLine Plumbing Services would breach the sector concentration limit for Construction Services, which is already at the maximum allowed exposure of 30%. The company's financials show stable revenue and margin trends, but the loan cannot be approved due to the portfolio fit issue.
- Replay valid: `True`
- Minimality: `True`
- Robustness: `1.0`

