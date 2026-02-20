"""
Mock LLM responses that simulate model-quality differences.

Bigger models:
  - Catch fraud signals (round numbers, circular transfers, fabricated consistency)
  - Detect bad business red flags (concentration, margin compression, decline)
  - Produce valid JSON reliably
  - Offer well-calibrated interest rates

Smaller models:
  - Miss some fraud signals
  - Miss subtle red flags (margin compression, grant dependency)
  - Sometimes produce malformed JSON
  - Offer poorly calibrated rates (too aggressive or too conservative)
"""

import hashlib
import json
import random
from .models import Borrower, LenderConfig, LenderDecision, TermSheet


# Model quality tiers mapped to behavior profiles
MODEL_TIERS = {
    # Tier 1: Large frontier models - excellent analysis
    "anthropic/claude-3.5-sonnet": "large",
    "anthropic/claude-sonnet-4": "large",
    "openai/gpt-4o": "large",
    "google/gemini-pro-1.5": "large",
    "meta-llama/llama-3.1-70b-instruct": "large",
    # Tier 2: Mid-size models - decent but miss subtleties
    "google/gemma-2-9b-it": "medium",
    "meta-llama/llama-3.1-8b-instruct": "medium",
    "mistralai/mistral-7b-instruct": "medium",
    "meta-llama/llama-3.2-3b-instruct": "small",
    # Tier 3: Small models - miss a lot
    "microsoft/phi-3-mini-128k-instruct": "small",
    "google/gemma-2-2b-it": "small",
    "qwen/qwen-2.5-3b-instruct": "small",
}


def _get_tier(model: str) -> str:
    """Look up model tier, default to medium."""
    return MODEL_TIERS.get(model, "medium")


def _deterministic_rand(lender_id: str, borrower_id: str, salt: str = "") -> float:
    """Deterministic random float [0,1) based on lender+borrower IDs."""
    h = hashlib.sha256(f"{lender_id}:{borrower_id}:{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _check_sector_exposure(lender: LenderConfig, borrower: Borrower) -> tuple[bool, float]:
    """Check if adding this loan would breach sector limits."""
    sector = borrower.dossier.sector
    current_exposure = sum(
        l.remaining_balance for l in lender.existing_portfolio if l.sector == sector
    )
    new_total = current_exposure + borrower.dossier.loan_request_amount
    limit = lender.sector_limits.get(sector, 0.25)
    actual_pct = new_total / lender.total_capital
    return actual_pct > limit, actual_pct


def _evaluate_mock(lender: LenderConfig, borrower: Borrower) -> LenderDecision:
    """Simulate an LLM evaluation with quality based on model tier."""
    tier = _get_tier(lender.model)
    bid = borrower.id
    lid = lender.id
    outcome = borrower.true_outcome
    sector = borrower.dossier.sector
    rand = _deterministic_rand(lid, bid)
    rand2 = _deterministic_rand(lid, bid, "rate")

    # === FRAUD DETECTION ===
    # Large models catch fraud ~95%, medium ~50%, small ~20%
    fraud_detection_rate = {"large": 0.95, "medium": 0.50, "small": 0.20}[tier]
    catches_fraud = rand < fraud_detection_rate

    if outcome == "fraud" and catches_fraud:
        reasons = {
            "BRW-010": "Suspicious round-number deposits detected across all 12 months. "
                       "Multiple deposits of exactly $50,000, $75,000, $100,000 suggest fabricated "
                       "bank statements. Revenue claims of $3.5M are inconsistent with deposit patterns.",
            "BRW-011": "Critical red flag: circular transfers between BioGenesis Research and "
                       "BioGenesis Holdings LLC / BGH Capital Partners. ~70% of deposits originate "
                       "from affiliated entities, indicating manufactured revenue.",
            "BRW-012": "Unnaturally consistent monthly figures across all 12 periods. Total deposits "
                       "vary by less than $400 month-to-month, which is statistically implausible "
                       "for a real operating business. Likely fabricated statements.",
        }
        return LenderDecision(
            lender_id=lid, borrower_id=bid, decision="REJECT",
            reasoning=reasons.get(bid, "Fraud indicators detected in financial statements."),
        )

    # === BAD BUSINESS DETECTION ===
    # Large models catch bad businesses ~85%, medium ~40%, small ~15%
    bad_detection_rate = {"large": 0.85, "medium": 0.40, "small": 0.15}[tier]
    catches_bad = _deterministic_rand(lid, bid, "bad") < bad_detection_rate

    if outcome == "bad" and catches_bad:
        reasons = {
            "BRW-006": "Dangerous customer concentration: approximately 85% of deposits come from "
                       "Titan Defense Corp and its divisions. Loss of this single client would be "
                       "catastrophic. Insufficient revenue diversification to support debt service.",
            "BRW-007": "Margin compression trend: expenses grew from $102K to $145K over 12 months "
                       "while revenue remained flat at ~$152K. At current trajectory, the business "
                       "will be cash-flow negative within 6 months. Cannot support new debt.",
            "BRW-008": "Revenue is almost entirely grant-dependent (DARPA, NSF, Quantum Horizons Fund). "
                       "No commercial revenue stream. Grant renewal is uncertain and cannot "
                       "reliably service debt obligations.",
            "BRW-009": "Clear declining revenue trend: deposits dropped from $182K to $100K over 12 "
                       "months (-45%). Expenses have not declined proportionally. Business is on a "
                       "trajectory toward negative cash flow.",
        }
        return LenderDecision(
            lender_id=lid, borrower_id=bid, decision="REJECT",
            reasoning=reasons.get(bid, "Fundamental business weaknesses detected."),
        )

    # === SECTOR CONCENTRATION CHECK ===
    over_limit, actual_pct = _check_sector_exposure(lender, borrower)
    # Large models always check limits, medium 70%, small 40%
    checks_limits = _deterministic_rand(lid, bid, "limits") < {"large": 1.0, "medium": 0.70, "small": 0.40}[tier]

    if over_limit and checks_limits:
        return LenderDecision(
            lender_id=lid, borrower_id=bid, decision="REJECT",
            reasoning=f"Sector concentration breach: {sector} exposure would reach "
                      f"{actual_pct*100:.1f}%, exceeding our {lender.sector_limits.get(sector, 0.25)*100:.0f}% limit. "
                      f"Must diversify portfolio.",
        )

    # === APPROVE WITH TERM SHEET ===
    # If we got here, the lender is going to approve

    # For fraud/bad that slipped through, the lender approves unknowingly
    loan_amount = borrower.dossier.loan_request_amount

    # Cap at lender's max
    if loan_amount > lender.max_single_loan:
        loan_amount = lender.max_single_loan

    # Interest rate calibration
    # Large models calibrate to risk; small models are noisier
    base_rate = lender.target_yield_pct
    if outcome == "good":
        # Good businesses - appropriate rate near target
        noise = {"large": 1.0, "medium": 2.5, "small": 4.0}[tier]
        rate = base_rate + (rand2 - 0.5) * noise
    elif outcome == "bad":
        # Bad business that slipped through - model doesn't see the risk
        # Small models may even offer lower rates
        rate = base_rate + (rand2 - 0.3) * 3.0
    else:
        # Fraud that slipped through
        rate = base_rate + (rand2 - 0.5) * 2.0

    rate = max(4.0, min(rate, 25.0))  # Clamp to reasonable range

    # Term months
    term_preferences = {
        "large": (18, 30),
        "medium": (12, 36),
        "small": (12, 36),
    }
    term_min, term_max = term_preferences[tier]
    term = term_min + int(rand2 * (term_max - term_min))
    term = max(12, min(term, 36))

    # Generate approval reasoning
    dossier = borrower.dossier
    monthly_fcf = (dossier.annual_revenue - dossier.annual_expenses) / 12
    dscr = monthly_fcf / (loan_amount / term) if loan_amount > 0 else 0

    if tier == "large":
        reasoning = (
            f"Strong application. Monthly free cash flow of ${monthly_fcf:,.0f} provides "
            f"a DSCR of {dscr:.1f}x on proposed payments. {dossier.years_in_business} years "
            f"in business with consistent revenue trends. Sector exposure within limits."
        )
    elif tier == "medium":
        reasoning = (
            f"Acceptable risk profile. Revenue of ${dossier.annual_revenue:,.0f} supports "
            f"the requested loan amount. Cash flow appears adequate for debt service."
        )
    else:
        reasoning = f"Business financials reviewed. Revenue and cash flow appear sufficient."

    return LenderDecision(
        lender_id=lid, borrower_id=bid, decision="APPROVE",
        reasoning=reasoning,
        term_sheet=TermSheet(
            loan_amount=round(loan_amount, 2),
            interest_rate=round(rate, 2),
            term_months=term,
        ),
    )


def mock_evaluate_all(
    lenders: list[LenderConfig],
    borrowers: list[Borrower],
) -> dict[str, list[LenderDecision]]:
    """Run mock evaluations for all lenders against all borrowers."""
    results = {}
    for lender in lenders:
        decisions = [_evaluate_mock(lender, b) for b in borrowers]
        results[lender.id] = decisions
    return results
