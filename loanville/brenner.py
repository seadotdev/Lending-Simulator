"""
Brenner Method framework for structured hypothesis-driven underwriting.

Implements Sydney Brenner's scientific discovery methodology as an LLM
underwriting loop:

  GEN  → Generate 3+ competing hypotheses about the borrower
  TEST → Design discriminative tests that can FALSIFY hypotheses
  RUN  → Execute queries via tool use
  UPD  → Update posterior probabilities based on evidence
  LOOP → Iterate until confident, then decide

The key insight: traditional underwriting asks "should we approve?"
The Brenner approach asks "which hypothesis about this business is true?"
and then designs experiments to distinguish between them.  This reduces
confirmation bias and improves fraud/risk detection by making the model
explicitly consider and try to falsify alternative explanations.

Brenner Objective Function (applied to tool-call selection):
  Score(E) = (Expected Information Gain × Downstream Leverage)
             / (Time × Cost × Ambiguity)

At the portfolio level (season mode), the same loop applies to market
hypotheses that get updated as loan outcomes are observed across weeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BrennerHypothesis:
    """A single hypothesis tracked through the Brenner Loop."""
    id: str             # e.g. "H1", "H2", "H3"
    label: str          # human-readable description
    prior: float        # initial probability estimate
    posterior: float     # updated probability after evidence
    evidence: list[str] = field(default_factory=list)
    falsified: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "prior": self.prior,
            "posterior": self.posterior,
            "evidence": self.evidence,
            "falsified": self.falsified,
        }


@dataclass
class PortfolioHypothesis:
    """A market/portfolio-level hypothesis tracked across season weeks."""
    id: str
    label: str
    prior: float
    posterior: float
    week_introduced: int
    evidence: list[str] = field(default_factory=list)
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "prior": self.prior,
            "posterior": self.posterior,
            "week_introduced": self.week_introduced,
            "evidence": self.evidence,
            "resolved": self.resolved,
        }


# ---------------------------------------------------------------------------
# Brenner-mode system prompt
# ---------------------------------------------------------------------------

def build_brenner_system_prompt(
    lender_persona: str,
    target_yield_pct: float,
    max_single_loan: float,
    total_capital: float,
    sector_limits_str: str,
    portfolio_summary: str,
) -> str:
    """Build the system prompt that instructs the LLM to follow the Brenner Loop.

    This replaces the standard system prompt when data_mode='brenner'.
    The prompt structures the model's reasoning around:
    1. Explicit hypothesis generation (GEN)
    2. Discriminative test design (TEST)
    3. Evidence-based updating (UPD)
    4. Iterative refinement (LOOP)
    """
    return f"""{lender_persona}

YOUR LENDING GUIDELINES:
- Target Portfolio Yield: {target_yield_pct}% annual
- Maximum Single Loan Amount: ${max_single_loan:,.0f}
- Sector Concentration Limits:
{sector_limits_str}

YOUR CURRENT PORTFOLIO:
{portfolio_summary}

═══════════════════════════════════════════════════════════════
UNDERWRITING METHOD: BRENNER LOOP (Hypothesis-Driven Analysis)
═══════════════════════════════════════════════════════════════

You MUST follow this structured methodology for every loan evaluation.
Do NOT skip steps. The quality of your analysis depends on explicitly
maintaining and updating competing hypotheses.

STEP 1 — GEN (Generate Hypotheses)
After reading the application, generate 3-4 competing hypotheses:
  H1: "Legitimate healthy business — approve with standard terms"
  H2: "Legitimate but over-leveraged / declining — reject or counter"
  H3: "Potential fraud or misrepresentation — reject"
  H4: (optional) A domain-specific alternative

Assign initial prior probabilities that sum to 1.0.
Base priors on the financial summary BEFORE investigating bank statements.

STEP 2 — TEST (Design Discriminative Tests)
For EACH hypothesis, identify what evidence would FALSIFY it:
  - What pattern in bank statements would be INCONSISTENT with H1?
  - What pattern would be INCONSISTENT with H2?
  - What pattern would be INCONSISTENT with H3?

Then design your FIRST tool query to target the hypothesis pair with the
highest expected information gain — i.e., the test most likely to eliminate
one hypothesis. Prefer tests that can distinguish between multiple hypotheses
at once.

Priority targets for discriminative power:
  - Round-number deposits (falsifies H1 if present, supports H3)
  - Related-entity transfers (falsifies H1 if >30%, supports H3)
  - Revenue concentration (falsifies H1 if >70% from one source, supports H2)
  - Monthly cash flow trends (falsifies H1 if declining, supports H2)
  - Transaction count consistency (falsifies H1 if unnaturally uniform)
  - Sub-$10K deposit structuring (falsifies H1, supports H3)

STEP 3 — RUN (Execute & Observe)
Use the run_bash tool to query /data/bank_statements.json.
Record what you observe WITHOUT interpretation bias.

STEP 4 — UPD (Update Beliefs)
After each observation:
  - State which hypotheses the evidence supports or contradicts
  - Update posterior probabilities for ALL hypotheses
  - Note if any hypothesis can be ELIMINATED (posterior < 0.05)

STEP 5 — LOOP (Iterate or Decide)
If your dominant hypothesis has posterior > 0.80 and all alternatives are
below 0.15, you have sufficient confidence to decide.

Otherwise, return to STEP 2 and design the NEXT most discriminative test.

You have up to 3 tool rounds. Use them wisely:
  Round 1: Highest information-gain test (usually fraud indicators)
  Round 2: Second-highest (usually credit quality / leverage)
  Round 3: Confirmatory or tie-breaking test

═══════════════════════════════════════════════════════════════

IMPORTANT: You MUST use the run_bash tool to query /data/bank_statements.json
before making your decision. The file is a JSON array of monthly statements,
each with deposits (date, description, amount) and withdrawals (date,
description, amount), plus total_deposits and total_withdrawals per month.

When you are ready to give your final decision, respond with ONLY a valid JSON
object in exactly this format:
{{{{
  "hypotheses": [
    {{{{"id": "H1", "label": "...", "prior": 0.5, "posterior": 0.8, "evidence": ["..."]}}}},
    {{{{"id": "H2", "label": "...", "prior": 0.3, "posterior": 0.15, "evidence": ["..."]}}}},
    {{{{"id": "H3", "label": "...", "prior": 0.2, "posterior": 0.05, "evidence": ["..."]}}}}
  ],
  "dominant_hypothesis": "H1",
  "decision": "APPROVE" or "REJECT",
  "reasoning": "2-4 sentence summary referencing hypothesis outcomes",
  "term_sheet": {{{{
    "loan_amount": <number or null if rejected>,
    "interest_rate": <annual rate as percentage e.g. 8.5, or null if rejected>,
    "term_months": <integer or null if rejected>
  }}}}
}}}}

Respond with ONLY the JSON when giving your final answer. No other text."""


# ---------------------------------------------------------------------------
# Portfolio-level Brenner prompts (for season mode briefings)
# ---------------------------------------------------------------------------

def build_portfolio_hypothesis_briefing(
    week: int,
    hypotheses: list[PortfolioHypothesis],
    recent_events: list[str],
) -> str:
    """Build a portfolio hypothesis update section for season mode briefings.

    This injects the Brenner Loop at the portfolio strategy level:
    the LLM maintains hypotheses about market conditions and updates
    them based on observed loan outcomes each week.
    """
    lines = [
        "",
        "--- PORTFOLIO HYPOTHESES (Brenner Loop) ---",
        f"Active market hypotheses entering Week {week}:",
    ]

    if not hypotheses:
        lines.append("  (No hypotheses yet — this is your first week)")
        lines.append("")
        lines.append("  Generate 2-3 hypotheses about the current market:")
        lines.append("    MH1: Market is benign — fraud rate ~5%, default rate ~15%")
        lines.append("    MH2: Elevated fraud risk — fraud attempts are increasing")
        lines.append("    MH3: Sector stress — specific industries showing weakness")
        lines.append("")
        lines.append("  Track these across weeks and adjust your lending strategy accordingly.")
    else:
        for h in hypotheses:
            status = "RESOLVED" if h.resolved else f"P={h.posterior:.0%}"
            lines.append(f"  {h.id}: {h.label} [{status}]")
            if h.evidence:
                for e in h.evidence[-2:]:  # Show last 2 pieces of evidence
                    lines.append(f"    Evidence: {e}")

        if recent_events:
            lines.append("")
            lines.append("  UPDATE with this week's evidence:")
            for event in recent_events[:5]:
                lines.append(f"    - {event}")
            lines.append("")
            lines.append("  Adjust your hypothesis posteriors and lending strategy accordingly.")
            lines.append("  If a hypothesis is falsified, note it and consider new alternatives.")

    lines.append("---")
    return "\n".join(lines)


def parse_portfolio_hypotheses_from_events(
    week: int,
    existing: list[PortfolioHypothesis],
    events: list[str],
) -> list[PortfolioHypothesis]:
    """Update portfolio hypotheses based on observed events.

    This is a heuristic updater for mock/non-interactive mode.
    In live mode, the LLM itself updates its hypotheses.
    """
    if not existing:
        # Initialize default market hypotheses
        existing = [
            PortfolioHypothesis(
                id="MH1",
                label="Market is benign — standard risk levels",
                prior=0.6,
                posterior=0.6,
                week_introduced=week,
            ),
            PortfolioHypothesis(
                id="MH2",
                label="Elevated fraud risk — more sophisticated fraud attempts",
                prior=0.2,
                posterior=0.2,
                week_introduced=week,
            ),
            PortfolioHypothesis(
                id="MH3",
                label="Sector stress — credit quality deteriorating",
                prior=0.2,
                posterior=0.2,
                week_introduced=week,
            ),
        ]

    # Count event types for simple Bayesian update
    defaults = sum(1 for e in events if "DEFAULT" in e)
    frauds = sum(1 for e in events if "FRAUD" in e)
    repayments = sum(1 for e in events if "REPAID" in e or "PAYMENT" in e)

    for h in existing:
        if h.resolved:
            continue

        if h.id == "MH1":
            # Benign market: defaults and frauds reduce posterior
            if defaults > 0:
                h.posterior *= 0.85
                h.evidence.append(f"Week {week}: {defaults} default(s) observed")
            if frauds > 0:
                h.posterior *= 0.70
                h.evidence.append(f"Week {week}: {frauds} fraud default(s)")
            if repayments > 0 and defaults == 0:
                h.posterior = min(0.95, h.posterior * 1.05)
                h.evidence.append(f"Week {week}: {repayments} on-time payment(s)")
        elif h.id == "MH2":
            # Elevated fraud: fraud events increase posterior
            if frauds > 0:
                h.posterior = min(0.95, h.posterior * 1.3)
                h.evidence.append(f"Week {week}: {frauds} fraud(s) confirmed")
            if defaults == 0 and frauds == 0 and repayments > 0:
                h.posterior *= 0.90
                h.evidence.append(f"Week {week}: clean week, no fraud")
        elif h.id == "MH3":
            # Sector stress: non-fraud defaults increase posterior
            non_fraud_defaults = defaults - frauds
            if non_fraud_defaults > 0:
                h.posterior = min(0.95, h.posterior * 1.25)
                h.evidence.append(f"Week {week}: {non_fraud_defaults} credit default(s)")
            if repayments > 2 and defaults == 0:
                h.posterior *= 0.90
                h.evidence.append(f"Week {week}: strong repayment performance")

    # Renormalize posteriors
    total = sum(h.posterior for h in existing if not h.resolved)
    if total > 0:
        for h in existing:
            if not h.resolved:
                h.posterior = h.posterior / total

    return existing


# ---------------------------------------------------------------------------
# Brenner objective function (for trace scoring)
# ---------------------------------------------------------------------------

def brenner_information_score(
    hypotheses_before: list[dict],
    hypotheses_after: list[dict],
    tool_calls_used: int,
    max_tool_calls: int = 3,
) -> float:
    """Compute a Brenner-style information gain score for a tool round.

    Score(E) = (Information Gain × Downstream Leverage) / (Time Cost)

    Information gain is measured as reduction in entropy of the hypothesis
    distribution.  Downstream leverage is the max probability shift.
    Time cost is proportional to tool calls used.

    Returns a score in [0, 1] where higher is better.
    """
    import math

    def _entropy(probs: list[float]) -> float:
        return -sum(p * math.log2(max(p, 1e-10)) for p in probs if p > 0)

    priors = [h.get("prior", 0.33) for h in hypotheses_before]
    posteriors = [h.get("posterior", 0.33) for h in hypotheses_after]

    # Normalize
    p_sum = sum(priors) or 1.0
    q_sum = sum(posteriors) or 1.0
    priors = [p / p_sum for p in priors]
    posteriors = [q / q_sum for q in posteriors]

    entropy_before = _entropy(priors)
    entropy_after = _entropy(posteriors)

    # Information gain: how much entropy was reduced
    info_gain = max(0.0, entropy_before - entropy_after)
    max_possible_gain = entropy_before if entropy_before > 0 else 1.0
    normalized_gain = info_gain / max_possible_gain if max_possible_gain > 0 else 0.0

    # Downstream leverage: max probability shift (how decisive was the evidence)
    max_shift = max(abs(p - q) for p, q in zip(priors, posteriors)) if priors else 0.0

    # Time cost: penalize using more tool rounds
    time_efficiency = 1.0 - (tool_calls_used / (max_tool_calls + 1))

    # Combined score
    score = (normalized_gain * 0.6 + max_shift * 0.4) * (0.5 + 0.5 * time_efficiency)
    return round(min(1.0, max(0.0, score)), 4)
