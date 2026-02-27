"""
Competitiveness metrics for Loanville season simulations.

Computes a suite of metrics after each simulation to quantify how
"interesting" and competitive a run was.  Flags runs with tight races,
Elo upsets, balanced underwriting, and unpredictable dynamics — filtering
out blowouts and degenerate runs.

The metrics are organized into five families:

  1. Score Parity    — how close were the final scores?
  2. Elo Dynamics    — volatility, upsets, lead changes in Elo trajectories
  3. Underwriting    — diversity and quality of approve/reject patterns
  4. Market Activity — deal flow, competition for borrowers, pricing spread
  5. Predictability  — BenchPress-inspired: can a simple model predict the
                       final standings from early-season data?  Lower
                       predictability = more interesting.

All metrics are pure functions of data already collected by SeasonEngine,
SeasonScore, and the leaderboard match records.  No new data collection
is needed.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CompetitivenessReport:
    """Full competitiveness analysis for a completed simulation."""

    # ---- 1. Score Parity ----
    score_gap: float = 0.0             # 1st - 2nd place final score gap
    score_range: float = 0.0           # max - min final score
    score_cv: float = 0.0              # coefficient of variation of final scores
    score_gini: float = 0.0            # Gini coefficient of final scores (0=equal, 1=dominated)
    pnl_gap: float = 0.0              # 1st - 2nd place net PnL gap
    pnl_range: float = 0.0            # max - min net PnL

    # ---- 2. Elo Dynamics ----
    elo_volatility: float = 0.0        # mean weekly Elo delta (absolute) across all lenders
    elo_max_swing: float = 0.0         # largest single-week Elo move by any lender
    lead_changes: int = 0              # number of times the Elo leader changed
    elo_upsets: int = 0                # weeks where a lower-rated lender beat a higher-rated one
    elo_convergence: float = 0.0       # final Elo range / initial range (>1=diverged, <1=converged)
    rank_inversions: int = 0           # total pairwise rank swaps across all weeks

    # ---- 3. Underwriting Quality ----
    decision_agreement: float = 0.0    # fraction of borrowers all lenders agreed on
    decision_entropy: float = 0.0      # avg Shannon entropy of approve/reject across lenders per borrower
    avg_approval_rate: float = 0.0     # mean approval rate across lenders
    approval_rate_spread: float = 0.0  # max - min approval rate
    fraud_detection_spread: float = 0.0  # max - min fraud detection rate

    # ---- 4. Market Activity ----
    deal_competition_rate: float = 0.0   # fraction of booked deals with multiple bidders
    avg_offers_per_borrower: float = 0.0 # mean offers per borrower
    pricing_spread: float = 0.0          # std dev of offered rates across all offers
    win_rate_balance: float = 0.0        # 1 - Gini of deal win counts (1=balanced, 0=monopoly)
    market_share_hhi: float = 0.0        # Herfindahl–Hirschman Index of deal wins

    # ---- 5. Predictability (BenchPress-inspired) ----
    midseason_rank_correlation: float = 0.0   # Spearman correlation: mid-season vs final ranks
    early_score_prediction_error: float = 0.0 # MedAPE: predict final from first-half scores
    outcome_entropy: float = 0.0              # Shannon entropy of final rank order
    upset_index: float = 0.0                  # fraction of pairwise matchups where the
                                               # lower-seeded lender finished higher

    # ---- Aggregate ----
    competitiveness_index: float = 0.0  # 0-100 composite score
    tags: list[str] = field(default_factory=list)  # ["tight_race", "elo_upset", "blowout", ...]

    def to_dict(self) -> dict:
        return {
            "score_parity": {
                "score_gap": round(self.score_gap, 2),
                "score_range": round(self.score_range, 2),
                "score_cv": round(self.score_cv, 4),
                "score_gini": round(self.score_gini, 4),
                "pnl_gap": round(self.pnl_gap, 2),
                "pnl_range": round(self.pnl_range, 2),
            },
            "elo_dynamics": {
                "elo_volatility": round(self.elo_volatility, 2),
                "elo_max_swing": round(self.elo_max_swing, 2),
                "lead_changes": self.lead_changes,
                "elo_upsets": self.elo_upsets,
                "elo_convergence": round(self.elo_convergence, 4),
                "rank_inversions": self.rank_inversions,
            },
            "underwriting": {
                "decision_agreement": round(self.decision_agreement, 4),
                "decision_entropy": round(self.decision_entropy, 4),
                "avg_approval_rate": round(self.avg_approval_rate, 4),
                "approval_rate_spread": round(self.approval_rate_spread, 4),
                "fraud_detection_spread": round(self.fraud_detection_spread, 4),
            },
            "market_activity": {
                "deal_competition_rate": round(self.deal_competition_rate, 4),
                "avg_offers_per_borrower": round(self.avg_offers_per_borrower, 2),
                "pricing_spread": round(self.pricing_spread, 4),
                "win_rate_balance": round(self.win_rate_balance, 4),
                "market_share_hhi": round(self.market_share_hhi, 4),
            },
            "predictability": {
                "midseason_rank_correlation": round(self.midseason_rank_correlation, 4),
                "early_score_prediction_error": round(self.early_score_prediction_error, 4),
                "outcome_entropy": round(self.outcome_entropy, 4),
                "upset_index": round(self.upset_index, 4),
            },
            "competitiveness_index": round(self.competitiveness_index, 1),
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(values: list[float]) -> float:
    """Gini coefficient for a list of non-negative values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def _shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy in bits for a probability distribution."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _hhi(shares: list[float]) -> float:
    """Herfindahl-Hirschman Index from market shares (0-1 each)."""
    return sum(s ** 2 for s in shares)


def _spearman_rank_correlation(ranks_a: list[int], ranks_b: list[int]) -> float:
    """Spearman rank correlation between two rank orderings."""
    n = len(ranks_a)
    if n < 2:
        return 1.0
    d_sq = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))


def _ranks_from_scores(scores: list[float]) -> list[int]:
    """Convert scores to 1-based ranks (higher score = rank 1)."""
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])
    ranks = [0] * len(scores)
    for rank, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = rank
    return ranks


def _count_inversions(ranks_a: list[int], ranks_b: list[int]) -> int:
    """Count pairwise rank inversions between two orderings."""
    n = len(ranks_a)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            # An inversion: i was ranked above j in A but below in B (or vice versa)
            if (ranks_a[i] < ranks_a[j]) != (ranks_b[i] < ranks_b[j]):
                inversions += 1
    return inversions


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_score_parity(
    final_scores: list[float],
    net_pnls: list[float],
) -> dict:
    """Compute score parity metrics from final season scores."""
    if len(final_scores) < 2:
        return {}

    sorted_scores = sorted(final_scores, reverse=True)
    score_gap = sorted_scores[0] - sorted_scores[1]
    score_range = sorted_scores[0] - sorted_scores[-1]

    mean = statistics.mean(final_scores)
    stdev = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
    score_cv = stdev / mean if mean > 0 else 0.0

    # Shift scores to non-negative for Gini
    min_score = min(final_scores)
    shifted = [s - min_score for s in final_scores]
    score_gini = _gini(shifted)

    sorted_pnl = sorted(net_pnls, reverse=True)
    pnl_gap = sorted_pnl[0] - sorted_pnl[1] if len(sorted_pnl) >= 2 else 0.0
    pnl_range = sorted_pnl[0] - sorted_pnl[-1]

    return {
        "score_gap": score_gap,
        "score_range": score_range,
        "score_cv": score_cv,
        "score_gini": score_gini,
        "pnl_gap": pnl_gap,
        "pnl_range": pnl_range,
    }


def compute_elo_dynamics(
    weekly_elo_snapshots: list[dict[str, float]],
    lender_ids: list[str],
) -> dict:
    """Compute Elo dynamics from weekly snapshots.

    weekly_elo_snapshots: list of {lender_id: composite_elo} per week.
    """
    n_weeks = len(weekly_elo_snapshots)
    n_lenders = len(lender_ids)

    if n_weeks < 2 or n_lenders < 2:
        return {
            "elo_volatility": 0.0,
            "elo_max_swing": 0.0,
            "lead_changes": 0,
            "elo_upsets": 0,
            "elo_convergence": 0.0,
            "rank_inversions": 0,
        }

    # Per-lender deltas
    all_deltas: list[float] = []
    max_swing = 0.0
    for lid in lender_ids:
        for w in range(1, n_weeks):
            prev = weekly_elo_snapshots[w - 1].get(lid, 1500.0)
            curr = weekly_elo_snapshots[w].get(lid, 1500.0)
            delta = abs(curr - prev)
            all_deltas.append(delta)
            max_swing = max(max_swing, delta)

    elo_volatility = statistics.mean(all_deltas) if all_deltas else 0.0

    # Lead changes
    leaders = []
    for snapshot in weekly_elo_snapshots:
        if snapshot:
            leader = max(snapshot, key=snapshot.get)
            leaders.append(leader)
    lead_changes = sum(
        1 for i in range(1, len(leaders)) if leaders[i] != leaders[i - 1]
    )

    # Elo upsets: count weeks where a previously lower-rated lender
    # gained more Elo than a higher-rated one
    elo_upsets = 0
    for w in range(1, n_weeks):
        prev = weekly_elo_snapshots[w - 1]
        curr = weekly_elo_snapshots[w]
        for i, lid_a in enumerate(lender_ids):
            for lid_b in lender_ids[i + 1:]:
                prev_a = prev.get(lid_a, 1500.0)
                prev_b = prev.get(lid_b, 1500.0)
                curr_a = curr.get(lid_a, 1500.0)
                curr_b = curr.get(lid_b, 1500.0)
                delta_a = curr_a - prev_a
                delta_b = curr_b - prev_b
                # Upset: the lower-rated lender gained more
                if prev_a > prev_b and delta_b > delta_a + 1.0:
                    elo_upsets += 1
                elif prev_b > prev_a and delta_a > delta_b + 1.0:
                    elo_upsets += 1

    # Convergence: final spread vs initial spread
    initial_range = max(weekly_elo_snapshots[0].values()) - min(weekly_elo_snapshots[0].values())
    final_range = max(weekly_elo_snapshots[-1].values()) - min(weekly_elo_snapshots[-1].values())
    elo_convergence = final_range / initial_range if initial_range > 0 else 1.0

    # Rank inversions across weeks
    total_inversions = 0
    prev_ranks = _ranks_from_scores([weekly_elo_snapshots[0].get(lid, 1500.0) for lid in lender_ids])
    for w in range(1, n_weeks):
        curr_ranks = _ranks_from_scores([weekly_elo_snapshots[w].get(lid, 1500.0) for lid in lender_ids])
        total_inversions += _count_inversions(prev_ranks, curr_ranks)
        prev_ranks = curr_ranks

    return {
        "elo_volatility": elo_volatility,
        "elo_max_swing": max_swing,
        "lead_changes": lead_changes,
        "elo_upsets": elo_upsets,
        "elo_convergence": elo_convergence,
        "rank_inversions": total_inversions,
    }


def compute_underwriting_metrics(
    all_decisions: dict[str, list],
    all_borrowers: list,
    lender_ids: list[str],
) -> dict:
    """Compute underwriting diversity and quality metrics."""
    if not lender_ids or not all_borrowers:
        return {
            "decision_agreement": 0.0,
            "decision_entropy": 0.0,
            "avg_approval_rate": 0.0,
            "approval_rate_spread": 0.0,
            "fraud_detection_spread": 0.0,
        }

    borrower_map = {b.id: b for b in all_borrowers}

    # Build per-borrower decision matrix: borrower_id -> {lender_id: "APPROVE"/"REJECT"/...}
    decision_matrix: dict[str, dict[str, str]] = {}
    for lid in lender_ids:
        for d in all_decisions.get(lid, []):
            if d.reasoning and d.reasoning.startswith("[LLM_ERROR]"):
                continue
            decision_matrix.setdefault(d.borrower_id, {})[lid] = d.decision

    # Agreement: fraction of borrowers where all lenders made the same decision
    agreed = 0
    total_decisions = 0
    entropies: list[float] = []
    for bid, decisions in decision_matrix.items():
        if len(decisions) < 2:
            continue
        total_decisions += 1

        # Count approve vs non-approve
        n_approve = sum(1 for d in decisions.values() if d == "APPROVE")
        n_total = len(decisions)
        if n_approve == 0 or n_approve == n_total:
            agreed += 1

        # Shannon entropy of this borrower's decisions
        p_approve = n_approve / n_total
        p_reject = 1.0 - p_approve
        probs = [p for p in [p_approve, p_reject] if p > 0]
        entropies.append(_shannon_entropy(probs))

    decision_agreement = agreed / total_decisions if total_decisions > 0 else 0.0
    decision_entropy = statistics.mean(entropies) if entropies else 0.0

    # Approval rates per lender
    approval_rates: list[float] = []
    fraud_detection_rates: list[float] = []
    for lid in lender_ids:
        decisions = all_decisions.get(lid, [])
        valid = [d for d in decisions if not (d.reasoning and d.reasoning.startswith("[LLM_ERROR]"))]
        if not valid:
            continue
        n_approve = sum(1 for d in valid if d.decision == "APPROVE")
        approval_rates.append(n_approve / len(valid))

        # Fraud detection: correctly rejected frauds
        fraud_borrowers = [d for d in valid if borrower_map.get(d.borrower_id) and
                          borrower_map[d.borrower_id].true_outcome == "fraud"]
        if fraud_borrowers:
            n_caught = sum(1 for d in fraud_borrowers if d.decision != "APPROVE")
            fraud_detection_rates.append(n_caught / len(fraud_borrowers))

    avg_approval_rate = statistics.mean(approval_rates) if approval_rates else 0.0
    approval_rate_spread = (max(approval_rates) - min(approval_rates)) if len(approval_rates) >= 2 else 0.0

    fraud_detection_spread = 0.0
    if len(fraud_detection_rates) >= 2:
        fraud_detection_spread = max(fraud_detection_rates) - min(fraud_detection_rates)

    return {
        "decision_agreement": decision_agreement,
        "decision_entropy": decision_entropy,
        "avg_approval_rate": avg_approval_rate,
        "approval_rate_spread": approval_rate_spread,
        "fraud_detection_spread": fraud_detection_spread,
    }


def compute_market_activity(
    all_deal_results: dict[str, dict],
    all_decisions: dict[str, list],
    lender_ids: list[str],
) -> dict:
    """Compute market activity metrics from deal results."""
    if not all_deal_results or not lender_ids:
        return {
            "deal_competition_rate": 0.0,
            "avg_offers_per_borrower": 0.0,
            "pricing_spread": 0.0,
            "win_rate_balance": 0.0,
            "market_share_hhi": 0.0,
        }

    # Deal competition: fraction of booked deals with multiple offers
    booked = {bid: d for bid, d in all_deal_results.items() if d.get("outcome") == "booked"}
    competitive = sum(1 for d in booked.values() if d.get("competitive", False))
    deal_competition_rate = competitive / len(booked) if booked else 0.0

    # Average offers per borrower (approvals, not just winners)
    offers_per_borrower: dict[str, int] = {}
    all_rates: list[float] = []
    for lid in lender_ids:
        for d in all_decisions.get(lid, []):
            if d.decision == "APPROVE" and d.term_sheet:
                offers_per_borrower[d.borrower_id] = offers_per_borrower.get(d.borrower_id, 0) + 1
                all_rates.append(d.term_sheet.interest_rate)

    avg_offers = statistics.mean(offers_per_borrower.values()) if offers_per_borrower else 0.0
    pricing_spread = statistics.stdev(all_rates) if len(all_rates) > 1 else 0.0

    # Win rate balance: how evenly distributed are deal wins?
    win_counts = {lid: 0 for lid in lender_ids}
    for deal in booked.values():
        winner = deal.get("winner")
        if winner and winner in win_counts:
            win_counts[winner] += 1

    counts = list(win_counts.values())
    win_rate_balance = 1.0 - _gini(counts) if sum(counts) > 0 else 0.0

    # HHI: market concentration
    total_wins = sum(counts)
    if total_wins > 0:
        shares = [c / total_wins for c in counts]
        market_share_hhi = _hhi(shares)
    else:
        market_share_hhi = 1.0 / len(lender_ids) if lender_ids else 1.0

    return {
        "deal_competition_rate": deal_competition_rate,
        "avg_offers_per_borrower": avg_offers,
        "pricing_spread": pricing_spread,
        "win_rate_balance": win_rate_balance,
        "market_share_hhi": market_share_hhi,
    }


def compute_predictability(
    weekly_scores: list[dict[str, float]],
    final_scores: dict[str, float],
    lender_ids: list[str],
) -> dict:
    """BenchPress-inspired predictability metrics.

    Lower predictability = more interesting run (harder to call the winner
    from early data, analogous to BenchPress's benchmark predictability
    hierarchy).

    weekly_scores: list of {lender_id: cumulative_score} at each week.
    final_scores: {lender_id: final_season_score}.
    """
    n_weeks = len(weekly_scores)
    n_lenders = len(lender_ids)

    if n_weeks < 2 or n_lenders < 2:
        return {
            "midseason_rank_correlation": 1.0,
            "early_score_prediction_error": 0.0,
            "outcome_entropy": 0.0,
            "upset_index": 0.0,
        }

    # Mid-season rank correlation (Spearman)
    mid_week = n_weeks // 2
    mid_scores = [weekly_scores[mid_week].get(lid, 0.0) for lid in lender_ids]
    final = [final_scores.get(lid, 0.0) for lid in lender_ids]

    mid_ranks = _ranks_from_scores(mid_scores)
    final_ranks = _ranks_from_scores(final)
    midseason_rank_corr = _spearman_rank_correlation(mid_ranks, final_ranks)

    # Early-score prediction error (MedAPE inspired by BenchPress)
    # Use first-half cumulative scores to "predict" final score, measure error
    first_half_scores = [weekly_scores[mid_week].get(lid, 0.0) for lid in lender_ids]
    apes: list[float] = []
    for i, lid in enumerate(lender_ids):
        actual = final_scores.get(lid, 0.0)
        predicted = first_half_scores[i]
        if abs(actual) > 0.01:
            ape = abs(predicted - actual) / abs(actual) * 100
            apes.append(ape)

    # Median APE (BenchPress's primary metric)
    early_prediction_error = statistics.median(apes) if apes else 0.0

    # Outcome entropy: how surprising is the final ranking?
    # Use the distribution of final score gaps — more uniform = more entropy
    if len(final) > 1:
        score_diffs = []
        total = sum(abs(s) for s in final) or 1.0
        shares = [abs(s) / total for s in final]
        outcome_entropy = _shannon_entropy(shares)
    else:
        outcome_entropy = 0.0

    # Upset index: fraction of pairwise comparisons where the lender
    # ranked lower at mid-season finished higher
    n_pairs = 0
    upsets = 0
    for i in range(n_lenders):
        for j in range(i + 1, n_lenders):
            n_pairs += 1
            mid_i = mid_scores[i]
            mid_j = mid_scores[j]
            final_i = final[i]
            final_j = final[j]
            # Upset: whoever was lower at midpoint ended up higher
            if (mid_i > mid_j and final_j > final_i) or (mid_j > mid_i and final_i > final_j):
                upsets += 1

    upset_index = upsets / n_pairs if n_pairs > 0 else 0.0

    return {
        "midseason_rank_correlation": midseason_rank_corr,
        "early_score_prediction_error": early_prediction_error,
        "outcome_entropy": outcome_entropy,
        "upset_index": upset_index,
    }


# ---------------------------------------------------------------------------
# Composite index and tagging
# ---------------------------------------------------------------------------

def _compute_competitiveness_index(report: CompetitivenessReport, n_lenders: int) -> float:
    """Compute a 0-100 composite competitiveness index.

    Higher = more competitive and interesting.  The index rewards:
    - Tight score gaps (high parity)
    - Elo lead changes and upsets (high drama)
    - Decision disagreement (diverse strategies)
    - High deal competition (active market)
    - Low predictability (surprising outcomes)
    """
    components: list[tuple[float, float]] = []  # (score 0-1, weight)

    # 1. Score parity (25%): tighter gap = more competitive
    # score_gap of 0 = perfect (1.0), gap of 20+ = boring (0.0)
    parity = max(0.0, 1.0 - report.score_gap / 20.0)
    components.append((parity, 0.25))

    # 2. Elo dynamics (25%): more lead changes and upsets = more interesting
    max_possible_lead_changes = max(1, n_lenders)  # crude upper bound
    drama = 0.0
    drama += min(1.0, report.lead_changes / max(1, max_possible_lead_changes)) * 0.4
    drama += min(1.0, report.elo_upsets / max(1, 5)) * 0.3
    drama += min(1.0, report.rank_inversions / max(1, 10)) * 0.3
    components.append((drama, 0.25))

    # 3. Strategic diversity (20%): different strategies = more interesting
    diversity = 0.0
    # High decision entropy = lenders disagree more = diverse approaches
    diversity += min(1.0, report.decision_entropy) * 0.5
    # Spread in approval rates = different risk appetites
    diversity += min(1.0, report.approval_rate_spread / 0.3) * 0.3
    # Spread in fraud detection = different skills
    diversity += min(1.0, report.fraud_detection_spread / 0.3) * 0.2
    components.append((diversity, 0.20))

    # 4. Market competitiveness (15%): active bidding
    market = 0.0
    market += report.deal_competition_rate * 0.4
    market += report.win_rate_balance * 0.4
    market += min(1.0, report.pricing_spread / 5.0) * 0.2
    components.append((market, 0.15))

    # 5. Unpredictability (15%): harder to predict = more interesting
    # Low midseason correlation = unpredictable
    unpredictability = max(0.0, 1.0 - report.midseason_rank_correlation)
    # Plus upset index
    unpredictability = 0.5 * unpredictability + 0.5 * report.upset_index
    components.append((unpredictability, 0.15))

    # Weighted sum, scaled to 0-100
    index = sum(score * weight for score, weight in components)
    return round(index * 100, 1)


def _assign_tags(report: CompetitivenessReport) -> list[str]:
    """Assign human-readable tags based on metric thresholds."""
    tags: list[str] = []

    # Score parity
    if report.score_gap < 3.0:
        tags.append("photo_finish")
    elif report.score_gap < 8.0:
        tags.append("tight_race")
    elif report.score_gap > 25.0:
        tags.append("blowout")

    if report.score_gini > 0.3:
        tags.append("one_sided")
    elif report.score_gini < 0.08:
        tags.append("highly_balanced")

    # Elo dynamics
    if report.lead_changes >= 3:
        tags.append("lead_changes")
    if report.elo_upsets >= 3:
        tags.append("elo_upsets")
    if report.elo_max_swing > 50:
        tags.append("volatile_elo")
    if report.elo_convergence < 0.5:
        tags.append("elo_converged")
    elif report.elo_convergence > 2.0:
        tags.append("elo_diverged")

    # Underwriting
    if report.decision_agreement > 0.8:
        tags.append("consensus_decisions")
    elif report.decision_agreement < 0.3:
        tags.append("strategic_diversity")
    if report.approval_rate_spread > 0.25:
        tags.append("mixed_risk_appetite")

    # Market
    if report.deal_competition_rate > 0.7:
        tags.append("competitive_market")
    elif report.deal_competition_rate < 0.2:
        tags.append("weak_competition")
    if report.market_share_hhi > 0.5:
        tags.append("market_concentration")
    elif report.market_share_hhi < 0.3:
        tags.append("balanced_market_share")

    # Predictability
    if report.midseason_rank_correlation < 0.3:
        tags.append("unpredictable")
    elif report.midseason_rank_correlation > 0.9:
        tags.append("predictable")
    if report.upset_index > 0.4:
        tags.append("upset_heavy")

    return tags


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_competitiveness(
    season_scores: list,
    season_engine,
    weekly_elo_snapshots: Optional[list[dict[str, float]]] = None,
    weekly_pnl_snapshots: Optional[list[dict[str, float]]] = None,
) -> CompetitivenessReport:
    """Run full competitiveness analysis on a completed season.

    Args:
        season_scores: list of SeasonScore from score_season()
        season_engine: SeasonEngine instance with all accumulated state
        weekly_elo_snapshots: optional list of {lender_id: composite_elo} per week
            (built from leaderboard Elo replay if available)
        weekly_pnl_snapshots: optional list of {lender_id: cumulative_pnl} per week

    Returns:
        CompetitivenessReport with all metrics computed.
    """
    report = CompetitivenessReport()

    if not season_scores or len(season_scores) < 2:
        report.tags = ["insufficient_data"]
        return report

    lender_ids = [s.lender_id for s in season_scores]
    final_scores = [s.final_score for s in season_scores]
    net_pnls = [s.net_pnl for s in season_scores]

    # 1. Score parity
    parity = compute_score_parity(final_scores, net_pnls)
    report.score_gap = parity.get("score_gap", 0.0)
    report.score_range = parity.get("score_range", 0.0)
    report.score_cv = parity.get("score_cv", 0.0)
    report.score_gini = parity.get("score_gini", 0.0)
    report.pnl_gap = parity.get("pnl_gap", 0.0)
    report.pnl_range = parity.get("pnl_range", 0.0)

    # 2. Elo dynamics
    if weekly_elo_snapshots and len(weekly_elo_snapshots) >= 2:
        elo = compute_elo_dynamics(weekly_elo_snapshots, lender_ids)
        report.elo_volatility = elo["elo_volatility"]
        report.elo_max_swing = elo["elo_max_swing"]
        report.lead_changes = elo["lead_changes"]
        report.elo_upsets = elo["elo_upsets"]
        report.elo_convergence = elo["elo_convergence"]
        report.rank_inversions = elo["rank_inversions"]
    else:
        # Build approximate Elo snapshots from weekly snapshots on lender states
        _elo_from_snapshots = _build_elo_proxy_from_engine(season_engine, lender_ids)
        if len(_elo_from_snapshots) >= 2:
            elo = compute_elo_dynamics(_elo_from_snapshots, lender_ids)
            report.elo_volatility = elo["elo_volatility"]
            report.elo_max_swing = elo["elo_max_swing"]
            report.lead_changes = elo["lead_changes"]
            report.elo_upsets = elo["elo_upsets"]
            report.elo_convergence = elo["elo_convergence"]
            report.rank_inversions = elo["rank_inversions"]

    # 3. Underwriting
    uw = compute_underwriting_metrics(
        season_engine.all_decisions,
        season_engine.all_borrowers,
        lender_ids,
    )
    report.decision_agreement = uw["decision_agreement"]
    report.decision_entropy = uw["decision_entropy"]
    report.avg_approval_rate = uw["avg_approval_rate"]
    report.approval_rate_spread = uw["approval_rate_spread"]
    report.fraud_detection_spread = uw["fraud_detection_spread"]

    # 4. Market activity
    market = compute_market_activity(
        season_engine.all_deal_results,
        season_engine.all_decisions,
        lender_ids,
    )
    report.deal_competition_rate = market["deal_competition_rate"]
    report.avg_offers_per_borrower = market["avg_offers_per_borrower"]
    report.pricing_spread = market["pricing_spread"]
    report.win_rate_balance = market["win_rate_balance"]
    report.market_share_hhi = market["market_share_hhi"]

    # 5. Predictability
    weekly_score_snapshots = weekly_pnl_snapshots
    if not weekly_score_snapshots:
        weekly_score_snapshots = _build_weekly_score_proxy(season_engine, lender_ids)

    final_score_map = {s.lender_id: s.final_score for s in season_scores}
    pred = compute_predictability(
        weekly_score_snapshots,
        final_score_map,
        lender_ids,
    )
    report.midseason_rank_correlation = pred["midseason_rank_correlation"]
    report.early_score_prediction_error = pred["early_score_prediction_error"]
    report.outcome_entropy = pred["outcome_entropy"]
    report.upset_index = pred["upset_index"]

    # Aggregate
    report.competitiveness_index = _compute_competitiveness_index(report, len(lender_ids))
    report.tags = _assign_tags(report)

    return report


# ---------------------------------------------------------------------------
# Engine data extraction helpers
# ---------------------------------------------------------------------------

def _build_elo_proxy_from_engine(season_engine, lender_ids: list[str]) -> list[dict[str, float]]:
    """Build approximate weekly Elo-like scores from SeasonLenderState snapshots.

    Uses cumulative PnL as a proxy ranking signal when actual Elo history
    is not available (i.e., when the leaderboard wasn't used).
    """
    snapshots: list[dict[str, float]] = []
    for state in season_engine.lender_states.values():
        n_snapshots = len(state.weekly_snapshots)
        break
    else:
        return snapshots

    for week_idx in range(n_snapshots):
        week_snap: dict[str, float] = {}
        for lid in lender_ids:
            state = season_engine.lender_states.get(lid)
            if state and week_idx < len(state.weekly_snapshots):
                snap = state.weekly_snapshots[week_idx]
                # Use cumulative PnL as proxy Elo
                pnl = (
                    snap.get("cumulative_interest", 0.0)
                    + snap.get("cumulative_fees", 0.0)
                    - snap.get("cumulative_losses", 0.0)
                    - snap.get("cumulative_workout_cost", 0.0)
                )
                # Scale to Elo-like range: 1500 + PnL/100
                week_snap[lid] = 1500.0 + pnl / 100.0
            else:
                week_snap[lid] = 1500.0
        snapshots.append(week_snap)

    return snapshots


def _build_weekly_score_proxy(season_engine, lender_ids: list[str]) -> list[dict[str, float]]:
    """Build weekly cumulative score snapshots from lender state snapshots."""
    snapshots: list[dict[str, float]] = []
    for state in season_engine.lender_states.values():
        n_snapshots = len(state.weekly_snapshots)
        break
    else:
        return snapshots

    for week_idx in range(n_snapshots):
        week_snap: dict[str, float] = {}
        for lid in lender_ids:
            state = season_engine.lender_states.get(lid)
            if state and week_idx < len(state.weekly_snapshots):
                snap = state.weekly_snapshots[week_idx]
                pnl = (
                    snap.get("cumulative_interest", 0.0)
                    + snap.get("cumulative_fees", 0.0)
                    - snap.get("cumulative_losses", 0.0)
                    - snap.get("cumulative_workout_cost", 0.0)
                )
                week_snap[lid] = pnl
            else:
                week_snap[lid] = 0.0
        snapshots.append(week_snap)

    return snapshots


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_competitiveness_report(report: CompetitivenessReport) -> None:
    """Print a formatted competitiveness report."""
    print(f"\n{'=' * 70}")
    print("  COMPETITIVENESS ANALYSIS")
    print(f"{'=' * 70}")

    # Tags
    if report.tags:
        tag_str = ", ".join(report.tags)
        print(f"  Tags: {tag_str}")

    # Composite
    idx = report.competitiveness_index
    if idx >= 70:
        quality = "HIGHLY COMPETITIVE"
    elif idx >= 45:
        quality = "COMPETITIVE"
    elif idx >= 25:
        quality = "MODERATE"
    else:
        quality = "LOW INTEREST"
    print(f"  Competitiveness Index: {idx:.1f}/100 ({quality})")

    print(f"\n  {'─' * 50}")
    print("  Score Parity")
    print(f"    1st-2nd gap:    {report.score_gap:.2f} pts")
    print(f"    Score range:    {report.score_range:.2f} pts")
    print(f"    Score CV:       {report.score_cv:.3f}")
    print(f"    Gini:           {report.score_gini:.3f}")
    print(f"    PnL gap:        ${report.pnl_gap:,.0f}")

    print(f"\n  {'─' * 50}")
    print("  Elo Dynamics")
    print(f"    Avg volatility: {report.elo_volatility:.1f} pts/week")
    print(f"    Max swing:      {report.elo_max_swing:.1f} pts")
    print(f"    Lead changes:   {report.lead_changes}")
    print(f"    Upsets:         {report.elo_upsets}")
    print(f"    Rank inversions:{report.rank_inversions}")
    conv = "converged" if report.elo_convergence < 1.0 else "diverged"
    print(f"    Convergence:    {report.elo_convergence:.2f}x ({conv})")

    print(f"\n  {'─' * 50}")
    print("  Underwriting Diversity")
    print(f"    Agreement:      {report.decision_agreement:.1%}")
    print(f"    Entropy:        {report.decision_entropy:.3f} bits")
    print(f"    Avg approval:   {report.avg_approval_rate:.1%}")
    print(f"    Approval spread:{report.approval_rate_spread:.1%}")
    print(f"    Fraud det spread:{report.fraud_detection_spread:.1%}")

    print(f"\n  {'─' * 50}")
    print("  Market Activity")
    print(f"    Competition:    {report.deal_competition_rate:.1%}")
    print(f"    Offers/borrower:{report.avg_offers_per_borrower:.1f}")
    print(f"    Rate spread:    {report.pricing_spread:.2f}%")
    print(f"    Win balance:    {report.win_rate_balance:.3f}")
    print(f"    HHI:            {report.market_share_hhi:.3f}")

    print(f"\n  {'─' * 50}")
    print("  Predictability")
    print(f"    Mid-season corr:{report.midseason_rank_correlation:.3f}")
    print(f"    MedAPE:         {report.early_score_prediction_error:.1f}%")
    print(f"    Outcome entropy:{report.outcome_entropy:.3f} bits")
    print(f"    Upset index:    {report.upset_index:.3f}")

    print(f"{'=' * 70}")
