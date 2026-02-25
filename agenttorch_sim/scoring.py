"""
RAROC scoring for AgentTorch lending simulation.

Ports Loanville2's scoring.py to operate on trajectory tensors from the
AgentTorch runner. Computes per-lender RAROC scores, default rates,
concentration metrics, and overall portfolio performance.

Score = actual_return_pct - benchmark_pct (risk-free rate over horizon)

Where actual_return_pct accounts for:
  - Net P&L (interest + fees - losses - funding cost - workout cost)
  - Risk penalty (lambda * sigma * sqrt(n))
  - Volume penalty (quadratic for under-deployment)
  - Fraud penalty (extra charge on fraud principal)
  - Concentration penalty (excess sector exposure)
"""

from dataclasses import dataclass, field
import math
import torch


@dataclass
class TorchLenderScore:
    """Per-lender scoring output (tensor-derived)."""
    lender_idx: int
    total_capital: float
    deployed_capital: float
    available_capital: float
    # Revenue
    total_interest_earned: float
    total_fees_earned: float
    # Costs
    total_principal_lost: float
    total_recovery: float
    total_workout_cost: float
    total_funding_cost: float
    # Counts
    loans_booked: int
    defaults_count: int
    fraud_count: int
    # Rates
    default_rate: float
    deployment_ratio: float
    # RAROC components
    net_pnl: float
    risk_penalty: float
    volume_penalty: float
    fraud_penalty: float
    concentration_penalty: float
    # Final
    adjusted_pnl: float
    actual_return_pct: float
    benchmark_pct: float
    raroc_score: float


def score_lenders(
    final_state: dict,
    economics: "TorchEconomicsConfig | None" = None,
    sim_horizon_months: int = 24,
) -> list[TorchLenderScore]:
    """Compute RAROC scores from the final simulation state.

    Args:
        final_state: The state dict from runner.state_trajectory[-1][-1]
                     or from a standalone simulation's final state.
        economics: Economic configuration (defaults to balanced preset).
        sim_horizon_months: Simulation horizon for benchmark calculation.

    Returns:
        List of TorchLenderScore objects, one per lender.
    """
    from .economics import TorchEconomicsConfig
    eco = economics or TorchEconomicsConfig()

    # Extract lender state tensors
    lenders = final_state["agents"]["lenders"]
    total_capital = lenders["total_capital"]  # (M, 1)
    available_capital = lenders["available_capital"]  # (M, 1)
    total_interest = lenders["total_interest_earned"]  # (M, 1)
    total_fees = lenders["total_fees_earned"]  # (M, 1)
    principal_lost = lenders["total_principal_lost"]  # (M, 1)
    recovery = lenders["total_recovery"]  # (M, 1)
    workout_cost = lenders["total_workout_cost"]  # (M, 1)
    funding_cost = lenders["total_funding_cost"]  # (M, 1)
    loans_booked = lenders["loans_booked"]  # (M, 1)
    defaults_count = lenders["defaults_count"]  # (M, 1)
    fraud_count = lenders["fraud_count"]  # (M, 1)
    sector_exposure = lenders["sector_exposure"]  # (M, num_sectors)
    sector_limit = lenders["sector_limit"]  # (M, 1)

    M = total_capital.shape[0]
    scores = []

    for m in range(M):
        cap = total_capital[m].item()
        avail = available_capital[m].item()
        deployed = cap - avail
        interest = total_interest[m].item()
        fees = total_fees[m].item()
        lost = principal_lost[m].item()
        recov = recovery[m].item()
        workout = workout_cost[m].item()
        funding = funding_cost[m].item()
        n_loans = int(loans_booked[m].item())
        n_defaults = int(defaults_count[m].item())
        n_fraud = int(fraud_count[m].item())

        # Default rate
        default_rate = n_defaults / max(n_loans, 1)

        # Deployment ratio
        deployment_ratio = deployed / max(cap, 1.0)

        # --- Net P&L ---
        net_pnl = interest + fees - lost - workout - funding

        # --- Risk penalty (loss volatility) ---
        # Simplified: use realized loss ratio variance estimate
        # In full implementation, this would use per-loan profit std dev
        if n_loans > 1:
            loss_ratio = lost / max(deployed, 1.0)
            # Estimate volatility from default rate and loss given default
            lgd = lost / max(deployed * default_rate, 1.0) if default_rate > 0 else 0
            sigma = math.sqrt(default_rate * (1 - default_rate)) * lgd * deployed / n_loans
            risk_penalty = eco.risk_lambda * sigma * math.sqrt(n_loans)
        else:
            risk_penalty = 0.0

        # --- Volume penalty (quadratic for under-deployment) ---
        if deployment_ratio < eco.min_deployment_ratio:
            shortfall = eco.min_deployment_ratio - deployment_ratio
            volume_penalty = eco.volume_penalty_lambda * shortfall ** 2 * cap
        else:
            volume_penalty = 0.0

        # --- Fraud penalty ---
        fraud_penalty = n_fraud * eco.fraud_penalty_rate * (deployed / max(n_loans, 1))

        # --- Concentration penalty ---
        concentration_penalty = 0.0
        sec_lim = sector_limit[m].item()
        for s in range(sector_exposure.shape[1]):
            exp = sector_exposure[m, s].item()
            limit = cap * sec_lim
            excess = max(0.0, exp - limit)
            concentration_penalty += excess * eco.concentration_penalty_rate

        # --- Adjusted P&L ---
        adjusted_pnl = (
            net_pnl
            - risk_penalty
            - volume_penalty
            - fraud_penalty
            - concentration_penalty
        )

        # --- RAROC score ---
        actual_return_pct = (adjusted_pnl / max(cap, 1.0)) * 100
        benchmark_pct = eco.risk_free_rate * (sim_horizon_months / 12.0) * 100
        raroc_score = actual_return_pct - benchmark_pct

        scores.append(TorchLenderScore(
            lender_idx=m,
            total_capital=cap,
            deployed_capital=deployed,
            available_capital=avail,
            total_interest_earned=interest,
            total_fees_earned=fees,
            total_principal_lost=lost,
            total_recovery=recov,
            total_workout_cost=workout,
            total_funding_cost=funding,
            loans_booked=n_loans,
            defaults_count=n_defaults,
            fraud_count=n_fraud,
            default_rate=default_rate,
            deployment_ratio=deployment_ratio,
            net_pnl=net_pnl,
            risk_penalty=risk_penalty,
            volume_penalty=volume_penalty,
            fraud_penalty=fraud_penalty,
            concentration_penalty=concentration_penalty,
            adjusted_pnl=adjusted_pnl,
            actual_return_pct=actual_return_pct,
            benchmark_pct=benchmark_pct,
            raroc_score=raroc_score,
        ))

    return scores


def print_scores(scores: list[TorchLenderScore]) -> None:
    """Print a formatted summary of lender scores."""
    print("\n" + "=" * 80)
    print("RAROC SCORING SUMMARY (AgentTorch)")
    print("=" * 80)

    for s in sorted(scores, key=lambda x: -x.raroc_score):
        print(f"\n  Lender {s.lender_idx}:")
        print(f"    Capital:     ${s.total_capital:>12,.0f}  |  Deployed: ${s.deployed_capital:>12,.0f} ({s.deployment_ratio:.1%})")
        print(f"    Interest:    ${s.total_interest_earned:>12,.2f}  |  Fees: ${s.total_fees_earned:>12,.2f}")
        print(f"    Losses:      ${s.total_principal_lost:>12,.2f}  |  Recovery: ${s.total_recovery:>12,.2f}")
        print(f"    Funding:     ${s.total_funding_cost:>12,.2f}  |  Workout: ${s.total_workout_cost:>12,.2f}")
        print(f"    Loans:       {s.loans_booked:>12}    |  Defaults: {s.defaults_count} ({s.default_rate:.1%}) | Fraud: {s.fraud_count}")
        print(f"    ---")
        print(f"    Net P&L:     ${s.net_pnl:>12,.2f}")
        print(f"    Risk pen:    ${s.risk_penalty:>12,.2f}  |  Vol pen: ${s.volume_penalty:>12,.2f}")
        print(f"    Fraud pen:   ${s.fraud_penalty:>12,.2f}  |  Conc pen: ${s.concentration_penalty:>12,.2f}")
        print(f"    Adj P&L:     ${s.adjusted_pnl:>12,.2f}")
        print(f"    Return:      {s.actual_return_pct:>11.2f}%  |  Benchmark: {s.benchmark_pct:.2f}%")
        print(f"    RAROC SCORE: {s.raroc_score:>11.2f}%")

    print("\n" + "=" * 80)
    best = max(scores, key=lambda x: x.raroc_score)
    worst = min(scores, key=lambda x: x.raroc_score)
    avg = sum(s.raroc_score for s in scores) / len(scores)
    print(f"  Best:  Lender {best.lender_idx} ({best.raroc_score:+.2f}%)")
    print(f"  Worst: Lender {worst.lender_idx} ({worst.raroc_score:+.2f}%)")
    print(f"  Avg:   {avg:+.2f}%")
    total_defaults = sum(s.defaults_count for s in scores)
    total_loans = sum(s.loans_booked for s in scores)
    total_fraud = sum(s.fraud_count for s in scores)
    print(f"  Market: {total_loans} loans, {total_defaults} defaults ({total_defaults/max(total_loans,1):.1%}), {total_fraud} fraud")
    print("=" * 80)
