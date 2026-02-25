"""
CLI entry point for the AgentTorch lending market simulation.

Supports two execution modes:

1. Standalone mode (default): Runs the simulation using PyTorch tensors directly,
   without requiring the agent-torch package. All substep modules are used as
   standard nn.Modules. This is the fastest path for development and testing.

2. AgentTorch mode (--agenttorch): Uses the full AgentTorch Runner/Controller
   infrastructure. Requires agent-torch to be installed.

Usage:
    # Standalone mode (no agent-torch required)
    python -m agenttorch_sim.run --num-borrowers 10000 --num-lenders 10 --months 24

    # With Loanville2 static borrowers (for validation)
    python -m agenttorch_sim.run --from-loanville --mix balanced

    # AgentTorch mode
    python -m agenttorch_sim.run --agenttorch --num-borrowers 100000 --device cuda
"""

import argparse
import time
import sys
import os
from pathlib import Path

import torch

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agenttorch_sim.population.generate import (
    from_loanville,
    generate_population,
    generate_lender_tensors,
    SECTOR_INDEX,
)
from agenttorch_sim.economics import TorchEconomicsConfig, TORCH_ECONOMICS_PRESETS
from agenttorch_sim.scoring import score_lenders, print_scores

# Import substep modules (triggers registration)
from agenttorch_sim.substeps.application import GenerateDemand, MarkApplicants
from agenttorch_sim.substeps.underwriting import CreditEvaluate, StoreEvaluations
from agenttorch_sim.substeps.adjudication import CompetitiveAuction
from agenttorch_sim.substeps.repayment import MonthlyAmortization
from agenttorch_sim.substeps.defaults import DefaultProcess
from agenttorch_sim.substeps.market_update import MarketDynamics


def build_initial_state(
    borrower_tensors: dict[str, torch.Tensor],
    lender_tensors: dict[str, torch.Tensor],
    economics: TorchEconomicsConfig,
    num_sectors: int = 15,
    device: str = "cpu",
) -> dict:
    """Construct the nested state dict from population tensors."""
    N = borrower_tensors["sector"].shape[0]
    M = lender_tensors["total_capital"].shape[0]

    dev = torch.device(device)

    def _to(t):
        return t.to(dev) if isinstance(t, torch.Tensor) else t

    env_overrides = economics.to_env_overrides()

    state = {
        "environment": {
            "funding_rate": torch.tensor([env_overrides["funding_rate"]], device=dev),
            "risk_free_rate": torch.tensor([env_overrides["risk_free_rate"]], device=dev),
            "recovery_rate_bad": torch.tensor([env_overrides["recovery_rate_bad"]], device=dev),
            "recovery_rate_fraud": torch.tensor([env_overrides["recovery_rate_fraud"]], device=dev),
            "origination_fee_rate": torch.tensor([env_overrides["origination_fee_rate"]], device=dev),
            "workout_cost_rate": torch.tensor([env_overrides["workout_cost_rate"]], device=dev),
            "fraud_penalty_rate": torch.tensor([env_overrides["fraud_penalty_rate"]], device=dev),
            "servicing_cost_rate": torch.tensor([env_overrides["servicing_cost_rate"]], device=dev),
            "current_month": torch.tensor([0], dtype=torch.long, device=dev),
            "market_cycle_phase": torch.tensor([0.0], device=dev),
            "total_originations": torch.tensor([0.0], device=dev),
            "total_defaults_this_step": torch.tensor([0.0], device=dev),
        },
        "agents": {
            "borrowers": {
                # Static financials
                "sector": _to(borrower_tensors["sector"]),
                "annual_revenue": _to(borrower_tensors["annual_revenue"]),
                "net_income": _to(borrower_tensors["net_income"]),
                "net_margin": _to(borrower_tensors["net_margin"]),
                "loan_request": _to(borrower_tensors["loan_request"]),
                "years_in_business": _to(borrower_tensors["years_in_business"]),
                # Risk parameters
                "default_prob": _to(borrower_tensors["default_prob"]),
                "months_to_default": _to(borrower_tensors["months_to_default"]),
                "fraud_prob": _to(borrower_tensors["fraud_prob"]),
                # Dynamic loan state
                "outstanding_debt": torch.zeros(N, 1, device=dev),
                "monthly_payment": torch.zeros(N, 1, device=dev),
                "months_remaining": torch.zeros(N, 1, device=dev),
                "loan_status": torch.zeros(N, 1, device=dev),
                "assigned_lender": torch.full((N, 1), -1.0, device=dev),
                "interest_rate": torch.zeros(N, 1, device=dev),
                "loan_origination_month": torch.full((N, 1), -1.0, device=dev),
            },
            "lenders": {
                "total_capital": _to(lender_tensors["total_capital"]),
                "available_capital": _to(lender_tensors["available_capital"]),
                "target_yield": _to(lender_tensors["target_yield"]),
                "risk_tolerance": _to(lender_tensors["risk_tolerance"]),
                "max_single_loan": _to(lender_tensors["max_single_loan"]),
                "sector_exposure": torch.zeros(M, num_sectors, device=dev),
                "sector_limit": torch.full((M, 1), 0.25, device=dev),
                "total_interest_earned": torch.zeros(M, 1, device=dev),
                "total_principal_lost": torch.zeros(M, 1, device=dev),
                "total_fees_earned": torch.zeros(M, 1, device=dev),
                "total_recovery": torch.zeros(M, 1, device=dev),
                "total_workout_cost": torch.zeros(M, 1, device=dev),
                "total_funding_cost": torch.zeros(M, 1, device=dev),
                "loans_booked": torch.zeros(M, 1, device=dev),
                "defaults_count": torch.zeros(M, 1, device=dev),
                "fraud_count": torch.zeros(M, 1, device=dev),
            },
        },
        "objects": None,
        "network": None,
    }

    return state


def run_standalone(
    state: dict,
    num_months: int = 24,
    verbose: bool = True,
) -> dict:
    """Run the simulation in standalone mode (no agent-torch required).

    Executes the 6 substeps sequentially for each month.
    Returns the final state dict.
    """
    # Instantiate substep modules with proper output_variables
    demand_policy = GenerateDemand(
        output_variables=["applying"],
        arguments={"rate_sensitivity": 1.0},
    )
    mark_applicants = MarkApplicants(output_variables=["loan_status"])
    credit_eval = CreditEvaluate(
        output_variables=["approval_prob", "offered_rate"],
        arguments={"margin_threshold": 0.10, "leverage_cap": 1.5, "dscr_floor": 1.25},
    )
    store_evals = StoreEvaluations(output_variables=["loan_status"])
    auction = CompetitiveAuction(
        output_variables=[
            "loan_status", "outstanding_debt", "monthly_payment",
            "months_remaining", "assigned_lender", "interest_rate",
            "loan_origination_month",
            "available_capital", "sector_exposure", "loans_booked",
            "total_fees_earned",
        ],
        arguments={"temperature": 0.1},
    )
    amortization = MonthlyAmortization(
        output_variables=[
            "outstanding_debt", "months_remaining", "loan_status",
            "total_interest_earned", "total_funding_cost",
        ],
    )
    default_process = DefaultProcess(
        output_variables=[
            "outstanding_debt", "loan_status",
            "total_principal_lost", "total_recovery", "total_workout_cost",
            "defaults_count", "fraud_count",
            "total_defaults_this_step",
        ],
        arguments={"hazard_steepness": 5.0, "fraud_steepness": 10.0},
    )
    market_dynamics = MarketDynamics(
        output_variables=["current_month", "market_cycle_phase", "risk_tolerance"],
        arguments={"cycle_alpha": 0.1, "cycle_beta": 0.05, "target_default_rate": 0.05},
    )

    N = state["agents"]["borrowers"]["loan_status"].shape[0]
    M = state["agents"]["lenders"]["total_capital"].shape[0]

    if verbose:
        print(f"\nStarting simulation: {N} borrowers, {M} lenders, {num_months} months")
        print("=" * 70)

    trajectory = []

    for month in range(num_months):
        # Reset step-level counters
        state["environment"]["total_defaults_this_step"] = torch.tensor(
            [0.0], device=state["environment"]["current_month"].device
        )

        # --- Substep 0: Loan Application ---
        demand_output = demand_policy(state)
        # MarkApplicants is a pass-through
        mark_applicants(state)

        # --- Substep 1: Underwriting ---
        eval_output = credit_eval(state)
        approval_prob = eval_output.get("approval_prob")
        offered_rate = eval_output.get("offered_rate")

        # --- Substep 2: Adjudication ---
        action = {"lenders": {"approval_prob": approval_prob, "offered_rate": offered_rate}}
        auction_output = auction(state, action=action)

        # Apply auction results to state
        borrowers = state["agents"]["borrowers"]
        lenders = state["agents"]["lenders"]
        auction_keys_borrower = [
            "loan_status", "outstanding_debt", "monthly_payment",
            "months_remaining", "assigned_lender", "interest_rate",
            "loan_origination_month",
        ]
        auction_keys_lender = [
            "available_capital", "sector_exposure", "loans_booked",
            "total_fees_earned",
        ]
        all_keys = auction_keys_borrower + auction_keys_lender
        for key in all_keys:
            if key in auction_output:
                if key in borrowers:
                    borrowers[key] = auction_output[key]
                elif key in lenders:
                    lenders[key] = auction_output[key]

        # --- Substep 3: Repayment ---
        repay_output = amortization(state)
        repay_map = {
            "outstanding_debt": "borrowers",
            "months_remaining": "borrowers",
            "loan_status": "borrowers",
            "total_interest_earned": "lenders",
            "total_funding_cost": "lenders",
        }
        for key, agent_type in repay_map.items():
            if key in repay_output:
                state["agents"][agent_type][key] = repay_output[key]

        # --- Substep 4: Defaults ---
        default_output = default_process(state)
        default_map = {
            "outstanding_debt": ("agents", "borrowers"),
            "loan_status": ("agents", "borrowers"),
            "total_principal_lost": ("agents", "lenders"),
            "total_recovery": ("agents", "lenders"),
            "total_workout_cost": ("agents", "lenders"),
            "defaults_count": ("agents", "lenders"),
            "fraud_count": ("agents", "lenders"),
            "total_defaults_this_step": ("environment", None),
        }
        for key, (level, agent_type) in default_map.items():
            if key in default_output:
                if level == "environment":
                    state["environment"][key] = default_output[key]
                else:
                    state["agents"][agent_type][key] = default_output[key]

        # --- Substep 5: Market Update ---
        market_output = market_dynamics(state)
        market_map = {
            "current_month": "environment",
            "market_cycle_phase": "environment",
            "risk_tolerance": "lenders",
        }
        for key, target in market_map.items():
            if key in market_output:
                if target == "environment":
                    state["environment"][key] = market_output[key]
                else:
                    state["agents"]["lenders"][key] = market_output[key]

        # --- Progress reporting ---
        if verbose and (month % 3 == 0 or month == num_months - 1):
            performing = (state["agents"]["borrowers"]["loan_status"] == 1).sum().item()
            defaulted = (state["agents"]["borrowers"]["loan_status"] == 2).sum().item()
            repaid = (state["agents"]["borrowers"]["loan_status"] == 3).sum().item()
            no_loan = (state["agents"]["borrowers"]["loan_status"] == 0).sum().item()
            cycle = state["environment"]["market_cycle_phase"].item()
            total_deployed = (
                state["agents"]["lenders"]["total_capital"].sum().item()
                - state["agents"]["lenders"]["available_capital"].sum().item()
            )
            print(
                f"  Month {month+1:>3}: "
                f"Performing={performing:>6} | Defaulted={defaulted:>5} | "
                f"Repaid={repaid:>5} | Unmatched={no_loan:>6} | "
                f"Cycle={cycle:.3f} | Deployed=${total_deployed:>12,.0f}"
            )

        # Store snapshot for trajectory
        trajectory.append({
            "month": month + 1,
            "performing": (state["agents"]["borrowers"]["loan_status"] == 1).sum().item(),
            "defaulted": (state["agents"]["borrowers"]["loan_status"] == 2).sum().item(),
            "repaid": (state["agents"]["borrowers"]["loan_status"] == 3).sum().item(),
            "cycle_phase": state["environment"]["market_cycle_phase"].item(),
        })

    return state


def main():
    parser = argparse.ArgumentParser(
        description="AgentTorch Lending Market Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 1K borrowers
  python -m agenttorch_sim.run --num-borrowers 1000 --num-lenders 5

  # Validate against Loanville2 static borrowers
  python -m agenttorch_sim.run --from-loanville --mix balanced

  # Full simulation with 100K borrowers
  python -m agenttorch_sim.run --num-borrowers 100000 --num-lenders 20

  # Conservative economics
  python -m agenttorch_sim.run --num-borrowers 10000 --economics conservative
        """,
    )
    parser.add_argument("--num-borrowers", type=int, default=10000,
                        help="Number of borrower agents (default: 10000)")
    parser.add_argument("--num-lenders", type=int, default=10,
                        help="Number of lender agents (default: 10)")
    parser.add_argument("--months", type=int, default=24,
                        help="Simulation horizon in months (default: 24)")
    parser.add_argument("--economics", choices=["balanced", "aggressive", "conservative"],
                        default="balanced", help="Economics preset (default: balanced)")
    parser.add_argument("--mix", choices=["gentle", "realistic", "adversarial"],
                        default="realistic", help="Borrower mix distribution (default: realistic)")
    parser.add_argument("--from-loanville", action="store_true",
                        help="Use Loanville2's static borrower pool (24-36 borrowers)")
    parser.add_argument("--loanville-mix", default="all",
                        help="Loanville2 mix preset when using --from-loanville")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Economics
    economics = TORCH_ECONOMICS_PRESETS[args.economics]

    # Generate populations
    if args.from_loanville:
        print(f"Loading Loanville2 static borrowers (mix={args.loanville_mix})...")
        borrower_tensors = from_loanville(mix=args.loanville_mix)
        num_borrowers = borrower_tensors["sector"].shape[0]
        print(f"  Loaded {num_borrowers} borrowers")
    else:
        print(f"Generating {args.num_borrowers} procedural borrowers (mix={args.mix}, seed={args.seed})...")
        borrower_tensors = generate_population(
            num_borrowers=args.num_borrowers,
            mix=args.mix,
            seed=args.seed,
        )
        num_borrowers = args.num_borrowers

    print(f"Generating {args.num_lenders} lender agents...")
    lender_tensors = generate_lender_tensors(
        num_lenders=args.num_lenders,
        seed=args.seed,
    )

    # Distribution summary
    n_good = (borrower_tensors["default_prob"] == 0).sum().item()
    n_fraud = (borrower_tensors["fraud_prob"] > 0.5).sum().item()
    n_bad = num_borrowers - n_good - n_fraud
    print(f"  Distribution: {n_good} good ({n_good/num_borrowers:.1%}), "
          f"{n_bad} bad ({n_bad/num_borrowers:.1%}), "
          f"{n_fraud} fraud ({n_fraud/num_borrowers:.1%})")

    # Build state
    state = build_initial_state(
        borrower_tensors=borrower_tensors,
        lender_tensors=lender_tensors,
        economics=economics,
        device=args.device,
    )

    # Run simulation
    t0 = time.time()
    final_state = run_standalone(
        state=state,
        num_months=args.months,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    print(f"\nSimulation complete in {elapsed:.2f}s")
    print(f"  {num_borrowers} borrowers x {args.num_lenders} lenders x {args.months} months")
    print(f"  Throughput: {num_borrowers * args.months / elapsed:,.0f} borrower-months/sec")

    # Score lenders
    scores = score_lenders(final_state, economics=economics, sim_horizon_months=args.months)
    print_scores(scores)


if __name__ == "__main__":
    main()
