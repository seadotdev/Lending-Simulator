"""
Entry point for: python -m loanville
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from .data import get_borrowers, get_lenders
from .engine import SimulationEngine
from .scoring import print_final_report, score_lenders


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        print("Set it in a .env file or export it directly:")
        print("  export OPENROUTER_API_KEY=your-key-here")
        sys.exit(1)

    print("=" * 70)
    print("  LOANVILLE — THE LLM LENDING SIMULATOR")
    print("=" * 70)

    borrowers = get_borrowers()
    lenders = get_lenders()

    print(f"\nLoaded {len(borrowers)} borrower applications")
    print(f"Loaded {len(lenders)} competing lenders:\n")
    for l in lenders:
        deployed = sum(x.remaining_balance for x in l.existing_portfolio)
        print(f"  {l.name} ({l.model})")
        print(f"    Capital: ${l.total_capital:,.0f} | "
              f"Deployed: ${deployed:,.0f} | "
              f"Target Yield: {l.target_yield_pct}%")

    engine = SimulationEngine(borrowers, lenders, api_key)

    # Run the simulation
    asyncio.run(engine.run())

    # Scoring
    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
    )
    print_final_report(scores)

    print("\nSimulation complete.\n")


if __name__ == "__main__":
    main()
