#!/usr/bin/env python3
"""LOS live smoke test using a cheap small model."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loanville.data import MIX_PRESETS, get_borrowers, get_lenders  # noqa: E402
from loanville.engine import SimulationEngine  # noqa: E402


def run() -> int:
    parser = argparse.ArgumentParser(description="Smoke test LOS underwriting flow")
    parser.add_argument(
        "--model",
        default="meta-llama/llama-3.1-8b-instruct",
        help="Cheap/small model to use for all lenders (via LOS)",
    )
    parser.add_argument("--mix", choices=list(MIX_PRESETS.keys()), default="easy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--los-url", default="http://localhost:3000")
    args = parser.parse_args()

    load_dotenv()

    borrowers = get_borrowers(args.mix, seed=args.seed, sample_size=args.sample_size)
    lenders = get_lenders()
    for lender in lenders:
        lender.model = args.model
        lender.name = f"{lender.name.split('[')[0].strip()} [{args.model.split('/')[-1]}]"

    expected = len(borrowers) * len(lenders)
    print(
        f"Running LOS smoke: model={args.model}, mix={args.mix}, "
        f"borrowers={len(borrowers)}, lenders={len(lenders)}, evals={expected}"
    )

    engine = SimulationEngine(
        borrowers=borrowers,
        lenders=lenders,
        max_concurrent_per_lender=1,
        mock=False,
        los_url=args.los_url,
        los_mode="full",
        formal_los_only=True,
    )
    asyncio.run(engine.run())

    decisions = [d for lender_decisions in engine.all_decisions.values() for d in lender_decisions]
    approvals = sum(1 for d in decisions if d.decision == "APPROVE")
    rejections = sum(1 for d in decisions if d.decision == "REJECT")
    run_count = len(engine.runs)

    print(
        f"Decision summary: total={len(decisions)}, approvals={approvals}, "
        f"rejections={rejections}, runs={run_count}"
    )

    failures: list[str] = []
    if len(decisions) != expected:
        failures.append(f"expected {expected} decisions, got {len(decisions)}")
    if run_count != expected:
        failures.append(f"expected {expected} runs, got {run_count}")
    if approvals + rejections == 0:
        failures.append("no underwriting decisions were produced")

    if failures:
        print("SMOKE FAILED:")
        for issue in failures:
            print(f"  - {issue}")
        return 1

    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
