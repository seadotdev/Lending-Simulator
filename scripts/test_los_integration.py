#!/usr/bin/env python3
"""Run a SIM smoke test against a live LOS API backend."""

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loanville.data import MIX_PRESETS, get_borrowers, get_lenders  # noqa: E402
from loanville.engine import SimulationEngine  # noqa: E402


def run() -> int:
    parser = argparse.ArgumentParser(description="Smoke test SIM <-> LOS integration")
    parser.add_argument("--los-base-url", default="http://localhost:3000")
    parser.add_argument("--los-timeout-s", type=float, default=20.0)
    parser.add_argument("--los-tenant-id", default="loanville-sim")
    parser.add_argument("--mix", choices=list(MIX_PRESETS.keys()), default="fraud")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=6)
    args = parser.parse_args()

    borrowers = get_borrowers(args.mix, seed=args.seed, sample_size=args.sample_size)
    lenders = get_lenders()
    expected_runs = len(borrowers) * len(lenders)

    print(
        f"Running LOS integration smoke: base_url={args.los_base_url}, "
        f"mix={args.mix}, borrowers={len(borrowers)}, lenders={len(lenders)}"
    )

    engine = SimulationEngine(
        borrowers=borrowers,
        lenders=lenders,
        mock=False,
        data_mode="full",
        los_url=args.los_base_url,
    )
    asyncio.run(engine.run())

    decisions = [d for lender_decisions in engine.all_decisions.values() for d in lender_decisions]
    approvals = sum(1 for d in decisions if d.decision == "APPROVE")
    rejections = sum(1 for d in decisions if d.decision == "REJECT")

    print(
        f"Decision summary: total={len(decisions)}, approvals={approvals}, "
        f"rejections={rejections}, runs={len(engine.runs)}"
    )

    failures: list[str] = []
    if len(engine.runs) != expected_runs:
        failures.append(f"expected {expected_runs} runs, got {len(engine.runs)}")
    if len(decisions) != expected_runs:
        failures.append(f"expected {expected_runs} decisions, got {len(decisions)}")
    if approvals == 0:
        failures.append("no approvals produced")
    if rejections == 0:
        failures.append("no rejections produced")

    if failures:
        print("SMOKE FAILED:")
        for issue in failures:
            print(f"  - {issue}")
        return 1

    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
