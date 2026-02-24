#!/usr/bin/env python3
"""Replay determinism check for SIM mock mode."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loanville.data import get_borrowers, get_lenders  # noqa: E402
from loanville.mock_llm import mock_evaluate_all  # noqa: E402


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_once(mix: str, data_mode: str, seed: int) -> tuple[str, dict]:
    borrowers = get_borrowers(mix=mix, seed=seed)
    lenders = get_lenders()
    decisions_by_lender = mock_evaluate_all(lenders=lenders, borrowers=borrowers, data_mode=data_mode)

    payload = {
        lender_id: [
            {
                "borrower_id": decision.borrower_id,
                "decision": decision.decision,
                "loan_amount": round(decision.term_sheet.loan_amount, 2) if decision.term_sheet else None,
                "interest_rate": round(decision.term_sheet.interest_rate, 4) if decision.term_sheet else None,
                "term_months": decision.term_sheet.term_months if decision.term_sheet else None,
            }
            for decision in decisions
        ]
        for lender_id, decisions in sorted(decisions_by_lender.items())
    }

    return _stable_hash(payload), payload


def run() -> int:
    parser = argparse.ArgumentParser(description="Check deterministic replay hash in mock mode")
    parser.add_argument("--mix", default="fraud")
    parser.add_argument("--data-mode", default="lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/replay/run_hashes.json")
    args = parser.parse_args()

    first_hash, first_payload = _run_once(args.mix, args.data_mode, args.seed)
    second_hash, second_payload = _run_once(args.mix, args.data_mode, args.seed)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "mix": args.mix,
                "data_mode": args.data_mode,
                "seed": args.seed,
                "first_hash": first_hash,
                "second_hash": second_hash,
                "match": first_hash == second_hash,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"first_hash={first_hash}")
    print(f"second_hash={second_hash}")
    print(f"hash_report={output_path}")

    if first_hash != second_hash:
        # Emit diff payloads for easier debugging if determinism breaks.
        diff_path = output_path.with_name("run_hash_mismatch_payloads.json")
        diff_path.write_text(
            json.dumps({"first": first_payload, "second": second_payload}, indent=2) + "\n"
        )
        print(f"mismatch_payloads={diff_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
