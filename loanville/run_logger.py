"""
Run Logger — append-only storage and replay for UnderwritingRun artifacts.

The RunLogger is the observability spine of the flywheel:
- Every underwriting evaluation emits a Run
- Runs are stored as newline-delimited JSON (one per line)
- Runs can be replayed, diffed, and scored

Storage: runs/{policy_id}/{run_id}.json  (individual files)
         runs/log.jsonl                   (append-only log for streaming)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .run_schema import UnderwritingRun


DEFAULT_RUNS_DIR = "runs"


class RunLogger:
    """Append-only logger for UnderwritingRun artifacts."""

    def __init__(self, base_dir: str = DEFAULT_RUNS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, run: UnderwritingRun) -> str:
        """Persist a run. Returns the file path written."""
        # Write individual run file
        policy_dir = self.base_dir / run.policy.policy_id.replace("/", "_")
        policy_dir.mkdir(parents=True, exist_ok=True)
        run_path = policy_dir / f"{run.run_id}.json"
        run_path.write_text(run.to_json())

        # Append to log
        log_path = self.base_dir / "log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(run.to_dict(), default=str) + "\n")

        return str(run_path)

    def load(self, run_id: str) -> Optional[UnderwritingRun]:
        """Load a run by ID (searches all policy dirs)."""
        for policy_dir in self.base_dir.iterdir():
            if not policy_dir.is_dir():
                continue
            run_path = policy_dir / f"{run_id}.json"
            if run_path.exists():
                data = json.loads(run_path.read_text())
                return UnderwritingRun.from_dict(data)
        return None

    def load_by_policy(self, policy_id: str) -> list[UnderwritingRun]:
        """Load all runs for a given policy."""
        policy_dir = self.base_dir / policy_id.replace("/", "_")
        if not policy_dir.exists():
            return []
        runs = []
        for f in sorted(policy_dir.glob("*.json")):
            data = json.loads(f.read_text())
            runs.append(UnderwritingRun.from_dict(data))
        return runs

    def load_all(self) -> list[UnderwritingRun]:
        """Load all runs from the log."""
        log_path = self.base_dir / "log.jsonl"
        if not log_path.exists():
            return []
        runs = []
        for line in log_path.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                runs.append(UnderwritingRun.from_dict(data))
        return runs

    def load_by_case(self, case_id: str) -> list[UnderwritingRun]:
        """Load all runs for a given case/borrower."""
        return [r for r in self.load_all() if r.case.case_id == case_id]

    def count(self) -> int:
        """Count total runs in the log."""
        log_path = self.base_dir / "log.jsonl"
        if not log_path.exists():
            return 0
        return sum(1 for line in log_path.read_text().splitlines() if line.strip())

    def policies(self) -> list[str]:
        """List all policy IDs that have runs."""
        return [
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]


# ---------------------------------------------------------------------------
# Diff utility
# ---------------------------------------------------------------------------

def diff_runs(
    run_a: UnderwritingRun,
    run_b: UnderwritingRun,
) -> dict:
    """Compare two runs on the same case. Returns a structured diff.

    Useful for:
    - Comparing champion vs challenger on the same borrower
    - Regression testing after policy changes
    - Finding disagreements for benchmark case generation
    """
    diff = {
        "case_id": run_a.case.case_id,
        "policy_a": run_a.policy.policy_id,
        "policy_b": run_b.policy.policy_id,
        "same_case": run_a.case.case_id == run_b.case.case_id,
        "decision_match": run_a.decision.action == run_b.decision.action,
        "decisions": {
            "a": run_a.decision.action,
            "b": run_b.decision.action,
        },
    }

    # Terms comparison (if both approved)
    if run_a.decision.action == "approve" and run_b.decision.action == "approve":
        diff["terms_diff"] = {
            "amount": {
                "a": run_a.decision.terms.amount,
                "b": run_b.decision.terms.amount,
                "delta": run_b.decision.terms.amount - run_a.decision.terms.amount,
            },
            "apr": {
                "a": run_a.decision.terms.apr,
                "b": run_b.decision.terms.apr,
                "delta": run_b.decision.terms.apr - run_a.decision.terms.apr,
            },
            "tenor": {
                "a": run_a.decision.terms.tenor_months,
                "b": run_b.decision.terms.tenor_months,
                "delta": run_b.decision.terms.tenor_months - run_a.decision.terms.tenor_months,
            },
        }

    # Score comparison
    if run_a.scores.overall and run_b.scores.overall:
        diff["score_diff"] = {
            "a": run_a.scores.overall.get("score", 0),
            "b": run_b.scores.overall.get("score", 0),
            "delta": (run_b.scores.overall.get("score", 0) -
                      run_a.scores.overall.get("score", 0)),
        }

    # Latency comparison
    diff["latency_diff"] = {
        "a_ms": run_a.trace.latency_ms,
        "b_ms": run_b.trace.latency_ms,
    }

    return diff


def find_disagreements(
    runs_a: list[UnderwritingRun],
    runs_b: list[UnderwritingRun],
) -> list[dict]:
    """Find cases where two policies disagree.

    This is the core of the feedback loop: disagreements become
    new benchmark cases and simulator scenarios.

    Returns disagreements sorted by "interestingness":
    - High confidence + wrong (benchmark) → most interesting
    - Approve vs decline splits → second most interesting
    - Terms disagreements → third
    """
    # Index by case_id
    a_by_case = {r.case.case_id: r for r in runs_a}
    b_by_case = {r.case.case_id: r for r in runs_b}

    common_cases = set(a_by_case.keys()) & set(b_by_case.keys())
    disagreements = []

    for cid in common_cases:
        ra = a_by_case[cid]
        rb = b_by_case[cid]

        if ra.decision.action != rb.decision.action:
            d = diff_runs(ra, rb)
            d["disagreement_type"] = "action"
            d["severity"] = 3  # highest

            # Check if one is correct (against gold)
            if ra.labels.available and ra.labels.gold:
                correct = ra.labels.gold.get("correct_action", "")
                d["a_correct"] = ra.decision.action == correct
                d["b_correct"] = rb.decision.action == correct
                # High confidence + wrong = most interesting
                if ra.decision.confidence > 0.7 and not d["a_correct"]:
                    d["severity"] = 5
                if rb.decision.confidence > 0.7 and not d["b_correct"]:
                    d["severity"] = 5

            disagreements.append(d)

        elif (ra.decision.action == "approve" and rb.decision.action == "approve"):
            # Both approved but terms differ significantly
            apr_a = ra.decision.terms.apr
            apr_b = rb.decision.terms.apr
            amt_a = ra.decision.terms.amount
            amt_b = rb.decision.terms.amount

            apr_diff = abs(apr_a - apr_b)
            amt_diff = abs(amt_a - amt_b)

            if apr_diff > 0.02 or (amt_a > 0 and amt_diff / max(amt_a, 1) > 0.15):
                d = diff_runs(ra, rb)
                d["disagreement_type"] = "terms"
                d["severity"] = 1
                disagreements.append(d)

    # Sort by severity descending
    disagreements.sort(key=lambda x: x.get("severity", 0), reverse=True)
    return disagreements
