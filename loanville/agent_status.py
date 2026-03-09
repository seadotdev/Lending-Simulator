"""Status dashboard for agent sim runs.

Usage: python -m loanville status [run_id]
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent_events import EventLogger


RUNS_DIR = Path("runs")


def latest_run(runs_dir: Path = RUNS_DIR) -> Path | None:
    """Return the most recent run directory, or None."""
    if not runs_dir.is_dir():
        return None
    dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[0] if dirs else None


def summarise_case(case_dir: Path, lender_override: str = "") -> dict:
    """Summarise a single case from its events.jsonl and summary.json."""
    result: dict = {
        "lender": lender_override or case_dir.parent.name,
        "borrower": case_dir.name,
        "status": "unknown",
        "turns": 0,
        "tool_calls": 0,
        "has_deal": False,
        "decision": "",
        "cost": "",
        "duration_s": 0,
    }

    # Read summary.json if available
    summary_path = case_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            # Handle both agent_sim format (termination) and Docker format (error/decision)
            result["status"] = (
                summary.get("termination")
                or summary.get("error")
                or ("done" if summary.get("finished_at") else "unknown")
            )
            result["turns"] = summary.get("turns", 0)
            result["tool_calls"] = summary.get("tool_call_count") or summary.get("tool_calls", 0)
            result["has_deal"] = summary.get("has_deal", False)
            # decision can be a string or a dict with "decision" key
            dec = summary.get("decision", "")
            if isinstance(dec, dict):
                dec = dec.get("decision", "")
            result["decision"] = dec
            result["duration_s"] = summary.get("duration_s", 0)
            result["cost"] = summary.get("cost_estimate") or ""
            # Docker format stores cost differently
            cost_obj = summary.get("cost", {})
            if isinstance(cost_obj, dict) and cost_obj.get("measured_usd"):
                result["cost"] = f"${cost_obj['measured_usd']:.4f}"
            # Use borrower from summary if available
            if summary.get("borrower"):
                result["borrower"] = summary["borrower"]
            if summary.get("lender"):
                result["lender"] = summary["lender"]
            if summary.get("alias"):
                result["lender"] = summary["alias"]
            return result
        except json.JSONDecodeError:
            pass

    # Fall back to reading events
    events_path = case_dir / "events.jsonl"
    events = EventLogger.read(events_path)
    if not events:
        result["status"] = "no_data"
        return result

    tool_calls = [e for e in events if e.get("type") in ("tool_call", "tool_execution_end")]
    result["tool_calls"] = len(tool_calls)

    # Find loop_end event
    loop_ends = [e for e in events if e.get("type") == "loop_end"]
    if loop_ends:
        end = loop_ends[-1]
        result["status"] = end.get("termination", "done")
        result["turns"] = end.get("turns", 0)
        result["decision"] = end.get("decision", "")
        result["has_deal"] = end.get("has_deal", False)
        result["duration_s"] = end.get("t", 0)
    else:
        result["status"] = "running"
        model_calls = [e for e in events if e.get("type") == "model_call"]
        result["turns"] = len(model_calls)

    return result


def print_status_table(cases: list[dict]) -> None:
    """Print a formatted status table."""
    if not cases:
        print("  No cases found.")
        return

    # Column widths
    headers = ["lender", "borrower", "status", "turns", "tools", "deal", "decision", "dur(s)"]
    rows = []
    for c in cases:
        rows.append([
            c["lender"],
            c["borrower"],
            c["status"],
            str(c["turns"]),
            str(c["tool_calls"]),
            "Y" if c["has_deal"] else "-",
            c["decision"] or "-",
            f"{c['duration_s']:.0f}" if c["duration_s"] else "-",
        ])

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    # Print
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'─' * len(header_line)}")
    for row in rows:
        line = "  ".join(val.ljust(widths[i]) for i, val in enumerate(row))
        print(f"  {line}")


def main(run_id: str | None = None) -> None:
    """Print status dashboard for a run."""
    if run_id:
        run_dir = RUNS_DIR / run_id
    else:
        run_dir = latest_run()

    if run_dir is None or not run_dir.is_dir():
        print("  No runs found in runs/ directory.")
        return

    print(f"\n  Agent Sim Status: {run_dir.name}")
    print(f"  {'=' * 60}")

    cases = []
    for sub_dir in sorted(run_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        # Check if this dir itself has summary.json (Docker flat layout: run/{alias}/)
        if (sub_dir / "summary.json").exists() or (sub_dir / "events.jsonl").exists():
            cases.append(summarise_case(sub_dir, lender_override=sub_dir.name))
        else:
            # Nested layout: run/{lender}/{borrower}/
            for borrower_dir in sorted(sub_dir.iterdir()):
                if not borrower_dir.is_dir():
                    continue
                cases.append(summarise_case(borrower_dir))

    print_status_table(cases)

    # Summary line
    done = sum(1 for c in cases if c["status"] not in ("running", "unknown", "no_data"))
    total = len(cases)
    print(f"\n  {done}/{total} cases complete")
    print()
