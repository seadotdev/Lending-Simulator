"""
CLI for the git-native leaderboard.

Usage:
  python -m loanville.leaderboard standings
  python -m loanville.leaderboard recompute
  python -m loanville.leaderboard ingest <file>
  python -m loanville.leaderboard validate <file>
  python -m loanville.leaderboard history <model_id>
  python -m loanville.leaderboard export --format md
"""

import argparse
import json
import sys
from pathlib import Path

from .core import (
    compute_leaderboard,
    load_all_matches,
    load_config,
    load_leaderboard,
    validate_match,
    write_leaderboard,
    write_match_record,
)


def cmd_standings(args):
    """Print current leaderboard standings."""
    lb = load_leaderboard()
    if lb is None:
        print("No leaderboard found. Run 'recompute' first or complete a sim with --leaderboard.")
        return

    standings = lb.get("standings", [])
    if not standings:
        print("Leaderboard is empty (no matches recorded).")
        return

    print(f"\n{'='*80}")
    print(f"  LOANVILLE LEADERBOARD — 3-Elo Rankings")
    print(f"  {lb.get('n_matches', 0)} matches | computed {lb.get('computed_at', '?')}")
    print(f"{'='*80}")

    print(f"\n  {'#':<4} {'Model':<35} {'Profit':>8} {'Credit':>8} {'Deal':>8} {'Matches':>8} {'Avg P&L':>12}")
    print(f"  {'─'*83}")

    for i, s in enumerate(standings, 1):
        name = s["display_name"]
        if len(name) > 33:
            name = name[:30] + "..."
        avg_pnl = s.get("avg_net_pnl", s.get("avg_raroc", 0.0))
        print(
            f"  {i:<4} {name:<35} "
            f"{s['profit_elo']:>7.0f} {s['credit_elo']:>7.0f} {s['dealshare_elo']:>7.0f} "
            f"{s['matches_played']:>8} ${avg_pnl:>+10,.0f}"
        )

    # Confusion matrix summary
    print(f"\n  {'Model':<35} {'Good→Appr':>10} {'Good→Rej':>10} {'Bad→Appr':>10} {'Bad→Rej':>10} {'Fraud→A':>8}")
    print(f"  {'─'*83}")
    for s in standings:
        cm = s.get("confusion_agg", {})
        name = s["display_name"]
        if len(name) > 33:
            name = name[:30] + "..."
        g_a = cm.get("good", {}).get("approved", 0)
        g_r = cm.get("good", {}).get("rejected", 0)
        b_a = cm.get("bad", {}).get("approved", 0)
        b_r = cm.get("bad", {}).get("rejected", 0)
        f_a = cm.get("fraud", {}).get("approved", 0)
        print(
            f"  {name:<35} {g_a:>10} {g_r:>10} {b_a:>10} {b_r:>10} {f_a:>8}"
        )

    print()


def cmd_recompute(args):
    """Full recomputation from all match files."""
    matches = load_all_matches()
    if not matches:
        print("No match files found in leaderboard/matches/")
        return

    print(f"Recomputing leaderboard from {len(matches)} match files...")
    leaderboard = compute_leaderboard(matches)
    path = write_leaderboard(leaderboard)
    print(f"Leaderboard written to {path}")
    print(f"  {leaderboard['n_matches']} valid matches, "
          f"{len(leaderboard['standings'])} models ranked")


def cmd_ingest(args):
    """Validate and add a match record."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    with open(filepath) as f:
        record = json.load(f)

    config = load_config()
    validation = validate_match(record, config)

    if not validation["valid"]:
        print(f"FAILED validation:")
        for err in validation["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    record["validation"] = validation
    match_path = write_match_record(record)
    print(f"Match record written to {match_path}")

    # Update leaderboard
    leaderboard = compute_leaderboard()
    lb_path = write_leaderboard(leaderboard)
    print(f"Leaderboard updated at {lb_path}")


def cmd_validate(args):
    """Dry-run validation of a match record."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    with open(filepath) as f:
        record = json.load(f)

    config = load_config()
    validation = validate_match(record, config)

    if validation["valid"]:
        print(f"PASS: {filepath.name}")
    else:
        print(f"FAIL: {filepath.name}")
        for err in validation["errors"]:
            print(f"  - {err}")
        sys.exit(1)


def cmd_history(args):
    """Show Elo trajectory over time for a model."""
    lb = load_leaderboard()
    if lb is None:
        print("No leaderboard found.")
        return

    model_id = args.model_id
    history = lb.get("elo_history", {}).get(model_id)

    if history is None:
        # Try partial match
        for mid in lb.get("elo_history", {}):
            if model_id.lower() in mid.lower():
                model_id = mid
                history = lb["elo_history"][mid]
                break

    if not history:
        print(f"No history for model: {args.model_id}")
        print(f"Available models:")
        for mid in lb.get("elo_history", {}):
            print(f"  {mid}")
        return

    # Find display name
    display_name = model_id
    for s in lb.get("standings", []):
        if s["model_id"] == model_id:
            display_name = s["display_name"]
            break

    print(f"\n  Elo History: {display_name} ({model_id})")
    print(f"  {'─'*60}")
    print(f"  {'Match':<6} {'Profit':>8} {'Credit':>8} {'Deal':>8}")
    print(f"  {'─'*60}")

    for i, h in enumerate(history, 1):
        print(f"  {i:<6} {h['profit_elo']:>7.0f} {h['credit_elo']:>7.0f} {h['dealshare_elo']:>7.0f}")

    if history:
        last = history[-1]
        print(f"  {'─'*60}")
        print(f"  Final: Profit={last['profit_elo']:.0f} "
              f"Credit={last['credit_elo']:.0f} "
              f"Deal={last['dealshare_elo']:.0f}")
    print()


def cmd_export(args):
    """Export leaderboard in various formats."""
    lb = load_leaderboard()
    if lb is None:
        print("No leaderboard found.")
        return

    fmt = args.format
    standings = lb.get("standings", [])

    if fmt == "md":
        _export_markdown(lb, standings)
    elif fmt == "html":
        print("HTML export: use leaderboard/index.html (reads leaderboard.json directly)")
    elif fmt == "json":
        print(json.dumps(lb, indent=2))
    else:
        print(f"Unknown format: {fmt}")
        sys.exit(1)


def _export_markdown(lb, standings):
    """Print Markdown table."""
    print(f"# Loanville Leaderboard")
    print(f"")
    print(f"*{lb.get('n_matches', 0)} matches | updated {lb.get('computed_at', '?')}*")
    print(f"")
    print(f"| # | Model | Profit Elo | Credit Elo | DealShare Elo | Matches | Avg Net P&L |")
    print(f"|---|-------|-----------|-----------|--------------|---------|-------------|")
    for i, s in enumerate(standings, 1):
        avg_pnl = s.get("avg_net_pnl", s.get("avg_raroc", 0.0))
        print(
            f"| {i} | {s['display_name']} | "
            f"{s['profit_elo']:.0f} | {s['credit_elo']:.0f} | {s['dealshare_elo']:.0f} | "
            f"{s['matches_played']} | ${avg_pnl:+,.0f} |"
        )

    print(f"")
    print(f"## Confusion Matrix (Aggregate)")
    print(f"")
    print(f"| Model | Good→Approve | Good→Reject | Bad→Approve | Bad→Reject | Fraud→Approve |")
    print(f"|-------|-------------|------------|------------|-----------|--------------|")
    for s in standings:
        cm = s.get("confusion_agg", {})
        print(
            f"| {s['display_name']} | "
            f"{cm.get('good', {}).get('approved', 0)} | "
            f"{cm.get('good', {}).get('rejected', 0)} | "
            f"{cm.get('bad', {}).get('approved', 0)} | "
            f"{cm.get('bad', {}).get('rejected', 0)} | "
            f"{cm.get('fraud', {}).get('approved', 0)} |"
        )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m loanville.leaderboard",
        description="Loanville 3-Elo Leaderboard",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("standings", help="Print current leaderboard standings")
    sub.add_parser("recompute", help="Full recomputation from all match files")

    p_ingest = sub.add_parser("ingest", help="Validate + add a match record")
    p_ingest.add_argument("file", help="Path to match record JSON")

    p_validate = sub.add_parser("validate", help="Dry-run validation")
    p_validate.add_argument("file", help="Path to match record JSON")

    p_history = sub.add_parser("history", help="Show Elo trajectory for a model")
    p_history.add_argument("model_id", help="Model ID (or partial match)")

    p_export = sub.add_parser("export", help="Export leaderboard")
    p_export.add_argument("--format", choices=["md", "html", "json"], default="md",
                          help="Output format (default: md)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "standings": cmd_standings,
        "recompute": cmd_recompute,
        "ingest": cmd_ingest,
        "validate": cmd_validate,
        "history": cmd_history,
        "export": cmd_export,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
