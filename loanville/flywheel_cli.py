#!/usr/bin/env python3
"""
Flywheel CLI — unified commands for the LOS → Benchmark → Simulator loop.

Commands:
  los run    --case <id> --policy <id>    Run a single underwriting evaluation
  los score  --run <run_id>               Score a run (writes scorecard)
  los tournament --policies p1 p2 p3      Run Elo tournament over policies
  los promote --policy <id>               Mark policy as champion
  los mine   --disagreements --last 7d    Find cases to turn into benchmark items
  los status                              Print champion/challenger status
  los replay --run <run_id>               Replay and display a run
  los diff   --run-a <id> --run-b <id>    Diff two runs on the same case
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .champion import ChampionTracker
from .run_logger import RunLogger, diff_runs, find_disagreements
from .run_schema import UnderwritingRun
from .scorecard import Scorecard, print_scorecard, score_run, score_runs


def cmd_status(args: argparse.Namespace) -> None:
    """Print champion/challenger status and run counts."""
    logger = RunLogger(args.runs_dir)
    tracker = ChampionTracker(args.runs_dir)

    print(f"\n  Run store: {args.runs_dir}")
    print(f"  Total runs logged: {logger.count()}")
    print(f"  Policies with runs: {logger.policies()}")
    tracker.print_status()


def cmd_score(args: argparse.Namespace) -> None:
    """Score one or more runs."""
    logger = RunLogger(args.runs_dir)

    if args.run_id:
        run = logger.load(args.run_id)
        if not run:
            print(f"  Run {args.run_id} not found.")
            return
        card = score_run(run)
        print_scorecard(card)
        # Write scorecard back
        logger.log(run)  # re-log with scores
        print(f"  Scorecard written to run {run.run_id[:12]}")

    elif args.policy:
        runs = logger.load_by_policy(args.policy)
        if not runs:
            print(f"  No runs found for policy '{args.policy}'.")
            return
        cards = score_runs(runs)
        print(f"\n  Scored {len(cards)} runs for policy '{args.policy}':")
        for card in cards[:10]:
            print_scorecard(card)
        if len(cards) > 10:
            print(f"  ... and {len(cards) - 10} more")

        # Aggregate
        avg = sum(c.overall_score for c in cards) / len(cards)
        gates_pass = sum(1 for c in cards if c.gates.passed) / len(cards)
        dec_acc = sum(c.uw_quality.decision_acc for c in cards) / len(cards)
        print(f"\n  Aggregate: avg_score={avg:.1f} gates_pass={gates_pass:.0%} "
              f"decision_acc={dec_acc:.0%}")

    else:
        print("  Specify --run-id or --policy to score.")


def cmd_promote(args: argparse.Namespace) -> None:
    """Promote a challenger to champion."""
    tracker = ChampionTracker(args.runs_dir)
    logger = RunLogger(args.runs_dir)

    if args.auto:
        # Auto-evaluate all challengers
        for ch in tracker.state.challengers:
            runs = logger.load_by_policy(ch.policy_id)
            if runs:
                cards = score_runs(runs)
                tracker.update_scores(ch.policy_id, cards)
                should, reason = tracker.should_promote(ch.policy_id)
                if should:
                    tracker.promote(ch.policy_id, reason=reason)
                    print(f"  PROMOTED: {ch.policy_id} — {reason}")
                else:
                    print(f"  {ch.policy_id}: not ready — {reason}")
        return

    policy_id = args.policy
    if not policy_id:
        print("  Specify --policy <id> or --auto.")
        return

    # Score the policy first
    runs = logger.load_by_policy(policy_id)
    if runs:
        cards = score_runs(runs)
        tracker.update_scores(policy_id, cards)

    should, reason = tracker.should_promote(policy_id)
    if args.force or should:
        tracker.promote(policy_id, reason=reason or "forced promotion")
        print(f"  PROMOTED: {policy_id}")
    else:
        print(f"  NOT promoting: {reason}")
        print(f"  Use --force to override.")


def cmd_mine(args: argparse.Namespace) -> None:
    """Find disagreements between champion and challengers."""
    tracker = ChampionTracker(args.runs_dir)
    logger = RunLogger(args.runs_dir)

    champion = tracker.get_champion()
    if not champion:
        print("  No champion set. Run 'los promote' first.")
        return

    champion_runs = logger.load_by_policy(champion.policy_id)
    if not champion_runs:
        print(f"  No runs for champion '{champion.policy_id}'.")
        return

    all_disagreements = []
    for ch in tracker.state.challengers:
        ch_runs = logger.load_by_policy(ch.policy_id)
        if ch_runs:
            disag = find_disagreements(champion_runs, ch_runs)
            for d in disag:
                d["challenger"] = ch.policy_id
            all_disagreements.extend(disag)

    # Sort by severity
    all_disagreements.sort(key=lambda x: x.get("severity", 0), reverse=True)

    n = args.top or 20
    print(f"\n  Top {min(n, len(all_disagreements))} disagreements "
          f"(champion: {champion.policy_id}):")
    print(f"  {'─' * 70}")

    for d in all_disagreements[:n]:
        case = d.get("case_id", "?")
        dtype = d.get("disagreement_type", "?")
        sev = d.get("severity", 0)
        dec_a = d.get("decisions", {}).get("a", "?")
        dec_b = d.get("decisions", {}).get("b", "?")
        challenger = d.get("challenger", "?")

        print(f"  [{sev}] {case}: champion={dec_a} vs {challenger}={dec_b} ({dtype})")

        if "terms_diff" in d:
            td = d["terms_diff"]
            print(f"       APR: {td['apr']['a']:.2%} vs {td['apr']['b']:.2%} "
                  f"(Δ{td['apr']['delta']:+.2%})")

    if not all_disagreements:
        print("  No disagreements found (or no challenger runs).")

    print(f"\n  Total disagreements: {len(all_disagreements)}")
    print(f"  Action splits: {sum(1 for d in all_disagreements if d.get('disagreement_type') == 'action')}")
    print(f"  Terms splits: {sum(1 for d in all_disagreements if d.get('disagreement_type') == 'terms')}")


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay and display a run."""
    logger = RunLogger(args.runs_dir)
    run = logger.load(args.run_id)
    if not run:
        print(f"  Run {args.run_id} not found.")
        return

    print(f"\n{'=' * 60}")
    print(f"  REPLAY — {run.run_id}")
    print(f"{'=' * 60}")
    print(f"  Case: {run.case.case_id} ({run.case.segment})")
    print(f"  Policy: {run.policy.policy_id} ({run.policy.model})")
    print(f"  Requested: ${run.case.requested_amount:,.0f} for {run.case.requested_purpose}")
    print(f"\n  Business: {run.inputs.business.company_name} "
          f"({run.inputs.business.industry}, {run.inputs.business.years_trading}yr)")
    print(f"  Revenue: ${run.inputs.financials.revenue_ttm:,.0f}")
    print(f"  Net Income: ${run.inputs.financials.net_income:,.0f}")
    print(f"  Gross Margin: {run.inputs.financials.gross_margin:.1%}")

    if run.trace.steps:
        print(f"\n  Trace ({len(run.trace.steps)} steps):")
        for s in run.trace.steps:
            if s.type == "tool_call":
                print(f"    [{s.t}] {s.name}({json.dumps(s.args)[:60]})")
            elif s.type == "note":
                print(f"    [{s.t}] NOTE: {s.content[:80]}")

    print(f"\n  DECISION: {run.decision.action.upper()}")
    if run.decision.action == "approve":
        t = run.decision.terms
        print(f"    Amount: ${t.amount:,.0f}")
        apr_display = t.apr * 100 if t.apr < 1.0 else t.apr
        print(f"    APR: {apr_display:.2f}%")
        print(f"    Tenor: {t.tenor_months}mo")
    if run.decision.rationale.summary:
        print(f"    Rationale: {run.decision.rationale.summary[:120]}")

    if run.labels.available:
        gold = run.labels.gold or {}
        print(f"\n  GROUND TRUTH: {gold.get('true_outcome', '?')} "
              f"(correct action: {gold.get('correct_action', '?')})")
        if run.labels.outcome:
            o = run.labels.outcome
            print(f"    Defaulted: {o.get('defaulted', '?')} | "
                  f"Loss: ${o.get('principal_lost', 0):,.0f}")

    if run.scores.overall:
        print(f"\n  SCORE: {run.scores.overall.get('score', 0):.1f}/100")

    print(f"{'=' * 60}")


def cmd_diff(args: argparse.Namespace) -> None:
    """Diff two runs."""
    logger = RunLogger(args.runs_dir)
    run_a = logger.load(args.run_a)
    run_b = logger.load(args.run_b)

    if not run_a:
        print(f"  Run A ({args.run_a}) not found.")
        return
    if not run_b:
        print(f"  Run B ({args.run_b}) not found.")
        return

    d = diff_runs(run_a, run_b)
    print(json.dumps(d, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="los",
        description="Loanville flywheel CLI — LOS + Benchmark + Simulator",
    )
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory for run artifacts (default: runs)")
    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Print system status")

    # score
    p_score = sub.add_parser("score", help="Score runs")
    p_score.add_argument("--run-id", type=str, help="Score a single run")
    p_score.add_argument("--policy", type=str, help="Score all runs for a policy")

    # promote
    p_promote = sub.add_parser("promote", help="Promote a challenger to champion")
    p_promote.add_argument("--policy", type=str, help="Policy ID to promote")
    p_promote.add_argument("--auto", action="store_true",
                           help="Auto-evaluate and promote if ready")
    p_promote.add_argument("--force", action="store_true",
                           help="Force promotion even if criteria not met")

    # mine
    p_mine = sub.add_parser("mine", help="Find disagreements for new benchmark cases")
    p_mine.add_argument("--top", type=int, default=20,
                        help="Show top N disagreements")

    # replay
    p_replay = sub.add_parser("replay", help="Replay a run")
    p_replay.add_argument("run_id", type=str, help="Run ID to replay")

    # diff
    p_diff = sub.add_parser("diff", help="Diff two runs")
    p_diff.add_argument("--run-a", type=str, required=True, help="Run A ID")
    p_diff.add_argument("--run-b", type=str, required=True, help="Run B ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "status": cmd_status,
        "score": cmd_score,
        "promote": cmd_promote,
        "mine": cmd_mine,
        "replay": cmd_replay,
        "diff": cmd_diff,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
