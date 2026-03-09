"""
Entry point for: python -m loanville

Usage:
  python -m loanville                    # LOS mode, formal interactions enforced
  python -m loanville --crm-sim          # High-volume CRM simulation benchmark
  python -m loanville --mock --allow-non-los-formal   # Legacy mock mode
  python -m loanville --mix hard         # Adversarial stress test
  python -m loanville --scenario realistic-10w --lenders budget-league
  python -m loanville --compare          # Compare big vs small models (mock)
  python -m loanville view               # Open web viewer in browser
  python -m loanville view season.json   # Open viewer with specific file
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .data import MIX_PRESETS, SCENARIOS, get_borrowers, get_lenders, get_scenario
from .engine import SimulationEngine
from .models import EconomicsConfig, ECONOMICS_PRESETS, LosConfig, SeasonConfig
from .presets import (
    apply_lender_preset,
    list_lender_presets,
    list_scenario_presets,
    load_lender_preset,
    load_scenario_preset,
)
from .scoring import print_final_report, print_season_report, score_lenders, score_season
from .season import SeasonEngine


def run_view(json_file=None, port=8765):
    """Start a local HTTP server and open the web viewer in the browser."""
    import http.server
    import json
    import shutil
    import threading
    import webbrowser
    from pathlib import Path

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if not web_dir.is_dir():
        print(f"ERROR: web directory not found at {web_dir}", file=sys.stderr)
        sys.exit(1)

    # If a JSON file was specified, copy it into web/seasons/ so the viewer picks it up
    if json_file:
        src = Path(json_file).resolve()
        if not src.exists():
            print(f"ERROR: file not found: {json_file}", file=sys.stderr)
            sys.exit(1)
        seasons_dir = web_dir / "seasons"
        seasons_dir.mkdir(exist_ok=True)
        dest = seasons_dir / src.name
        if not dest.exists() or dest.resolve() != src:
            shutil.copy2(src, dest)
        # Rebuild index
        all_files = sorted(
            [f.name for f in seasons_dir.glob("*.json") if f.name != "index.json"],
            reverse=True,
        )
        with open(seasons_dir / "index.json", "w") as f:
            json.dump(all_files, f, indent=2)
        print(f"  Loaded: {src.name}")

    os.chdir(web_dir)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # suppress request logs

    try:
        httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    except OSError:
        # Port in use — try next few ports
        for p in range(port + 1, port + 10):
            try:
                httpd = http.server.HTTPServer(("127.0.0.1", p), handler)
                port = p
                break
            except OSError:
                continue
        else:
            print(f"ERROR: Could not find an open port near {port}", file=sys.stderr)
            sys.exit(1)

    url = f"http://127.0.0.1:{port}"
    print(f"\n  Loanville Web Viewer")
    print(f"  {url}")
    print(f"  Press Ctrl+C to stop\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Viewer stopped.")
        httpd.shutdown()


def _run_single(borrowers, lenders, mock=False, data_mode="full",
                 los_url="http://localhost:3000",
                 los_provider="openrouter", los_mode="rules_only",
                 underwrite_only=False, los_model=None,
                 economics=None, info_asymmetry="none",
                 formal_los_only=False):
    """Run a single simulation and return (scores, engine)."""
    engine = SimulationEngine(
        borrowers, lenders, mock=mock, data_mode=data_mode,
        los_url=los_url,
        los_provider=los_provider, los_mode=los_mode,
        underwrite_only=underwrite_only, los_model=los_model,
        formal_los_only=formal_los_only,
        economics=economics,
        info_asymmetry=info_asymmetry,
    )
    asyncio.run(engine.run())

    scores = score_lenders(
        lenders,
        engine.all_decisions,
        engine.booked_loans,
        engine.loan_outcomes,
        engine.deal_results,
        borrowers=borrowers,
        economics=economics,
        runs=engine.runs,
    )
    print_final_report(scores, economics=economics)
    return scores, engine


def _print_mix_info(mix: str, borrowers) -> None:
    """Print the borrower population breakdown."""
    from collections import Counter
    outcomes = Counter(b.true_outcome for b in borrowers)
    total = len(borrowers)
    parts = []
    for k in ["good", "bad", "fraud"]:
        n = outcomes.get(k, 0)
        parts.append(f"{n} {k} ({n/total*100:.0f}%)")
    print(f"  Pipeline mix: {mix} — {', '.join(parts)}")


def run_compare(mix: str):
    """Run two simulations: big models vs small models, then compare."""
    borrowers = get_borrowers(mix)

    # --- Round 1: Big frontier models (tool-use capable) ---
    big_lenders = get_lenders()
    big_lenders[0].model = "deepseek/deepseek-chat-v3-0324"
    big_lenders[0].name = "Velocity Capital [DeepSeek V3]"
    big_lenders[1].model = "qwen/qwen3-235b-a22b"
    big_lenders[1].name = "Heritage Trust [Qwen3-235B]"
    big_lenders[2].model = "meta-llama/llama-3.3-70b-instruct"
    big_lenders[2].name = "Meridian Partners [Llama-3.3-70B]"

    print("\n" + "#" * 70)
    print("#  ROUND 1: FRONTIER MODELS (tool-use capable)")
    print("#" * 70)
    big_scores, _ = _run_single(borrowers, big_lenders, mock=True, data_mode="full")

    # --- Round 2: Small / mid-tier models ---
    small_lenders = get_lenders()
    small_lenders[0].model = "meta-llama/llama-3.1-8b-instruct"
    small_lenders[0].name = "Velocity Capital [Llama-8B]"
    small_lenders[1].model = "qwen/qwen-2.5-7b-instruct"
    small_lenders[1].name = "Heritage Trust [Qwen-7B]"
    small_lenders[2].model = "qwen/qwen3-30b-a3b"
    small_lenders[2].name = "Meridian Partners [Qwen3-30B]"

    print("\n\n" + "#" * 70)
    print("#  ROUND 2: SMALL MODELS")
    print("#" * 70)
    small_scores, _ = _run_single(borrowers, small_lenders, mock=True, data_mode="full")

    # --- Comparison ---
    print("\n\n" + "=" * 70)
    print("  HEAD-TO-HEAD: FRONTIER vs SMALL MODELS")
    print("=" * 70)

    print(f"\n  {'Metric':<30s} {'Frontier':>15s} {'Small':>15s}")
    print(f"  {'─'*60}")

    for label, big_list, small_list in [
        ("Avg Final Score (%)", big_scores, small_scores),
    ]:
        big_avg = sum(s.final_adjusted_score for s in big_list) / len(big_list)
        small_avg = sum(s.final_adjusted_score for s in small_list) / len(small_list)
        print(f"  {label:<30s} {big_avg:>14.2f}% {small_avg:>14.2f}%")

    big_fraud = sum(s.frauds_funded for s in big_scores)
    small_fraud = sum(s.frauds_funded for s in small_scores)
    print(f"  {'Total Frauds Funded':<30s} {big_fraud:>15d} {small_fraud:>15d}")

    big_defaults = sum(s.defaults_count for s in big_scores)
    small_defaults = sum(s.defaults_count for s in small_scores)
    print(f"  {'Total Defaults':<30s} {big_defaults:>15d} {small_defaults:>15d}")

    big_return = sum(s.net_return for s in big_scores)
    small_return = sum(s.net_return for s in small_scores)
    print(f"  {'Total Net Return ($)':<30s} ${big_return:>13,.0f} ${small_return:>13,.0f}")

    big_deployed = sum(s.total_deployed for s in big_scores)
    small_deployed = sum(s.total_deployed for s in small_scores)
    print(f"  {'Total Capital Deployed ($)':<30s} ${big_deployed:>13,.0f} ${small_deployed:>13,.0f}")

    big_lost = sum(s.total_principal_lost for s in big_scores)
    small_lost = sum(s.total_principal_lost for s in small_scores)
    print(f"  {'Total Principal Lost ($)':<30s} ${big_lost:>13,.0f} ${small_lost:>13,.0f}")

    big_violations = sum(len(s.concentration_violations) for s in big_scores)
    small_violations = sum(len(s.concentration_violations) for s in small_scores)
    print(f"  {'Concentration Violations':<30s} {big_violations:>15d} {small_violations:>15d}")

    print(f"\n  {'─'*60}")
    print(f"\n  Per-Lender Breakdown:")
    print(f"\n  {'Lender Slot':<18s} {'Frontier Score':>15s} {'Small Score':>17s} {'Delta':>10s}")
    print(f"  {'─'*60}")
    for bs, ss in zip(big_scores, small_scores):
        delta = bs.final_adjusted_score - ss.final_adjusted_score
        sign = "+" if delta >= 0 else ""
        print(f"  {bs.lender_name.split('[')[0].strip():<18s} "
              f"{bs.final_adjusted_score:>14.2f}% "
              f"{ss.final_adjusted_score:>16.2f}% "
              f"{sign}{delta:>8.2f}%")

    print(f"\n{'*'*70}")
    big_avg = sum(s.final_adjusted_score for s in big_scores) / len(big_scores)
    small_avg = sum(s.final_adjusted_score for s in small_scores) / len(small_scores)
    if big_avg > small_avg:
        print(f"  FRONTIER MODELS WIN by {big_avg - small_avg:.2f} percentage points")
    elif small_avg > big_avg:
        print(f"  SMALL MODELS WIN by {small_avg - big_avg:.2f} percentage points")
    else:
        print(f"  TIE!")
    print(f"  Frontier: {big_avg:.2f}% avg | Small: {small_avg:.2f}% avg")
    print(f"  Frontier frauds funded: {big_fraud} | Small frauds funded: {small_fraud}")
    print(f"{'*'*70}\n")


SEASON_DEFAULTS = {
    "weeks": 10,
    "cohort_size": 5,
    "months_per_week": 2,
    "season_mix": "realistic",
    "seed": 42,
    "arrival_phases": 1,
    "deep_uw_slots_per_week": 0,
    "speed_scoring": False,
    "custom_tools": False,
    "info_asymmetry": "none",
    "data_mode": "full",
    "borrower_patience_weeks": 1,
    "offer_validity_weeks": 1,
}


def _format_source(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Loanville — The LLM Lending Simulator")
    subparsers = parser.add_subparsers(dest="command")
    lenders_list = ", ".join(list_lender_presets()) or "(none found)"
    scenarios_list = ", ".join(list_scenario_presets()) or "(none found)"
    builtin_scenarios = ", ".join(sorted(SCENARIOS.keys()))

    # `view` subcommand
    view_parser = subparsers.add_parser("view", help="Open the web viewer in a browser")
    view_parser.add_argument("file", nargs="?", default=None,
                             help="Optional season JSON file to load")
    view_parser.add_argument("--port", type=int, default=8765,
                             help="HTTP server port (default: 8765)")

    # `status` subcommand
    status_parser = subparsers.add_parser("status", help="Show agent sim run status dashboard")
    status_parser.add_argument("run_id", nargs="?", default=None,
                               help="Run ID to inspect (default: latest)")

    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM responses (no API key needed)")
    parser.add_argument("--allow-non-los-formal", action="store_true",
                        help="Allow legacy non-formal paths (mock, rules_only, underwrite-only). "
                             "Default behavior enforces formal LOS-only interactions.")
    parser.add_argument("--compare", action="store_true",
                        help="Run frontier-vs-small model comparison (uses mock mode)")
    parser.add_argument("--lenders", default=None,
                        help="Lender preset name or file path (persona selection + model mapping). "
                             f"Built-in presets: {lenders_list}")
    parser.add_argument("--scenario", default=None,
                        help="Scenario preset name or file path (season bundle). "
                             "Implies --season. "
                             f"Built-in presets: {scenarios_list}. "
                             f"In-code scenarios: {builtin_scenarios}")
    parser.add_argument("--mix", choices=list(MIX_PRESETS.keys()), default="realistic",
                        help="Borrower population mix (default: realistic)")
    parser.add_argument("--data-mode",
                        choices=["full", "quarterly_only", "aggregate_only", "statements_inline", "lite"],
                        default=None,
                        help="Financial data presentation mode. "
                             "Default: full (or scenario preset value). "
                             "'lite' uses compact prompts optimized for small models (3B-30B).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deterministic borrower ordering")
    parser.add_argument("--los-url", default="http://localhost:3000",
                        help="Open LOS API URL (default: http://localhost:3000)")
    parser.add_argument("--los-provider", default="openrouter",
                        choices=["openrouter", "anthropic", "openai", "vercel"],
                        help="LLM provider for LOS evaluation (default: openrouter)")
    parser.add_argument("--los-mode", default="full",
                        choices=["full", "rules_only"],
                        help="LOS evaluation mode: 'full' uses LLM agent, "
                             "'rules_only' uses deterministic rules (default: full)")
    parser.add_argument("--underwrite-only", action="store_true",
                        help="Skip LOS pipeline ceremony (entity/deal/docs/spread/stages), "
                             "just POST dossier to /v1/underwrite for standalone LLM evaluation")
    parser.add_argument("--los-model", default=None,
                        help="Override the LLM model used by the LOS for underwriting "
                             "(default: uses each lender's model)")
    parser.add_argument("--crm-sim", action="store_true",
                        help="Run CRM-style high-volume LOS simulation benchmark "
                             "(throughput/correctness/responsiveness).")
    parser.add_argument("--crm-cases", type=int, default=120,
                        help="Number of CRM applications to simulate (default: 120)")
    parser.add_argument("--crm-mix",
                        choices=["gentle", "realistic", "adversarial", "stress", "escalating"],
                        default="realistic",
                        help="CRM simulation mix profile (default: realistic)")
    parser.add_argument("--crm-incomplete-ratio", type=float, default=0.35,
                        help="Fraction of CRM cases with incomplete info requiring follow-up "
                             "(default: 0.35)")
    parser.add_argument("--crm-concurrency", type=int, default=12,
                        help="Max in-flight LOS evaluations per lender in CRM simulation "
                             "(default: 12)")
    parser.add_argument("--crm-apr-tolerance", type=float, default=1.5,
                        help="APR tolerance (percentage points) for formulaic pricing correctness "
                             "(default: 1.5)")
    parser.add_argument("--crm-export", type=str, default=None, metavar="FILE",
                        help="Save CRM simulation summary JSON to FILE")
    parser.add_argument("--economics",
                        choices=list(ECONOMICS_PRESETS.keys()),
                        default="balanced",
                        help="Economics preset: balanced (default), aggressive, conservative")
    # Season mode arguments
    parser.add_argument("--season", action="store_true",
                        help="Run multi-week season mode")
    parser.add_argument("--weeks", type=int, default=None,
                        help="Number of weeks in season (default: 10, or scenario preset value)")
    parser.add_argument("--cohort-size", type=int, default=None,
                        help="Borrowers per week in season mode (default: 5, or scenario preset value)")
    parser.add_argument("--season-mix",
                        choices=["gentle", "realistic", "adversarial", "stress", "escalating"],
                        default=None,
                        help="Season borrower mix (default: realistic, or scenario preset value)")
    parser.add_argument("--speed-scoring", action="store_true",
                        help="Enable speed-to-offer scoring in season mode")
    parser.add_argument("--custom-tools", action="store_true",
                        help="Enable custom tool creation in season mode")
    parser.add_argument("--months-per-week", type=int, default=None,
                        help="Months of loan aging per season week (default: 2, or scenario preset value)")
    parser.add_argument("--borrower-patience-weeks", type=int, default=None,
                        help="How many weeks borrowers stay in market (default: 1, or scenario preset value)")
    parser.add_argument("--offer-validity-weeks", type=int, default=None,
                        help="How many weeks offers stay open (default: 1, or scenario preset value)")
    parser.add_argument("--arrival-phases", type=int, default=None,
                        help="Number of intra-week arrival phases (default: 1, or scenario preset value)")
    parser.add_argument("--deep-uw-slots", type=int, default=None,
                        help="Weekly cap on deep-underwrite approvals; 0 disables cap "
                             "(default: 0, or scenario preset value)")
    parser.add_argument("--info-asymmetry",
                        choices=["none", "partial_statements", "redacted"],
                        default=None,
                        help="Per-lender borrower view differences "
                             "(default: none, or scenario preset value)")
    # Agent sim arguments
    parser.add_argument("--agent-sim", action="store_true",
                        help="Run agentic LOS simulation (models drive the LOS autonomously)")
    parser.add_argument("--agent-mode", default="tool_call",
                        choices=["tool_call", "cli", "repl"],
                        help="Agent interaction mode (default: tool_call)")
    parser.add_argument("--agent-max-turns", type=int, default=20,
                        help="Max turns per agent task (default: 20)")
    parser.add_argument("--agent-tasks", default="simple_underwrite",
                        help="Comma-separated task types (default: simple_underwrite)")
    parser.add_argument("--agent-cases", type=int, default=3,
                        help="Number of borrower cases for agent sim (default: 3)")
    parser.add_argument("--agent-budget", type=float, default=None,
                        help="Per-model budget in USD for agent sim (requires OR_ADMIN_KEY)")
    parser.add_argument("--agent-parallel", action="store_true",
                        help="Run agent sim lenders in parallel")
    parser.add_argument("--agent-eject", action="store_true",
                        help="Enable eject policies (no-progress + quality) for agent sim")
    parser.add_argument("--agent-docker", action="store_true",
                        help="Run agent sim in Docker containers (full bash+curl, pi-style). "
                             "Requires Docker, OR_ADMIN_KEY, and --agent-budget.")
    parser.add_argument("--agent-build", action="store_true",
                        help="Build the Docker agent image and exit")

    # Leaderboard
    parser.add_argument("--leaderboard", action="store_true", default=True,
                        help="Emit match record to leaderboard after scoring (default: on)")
    parser.add_argument("--no-leaderboard", action="store_false", dest="leaderboard",
                        help="Disable leaderboard match record emission")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight model validation before season runs")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable detailed logging (LOS calls, per-borrower progress)")
    parser.add_argument("--export", type=str, default=None, metavar="FILE",
                        help="Save an extra copy of season results JSON to FILE")
    args = parser.parse_args()

    # Handle `view` subcommand
    if args.command == "view":
        run_view(json_file=args.file, port=args.port)
        return

    # Handle `status` subcommand
    if args.command == "status":
        from .agent_status import main as status_main
        status_main(run_id=args.run_id)
        return

    season_mode = args.season or bool(args.scenario)
    formal_los_only = not args.allow_non_los_formal

    if season_mode and args.compare:
        parser.error("--season/--scenario and --compare cannot be used together.")
    if args.compare and (args.lenders or args.scenario):
        parser.error("--compare cannot be combined with --lenders or --scenario.")
    if args.crm_sim and season_mode:
        parser.error("--crm-sim cannot be combined with --season/--scenario.")
    if args.crm_sim and args.compare:
        parser.error("--crm-sim cannot be combined with --compare.")
    if args.crm_sim and args.mock:
        parser.error("--crm-sim requires live LOS mode (no --mock).")

    if formal_los_only:
        if args.compare:
            parser.error(
                "--compare uses mock evaluation and is blocked in formal LOS-only mode. "
                "Use --allow-non-los-formal to run it intentionally."
            )
        if args.mock:
            parser.error(
                "--mock is blocked in formal LOS-only mode. "
                "Use --allow-non-los-formal to bypass."
            )
        if args.underwrite_only:
            parser.error(
                "--underwrite-only is blocked in formal LOS-only mode. "
                "Formal submissions/offers must flow through full LOS pipeline."
            )
        if args.los_mode != "full":
            parser.error(
                f"--los-mode {args.los_mode!r} is blocked in formal LOS-only mode; use --los-mode full."
            )

    # Configure logging — errors always shown, -v adds per-call progress
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("loanville.los_adapter").setLevel(
        logging.INFO if args.verbose else logging.ERROR
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if args.compare:
        print("=" * 70)
        print("  LOANVILLE — MODEL SIZE COMPARISON")
        print("=" * 70)
        run_compare(args.mix)
        print("Comparison complete.\n")
        return

    mock = args.mock

    economics = ECONOMICS_PRESETS[args.economics]

    scenario_overrides = {}
    scenario_source = None
    scenario_lenders = None
    if args.scenario:
        builtin = get_scenario(args.scenario)
        if builtin is not None:
            scenario_source = f"builtin:{args.scenario}"
            scenario_overrides = builtin.get("season", builtin)
            scenario_lenders = builtin.get("lenders")
        else:
            try:
                scenario_overrides, scenario_path = load_scenario_preset(args.scenario)
                scenario_source = _format_source(scenario_path)
            except (ValueError, RuntimeError) as exc:
                parser.error(str(exc))

    data_mode = args.data_mode or scenario_overrides.get("data_mode", SEASON_DEFAULTS["data_mode"])
    info_asymmetry = (
        args.info_asymmetry
        or scenario_overrides.get("info_asymmetry", SEASON_DEFAULTS["info_asymmetry"])
    )

    lenders = scenario_lenders or get_lenders()
    lenders_source = None
    if args.lenders:
        try:
            lender_assignments, lenders_path = load_lender_preset(args.lenders)
            lenders = apply_lender_preset(get_lenders(), lender_assignments)
            lenders_source = _format_source(lenders_path)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    if args.crm_sim:
        from .crm_sim import (
            CRMSimulationConfig,
            print_crm_simulation_report,
            run_crm_simulation,
        )

        if lenders_source:
            print(f"  Lender preset: {lenders_source}")

        crm_config = CRMSimulationConfig(
            cases=args.crm_cases,
            mix=args.crm_mix,
            seed=args.seed if args.seed is not None else SEASON_DEFAULTS["seed"],
            incomplete_ratio=args.crm_incomplete_ratio,
            max_concurrent_per_lender=args.crm_concurrency,
            apr_tolerance_pct=args.crm_apr_tolerance,
            los_url=args.los_url,
            los_provider=args.los_provider,
            los_model=args.los_model,
            require_formal_offer_trace=formal_los_only,
        )
        report = asyncio.run(run_crm_simulation(config=crm_config, lenders=lenders))
        print_crm_simulation_report(report)

        if args.crm_export:
            import json
            with open(args.crm_export, "w") as f:
                json.dump(report, f, indent=2)
            print(f"CRM simulation summary exported to: {args.crm_export}")
        return

    if args.agent_build:
        from .agent_docker import build_image
        build_image()
        return

    if args.agent_docker:
        from .agent_docker import run_agent_docker
        from .agent_tasks import get_task
        from .data import get_borrowers

        admin_key = os.environ.get("OR_ADMIN_KEY", "")
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not admin_key:
            parser.error("--agent-docker requires OR_ADMIN_KEY environment variable")
        if not args.agent_budget:
            parser.error("--agent-docker requires --agent-budget")

        task_def = get_task(args.agent_tasks.split(",")[0].strip())
        borrowers = get_borrowers(args.mix, seed=args.seed if args.seed is not None else 42)
        if len(borrowers) > args.agent_cases:
            borrowers = borrowers[:args.agent_cases]

        # Build (model_id, alias) list from lenders
        models = []
        for l in lenders:
            model = args.los_model or l.model
            alias = l.id[:12]
            models.append((model, alias))

        run_agent_docker(
            models=models,
            lenders=lenders,
            borrowers=borrowers,
            task_prompt=task_def.task_prompt,
            los_url=args.los_url,
            budget_usd=args.agent_budget,
            admin_key=admin_key,
            or_key=or_key,
            parallel=args.agent_parallel,
        )
        return

    if args.agent_sim:
        from .agent_sim import AgentSimConfig, run_agent_sim

        agent_config = AgentSimConfig(
            mode=args.agent_mode,
            max_turns=args.agent_max_turns,
            tasks=[t.strip() for t in args.agent_tasks.split(",")],
            cases=args.agent_cases,
            los_url=args.los_url,
            provider=args.los_provider,
            mix=args.mix,
            seed=args.seed if args.seed is not None else 42,
            los_model=args.los_model,
            budget_usd=args.agent_budget,
            parallel=args.agent_parallel,
            eject=args.agent_eject,
        )
        asyncio.run(run_agent_sim(config=agent_config, lenders=lenders))
        return

    if season_mode:
        weeks = args.weeks if args.weeks is not None else scenario_overrides.get("weeks", SEASON_DEFAULTS["weeks"])
        cohort_size = (
            args.cohort_size
            if args.cohort_size is not None
            else scenario_overrides.get("cohort_size", SEASON_DEFAULTS["cohort_size"])
        )
        months_per_week = (
            args.months_per_week
            if args.months_per_week is not None
            else scenario_overrides.get("months_per_week", SEASON_DEFAULTS["months_per_week"])
        )
        season_mix = (
            args.season_mix
            or scenario_overrides.get("season_mix", SEASON_DEFAULTS["season_mix"])
        )
        season_seed = (
            args.seed
            if args.seed is not None
            else scenario_overrides.get("seed", SEASON_DEFAULTS["seed"])
        )
        arrival_phases = (
            args.arrival_phases
            if args.arrival_phases is not None
            else scenario_overrides.get("arrival_phases", SEASON_DEFAULTS["arrival_phases"])
        )
        deep_uw_slots = (
            args.deep_uw_slots
            if args.deep_uw_slots is not None
            else scenario_overrides.get(
                "deep_uw_slots_per_week",
                SEASON_DEFAULTS["deep_uw_slots_per_week"],
            )
        )
        borrower_patience_weeks = (
            args.borrower_patience_weeks
            if args.borrower_patience_weeks is not None
            else scenario_overrides.get(
                "borrower_patience_weeks",
                SEASON_DEFAULTS["borrower_patience_weeks"],
            )
        )
        offer_validity_weeks = (
            args.offer_validity_weeks
            if args.offer_validity_weeks is not None
            else scenario_overrides.get(
                "offer_validity_weeks",
                SEASON_DEFAULTS["offer_validity_weeks"],
            )
        )
        speed_scoring = (
            bool(scenario_overrides.get("speed_scoring", SEASON_DEFAULTS["speed_scoring"]))
            or args.speed_scoring
        )
        custom_tools = (
            bool(scenario_overrides.get("custom_tools", SEASON_DEFAULTS["custom_tools"]))
            or args.custom_tools
        )

        # Parse los_config from scenario overrides
        los_config_raw = scenario_overrides.get("los_config", {})
        los_config = LosConfig(
            disabled_guards=los_config_raw.get("disabled_guards", []),
            gate_policies=los_config_raw.get("gate_policies", []),
        ) if los_config_raw else LosConfig()

        season_config = SeasonConfig(
            weeks=weeks,
            cohort_size=cohort_size,
            months_per_week=months_per_week,
            season_mix=season_mix,
            seed=season_seed,
            speed_scoring=speed_scoring,
            custom_tools=custom_tools,
            economics=economics,
            los_config=los_config,
            arrival_phases=arrival_phases,
            deep_uw_slots_per_week=deep_uw_slots,
            info_asymmetry=info_asymmetry,
            borrower_patience_weeks=borrower_patience_weeks,
            offer_validity_weeks=offer_validity_weeks,
        )

        if scenario_source:
            print(f"  Scenario preset: {scenario_source}")
        if lenders_source:
            print(f"  Lender preset: {lenders_source}")

        season = SeasonEngine(
            config=season_config,
            lenders=lenders,
            mock=mock,
            data_mode=data_mode,
            los_url=args.los_url,
            los_provider=args.los_provider,
            los_mode=args.los_mode,
            underwrite_only=args.underwrite_only,
            los_model=args.los_model,
            formal_los_only=formal_los_only,
        )

        # Preflight checks — validate LOS, credits, and models before burning evaluations
        if args.los_mode == "full" and not mock and not args.skip_preflight:
            from .los_adapter import preflight_season
            total_evals = weeks * cohort_size * len(lenders)
            asyncio.run(preflight_season(
                lenders=lenders,
                los_url=args.los_url,
                provider=args.los_provider,
                total_evaluations=total_evals,
            ))

        asyncio.run(season.run_season())

        # Score and report
        season_scores = score_season(season.lender_states, season_config)
        print_season_report(season_scores)

        # Leaderboard integration for season mode
        if args.leaderboard:
            from .leaderboard import (
                compute_leaderboard,
                emit_match_record_from_season,
                emit_match_records_from_season_weeks,
                write_leaderboard,
                write_match_record,
            )
            week_records = emit_match_records_from_season_weeks(
                season_engine=season,
                lenders=lenders,
                mix=season_mix,
            )
            aggregate_record = emit_match_record_from_season(
                season_engine=season,
                season_scores=season_scores,
                lenders=lenders,
                mix=season_mix,
            )

            weekly_paths = [write_match_record(r) for r in week_records]
            aggregate_path = write_match_record(aggregate_record)
            lb_path = write_leaderboard(compute_leaderboard())

            mock_tag = " [MOCK]" if mock else ""
            print(f"\n  Leaderboard{mock_tag}: wrote {len(weekly_paths)} weekly match record(s)")
            if weekly_paths:
                print(f"  Leaderboard{mock_tag}: latest weekly -> {weekly_paths[-1].name}")
            print(f"  Leaderboard{mock_tag}: aggregate -> {aggregate_path.name}")
            print(f"  Leaderboard: standings -> {lb_path.name}")

        # JSON export for web viewer
        import json
        from datetime import datetime
        season_json = season.to_json()

        # Save to web/seasons/ with timestamped name and rebuild index
        web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
        seasons_dir = os.path.join(web_dir, "seasons")
        if os.path.isdir(web_dir):
            os.makedirs(seasons_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            lender_names = "-".join(l["name"].split()[0] for l in season_json.get("lenders", [])[:3])
            mode_tag = "_mock" if mock else ""
            filename = f"{ts}_{lender_names}{mode_tag}.json"
            filepath = os.path.join(seasons_dir, filename)
            with open(filepath, "w") as f:
                json.dump(season_json, f, indent=2)
            # Rebuild index: list all .json files sorted newest first
            all_files = sorted(
                [f for f in os.listdir(seasons_dir) if f.endswith(".json") and f != "index.json"],
                reverse=True,
            )
            with open(os.path.join(seasons_dir, "index.json"), "w") as f:
                json.dump(all_files, f, indent=2)
            print(f"\nSeason data saved to: seasons/{filename}")

        if args.export:
            with open(args.export, "w") as f:
                json.dump(season_json, f, indent=2)
            print(f"Season data also exported to: {args.export}")

        print("\nSeason complete.\n")
        return

    print("=" * 70)
    print("  LOANVILLE — THE LLM LENDING SIMULATOR")
    if mock:
        print("  [MOCK MODE]")
    else:
        uw_flag = " underwrite-only" if args.underwrite_only else ""
        model_flag = f" model={args.los_model}" if args.los_model else ""
        print(f"  [LOS MODE{uw_flag}] → {args.los_url} (mode={args.los_mode}{model_flag})")
    if formal_los_only:
        print("  Formal interactions: LOS-only enforced")
    else:
        print("  Formal interactions: legacy mode allowed")
    print(f"  Economics: {economics.name}")
    print("=" * 70)

    borrowers = get_borrowers(args.mix, seed=args.seed)

    if lenders_source:
        print(f"  Lender preset: {lenders_source}")

    print(f"\nLoaded {len(borrowers)} borrower applications")
    _print_mix_info(args.mix, borrowers)
    print(f"Loaded {len(lenders)} competing lenders:\n")
    for l in lenders:
        deployed = sum(x.remaining_balance for x in l.existing_portfolio)
        print(f"  {l.name} ({l.model})")
        print(f"    Capital: ${l.total_capital:,.0f} | "
              f"Deployed: ${deployed:,.0f} | "
              f"Target Yield: {l.target_yield_pct}%")

    scores, engine = _run_single(
        borrowers, lenders, mock=mock, data_mode=data_mode,
        los_url=args.los_url,
        los_provider=args.los_provider, los_mode=args.los_mode,
        underwrite_only=args.underwrite_only, los_model=args.los_model,
        economics=economics,
        info_asymmetry=info_asymmetry,
        formal_los_only=formal_los_only,
    )

    # Emit match record to leaderboard
    if args.leaderboard:
        from .leaderboard import emit_match_record_from_sim, emit_and_update

        # Use raw model IDs; dedup only when the same model appears in multiple slots
        seen = {}
        models_info = []
        for l in lenders:
            raw = l.model
            seen[raw] = seen.get(raw, 0) + 1
            mid = raw if seen[raw] == 1 else f"{raw}::{seen[raw]}"
            models_info.append({"model_id": mid, "display_name": l.name})
        record = emit_match_record_from_sim(
            lenders=lenders,
            models_info=models_info,
            engine=engine,
            scores=scores,
            borrowers=borrowers,
            mix=args.mix,
        )
        match_path, lb_path = emit_and_update(record)
        mock_tag = " [MOCK]" if mock else ""
        print(f"\n  Leaderboard{mock_tag}: match → {match_path.name}")
        print(f"  Leaderboard{mock_tag}: standings → {lb_path.name}")

    # Export to web viewer (same format as season mode)
    import json
    from datetime import datetime
    sim_json = engine.to_json()
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    seasons_dir = os.path.join(web_dir, "seasons")
    if os.path.isdir(web_dir):
        os.makedirs(seasons_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lender_names = "-".join(l.name.split()[0] for l in lenders[:3])
        mode_tag = "mock" if mock else args.los_provider
        filename = f"{ts}_{lender_names}_{mode_tag}.json"
        filepath = os.path.join(seasons_dir, filename)
        with open(filepath, "w") as f:
            json.dump(sim_json, f, indent=2)
        # Rebuild index
        all_files = sorted(
            [f for f in os.listdir(seasons_dir) if f.endswith(".json") and f != "index.json"],
            reverse=True,
        )
        with open(os.path.join(seasons_dir, "index.json"), "w") as f:
            json.dump(all_files, f, indent=2)
        print(f"\nWeb viewer: seasons/{filename}")

    if args.export:
        import json as json_mod
        with open(args.export, "w") as f:
            json_mod.dump(engine.to_json(), f, indent=2)
        print(f"Data exported to: {args.export}")

    print("\nSimulation complete.\n")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
