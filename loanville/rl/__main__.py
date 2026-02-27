"""
CLI entry point for the RL pipeline.

Usage::

    python -m loanville.rl generate --mix easy --seed 42
    python -m loanville.rl evaluate --from artifacts/
    python -m loanville.rl score --from artifacts/ --economics aggressive
    python -m loanville.rl export --from artifacts/ --format harbor
    python -m loanville.rl run --mix easy --seed 42   # all stages
    python -m loanville.rl cache-stats
    python -m loanville.rl cache-clear

Each stage persists its output.  Later stages load from disk,
skipping expensive LLM calls.
"""

import argparse
import json
import sys

from .pipeline import Pipeline, PipelineConfig
from .cache import ResponseCache


def cmd_generate(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        mix=args.mix,
        n_borrowers=args.n_borrowers,
        seed=args.seed,
        artifacts_dir=args.out,
    )
    pipeline = Pipeline(config)
    result = pipeline.generate()
    print(f"  Stage: generate ({result.duration_ms:.0f}ms)")
    print(f"  Borrowers: {result.data['n_borrowers']}")
    print(f"  Lenders: {result.data['n_lenders']}")
    print(f"  Artifacts: {args.out}/generated/")


def cmd_evaluate(args: argparse.Namespace) -> None:
    pipeline = Pipeline.load(args.source)
    pipeline.config.use_cache = not args.no_cache
    pipeline.config.cache_dir = args.cache_dir
    pipeline.config.mock = not args.los

    # Ensure generate stage has run
    if not (pipeline.artifacts_dir / "generated" / "manifest.json").exists():
        print("  Running generate stage first...")
        pipeline.generate()

    result = pipeline.evaluate()
    hits = result.data.get("cache_hits", 0)
    misses = result.data.get("cache_misses", 0)
    total = hits + misses
    print(f"  Stage: evaluate ({result.duration_ms:.0f}ms)")
    print(f"  Decisions: {result.data['n_decisions']}")
    if total > 0:
        print(f"  Cache: {hits}/{total} hits ({hits/total*100:.0f}%)")
    print(f"  Artifacts: {args.source}/evaluated/")


def cmd_resolve(args: argparse.Namespace) -> None:
    pipeline = Pipeline.load(args.source)

    # Ensure evaluate stage has run
    if not (pipeline.artifacts_dir / "evaluated" / "decisions.json").exists():
        print("  Running evaluate stage first...")
        pipeline.config.mock = True
        pipeline.evaluate()

    result = pipeline.resolve()
    print(f"  Stage: resolve ({result.duration_ms:.0f}ms)")
    print(f"  Booked loans: {result.data['n_booked']}")
    print(f"  Outcomes: {result.data['n_outcomes']}")
    print(f"  Rollouts: {result.data['n_rollouts']}")
    print(f"  Artifacts: {args.source}/resolved/")


def cmd_score(args: argparse.Namespace) -> None:
    pipeline = Pipeline.load(args.source)

    # Ensure resolve stage has run
    if not (pipeline.artifacts_dir / "rollouts").exists():
        print("  Running resolve stage first...")
        pipeline.resolve()

    rubric = None
    if args.rubric_weights:
        from .verifier import (
            Rubric,
            CreditQualityVerifier,
            PortfolioVerifier,
            GateVerifier,
        )
        weights = json.loads(args.rubric_weights)
        rubric = Rubric()
        rubric.add(CreditQualityVerifier(), weight=weights.get("credit", 0.7), name="credit_quality")
        rubric.add(PortfolioVerifier(), weight=weights.get("portfolio", 0.2), name="portfolio_mgmt")
        rubric.add(GateVerifier(), weight=weights.get("gates", 0.1), name="gates")

    result = pipeline.score(rubric=rubric, economics=args.economics)
    print(f"  Stage: score ({result.duration_ms:.0f}ms)")
    print(f"  Economics: {args.economics}")
    print(f"  Scored: {result.data['n_scored']} lenders")
    for agent_id, score in result.data.get("scores", {}).items():
        print(f"    {agent_id}: {score:.4f}")
    print(f"  Artifacts: {args.source}/scored/")


def cmd_export(args: argparse.Namespace) -> None:
    pipeline = Pipeline.load(args.source)
    result = pipeline.export(format=args.format, output_dir=args.export_dir)
    print(f"  Stage: export ({result.duration_ms:.0f}ms)")
    print(f"  Format: {args.format}")
    for k, v in result.data.items():
        if k != "format":
            print(f"  {k}: {v}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run all stages."""
    config = PipelineConfig(
        mix=args.mix,
        n_borrowers=args.n_borrowers,
        seed=args.seed,
        artifacts_dir=args.out,
        mock=not args.los,
        economics=args.economics,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    pipeline = Pipeline(config)
    results = pipeline.run_all()

    print(f"\n  Pipeline complete ({sum(r.duration_ms for r in results):.0f}ms total)")
    for r in results:
        print(f"    {r.stage}: {r.duration_ms:.0f}ms")
    print(f"\n  Artifacts: {args.out}/")


def cmd_cache_stats(args: argparse.Namespace) -> None:
    cache = ResponseCache(args.cache_dir)
    size = cache.size()
    print(f"  Cache: {args.cache_dir}")
    print(f"  Entries: {size}")


def cmd_cache_clear(args: argparse.Namespace) -> None:
    cache = ResponseCache(args.cache_dir)
    removed = cache.clear()
    print(f"  Cleared {removed} cached entries from {args.cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m loanville.rl",
        description="Loanville RL Pipeline — staged evaluation with caching",
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    p_gen = sub.add_parser("generate", help="Stage 1: Generate borrower tasks")
    p_gen.add_argument("--mix", default="easy", help="Borrower mix preset")
    p_gen.add_argument("--n-borrowers", type=int, default=None, help="Limit borrower count")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--out", default="artifacts", help="Output directory")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Stage 2: Run LLM evaluations (cached)")
    p_eval.add_argument("--from", dest="source", default="artifacts", help="Artifacts directory")
    p_eval.add_argument("--no-cache", action="store_true", help="Disable response cache")
    p_eval.add_argument("--cache-dir", default="cache/responses", help="Cache directory")
    p_eval.add_argument("--los", action="store_true", help="Use LOS mode (default: mock)")

    # resolve
    p_res = sub.add_parser("resolve", help="Stage 3: Adjudicate + resolve loans")
    p_res.add_argument("--from", dest="source", default="artifacts", help="Artifacts directory")

    # score
    p_score = sub.add_parser("score", help="Stage 4: Score with rubric (re-runnable)")
    p_score.add_argument("--from", dest="source", default="artifacts", help="Artifacts directory")
    p_score.add_argument("--economics", default="balanced",
                         help="Economics preset (balanced, aggressive, conservative)")
    p_score.add_argument("--rubric-weights", default=None,
                         help='JSON rubric weights: \'{"credit": 0.7, "portfolio": 0.2, "gates": 0.1}\'')

    # export
    p_export = sub.add_parser("export", help="Stage 5: Export for RL training")
    p_export.add_argument("--from", dest="source", default="artifacts", help="Artifacts directory")
    p_export.add_argument("--format", default="rollouts",
                          choices=["rollouts", "harbor", "verifier_states"],
                          help="Export format")
    p_export.add_argument("--export-dir", default=None, help="Override output directory")

    # run (all stages)
    p_run = sub.add_parser("run", help="Run all stages")
    p_run.add_argument("--mix", default="easy", help="Borrower mix preset")
    p_run.add_argument("--n-borrowers", type=int, default=None, help="Limit borrower count")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--out", default="artifacts", help="Output directory")
    p_run.add_argument("--economics", default="balanced", help="Economics preset")
    p_run.add_argument("--no-cache", action="store_true", help="Disable response cache")
    p_run.add_argument("--cache-dir", default="cache/responses", help="Cache directory")
    p_run.add_argument("--los", action="store_true", help="Use LOS mode")

    # cache management
    sub.add_parser("cache-stats", help="Show cache statistics")
    p_clear = sub.add_parser("cache-clear", help="Clear the response cache")

    # Add cache-dir to cache commands
    for p in [parser]:
        p.add_argument("--cache-dir", default="cache/responses", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "generate": cmd_generate,
        "evaluate": cmd_evaluate,
        "resolve": cmd_resolve,
        "score": cmd_score,
        "export": cmd_export,
        "run": cmd_run,
        "cache-stats": cmd_cache_stats,
        "cache-clear": cmd_cache_clear,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
