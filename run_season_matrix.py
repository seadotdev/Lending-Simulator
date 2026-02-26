#!/usr/bin/env python3
"""
Run a season test matrix across an arbitrary model set.

Example:
  python run_season_matrix.py --runs run1 --models \
    meta-llama/llama-4-maverick \
    nvidia/llama-3.3-nemotron-super-49b-v1.5 \
    nvidia/nemotron-nano-9b-v2 \
    google/gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from loanville.data import get_lenders
from loanville.models import ECONOMICS_PRESETS, SeasonConfig
from loanville.scoring import print_season_report, score_season
from loanville.season import SeasonEngine


DEFAULT_MODELS = [
    "meta-llama/llama-4-maverick",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-nano-9b-v2",
    "google/gemini-2.5-flash",
]

RUN_MATRIX = {
    "run1": {
        "weeks": 5,
        "cohort_size": 5,
        "season_mix": "realistic",
        "economics": "balanced",
        "speed_scoring": False,
        "info_asymmetry": "none",
        "arrival_phases": 1,
        "deep_uw_slots_per_week": 0,
    },
    "run2": {
        "weeks": 10,
        "cohort_size": 15,
        "season_mix": "realistic",
        "economics": "balanced",
        "speed_scoring": False,
        "info_asymmetry": "none",
        "arrival_phases": 1,
        "deep_uw_slots_per_week": 0,
    },
    "run3": {
        "weeks": 15,
        "cohort_size": 10,
        "season_mix": "escalating",
        "economics": "balanced",
        "speed_scoring": False,
        "info_asymmetry": "none",
        "arrival_phases": 1,
        "deep_uw_slots_per_week": 0,
    },
    "run4": {
        "weeks": 15,
        "cohort_size": 10,
        "season_mix": "escalating",
        "economics": "conservative",
        "speed_scoring": True,
        "info_asymmetry": "partial_statements",
        "arrival_phases": 2,
        "deep_uw_slots_per_week": 3,
    },
    "run5": {
        "weeks": 20,
        "cohort_size": 20,
        "season_mix": "stress",
        "economics": "aggressive",
        "speed_scoring": True,
        "info_asymmetry": "none",
        "arrival_phases": 3,
        "deep_uw_slots_per_week": 0,
    },
}


def _build_lenders(model_ids: list[str]):
    lenders = get_lenders()
    if len(model_ids) > len(lenders):
        raise ValueError(
            f"Requested {len(model_ids)} models but only {len(lenders)} lender slots exist."
        )
    for idx, model_id in enumerate(model_ids):
        lenders[idx].model = model_id
        alias = model_id.split("/")[-1]
        base_name = lenders[idx].name.split("[")[0].strip()
        lenders[idx].name = f"{base_name} [{alias}]"
    return lenders[: len(model_ids)]


def _run_metrics(run_id: str, season: SeasonEngine, elapsed_s: float, run_cfg: dict) -> dict:
    lenders = []
    eliminated = 0
    for state in season.lender_states.values():
        evaluations = state.total_evaluations
        error_rate = (state.deals_errored / evaluations) if evaluations else 0.0
        approve_rate = (state.deals_won + state.deals_lost) / evaluations if evaluations else 0.0
        deals_not_won = max(0, evaluations - state.deals_won - state.deals_errored)
        if state.eliminated:
            eliminated += 1
        lenders.append(
            {
                "lender_id": state.lender_id,
                "name": state.lender_name,
                "model": state.model,
                "evaluations": evaluations,
                "deals_won": state.deals_won,
                "deals_lost": state.deals_lost,
                "deals_rejected": state.deals_rejected,
                "deals_not_won": deals_not_won,
                "deals_errored": state.deals_errored,
                "error_rate": round(error_rate, 4),
                "approve_rate": round(approve_rate, 4),
                "speed_wins": state.speed_wins,
                "deep_uw_deferred": state.deep_uw_deferred,
                "tokens_in": state.cumulative_tokens_in,
                "tokens_out": state.cumulative_tokens_out,
                "cost_usd": round(state.cumulative_cost_usd, 4),
                "eliminated": state.eliminated,
                "eliminated_week": state.eliminated_week,
            }
        )

    return {
        "run_id": run_id,
        "config": run_cfg,
        "elapsed_seconds": round(elapsed_s, 2),
        "lenders_eliminated": eliminated,
        "total_tokens_in": sum(l["tokens_in"] for l in lenders),
        "total_tokens_out": sum(l["tokens_out"] for l in lenders),
        "total_cost_usd": round(sum(l["cost_usd"] for l in lenders), 4),
        "lenders": lenders,
    }


def _print_run_banner(run_id: str, cfg: dict, model_ids: list[str]) -> None:
    print("\n" + "=" * 72)
    print(f"  MATRIX {run_id.upper()} | {cfg['weeks']}w x {cfg['cohort_size']} cohort | mix={cfg['season_mix']}")
    print(
        f"  econ={cfg['economics']} speed={cfg['speed_scoring']} "
        f"asym={cfg['info_asymmetry']} phases={cfg['arrival_phases']} deep_slots={cfg['deep_uw_slots_per_week']}"
    )
    print(f"  models ({len(model_ids)}):")
    for mid in model_ids:
        print(f"    - {mid}")
    print("=" * 72)


def _rebuild_web_index(seasons_dir: Path) -> None:
    season_files = sorted(
        [p.name for p in seasons_dir.glob("*.json") if p.name != "index.json"],
        reverse=True,
    )
    with (seasons_dir / "index.json").open("w") as f:
        json.dump(season_files, f, indent=2)


def _publish_season_to_web(
    season_path: Path,
    run_id: str,
    season_json: dict,
    seasons_dir: Path,
) -> Path:
    seasons_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lender_names = "-".join(
        str(l.get("model", "")).split("/")[-1]
        for l in season_json.get("lenders", [])[:3]
    )
    filename = f"{ts}_{run_id}_{lender_names}.json"
    dest = seasons_dir / filename
    shutil.copy2(season_path, dest)
    _rebuild_web_index(seasons_dir)
    return dest


def _drop_high_error_models(
    record: dict,
    max_error_rate: float,
) -> tuple[dict | None, int]:
    from loanville.leaderboard import build_match_record

    keep_results = []
    keep_ids = set()
    dropped = 0
    for r in record.get("results", []):
        total = (
            int(r.get("deals_won", 0))
            + int(r.get("deals_rejected", 0))
            + int(r.get("deals_errored", 0))
        )
        errored = int(r.get("deals_errored", 0))
        if total > 0 and (errored / total) > max_error_rate:
            dropped += 1
            continue
        keep_results.append(r)
        keep_ids.add(r.get("model_id"))

    if len(keep_results) < 2:
        return None, dropped

    keep_models = [
        m for m in record.get("models", [])
        if m.get("model_id") in keep_ids
    ]
    filtered = build_match_record(
        models=keep_models,
        results=keep_results,
        mix=str(record.get("mix", "")),
        n_borrowers=int(record.get("n_borrowers", 0)),
    )
    return filtered, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run season matrix scenarios against chosen models.")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(RUN_MATRIX.keys()),
        default=["run1"],
        help="Matrix run ids to execute in order (default: run1).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="OpenRouter model ids mapped to lender slots.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Season random seed.")
    parser.add_argument("--months-per-week", type=int, default=2, help="Loan aging months per week.")
    parser.add_argument(
        "--data-mode",
        choices=["full", "quarterly_only", "aggregate_only", "statements_inline", "lite"],
        default="lite",
        help="Borrower data mode used during underwriting.",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock underwriting responses.")
    parser.add_argument("--los-url", default="http://localhost:3000", help="Open LOS base URL.")
    parser.add_argument(
        "--los-provider",
        choices=["openrouter", "anthropic", "openai", "vercel"],
        default="openrouter",
        help="LLM provider passed to Open LOS.",
    )
    parser.add_argument(
        "--los-mode",
        choices=["full", "rules_only"],
        default="full",
        help="LOS evaluation mode when not using --underwrite-only.",
    )
    parser.add_argument(
        "--underwrite-only",
        action="store_true",
        help="Use /v1/underwrite path instead of full LOS ceremony.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/season-matrix",
        help="Directory for per-run JSON outputs and summary.",
    )
    parser.add_argument(
        "--web-export",
        action="store_true",
        help="Copy each run's season JSON into web/seasons and rebuild index.json.",
    )
    parser.add_argument(
        "--web-seasons-dir",
        default="web/seasons",
        help="Web seasons directory for viewer exports (default: web/seasons).",
    )
    parser.add_argument(
        "--leaderboard",
        choices=["none", "aggregate", "weekly", "both"],
        default="none",
        help="Publish leaderboard match records from each run (default: none).",
    )
    parser.add_argument(
        "--leaderboard-auto-drop-high-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a record is invalid due model error-rate, auto-drop over-threshold models and publish remaining valid models.",
    )
    args = parser.parse_args()

    load_dotenv(os.path.expanduser("~/.env"))
    load_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    web_seasons_dir = Path(args.web_seasons_dir)

    lb_max_error_rate = 0.25
    if args.leaderboard != "none":
        from loanville.leaderboard import load_config

        cfg = load_config()
        lb_max_error_rate = float(cfg.get("max_error_rate", 0.25))

    all_results = []
    for run_id in args.runs:
        run_cfg = dict(RUN_MATRIX[run_id])
        _print_run_banner(run_id, run_cfg, args.models)

        lenders = _build_lenders(args.models)
        config = SeasonConfig(
            weeks=run_cfg["weeks"],
            cohort_size=run_cfg["cohort_size"],
            months_per_week=args.months_per_week,
            season_mix=run_cfg["season_mix"],
            seed=args.seed,
            speed_scoring=run_cfg["speed_scoring"],
            custom_tools=False,
            economics=ECONOMICS_PRESETS[run_cfg["economics"]],
            info_asymmetry=run_cfg["info_asymmetry"],
            arrival_phases=run_cfg["arrival_phases"],
            deep_uw_slots_per_week=run_cfg["deep_uw_slots_per_week"],
        )

        t0 = time.time()
        season = SeasonEngine(
            config=config,
            lenders=lenders,
            mock=args.mock,
            data_mode=args.data_mode,
            los_url=args.los_url,
            los_provider=args.los_provider,
            los_mode=args.los_mode,
            underwrite_only=args.underwrite_only,
        )
        asyncio.run(season.run_season())
        elapsed_s = time.time() - t0

        season_scores = score_season(season.lender_states, config)
        print_season_report(season_scores)

        season_json = season.to_json()
        season_path = output_dir / f"{run_id}_season.json"
        with season_path.open("w") as f:
            json.dump(season_json, f, indent=2)

        result = _run_metrics(run_id, season, elapsed_s, run_cfg)
        result["season_json"] = str(season_path)
        metrics_path = output_dir / f"{run_id}_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

        if args.web_export:
            web_path = _publish_season_to_web(
                season_path=season_path,
                run_id=run_id,
                season_json=season_json,
                seasons_dir=web_seasons_dir,
            )
            result["web_viewer_json"] = str(web_path)
            print(f"  Web viewer: {web_path}")

        if args.leaderboard != "none":
            from loanville.leaderboard import (
                compute_leaderboard,
                emit_match_record_from_season,
                emit_match_records_from_season_weeks,
                write_leaderboard,
                write_match_record,
            )

            records = []
            if args.leaderboard in ("weekly", "both"):
                records.extend(
                    emit_match_records_from_season_weeks(
                        season_engine=season,
                        lenders=lenders,
                        mix=run_cfg["season_mix"],
                    )
                )
            if args.leaderboard in ("aggregate", "both"):
                records.append(
                    emit_match_record_from_season(
                        season_engine=season,
                        season_scores=season_scores,
                        lenders=lenders,
                        mix=run_cfg["season_mix"],
                    )
                )

            published = []
            skipped = []
            dropped_total = 0
            for rec in records:
                to_write = rec
                if (
                    not rec.get("validation", {}).get("valid", False)
                    and args.leaderboard_auto_drop_high_error
                ):
                    filtered, dropped = _drop_high_error_models(rec, lb_max_error_rate)
                    if filtered is not None:
                        to_write = filtered
                    dropped_total += dropped
                if to_write.get("validation", {}).get("valid", False):
                    published.append(write_match_record(to_write))
                else:
                    skipped.append(to_write.get("validation", {}).get("errors", []))

            if published:
                lb_path = write_leaderboard(compute_leaderboard())
                result["leaderboard_matches"] = [str(p) for p in published]
                result["leaderboard_file"] = str(lb_path)
                if dropped_total:
                    result["leaderboard_models_dropped"] = dropped_total
                print(f"  Leaderboard: wrote {len(published)} record(s)")
                print(f"  Leaderboard: standings -> {lb_path}")
            if skipped:
                result["leaderboard_skipped"] = skipped
                print(f"  Leaderboard: skipped {len(skipped)} invalid record(s)")

        print(
            f"\n  Saved: {season_path}\n"
            f"  Saved: {metrics_path}\n"
            f"  Elapsed: {elapsed_s:.1f}s | Total cost: ${result['total_cost_usd']:.4f}"
        )

    summary = {
        "runs": args.runs,
        "models": args.models,
        "results": all_results,
    }
    summary_path = output_dir / "matrix_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMatrix complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
