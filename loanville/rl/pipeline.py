"""
Staged pipeline — run Loanville from any point, skip expensive stages.

This is what verifiers and Harbor enable at an architectural level:
each stage persists its output, and the next stage can load from disk
instead of re-running.  The pipeline makes the implicit stages in
Loanville's monolithic ``asyncio.run(engine.run())`` into explicit,
independently-runnable steps.

Stages::

    1. generate   — Create borrower tasks + lender configs (cheap, no LLM)
    2. evaluate   — Run LLM underwriting (expensive, cached for replay)
    3. resolve    — Adjudicate + resolve loans (cheap, deterministic)
    4. score      — Score with rubric/economics (cheap, re-runnable)
    5. export     — Export rollouts, Harbor tasks, or training data

Token-efficiency wins:
  - Stage 2 outputs are cached: re-running with different scoring = FREE
  - Stage 4 can be re-run with different rubrics/economics without LLM calls
  - Stage 5 can export to verifiers/Harbor format for external training

Usage::

    from loanville.rl.pipeline import Pipeline, PipelineConfig

    # Full run (all stages)
    pipeline = Pipeline(PipelineConfig(mix="easy", seed=42))
    pipeline.run_all()

    # Re-score only (stages 4-5, skipping LLM)
    pipeline = Pipeline.load("artifacts/run_20240101")
    pipeline.score(rubric=my_custom_rubric)
    pipeline.export(format="rollouts")

CLI::

    python -m loanville.rl generate --mix easy --seed 42 --out artifacts/
    python -m loanville.rl evaluate --from artifacts/ --cache cache/
    python -m loanville.rl score --from artifacts/ --economics aggressive
    python -m loanville.rl export --from artifacts/ --format harbor --out tasks/
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..data import get_borrowers, get_lenders, MIX_PRESETS
from ..models import (
    Borrower,
    EconomicsConfig,
    ECONOMICS_PRESETS,
    LenderConfig,
    LenderDecision,
)
from .cache import ResponseCache, make_cache_key
from .rollout import Rollout, RolloutLogger
from .verifier import Rubric, build_default_rubric


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""
    # Borrower generation
    mix: str = "easy"
    n_borrowers: Optional[int] = None  # None = use all from mix
    seed: int = 42
    # Evaluation
    mock: bool = True
    data_mode: str = "full"
    use_cache: bool = True
    cache_dir: str = "cache/responses"
    # Scoring
    economics: str = "balanced"
    # Output
    artifacts_dir: str = "artifacts"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage artifacts (what each stage persists)
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """What a pipeline stage produces."""
    stage: str
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Staged pipeline for Loanville evaluation and RL data generation.

    Each stage persists its output to disk.  Later stages can load from
    the persisted artifacts, skipping expensive earlier stages.

    This is the Loanville equivalent of:
    - verifiers: Dataset → Rollout → Score → Train
    - Harbor: Task → Agent → Verify → Report
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Stage outputs (populated as stages run)
        self.borrowers: list[Borrower] = []
        self.lenders: list[LenderConfig] = []
        self.decisions: dict[str, list[LenderDecision]] = {}
        self.rollouts: list[Rollout] = []
        self.scores: dict[str, Any] = {}

        # Cache
        self._cache: Optional[ResponseCache] = None
        if self.config.use_cache:
            self._cache = ResponseCache(self.config.cache_dir)

    # ------------------------------------------------------------------
    # Stage 1: Generate
    # ------------------------------------------------------------------

    def generate(self) -> StageResult:
        """Stage 1: Generate borrower tasks and lender configs.

        Cheap, no LLM calls.  Persists borrowers + lenders to disk.
        """
        t0 = time.monotonic()

        self.borrowers = get_borrowers(self.config.mix, seed=self.config.seed)
        if self.config.n_borrowers:
            self.borrowers = self.borrowers[:self.config.n_borrowers]
        self.lenders = get_lenders()

        # Persist
        gen_dir = self.artifacts_dir / "generated"
        gen_dir.mkdir(exist_ok=True)

        borrowers_data = []
        for b in self.borrowers:
            d = b.dossier
            borrowers_data.append({
                "id": b.id,
                "company_name": d.company_name,
                "sector": d.sector,
                "annual_revenue": d.annual_revenue,
                "net_income": d.net_income,
                "loan_request_amount": d.loan_request_amount,
                "loan_purpose": d.loan_purpose,
                "true_outcome": b.true_outcome,
                "months_before_default": b.months_before_default,
            })

        lenders_data = []
        for l in self.lenders:
            lenders_data.append({
                "id": l.id,
                "name": l.name,
                "model": l.model,
                "total_capital": l.total_capital,
                "target_yield_pct": l.target_yield_pct,
            })

        manifest = {
            "config": self.config.to_dict(),
            "n_borrowers": len(self.borrowers),
            "n_lenders": len(self.lenders),
            "borrowers": borrowers_data,
            "lenders": lenders_data,
        }
        (gen_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        elapsed = (time.monotonic() - t0) * 1000
        return StageResult(
            stage="generate",
            duration_ms=elapsed,
            data={
                "n_borrowers": len(self.borrowers),
                "n_lenders": len(self.lenders),
            },
        )

    # ------------------------------------------------------------------
    # Stage 2: Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> StageResult:
        """Stage 2: Run LLM underwriting evaluations.

        This is the expensive stage.  Results are cached by default so
        re-running with the same borrowers + models serves from cache.

        Requires: Stage 1 (generate) has been run, or borrowers are loaded.
        """
        t0 = time.monotonic()

        if not self.borrowers:
            self._load_generated()

        from ..mock_llm import mock_evaluate_all

        cache_hits = 0
        cache_misses = 0

        if self.config.mock:
            # Check cache first
            if self._cache:
                cached_decisions: dict[str, list[LenderDecision]] = {}
                uncached_borrowers: list[Borrower] = []
                all_cached = True

                for lender in self.lenders:
                    cached_decisions[lender.id] = []
                    for borrower in self.borrowers:
                        dossier_dict = _dossier_to_dict(borrower)
                        key = make_cache_key(
                            model=lender.model,
                            borrower_id=borrower.id,
                            dossier=dossier_dict,
                            data_mode=self.config.data_mode,
                        )
                        entry = self._cache.get(key)
                        if entry is not None:
                            cache_hits += 1
                            ts = None
                            if entry.term_sheet:
                                from ..models import TermSheet
                                ts = TermSheet(
                                    loan_amount=entry.term_sheet.get("loan_amount", 0),
                                    interest_rate=entry.term_sheet.get("interest_rate", 0),
                                    term_months=entry.term_sheet.get("term_months", 24),
                                )
                            cached_decisions[lender.id].append(LenderDecision(
                                lender_id=lender.id,
                                borrower_id=borrower.id,
                                decision=entry.decision,
                                reasoning=entry.reasoning,
                                term_sheet=ts,
                            ))
                        else:
                            cache_misses += 1
                            all_cached = False

                if all_cached:
                    self.decisions = cached_decisions
                else:
                    # Cache miss: run full mock eval, then cache results
                    self.decisions = mock_evaluate_all(
                        self.lenders, self.borrowers, self.config.data_mode,
                    )
                    self._cache_all_decisions()
            else:
                self.decisions = mock_evaluate_all(
                    self.lenders, self.borrowers, self.config.data_mode,
                )
        else:
            # LOS mode: would need async runner
            # For now, fall through to mock with a warning
            import warnings
            warnings.warn(
                "Pipeline.evaluate() currently only supports mock mode. "
                "For LOS mode, use the main CLI: python -m loanville",
                stacklevel=2,
            )
            self.decisions = mock_evaluate_all(
                self.lenders, self.borrowers, self.config.data_mode,
            )
            if self._cache:
                self._cache_all_decisions()

        # Persist decisions
        eval_dir = self.artifacts_dir / "evaluated"
        eval_dir.mkdir(exist_ok=True)

        decisions_data: dict[str, list[dict]] = {}
        for lender_id, decs in self.decisions.items():
            decisions_data[lender_id] = [
                {
                    "borrower_id": d.borrower_id,
                    "decision": d.decision,
                    "reasoning": d.reasoning,
                    "term_sheet": {
                        "loan_amount": d.term_sheet.loan_amount,
                        "interest_rate": d.term_sheet.interest_rate,
                        "term_months": d.term_sheet.term_months,
                    } if d.term_sheet else None,
                }
                for d in decs
            ]
        (eval_dir / "decisions.json").write_text(json.dumps(decisions_data, indent=2))

        elapsed = (time.monotonic() - t0) * 1000
        return StageResult(
            stage="evaluate",
            duration_ms=elapsed,
            data={
                "n_decisions": sum(len(d) for d in self.decisions.values()),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            },
        )

    def _cache_all_decisions(self) -> None:
        """Write all current decisions to the response cache."""
        if not self._cache:
            return
        for lender in self.lenders:
            for borrower in self.borrowers:
                dossier_dict = _dossier_to_dict(borrower)
                key = make_cache_key(
                    model=lender.model,
                    borrower_id=borrower.id,
                    dossier=dossier_dict,
                    data_mode=self.config.data_mode,
                )
                dec = next(
                    (d for d in self.decisions.get(lender.id, [])
                     if d.borrower_id == borrower.id),
                    None,
                )
                if dec and not self._cache.has(key):
                    ts_dict = None
                    if dec.term_sheet:
                        ts_dict = {
                            "loan_amount": dec.term_sheet.loan_amount,
                            "interest_rate": dec.term_sheet.interest_rate,
                            "term_months": dec.term_sheet.term_months,
                        }
                    self._cache.put(
                        key=key,
                        model=lender.model,
                        borrower_id=borrower.id,
                        decision=dec.decision,
                        reasoning=dec.reasoning or "",
                        term_sheet=ts_dict,
                        created_from="mock" if self.config.mock else "los",
                    )

    # ------------------------------------------------------------------
    # Stage 3: Resolve (adjudicate + fast-forward loans)
    # ------------------------------------------------------------------

    def resolve(self) -> StageResult:
        """Stage 3: Adjudicate deals and resolve loan outcomes.

        Cheap, deterministic.  Uses the decisions from Stage 2.
        """
        t0 = time.monotonic()

        if not self.decisions:
            self._load_evaluated()

        from ..engine import SimulationEngine

        # Build a SimulationEngine from our artifacts and re-run adjudication + resolution
        engine = SimulationEngine(
            self.borrowers, self.lenders,
            mock=True,  # won't matter, we inject decisions
            data_mode=self.config.data_mode,
            economics=ECONOMICS_PRESETS.get(self.config.economics, EconomicsConfig()),
        )
        # Inject pre-computed decisions (skip LLM stage)
        engine.all_decisions = self.decisions

        # Run just adjudication + resolution
        engine.adjudicate_deals()
        engine.resolve_loans()
        engine._finalize_runs()

        # Extract rollouts
        self.rollouts = engine.to_rollouts()
        self._engine = engine  # Keep for scoring

        # Persist
        resolve_dir = self.artifacts_dir / "resolved"
        resolve_dir.mkdir(exist_ok=True)

        outcomes_data = []
        for outcome in engine.loan_outcomes:
            outcomes_data.append({
                "loan_id": outcome.loan_id,
                "borrower_id": outcome.borrower_id,
                "lender_id": outcome.lender_id,
                "defaulted": outcome.defaulted,
                "was_fraud": outcome.was_fraud,
                "interest_paid": outcome.total_interest_paid,
                "principal_lost": outcome.principal_lost,
            })
        (resolve_dir / "outcomes.json").write_text(json.dumps(outcomes_data, indent=2))

        # Save rollouts
        rollout_logger = RolloutLogger(self.artifacts_dir / "rollouts")
        for rollout in self.rollouts:
            rollout_logger.log(rollout)

        elapsed = (time.monotonic() - t0) * 1000
        return StageResult(
            stage="resolve",
            duration_ms=elapsed,
            data={
                "n_booked": len(engine.booked_loans),
                "n_outcomes": len(engine.loan_outcomes),
                "n_rollouts": len(self.rollouts),
            },
        )

    # ------------------------------------------------------------------
    # Stage 4: Score
    # ------------------------------------------------------------------

    def score(self, rubric: Optional[Rubric] = None, economics: Optional[str] = None) -> StageResult:
        """Stage 4: Score with rubric and economics config.

        Cheap, re-runnable.  Can be called multiple times with different
        rubrics or economics presets without re-running LLM calls.
        """
        t0 = time.monotonic()

        rubric = rubric or build_default_rubric()

        if not self.rollouts:
            self._load_rollouts()

        eco_name = economics or self.config.economics
        eco = ECONOMICS_PRESETS.get(eco_name, EconomicsConfig())

        results: dict[str, Any] = {}
        for rollout in self.rollouts:
            # Build verifier state from rollout
            state = _rollout_to_verifier_state(rollout, eco)
            result = rubric.score(state)

            results[rollout.agent_id] = {
                "terminal_reward": result.reward,
                "rewards": result.rewards,
                "info": result.info,
                "n_steps": rollout.n_steps,
                "cumulative_reward": rollout.cumulative_reward,
            }

        self.scores = results

        # Persist
        score_dir = self.artifacts_dir / "scored"
        score_dir.mkdir(exist_ok=True)
        (score_dir / f"scores_{eco_name}.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

        elapsed = (time.monotonic() - t0) * 1000
        return StageResult(
            stage="score",
            duration_ms=elapsed,
            data={
                "n_scored": len(results),
                "economics": eco_name,
                "scores": {k: v["terminal_reward"] for k, v in results.items()},
            },
        )

    # ------------------------------------------------------------------
    # Stage 5: Export
    # ------------------------------------------------------------------

    def export(self, format: str = "rollouts", output_dir: Optional[str] = None) -> StageResult:
        """Stage 5: Export for RL training or external evaluation.

        Formats:
          - "rollouts": JSON rollout files (for offline RL)
          - "harbor": Harbor task directories (for harbor run / HarborEnv)
          - "verifier_states": State dicts for vf.Rubric.score_rollout()
        """
        t0 = time.monotonic()

        out_dir = Path(output_dir) if output_dir else self.artifacts_dir / "exported"
        out_dir.mkdir(parents=True, exist_ok=True)

        if format == "rollouts":
            if not self.rollouts:
                self._load_rollouts()
            logger = RolloutLogger(out_dir / "rollouts")
            for rollout in self.rollouts:
                logger.log(rollout)
            data = {"n_exported": len(self.rollouts), "format": "rollouts"}

        elif format == "harbor":
            from .tasks import export_harbor_dataset, TaskConfig
            export_harbor_dataset(
                out_dir / "harbor_tasks",
                n_tasks=5,
                base_seed=self.config.seed,
            )
            data = {"format": "harbor", "output": str(out_dir / "harbor_tasks")}

        elif format == "verifier_states":
            if not self.rollouts:
                self._load_rollouts()
            all_states = []
            for rollout in self.rollouts:
                all_states.extend(rollout.to_verifier_states())
            (out_dir / "verifier_states.json").write_text(
                json.dumps(all_states, indent=2, default=str)
            )
            data = {"n_states": len(all_states), "format": "verifier_states"}

        else:
            raise ValueError(f"Unknown export format: {format}")

        elapsed = (time.monotonic() - t0) * 1000
        return StageResult(stage="export", duration_ms=elapsed, data=data)

    # ------------------------------------------------------------------
    # Run all stages
    # ------------------------------------------------------------------

    def run_all(self, rubric: Optional[Rubric] = None) -> list[StageResult]:
        """Run the complete pipeline (all 5 stages)."""
        results = []
        results.append(self.generate())
        results.append(self.evaluate())
        results.append(self.resolve())
        results.append(self.score(rubric=rubric))
        results.append(self.export())
        return results

    # ------------------------------------------------------------------
    # Load from persisted artifacts (enter pipeline mid-stream)
    # ------------------------------------------------------------------

    def _load_generated(self) -> None:
        """Load Stage 1 artifacts from disk."""
        manifest_path = self.artifacts_dir / "generated" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No generated artifacts found at {manifest_path}. "
                f"Run pipeline.generate() first."
            )
        manifest = json.loads(manifest_path.read_text())
        config = manifest.get("config", {})

        # Regenerate borrowers from config (deterministic)
        mix = config.get("mix", self.config.mix)
        seed = config.get("seed", self.config.seed)
        self.borrowers = get_borrowers(mix, seed=seed)
        n = config.get("n_borrowers") or manifest.get("n_borrowers")
        if n:
            self.borrowers = self.borrowers[:n]
        self.lenders = get_lenders()

    def _load_evaluated(self) -> None:
        """Load Stage 2 artifacts from disk."""
        if not self.borrowers:
            self._load_generated()

        decisions_path = self.artifacts_dir / "evaluated" / "decisions.json"
        if not decisions_path.exists():
            raise FileNotFoundError(
                f"No evaluated artifacts found at {decisions_path}. "
                f"Run pipeline.evaluate() first."
            )
        decisions_data = json.loads(decisions_path.read_text())

        from ..models import TermSheet

        self.decisions = {}
        for lender_id, decs in decisions_data.items():
            self.decisions[lender_id] = []
            for d in decs:
                ts = None
                if d.get("term_sheet"):
                    ts = TermSheet(
                        loan_amount=d["term_sheet"]["loan_amount"],
                        interest_rate=d["term_sheet"]["interest_rate"],
                        term_months=d["term_sheet"]["term_months"],
                    )
                self.decisions[lender_id].append(LenderDecision(
                    lender_id=lender_id,
                    borrower_id=d["borrower_id"],
                    decision=d["decision"],
                    reasoning=d.get("reasoning", ""),
                    term_sheet=ts,
                ))

    def _load_rollouts(self) -> None:
        """Load Stage 3 rollout artifacts from disk."""
        rollout_dir = self.artifacts_dir / "rollouts"
        if not rollout_dir.exists():
            raise FileNotFoundError(
                f"No rollout artifacts found at {rollout_dir}. "
                f"Run pipeline.resolve() first."
            )
        logger = RolloutLogger(rollout_dir)
        self.rollouts = logger.load_all()

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> Pipeline:
        """Load a pipeline from persisted artifacts.

        Determines which stages have already run and positions the
        pipeline to resume from the latest completed stage.
        """
        artifacts_dir = Path(artifacts_dir)
        config_path = artifacts_dir / "generated" / "manifest.json"

        if config_path.exists():
            manifest = json.loads(config_path.read_text())
            config = PipelineConfig(**{
                k: v for k, v in manifest.get("config", {}).items()
                if k in PipelineConfig.__dataclass_fields__
            })
            config.artifacts_dir = str(artifacts_dir)
        else:
            config = PipelineConfig(artifacts_dir=str(artifacts_dir))

        return cls(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dossier_to_dict(borrower: Borrower) -> dict[str, Any]:
    """Extract decision-relevant dossier fields as a dict."""
    d = borrower.dossier
    return {
        "company_name": d.company_name,
        "sector": d.sector,
        "annual_revenue": d.annual_revenue,
        "net_income": d.net_income,
        "loan_request_amount": d.loan_request_amount,
        "loan_purpose": d.loan_purpose,
        "years_in_business": d.years_in_business,
        "employee_count": d.employee_count,
    }


def _rollout_to_verifier_state(rollout: Rollout, eco: EconomicsConfig) -> dict[str, Any]:
    """Convert a rollout into a verifier state dict for scoring."""
    # Aggregate from steps
    total_deployed = 0.0
    total_interest = 0.0
    total_losses = 0.0
    deals_won = 0
    deals_rejected = 0
    frauds_funded = 0
    defaults_count = 0

    for step in rollout.steps:
        action = step.action.get("decision", "REJECT")
        if action == "APPROVE":
            deals_won += 1
            rb = step.reward_breakdown
            if rb:
                total_interest += rb.get("interest_paid", rb.get("interest_earned", 0))
                total_losses += rb.get("principal_lost", 0)
                ts = step.action.get("term_sheet", {})
                total_deployed += ts.get("loan_amount", 0)
            gt = step.ground_truth
            if gt.get("true_outcome") == "fraud":
                frauds_funded += 1
            if gt.get("true_outcome") in ("bad", "fraud"):
                defaults_count += 1
        else:
            deals_rejected += 1

    # Fall back to terminal rewards if step-level data is sparse
    if total_deployed == 0 and rollout.terminal_rewards:
        return {
            **rollout.terminal_rewards,
            "gates_passed": True,
            "avg_utilization": 0.0,
            "concentration_violations": 0,
            "total_weeks": 1,
            "economics": asdict(eco),
        }

    available = rollout.config.get("total_capital", 5_000_000)
    return {
        "total_interest": total_interest,
        "total_losses": total_losses,
        "total_fees": 0,
        "total_deployed": total_deployed,
        "available_capital": available,
        "frauds_funded": frauds_funded,
        "defaults_count": defaults_count,
        "deals_won": deals_won,
        "deals_rejected": deals_rejected,
        "deals_errored": 0,
        "gates_passed": True,
        "avg_utilization": total_deployed / available if available > 0 else 0,
        "concentration_violations": 0,
        "total_weeks": 1,
        "economics": asdict(eco),
    }
