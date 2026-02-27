"""
Verifiers framework adapter — subclass of ``vf.CliAgentEnv``.

This is the primary integration point for using Loanville with the
verifiers framework (``pip install verifiers``) and its GRPO trainer.

Requires: ``pip install verifiers[rl]``

Usage with verifiers::

    from loanville.rl.vf_adapter import LoanvilleVerifierEnv

    env = LoanvilleVerifierEnv(
        dataset_path="tasks/loanville_dataset",
        docker_image="python:3.11-slim",
    )

    # Use with GRPOTrainer
    from verifiers import GRPOTrainer, GRPOConfig
    trainer = GRPOTrainer(
        model="your-model",
        env=env,
        config=GRPOConfig(...),
    )
    trainer.train()

Usage with Harbor-format tasks::

    # Export Loanville tasks to Harbor format first:
    from loanville.rl.tasks import export_harbor_dataset
    export_harbor_dataset("tasks/loanville_dataset", n_tasks=10)

    # Then use HarborEnv directly:
    import verifiers as vf
    env = vf.HarborEnv(
        run_command="python3 /task/agent.py",
        dataset_path="tasks/loanville_dataset",
    )

This module is only importable when verifiers is installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

try:
    import verifiers as vf
    from datasets import Dataset
except ImportError as e:
    raise ImportError(
        "The verifiers framework is required for this adapter. "
        "Install it with: pip install verifiers[rl]\n"
        f"Original error: {e}"
    ) from e

from ..data import get_borrowers, MIX_PRESETS
from ..models import Borrower, EconomicsConfig, ECONOMICS_PRESETS
from .verifier import (
    CreditQualityVerifier,
    GateVerifier,
    PortfolioVerifier,
    RewardResult,
    build_default_rubric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward functions in verifiers convention (async, returns float)
# ---------------------------------------------------------------------------

async def loanville_credit_reward(state: dict, **kwargs: Any) -> float:
    """Credit quality reward function for vf.Rubric."""
    verifier = CreditQualityVerifier()
    return verifier(state).reward


async def loanville_portfolio_reward(state: dict, **kwargs: Any) -> float:
    """Portfolio management reward function for vf.Rubric."""
    verifier = PortfolioVerifier()
    return verifier(state).reward


async def loanville_gate_reward(state: dict, **kwargs: Any) -> float:
    """Gate verification reward function for vf.Rubric."""
    verifier = GateVerifier()
    return verifier(state).reward


def build_vf_rubric(
    credit_weight: float = 0.7,
    portfolio_weight: float = 0.2,
    gate_weight: float = 0.1,
) -> vf.Rubric:
    """Build a native ``vf.Rubric`` with Loanville reward functions.

    This is the rubric you pass to verifiers-framework environments
    and trainers.  It uses the same reward logic as the standalone
    verifiers but in the ``async (state, **kwargs) -> float`` form
    that ``vf.Rubric`` expects.
    """
    rubric = vf.Rubric(
        funcs=[
            loanville_credit_reward,
            loanville_portfolio_reward,
            loanville_gate_reward,
        ],
        weights=[credit_weight, portfolio_weight, gate_weight],
    )
    return rubric


# ---------------------------------------------------------------------------
# Dataset builder: Loanville borrowers → HuggingFace Dataset
# ---------------------------------------------------------------------------

def build_loanville_dataset(
    n_tasks: int = 10,
    borrowers_per_task: int = 5,
    mix: str = "realistic",
    data_mode: str = "full",
    economics: str = "balanced",
    base_seed: int = 42,
) -> Dataset:
    """Build a HuggingFace Dataset of Loanville underwriting tasks.

    Each row is a task with a prompt (borrower dossier) and metadata.
    Compatible with ``vf.Environment.dataset`` parameter.
    """
    eco = ECONOMICS_PRESETS.get(economics, EconomicsConfig())
    tasks = []

    for i in range(n_tasks):
        seed = base_seed + i
        # Map season-style mix names to data-module mix names
        mix_map = {
            "gentle": "easy",
            "realistic": "balanced",
            "adversarial": "hard",
            "stress": "stress",
        }
        data_mix = mix_map.get(mix, mix)
        if data_mix not in MIX_PRESETS:
            data_mix = "easy"
        all_borrowers = get_borrowers(data_mix, seed=seed)
        borrowers = all_borrowers[:borrowers_per_task]

        # Build the prompt (what the model sees)
        prompt_parts = [
            "You are a commercial credit underwriter. Evaluate the following "
            "loan applications and make APPROVE/REJECT decisions with terms.\n\n"
            f"Available capital: $5,000,000 | Target yield: 10% | "
            f"Funding cost: {eco.funding_rate * 100:.1f}%\n\n"
        ]

        for b in borrowers:
            d = b.dossier
            prompt_parts.append(
                f"## {d.company_name}\n"
                f"Sector: {d.sector} | Years: {d.years_in_business} | "
                f"Revenue: ${d.annual_revenue:,.0f} | "
                f"Net Income: ${d.net_income:,.0f}\n"
                f"Loan: ${d.loan_request_amount:,.0f} for {d.loan_purpose}\n"
                f"{d.narrative}\n\n"
            )

        prompt_parts.append(
            "Output JSON array of decisions, each with: "
            "borrower_id, decision (APPROVE/REJECT), reasoning, "
            "and term_sheet (loan_amount, interest_rate, term_months) if approved."
        )

        prompt = "".join(prompt_parts)
        messages = [{"role": "user", "content": prompt}]

        # Ground truth for reward computation (hidden from model)
        ground_truth = {
            b.dossier.company_name: {
                "true_outcome": b.true_outcome,
                "months_before_default": b.months_before_default,
                "loan_request_amount": b.dossier.loan_request_amount,
            }
            for b in borrowers
        }

        tasks.append({
            "example_id": i,
            "task": f"loanville_{mix}_{i:03d}",
            "prompt": messages,
            "info": {
                "seed": seed,
                "mix": mix,
                "data_mode": data_mode,
                "economics": economics,
                "n_borrowers": borrowers_per_task,
                "ground_truth": ground_truth,
            },
        })

    return Dataset.from_list(tasks)


# ---------------------------------------------------------------------------
# LoanvilleVerifierEnv — vf.SingleTurnEnv subclass
# ---------------------------------------------------------------------------

class LoanvilleVerifierEnv(vf.SingleTurnEnv):
    """Verifiers-framework environment for Loanville underwriting tasks.

    This is a ``vf.SingleTurnEnv`` subclass that presents borrower
    dossiers as prompts and scores model responses against ground truth.

    For sandboxed (Harbor-format) tasks, use ``vf.HarborEnv`` directly
    with tasks exported by ``loanville.rl.tasks.export_harbor_dataset()``.

    For direct (non-sandboxed) evaluation and training, use this class::

        env = LoanvilleVerifierEnv(
            n_tasks=20,
            mix="realistic",
        )

        # With GRPOTrainer
        trainer = vf.GRPOTrainer(model="...", env=env, config=...)
        trainer.train()

        # Or standalone evaluation
        results = env.evaluate_sync(model="...", provider="openai")
    """

    def __init__(
        self,
        n_tasks: int = 20,
        borrowers_per_task: int = 5,
        mix: str = "realistic",
        data_mode: str = "full",
        economics: str = "balanced",
        base_seed: int = 42,
        credit_weight: float = 0.7,
        portfolio_weight: float = 0.2,
        gate_weight: float = 0.1,
        **kwargs: Any,
    ):
        self._mix = mix
        self._data_mode = data_mode
        self._economics_name = economics
        self._economics = ECONOMICS_PRESETS.get(economics, EconomicsConfig())
        self._base_seed = base_seed
        self._borrowers_per_task = borrowers_per_task

        dataset = build_loanville_dataset(
            n_tasks=n_tasks,
            borrowers_per_task=borrowers_per_task,
            mix=mix,
            data_mode=data_mode,
            economics=economics,
            base_seed=base_seed,
        )

        rubric = build_vf_rubric(
            credit_weight=credit_weight,
            portfolio_weight=portfolio_weight,
            gate_weight=gate_weight,
        )

        super().__init__(
            dataset=dataset,
            rubric=rubric,
            **kwargs,
        )

    async def score_response(
        self,
        state: dict[str, Any],
        response: str,
    ) -> dict[str, Any]:
        """Score a model's response against ground truth.

        Parses the model's JSON decisions and computes RAROC-based reward.
        This is called by the verifiers framework during rollout scoring.
        """
        info = state.get("info", {})
        ground_truth = info.get("ground_truth", {})

        # Parse model response
        try:
            decisions = json.loads(response)
            if not isinstance(decisions, list):
                decisions = [decisions]
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
            if match:
                try:
                    decisions = json.loads(match.group(1))
                except json.JSONDecodeError:
                    decisions = []
            else:
                decisions = []

        if not decisions:
            state["reward"] = 0.0
            return state

        # Score against ground truth
        total_deployed = 0.0
        total_interest = 0.0
        total_losses = 0.0
        total_fees = 0.0
        frauds_funded = 0
        defaults_count = 0
        deals_won = 0
        deals_rejected = 0

        for dec in decisions:
            bid = dec.get("borrower_id", "")
            gt = ground_truth.get(bid)
            if gt is None:
                continue

            action = dec.get("decision", "REJECT").upper()
            true_outcome = gt["true_outcome"]

            if action == "APPROVE":
                ts = dec.get("term_sheet", {})
                amount = float(ts.get("loan_amount", 0))
                rate = float(ts.get("interest_rate", 10.0))
                term = int(ts.get("term_months", 24))
                amount = min(amount, 500_000.0)

                if amount <= 0:
                    deals_rejected += 1
                    continue

                deals_won += 1
                total_deployed += amount
                horizon_years = term / 12.0

                if true_outcome == "good":
                    total_interest += amount * (rate / 100.0) * horizon_years
                    total_fees += amount * self._economics.origination_fee_rate
                elif true_outcome == "bad":
                    mbd = gt.get("months_before_default") or 6
                    months_paid = min(mbd, term)
                    total_interest += amount * (rate / 100.0) * (months_paid / 12.0)
                    total_losses += amount * (1.0 - self._economics.recovery_rate_bad)
                    defaults_count += 1
                elif true_outcome == "fraud":
                    total_losses += amount * (1.0 - self._economics.recovery_rate_fraud)
                    frauds_funded += 1
                    defaults_count += 1
            else:
                deals_rejected += 1

        # Build verifier state for reward computation
        from dataclasses import asdict
        state.update({
            "total_interest": total_interest,
            "total_losses": total_losses,
            "total_fees": total_fees,
            "total_deployed": total_deployed,
            "available_capital": 5_000_000.0,
            "frauds_funded": frauds_funded,
            "defaults_count": defaults_count,
            "deals_won": deals_won,
            "deals_rejected": deals_rejected,
            "gates_passed": True,  # no gate checking in this mode
            "avg_utilization": total_deployed / 5_000_000.0,
            "concentration_violations": 0,
            "total_weeks": 1,
            "economics": asdict(self._economics),
        })

        return state
