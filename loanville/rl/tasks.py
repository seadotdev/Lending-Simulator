"""
Harbor-compatible task export.

Converts Loanville scenarios into the Harbor task directory format,
enabling evaluation via Harbor's CLI (``harbor run``) or the verifiers
framework's ``HarborEnv``.

Harbor task structure::

    task_name/
    ├── instruction.md      # What the agent sees
    ├── task.toml           # Task configuration
    ├── solution/           # Oracle/reference (uploaded after agent completes)
    │   └── solve.sh
    └── tests/              # Verification (uploaded after agent completes)
        └── test.sh

This module generates all four artifacts from Loanville's borrower data
and scoring configuration.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..data import get_borrowers, MIX_PRESETS
from ..models import Borrower, EconomicsConfig, ECONOMICS_PRESETS


@dataclass
class TaskConfig:
    """Configuration for a Harbor-compatible task."""
    task_name: str = ""
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    docker_image: str = "python:3.11-slim"
    timeout_seconds: int = 300
    # Loanville-specific
    n_borrowers: int = 5
    season_mix: str = "realistic"
    economics: str = "balanced"
    data_mode: str = "full"
    seed: int = 42
    # Reward thresholds
    pass_threshold: float = 0.5  # reward >= this to pass
    # Tags for categorization
    tags: list[str] = field(default_factory=lambda: ["lending", "underwriting"])


def _build_instruction(
    borrowers: list[Borrower],
    config: TaskConfig,
    economics: EconomicsConfig,
) -> str:
    """Generate the instruction.md content for a task."""
    borrower_summaries = []
    for b in borrowers:
        d = b.dossier
        summary = textwrap.dedent(f"""\
        ### Borrower: {d.company_name}
        - **Sector:** {d.sector}
        - **Years in business:** {d.years_in_business}
        - **Employees:** {d.employee_count}
        - **Annual revenue:** ${d.annual_revenue:,.0f}
        - **Annual expenses:** ${d.annual_expenses:,.0f}
        - **Net income:** ${d.net_income:,.0f}
        - **Loan request:** ${d.loan_request_amount:,.0f} for {d.loan_purpose}
        - **Narrative:** {d.narrative}
        """)

        if d.quarterly_income and config.data_mode != "aggregate_only":
            summary += "\n**Quarterly Income:**\n"
            for q in d.quarterly_income:
                summary += (
                    f"- {q.quarter}: revenue=${q.revenue:,.0f}, "
                    f"expenses=${q.expenses:,.0f}, "
                    f"net=${q.net_income:,.0f}, "
                    f"margin={q.net_margin_pct:.1f}%\n"
                )

        if d.bank_statements and config.data_mode in ("full", "statements_inline"):
            summary += "\n**Bank Statement Summaries:**\n"
            for s in d.bank_statements:
                summary += (
                    f"- {s.month}: opening=${s.opening_balance:,.0f}, "
                    f"deposits=${s.total_deposits:,.0f}, "
                    f"withdrawals=${s.total_withdrawals:,.0f}, "
                    f"ending=${s.ending_balance:,.0f}\n"
                )

        borrower_summaries.append(summary)

    all_borrowers = "\n---\n".join(borrower_summaries)

    return textwrap.dedent(f"""\
    # Loanville Underwriting Task

    You are a commercial credit underwriter evaluating loan applications.

    ## Your Role

    - Evaluate each borrower application below
    - For each borrower, decide: **APPROVE** or **REJECT**
    - If you approve, propose terms: loan amount, interest rate (annual %), and term (months)
    - Your goal is to maximize risk-adjusted return on capital (RAROC)

    ## Constraints

    - **Total capital available:** $5,000,000
    - **Maximum single loan:** $500,000
    - **Target yield:** 10% annual
    - **Funding cost:** {economics.funding_rate * 100:.1f}% annual
    - **Risk-free benchmark:** {economics.risk_free_rate * 100:.1f}% annual

    ## Scoring

    You will be scored on:
    1. **Credit quality** (70%): Net P&L after defaults, fraud penalties, and funding cost
    2. **Portfolio management** (20%): Capital deployment efficiency and diversification
    3. **Decision quality** (10%): Correct approve/reject relative to true outcomes

    Rejecting everything scores poorly (missed opportunity cost).
    Approving everything scores poorly (excessive defaults).
    The optimal strategy requires genuine credit analysis.

    ## Output Format

    For each borrower, output a JSON object:

    ```json
    {{
      "borrower_id": "<company_name>",
      "decision": "APPROVE" or "REJECT",
      "reasoning": "<brief rationale>",
      "term_sheet": {{
        "loan_amount": <number>,
        "interest_rate": <annual percentage>,
        "term_months": <number>
      }}
    }}
    ```

    Write all decisions to `/task/decisions.json` as a JSON array.

    ## Borrower Applications

    {all_borrowers}
    """)


def _build_task_toml(config: TaskConfig) -> str:
    """Generate the task.toml configuration."""
    tags_str = ", ".join(f'"{t}"' for t in config.tags)
    return textwrap.dedent(f"""\
    [task]
    name = "{config.task_name}"
    difficulty = "{config.difficulty}"
    tags = [{tags_str}]

    [environment]
    docker_image = "{config.docker_image}"
    timeout = {config.timeout_seconds}

    [loanville]
    n_borrowers = {config.n_borrowers}
    season_mix = "{config.season_mix}"
    economics = "{config.economics}"
    data_mode = "{config.data_mode}"
    seed = {config.seed}
    pass_threshold = {config.pass_threshold}
    """)


def _build_solution_script(
    borrowers: list[Borrower],
    economics: EconomicsConfig,
) -> str:
    """Generate the oracle solution (solve.sh).

    The oracle knows ground truth and makes perfect decisions.
    """
    decisions = []
    for b in borrowers:
        d = b.dossier
        if b.true_outcome == "good":
            decisions.append({
                "borrower_id": d.company_name,
                "decision": "APPROVE",
                "reasoning": "Good credit quality — reliable borrower.",
                "term_sheet": {
                    "loan_amount": min(d.loan_request_amount, 500_000.0),
                    "interest_rate": 10.0,
                    "term_months": 24,
                },
            })
        else:
            decisions.append({
                "borrower_id": d.company_name,
                "decision": "REJECT",
                "reasoning": f"High risk — true outcome is {b.true_outcome}.",
            })

    decisions_json = json.dumps(decisions, indent=2)

    return textwrap.dedent(f"""\
    #!/bin/bash
    # Oracle solution — uses ground truth to make perfect decisions.
    # This is the reference implementation; agent solutions are compared against it.

    mkdir -p /task

    cat > /task/decisions.json << 'ORACLE_EOF'
    {decisions_json}
    ORACLE_EOF

    echo "Oracle solution written to /task/decisions.json"
    """)


def _build_test_script(
    borrowers: list[Borrower],
    config: TaskConfig,
    economics: EconomicsConfig,
) -> str:
    """Generate the test/verification script (test.sh).

    This script runs after the agent completes and computes the reward.
    It reads /task/decisions.json, scores it against ground truth,
    and writes the reward to /logs/verifier/reward.json.
    """
    ground_truth = {}
    for b in borrowers:
        ground_truth[b.dossier.company_name] = {
            "true_outcome": b.true_outcome,
            "months_before_default": b.months_before_default,
            "loan_request_amount": b.dossier.loan_request_amount,
        }

    gt_json = json.dumps(ground_truth, indent=2)

    return textwrap.dedent(f"""\
    #!/bin/bash
    # Loanville task verifier — scores agent decisions against ground truth.
    # Writes reward to /logs/verifier/reward.json (Harbor convention).

    set -e
    mkdir -p /logs/verifier

    # Check that decisions file exists
    if [ ! -f /task/decisions.json ]; then
        echo '{{"reward": 0.0, "error": "decisions.json not found"}}' > /logs/verifier/reward.json
        echo "FAIL: /task/decisions.json not found"
        exit 0
    fi

    # Inline Python scorer
    python3 << 'SCORER_EOF'
    import json
    import sys

    # Ground truth
    GROUND_TRUTH = json.loads('''{gt_json}''')

    PASS_THRESHOLD = {config.pass_threshold}
    FUNDING_RATE = {economics.funding_rate}
    RISK_FREE_RATE = {economics.risk_free_rate}
    FRAUD_PENALTY_RATE = {economics.fraud_penalty_rate}
    TOTAL_CAPITAL = 5_000_000.0

    # Load agent decisions
    try:
        with open("/task/decisions.json") as f:
            decisions = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        result = {{"reward": 0.0, "error": f"Failed to parse decisions: {{e}}"}}
        with open("/logs/verifier/reward.json", "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(0)

    # Score decisions
    total_deployed = 0.0
    total_interest = 0.0
    total_losses = 0.0
    frauds_funded = 0
    defaults = 0
    correct_decisions = 0
    n_decisions = 0

    for dec in decisions:
        bid = dec.get("borrower_id", "")
        gt = GROUND_TRUTH.get(bid)
        if gt is None:
            continue
        n_decisions += 1

        action = dec.get("decision", "REJECT").upper()
        true_outcome = gt["true_outcome"]

        # Decision correctness
        correct_action = "REJECT" if true_outcome in ("bad", "fraud") else "APPROVE"
        if action == correct_action:
            correct_decisions += 1

        if action == "APPROVE":
            ts = dec.get("term_sheet", {{}})
            amount = float(ts.get("loan_amount", 0))
            rate = float(ts.get("interest_rate", 10.0))
            term = int(ts.get("term_months", 24))
            amount = min(amount, 500_000.0)

            if amount <= 0:
                continue

            total_deployed += amount
            horizon_years = term / 12.0

            if true_outcome == "good":
                interest = amount * (rate / 100.0) * horizon_years
                total_interest += interest
            elif true_outcome == "bad":
                mbd = gt.get("months_before_default") or 6
                months_paid = min(mbd, term)
                interest = amount * (rate / 100.0) * (months_paid / 12.0)
                total_interest += interest
                total_losses += amount * 0.75  # 75% loss on default
                defaults += 1
            elif true_outcome == "fraud":
                total_losses += amount * 0.98  # 98% loss on fraud
                frauds_funded += 1
                defaults += 1

    # Compute RAROC-style reward
    funding_cost = total_deployed * FUNDING_RATE * 2.0  # 2-year horizon
    net_pnl = total_interest - total_losses - funding_cost
    fraud_penalty = frauds_funded * FRAUD_PENALTY_RATE * (total_deployed / max(1, n_decisions))
    opportunity = TOTAL_CAPITAL * RISK_FREE_RATE * 2.0

    if TOTAL_CAPITAL > 0:
        raroc = (net_pnl - fraud_penalty) / TOTAL_CAPITAL
    else:
        raroc = 0.0

    # Normalize to [0, 1]
    reward = max(0.0, min(1.0, (raroc + 0.20) / 0.40))

    # Decision accuracy
    accuracy = correct_decisions / max(1, n_decisions)

    result = {{
        "reward": round(reward, 4),
        "raroc": round(raroc, 4),
        "net_pnl": round(net_pnl, 2),
        "accuracy": round(accuracy, 4),
        "total_deployed": round(total_deployed, 2),
        "frauds_funded": frauds_funded,
        "defaults": defaults,
        "n_decisions": n_decisions,
        "passed": reward >= PASS_THRESHOLD,
    }}

    with open("/logs/verifier/reward.json", "w") as f:
        json.dump(result, f, indent=2)

    # Also write scalar reward for reward.txt fallback
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(str(round(reward, 4)))

    print(f"Reward: {{reward:.4f}} (RAROC: {{raroc:.4f}}, Accuracy: {{accuracy:.2%}})")
    print(f"Deployed: ${{total_deployed:,.0f}} | PnL: ${{net_pnl:,.0f}} | "
          f"Frauds: {{frauds_funded}} | Defaults: {{defaults}}")
    SCORER_EOF
    """)


def export_harbor_task(
    task_dir: str | Path,
    config: Optional[TaskConfig] = None,
    borrowers: Optional[list[Borrower]] = None,
) -> Path:
    """Export a single Loanville scenario as a Harbor-format task.

    Creates the task directory with instruction.md, task.toml,
    solution/solve.sh, and tests/test.sh.

    Args:
        task_dir: Where to create the task directory.
        config: Task configuration. If None, uses defaults.
        borrowers: Pre-generated borrowers. If None, generates from config.

    Returns:
        Path to the created task directory.
    """
    config = config or TaskConfig()
    task_path = Path(task_dir)
    task_path.mkdir(parents=True, exist_ok=True)

    if not config.task_name:
        config.task_name = task_path.name

    # Get economics config
    economics = ECONOMICS_PRESETS.get(config.economics, EconomicsConfig())

    # Generate borrowers if not provided
    if borrowers is None:
        # Map season-style mix names to data-module mix names
        mix_map = {
            "gentle": "easy",
            "realistic": "balanced",
            "adversarial": "hard",
            "stress": "stress",
        }
        data_mix = mix_map.get(config.season_mix, config.season_mix)
        if data_mix not in MIX_PRESETS:
            data_mix = "easy"
        all_borrowers = get_borrowers(data_mix, seed=config.seed)
        borrowers = all_borrowers[:config.n_borrowers]

    # Write instruction.md
    instruction = _build_instruction(borrowers, config, economics)
    (task_path / "instruction.md").write_text(instruction)

    # Write task.toml
    toml_content = _build_task_toml(config)
    (task_path / "task.toml").write_text(toml_content)

    # Write solution/
    solution_dir = task_path / "solution"
    solution_dir.mkdir(exist_ok=True)
    solution = _build_solution_script(borrowers, economics)
    solve_path = solution_dir / "solve.sh"
    solve_path.write_text(solution)
    solve_path.chmod(0o755)

    # Write tests/
    tests_dir = task_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_script = _build_test_script(borrowers, config, economics)
    test_path = tests_dir / "test.sh"
    test_path.write_text(test_script)
    test_path.chmod(0o755)

    return task_path


def export_harbor_dataset(
    dataset_dir: str | Path,
    n_tasks: int = 5,
    difficulties: Optional[list[str]] = None,
    base_seed: int = 42,
) -> Path:
    """Export multiple Loanville scenarios as a Harbor dataset.

    Creates a directory of tasks suitable for ``HarborEnv`` or
    ``harbor run --dataset``.

    Args:
        dataset_dir: Root directory for the dataset.
        n_tasks: Number of tasks to generate.
        difficulties: List of difficulty levels to cycle through.
        base_seed: Starting seed (incremented per task).

    Returns:
        Path to the dataset directory.
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    difficulties = difficulties or ["easy", "medium", "hard"]
    mixes = {
        "easy": "gentle",
        "medium": "realistic",
        "hard": "adversarial",
    }

    for i in range(n_tasks):
        diff = difficulties[i % len(difficulties)]
        mix = mixes.get(diff, "realistic")
        task_name = f"loanville_{diff}_{i:03d}"

        config = TaskConfig(
            task_name=task_name,
            difficulty=diff,
            n_borrowers=5 if diff == "easy" else 10 if diff == "medium" else 15,
            season_mix=mix,
            seed=base_seed + i,
            pass_threshold=0.6 if diff == "easy" else 0.5 if diff == "medium" else 0.4,
            tags=["lending", "underwriting", diff],
        )

        export_harbor_task(
            task_dir=dataset_path / task_name,
            config=config,
        )

    return dataset_path
