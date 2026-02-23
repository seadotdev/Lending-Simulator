#!/usr/bin/env python3
"""
JQ Benchmark: pure jq-skill evaluation for small language models.

Inspired by OpenThoughts-TBLite: difficulty-calibrated tasks with objective
pass/fail grading so that improvements in 3B-30B models produce visible signal
instead of sitting at the floor on hard benchmarks.

Architecture:
  - Pulls bank statement JSON from loanville.data (the only coupling)
  - Defines 25 tasks across 4 difficulty tiers (easy/medium/hard/extreme)
  - Each task: natural-language question → model writes jq command → execute → grade
  - Grading is objective: exact match, numeric tolerance, or set equality
  - Results per-model: pass rate by tier, overall score, query quality metrics

Usage:
  python test_jq_benchmark.py                          # All default models
  python test_jq_benchmark.py --model qwen/qwen3-8b   # Single model
  python test_jq_benchmark.py --borrower BRW-010       # Specific borrower
  python test_jq_benchmark.py --dry-run                # Show tasks + expected answers only
  python test_jq_benchmark.py --attempts 3             # Multiple attempts per task
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from loanville.data import get_borrowers
from loanville.llm import _bank_statements_to_json, MODEL_PRICING

# ---------------------------------------------------------------------------
# Models — focus on 3B-30B range, with one larger reference
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    # Reference (large)
    "meta-llama/llama-3.3-70b-instruct",
    # Mid-tier
    "qwen/qwq-32b",
    "qwen/qwen3-32b",
    "qwen/qwen3-30b-a3b",
    # Small
    "qwen/qwen3-14b",
    "qwen/qwen3-8b",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "google/gemma-3-27b-it",
    "mistralai/mistral-small-3.2-24b-instruct",
    # Tiny
    "nvidia/nemotron-nano-9b-v2",
    "mistralai/mistral-nemo",
]


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class JqTask:
    """A single jq benchmark task."""
    id: str
    difficulty: str           # easy, medium, hard, extreme
    question: str             # Natural-language question
    compute_expected: object  # callable(data: list[dict]) -> str
    grade: object             # callable(actual: str, expected: str) -> (bool, str)
    hint: str = ""            # Optional hint about jq approach


def grade_exact(actual: str, expected: str) -> tuple[bool, str]:
    """Grade by exact string match (after stripping whitespace)."""
    a = actual.strip()
    e = expected.strip()
    if a == e:
        return True, "exact match"
    return False, f"expected {e!r}, got {a!r}"


def grade_numeric(tolerance: float = 0.01):
    """Grade by numeric closeness (relative tolerance)."""
    def _grade(actual: str, expected: str) -> tuple[bool, str]:
        try:
            a = float(actual.strip())
            e = float(expected.strip())
        except (ValueError, TypeError):
            return False, f"cannot parse as number: actual={actual.strip()!r}"
        if e == 0:
            ok = abs(a) < tolerance
        else:
            ok = abs(a - e) / abs(e) < tolerance
        if ok:
            return True, f"numeric match ({a} ≈ {e})"
        return False, f"expected ≈{e}, got {a}"
    return _grade


def grade_number_exact(actual: str, expected: str) -> tuple[bool, str]:
    """Grade numeric values that should match exactly (integers, counts)."""
    try:
        a = float(actual.strip())
        e = float(expected.strip())
    except (ValueError, TypeError):
        return False, f"cannot parse as number: actual={actual.strip()!r}"
    if a == e:
        return True, f"exact numeric match ({a})"
    return False, f"expected {e}, got {a}"


def grade_json_set(actual: str, expected: str) -> tuple[bool, str]:
    """Grade by comparing as unordered sets of JSON values."""
    try:
        a = set(json.loads(actual)) if isinstance(actual, str) else set(actual)
        e = set(json.loads(expected)) if isinstance(expected, str) else set(expected)
    except (json.JSONDecodeError, TypeError):
        return False, f"cannot parse as JSON array: {actual.strip()[:80]!r}"
    if a == e:
        return True, "set match"
    missing = e - a
    extra = a - e
    detail = []
    if missing:
        detail.append(f"missing: {missing}")
    if extra:
        detail.append(f"extra: {extra}")
    return False, "; ".join(detail)


def grade_sorted_json_list(actual: str, expected: str) -> tuple[bool, str]:
    """Grade by comparing as sorted JSON arrays."""
    try:
        a = sorted(json.loads(actual))
        e = sorted(json.loads(expected))
    except (json.JSONDecodeError, TypeError):
        return False, f"cannot parse as JSON array: {actual.strip()[:80]!r}"
    if a == e:
        return True, "sorted list match"
    return False, f"expected {e}, got {a}"


def grade_numeric_list(tolerance: float = 0.01):
    """Grade a JSON array of numbers with tolerance per element."""
    def _grade(actual: str, expected: str) -> tuple[bool, str]:
        try:
            a = json.loads(actual.strip())
            e = json.loads(expected.strip())
        except (json.JSONDecodeError, TypeError):
            return False, f"cannot parse as JSON array"
        if not isinstance(a, list) or not isinstance(e, list):
            return False, "expected JSON array"
        if len(a) != len(e):
            return False, f"length mismatch: expected {len(e)}, got {len(a)}"
        for i, (av, ev) in enumerate(zip(a, e)):
            try:
                af, ef = float(av), float(ev)
            except (ValueError, TypeError):
                return False, f"element {i} not numeric"
            if ef == 0:
                if abs(af) > tolerance:
                    return False, f"element {i}: expected ≈0, got {af}"
            elif abs(af - ef) / abs(ef) > tolerance:
                return False, f"element {i}: expected ≈{ef}, got {af}"
        return True, "numeric list match"
    return _grade


def grade_contains_all(actual: str, expected: str) -> tuple[bool, str]:
    """Grade: actual output must contain all items from expected JSON array."""
    try:
        e = set(json.loads(expected))
    except (json.JSONDecodeError, TypeError):
        return False, "bad expected value"
    try:
        a = set(json.loads(actual))
    except (json.JSONDecodeError, TypeError):
        # Try line-by-line
        a = set(actual.strip().split("\n"))
    missing = e - a
    if not missing:
        return True, "contains all expected items"
    return False, f"missing: {missing}"


# ---------------------------------------------------------------------------
# Task bank — 25 tasks across 4 difficulty levels
# ---------------------------------------------------------------------------

def build_tasks(data: list[dict]) -> list[JqTask]:
    """Build the task bank with expected answers computed from actual data."""
    tasks = []

    # ===================== EASY (7 tasks) =====================
    # Targeting: basic jq syntax, field access, simple filters

    tasks.append(JqTask(
        id="E01",
        difficulty="easy",
        question="How many months of bank statement data are there?",
        compute_expected=lambda d: str(len(d)),
        grade=grade_number_exact,
        hint="Use jq 'length'",
    ))

    tasks.append(JqTask(
        id="E02",
        difficulty="easy",
        question="What is the opening balance for the first month?",
        compute_expected=lambda d: str(d[0]["opening_balance"]),
        grade=grade_number_exact,
    ))

    tasks.append(JqTask(
        id="E03",
        difficulty="easy",
        question="What is the ending balance for the last month (month 12)?",
        compute_expected=lambda d: str(d[-1]["ending_balance"]),
        grade=grade_number_exact,
    ))

    tasks.append(JqTask(
        id="E04",
        difficulty="easy",
        question="List all the month names in the data as a JSON array.",
        compute_expected=lambda d: json.dumps([m["month"] for m in d]),
        grade=grade_sorted_json_list,
    ))

    tasks.append(JqTask(
        id="E05",
        difficulty="easy",
        question="How many deposit transactions are there in the first month?",
        compute_expected=lambda d: str(len(d[0]["deposits"])),
        grade=grade_number_exact,
    ))

    tasks.append(JqTask(
        id="E06",
        difficulty="easy",
        question="What is the total_deposits value for the first month?",
        compute_expected=lambda d: str(d[0]["total_deposits"]),
        grade=grade_number_exact,
    ))

    tasks.append(JqTask(
        id="E07",
        difficulty="easy",
        question="What are the keys of the first element in the array? Return as a JSON array.",
        compute_expected=lambda d: json.dumps(sorted(d[0].keys())),
        grade=grade_sorted_json_list,
    ))

    # ===================== MEDIUM (7 tasks) =====================
    # Targeting: aggregation, map/reduce, sorting, basic computation

    tasks.append(JqTask(
        id="M01",
        difficulty="medium",
        question="What is the sum of total_deposits across all 12 months?",
        compute_expected=lambda d: str(round(sum(m["total_deposits"] for m in d), 2)),
        grade=grade_numeric(0.001),
    ))

    tasks.append(JqTask(
        id="M02",
        difficulty="medium",
        question="What is the sum of total_withdrawals across all 12 months?",
        compute_expected=lambda d: str(round(sum(m["total_withdrawals"] for m in d), 2)),
        grade=grade_numeric(0.001),
    ))

    tasks.append(JqTask(
        id="M03",
        difficulty="medium",
        question="Which month has the highest total_deposits? Return just the month name string.",
        compute_expected=lambda d: max(d, key=lambda m: m["total_deposits"])["month"],
        grade=grade_exact,
    ))

    tasks.append(JqTask(
        id="M04",
        difficulty="medium",
        question=(
            "How many total individual deposit transactions are there across all 12 months? "
            "(Count each transaction, not the monthly totals.)"
        ),
        compute_expected=lambda d: str(sum(len(m["deposits"]) for m in d)),
        grade=grade_number_exact,
    ))

    tasks.append(JqTask(
        id="M05",
        difficulty="medium",
        question=(
            "How many total individual withdrawal transactions are there across all 12 months?"
        ),
        compute_expected=lambda d: str(sum(len(m["withdrawals"]) for m in d)),
        grade=grade_number_exact,
    ))

    tasks.append(JqTask(
        id="M06",
        difficulty="medium",
        question=(
            "What is the largest single deposit transaction amount across all months? "
            "Return just the number."
        ),
        compute_expected=lambda d: str(max(
            t["amount"]
            for m in d
            for t in m["deposits"]
        )),
        grade=grade_numeric(0.001),
    ))

    tasks.append(JqTask(
        id="M07",
        difficulty="medium",
        question=(
            "Return a JSON array of the 12 monthly net cash flows "
            "(total_deposits minus total_withdrawals for each month), in order."
        ),
        compute_expected=lambda d: json.dumps([
            round(m["total_deposits"] - m["total_withdrawals"], 2)
            for m in d
        ]),
        grade=grade_numeric_list(0.001),
    ))

    # ===================== HARD (7 tasks) =====================
    # Targeting: multi-step computation, group-by, cross-month analysis

    tasks.append(JqTask(
        id="H01",
        difficulty="hard",
        question=(
            "What is the average monthly net cash flow (total_deposits minus total_withdrawals) "
            "across all 12 months? Return a single number."
        ),
        compute_expected=lambda d: str(round(
            sum(m["total_deposits"] - m["total_withdrawals"] for m in d) / len(d), 2
        )),
        grade=grade_numeric(0.01),
    ))

    tasks.append(JqTask(
        id="H02",
        difficulty="hard",
        question=(
            "List all unique deposit description strings across all months as a JSON array."
        ),
        compute_expected=lambda d: json.dumps(sorted(set(
            t["description"]
            for m in d
            for t in m["deposits"]
        ))),
        grade=grade_sorted_json_list,
    ))

    tasks.append(JqTask(
        id="H03",
        difficulty="hard",
        question=(
            "List all unique withdrawal description strings across all months as a JSON array."
        ),
        compute_expected=lambda d: json.dumps(sorted(set(
            t["description"]
            for m in d
            for t in m["withdrawals"]
        ))),
        grade=grade_sorted_json_list,
    ))

    tasks.append(JqTask(
        id="H04",
        difficulty="hard",
        question=(
            "For each unique deposit description, calculate the total amount deposited "
            "across all months. Return as a JSON object mapping description to total amount. "
            "Round values to 2 decimal places."
        ),
        compute_expected=lambda d: json.dumps({
            desc: round(total, 2)
            for desc, total in sorted(
                _group_sum(d, "deposits").items()
            )
        }),
        grade=_grade_json_object_numeric(0.01),
    ))

    tasks.append(JqTask(
        id="H05",
        difficulty="hard",
        question=(
            "How many months had a negative net cash flow "
            "(where total_withdrawals exceeded total_deposits)?"
        ),
        compute_expected=lambda d: str(sum(
            1 for m in d
            if m["total_withdrawals"] > m["total_deposits"]
        )),
        grade=grade_number_exact,
    ))

    tasks.append(JqTask(
        id="H06",
        difficulty="hard",
        question=(
            "What is the maximum ending_balance across all 12 months? Return just the number."
        ),
        compute_expected=lambda d: str(max(m["ending_balance"] for m in d)),
        grade=grade_numeric(0.001),
    ))

    tasks.append(JqTask(
        id="H07",
        difficulty="hard",
        question=(
            "What is the total number of deposit transactions where the amount "
            "is greater than 50000?"
        ),
        compute_expected=lambda d: str(sum(
            1
            for m in d
            for t in m["deposits"]
            if t["amount"] > 50000
        )),
        grade=grade_number_exact,
    ))

    # ===================== EXTREME (4 tasks) =====================
    # Targeting: complex multi-step, statistical, pattern detection

    tasks.append(JqTask(
        id="X01",
        difficulty="extreme",
        question=(
            "Calculate the standard deviation of the 12 monthly total_deposits values. "
            "Use population standard deviation (divide by N, not N-1). "
            "Return a single number rounded to 2 decimal places."
        ),
        compute_expected=lambda d: str(round(_population_stddev([
            m["total_deposits"] for m in d
        ]), 2)),
        grade=grade_numeric(0.02),
    ))

    tasks.append(JqTask(
        id="X02",
        difficulty="extreme",
        question=(
            "Find the top 3 deposit descriptions by total amount across all months. "
            "Return as a JSON array of 3 strings, ordered from highest total to lowest."
        ),
        compute_expected=lambda d: json.dumps(
            [desc for desc, _ in sorted(
                _group_sum(d, "deposits").items(),
                key=lambda x: -x[1]
            )[:3]]
        ),
        grade=_grade_ordered_list,
    ))

    tasks.append(JqTask(
        id="X03",
        difficulty="extreme",
        question=(
            "Calculate the month-over-month percentage change in total_deposits "
            "for months 2 through 12 (11 values). Return as a JSON array of numbers "
            "rounded to 2 decimal places. Formula: ((current - previous) / previous) * 100."
        ),
        compute_expected=lambda d: json.dumps([
            round((d[i]["total_deposits"] - d[i-1]["total_deposits"])
                  / d[i-1]["total_deposits"] * 100, 2)
            for i in range(1, len(d))
        ]),
        grade=grade_numeric_list(0.05),
    ))

    tasks.append(JqTask(
        id="X04",
        difficulty="extreme",
        question=(
            "For each month, count the number of deposit transactions whose amount "
            "is above that month's average deposit amount. Return a JSON array of "
            "12 integers (one per month, in order)."
        ),
        compute_expected=lambda d: json.dumps([
            sum(1 for t in m["deposits"]
                if t["amount"] > (sum(t2["amount"] for t2 in m["deposits"]) / len(m["deposits"])))
            for m in d
        ]),
        grade=lambda actual, expected: (
            json.loads(actual.strip()) == json.loads(expected.strip()),
            "match" if json.loads(actual.strip()) == json.loads(expected.strip())
            else f"expected {expected}, got {actual.strip()}"
        ) if _safe_json_parse(actual) is not None else (False, f"cannot parse: {actual.strip()[:60]}"),
    ))

    return tasks


# ---------------------------------------------------------------------------
# Helper functions for task computation
# ---------------------------------------------------------------------------

def _safe_json_parse(s: str):
    try:
        return json.loads(s.strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _group_sum(data: list[dict], txn_type: str) -> dict[str, float]:
    """Group transactions by description and sum amounts."""
    totals: dict[str, float] = {}
    for m in data:
        for t in m[txn_type]:
            desc = t["description"]
            totals[desc] = totals.get(desc, 0) + t["amount"]
    return totals


def _population_stddev(values: list[float]) -> float:
    """Population standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def _grade_json_object_numeric(tolerance: float = 0.01):
    """Grade a JSON object where values are numbers, with tolerance."""
    def _grade(actual: str, expected: str) -> tuple[bool, str]:
        try:
            a = json.loads(actual.strip())
            e = json.loads(expected.strip())
        except (json.JSONDecodeError, TypeError):
            return False, f"cannot parse as JSON object"
        if not isinstance(a, dict) or not isinstance(e, dict):
            return False, "expected JSON object"
        if set(a.keys()) != set(e.keys()):
            missing = set(e.keys()) - set(a.keys())
            extra = set(a.keys()) - set(e.keys())
            parts = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"extra keys: {extra}")
            return False, "; ".join(parts)
        for k in e:
            try:
                af, ef = float(a[k]), float(e[k])
            except (ValueError, TypeError):
                return False, f"key {k!r}: not numeric"
            if ef == 0:
                if abs(af) > tolerance:
                    return False, f"key {k!r}: expected ≈0, got {af}"
            elif abs(af - ef) / abs(ef) > tolerance:
                return False, f"key {k!r}: expected ≈{ef}, got {af}"
        return True, "JSON object numeric match"
    return _grade


def _grade_ordered_list(actual: str, expected: str) -> tuple[bool, str]:
    """Grade an ordered JSON list by exact element match."""
    try:
        a = json.loads(actual.strip())
        e = json.loads(expected.strip())
    except (json.JSONDecodeError, TypeError):
        return False, f"cannot parse as JSON array"
    if not isinstance(a, list) or not isinstance(e, list):
        return False, "expected JSON array"
    if a == e:
        return True, "ordered list match"
    return False, f"expected {e}, got {a}"


# ---------------------------------------------------------------------------
# Sandbox execution (uses just-bash like the main codebase)
# ---------------------------------------------------------------------------

async def run_jq_in_sandbox(command: str, bank_json: str) -> tuple[str, int, str]:
    """Execute a jq command in a just-bash sandbox. Returns (stdout, exit_code, stderr)."""
    from just_bash import Bash as JustBash
    from just_bash.types import ExecutionLimits

    sandbox = JustBash(
        files={"/data/bank_statements.json": bank_json},
        limits=ExecutionLimits(
            max_command_count=500,
            max_loop_iterations=1000,
            max_awk_iterations=1000,
        ),
    )
    result = await sandbox.exec(command)
    return (
        result.stdout.strip() if result.stdout else "",
        result.exit_code,
        result.stderr.strip() if result.stderr else "",
    )


# ---------------------------------------------------------------------------
# LLM interaction — ask model to produce jq command
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a jq expert. You will be given a question about JSON data stored at \
/data/bank_statements.json. The file contains a JSON array of 12 monthly bank \
statements with this structure:

[
  {
    "month": "January 2025",
    "opening_balance": 180000,
    "ending_balance": 227000,
    "deposits": [
      {"date": "01/03/2025", "description": "Customer Name", "amount": 57632.98},
      ...
    ],
    "total_deposits": 185000.0,
    "withdrawals": [
      {"date": "01/01/2025", "description": "Vendor Name", "amount": 7513.37},
      ...
    ],
    "total_withdrawals": 138000.0
  },
  ... (12 months total)
]

Your task: write a single bash command using jq (and optionally other standard \
Unix tools) that answers the question. The command should read from \
/data/bank_statements.json and output ONLY the answer to stdout.

Rules:
- Output ONLY the bash command, nothing else
- No explanation, no markdown fences, no commentary
- The command must be a single line
- Use jq as the primary tool; you may pipe through other tools if needed
- The output should be the final answer (a number, string, or JSON value)
- Do not output extra formatting, labels, or text — just the raw answer"""


async def ask_model_for_jq(
    client,
    model: str,
    question: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict]:
    """Ask a model to write a jq command for the given question.

    Returns (command_string, usage_dict).
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            usage = getattr(response, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            } if usage else {"prompt_tokens": 0, "completion_tokens": 0}

            raw = response.choices[0].message.content or ""
            # Extract command — strip markdown fences if present
            command = _extract_command(raw)
            return command, usage_dict

        except Exception as e:
            return f"echo 'ERROR: {e}'", {"prompt_tokens": 0, "completion_tokens": 0}


def _extract_command(raw: str) -> str:
    """Extract a bash command from model output, stripping markdown fences."""
    raw = raw.strip()

    # Remove markdown code fences
    patterns = [
        r"```(?:bash|sh|shell)?\s*\n?(.*?)\n?\s*```",
        r"`([^`]+)`",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no fences, take the first non-empty line that looks like a command
    for line in raw.split("\n"):
        line = line.strip()
        if line and (line.startswith("jq") or line.startswith("cat") or
                     line.startswith("echo") or "|" in line):
            return line

    # Fallback: return the whole thing
    return raw.split("\n")[0].strip() if raw else "echo 'no command generated'"


# ---------------------------------------------------------------------------
# Single task evaluation
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    difficulty: str
    question: str
    model: str
    command: str
    stdout: str
    stderr: str
    exit_code: int
    expected: str
    passed: bool
    grade_detail: str
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    attempt: int = 1
    error: str = ""


async def evaluate_task(
    client,
    model: str,
    task: JqTask,
    bank_json: str,
    expected: str,
    semaphore: asyncio.Semaphore,
    attempt: int = 1,
) -> TaskResult:
    """Evaluate a single task: ask model → execute → grade."""
    start = time.time()

    # Step 1: ask model for jq command
    command, usage = await ask_model_for_jq(client, model, task.question, semaphore)

    # Step 2: execute in sandbox
    try:
        stdout, exit_code, stderr = await run_jq_in_sandbox(command, bank_json)
    except Exception as e:
        stdout, exit_code, stderr = "", 1, f"sandbox error: {e}"

    # Step 3: grade
    error = ""
    if exit_code != 0:
        passed = False
        grade_detail = f"command failed (exit {exit_code}): {stderr[:120]}"
        error = stderr[:200]
    else:
        try:
            passed, grade_detail = task.grade(stdout, expected)
        except Exception as e:
            passed = False
            grade_detail = f"grading error: {e}"
            error = str(e)

    elapsed = time.time() - start

    return TaskResult(
        task_id=task.id,
        difficulty=task.difficulty,
        question=task.question,
        model=model,
        command=command,
        stdout=stdout[:500],
        stderr=stderr[:200],
        exit_code=exit_code,
        expected=expected[:500],
        passed=passed,
        grade_detail=grade_detail,
        elapsed_s=round(elapsed, 2),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        attempt=attempt,
        error=error,
    )


# ---------------------------------------------------------------------------
# Run all tasks for one model
# ---------------------------------------------------------------------------

async def run_model_benchmark(
    client,
    model: str,
    tasks: list[JqTask],
    bank_json: str,
    data: list[dict],
    concurrency: int = 5,
    attempts: int = 1,
) -> list[TaskResult]:
    """Run all tasks for a single model."""
    semaphore = asyncio.Semaphore(concurrency)
    all_results = []

    for attempt_num in range(1, attempts + 1):
        coros = []
        for task in tasks:
            expected = task.compute_expected(data)
            coros.append(evaluate_task(
                client, model, task, bank_json, expected, semaphore,
                attempt=attempt_num,
            ))
        results = await asyncio.gather(*coros)
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

DIFFICULTY_ORDER = ["easy", "medium", "hard", "extreme"]


def print_results_table(all_results: list[TaskResult], models: list[str]):
    """Print a per-task pass/fail matrix."""
    tasks_seen = list(dict.fromkeys((r.task_id, r.difficulty) for r in all_results))

    print(f"\n{'='*110}")
    print(f"  JQ BENCHMARK — PER-TASK RESULTS")
    print(f"{'='*110}")

    # Header
    header = f"  {'Task':<6s} {'Diff':<8s}"
    for m in models:
        short = m.split("/")[-1][:14]
        header += f" {short:>15s}"
    print(header)
    print(f"  {'─'*6} {'─'*8}" + f" {'─'*15}" * len(models))

    for tid, diff in tasks_seen:
        row = f"  {tid:<6s} {diff:<8s}"
        for m in models:
            # If multiple attempts, pass if ANY attempt passed
            task_results = [r for r in all_results
                           if r.task_id == tid and r.model == m]
            if not task_results:
                row += f" {'?':>15s}"
            else:
                any_pass = any(r.passed for r in task_results)
                best = next((r for r in task_results if r.passed), task_results[0])
                if any_pass:
                    row += f" {'PASS':>15s}"
                elif best.exit_code != 0:
                    row += f" {'FAIL(exec)':>15s}"
                else:
                    row += f" {'FAIL(wrong)':>15s}"
        print(row)


def print_summary(all_results: list[TaskResult], models: list[str]):
    """Print summary statistics per model."""
    print(f"\n{'='*110}")
    print(f"  JQ BENCHMARK — SUMMARY BY MODEL")
    print(f"{'='*110}")

    print(f"\n  {'Model':<45s} {'Easy':>7s} {'Med':>7s} {'Hard':>7s} {'Ext':>7s} "
          f"{'Total':>8s} {'Exec%':>7s} {'Cost':>8s}")
    print(f"  {'─'*100}")

    for m in models:
        model_results = [r for r in all_results if r.model == m]
        short = m.split("/")[-1]

        # Deduplicate by task (best of attempts)
        best_by_task: dict[str, TaskResult] = {}
        for r in model_results:
            key = r.task_id
            if key not in best_by_task or (r.passed and not best_by_task[key].passed):
                best_by_task[key] = r
        deduped = list(best_by_task.values())

        by_diff = {}
        for diff in DIFFICULTY_ORDER:
            diff_results = [r for r in deduped if r.difficulty == diff]
            n = len(diff_results)
            passed = sum(1 for r in diff_results if r.passed)
            by_diff[diff] = (passed, n)

        total_passed = sum(p for p, _ in by_diff.values())
        total_n = sum(n for _, n in by_diff.values())
        exec_ok = sum(1 for r in deduped if r.exit_code == 0)
        exec_pct = exec_ok / total_n * 100 if total_n else 0

        # Cost estimate
        total_prompt = sum(r.prompt_tokens for r in model_results)
        total_completion = sum(r.completion_tokens for r in model_results)
        in_price, out_price = MODEL_PRICING.get(m, (1.0, 3.0))
        cost = (total_prompt / 1_000_000 * in_price +
                total_completion / 1_000_000 * out_price)

        diff_strs = []
        for diff in DIFFICULTY_ORDER:
            p, n = by_diff.get(diff, (0, 0))
            diff_strs.append(f"{p}/{n}")

        print(f"  {short:<45s} {diff_strs[0]:>7s} {diff_strs[1]:>7s} "
              f"{diff_strs[2]:>7s} {diff_strs[3]:>7s} "
              f"{total_passed}/{total_n:>3d}   {exec_pct:>5.0f}%  ${cost:>.4f}")

    # TBLite-style difficulty calibration report
    print(f"\n{'='*110}")
    print(f"  DIFFICULTY CALIBRATION (TBLite-style)")
    print(f"{'='*110}")
    print(f"\n  {'Difficulty':<12s} {'Avg Pass%':>10s} {'Models@0%':>10s} {'Models@100%':>12s} {'Discriminating':>15s}")
    print(f"  {'─'*62}")

    for diff in DIFFICULTY_ORDER:
        pass_rates = []
        for m in models:
            best_by_task: dict[str, TaskResult] = {}
            for r in all_results:
                if r.model == m and r.difficulty == diff:
                    key = r.task_id
                    if key not in best_by_task or (r.passed and not best_by_task[key].passed):
                        best_by_task[key] = r
            if best_by_task:
                n = len(best_by_task)
                p = sum(1 for r in best_by_task.values() if r.passed)
                pass_rates.append(p / n * 100)

        if not pass_rates:
            continue
        avg = sum(pass_rates) / len(pass_rates)
        at_zero = sum(1 for r in pass_rates if r == 0)
        at_100 = sum(1 for r in pass_rates if r == 100)
        disc = len(pass_rates) - at_zero - at_100
        disc_label = "YES" if disc >= 2 else "marginal" if disc == 1 else "no"

        print(f"  {diff:<12s} {avg:>9.1f}% {at_zero:>10d} {at_100:>12d} {disc_label:>15s}")


def print_query_samples(all_results: list[TaskResult], models: list[str]):
    """Print sample jq queries per model."""
    print(f"\n{'='*110}")
    print(f"  JQ QUERY SAMPLES")
    print(f"{'='*110}")

    for m in models:
        model_results = [r for r in all_results if r.model == m]
        short = m.split("/")[-1]

        passed = [r for r in model_results if r.passed]
        failed = [r for r in model_results if not r.passed]

        print(f"\n  {short}:")
        if passed:
            print(f"    Passing queries ({len(passed)}):")
            for r in passed[:3]:
                cmd_display = r.command[:85] + "..." if len(r.command) > 85 else r.command
                print(f"      [{r.task_id}] {cmd_display}")
            if len(passed) > 3:
                print(f"      ... and {len(passed) - 3} more")

        if failed:
            print(f"    Failing queries ({len(failed)}):")
            for r in failed[:3]:
                cmd_display = r.command[:85] + "..." if len(r.command) > 85 else r.command
                detail = r.grade_detail[:60]
                print(f"      [{r.task_id}] {cmd_display}")
                print(f"              → {detail}")
            if len(failed) > 3:
                print(f"      ... and {len(failed) - 3} more")


def print_dry_run(tasks: list[JqTask], data: list[dict]):
    """Print all tasks with expected answers (no API calls)."""
    print(f"\n{'='*90}")
    print(f"  JQ BENCHMARK — DRY RUN (tasks + expected answers)")
    print(f"{'='*90}")

    for diff in DIFFICULTY_ORDER:
        diff_tasks = [t for t in tasks if t.difficulty == diff]
        print(f"\n  --- {diff.upper()} ({len(diff_tasks)} tasks) ---")
        for t in diff_tasks:
            expected = t.compute_expected(data)
            exp_display = expected[:80] + "..." if len(expected) > 80 else expected
            print(f"\n  [{t.id}] {t.question}")
            print(f"       Expected: {exp_display}")
            if t.hint:
                print(f"       Hint: {t.hint}")

    print(f"\n  Total: {len(tasks)} tasks")
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JQ Benchmark: test jq capabilities of small language models"
    )
    parser.add_argument("--model", type=str, action="append", default=None,
                        help="Model to test (can be repeated; default: all)")
    parser.add_argument("--borrower", type=str, default="BRW-001",
                        help="Borrower ID to use for bank statement data (default: BRW-001)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent API calls per model (default: 5)")
    parser.add_argument("--attempts", type=int, default=1,
                        help="Number of attempts per task (best-of-N; default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show tasks and expected answers without calling APIs")
    parser.add_argument("--output", type=str, default="jq_benchmark_results.json",
                        help="Output JSON file (default: jq_benchmark_results.json)")
    args = parser.parse_args()

    # Load bank statement data
    all_borrowers = get_borrowers("all")
    borrower = next((b for b in all_borrowers if b.id == args.borrower), None)
    if not borrower:
        print(f"ERROR: borrower {args.borrower} not found")
        sys.exit(1)

    bank_json = _bank_statements_to_json(borrower)
    data = json.loads(bank_json)

    # Build tasks
    tasks = build_tasks(data)

    if args.dry_run:
        print_dry_run(tasks, data)
        return

    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    models = args.model if args.model else DEFAULT_MODELS

    print("=" * 110)
    print("  JQ BENCHMARK: Pure jq-skill evaluation for language models")
    print("=" * 110)
    print(f"\n  Borrower: {borrower.dossier.company_name} ({borrower.id})")
    print(f"  Tasks: {len(tasks)} ({', '.join(f'{d}: {sum(1 for t in tasks if t.difficulty == d)}' for d in DIFFICULTY_ORDER)})")
    print(f"  Models: {len(models)}")
    print(f"  Attempts per task: {args.attempts}")
    print(f"  Total API calls: {len(models) * len(tasks) * args.attempts}")

    all_results: list[TaskResult] = []

    for model in models:
        short = model.split("/")[-1]
        print(f"\n{'─'*110}")
        print(f"  MODEL: {model}")
        print(f"{'─'*110}")

        try:
            results = asyncio.run(
                run_model_benchmark(
                    client, model, tasks, bank_json, data,
                    concurrency=args.concurrency,
                    attempts=args.attempts,
                )
            )
            all_results.extend(results)

            # Inline progress: count passes
            best_by_task: dict[str, TaskResult] = {}
            for r in results:
                key = r.task_id
                if key not in best_by_task or (r.passed and not best_by_task[key].passed):
                    best_by_task[key] = r

            for r in sorted(best_by_task.values(), key=lambda r: r.task_id):
                mark = "PASS" if r.passed else "FAIL"
                cmd_short = r.command[:60] + "..." if len(r.command) > 60 else r.command
                detail = r.grade_detail[:50]
                print(f"    [{r.task_id}] {mark:<4s} | {cmd_short}")
                if not r.passed:
                    print(f"           → {detail}")

            passed = sum(1 for r in best_by_task.values() if r.passed)
            total = len(best_by_task)
            print(f"\n    Score: {passed}/{total} ({passed/total*100:.0f}%)")

        except Exception as e:
            print(f"  FAILED: {e}")

    # Print reports
    print_results_table(all_results, models)
    print_summary(all_results, models)
    print_query_samples(all_results, models)

    # Save raw results
    raw_output = [
        {
            "task_id": r.task_id,
            "difficulty": r.difficulty,
            "question": r.question,
            "model": r.model,
            "command": r.command,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "exit_code": r.exit_code,
            "expected": r.expected,
            "passed": r.passed,
            "grade_detail": r.grade_detail,
            "elapsed_s": r.elapsed_s,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "attempt": r.attempt,
            "error": r.error,
        }
        for r in all_results
    ]

    with open(args.output, "w") as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n  Raw results saved to {args.output}")

    print(f"\n{'='*110}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()
