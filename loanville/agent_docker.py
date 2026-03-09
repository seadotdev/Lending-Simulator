"""
Docker-based agent runner.

Each model gets a full agentic environment (pi + bash + curl) in a Docker
container, with a provisioned sub-key and budget cap. The model interacts
with the LOS via curl, reads docs, writes files — like a human would.

Modeled on snake-arena's pi-docker experiment runner (v2).
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .agent_budget import (
    BudgetTracker,
    get_account_balance,
    get_usage,
    preflight_budget,
    provision_key,
)
from .agent_events import EventLogger
from .data import get_borrowers, get_lenders
from .agent_tasks import get_task
from .models import Borrower, LenderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_NAME = "loanville-agent"
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKER_DIR = REPO_ROOT / "docker" / "agent"
RUNS_DIR = Path("runs")

MAX_IDLE_SECS = 300  # kill container if no events for 5 min
BUDGET_POLL_INTERVAL_S = 30

# ---------------------------------------------------------------------------
# Thread-local logger (prefixed stdout + per-model log file)
# ---------------------------------------------------------------------------

_thread_log = threading.local()


def tlog(*args, **kwargs):
    fn = getattr(_thread_log, "fn", print)
    fn(*args, **kwargs)


def _make_logger(log_path: Path, alias: str):
    lock = threading.Lock()

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        with lock:
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            print(f"[{alias}] {msg}", flush=True)

    return log


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def build_image() -> None:
    print(f"Building Docker image {IMAGE_NAME}...")
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "-f", "docker/agent/Dockerfile", "."],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("Docker build failed.", file=sys.stderr)
        sys.exit(1)
    print("Image built.\n")


def _image_exists() -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# ---------------------------------------------------------------------------
# Task doc rendering
# ---------------------------------------------------------------------------

def _render_task_doc(
    borrower: Borrower,
    task_prompt: str,
    los_url: str,
    tenant_id: str,
) -> str:
    """Render the TASK.md template with borrower + config details."""
    template = (DOCKER_DIR / "TASK.md").read_text()

    # Build borrower dossier text
    d = borrower.dossier
    dossier_lines = [
        f"Company: {d.company_name}",
        f"Sector: {d.sector}",
        f"Loan Amount Requested: ${d.loan_request_amount:,.0f}",
    ]
    if hasattr(d, "loan_purpose") and d.loan_purpose:
        dossier_lines.append(f"Purpose: {d.loan_purpose}")
    if hasattr(d, "years_in_business") and d.years_in_business:
        dossier_lines.append(f"Years in Business: {d.years_in_business}")
    if hasattr(d, "annual_revenue") and d.annual_revenue:
        dossier_lines.append(f"Annual Revenue: ${d.annual_revenue:,.0f}")

    dossier_text = "\n".join(dossier_lines)

    return (
        template
        .replace("{LOS_URL}", los_url)
        .replace("{TENANT_ID}", tenant_id)
        .replace("{TASK_PROMPT}", task_prompt)
        .replace("{BORROWER_DOSSIER}", dossier_text)
    )


# ---------------------------------------------------------------------------
# Run a single model in a Docker container
# ---------------------------------------------------------------------------

def run_model_container(
    model_id: str,
    alias: str,
    lender: LenderConfig,
    borrower: Borrower,
    task_prompt: str,
    los_url: str,
    tenant_id: str,
    run_id: str,
    run_dir: Path,
    admin_key: str,
    budget_usd: float,
) -> dict:
    """Spawn a Docker container for one model, stream events, monitor budget."""
    model_dir = run_dir / alias
    workspace = model_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    _thread_log.fn = _make_logger(model_dir / "run.log", alias)

    # Provision a capped sub-key
    try:
        model_key = provision_key(admin_key, f"loanville-{run_id}-{alias}", budget_usd)
    except Exception as e:
        tlog(f"ERROR: Key provisioning failed: {e}")
        return {"alias": alias, "model": model_id, "error": f"provisioning_failed: {e}"}

    tlog(f"\n{'='*60}")
    tlog(f"  Model: {alias} ({model_id})")
    tlog(f"  Borrower: {borrower.dossier.company_name}")
    tlog(f"  Budget: ${budget_usd:.2f}")
    tlog(f"{'='*60}\n")

    container_name = f"lv-agent-{alias}-{borrower.id[:8]}"
    _remove_container(container_name)

    # Render per-run task document
    task_doc = _render_task_doc(borrower, task_prompt, los_url, tenant_id)
    task_path = model_dir / "TASK.md"
    task_path.write_text(task_doc)

    # Docker needs to reach the host LOS — use host.docker.internal on macOS
    container_los_url = los_url.replace("localhost", "host.docker.internal")

    docker_cmd = [
        "docker", "run", "--rm", "--init", "-i",
        "--name", container_name,
        "--add-host", "host.docker.internal:host-gateway",
        "-e", f"OPENROUTER_API_KEY={model_key}",
        "-e", f"LOS_URL={container_los_url}",
        "-e", f"TENANT_ID={tenant_id}",
        "-v", f"{task_path.resolve()}:/workspace/TASK.md:ro",
        "-v", f"{workspace.resolve()}:/workspace/output",
        IMAGE_NAME,
        "--model", model_id,
    ]

    events: list[dict] = []
    summary: dict = {
        "run_id": run_id,
        "model": model_id,
        "alias": alias,
        "lender": lender.name,
        "borrower": borrower.dossier.company_name,
        "tenant_id": tenant_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
        "decision": None,
        "cost": {},
    }

    prompt = (
        f"Read /workspace/TASK.md for your full task and borrower information.\n"
        f"Read /workspace/LOS_DOCS.md for the LOS architecture.\n"
        f"Read /workspace/openapi.yaml for the full API spec.\n"
        f"Then process the loan application."
    )

    try:
        proc = subprocess.Popen(
            docker_cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        # Send initial prompt via RPC
        proc.stdin.write(json.dumps({
            "id": "init", "type": "prompt", "message": prompt,
        }) + "\n")
        proc.stdin.flush()

        tlog("  Prompt sent. Streaming events...")
        last_event_ts = time.time()
        last_budget_poll_ts = time.time()
        last_injected_usage: float | None = None
        tool_call_count = 0
        has_deal = False

        ev_logger = EventLogger(model_dir / "events.jsonl")

        while True:
            # Idle timeout
            if time.time() - last_event_ts > MAX_IDLE_SECS:
                tlog(f"\n  TIMEOUT: no events for {MAX_IDLE_SECS}s, killing.")
                proc.kill()
                summary["error"] = "idle_timeout"
                break

            # Non-blocking read
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if proc.poll() is not None:
                    break

                # Budget polling
                if time.time() - last_budget_poll_ts >= BUDGET_POLL_INTERVAL_S:
                    last_budget_poll_ts = time.time()
                    usage = get_usage(model_key)
                    if usage is not None and usage != last_injected_usage:
                        remaining = max(0.0, budget_usd - usage)
                        pct = int(remaining / budget_usd * 100) if budget_usd > 0 else 0
                        budget_line = f"Budget: ${remaining:.3f} remaining of ${budget_usd:.2f} ({pct}% left)"
                        # Write budget.txt into workspace
                        try:
                            subprocess.run(
                                ["docker", "exec", container_name, "sh", "-c",
                                 f"echo '{budget_line}' > /workspace/output/budget.txt"],
                                capture_output=True, timeout=5,
                            )
                        except Exception:
                            pass
                        last_injected_usage = usage
                        ev_logger.log("budget_poll", usage=usage, remaining=remaining)
                        tlog(f"  [budget] {budget_line}")

                        # Budget exhausted — kill
                        if remaining <= 0:
                            tlog(f"\n  BUDGET EXHAUSTED: ${usage:.3f} spent.")
                            proc.kill()
                            summary["error"] = "budget_exhausted"
                            break
                continue

            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue

            line = line.strip()
            if not line:
                continue
            last_event_ts = time.time()

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "raw", "data": line}

            events.append(event)
            ev_logger.log(event.get("type", "raw"), **{
                k: v for k, v in event.items() if k != "type"
            })

            # Print relevant events
            _print_event(event)

            # Track tool calls
            if event.get("type") == "tool_execution_end":
                tool_call_count += 1
                result_text = str(event.get("result", ""))

                # Detect deal creation
                if "deal" in result_text.lower() and "id" in result_text:
                    has_deal = True

            # Agent finished
            if event.get("type") == "agent_end":
                tlog("\n  Agent finished.")
                time.sleep(2)
                proc.terminate()
                break

        proc.wait(timeout=10)
        ev_logger.close()

    except Exception as e:
        summary["error"] = str(e)
        tlog(f"\n  ERROR: {e}")

    # Read decision from output
    decision_path = workspace / "decision.json"
    if decision_path.exists():
        try:
            summary["decision"] = json.loads(decision_path.read_text())
            tlog(f"  Decision: {summary['decision'].get('decision', '?')}")
        except Exception:
            pass

    # Cost
    finished_at = datetime.now(timezone.utc)
    summary["finished_at"] = finished_at.isoformat()
    started_at = datetime.fromisoformat(summary["started_at"])
    summary["duration_s"] = round((finished_at - started_at).total_seconds())
    summary["tool_calls"] = tool_call_count
    summary["has_deal"] = has_deal

    # Final usage poll
    time.sleep(3)
    final_usage = get_usage(model_key)
    if final_usage is not None:
        summary["cost"] = {"measured_usd": round(final_usage, 4)}
        tlog(f"  Cost: ${final_usage:.4f}  Duration: {summary['duration_s']}s")
    else:
        tlog(f"  Cost: unknown  Duration: {summary['duration_s']}s")

    # Write summary
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tlog(f"  Summary saved.")
    return summary


def _print_event(event: dict):
    t = event.get("type", "")
    if t == "tool_execution_end":
        result = str(event.get("result", ""))[:120]
        tlog(f"  [tool:{event.get('toolName', '?')}] {result}")
    elif t == "error":
        tlog(f"  [ERROR] {event}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_agent_docker(
    models: list[tuple[str, str]],  # [(model_id, alias), ...]
    lenders: list[LenderConfig],
    borrowers: list[Borrower],
    task_prompt: str,
    los_url: str,
    budget_usd: float,
    admin_key: str,
    or_key: str,
    parallel: bool = True,
) -> list[dict]:
    """Run all models in Docker containers, optionally in parallel."""
    run_id = f"docker-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Preflight
    model_ids = [m for m, _ in models]
    preflight_budget(admin_key, or_key, model_ids, budget_usd)

    # Build image if needed
    if not _image_exists():
        build_image()

    # Clean stale containers
    for _, alias in models:
        for b in borrowers:
            _remove_container(f"lv-agent-{alias}-{b.id[:8]}")

    print(f"\nStarting {len(models)} model(s) x {len(borrowers)} borrower(s).")
    print(f"Run dir: {run_dir}")
    print(f"Monitor: python -m loanville status {run_id}\n")

    results: list[dict] = []

    def _run_one(model_id: str, alias: str, lender: LenderConfig, borrower: Borrower) -> dict:
        import uuid
        tenant_id = f"r{run_id}_{alias}_{borrower.id}"
        return run_model_container(
            model_id=model_id,
            alias=alias,
            lender=lender,
            borrower=borrower,
            task_prompt=task_prompt,
            los_url=los_url,
            tenant_id=tenant_id,
            run_id=run_id,
            run_dir=run_dir,
            admin_key=admin_key,
            budget_usd=budget_usd,
        )

    # Build work items: each (model, lender) x borrower
    work = []
    for (model_id, alias), lender in zip(models, lenders):
        for borrower in borrowers:
            work.append((model_id, alias, lender, borrower))

    if parallel:
        max_workers = min(len(work), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_one, *item): item[1]
                for item in work
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    summary = future.result()
                    results.append(summary)
                    cost = summary.get("cost", {}).get("measured_usd", 0)
                    print(f"  [done] {alias}  cost=${cost:.3f}  "
                          f"time={summary.get('duration_s', 0)}s  "
                          f"{summary.get('error') or 'ok'}")
                except Exception as e:
                    print(f"  [done] {alias}  EXCEPTION: {e}")
    else:
        for item in work:
            summary = _run_one(*item)
            results.append(summary)

    # Final report
    print(f"\n{'='*60}")
    print(f"  RUN COMPLETE: {run_id}")
    print(f"{'='*60}")
    print(f"  {'alias':<14}  {'borrower':<20}  {'cost':>7}  {'time':>6}  {'decision':<10}  status")
    print(f"  {'-'*80}")
    total_cost = 0.0
    for r in sorted(results, key=lambda x: x.get("alias", "")):
        c = r.get("cost", {}).get("measured_usd", 0)
        total_cost += c
        dur_s = r.get("duration_s", 0)
        dur = f"{dur_s // 60}m{dur_s % 60:02d}s"
        dec = (r.get("decision") or {}).get("decision", "-")
        status = r.get("error") or "ok"
        borrower_name = r.get("borrower", "?")[:20]
        print(f"  {r.get('alias','?'):<14}  {borrower_name:<20}  ${c:>6.3f}  {dur:>6}  {dec:<10}  {status}")
    print(f"  {'TOTAL':<14}  {'':>20}  ${total_cost:>6.3f}")
    print(f"\nResults in: {run_dir}\n")

    return results
