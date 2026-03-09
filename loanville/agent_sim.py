"""
Agentic LOS simulation orchestrator.

Runs the agent loop for each lender × borrower pair, then inspects the
LOS to determine what actually happened.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .agent_budget import BudgetTracker, preflight_budget, provision_key
from .agent_eject import default_eject_policies
from .agent_events import EventLogger
from .agent_loop import AgentLoopConfig, AgentLoopResult, run_agent_loop
from .agent_tasks import get_task
from .custom_tools import LenderToolkit
from .data import get_borrowers, get_lenders
from .los_adapter import DEFAULT_LOS_URL, check_los_health
from .models import Borrower, LenderConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentSimConfig:
    """Configuration for the agentic simulation."""
    mode: str = "tool_call"
    max_turns: int = 20
    tasks: list[str] = field(default_factory=lambda: ["simple_underwrite"])
    cases: int = 3
    los_url: str = DEFAULT_LOS_URL
    provider: str = "openrouter"
    mix: str = "realistic"
    seed: int = 42
    los_model: str | None = None
    budget_usd: float | None = None
    parallel: bool = False
    eject: bool = False


@dataclass
class LOSState:
    """What the LOS actually contains after an agent loop."""
    deals: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    has_spread: bool = False
    has_evaluation: bool = False
    deal_stage: str = ""


@dataclass
class CaseResult:
    lender_id: str
    lender_name: str
    borrower_id: str
    borrower_name: str
    task_name: str
    loop_result: AgentLoopResult
    los_state: LOSState
    borrower_quality: str = ""  # "good" or "bad"/"fraud"


@dataclass
class AgentSimReport:
    config: AgentSimConfig
    cases: list[CaseResult] = field(default_factory=list)
    total_latency_ms: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0


async def _inspect_los_state(
    los_url: str, tenant_id: str,
) -> LOSState:
    """Query the LOS to see what the agent actually created."""
    base = los_url.rstrip("/")
    headers = {"X-Tenant-Id": tenant_id}
    state = LOSState()

    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        try:
            resp = await client.get(f"{base}/v1/deals")
            if resp.status_code == 200:
                data = resp.json()
                state.deals = data.get("deals", data) if isinstance(data, dict) else data
        except Exception:
            pass

        try:
            resp = await client.get(f"{base}/v1/entities")
            if resp.status_code == 200:
                data = resp.json()
                state.entities = data.get("entities", data) if isinstance(data, dict) else data
        except Exception:
            pass

        for deal in state.deals[:5]:
            deal_id = deal.get("id", "")
            if not deal_id:
                continue

            state.deal_stage = deal.get("stage", "")

            try:
                resp = await client.get(f"{base}/v1/deals/{deal_id}/documents")
                if resp.status_code == 200:
                    data = resp.json()
                    docs = data.get("documents", data) if isinstance(data, dict) else data
                    state.documents.extend(docs)
            except Exception:
                pass

            try:
                resp = await client.get(f"{base}/v1/deals/{deal_id}/ratios")
                if resp.status_code == 200:
                    data = resp.json()
                    ratios = data.get("ratios", []) if isinstance(data, dict) else data
                    if ratios:
                        state.has_spread = True
            except Exception:
                pass

            try:
                resp = await client.get(f"{base}/v1/deals/{deal_id}/audit")
                if resp.status_code == 200:
                    data = resp.json()
                    events = data.get("events", data) if isinstance(data, dict) else data
                    for ev in (events if isinstance(events, list) else []):
                        if "evaluat" in str(ev.get("type", "")).lower():
                            state.has_evaluation = True
                            break
            except Exception:
                pass

    return state


# ---------------------------------------------------------------------------
# Describe what happened in plain English
# ---------------------------------------------------------------------------

def _describe_los_pipeline(los: LOSState) -> str:
    """One-line description of how far the model got through the LOS."""
    if not los.deals:
        return "nothing in LOS"
    parts = []
    parts.append(f"{len(los.deals)} deal{'s' if len(los.deals) != 1 else ''}")
    if los.entities:
        parts.append(f"{len(los.entities)} entit{'ies' if len(los.entities) != 1 else 'y'}")
    if los.documents:
        parts.append(f"{len(los.documents)} doc{'s' if len(los.documents) != 1 else ''}")
    if los.has_spread:
        parts.append("spread")
    if los.has_evaluation:
        parts.append("evaluation")
    stage = los.deal_stage or "broker"
    parts.append(f"stage={stage}")
    return ", ".join(parts)


def _describe_case(case: CaseResult) -> str:
    """Human-readable summary of one case."""
    lr = case.loop_result
    los = case.los_state

    # What happened?
    if lr.termination == "error":
        outcome = f"CRASHED after {lr.turns} turn{'s' if lr.turns != 1 else ''}"
        if lr.error:
            outcome += f" ({lr.error[:60]})"
    elif lr.termination == "max_turns":
        outcome = f"RAN OUT OF TURNS ({lr.turns})"
    else:
        decision = "?"
        if lr.final_decision:
            decision = lr.final_decision.get("decision", "?").upper()
        outcome = f"decided {decision} in {lr.turns} turn{'s' if lr.turns != 1 else ''}"

    # Was the decision right?
    quality = case.borrower_quality
    agent_decision = (lr.final_decision or {}).get("decision", "").lower()
    if quality in ("bad", "fraud"):
        if agent_decision == "decline":
            judgment = "CORRECT (risky borrower, declined)"
        elif agent_decision in ("approve", "counter"):
            judgment = "WRONG — approved a risky borrower"
        else:
            judgment = "UNCLEAR"
    else:
        if agent_decision in ("approve", "counter"):
            judgment = "CORRECT (good borrower, approved)"
        elif agent_decision == "decline":
            judgment = "WRONG — declined a good borrower"
        else:
            judgment = "UNCLEAR"

    # What's in the LOS?
    pipeline = _describe_los_pipeline(los)
    if not los.deals and agent_decision in ("approve", "counter"):
        pipeline += " — approval is meaningless without a deal!"

    return (
        f"      {outcome}\n"
        f"      Decision: {judgment}\n"
        f"      LOS: {pipeline}"
    )


# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------

def _tlog(prefix: str, msg: str) -> None:
    """Thread-safe prefixed print."""
    print(f"{prefix}{msg}", flush=True)


def _write_case_summary(
    case_dir: Path, case: CaseResult, duration_s: float,
) -> None:
    """Write summary.json for a completed case."""
    lr = case.loop_result
    cost_est = (lr.tokens_in * 0.25 + lr.tokens_out * 1.0) / 1_000_000
    ls = case.los_state
    summary = {
        "lender": case.lender_name,
        "borrower": case.borrower_name,
        "termination": lr.termination,
        "turns": lr.turns,
        "tool_call_count": lr.tool_call_count,
        "has_deal": bool(ls.deals),
        "decision": (lr.final_decision or {}).get("decision", ""),
        "tokens_in": lr.tokens_in,
        "tokens_out": lr.tokens_out,
        "cost_estimate": f"${cost_est:.4f}",
        "duration_s": round(duration_s, 1),
        "custom_tools_used": lr.custom_tools_used,
        # Behavioral telemetry
        "spread_created": ls.has_spread,
        "ratios_reviewed": ls.has_evaluation,  # evaluate fetches ratios
        "stage_reached": ls.deal_stage or (ls.deals[0].get("stage", "") if ls.deals else ""),
        "doc_count": len(ls.documents),
        "independent_decision": ls.has_spread,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))


async def run_agent_sim(
    config: AgentSimConfig,
    lenders: list[LenderConfig] | None = None,
) -> AgentSimReport:
    """Run the agentic simulation."""
    import uuid
    run_id = uuid.uuid4().hex[:8]
    ts = time.strftime("%Y%m%dT%H%M%S")
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "runs" / f"{ts}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    await check_los_health(config.los_url)

    if lenders is None:
        lenders = get_lenders()

    borrowers = get_borrowers(config.mix, seed=config.seed)
    if len(borrowers) > config.cases:
        borrowers = borrowers[:config.cases]

    # Budget: provision sub-keys if requested
    admin_key = os.environ.get("OR_ADMIN_KEY", "")
    provisioned_keys: dict[str, str] = {}  # model -> sub-key
    if config.budget_usd and admin_key:
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        model_ids = list(dict.fromkeys(
            config.los_model or lender.model for lender in lenders
        ))
        preflight_budget(admin_key, or_key, model_ids, config.budget_usd)
        for lender in lenders:
            model = config.los_model or lender.model
            if model not in provisioned_keys:
                try:
                    label = f"loanville-{run_id}-{lender.id}"
                    key = provision_key(admin_key, label, config.budget_usd)
                    provisioned_keys[model] = key
                except Exception as exc:
                    logger.warning("Failed to provision key for %s: %s", model, exc)

    report = AgentSimReport(config=config)
    start = time.time()

    task_def = get_task(config.tasks[0] if config.tasks else "simple_underwrite")

    print(f"\n{'=' * 70}")
    print(f"  LOANVILLE — AGENTIC LOS SIMULATION")
    print(f"{'=' * 70}")
    print(f"  Mode: {config.mode} | Task: {task_def.name}")
    print(f"  {len(borrowers)} borrowers × {len(lenders)} lenders | max {config.max_turns} turns")
    print(f"  LOS: {config.los_url}")
    if config.budget_usd:
        print(f"  Budget: ${config.budget_usd:.2f}/model")
    if config.eject:
        print(f"  Eject policies: enabled")
    if config.parallel:
        print(f"  Execution: parallel")
    print(f"  Run dir: {run_dir}")

    # Show borrower lineup
    print(f"\n  Borrowers:")
    for b in borrowers:
        quality = "RISKY" if b.true_outcome in ("bad", "fraud") else "good"
        print(f"    {b.dossier.company_name} ({b.dossier.sector}) "
              f"— ${b.dossier.loan_request_amount:,.0f} [{quality}]")
    print()

    async def run_case(
        lender: LenderConfig, borrower: Borrower, prefix: str = "",
    ) -> CaseResult:
        model = config.los_model or lender.model
        quality = borrower.true_outcome or "good"
        case_start = time.time()

        _tlog(prefix, f"  {borrower.dossier.company_name}: starting...")

        tenant_id = f"r{run_id}_{lender.id}_{borrower.id}"
        case_dir = run_dir / lender.id / borrower.id
        case_dir.mkdir(parents=True, exist_ok=True)

        ev = EventLogger(case_dir / "events.jsonl")

        # Budget tracker
        budget_tracker = None
        api_key_for_model = provisioned_keys.get(model)
        if api_key_for_model and config.budget_usd:
            budget_tracker = BudgetTracker(
                api_key=api_key_for_model,
                limit_usd=config.budget_usd,
            )

        # Eject policies
        eject_policies = default_eject_policies() if config.eject else []

        portfolio_summary = ""
        if lender.existing_portfolio:
            deployed = sum(x.remaining_balance for x in lender.existing_portfolio)
            portfolio_summary = f"Deployed: ${deployed:,.0f} across {len(lender.existing_portfolio)} loans."

        toolkit = LenderToolkit(lender.id)

        loop_config = AgentLoopConfig(
            model=model,
            mode=config.mode,
            max_turns=config.max_turns,
            tenant_id=tenant_id,
            actor=f"agent:{lender.id}",
            los_url=config.los_url,
            provider=config.provider,
            toolkit=toolkit,
            event_logger=ev,
            budget_tracker=budget_tracker,
            eject_policies=eject_policies,
            log_prefix=prefix,
        )

        loop_result = await run_agent_loop(
            config=loop_config,
            lender=lender,
            borrower=borrower,
            task_description=task_def.task_prompt,
            portfolio_summary=portfolio_summary,
        )

        ev.close()

        los_state = await _inspect_los_state(config.los_url, tenant_id)

        case = CaseResult(
            lender_id=lender.id,
            lender_name=lender.name,
            borrower_id=borrower.id,
            borrower_name=borrower.dossier.company_name,
            task_name=task_def.name,
            loop_result=loop_result,
            los_state=los_state,
            borrower_quality=quality,
        )

        duration_s = time.time() - case_start
        _write_case_summary(case_dir, case, duration_s)

        _tlog(prefix, f"  {borrower.dossier.company_name}:")
        _tlog("", _describe_case(case))

        return case

    for lender in lenders:
        model = config.los_model or lender.model
        alias = lender.id[:12]
        print(f"  --- {lender.name} ({model}) ---")

        if config.parallel:
            # Run all borrowers for this lender concurrently
            prefix = f"[{alias}] "
            tasks = [
                run_case(lender, borrower, prefix=prefix)
                for borrower in borrowers
            ]
            cases = await asyncio.gather(*tasks, return_exceptions=True)
            for c in cases:
                if isinstance(c, Exception):
                    logger.error("%sCase failed: %s", prefix, c)
                    continue
                report.cases.append(c)
                report.total_tokens_in += c.loop_result.tokens_in
                report.total_tokens_out += c.loop_result.tokens_out
        else:
            # Sequential execution
            for borrower in borrowers:
                case = await run_case(lender, borrower)
                report.cases.append(case)
                report.total_tokens_in += case.loop_result.tokens_in
                report.total_tokens_out += case.loop_result.tokens_out

    report.total_latency_ms = int((time.time() - start) * 1000)

    print_agent_report(report)
    return report


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_agent_report(report: AgentSimReport) -> None:
    """Print a human-readable summary."""
    secs = report.total_latency_ms / 1000
    cost_est = (report.total_tokens_in * 0.25 + report.total_tokens_out * 1.0) / 1_000_000

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Took {secs:.0f}s | ~${cost_est:.3f} estimated API cost")
    print(f"  {report.total_tokens_in:,} tokens in, {report.total_tokens_out:,} tokens out")

    # Per-lender summary
    lender_data: dict[str, dict] = {}
    for case in report.cases:
        name = case.lender_name
        d = lender_data.setdefault(name, {
            "turns": [], "deals": 0,
            "correct": 0, "wrong": 0, "total": 0,
        })
        d["turns"].append(case.loop_result.turns)
        d["deals"] += len(case.los_state.deals)
        d["total"] += 1

        agent_dec = (case.loop_result.final_decision or {}).get("decision", "").lower()
        expected = "decline" if case.borrower_quality in ("bad", "fraud") else "approve"
        if agent_dec == expected or (agent_dec == "counter" and expected == "approve"):
            d["correct"] += 1
        elif agent_dec:
            d["wrong"] += 1

    print()
    for name, d in lender_data.items():
        avg_t = sum(d["turns"]) / len(d["turns"])
        print(f"  {name}")
        print(f"    Decisions: {d['correct']} correct, {d['wrong']} wrong out of {d['total']}")
        print(f"    LOS deals created: {d['deals']} | Avg turns: {avg_t:.0f}")
        print()

    # What went wrong?
    problems = []
    for case in report.cases:
        los = case.los_state
        lr = case.loop_result
        dec = (lr.final_decision or {}).get("decision", "").lower()

        if not los.deals and dec in ("approve", "counter"):
            problems.append(
                f"  {case.lender_name} approved {case.borrower_name} "
                f"but never created a deal — approval is meaningless"
            )
        if los.deals and not los.documents:
            problems.append(
                f"  {case.lender_name} created a deal for {case.borrower_name} "
                f"but uploaded no documents — weak underwriting"
            )
        if los.deals and not los.has_spread:
            problems.append(
                f"  {case.lender_name} skipped financial spread for {case.borrower_name} "
                f"— no ratio analysis"
            )
        if case.borrower_quality in ("bad", "fraud") and dec in ("approve", "counter"):
            problems.append(
                f"  {case.lender_name} approved {case.borrower_name} "
                f"who is a {'fraudulent' if case.borrower_quality == 'fraud' else 'risky'} borrower"
            )
        if lr.termination == "error":
            problems.append(
                f"  {case.lender_name} crashed on {case.borrower_name}: {lr.error or 'unknown'}"
            )

    if problems:
        print("  Issues:")
        for p in problems:
            print(p)
        print()
