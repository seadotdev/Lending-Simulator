"""
Agentic LOS simulation orchestrator.

Runs the agent loop for each lender × borrower pair, then inspects the
LOS to determine what actually happened.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from .agent_loop import AgentLoopConfig, AgentLoopResult, run_agent_loop
from .agent_tasks import get_task
from .custom_tools import LenderToolkit
from .data import get_borrowers, get_lenders
from .los_adapter import DEFAULT_LOS_URL, check_los_health
from .models import LenderConfig

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

async def run_agent_sim(
    config: AgentSimConfig,
    lenders: list[LenderConfig] | None = None,
) -> AgentSimReport:
    """Run the agentic simulation."""
    import uuid
    run_id = uuid.uuid4().hex[:8]

    await check_los_health(config.los_url)

    if lenders is None:
        lenders = get_lenders()

    borrowers = get_borrowers(config.mix, seed=config.seed)
    if len(borrowers) > config.cases:
        borrowers = borrowers[:config.cases]

    report = AgentSimReport(config=config)
    start = time.time()

    task_def = get_task(config.tasks[0] if config.tasks else "simple_underwrite")

    print(f"\n{'=' * 70}")
    print(f"  LOANVILLE — AGENTIC LOS SIMULATION")
    print(f"{'=' * 70}")
    print(f"  Mode: {config.mode} | Task: {task_def.name}")
    print(f"  {len(borrowers)} borrowers × {len(lenders)} lenders | max {config.max_turns} turns")
    print(f"  LOS: {config.los_url}")

    # Show borrower lineup
    print(f"\n  Borrowers:")
    for b in borrowers:
        quality = "RISKY" if b.true_outcome in ("bad", "fraud") else "good"
        print(f"    {b.dossier.company_name} ({b.dossier.sector}) "
              f"— ${b.dossier.loan_request_amount:,.0f} [{quality}]")
    print()

    for lender in lenders:
        model = config.los_model or lender.model
        print(f"  --- {lender.name} ({model}) ---")

        portfolio_summary = ""
        if lender.existing_portfolio:
            deployed = sum(x.remaining_balance for x in lender.existing_portfolio)
            portfolio_summary = f"Deployed: ${deployed:,.0f} across {len(lender.existing_portfolio)} loans."

        toolkit = LenderToolkit(lender.id)

        for i, borrower in enumerate(borrowers):
            quality = borrower.true_outcome or "good"
            print(f"    {borrower.dossier.company_name}:", end=" ", flush=True)

            tenant_id = f"r{run_id}_{lender.id}_{borrower.id}"

            loop_config = AgentLoopConfig(
                model=model,
                mode=config.mode,
                max_turns=config.max_turns,
                tenant_id=tenant_id,
                actor=f"agent:{lender.id}",
                los_url=config.los_url,
                provider=config.provider,
                toolkit=toolkit,
            )

            loop_result = await run_agent_loop(
                config=loop_config,
                lender=lender,
                borrower=borrower,
                task_description=task_def.task_prompt,
                portfolio_summary=portfolio_summary,
            )

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
            report.cases.append(case)

            report.total_tokens_in += loop_result.tokens_in
            report.total_tokens_out += loop_result.tokens_out

            # Print inline result
            print()
            print(_describe_case(case))

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
