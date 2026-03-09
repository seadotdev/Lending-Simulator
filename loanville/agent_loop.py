"""
Agentic loop for LOS-driven underwriting.

The model autonomously drives the LOS — creating entities, deals, uploading
docs, advancing stages, making decisions — with the sim providing a task
and scoring the outcome.

Three interaction modes:
  - tool_call: OpenAI function-calling → REST API
  - cli: model emits `los` commands → subprocess execution
  - repl: persistent LOS session → subprocess with state
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
import httpx

load_dotenv()

from .agent_budget import BudgetTracker
from .agent_eject import EjectDecision, EjectPolicy
from .agent_events import EventLogger
from .agent_prompts import build_system_prompt, build_task_prompt
from .custom_tools import LenderToolkit
from .executors import CLIExecutor, REPLExecutor, ToolCallExecutor, extract_los_commands
from .los_tools import LOS_TOOLS
from .models import Borrower, LenderConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopConfig:
    model: str
    mode: str = "tool_call"  # "tool_call" | "cli" | "repl"
    max_turns: int = 20
    tenant_id: str = ""
    actor: str = ""
    los_url: str = "http://localhost:3000"
    provider: str = "openrouter"
    toolkit: LenderToolkit | None = None
    event_logger: EventLogger | None = None
    budget_tracker: BudgetTracker | None = None
    eject_policies: list[EjectPolicy] = field(default_factory=list)
    log_prefix: str = ""  # e.g. "[alias] " for parallel runs


@dataclass
class AgentLoopResult:
    messages: list[dict] = field(default_factory=list)
    turns: int = 0
    tool_call_count: int = 0
    termination: str = ""  # "model_done" | "max_turns" | "error"
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    custom_tools_used: list[str] = field(default_factory=list)
    final_decision: dict | None = None
    error: str | None = None


async def run_agent_loop(
    config: AgentLoopConfig,
    lender: LenderConfig,
    borrower: Borrower,
    task_description: str = "",
    portfolio_summary: str = "",
) -> AgentLoopResult:
    """Run the agentic loop: model drives LOS autonomously."""
    start = time.time()
    result = AgentLoopResult()

    # Build prompts
    custom_tools_info = ""
    if config.toolkit and config.toolkit.tools:
        custom_tools_info = "\n".join(
            f"- {t.name}: {t.description}"
            for t in config.toolkit.tools
        )

    system_prompt = build_system_prompt(
        mode=config.mode,
        lender=lender,
        portfolio_summary=portfolio_summary,
        custom_tools_info=custom_tools_info,
    )
    user_prompt = build_task_prompt(borrower, task_description)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Build tool list for function-calling mode
    tools = None
    if config.mode == "tool_call":
        tools = list(LOS_TOOLS)
        if config.toolkit:
            tools.extend(config.toolkit.get_tool_definitions())

    # Create executor
    executor = ToolCallExecutor(
        los_url=config.los_url,
        tenant_id=config.tenant_id,
        actor=config.actor,
        toolkit=config.toolkit,
        provider=config.provider,
    )
    cli_executor = None
    if config.mode in ("cli", "repl"):
        if config.mode == "repl":
            cli_executor = REPLExecutor(config.los_url, config.tenant_id, config.actor)
        else:
            cli_executor = CLIExecutor(config.los_url, config.tenant_id, config.actor)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if config.budget_tracker:
        api_key = config.budget_tracker.api_key or api_key
    if not api_key:
        result.termination = "error"
        result.error = "OPENROUTER_API_KEY not set"
        return result

    ev = config.event_logger
    prefix = config.log_prefix
    budget = config.budget_tracker
    _last_budget_turn = -999

    try:
        for turn in range(config.max_turns):
            result.turns = turn + 1

            # Budget check: poll and inject message periodically
            if budget:
                budget.poll()
                if budget.exhausted:
                    result.termination = "budget_exhausted"
                    if ev:
                        ev.log("budget_exhausted", turn=turn + 1)
                    break
                # Inject budget status every 5 turns
                if turn - _last_budget_turn >= 5:
                    _last_budget_turn = turn
                    msg = budget.inject_message()
                    if msg:
                        messages.append({"role": "system", "content": msg})

            # Progressive disclosure: summarize old tool results after turn 3
            # to reduce token waste. Keep system, user, and last 4 messages intact.
            if config.mode == "tool_call" and turn >= 3 and len(messages) > 8:
                messages = _compress_messages(messages)

            # Call the model
            if ev:
                ev.log("model_call", turn=turn + 1, model=config.model)

            resp = await _call_model(
                messages=messages,
                model=config.model,
                tools=tools,
                api_key=api_key,
                provider=config.provider,
            )

            if resp is None:
                result.termination = "error"
                result.error = "Model call returned None"
                break

            # Track tokens
            usage = resp.get("usage", {})
            result.tokens_in += usage.get("prompt_tokens", 0)
            result.tokens_out += usage.get("completion_tokens", 0)

            choice = resp.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")

            messages.append(message)

            if config.mode == "tool_call":
                tool_calls = message.get("tool_calls", [])

                if tool_calls:
                    result.tool_call_count += len(tool_calls)
                    tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    logger.info("%s  turn %d tools: %s", prefix, turn + 1, ", ".join(tool_names))

                    if ev:
                        ev.log("tool_call", turn=turn + 1, tools=tool_names)

                    # Check for agent_done
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        if func.get("name") == "agent_done":
                            try:
                                args = json.loads(func.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            result.final_decision = args

                    # Execute all tool calls
                    tool_results = await executor.execute(tool_calls)

                    if ev:
                        ev.log("tool_result", turn=turn + 1,
                               results=[{"name": tr.name, "len": len(tr.content)} for tr in tool_results])

                    # Track custom tool usage
                    for tr in tool_results:
                        if config.toolkit and config.toolkit.get_tool(tr.name):
                            if tr.name not in result.custom_tools_used:
                                result.custom_tools_used.append(tr.name)

                    # Append results as tool messages
                    for tr in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id,
                            "content": tr.content,
                        })

                    # If agent signaled done
                    if result.final_decision is not None:
                        result.termination = "model_done"
                        break
                else:
                    # No tool calls — check if model is done
                    content = message.get("content", "")
                    decision = _try_parse_decision(content)
                    if decision:
                        result.final_decision = decision
                        result.termination = "model_done"
                        break

                    if finish_reason == "stop":
                        result.termination = "model_done"
                        break

            elif config.mode in ("cli", "repl"):
                content = message.get("content", "")
                commands = extract_los_commands(content)

                if commands:
                    result.tool_call_count += len(commands)
                    if ev:
                        ev.log("tool_call", turn=turn + 1, tools=commands)
                    output = cli_executor.execute(commands)
                    if ev:
                        ev.log("tool_result", turn=turn + 1, output_len=len(output))
                    messages.append({
                        "role": "user",
                        "content": f"[LOS Output]\n{output}",
                    })
                else:
                    # No commands — check for decision
                    decision = _try_parse_decision(content)
                    if decision:
                        result.final_decision = decision
                        result.termination = "model_done"
                        break

                    if finish_reason == "stop":
                        result.termination = "model_done"
                        break

            # Eject policy checks
            if config.eject_policies:
                los_dict = await _quick_los_state(config.los_url, config.tenant_id)
                bfs = budget.fraction_spent() if budget else None
                for policy in config.eject_policies:
                    decision_ej = policy.check(turn + 1, config.max_turns, los_dict, bfs)
                    if decision_ej.reason and not decision_ej.should_eject:
                        # Warning
                        if ev:
                            ev.log("eject_warning", turn=turn + 1, reason=decision_ej.reason)
                        logger.info("%s  eject warning: %s", prefix, decision_ej.reason)
                    if decision_ej.should_eject:
                        result.termination = f"ejected:{decision_ej.reason}"
                        if ev:
                            ev.log("ejected", turn=turn + 1, reason=decision_ej.reason)
                        logger.info("%s  EJECTED: %s", prefix, decision_ej.reason)
                        break
                if result.termination.startswith("ejected:"):
                    break

        else:
            result.termination = "max_turns"

    except Exception as exc:
        result.termination = "error"
        result.error = str(exc)
        logger.exception("Agent loop error")

    finally:
        if ev:
            ev.log("loop_end", turns=result.turns, termination=result.termination,
                   tool_call_count=result.tool_call_count,
                   decision=(result.final_decision or {}).get("decision", ""),
                   tokens_in=result.tokens_in, tokens_out=result.tokens_out)
        await executor.close()
        if hasattr(cli_executor, "close"):
            cli_executor.close()

    result.messages = messages
    result.latency_ms = int((time.time() - start) * 1000)
    return result


def _compress_messages(messages: list[dict]) -> list[dict]:
    """Compress older tool results to reduce token usage.

    Keeps: system prompt, initial user prompt, last 4 messages.
    Middle messages: tool results get truncated to first 200 chars,
    assistant messages with tool_calls keep only the call names.
    """
    if len(messages) <= 8:
        return messages

    # System + user prompt (first 2) + last 4 = 6 preserved
    head = messages[:2]
    tail = messages[-4:]
    middle = messages[2:-4]

    compressed = []
    for msg in middle:
        role = msg.get("role", "")
        if role == "tool":
            # Truncate tool results
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "... [truncated]"
            compressed.append({**msg, "content": content})
        elif role == "assistant" and msg.get("tool_calls"):
            # Keep tool call structure but clear large arguments
            tc_summary = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "")
                if len(args) > 100:
                    args = args[:100] + "..."
                tc_summary.append({
                    **tc,
                    "function": {**func, "arguments": args},
                })
            compressed.append({**msg, "tool_calls": tc_summary})
        else:
            compressed.append(msg)

    return head + compressed + tail


async def _quick_los_state(los_url: str, tenant_id: str) -> dict:
    """Lightweight LOS state check for eject policies."""
    base = los_url.rstrip("/")
    headers = {"X-Tenant-Id": tenant_id}
    state: dict = {"deals": [], "entities": [], "spread_count": 0, "doc_count": 0}
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
            resp = await client.get(f"{base}/v1/deals")
            if resp.status_code == 200:
                data = resp.json()
                state["deals"] = data.get("deals", data) if isinstance(data, dict) else data
                for deal in state["deals"][:1]:
                    state["stage"] = deal.get("stage", "")
                    deal_id = deal.get("id")
                    if deal_id:
                        # Check docs and spreads for AnalysisEject
                        doc_resp = await client.get(f"{base}/v1/deals/{deal_id}/documents")
                        if doc_resp.status_code == 200:
                            docs = doc_resp.json()
                            doc_list = docs.get("documents", docs) if isinstance(docs, dict) else docs
                            state["doc_count"] = len(doc_list) if isinstance(doc_list, list) else 0
                        ratio_resp = await client.get(f"{base}/v1/deals/{deal_id}/ratios")
                        if ratio_resp.status_code == 200:
                            ratios = ratio_resp.json()
                            r_list = ratios.get("ratios", []) if isinstance(ratios, dict) else []
                            state["spread_count"] = len(r_list)
    except Exception:
        pass
    return state


def _try_parse_decision(text: str | None) -> dict | None:
    if not text:
        return None
    """Try to extract a JSON decision from model text."""
    import re
    # Look for JSON block with decision field
    patterns = [
        re.compile(r"```json\s*\n({.*?})\s*```", re.DOTALL),
        re.compile(r'(\{[^{}]*"decision"[^{}]*\})', re.DOTALL),
    ]
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            try:
                data = json.loads(m.group(1))
                if "decision" in data:
                    return data
            except json.JSONDecodeError:
                continue
    return None


async def _call_model(
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    api_key: str,
    provider: str = "openrouter",
) -> dict | None:
    """Call the model via OpenRouter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://loanville.dev",
        "X-Title": "Loanville Agent Sim",
    }

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            logger.error("Model API error %d: %s", resp.status_code, resp.text[:500])
            return None
        return resp.json()
