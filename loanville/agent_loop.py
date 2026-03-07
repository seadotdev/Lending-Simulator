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
    if not api_key:
        result.termination = "error"
        result.error = "OPENROUTER_API_KEY not set"
        return result

    try:
        for turn in range(config.max_turns):
            result.turns = turn + 1

            # Call the model
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
                    logger.info("  turn %d tools: %s", turn + 1, ", ".join(tool_names))

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
                    output = cli_executor.execute(commands)
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

        else:
            result.termination = "max_turns"

    except Exception as exc:
        result.termination = "error"
        result.error = str(exc)
        logger.exception("Agent loop error")

    finally:
        await executor.close()
        if hasattr(cli_executor, "close"):
            cli_executor.close()

    result.messages = messages
    result.latency_ms = int((time.time() - start) * 1000)
    return result


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
