"""
Executors for the agentic LOS simulation.

Three modes:
  - tool_call: function-calling → LOS REST API via httpx
  - cli: extract `los` commands from model text → subprocess
  - repl: persistent `los` subprocess session

Also handles custom tool execution from LenderToolkit.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import httpx

from .custom_tools import LenderToolkit
from .los_tools import BODY_REMAP, PATH_ARGS, TOOL_ROUTES

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str  # JSON string
    is_error: bool = False


# ---------------------------------------------------------------------------
# Tool-call executor
# ---------------------------------------------------------------------------

class ToolCallExecutor:
    """Dispatches OpenAI-style tool calls to LOS REST API or custom tools."""

    def __init__(
        self,
        los_url: str = "http://localhost:3000",
        tenant_id: str = "default",
        actor: str = "agent",
        toolkit: LenderToolkit | None = None,
        provider: str = "openrouter",
    ):
        self.los_url = los_url.rstrip("/")
        self.tenant_id = tenant_id
        self.actor = actor
        self.toolkit = toolkit
        self.provider = provider
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={
                    "Content-Type": "application/json",
                    "X-Tenant-Id": self.tenant_id,
                    "X-Actor": self.actor,
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, tool_calls: list[dict]) -> list[ToolResult]:
        """Execute a batch of tool calls, returning results."""
        results = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func.get("name", "")
            tc_id = tc.get("id", name)
            raw_args = func.get("arguments", "{}")

            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args

            result = await self._dispatch(name, args, tc_id)
            results.append(result)
        return results

    async def _dispatch(self, name: str, args: dict, tc_id: str) -> ToolResult:
        # Check for completion signal
        if name == "agent_done":
            return ToolResult(
                tool_call_id=tc_id,
                name=name,
                content=json.dumps({"status": "done", **args}),
            )

        # Check custom tools first
        if self.toolkit:
            tool = self.toolkit.get_tool(name)
            if tool:
                return await self._execute_custom_tool(name, args, tc_id, tool.implementation)

        # LOS API tool
        if name in TOOL_ROUTES:
            return await self._execute_los_tool(name, args, tc_id)

        return ToolResult(
            tool_call_id=tc_id,
            name=name,
            content=json.dumps({"error": f"Unknown tool: {name}"}),
            is_error=True,
        )

    async def _execute_los_tool(self, name: str, args: dict, tc_id: str) -> ToolResult:
        method, url_template = TOOL_ROUTES[name]
        client = await self._get_client()

        # Separate path args from body/query args
        path_keys = PATH_ARGS.get(name, [])
        url_args = {k: args[k] for k in path_keys if k in args}
        body_args = {k: v for k, v in args.items() if k not in path_keys}

        # Apply body remapping
        remap = BODY_REMAP.get(name, {})
        remapped = {}
        for k, v in body_args.items():
            remapped[remap.get(k, k)] = v
        body_args = remapped

        # Build URL — handle missing path args gracefully
        try:
            url = self.los_url + url_template.format(**url_args)
        except KeyError as exc:
            return ToolResult(
                tool_call_id=tc_id,
                name=name,
                content=json.dumps({
                    "error": f"Missing required argument: {exc}",
                    "provided": list(args.keys()),
                }),
                is_error=True,
            )

        # Special handling for certain tools
        if name == "los_deal_evaluate":
            body_args.setdefault("mode", "full")
            body_args["provider"] = self.provider
            body_args["allow_rules_fallback"] = True
        elif name == "los_underwrite":
            dossier_str = body_args.pop("dossier_json", "{}")
            try:
                dossier = json.loads(dossier_str)
            except json.JSONDecodeError:
                dossier = {}
            body_args = {"dossier": dossier, "provider": self.provider}
        elif name == "los_spread_create":
            # Parse metrics from JSON string if provided
            metrics = body_args.get("metrics")
            if isinstance(metrics, str):
                try:
                    body_args["metrics"] = json.loads(metrics)
                except json.JSONDecodeError:
                    pass
            # Parse line_items from JSON string if provided
            items = body_args.get("line_items")
            if isinstance(items, str):
                try:
                    body_args["line_items"] = json.loads(items)
                except json.JSONDecodeError:
                    pass
        elif name == "los_deal_advance":
            # Remap id to URL, rest is body
            pass

        try:
            if method == "GET":
                resp = await client.get(url, params=body_args if body_args else None)
            elif method == "POST":
                resp = await client.post(url, json=body_args)
            elif method == "PATCH":
                resp = await client.patch(url, json=body_args)
            elif method == "DELETE":
                resp = await client.delete(url)
            else:
                return ToolResult(tc_id, name, json.dumps({"error": f"Unsupported method: {method}"}), True)

            if resp.status_code >= 400:
                return ToolResult(
                    tool_call_id=tc_id,
                    name=name,
                    content=json.dumps({
                        "error": f"HTTP {resp.status_code}",
                        "detail": resp.text[:500],
                    }),
                    is_error=True,
                )

            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text[:1000]}

            return ToolResult(
                tool_call_id=tc_id,
                name=name,
                content=json.dumps(data, default=str),
            )

        except Exception as exc:
            return ToolResult(
                tool_call_id=tc_id,
                name=name,
                content=json.dumps({"error": str(exc)}),
                is_error=True,
            )

    async def _execute_custom_tool(
        self, name: str, args: dict, tc_id: str, implementation: str,
    ) -> ToolResult:
        """Execute a custom tool's bash implementation in a temp sandbox."""
        if self.toolkit:
            self.toolkit.record_usage(name)

        sandbox_dir = tempfile.mkdtemp(prefix="loanville_sandbox_")
        data_dir = os.path.join(sandbox_dir, "data")
        os.makedirs(data_dir)

        # Write args as data files
        if args:
            with open(os.path.join(data_dir, "args.json"), "w") as f:
                json.dump(args, f)

        try:
            result = subprocess.run(
                ["bash", "-c", implementation],
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DATA_DIR": data_dir},
            )
            output = result.stdout.strip() or result.stderr.strip() or "(no output)"
            return ToolResult(
                tool_call_id=tc_id,
                name=name,
                content=json.dumps({"output": output, "exit_code": result.returncode}),
                is_error=result.returncode != 0,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(tc_id, name, json.dumps({"error": "Tool execution timed out"}), True)
        except Exception as exc:
            return ToolResult(tc_id, name, json.dumps({"error": str(exc)}), True)


# ---------------------------------------------------------------------------
# CLI executor
# ---------------------------------------------------------------------------

_LOS_CMD_RE = re.compile(r"```(?:bash|sh)?\s*\n(los\s+.*?)```", re.DOTALL)
_LOS_INLINE_RE = re.compile(r"^(los\s+\S+.*)$", re.MULTILINE)


def extract_los_commands(text: str) -> list[str]:
    """Extract `los ...` commands from model text output."""
    cmds = []
    for m in _LOS_CMD_RE.finditer(text):
        for line in m.group(1).strip().splitlines():
            line = line.strip()
            if line.startswith("los "):
                cmds.append(line)
    if not cmds:
        for m in _LOS_INLINE_RE.finditer(text):
            cmds.append(m.group(1).strip())
    return cmds


class CLIExecutor:
    """Executes `los` CLI commands via subprocess."""

    def __init__(
        self,
        los_url: str = "http://localhost:3000",
        tenant_id: str = "default",
        actor: str = "agent",
    ):
        self.env = {
            **os.environ,
            "LOS_API_URL": los_url,
            "LOS_TENANT_ID": tenant_id,
            "LOS_ACTOR": actor,
            "LOS_FORMAT": "json",
        }

    def execute(self, commands: list[str]) -> str:
        """Run CLI commands sequentially, return combined output."""
        outputs = []
        for cmd in commands:
            parts = cmd.split()
            try:
                result = subprocess.run(
                    ["npx", *parts],
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                out = result.stdout.strip() or result.stderr.strip()
                outputs.append(f"$ {cmd}\n{out}")
            except subprocess.TimeoutExpired:
                outputs.append(f"$ {cmd}\nERROR: Command timed out")
            except Exception as exc:
                outputs.append(f"$ {cmd}\nERROR: {exc}")
        return "\n\n".join(outputs)


# ---------------------------------------------------------------------------
# REPL executor
# ---------------------------------------------------------------------------

class REPLExecutor:
    """Persistent `los` REPL subprocess for session-based interaction."""

    def __init__(
        self,
        los_url: str = "http://localhost:3000",
        tenant_id: str = "default",
        actor: str = "agent",
    ):
        self.env = {
            **os.environ,
            "LOS_API_URL": los_url,
            "LOS_TENANT_ID": tenant_id,
            "LOS_ACTOR": actor,
            "LOS_FORMAT": "json",
        }
        self._proc: subprocess.Popen | None = None

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                ["npx", "los", "repl"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
        return self._proc

    def execute(self, commands: list[str]) -> str:
        """Send commands to REPL, return combined output."""
        # For now, delegate to CLI executor since REPL may not be available.
        # This can be upgraded to true REPL interaction later.
        cli = CLIExecutor(
            los_url=self.env.get("LOS_API_URL", "http://localhost:3000"),
            tenant_id=self.env.get("LOS_TENANT_ID", "default"),
            actor=self.env.get("LOS_ACTOR", "agent"),
        )
        return cli.execute(commands)

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc = None
