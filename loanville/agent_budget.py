"""OpenRouter key provisioning and budget tracking for agent sim."""

from __future__ import annotations

import logging
import os
import time

import httpx

# ---------------------------------------------------------------------------
# Tool call cost table (for CallBudgetTracker)
# ---------------------------------------------------------------------------

# How many call credits each tool consumes.
# Unspecified tools cost 1 credit.  Free tools cost 0.  Expensive tools cost more.
TOOL_CALL_CREDITS: dict[str, int] = {
    # Free (no LOS traffic)
    "agent_done": 0,
    "los_quick_assess": 0,       # cheap/noisy — Option 2
    # Standard LOS calls: 1 credit each (default)
    # Expensive reliable evaluation: 2 credits
    "los_deal_evaluate": 2,
    "los_underwrite": 2,
}

logger = logging.getLogger(__name__)

OR_BASE = "https://openrouter.ai/api/v1"


def provision_key(admin_key: str, label: str, limit_usd: float) -> str:
    """Provision a sub-key with a spending limit via OpenRouter admin API.

    Returns the provisioned API key string.
    """
    resp = httpx.post(
        f"{OR_BASE}/keys",
        headers={
            "Authorization": f"Bearer {admin_key}",
            "Content-Type": "application/json",
        },
        json={
            "name": label,
            "limit": limit_usd,
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()
    key = data.get("key") or data.get("data", {}).get("key", "")
    if not key:
        raise RuntimeError(f"provision_key: no key in response: {data}")
    logger.info("Provisioned sub-key %s… (limit $%.2f)", key[:12], limit_usd)
    return key


def get_usage(api_key: str) -> float | None:
    """Query current usage in USD for an API key. Returns None on error."""
    try:
        resp = httpx.get(
            f"{OR_BASE}/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", {})
        return data.get("usage", 0.0)
    except Exception:
        return None


class BudgetTracker:
    """Tracks spend against a budget limit, with periodic polling."""

    def __init__(
        self,
        api_key: str,
        limit_usd: float,
        poll_interval_s: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.limit_usd = limit_usd
        self.poll_interval_s = poll_interval_s
        self._last_poll: float = 0.0
        self._last_usage: float = 0.0

    def poll(self) -> float | None:
        """Poll usage if enough time has passed. Returns remaining USD."""
        now = time.time()
        if now - self._last_poll < self.poll_interval_s:
            return self.limit_usd - self._last_usage
        self._last_poll = now
        usage = get_usage(self.api_key)
        if usage is not None:
            self._last_usage = usage
        remaining = self.limit_usd - self._last_usage
        return remaining

    def inject_message(self) -> str:
        """Budget status string suitable for injection as a system message."""
        remaining = self.poll()
        if remaining is None:
            return ""
        pct = (1.0 - self.fraction_spent()) * 100
        return f"Budget: ${remaining:.2f} remaining of ${self.limit_usd:.2f} ({pct:.0f}%)"

    def fraction_spent(self) -> float:
        if self.limit_usd <= 0:
            return 1.0
        return self._last_usage / self.limit_usd

    @property
    def exhausted(self) -> bool:
        return self.fraction_spent() >= 1.0


# ---------------------------------------------------------------------------
# LOS call-count budget (for multi-application allocation experiments)
# ---------------------------------------------------------------------------

class CallBudgetTracker:
    """Shared LOS-call-count budget across multiple agent loops.

    Unlike BudgetTracker (which tracks real API spend), this counts
    *LOS tool calls* made by the agent — a simulated resource that
    the model must allocate across multiple borrower applications.

    Experiment design (Option 1 — multi-app allocation):
      - N borrowers share a pool of `total_calls` LOS call credits
      - Each loop receives a "X calls remaining for Y remaining apps" message
      - Score = average decision accuracy across all N apps
      - A model that spreads evenly (~total/N per app) beats one that
        burns all credits on app 1 and guesses for the rest

    Tool cost table (TOOL_CALL_CREDITS in this module):
      - los_quick_assess: 0  (free, noisy — Option 2)
      - los_deal_evaluate: 2 (expensive, reliable)
      - everything else: 1
    """

    def __init__(self, total_calls: int) -> None:
        self.total_calls = total_calls
        self._calls_made = 0

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def calls_remaining(self) -> int:
        return max(0, self.total_calls - self._calls_made)

    @property
    def exhausted(self) -> bool:
        return self._calls_made >= self.total_calls

    def charge(self, tool_names: list[str]) -> int:
        """Deduct credits for the given tool names. Returns remaining credits."""
        cost = sum(TOOL_CALL_CREDITS.get(n, 1) for n in tool_names)
        self._calls_made += cost
        return self.calls_remaining

    def fraction_spent(self) -> float:
        if self.total_calls <= 0:
            return 1.0
        return min(1.0, self._calls_made / self.total_calls)

    def inject_message(self, app_num: int, total_apps: int) -> str:
        """Budget status string for injection at start of each loop."""
        remaining = self.calls_remaining
        apps_left = total_apps - app_num + 1
        optimal = remaining // apps_left if apps_left > 0 else remaining
        return (
            f"LOS Call Budget: {remaining} of {self.total_calls} credits remaining. "
            f"This is application {app_num} of {total_apps}. "
            f"Recommended spend: ~{optimal} credits on this application "
            f"({apps_left - 1} application(s) still need credits after this one). "
            f"los_quick_assess is FREE. los_deal_evaluate costs 2. Other tools cost 1."
        )


def _check(name: str, ok: bool | None, detail: str = "") -> bool:
    symbol = "v" if ok is True else ("!" if ok is None else "x")
    detail_str = f"  {detail}" if detail else ""
    print(f"  [{symbol}] {name}{detail_str}")
    return ok is True


def get_account_balance(api_key: str) -> tuple[float | None, float | None, float | None]:
    """Return (balance, total_credits, total_usage) or Nones on error."""
    try:
        resp = httpx.get(
            f"{OR_BASE}/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None, None, None
        data = resp.json().get("data", {})
        total = float(data.get("total_credits", 0))
        used = float(data.get("total_usage", 0))
        return total - used, total, used
    except Exception:
        return None, None, None


def get_available_models(api_key: str) -> dict[str, dict]:
    """Return {model_id: {pricing info}} from OpenRouter."""
    try:
        resp = httpx.get(
            f"{OR_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        if resp.status_code != 200:
            return {}
        models = resp.json().get("data", [])
        return {m["id"]: m for m in models}
    except Exception:
        return {}


def preflight_budget(
    admin_key: str,
    or_key: str,
    models: list[str],
    budget_per_model: float,
) -> None:
    """Full preflight: balance, provisioning budget, model availability, 1c sanity.

    Raises RuntimeError on any failure that should block the run.
    """
    print(f"\n  {'='*60}")
    print(f"  PREFLIGHT CHECKS")
    print(f"  {'='*60}")
    failures: list[str] = []

    # 1. Account balance
    balance, total, used = get_account_balance(or_key)
    if balance is not None:
        ok = balance > 0.10
        _check("Account balance", ok,
               f"${balance:.2f} available (${total:.2f} purchased, ${used:.2f} used)")
        if not ok:
            failures.append(f"Account balance too low (${balance:.2f})")
    else:
        _check("Account balance", False, "could not fetch")
        failures.append("Could not fetch account balance")

    # 2. Provisioning budget math
    unique_models = list(dict.fromkeys(models))
    n = len(unique_models)
    needed = n * budget_per_model
    if balance is not None:
        ok = balance >= needed
        detail = (f"{n} models x ${budget_per_model:.2f} = ${needed:.2f} "
                  f"({'OK' if ok else 'INSUFFICIENT'}: balance ${balance:.2f})")
        _check("Provisioning budget", ok, detail)
        if not ok:
            failures.append(f"Insufficient balance: need ${needed:.2f}, have ${balance:.2f}")
    else:
        _check("Provisioning budget", None,
               f"{n} x ${budget_per_model:.2f} = ${needed:.2f} (balance unknown)")

    # 3. Model availability + pricing
    available = get_available_models(or_key)
    if available:
        for model_id in unique_models:
            if model_id in available:
                p = available[model_id].get("pricing", {})
                inp = float(p.get("prompt", 0)) * 1_000_000
                out = float(p.get("completion", 0)) * 1_000_000
                _check(f"Model {model_id}", True, f"${inp:.2f}/${out:.2f} per M tokens in/out")
            else:
                _check(f"Model {model_id}", False, "not found on OpenRouter")
                failures.append(f"Model {model_id} not on OpenRouter")
    else:
        _check("Model availability", False, "could not fetch model list")
        failures.append("Could not fetch model list from OpenRouter")

    # 4. Key provisioning sanity (1c)
    try:
        test_key = provision_key(admin_key, "loanville-preflight-1c", limit_usd=0.01)
        # Make a tiny API call through the provisioned key
        resp = httpx.post(
            f"{OR_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {test_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://loanville.dev",
                "X-Title": "Loanville Preflight",
            },
            json={
                "model": "openai/gpt-4.1-nano",
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 16,
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API call failed ({resp.status_code}): {resp.text[:120]}")
        reply = (resp.json().get("choices", [{}])[0]
                 .get("message", {}).get("content", ""))
        usage = get_usage(test_key)
        usage_str = f", usage=${usage:.6f}" if usage is not None else ""
        _check("Key provisioning (1c)", True,
               f"key={test_key[:16]}... reply={reply!r}{usage_str}")
    except Exception as exc:
        _check("Key provisioning (1c)", False, str(exc))
        failures.append(f"Key provisioning failed: {exc}")

    # 5. LOS health (informational — already checked elsewhere)
    print()
    if failures:
        print(f"  PREFLIGHT FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"    x {f}")
        print()
        raise RuntimeError(f"Preflight failed: {failures[0]}")
    print("  All preflight checks passed.\n")
