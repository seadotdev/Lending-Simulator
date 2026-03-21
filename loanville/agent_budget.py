"""OpenRouter key provisioning and budget tracking — thin re-exports from agent-preflight.

All budget, provisioning, and preflight functionality now lives in the
``agent-preflight`` library.  This module re-exports the public API so that
existing callers (agent_loop, agent_sim, agent_docker) continue to work
without import changes.
"""

from agent_preflight import (  # noqa: F401
    BudgetTracker,
    BudgetStatus,
    CacheMetrics,
    CacheStats,
    PreflightError,
    RunResult,
    delete_key,
    get_account_balance,
    get_available_models,
    get_usage,
    preflight,
    provision_key,
    provision_key_full,
)

# Backwards-compat alias — old call sites use preflight_budget()
preflight_budget = preflight
