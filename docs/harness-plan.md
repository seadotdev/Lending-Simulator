# Plan: Shared Experiment Harness + Loanville Integration

## Context

Two projects run LLM models against benchmarks via OpenRouter:

- **snake-arena** (`experiments/pi-docker/run_experiment_v2.py`) — Models build game strategies in Docker containers. Pi coding agent drives tool use. 920 lines → 1150 lines in v2.
- **loanville** (`loanville/agent_sim.py` + `agent_loop.py`) — Models autonomously drive a Loan Origination System. Python drives OpenRouter directly. ~1650 lines across 6 files.

Both projects independently implement: OpenRouter balance checking, model validation, cost tracking, preflight checks, budget estimation, and run lifecycle management. The v2 adds parallelism, per-model logging, P25 eject, and Pareto plots.

## v1 → v2 diff (snake-arena)

| Feature | v1 | v2 |
|---|---|---|
| Execution | Sequential | `ThreadPoolExecutor` — all models in parallel |
| Logging | `print()` | Thread-local `tlog()` → per-model `run.log` + `[alias]`-prefixed stdout |
| Eject policy | No-test after 20% budget | + P25 eject: pause container, benchmark best snapshot, kill if below leaderboard P25 |
| `--model` | Single alias | Multiple aliases (`--model glm5 kimi`) |
| `--budget` | Not available | Override `KEY_LIMIT_PER_MODEL` for testing |
| Duration | Not tracked | `duration_s` in summary |
| Cost timing | Poll OR after run | Use last injected usage (avoids OR lag) |
| Post-run | Simple table | + 3-axis Pareto plot (ELO vs cost vs duration) |
| Smoke prompt | 5 tasks | 7 tasks + mandatory final results.txt |

## What to share vs keep project-specific

### Shared package: `or_harness`

A small Python package (~400 lines) in its own repo or as a git submodule, installable via pip. Zero non-stdlib dependencies (uses `urllib.request` like snake-arena does).

```
or_harness/
├── __init__.py
├── keys.py          # Key provisioning, balance, usage polling
├── preflight.py     # Preflight check framework
├── pricing.py       # Model pricing lookup + cost estimation
├── budget.py        # Budget tracking + injection
├── eject.py         # Eject policy framework
├── events.py        # JSONL event logger
├── status.py        # Live status dashboard data
└── report.py        # Post-run summary + Pareto plotting
```

#### `keys.py` — OpenRouter key management
```python
def load_or_key(env_paths: list[Path] = None) -> str
def load_admin_key(env_paths: list[Path] = None) -> str
def provision_key(admin_key: str, label: str, limit_usd: float) -> str
def get_usage(key: str) -> float | None
def get_balance(key: str) -> tuple[float, float, float]  # total, used, available
```

Both projects currently inline these as `_load_or_key()`, `_load_admin_key()`, `provision_key()`, `get_openrouter_usage()`. Identical logic, different variable names.

#### `pricing.py` — Model pricing + cost estimation
```python
def fetch_pricing(or_key: str) -> dict[str, tuple[float, float]]
def estimate_cost(model_id: str, input_tokens: int, output_tokens: int, pricing: dict) -> float
def estimate_budget_tokens(model_id: str, budget_usd: float, pricing: dict, io_ratio: float = 2.0) -> int
```

Snake-arena fetches live pricing from OR `/v1/models`. Loanville has a hardcoded `MODEL_PRICING` dict in `cost_tracking.py`. The shared version fetches live but falls back to a bundled cache.

#### `preflight.py` — Preflight check framework
```python
@dataclass
class CheckResult:
    name: str
    ok: bool | None  # True=pass, False=fail, None=warning
    detail: str = ""

def check_balance(or_key: str, min_balance: float = 0.10) -> CheckResult
def check_provisioning(admin_key: str) -> CheckResult
def check_models(or_key: str, models: list[str]) -> list[CheckResult]
def check_service(url: str, name: str = "service") -> CheckResult
def run_preflight(checks: list[CheckResult]) -> bool  # prints + returns pass/fail
```

Loanville's `preflight_season()` (in `los_adapter.py`) checks LOS health, model availability, credits, and does a smoke evaluation. Snake-arena checks Docker, OR balance, key provisioning, model availability, API health, and pi CLI. Same pattern, different checks. The framework should be shared; each project registers its own checks.

#### `budget.py` — Budget tracking
```python
@dataclass
class BudgetTracker:
    limit_usd: float
    model_key: str
    poll_interval_s: float = 30.0

    def poll(self) -> float | None  # returns remaining USD, or None
    def inject_prompt(self) -> str  # "Budget: $0.35 remaining of $0.50 (70% left)"
    def fraction_spent(self) -> float  # 0.0 to 1.0
```

Snake-arena writes `budget.txt` into the container every 30s. Loanville has no budget awareness — the agent loop runs blind until max_turns. This is the single most valuable feature to port.

#### `eject.py` — Eject policy framework
```python
@dataclass
class EjectDecision:
    should_eject: bool
    reason: str = ""

class EjectPolicy:
    def check(self, ctx: dict) -> EjectDecision

class NoProgressEject(EjectPolicy):
    """Eject if no meaningful progress after X% of budget."""
    min_actions: int = 25
    budget_fraction: float = 0.20

class QualityEject(EjectPolicy):
    """Eject if performance below threshold after X% of budget."""
    budget_fraction: float = 0.25
    quality_threshold: float = 0.25  # project-specific metric
```

Snake-arena has two: no-test eject (20% budget, no `snake-arena test` run) and P25 eject (25% budget, WR below leaderboard P25). Loanville equivalent: no-deal eject (agent hasn't created a deal after 30% of turns) and quality eject (agent's decisions are all wrong after 50% of turns).

#### `events.py` — JSONL event logger
```python
class EventLogger:
    def __init__(self, path: Path)
    def log(self, event: dict) -> None  # append to JSONL
    def close(self) -> None

    @staticmethod
    def read(path: Path) -> list[dict]
```

Snake-arena streams pi events to `events.jsonl`. Loanville keeps messages in memory only. Adding JSONL logging to agent_loop.py enables post-run debugging, the status dashboard, and replay.

#### `status.py` — Live status data
```python
@dataclass
class ModelStatus:
    alias: str
    model_id: str
    running: bool
    events: int
    tool_calls: int
    budget_remaining: float | None
    has_progress: bool  # project-specific: has_tested / has_deal
    error: str | None
    done: bool
    cost: float

def summarize_run(run_dir: Path, status_fn: Callable) -> list[ModelStatus]
def print_status_table(statuses: list[ModelStatus]) -> None
```

Snake-arena's `status.py` is already a standalone script. The shared version provides the table formatting; each project provides the `status_fn` that reads project-specific progress markers.

#### `report.py` — Post-run report + Pareto
```python
def print_summary_table(results: list[dict], columns: list[str]) -> None
def generate_pareto_plot(results: list[dict], run_dir: Path,
                         quality_key: str, cost_key: str, time_key: str) -> Path | None
```

Snake-arena's v2 Pareto plot is generic — just needs (quality, cost, time) tuples. Loanville can use (decision_accuracy, cost, latency).

### Project-specific (NOT shared)

| Concern | Snake-arena | Loanville |
|---|---|---|
| Isolation | Docker containers | LOS tenant IDs |
| Agent runtime | Pi coding agent (RPC subprocess) | Python httpx → OpenRouter |
| Progress metric | `snake-arena test` win rate | Deal created + correct decision |
| Output artifact | `strategy.py` snapshots | LOS state (deals, entities, docs) |
| Submission | `snake-arena submit` | Leaderboard match record |
| Eject criteria | No test / below P25 WR | No deal / wrong decisions |

## Implementation plan for Loanville

Rather than building `or_harness` as a separate package immediately, start by adding the features directly to Loanville's agent sim. Extract to shared package when snake-arena needs the same code (or vice versa).

### Phase 1: Budget + eject (highest impact)

**Files:** `loanville/agent_loop.py`, `loanville/agent_sim.py`

1. **Per-model key provisioning** — Before each agent loop, provision a capped OR sub-key via `OR_ADMIN_KEY`. Pass the sub-key to `_call_model()` instead of the shared key. Add `--agent-budget` CLI flag (default: $0.50/model).

2. **Budget tracking in agent loop** — Poll OR usage every 30s during the loop. Inject budget info as a system message: "Budget: $0.35 remaining. Pace yourself." This is cheaper than snake-arena's file injection since we control the message stream directly.

3. **No-progress eject** — After 30% of max_turns with no deal created in the LOS, terminate the loop early. Check `_inspect_los_state()` for deal count.

4. **Quality eject** — At 50% of max_turns, if the agent has created a deal but hasn't advanced past origination stage, inject a warning. At 75%, terminate if still stuck.

### Phase 2: Event logging + status

**Files:** `loanville/agent_loop.py`, new `loanville/agent_status.py`

5. **JSONL event log** — Write each tool call and model response to `runs/{run_id}/{lender}/{borrower}/events.jsonl`. Events: `model_call`, `tool_call`, `tool_result`, `budget_poll`, `eject`, `done`.

6. **Status dashboard** — `python -m loanville status` reads the latest run dir, shows per-lender×borrower progress table (like snake-arena's `status.py`). Columns: lender, borrower, turns, tool_calls, has_deal, has_spread, decision, cost, status.

### Phase 3: Parallel + reporting

**Files:** `loanville/agent_sim.py`

7. **Parallel lender execution** — Run all lenders concurrently per borrower using `asyncio.gather()` (already async). Each lender gets its own provisioned key and tenant.

8. **Per-model logging** — Thread-safe logger (like v2's `tlog()`) that writes to per-lender log file and `[alias]`-prefixed stdout.

9. **Duration tracking** — Add `duration_s` to `CaseResult`. Already have `latency_ms` in `AgentLoopResult` but not at the case level.

10. **Pareto plot** — After all cases, generate decision_accuracy vs cost vs latency Pareto plot. Save to `runs/{run_id}/pareto.png`.

### Phase 4: Extract shared package

Once both projects use the same patterns:

11. **Create `or_harness` package** — Extract `keys.py`, `pricing.py`, `preflight.py`, `budget.py`, `events.py`, `status.py`, `report.py` into a pip-installable package.

12. **Refactor both projects** — Replace inline implementations with `from or_harness import ...`.

## CLI changes

```
# New flags
--agent-budget 0.50        # USD cap per model (default: 0.50)
--agent-parallel            # Run lenders concurrently (default: sequential)
--agent-eject               # Enable eject policies (default: off)

# New subcommand
python -m loanville status  # Live status dashboard for running agent sim
```

## File changes summary

| File | Change |
|---|---|
| `loanville/agent_loop.py` | Add budget tracking, budget injection messages, JSONL event logging |
| `loanville/agent_sim.py` | Add key provisioning, eject policies, parallel execution, duration tracking, Pareto plot |
| `loanville/agent_status.py` | New — status dashboard (reads events.jsonl + summary.json) |
| `loanville/__main__.py` | Add `--agent-budget`, `--agent-parallel`, `--agent-eject`, `status` subcommand |
| `loanville/cost_tracking.py` | Replace hardcoded pricing with live OR fetch + fallback cache |

## Verification

1. `python -m loanville --agent-sim --agent-cases 1 --agent-budget 0.10` — single case with $0.10 cap, verify key provisioned + usage tracked
2. Budget injection visible in agent messages after 30s
3. No-progress eject fires when model loops without creating a deal
4. `python -m loanville status` shows live table during a run
5. JSONL events written and readable
6. `--agent-parallel` runs 3 lenders concurrently, each with own key
7. Pareto plot generated at end of run
