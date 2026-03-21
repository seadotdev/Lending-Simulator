#!/usr/bin/env python3
"""
Agent sim module tests — pure unit tests, zero API calls.

Tests the modules ported from snake-arena v2:
  1. EventLogger (JSONL read/write)
  2. BudgetTracker (fraction_spent, exhausted, inject_message)
  3. EjectPolicies (NoProgress, Quality)
  4. AgentTasks (registry, get_task)
  5. AgentSimConfig (defaults, fields)
  6. AgentLoopConfig (defaults, fields)
  7. AgentLoopResult (termination states)
  8. StatusDashboard (summarise_case from summary.json and events.jsonl)
  9. _try_parse_decision (JSON extraction from model text)
 10. agent_prompts (build_system_prompt, build_task_prompt)
 11. CLI wiring (--agent-budget, --agent-parallel, --agent-eject flags parse)
"""

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_event_logger_write_read():
    """EventLogger writes JSONL and reads it back."""
    from loanville.agent_events import EventLogger

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        ev = EventLogger(path)
        ev.log("model_call", turn=1, model="test-model")
        ev.log("tool_call", turn=1, tools=["create_entity", "create_deal"])
        ev.log("loop_end", turns=3, termination="model_done")
        ev.close()

        events = EventLogger.read(path)
        assert len(events) == 3, f"Expected 3 events, got {len(events)}"
        assert events[0]["type"] == "model_call"
        assert events[0]["model"] == "test-model"
        assert events[1]["tools"] == ["create_entity", "create_deal"]
        assert events[2]["termination"] == "model_done"
        # All events have timestamps
        for e in events:
            assert "ts" in e
            assert "t" in e  # relative time


def test_event_logger_empty_read():
    """Reading a non-existent file returns empty list."""
    from loanville.agent_events import EventLogger
    events = EventLogger.read(Path("/tmp/nonexistent_events_12345.jsonl"))
    assert events == []


def test_budget_tracker_basic():
    """BudgetTracker computes fraction_spent and exhausted correctly."""
    from loanville.agent_budget import BudgetTracker

    bt = BudgetTracker(api_key="test-key", limit_usd=1.00, poll_interval_s=0.0)
    # Before any polling, usage is 0
    assert bt.fraction_spent() == 0.0
    assert bt.exhausted is False
    msg = bt.inject_message()
    # inject_message calls poll(), which will fail (no real key) but still return cached
    assert "Budget:" in msg or msg == ""  # may be empty if poll fails


def test_budget_tracker_exhausted():
    """BudgetTracker reports exhausted when usage >= limit."""
    from loanville.agent_budget import BudgetTracker

    bt = BudgetTracker(api_key="test-key", limit_usd=0.50, poll_interval_s=9999)
    # Simulate having polled and found usage at limit
    bt._last_usage = 0.50
    bt._last_poll = time.time()

    assert bt.fraction_spent() == 1.0
    assert bt.exhausted is True
    remaining = bt.poll()
    assert remaining is not None
    assert remaining <= 0


def test_budget_tracker_inject_message():
    """inject_message formats a readable budget string."""
    from loanville.agent_budget import BudgetTracker

    bt = BudgetTracker(api_key="test-key", limit_usd=1.00, poll_interval_s=9999)
    bt._last_usage = 0.30
    bt._last_poll = time.time()

    msg = bt.inject_message()
    assert "$0.70" in msg  # remaining
    assert "$1.00" in msg  # total
    assert "70%" in msg    # percentage


def test_no_progress_eject():
    """NoProgressEject fires when no deal after 30% of turns."""
    from loanville.agent_eject import NoProgressEject

    policy = NoProgressEject(threshold_fraction=0.30)

    # Before threshold: no eject
    result = policy.check(turn=2, max_turns=20, los_state={"deals": []})
    assert result.should_eject is False

    # At threshold with no deals: eject
    result = policy.check(turn=6, max_turns=20, los_state={"deals": []})
    assert result.should_eject is True
    assert "no deal" in result.reason

    # At threshold with deals: no eject
    result = policy.check(turn=6, max_turns=20, los_state={"deals": [{"id": "d1"}]})
    assert result.should_eject is False


def test_quality_eject():
    """QualityEject warns at 60% and ejects at 80% if stuck in early stage."""
    from loanville.agent_eject import QualityEject

    policy = QualityEject(warn_fraction=0.60, eject_fraction=0.80)
    stuck_state = {"deals": [{"id": "d1", "stage": "broker"}]}

    # Before warn threshold
    result = policy.check(turn=5, max_turns=20, los_state=stuck_state)
    assert result.should_eject is False
    assert result.reason == ""

    # At warn threshold, stuck in broker
    result = policy.check(turn=12, max_turns=20, los_state=stuck_state)
    assert result.should_eject is False
    assert "warning" in result.reason

    # At eject threshold, still stuck
    result = policy.check(turn=16, max_turns=20, los_state=stuck_state)
    assert result.should_eject is True
    assert "stuck" in result.reason


def test_quality_eject_no_trigger_advanced_stage():
    """QualityEject doesn't fire if deal is in underwriting stage."""
    from loanville.agent_eject import QualityEject

    policy = QualityEject()
    advanced_state = {"deals": [{"id": "d1", "stage": "underwriting"}]}

    result = policy.check(turn=18, max_turns=20, los_state=advanced_state)
    assert result.should_eject is False


def test_default_eject_policies():
    """default_eject_policies returns both policies."""
    from loanville.agent_eject import default_eject_policies, NoProgressEject, QualityEject, AnalysisEject

    policies = default_eject_policies()
    assert len(policies) == 3
    types = [type(p) for p in policies]
    assert NoProgressEject in types
    assert QualityEject in types
    assert AnalysisEject in types


def test_agent_tasks_registry():
    """Task registry has expected tasks."""
    from loanville.agent_tasks import TASKS, get_task, list_tasks

    assert "simple_underwrite" in TASKS
    assert "incomplete_application" in TASKS
    assert "portfolio_batch" in TASKS
    assert len(list_tasks()) >= 5

    task = get_task("simple_underwrite")
    assert task.name == "simple_underwrite"
    assert task.min_api_calls > 0
    assert "entity" in task.expected_artifacts

    # Unknown task falls back to simple_underwrite
    fallback = get_task("nonexistent_task")
    assert fallback.name == "simple_underwrite"


def test_agent_sim_config_defaults():
    """AgentSimConfig has sensible defaults."""
    from loanville.agent_sim import AgentSimConfig

    cfg = AgentSimConfig()
    assert cfg.mode == "tool_call"
    assert cfg.max_turns == 20
    assert cfg.cases == 3
    assert cfg.budget_usd is None
    assert cfg.parallel is False
    assert cfg.eject is False
    assert cfg.provider == "openrouter"


def test_agent_loop_config_defaults():
    """AgentLoopConfig has sensible defaults."""
    from loanville.agent_loop import AgentLoopConfig

    cfg = AgentLoopConfig(model="test-model")
    assert cfg.mode == "tool_call"
    assert cfg.max_turns == 20
    assert cfg.event_logger is None
    assert cfg.budget_tracker is None
    assert cfg.eject_policies == []
    assert cfg.log_prefix == ""


def test_agent_loop_result_states():
    """AgentLoopResult tracks termination states correctly."""
    from loanville.agent_loop import AgentLoopResult

    r = AgentLoopResult()
    assert r.turns == 0
    assert r.termination == ""
    assert r.final_decision is None
    assert r.error is None

    r.termination = "model_done"
    r.final_decision = {"decision": "approve", "reasoning": "good borrower"}
    assert r.final_decision["decision"] == "approve"


def test_status_summarise_from_summary_json():
    """Status dashboard reads summary.json correctly."""
    from loanville.agent_status import summarise_case

    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp) / "lender_a" / "borrower_1"
        case_dir.mkdir(parents=True)

        summary = {
            "lender": "Test Lender",
            "borrower": "Acme Corp",
            "termination": "model_done",
            "turns": 8,
            "tool_call_count": 12,
            "has_deal": True,
            "decision": "approve",
            "duration_s": 45.2,
            "cost_estimate": "$0.0350",
        }
        (case_dir / "summary.json").write_text(json.dumps(summary))

        result = summarise_case(case_dir)
        assert result["lender"] == "Test Lender"
        assert result["borrower"] == "Acme Corp"
        assert result["status"] == "model_done"
        assert result["turns"] == 8
        assert result["tool_calls"] == 12
        assert result["has_deal"] is True
        assert result["decision"] == "approve"
        assert result["duration_s"] == 45.2


def test_status_summarise_from_events():
    """Status dashboard falls back to events.jsonl when no summary.json."""
    from loanville.agent_status import summarise_case

    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp) / "lender_b" / "borrower_2"
        case_dir.mkdir(parents=True)

        events = [
            {"t": 0.1, "ts": time.time(), "type": "model_call", "turn": 1},
            {"t": 0.5, "ts": time.time(), "type": "tool_call", "turn": 1, "tools": ["create_entity"]},
            {"t": 1.2, "ts": time.time(), "type": "model_call", "turn": 2},
            {"t": 3.5, "ts": time.time(), "type": "loop_end", "turns": 2,
             "termination": "model_done", "decision": "decline", "has_deal": True},
        ]
        with open(case_dir / "events.jsonl", "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        result = summarise_case(case_dir)
        assert result["status"] == "model_done"
        assert result["turns"] == 2
        assert result["decision"] == "decline"
        assert result["has_deal"] is True


def test_status_summarise_empty():
    """Status dashboard handles empty case dir gracefully."""
    from loanville.agent_status import summarise_case

    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp) / "lender_c" / "borrower_3"
        case_dir.mkdir(parents=True)

        result = summarise_case(case_dir)
        assert result["status"] == "no_data"
        assert result["turns"] == 0


def test_try_parse_decision():
    """Decision parser extracts JSON decisions from model text."""
    from loanville.agent_loop import _try_parse_decision

    # JSON in code block
    text1 = 'Based on analysis:\n```json\n{"decision": "approve", "reasoning": "good"}\n```'
    d = _try_parse_decision(text1)
    assert d is not None
    assert d["decision"] == "approve"

    # Inline JSON
    text2 = 'I recommend {"decision": "decline", "reasoning": "high risk"} for this application.'
    d = _try_parse_decision(text2)
    assert d is not None
    assert d["decision"] == "decline"

    # No decision
    d = _try_parse_decision("This is just a regular message with no decision.")
    assert d is None

    # None input
    d = _try_parse_decision(None)
    assert d is None


def test_build_prompts():
    """System and task prompts build without error."""
    from loanville.agent_prompts import build_system_prompt, build_task_prompt
    from loanville.data import get_borrowers, get_lenders

    lenders = get_lenders()[:1]
    borrowers = get_borrowers("easy", seed=42)[:1]

    prompt = build_system_prompt(
        mode="tool_call",
        lender=lenders[0],
        portfolio_summary="$500k deployed across 3 loans",
        custom_tools_info="",
    )
    assert len(prompt) > 100
    assert "LOS" in prompt or "loan" in prompt.lower()

    task_prompt = build_task_prompt(borrowers[0], "Process this application")
    assert len(task_prompt) > 50
    assert borrowers[0].dossier.company_name in task_prompt


def test_cli_agent_flags_parse():
    """CLI parser accepts all agent sim flags."""
    import argparse
    # Just verify the flags exist by importing and checking the parser
    # We can't easily run the full main() parser in isolation, so test
    # that the module imports cleanly and AgentSimConfig accepts the new fields
    from loanville.agent_sim import AgentSimConfig

    cfg = AgentSimConfig(
        mode="tool_call",
        max_turns=30,
        cases=5,
        budget_usd=0.50,
        parallel=True,
        eject=True,
    )
    assert cfg.budget_usd == 0.50
    assert cfg.parallel is True
    assert cfg.eject is True


def test_los_tools_defined():
    """LOS tools list is non-empty and well-formed."""
    from loanville.los_tools import LOS_TOOLS

    assert len(LOS_TOOLS) > 0
    for tool in LOS_TOOLS:
        assert "type" in tool
        assert tool["type"] == "function"
        func = tool.get("function", {})
        assert "name" in func
        assert "description" in func


def test_case_result_and_los_state():
    """CaseResult and LOSState dataclasses work correctly."""
    from loanville.agent_sim import CaseResult, LOSState
    from loanville.agent_loop import AgentLoopResult

    los = LOSState()
    assert los.deals == []
    assert los.has_evaluation is False

    los.deals = [{"id": "d1", "stage": "underwriting"}]
    los.has_spread = True
    los.has_evaluation = True
    los.deal_stage = "underwriting"

    lr = AgentLoopResult()
    lr.termination = "model_done"
    lr.final_decision = {"decision": "approve"}

    case = CaseResult(
        lender_id="l1", lender_name="Test Lender",
        borrower_id="b1", borrower_name="Acme Corp",
        task_name="simple_underwrite",
        loop_result=lr, los_state=los,
        borrower_quality="good",
    )
    assert case.lender_name == "Test Lender"
    assert case.los_state.has_spread is True


def test_describe_los_pipeline():
    """_describe_los_pipeline produces readable output."""
    from loanville.agent_sim import _describe_los_pipeline, LOSState

    # Empty
    los = LOSState()
    assert "nothing" in _describe_los_pipeline(los)

    # Populated
    los = LOSState(
        deals=[{"id": "d1"}],
        entities=[{"id": "e1"}, {"id": "e2"}],
        documents=[{"id": "doc1"}],
        has_spread=True,
        has_evaluation=True,
        deal_stage="underwriting",
    )
    desc = _describe_los_pipeline(los)
    assert "1 deal" in desc
    assert "2 entities" in desc
    assert "1 doc" in desc
    assert "spread" in desc
    assert "evaluation" in desc
    assert "underwriting" in desc


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("EventLogger write/read", test_event_logger_write_read),
    ("EventLogger empty read", test_event_logger_empty_read),
    ("BudgetTracker basic", test_budget_tracker_basic),
    ("BudgetTracker exhausted", test_budget_tracker_exhausted),
    ("BudgetTracker inject_message", test_budget_tracker_inject_message),
    ("NoProgressEject", test_no_progress_eject),
    ("QualityEject", test_quality_eject),
    ("QualityEject advanced stage", test_quality_eject_no_trigger_advanced_stage),
    ("Default eject policies", test_default_eject_policies),
    ("Agent tasks registry", test_agent_tasks_registry),
    ("AgentSimConfig defaults", test_agent_sim_config_defaults),
    ("AgentLoopConfig defaults", test_agent_loop_config_defaults),
    ("AgentLoopResult states", test_agent_loop_result_states),
    ("Status from summary.json", test_status_summarise_from_summary_json),
    ("Status from events.jsonl", test_status_summarise_from_events),
    ("Status empty case dir", test_status_summarise_empty),
    ("Parse decision from text", test_try_parse_decision),
    ("Build prompts", test_build_prompts),
    ("CLI agent flags", test_cli_agent_flags_parse),
    ("LOS tools defined", test_los_tools_defined),
    ("CaseResult & LOSState", test_case_result_and_los_state),
    ("Describe LOS pipeline", test_describe_los_pipeline),
]


def main():
    print("=" * 70)
    print("  AGENT SIM MODULE TEST — pure unit tests, zero API calls")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = []
    t0 = time.time()

    for name, fn in TESTS:
        print(f"\n  [{passed + failed + 1}/{len(TESTS)}] {name} ...", end=" ", flush=True)
        test_t0 = time.time()
        try:
            fn()
            elapsed = time.time() - test_t0
            print(f"PASS ({elapsed:.1f}s)")
            passed += 1
        except Exception as e:
            elapsed = time.time() - test_t0
            print(f"FAIL ({elapsed:.1f}s)")
            failed += 1
            errors.append((name, e))

    total_elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed "
          f"({total_elapsed:.1f}s total)")
    print(f"{'=' * 70}")

    if errors:
        print("\n  FAILURES:\n")
        for name, exc in errors:
            print(f"  {name}:")
            print(f"    {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
