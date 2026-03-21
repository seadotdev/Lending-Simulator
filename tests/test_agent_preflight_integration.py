"""Tests for agent-preflight integration into Loanville.

Tests that:
1. Re-exported modules work correctly
2. Loanville-specific eject policies work with agent-preflight's interface
3. EventLogger, BudgetTracker, and BudgetStatus are compatible
4. Preflight function is callable through the re-export layer
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_preflight import BudgetStatus, EjectDecision


# ---------------------------------------------------------------------------
# Eject policy tests (domain-specific policies on library base class)
# ---------------------------------------------------------------------------


class TestNoProgressEject:
    """NoProgressEject fires when no deal is created after a fraction of turns."""

    def _make_kwargs(self, turn, max_turns, deals=None):
        return dict(
            elapsed_s=turn * 5.0,
            budget_status=BudgetStatus(used=0.01, limit=0.25),
            events=[],
            context={
                "turn": turn,
                "max_turns": max_turns,
                "los_state": {"deals": deals or []},
            },
        )

    def test_no_eject_before_threshold(self):
        from loanville.agent_eject import NoProgressEject

        policy = NoProgressEject(threshold_fraction=0.30)
        result = policy.check(**self._make_kwargs(turn=3, max_turns=20))
        assert not result.should_eject

    def test_eject_at_threshold_no_deals(self):
        from loanville.agent_eject import NoProgressEject

        policy = NoProgressEject(threshold_fraction=0.30)
        result = policy.check(**self._make_kwargs(turn=6, max_turns=20))
        assert result.should_eject
        assert "no deal" in result.reason

    def test_no_eject_at_threshold_with_deals(self):
        from loanville.agent_eject import NoProgressEject

        policy = NoProgressEject(threshold_fraction=0.30)
        result = policy.check(
            **self._make_kwargs(turn=6, max_turns=20, deals=[{"id": "d1"}])
        )
        assert not result.should_eject

    def test_custom_threshold(self):
        from loanville.agent_eject import NoProgressEject

        policy = NoProgressEject(threshold_fraction=0.50)
        # Turn 9 out of 20 = 45%, below 50%
        result = policy.check(**self._make_kwargs(turn=9, max_turns=20))
        assert not result.should_eject
        # Turn 10 out of 20 = 50%
        result = policy.check(**self._make_kwargs(turn=10, max_turns=20))
        assert result.should_eject


class TestQualityEject:
    """QualityEject fires when stuck in early stages after a fraction of turns."""

    def _make_kwargs(self, turn, max_turns, deals=None):
        return dict(
            elapsed_s=turn * 5.0,
            budget_status=BudgetStatus(used=0.01, limit=0.25),
            events=[],
            context={
                "turn": turn,
                "max_turns": max_turns,
                "los_state": {"deals": deals or []},
            },
        )

    def test_no_eject_without_deals(self):
        from loanville.agent_eject import QualityEject

        policy = QualityEject()
        result = policy.check(**self._make_kwargs(turn=18, max_turns=20))
        assert not result.should_eject

    def test_no_eject_advanced_stage(self):
        from loanville.agent_eject import QualityEject

        policy = QualityEject()
        result = policy.check(
            **self._make_kwargs(
                turn=18, max_turns=20,
                deals=[{"stage": "underwriting"}],
            )
        )
        assert not result.should_eject

    def test_warning_at_warn_fraction(self):
        from loanville.agent_eject import QualityEject

        policy = QualityEject(warn_fraction=0.60, eject_fraction=0.80)
        result = policy.check(
            **self._make_kwargs(
                turn=12, max_turns=20,
                deals=[{"stage": "origination"}],
            )
        )
        assert not result.should_eject
        assert "warning" in result.reason

    def test_eject_at_eject_fraction(self):
        from loanville.agent_eject import QualityEject

        policy = QualityEject(warn_fraction=0.60, eject_fraction=0.80)
        result = policy.check(
            **self._make_kwargs(
                turn=16, max_turns=20,
                deals=[{"stage": "origination"}],
            )
        )
        assert result.should_eject
        assert "stuck" in result.reason

    def test_stuck_stages(self):
        from loanville.agent_eject import QualityEject

        for stage in ("", "broker", "origination", "processing"):
            policy = QualityEject(eject_fraction=0.80)
            result = policy.check(
                **self._make_kwargs(
                    turn=16, max_turns=20,
                    deals=[{"stage": stage}],
                )
            )
            assert result.should_eject, f"Expected eject for stage={stage!r}"


class TestDefaultEjectPolicies:
    def test_returns_list(self):
        from loanville.agent_eject import default_eject_policies

        policies = default_eject_policies()
        assert isinstance(policies, list)
        assert len(policies) == 2

    def test_policies_are_eject_policy_subclasses(self):
        from agent_preflight import EjectPolicy
        from loanville.agent_eject import default_eject_policies

        for policy in default_eject_policies():
            assert isinstance(policy, EjectPolicy)


# ---------------------------------------------------------------------------
# Re-export compatibility tests
# ---------------------------------------------------------------------------


class TestBudgetReExports:
    """Verify that agent_budget.py re-exports match the library API."""

    def test_budget_tracker_importable(self):
        from loanville.agent_budget import BudgetTracker
        from agent_preflight import BudgetTracker as LibTracker
        assert BudgetTracker is LibTracker

    def test_provision_key_importable(self):
        from loanville.agent_budget import provision_key
        from agent_preflight import provision_key as lib_provision
        assert provision_key is lib_provision

    def test_get_usage_importable(self):
        from loanville.agent_budget import get_usage
        from agent_preflight import get_usage as lib_get_usage
        assert get_usage is lib_get_usage

    def test_preflight_budget_alias(self):
        from loanville.agent_budget import preflight_budget
        from agent_preflight import preflight
        assert preflight_budget is preflight

    def test_get_account_balance_importable(self):
        from loanville.agent_budget import get_account_balance
        from agent_preflight import get_account_balance as lib_gab
        assert get_account_balance is lib_gab

    def test_get_available_models_importable(self):
        from loanville.agent_budget import get_available_models
        from agent_preflight import get_available_models as lib_gam
        assert get_available_models is lib_gam

    def test_new_exports_available(self):
        """New library features are available through the re-export."""
        from loanville.agent_budget import BudgetStatus, provision_key_full, delete_key
        assert BudgetStatus is not None
        assert provision_key_full is not None
        assert delete_key is not None


class TestEjectReExports:
    """Library eject classes available through Loanville's agent_eject."""

    def test_library_policies_importable(self):
        from loanville.agent_eject import (
            BudgetExceededEject,
            BudgetFractionEject,
            CompositeEject,
            IdleTimeoutEject,
        )
        assert BudgetExceededEject is not None
        assert BudgetFractionEject is not None
        assert CompositeEject is not None
        assert IdleTimeoutEject is not None


class TestEventLoggerReExport:
    def test_importable(self):
        from loanville.agent_events import EventLogger
        from agent_preflight import EventLogger as LibLogger
        assert EventLogger is LibLogger


# ---------------------------------------------------------------------------
# EventLogger functional tests
# ---------------------------------------------------------------------------


class TestEventLogger:
    def test_write_and_read(self, tmp_path):
        from loanville.agent_events import EventLogger

        log_path = tmp_path / "events.jsonl"
        with EventLogger(log_path) as ev:
            ev.log("tool_start", tool="bash")
            ev.log("tool_end", tool="bash", output="hello")

        events = EventLogger.read(log_path)
        assert len(events) == 2
        assert events[0]["type"] == "tool_start"
        assert events[0]["tool"] == "bash"
        assert events[1]["type"] == "tool_end"
        assert "timestamp" in events[0]

    def test_read_nonexistent(self, tmp_path):
        from loanville.agent_events import EventLogger

        events = EventLogger.read(tmp_path / "nope.jsonl")
        assert events == []


# ---------------------------------------------------------------------------
# BudgetTracker tests
# ---------------------------------------------------------------------------


class TestBudgetTracker:
    def test_construction(self):
        from loanville.agent_budget import BudgetTracker

        tracker = BudgetTracker(api_key="sk-test", limit_usd=0.25)
        assert tracker.api_key == "sk-test"
        assert tracker.limit_usd == 0.25
        assert not tracker.exceeded

    def test_status_message(self):
        from loanville.agent_budget import BudgetTracker

        tracker = BudgetTracker(api_key="sk-test", limit_usd=0.25)
        msg = tracker.status_message()
        assert "remaining" in msg
        assert "$0.25" in msg

    @patch("agent_preflight.budget.get_usage", return_value=0.25)
    def test_exceeded_after_poll(self, mock_usage):
        from loanville.agent_budget import BudgetTracker

        tracker = BudgetTracker(api_key="sk-test", limit_usd=0.25)
        status = tracker.poll(force=True)
        assert status.exceeded
        assert tracker.exceeded


# ---------------------------------------------------------------------------
# BudgetStatus tests
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    def test_remaining(self):
        s = BudgetStatus(used=0.10, limit=0.25)
        assert s.remaining == pytest.approx(0.15)

    def test_pct_spent(self):
        s = BudgetStatus(used=0.10, limit=0.25)
        assert s.pct_spent == pytest.approx(0.40)

    def test_exceeded_false(self):
        s = BudgetStatus(used=0.10, limit=0.25)
        assert not s.exceeded

    def test_exceeded_true(self):
        s = BudgetStatus(used=0.25, limit=0.25)
        assert s.exceeded

    def test_format_simple(self):
        s = BudgetStatus(used=0.10, limit=0.25)
        text = s.format_simple()
        assert "remaining" in text
        assert "$0.25" in text


# ---------------------------------------------------------------------------
# Composite eject (library + domain policies together)
# ---------------------------------------------------------------------------


class TestCompositeWithDomainPolicies:
    """Test that library CompositeEject works with Loanville domain policies."""

    def test_composite_with_no_progress_and_budget(self):
        from loanville.agent_eject import (
            CompositeEject,
            NoProgressEject,
            BudgetExceededEject,
        )

        composite = CompositeEject([
            BudgetExceededEject(),
            NoProgressEject(threshold_fraction=0.30),
        ])

        # Budget not exceeded, but no progress after threshold
        result = composite.check(
            elapsed_s=30.0,
            budget_status=BudgetStatus(used=0.05, limit=0.25),
            events=[],
            context={
                "turn": 7,
                "max_turns": 20,
                "los_state": {"deals": []},
            },
        )
        assert result.should_eject
        assert "no deal" in result.reason

    def test_budget_exceeded_takes_priority(self):
        from loanville.agent_eject import (
            CompositeEject,
            NoProgressEject,
            BudgetExceededEject,
        )

        composite = CompositeEject([
            BudgetExceededEject(),
            NoProgressEject(threshold_fraction=0.30),
        ])

        result = composite.check(
            elapsed_s=30.0,
            budget_status=BudgetStatus(used=0.25, limit=0.25),
            events=[],
            context={
                "turn": 2,
                "max_turns": 20,
                "los_state": {"deals": []},
            },
        )
        assert result.should_eject
        assert "budget" in result.reason.lower()

    def test_idle_timeout_with_domain_policies(self):
        import time
        from loanville.agent_eject import (
            CompositeEject,
            QualityEject,
            IdleTimeoutEject,
        )

        idle = IdleTimeoutEject(timeout_s=0.01)
        composite = CompositeEject([idle, QualityEject()])

        # Simulate idle — set last event time far in the past
        idle._last_event_ts = time.time() - 1

        result = composite.check(
            elapsed_s=100.0,
            budget_status=BudgetStatus(used=0.01, limit=0.25),
            events=[],
            context={
                "turn": 5,
                "max_turns": 20,
                "los_state": {"deals": []},
            },
        )
        assert result.should_eject
        assert "no events" in result.reason


# ---------------------------------------------------------------------------
# Preflight smoke test (mocked — no real API calls)
# ---------------------------------------------------------------------------


class TestPreflightCallable:
    """Ensure the preflight function is callable through the re-export."""

    @patch("agent_preflight.preflight.get_account_balance", return_value=(10.0, 20.0, 10.0))
    def test_preflight_via_re_export(self, mock_balance):
        from loanville.agent_budget import preflight_budget

        # Only run balance + math checks (skip smoke tests)
        failures = preflight_budget(
            admin_key="sk-admin-test",
            or_key="sk-test",
            models=["openai/gpt-4.1-nano"],
            budget_per_model=0.25,
            checks=["balance", "math"],
            raise_on_failure=False,
        )
        assert failures == []
