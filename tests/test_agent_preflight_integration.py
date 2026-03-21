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


class TestAnalysisEject:
    """AnalysisEject detects rubber-stamping via context dict."""

    def _make_kwargs(self, turn, los_state):
        return dict(
            elapsed_s=turn * 5.0,
            budget_status=BudgetStatus(used=0.01, limit=0.25),
            events=[],
            context={
                "turn": turn,
                "max_turns": 20,
                "los_state": los_state,
            },
        )

    def test_no_eject_without_deals(self):
        from loanville.agent_eject import AnalysisEject

        policy = AnalysisEject()
        result = policy.check(**self._make_kwargs(5, {"deals": []}))
        assert not result.should_eject

    def test_no_eject_early_stage(self):
        from loanville.agent_eject import AnalysisEject

        policy = AnalysisEject()
        result = policy.check(**self._make_kwargs(5, {
            "deals": [{"stage": "origination"}],
            "spread_count": 0,
            "doc_count": 0,
        }))
        assert not result.should_eject

    def test_eject_rubber_stamp_no_analysis(self):
        from loanville.agent_eject import AnalysisEject

        policy = AnalysisEject()
        result = policy.check(**self._make_kwargs(2, {
            "deals": [{"stage": "underwriting"}],
            "spread_count": 0,
            "doc_count": 0,
        }))
        assert result.should_eject
        assert "rubber-stamp" in result.reason

    def test_no_eject_with_spread(self):
        from loanville.agent_eject import AnalysisEject

        policy = AnalysisEject()
        result = policy.check(**self._make_kwargs(5, {
            "deals": [{"stage": "underwriting"}],
            "spread_count": 2,
            "doc_count": 3,
        }))
        assert not result.should_eject


class TestDefaultEjectPolicies:
    def test_returns_list(self):
        from loanville.agent_eject import default_eject_policies

        policies = default_eject_policies()
        assert isinstance(policies, list)
        assert len(policies) == 3

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

    def test_v02_exports_available(self):
        """v0.2.0 exports: CacheMetrics, CacheStats, PreflightError, RunResult."""
        from loanville.agent_budget import CacheMetrics, CacheStats, PreflightError, RunResult
        assert CacheMetrics is not None
        assert CacheStats is not None
        assert PreflightError is not None
        assert RunResult is not None


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

    def test_library_default_eject_policies_re_exported(self):
        from loanville.agent_eject import library_default_eject_policies
        from agent_preflight import CompositeEject

        policies = library_default_eject_policies()
        assert isinstance(policies, CompositeEject)


class TestLibraryDefaultEjectPolicies:
    """The library's default_eject_policies (budget+idle) work for Docker runs."""

    def test_budget_exceeded_ejects(self):
        from agent_preflight import default_eject_policies

        composite = default_eject_policies()
        result = composite.check(
            elapsed_s=10.0,
            budget_status=BudgetStatus(used=0.25, limit=0.25),
            events=[],
            context={},
        )
        assert result.should_eject
        assert "budget" in result.reason.lower()

    def test_idle_timeout_ejects(self):
        import time
        from agent_preflight import default_eject_policies

        composite = default_eject_policies(idle_timeout_s=0.01)
        # Simulate idle
        for p in composite.policies:
            if hasattr(p, "_last_event_ts"):
                p._last_event_ts = time.time() - 1

        result = composite.check(
            elapsed_s=100.0,
            budget_status=BudgetStatus(used=0.01, limit=0.25),
            events=[],
            context={},
        )
        assert result.should_eject
        assert "no events" in result.reason

    def test_no_eject_when_healthy(self):
        from agent_preflight import default_eject_policies

        composite = default_eject_policies()
        result = composite.check(
            elapsed_s=10.0,
            budget_status=BudgetStatus(used=0.01, limit=0.25),
            events=[{"type": "tool"}],
            context={},
        )
        assert not result.should_eject


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

    @patch("agent_preflight.preflight.get_account_balance", return_value=(0.01, 0.01, 0.00))
    def test_preflight_raises_preflight_error(self, mock_balance):
        from loanville.agent_budget import preflight_budget, PreflightError

        with pytest.raises(PreflightError) as exc_info:
            preflight_budget(
                admin_key="sk-admin-test",
                or_key="sk-test",
                models=["openai/gpt-4.1-nano"],
                budget_per_model=5.00,
                checks=["balance", "math"],
            )
        assert len(exc_info.value.failures) > 0


# ---------------------------------------------------------------------------
# CacheMetrics tests
# ---------------------------------------------------------------------------


class TestCacheMetrics:
    """CacheMetrics tracks prompt cache performance."""

    def test_empty_metrics(self):
        from loanville.agent_budget import CacheMetrics

        m = CacheMetrics()
        assert m.hit_rate == 0.0
        assert m.request_count == 0
        assert "no requests" in m.summary()

    def test_record_openrouter_format(self):
        from loanville.agent_budget import CacheMetrics

        m = CacheMetrics()
        m.record({
            "prompt_tokens": 2000,
            "completion_tokens": 100,
            "prompt_tokens_details": {
                "cached_tokens": 1800,
                "cache_write_tokens": 200,
            },
        })
        assert m.request_count == 1
        assert m.cached_tokens == 1800
        assert m.cache_creation_tokens == 200
        assert m.hit_rate == pytest.approx(0.90)
        assert m.cache_hit_requests == 1

    def test_record_anthropic_native_format(self):
        from loanville.agent_budget import CacheMetrics

        m = CacheMetrics()
        m.record({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200,
        })
        assert m.cached_tokens == 800
        assert m.cache_creation_tokens == 200

    def test_to_dict(self):
        from loanville.agent_budget import CacheMetrics

        m = CacheMetrics()
        m.record({"prompt_tokens": 100, "completion_tokens": 10})
        d = m.to_dict()
        assert "hit_rate" in d
        assert "request_count" in d
        assert d["request_count"] == 1
