"""
Tests for the loanville.rl package.

These tests verify the standalone RL components (no verifiers installation
required). They test the verifier protocol, rollout capture, metrics,
environment adapter, and Harbor task export.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from loanville.rl import (
    HAS_VERIFIERS,
    # Verifiers
    RewardResult,
    Rubric,
    CreditQualityVerifier,
    PortfolioVerifier,
    GateVerifier,
    CompoundVerifier,
    build_default_rubric,
    # Rollouts
    Step,
    Rollout,
    RolloutLogger,
    # Environment
    LoanvilleEnv,
    EnvConfig,
    # Metrics
    MeanMetric,
    MaxMetric,
    MinMetric,
    WeightedMetric,
    MetricSuite,
    # Tasks
    TaskConfig,
    export_harbor_task,
    export_harbor_dataset,
)


# ---------------------------------------------------------------------------
# Verifier tests
# ---------------------------------------------------------------------------

class TestRewardResult:
    def test_basic(self):
        r = RewardResult(reward=0.75, rewards={"a": 0.5, "b": 1.0})
        assert r.reward == 0.75
        assert r.rewards["a"] == 0.5

    def test_to_dict(self):
        r = RewardResult(reward=0.5)
        d = r.to_dict()
        assert d["reward"] == 0.5
        assert isinstance(d["rewards"], dict)

    def test_harbor_json(self):
        r = RewardResult(reward=0.8, rewards={"raroc": 0.1, "pnl": 5000})
        h = r.to_harbor_json()
        assert h["reward"] == 0.8
        assert h["raroc"] == 0.1


class TestCreditQualityVerifier:
    def test_good_portfolio(self):
        """A profitable portfolio should score above 0.5."""
        state = {
            "total_interest": 100_000,
            "total_losses": 10_000,
            "total_fees": 5_000,
            "total_deployed": 500_000,
            "available_capital": 1_000_000,
            "frauds_funded": 0,
            "defaults_count": 0,
            "deals_won": 5,
            "economics": {
                "funding_rate": 0.04,
                "fraud_penalty_rate": 0.25,
                "risk_free_rate": 0.05,
                "sim_horizon_months": 24,
            },
        }
        v = CreditQualityVerifier()
        result = v(state)
        assert result.reward > 0.5
        assert "raroc" in result.rewards
        assert "net_pnl" in result.rewards

    def test_terrible_portfolio(self):
        """A portfolio of all defaults should score near 0."""
        state = {
            "total_interest": 1_000,
            "total_losses": 400_000,
            "total_fees": 0,
            "total_deployed": 500_000,
            "available_capital": 1_000_000,
            "frauds_funded": 3,
            "defaults_count": 5,
            "deals_won": 5,
            "economics": {},
        }
        v = CreditQualityVerifier()
        result = v(state)
        assert result.reward < 0.3

    def test_empty_portfolio(self):
        """No lending should score around 0.5 (neutral)."""
        state = {
            "total_interest": 0,
            "total_losses": 0,
            "total_deployed": 0,
            "available_capital": 1_000_000,
            "deals_won": 0,
        }
        v = CreditQualityVerifier()
        result = v(state)
        assert 0.4 <= result.reward <= 0.6


class TestPortfolioVerifier:
    def test_balanced_portfolio(self):
        state = {
            "avg_utilization": 0.5,
            "concentration_violations": 0,
            "total_weeks": 10,
            "deals_won": 5,
            "deals_rejected": 3,
            "deals_errored": 0,
        }
        v = PortfolioVerifier()
        result = v(state)
        assert result.reward > 0.7

    def test_over_concentrated(self):
        state = {
            "avg_utilization": 0.5,
            "concentration_violations": 10,
            "total_weeks": 10,
            "deals_won": 5,
            "deals_rejected": 3,
            "deals_errored": 0,
        }
        v = PortfolioVerifier()
        result = v(state)
        # Concentration discipline = 0.0, utilization = 1.0, error = 1.0
        # Score = 0.4*1.0 + 0.4*0.0 + 0.2*1.0 = 0.6
        assert result.reward < 0.7
        assert result.rewards["concentration_discipline"] == 0.0


class TestGateVerifier:
    def test_all_pass(self):
        v = GateVerifier()
        result = v({"gates_passed": True})
        assert result.reward == 1.0

    def test_fail(self):
        v = GateVerifier()
        result = v({"gates_passed": False})
        assert result.reward == 0.0

    def test_detailed_gates(self):
        v = GateVerifier()
        result = v({
            "gate_results": [
                {"gate_id": "default_rate", "passed": True},
                {"gate_id": "min_roe", "passed": False},
            ]
        })
        assert result.reward == 0.0
        assert "min_roe" in result.info["failures"]


class TestRubric:
    def test_weighted_scoring(self):
        rubric = Rubric()
        rubric.add(CreditQualityVerifier(), weight=0.7, name="credit")
        rubric.add(GateVerifier(), weight=0.3, name="gates")

        state = {
            "total_interest": 50_000,
            "total_losses": 5_000,
            "total_deployed": 200_000,
            "available_capital": 500_000,
            "deals_won": 3,
            "gates_passed": True,
        }

        result = rubric.score(state)
        assert result.reward > 0
        assert "credit" in result.rewards
        assert "gates" in result.rewards

    def test_default_rubric(self):
        rubric = build_default_rubric()
        assert len(rubric.names) == 3
        assert "credit_quality" in rubric.names


class TestCompoundVerifier:
    def test_compound(self):
        cv = CompoundVerifier()
        cv.add("credit", CreditQualityVerifier())
        cv.add("gates", GateVerifier())

        state = {
            "total_interest": 50_000,
            "total_losses": 5_000,
            "total_deployed": 200_000,
            "available_capital": 500_000,
            "deals_won": 3,
            "gates_passed": True,
        }

        result = cv(state)
        assert "credit" in result.rewards
        assert "gates" in result.rewards
        assert result.reward > 0


# ---------------------------------------------------------------------------
# Rollout tests
# ---------------------------------------------------------------------------

class TestRollout:
    def test_basic_rollout(self):
        r = Rollout(episode_type="single", scenario="realistic")
        r.add_step(
            observation={"borrower": "test"},
            action={"decision": "APPROVE"},
            reward=0.5,
        )
        r.add_step(
            observation={"borrower": "test2"},
            action={"decision": "REJECT"},
            reward=0.0,
        )
        assert r.n_steps == 2
        assert r.cumulative_reward == 0.5
        assert not r.finalized

        r.finalize(terminal_reward=0.7, terminal_rewards={"credit": 0.7})
        assert r.finalized
        assert r.terminal_reward == 0.7

    def test_cannot_add_after_finalize(self):
        r = Rollout()
        r.finalize(terminal_reward=1.0)
        with pytest.raises(RuntimeError):
            r.add_step(observation={}, action={})

    def test_serialization(self):
        r = Rollout(episode_type="single", scenario="test", seed=42)
        r.add_step(
            observation={"x": 1},
            action={"y": 2},
            reward=0.3,
        )
        r.finalize(terminal_reward=0.6)

        # to_dict / from_dict roundtrip
        d = r.to_dict()
        r2 = Rollout.from_dict(d)
        assert r2.n_steps == 1
        assert r2.terminal_reward == 0.6
        assert r2.seed == 42

        # to_json
        j = r.to_json()
        data = json.loads(j)
        assert data["n_steps"] == 1

    def test_harbor_reward(self):
        r = Rollout()
        r.finalize(terminal_reward=0.8, terminal_rewards={"raroc": 0.1})
        h = r.to_harbor_reward()
        assert h["reward"] == 0.8
        assert h["raroc"] == 0.1

    def test_verifier_states(self):
        r = Rollout()
        r.add_step(
            observation={"capital": 1_000_000},
            action={"decision": "APPROVE"},
            reward=0.5,
        )
        states = r.to_verifier_states()
        assert len(states) == 1
        assert states[0]["capital"] == 1_000_000
        assert states[0]["step_idx"] == 0


class TestRolloutLogger:
    def test_log_and_load(self, tmp_path):
        logger = RolloutLogger(tmp_path / "rollouts")

        r = Rollout(scenario="test")
        r.add_step(observation={"a": 1}, action={"b": 2}, reward=0.5)
        r.finalize(terminal_reward=0.7)

        path = logger.log(r)
        assert path.exists()

        r2 = logger.load(r.rollout_id)
        assert r2 is not None
        assert r2.terminal_reward == 0.7

    def test_load_all(self, tmp_path):
        logger = RolloutLogger(tmp_path / "rollouts")

        for i in range(3):
            r = Rollout(scenario=f"test_{i}")
            r.finalize(terminal_reward=float(i) / 3)
            logger.log(r)

        all_rollouts = logger.load_all()
        assert len(all_rollouts) == 3


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mean(self):
        m = MeanMetric()
        result = m.compute([1.0, 2.0, 3.0])
        assert result["mean"] == 2.0
        assert result["count"] == 3.0

    def test_mean_with_none(self):
        m = MeanMetric()
        result = m.compute([1.0, None, 3.0])
        assert result["mean"] == 2.0
        assert result["count"] == 2.0

    def test_max(self):
        m = MaxMetric()
        assert m.compute([1.0, 3.0, 2.0])["max"] == 3.0

    def test_min(self):
        m = MinMetric()
        assert m.compute([1.0, 3.0, 2.0])["min"] == 1.0

    def test_weighted(self):
        m = WeightedMetric({"credit": 0.7, "portfolio": 0.3})
        result = m.compute_from_dicts([
            {"credit": 0.8, "portfolio": 0.6},
            {"credit": 0.9, "portfolio": 0.4},
        ])
        assert result["weighted_score"] > 0
        assert "credit_mean" in result
        assert "portfolio_mean" in result

    def test_metric_suite(self):
        suite = MetricSuite()
        suite.add("mean", MeanMetric())
        suite.add("max", MaxMetric())

        results = suite.compute([0.5, 0.8, 0.3])
        assert "mean" in results
        assert "max" in results
        assert results["mean"]["mean"] == pytest.approx(0.533, abs=0.01)


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestLoanvilleEnv:
    def test_reset_and_step(self):
        config = EnvConfig(n_borrowers=3, season_mix="easy", seed=42)
        env = LoanvilleEnv(config)
        obs, info = env.reset()

        assert "borrower_id" in obs or "company_name" in obs
        assert info["n_borrowers"] == 3

        # Step through all borrowers
        for i in range(3):
            action = {"decision": "REJECT"}
            obs, reward, done, truncated, info = env.step(action)

        assert done
        assert "terminal_reward" in info
        assert "rollout" in info
        assert info["rollout"].finalized

    def test_approve_flow(self):
        config = EnvConfig(n_borrowers=2, season_mix="easy", seed=42)
        env = LoanvilleEnv(config)
        obs, info = env.reset()

        # Approve first
        action = {
            "decision": "APPROVE",
            "term_sheet": {
                "loan_amount": 100_000,
                "interest_rate": 10.0,
                "term_months": 24,
            },
        }
        obs, reward, done, truncated, info = env.step(action)
        assert not done

        # Reject second
        obs, reward, done, truncated, info = env.step({"decision": "REJECT"})
        assert done
        assert info["deals_won"] == 1
        assert info["deals_rejected"] == 1

    def test_rollout_capture(self):
        config = EnvConfig(n_borrowers=2, season_mix="easy", seed=42)
        env = LoanvilleEnv(config)
        env.reset()

        env.step({"decision": "REJECT"})
        _, _, done, _, info = env.step({"decision": "REJECT"})

        assert done
        rollout = info["rollout"]
        assert rollout.n_steps == 2
        assert rollout.finalized

    def test_deterministic_seeding(self):
        """Same seed should produce same borrowers."""
        config1 = EnvConfig(n_borrowers=3, season_mix="easy", seed=123)
        config2 = EnvConfig(n_borrowers=3, season_mix="easy", seed=123)

        env1 = LoanvilleEnv(config1)
        env2 = LoanvilleEnv(config2)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        assert obs1["company_name"] == obs2["company_name"]


# ---------------------------------------------------------------------------
# Harbor task export tests
# ---------------------------------------------------------------------------

class TestHarborTasks:
    def test_export_single_task(self, tmp_path):
        task_dir = tmp_path / "test_task"
        config = TaskConfig(
            task_name="test_task",
            n_borrowers=3,
            season_mix="easy",
            seed=42,
        )
        path = export_harbor_task(task_dir, config)

        assert (path / "instruction.md").exists()
        assert (path / "task.toml").exists()
        assert (path / "solution" / "solve.sh").exists()
        assert (path / "tests" / "test.sh").exists()

        # Verify instruction content
        instruction = (path / "instruction.md").read_text()
        assert "Loanville" in instruction
        assert "APPROVE" in instruction

        # Verify task.toml content
        toml = (path / "task.toml").read_text()
        assert "test_task" in toml

        # Verify test.sh is executable
        test_sh = path / "tests" / "test.sh"
        assert os.access(test_sh, os.X_OK)

    def test_export_dataset(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        path = export_harbor_dataset(dataset_dir, n_tasks=3)

        assert path.exists()
        tasks = list(path.iterdir())
        assert len(tasks) == 3

        # Each task should have the right structure
        for task in tasks:
            assert (task / "instruction.md").exists()
            assert (task / "task.toml").exists()


# ---------------------------------------------------------------------------
# Integration: engine.to_rollouts()
# ---------------------------------------------------------------------------

class TestEngineRollouts:
    def test_mock_engine_rollouts(self):
        """Test that SimulationEngine.to_rollouts() works after a mock run."""
        from loanville.data import get_borrowers, get_lenders
        from loanville.engine import SimulationEngine
        import asyncio

        borrowers = get_borrowers("easy")[:3]
        lenders = get_lenders()[:1]

        engine = SimulationEngine(
            borrowers, lenders, mock=True, data_mode="full",
        )
        asyncio.run(engine.run())

        rollouts = engine.to_rollouts()
        assert len(rollouts) == 1  # one per lender
        assert rollouts[0].finalized
        assert rollouts[0].n_steps == 3
        assert rollouts[0].terminal_reward > 0
        assert "credit_quality" in rollouts[0].terminal_rewards
