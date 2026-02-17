"""Tests for EV-1: Eval schema and scoring models."""

from __future__ import annotations

import pytest

from pkg.models.eval import (
    EvalConfig,
    EvalResult,
    EvalRun,
    EvalStatus,
    RegressionAlert,
    RegressionSeverity,
    ScoreCard,
    ScoreDimension,
)


class TestEvalConfig:
    def test_defaults(self):
        config = EvalConfig()
        assert config.name == "default"
        assert config.episode_store_url == "http://localhost:8100"
        assert config.max_episodes == 100
        assert "correctness" in config.score_weights

    def test_custom_weights(self):
        config = EvalConfig(score_weights={"correctness": 1.0})
        assert config.score_weights["correctness"] == 1.0


class TestScoreCard:
    def test_compute_weighted(self):
        card = ScoreCard(
            episode_id="ep-1",
            agent_id="agent-1",
            correctness=1.0,
            cost_delta=0.0,
            latency_delta=0.0,
            tool_match=1.0,
            safety=1.0,
        )
        weights = EvalConfig().score_weights
        score = card.compute_weighted(weights)
        assert score > 0.9  # all perfect scores
    def test_zero_scores(self):
        card = ScoreCard(episode_id="ep-1", agent_id="a")
        card.compute_weighted(EvalConfig().score_weights)
        # safety defaults to 1.0, so not fully zero
        assert card.weighted_score >= 0.0

    def test_high_cost_delta_lowers_score(self):
        card = ScoreCard(
            episode_id="ep-1", agent_id="a",
            correctness=1.0, cost_delta=80.0,
            latency_delta=0.0, tool_match=1.0, safety=1.0,
        )
        score = card.compute_weighted(EvalConfig().score_weights)
        # 80% cost increase should lower the cost_delta dimension score
        assert score < 0.95


class TestEvalRun:
    def test_compute_summary(self):
        run = EvalRun()
        run.results = [
            EvalResult(
                episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
                score_card=ScoreCard(episode_id="ep-1", agent_id="a", weighted_score=0.9),
            ),
            EvalResult(
                episode_id="ep-2", agent_id="a", status=EvalStatus.COMPLETED,
                score_card=ScoreCard(episode_id="ep-2", agent_id="a", weighted_score=0.3),
            ),
        ]
        run.compute_summary()
        assert run.total_episodes == 2
        assert run.passed == 1  # >= 0.7
        assert run.failed == 1  # < 0.7
        assert 0.5 < run.avg_weighted_score < 0.7


class TestRegressionAlert:
    def test_defaults(self):
        alert = RegressionAlert(
            eval_run_id="run-1",
            episode_id="ep-1",
            dimension=ScoreDimension.COST_DELTA,
            severity=RegressionSeverity.WARNING,
            message="Cost went up",
        )
        assert alert.alert_id is not None
        assert alert.created_at is not None