"""Tests for EV-5: Regression detector."""

from __future__ import annotations

import pytest

from pkg.models.eval import (
    EvalConfig,
    EvalResult,
    EvalStatus,
    RegressionSeverity,
    ScoreCard,
    ScoreDimension,
)
from pkg.regression.detector import RegressionDetector


class TestRegressionDetector:
    def test_no_regression_on_good_scores(self, config):
        """Good scores should produce no alerts."""
        detector = RegressionDetector(config)
        result = EvalResult(
            episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
            score_card=ScoreCard(
                episode_id="ep-1", agent_id="a",
                correctness=0.9, cost_delta=5.0, latency_delta=10.0,
                tool_match=0.9, safety=1.0, weighted_score=0.85,
            ),
        )
        alerts = detector.check(result)
        assert len(alerts) == 0

    def test_cost_regression_warning(self, config):
        """Cost exceeding threshold should trigger warning."""
        detector = RegressionDetector(config)
        result = EvalResult(
            episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
            score_card=ScoreCard(
                episode_id="ep-1", agent_id="a",
                cost_delta=25.0,  # above 20% threshold
                correctness=0.9, tool_match=0.9, safety=1.0, weighted_score=0.8,
            ),
        )
        alerts = detector.check(result)
        cost_alerts = [a for a in alerts if a.dimension == ScoreDimension.COST_DELTA]
        assert len(cost_alerts) == 1
        assert cost_alerts[0].severity == RegressionSeverity.WARNING
    def test_cost_regression_critical(self, config):
        """Cost exceeding 2x threshold should trigger critical."""
        detector = RegressionDetector(config)
        result = EvalResult(
            episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
            score_card=ScoreCard(
                episode_id="ep-1", agent_id="a",
                cost_delta=50.0,  # above 40% (2x threshold)
                correctness=0.9, tool_match=0.9, safety=1.0, weighted_score=0.8,
            ),
        )
        alerts = detector.check(result)
        cost_alerts = [a for a in alerts if a.dimension == ScoreDimension.COST_DELTA]
        assert len(cost_alerts) == 1
        assert cost_alerts[0].severity == RegressionSeverity.CRITICAL

    def test_latency_regression(self, config):
        """Latency exceeding threshold should trigger alert."""
        detector = RegressionDetector(config)
        result = EvalResult(
            episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
            score_card=ScoreCard(
                episode_id="ep-1", agent_id="a",
                latency_delta=35.0,  # above 30%
                correctness=0.9, cost_delta=0.0, tool_match=0.9, safety=1.0, weighted_score=0.8,
            ),
        )
        alerts = detector.check(result)
        lat_alerts = [a for a in alerts if a.dimension == ScoreDimension.LATENCY_DELTA]
        assert len(lat_alerts) == 1
    def test_correctness_regression(self, config):
        """Low correctness should trigger critical alert."""
        detector = RegressionDetector(config)
        result = EvalResult(
            episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
            score_card=ScoreCard(
                episode_id="ep-1", agent_id="a",
                correctness=0.3,
                cost_delta=0.0, latency_delta=0.0, tool_match=0.9, safety=1.0, weighted_score=0.6,
            ),
        )
        alerts = detector.check(result)
        corr_alerts = [a for a in alerts if a.dimension == ScoreDimension.CORRECTNESS]
        assert len(corr_alerts) >= 1
        assert any(a.severity == RegressionSeverity.CRITICAL for a in corr_alerts)

    def test_safety_regression(self, config):
        """Low safety should trigger critical alert."""
        detector = RegressionDetector(config)
        result = EvalResult(
            episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
            score_card=ScoreCard(
                episode_id="ep-1", agent_id="a",
                safety=0.5,
                correctness=0.9, cost_delta=0.0, latency_delta=0.0, tool_match=0.9, weighted_score=0.7,
            ),
        )
        alerts = detector.check(result)
        safety_alerts = [a for a in alerts if a.dimension == ScoreDimension.SAFETY]
        assert len(safety_alerts) == 1
        assert safety_alerts[0].severity == RegressionSeverity.CRITICAL
    def test_batch_check(self, config):
        """Batch check should tag all alerts with run_id."""
        detector = RegressionDetector(config)
        results = [
            EvalResult(
                episode_id="ep-1", agent_id="a", status=EvalStatus.COMPLETED,
                score_card=ScoreCard(
                    episode_id="ep-1", agent_id="a",
                    cost_delta=50.0, correctness=0.9, tool_match=0.9, safety=1.0, weighted_score=0.8,
                ),
            ),
        ]
        alerts = detector.check_batch(results, "run-123")
        assert all(a.eval_run_id == "run-123" for a in alerts)

    def test_no_scorecard_no_alerts(self, config):
        """Result without scorecard should produce no alerts."""
        detector = RegressionDetector(config)
        result = EvalResult(episode_id="ep-1", agent_id="a", status=EvalStatus.FAILED)
        alerts = detector.check(result)
        assert len(alerts) == 0