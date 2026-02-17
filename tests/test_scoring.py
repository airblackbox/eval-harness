"""Tests for EV-4: Scoring engine."""

from __future__ import annotations

import pytest

from pkg.models.eval import EvalConfig
from pkg.scoring.engine import ScoringEngine


class TestScoringEngine:
    def test_identical_episodes(self, sample_episode, sample_baseline):
        """Scoring identical episodes should produce near-perfect scores."""
        scorer = ScoringEngine(EvalConfig())
        card = scorer.score(sample_episode, sample_baseline)
        assert card.correctness >= 0.9
        assert card.cost_delta == 0.0
        assert card.latency_delta == 0.0
        assert card.tool_match >= 0.9
        assert card.safety >= 0.9
        assert card.weighted_score > 0.8

    def test_self_score(self, sample_episode):
        """Scoring against self (no baseline) should work."""
        scorer = ScoringEngine(EvalConfig())
        card = scorer.score(sample_episode)
        assert card.weighted_score > 0.8
        assert card.baseline_episode_id is None

    def test_cost_increase_detected(self, sample_episode, sample_baseline):
        """Higher cost should produce positive cost_delta."""
        expensive = {**sample_episode, "total_cost_usd": 0.012}
        scorer = ScoringEngine(EvalConfig())
        card = scorer.score(expensive, sample_baseline)
        assert card.cost_delta > 0  # cost went up

    def test_latency_increase_detected(self, sample_episode, sample_baseline):
        """Higher latency should produce positive latency_delta."""
        slow = {**sample_episode, "total_duration_ms": 1200}
        scorer = ScoringEngine(EvalConfig())
        card = scorer.score(slow, sample_baseline)
        assert card.latency_delta > 0  # latency went up
    def test_different_tools_lower_match(self, sample_episode, sample_baseline):
        """Different tools should lower tool_match score."""
        different = {**sample_episode, "tools_used": ["web_search", "file_read"]}
        scorer = ScoringEngine(EvalConfig())
        card = scorer.score(different, sample_baseline)
        assert card.tool_match < 0.8  # tools changed

    def test_failure_lowers_correctness(self, sample_episode, sample_baseline):
        """Failed episode should have lower correctness."""
        failed = {**sample_episode, "status": "failure"}
        scorer = ScoringEngine(EvalConfig())
        card = scorer.score(failed, sample_baseline)
        assert card.correctness < 0.5

    def test_error_steps_lower_safety(self, sample_episode):
        """Error steps should lower safety score."""
        errored = {**sample_episode}
        errored["steps"] = [
            {"step_index": 0, "step_type": "error"},
            {"step_index": 1, "step_type": "error"},
        ]
        scorer = ScoringEngine(EvalConfig())
        card = scorer.score(errored)
        assert card.safety == 0.0  # all error steps