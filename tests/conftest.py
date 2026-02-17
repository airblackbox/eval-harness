"""Shared test fixtures for eval-harness tests."""

from __future__ import annotations

import pytest

from pkg.models.eval import EvalConfig, EvalResult, EvalRun, EvalStatus, ScoreCard


@pytest.fixture
def config():
    """Default eval config for tests."""
    return EvalConfig(
        name="test-eval",
        episode_store_url="http://localhost:8100",
        cost_threshold_pct=20.0,
        latency_threshold_pct=30.0,
    )


@pytest.fixture
def sample_episode():
    """A sample episode dict as returned by the store API."""
    return {
        "episode_id": "ep-test-001",
        "agent_id": "test-agent",
        "status": "success",
        "steps": [
            {
                "step_index": 0,
                "step_type": "llm_call",
                "model": "gpt-4",
                "provider": "openai",
                "input_summary": "What is 2+2?",
                "output_summary": "4",
                "tokens": 100,
                "cost_usd": 0.003,
                "duration_ms": 500,
                "metadata": {},
            },
            {
                "step_index": 1,
                "step_type": "tool_call",
                "tool_name": "calculator",
                "tokens": 50,
                "cost_usd": 0.001,
                "duration_ms": 100,
                "metadata": {},
            },
        ],
        "tools_used": ["calculator"],
        "total_tokens": 150,
        "total_cost_usd": 0.004,
        "total_duration_ms": 600,
        "step_count": 2,
        "started_at": "2026-02-16T00:00:00Z",
        "ended_at": "2026-02-16T00:00:01Z",
        "metadata": {},
    }

@pytest.fixture
def sample_baseline():
    """A baseline episode for comparison."""
    return {
        "episode_id": "ep-baseline-001",
        "agent_id": "test-agent",
        "status": "success",
        "steps": [
            {
                "step_index": 0,
                "step_type": "llm_call",
                "model": "gpt-4",
                "provider": "openai",
                "input_summary": "What is 2+2?",
                "output_summary": "4",
                "tokens": 100,
                "cost_usd": 0.003,
                "duration_ms": 500,
                "metadata": {},
            },
            {
                "step_index": 1,
                "step_type": "tool_call",
                "tool_name": "calculator",
                "tokens": 50,
                "cost_usd": 0.001,
                "duration_ms": 100,
                "metadata": {},
            },
        ],
        "tools_used": ["calculator"],
        "total_tokens": 150,
        "total_cost_usd": 0.004,
        "total_duration_ms": 600,
        "step_count": 2,
        "started_at": "2026-02-15T00:00:00Z",
        "ended_at": "2026-02-15T00:00:01Z",
        "metadata": {},
    }