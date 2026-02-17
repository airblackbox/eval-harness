"""
EV-1: Eval schema and scoring models.

Defines the data structures for eval runs, score cards, and regression alerts.
An eval run replays one or more episodes and produces a score card for each.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EvalStatus(str, Enum):
    """Status of an eval run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScoreDimension(str, Enum):
    """Scoring dimensions for episode evaluation."""
    CORRECTNESS = "correctness"
    COST_DELTA = "cost_delta"
    LATENCY_DELTA = "latency_delta"
    TOOL_MATCH = "tool_match"
    SAFETY = "safety"


class RegressionSeverity(str, Enum):
    """How bad is the regression."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EvalConfig(BaseModel):
    """Configuration for an eval run."""
    name: str = "default"
    episode_store_url: str = "http://localhost:8100"
    agent_id: str | None = None
    baseline_tag: str | None = None
    model_filter: str | None = None
    provider_filter: str | None = None
    tool_filter: str | None = None
    max_episodes: int = 100
    score_weights: dict[str, float] = Field(default_factory=lambda: {
        "correctness": 0.4,
        "cost_delta": 0.2,
        "latency_delta": 0.1,
        "tool_match": 0.2,
        "safety": 0.1,
    })
    cost_threshold_pct: float = 20.0  # alert if cost increases by more than 20%
    latency_threshold_pct: float = 30.0  # alert if latency increases by more than 30%
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreCard(BaseModel):
    """Score card for a single episode evaluation."""
    episode_id: str
    agent_id: str
    baseline_episode_id: str | None = None
    correctness: float = 0.0  # 0.0 to 1.0 — did it reach the same result?
    cost_delta: float = 0.0  # percentage change in cost vs baseline
    latency_delta: float = 0.0  # percentage change in latency vs baseline
    tool_match: float = 0.0  # 0.0 to 1.0 — same tools in same order?
    safety: float = 1.0  # 0.0 to 1.0 — policy violations found?
    weighted_score: float = 0.0  # overall weighted score
    details: dict[str, Any] = Field(default_factory=dict)

    def compute_weighted(self, weights: dict[str, float]) -> float:
        """Compute weighted overall score from dimension scores."""
        total = 0.0
        for dim, weight in weights.items():
            value = getattr(self, dim, 0.0)
            # For deltas, convert to a 0-1 score (lower delta = better)
            if dim in ("cost_delta", "latency_delta"):
                # 0% delta = 1.0, 100% delta = 0.0, negative delta (improvement) = 1.0
                value = max(0.0, 1.0 - abs(value) / 100.0)
            total += value * weight
        self.weighted_score = round(total, 4)
        return self.weighted_score


class RegressionAlert(BaseModel):
    """Alert generated when a regression is detected."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    eval_run_id: str
    episode_id: str
    dimension: ScoreDimension
    severity: RegressionSeverity
    message: str
    baseline_value: float = 0.0
    current_value: float = 0.0
    delta_pct: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalResult(BaseModel):
    """Result for a single episode in an eval run."""
    episode_id: str
    agent_id: str
    status: EvalStatus = EvalStatus.PENDING
    score_card: ScoreCard | None = None
    alerts: list[RegressionAlert] = Field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0


class EvalRun(BaseModel):
    """A complete eval run across multiple episodes."""
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    config: EvalConfig = Field(default_factory=EvalConfig)
    status: EvalStatus = EvalStatus.PENDING
    results: list[EvalResult] = Field(default_factory=list)
    alerts: list[RegressionAlert] = Field(default_factory=list)
    total_episodes: int = 0
    passed: int = 0
    failed: int = 0
    avg_weighted_score: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_summary(self) -> None:
        """Compute summary stats from individual results."""
        completed = [r for r in self.results if r.score_card is not None]
        self.total_episodes = len(self.results)
        self.passed = len([r for r in completed if r.score_card.weighted_score >= 0.7])
        self.failed = len([r for r in completed if r.score_card.weighted_score < 0.7])
        if completed:
            self.avg_weighted_score = round(
                sum(r.score_card.weighted_score for r in completed) / len(completed), 4
            )
        # Collect all alerts from results
        self.alerts = []
        for r in self.results:
            self.alerts.extend(r.alerts)
