"""
EV-4: Scoring engine.

Takes original and replay data, produces a ScoreCard with dimension scores:
- Correctness: did the agent reach the same result?
- Cost delta: percentage change in token spend
- Latency delta: percentage change in duration
- Tool match: did the agent use the same tools in the same order?
- Safety: placeholder for policy violation scoring (1.0 = clean)
"""

from __future__ import annotations

from typing import Any

from pkg.models.eval import EvalConfig, ScoreCard


class ScoringEngine:
    """Scores episode replays across multiple dimensions."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def score(
        self,
        original: dict[str, Any],
        baseline: dict[str, Any] | None = None,
    ) -> ScoreCard:
        """Score an episode against a baseline.

        If no baseline is provided, scores against itself (sanity check).
        """
        compare = baseline or original

        card = ScoreCard(
            episode_id=original.get("episode_id", ""),
            agent_id=original.get("agent_id", ""),
            baseline_episode_id=compare.get("episode_id") if baseline else None,
        )

        card.correctness = self._score_correctness(original, compare)
        card.cost_delta = self._score_cost_delta(original, compare)
        card.latency_delta = self._score_latency_delta(original, compare)
        card.tool_match = self._score_tool_match(original, compare)
        card.safety = self._score_safety(original)
        card.compute_weighted(self.config.score_weights)

        return card

    def _score_correctness(self, current: dict, baseline: dict) -> float:
        """Compare episode outcomes. 1.0 = same result, 0.0 = different."""
        current_status = current.get("status", "")
        baseline_status = baseline.get("status", "")

        if current_status == baseline_status:
            score = 1.0
        elif current_status == "success":
            score = 0.8  # improved over baseline
        else:
            score = 0.0  # regressed

        # Check step count similarity
        current_steps = len(current.get("steps", []))
        baseline_steps = len(baseline.get("steps", []))

        if current_steps > 0 and baseline_steps > 0:
            step_ratio = min(current_steps, baseline_steps) / max(current_steps, baseline_steps)
            score = score * 0.7 + step_ratio * 0.3

        return round(score, 4)


    def _score_cost_delta(self, current: dict, baseline: dict) -> float:
        """Percentage change in cost. Negative = cheaper (good)."""
        current_cost = current.get("total_cost_usd", 0.0)
        baseline_cost = baseline.get("total_cost_usd", 0.0)

        if baseline_cost == 0:
            return 0.0

        delta_pct = ((current_cost - baseline_cost) / baseline_cost) * 100
        return round(delta_pct, 2)

    def _score_latency_delta(self, current: dict, baseline: dict) -> float:
        """Percentage change in duration. Negative = faster (good)."""
        current_ms = current.get("total_duration_ms", 0)
        baseline_ms = baseline.get("total_duration_ms", 0)

        if baseline_ms == 0:
            return 0.0

        delta_pct = ((current_ms - baseline_ms) / baseline_ms) * 100
        return round(delta_pct, 2)

    def _score_tool_match(self, current: dict, baseline: dict) -> float:
        """How well do the tool sequences match? 1.0 = identical order."""
        current_tools = current.get("tools_used", [])
        baseline_tools = baseline.get("tools_used", [])

        if not baseline_tools and not current_tools:
            return 1.0
        if not baseline_tools or not current_tools:
            return 0.0

        # Check set overlap first
        current_set = set(current_tools)
        baseline_set = set(baseline_tools)
        overlap = len(current_set & baseline_set)
        union = len(current_set | baseline_set)
        set_score = overlap / union if union > 0 else 1.0

        # Check order match
        order_score = 0.0
        min_len = min(len(current_tools), len(baseline_tools))
        if min_len > 0:
            matches = sum(
                1 for i in range(min_len)
                if current_tools[i] == baseline_tools[i]
            )
            order_score = matches / min_len

        # Weighted: 60% set overlap, 40% order match
        return round(set_score * 0.6 + order_score * 0.4, 4)

    def _score_safety(self, episode: dict) -> float:
        """Safety score. 1.0 = no violations detected.

        Placeholder — will be replaced by policy engine integration.
        Currently checks for error steps and basic heuristics.
        """
        steps = episode.get("steps", [])
        if not steps:
            return 1.0

        error_steps = [s for s in steps if s.get("step_type") == "error"]
        error_ratio = len(error_steps) / len(steps)

        return round(1.0 - error_ratio, 4)
