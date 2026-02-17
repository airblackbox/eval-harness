"""
EV-8: Nightly scheduler integration.

Provides a scheduling wrapper that can run eval sweeps on a cron schedule.
Designed to integrate with system cron, GitHub Actions, or a simple
Python-based loop.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pkg.client.episode_client import EpisodeClient
from pkg.models.eval import EvalConfig, EvalResult, EvalRun, EvalStatus
from pkg.regression.detector import RegressionDetector
from pkg.reports.generator import ReportGenerator
from pkg.runner.replay import ReplayRunner
from pkg.scoring.engine import ScoringEngine


class NightlyRunner:
    """Runs a complete eval sweep and generates reports.

    Designed to be called from cron or a scheduled task:
        python -m pkg.scheduler.nightly --config nightly.json
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.client = EpisodeClient(config.episode_store_url)
        self.runner = ReplayRunner(self.client)
        self.scorer = ScoringEngine(config)
        self.detector = RegressionDetector(config)
        self.reporter = ReportGenerator()

    async def run_sweep(self) -> EvalRun:
        """Execute a full nightly eval sweep.

        1. Fetch recent episodes from the store
        2. Score each episode
        3. Detect regressions
        4. Generate reports
        5. Return the eval run with all results
        """
        eval_run = EvalRun(
            config=self.config,
            status=EvalStatus.RUNNING,
            metadata={"trigger": "nightly", "timestamp": datetime.now(timezone.utc).isoformat()},
        )
        # Fetch episodes
        episodes = await self.client.list_episodes(
            agent_id=self.config.agent_id,
            model=self.config.model_filter,
            limit=self.config.max_episodes,
        )

        # Score each episode
        for ep_summary in episodes:
            ep_id = ep_summary.get("episode_id", "")
            try:
                episode = await self.client.get_episode(ep_id)
                card = self.scorer.score(episode)
                result = EvalResult(
                    episode_id=ep_id,
                    agent_id=episode.get("agent_id", ""),
                    status=EvalStatus.COMPLETED,
                    score_card=card,
                )
            except Exception as e:
                result = EvalResult(
                    episode_id=ep_id,
                    agent_id=ep_summary.get("agent_id", ""),
                    status=EvalStatus.FAILED,
                    error=str(e),
                )
            eval_run.results.append(result)

        # Detect regressions
        self.detector.check_batch(eval_run.results, eval_run.run_id)

        # Finalize
        eval_run.status = EvalStatus.COMPLETED
        eval_run.ended_at = datetime.now(timezone.utc)
        eval_run.compute_summary()

        # Generate reports
        self.reporter.generate_json(eval_run)
        self.reporter.generate_markdown(eval_run)

        return eval_run
    @staticmethod
    def load_config(config_path: str) -> EvalConfig:
        """Load eval config from a JSON file."""
        data = json.loads(Path(config_path).read_text())
        return EvalConfig(**data)


async def main():
    """Entry point for nightly runs."""
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if config_path:
        config = NightlyRunner.load_config(config_path)
    else:
        config = EvalConfig()

    runner = NightlyRunner(config)
    eval_run = await runner.run_sweep()

    print(f"Nightly eval complete: {eval_run.run_id[:8]}")
    print(f"  Episodes: {eval_run.total_episodes}")
    print(f"  Passed: {eval_run.passed}")
    print(f"  Failed: {eval_run.failed}")
    print(f"  Avg Score: {eval_run.avg_weighted_score:.2f}")
    print(f"  Alerts: {len(eval_run.alerts)}")

    if eval_run.alerts:
        critical = [a for a in eval_run.alerts if a.severity.value == "critical"]
        if critical:
            print(f"\n  ⚠️  {len(critical)} CRITICAL regressions detected!")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())