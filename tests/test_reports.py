"""Tests for EV-7: Report generator."""

from __future__ import annotations

import json
import tempfile

import pytest

from pkg.models.eval import EvalResult, EvalRun, EvalStatus, ScoreCard
from pkg.reports.generator import ReportGenerator


class TestReportGenerator:
    def test_generate_json(self):
        """JSON report should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            run = _make_eval_run()
            path = gen.generate_json(run)

            with open(path) as f:
                data = json.load(f)
            assert data["run_id"] == run.run_id
            assert len(data["results"]) == 2

    def test_generate_markdown(self):
        """Markdown report should contain key sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            run = _make_eval_run()
            path = gen.generate_markdown(run)

            with open(path) as f:
                content = f.read()
            assert "# Eval Report" in content
            assert "Summary" in content
            assert "Episode Scores" in content

    def test_summary_dict(self):
        """Summary dict should have correct structure."""
        gen = ReportGenerator()
        run = _make_eval_run()
        summary = gen.generate_summary_dict(run)
        assert summary["total_episodes"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert "pass_rate" in summary


def _make_eval_run() -> EvalRun:
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
    return run