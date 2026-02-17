"""
EV-6: CLI runner.

Click-based command-line interface for running evals, viewing reports,
and checking regression status.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.table import Table

from pkg.client.episode_client import EpisodeClient
from pkg.models.eval import EvalConfig, EvalResult, EvalRun, EvalStatus
from pkg.regression.detector import RegressionDetector
from pkg.runner.replay import ReplayRunner
from pkg.scoring.engine import ScoringEngine

console = Console()


@click.group()
def cli():
    """eval-harness — Replay episodes, score results, detect regressions."""
    pass


@cli.command()
@click.option("--store-url", default="http://localhost:8100", help="Episode store URL")
@click.option("--agent-id", default=None, help="Filter by agent ID")
@click.option("--model", default=None, help="Filter by model name")
@click.option("--limit", default=20, help="Max episodes to evaluate")
@click.option("--gateway-url", default=None, help="Gateway URL for live replay")
@click.option("--output", default=None, help="Output file for results JSON")
def run(store_url, agent_id, model, limit, gateway_url, output):
    """Run an eval across episodes from the store."""
    config = EvalConfig(
        episode_store_url=store_url,
        agent_id=agent_id,
        model_filter=model,
        max_episodes=limit,
    )
    asyncio.run(_run_eval(config, gateway_url, output))

async def _run_eval(config: EvalConfig, gateway_url: str | None, output: str | None):
    """Core eval execution logic."""
    client = EpisodeClient(config.episode_store_url)
    runner = ReplayRunner(client, gateway_url)
    scorer = ScoringEngine(config)
    detector = RegressionDetector(config)

    eval_run = EvalRun(config=config, status=EvalStatus.RUNNING)
    console.print(f"[bold blue]Starting eval run {eval_run.run_id[:8]}...[/]")

    # Fetch episodes
    try:
        episodes = await client.list_episodes(
            agent_id=config.agent_id,
            model=config.model_filter,
            limit=config.max_episodes,
        )
    except Exception as e:
        console.print(f"[red]Failed to fetch episodes: {e}[/]")
        sys.exit(1)

    console.print(f"Found {len(episodes)} episodes to evaluate")

    # Score each episode
    for i, ep_summary in enumerate(episodes):
        ep_id = ep_summary.get("episode_id", "")
        console.print(f"  [{i+1}/{len(episodes)}] Evaluating {ep_id[:8]}...", end=" ")

        try:
            episode = await client.get_episode(ep_id)
            card = scorer.score(episode)
            result = EvalResult(
                episode_id=ep_id,
                agent_id=episode.get("agent_id", ""),
                status=EvalStatus.COMPLETED,
                score_card=card,
            )
            console.print(f"score={card.weighted_score:.2f}")
        except Exception as e:
            result = EvalResult(
                episode_id=ep_id,
                agent_id=ep_summary.get("agent_id", ""),
                status=EvalStatus.FAILED,
                error=str(e),
            )
            console.print(f"[red]ERROR: {e}[/]")

        eval_run.results.append(result)
    # Detect regressions
    alerts = detector.check_batch(eval_run.results, eval_run.run_id)
    eval_run.status = EvalStatus.COMPLETED
    eval_run.ended_at = datetime.now(timezone.utc)
    eval_run.compute_summary()

    # Print summary
    _print_summary(eval_run)

    # Save output
    if output:
        with open(output, "w") as f:
            json.dump(eval_run.model_dump(mode="json"), f, indent=2, default=str)
        console.print(f"\n[green]Results saved to {output}[/]")


def _print_summary(eval_run: EvalRun):
    """Print a rich summary table."""
    console.print(f"\n[bold]Eval Run: {eval_run.run_id[:8]}[/]")
    console.print(f"Status: {eval_run.status.value}")
    console.print(f"Episodes: {eval_run.total_episodes} (passed: {eval_run.passed}, failed: {eval_run.failed})")
    console.print(f"Avg Score: {eval_run.avg_weighted_score:.2f}")

    if eval_run.alerts:
        console.print(f"\n[bold red]Alerts ({len(eval_run.alerts)}):[/]")
        table = Table()
        table.add_column("Severity")
        table.add_column("Dimension")
        table.add_column("Episode")
        table.add_column("Message")

        for alert in eval_run.alerts:
            color = {"critical": "red", "warning": "yellow", "info": "blue"}.get(
                alert.severity.value, "white"
            )
            table.add_row(
                f"[{color}]{alert.severity.value}[/]",
                alert.dimension.value,
                alert.episode_id[:8],
                alert.message,
            )
        console.print(table)
    else:
        console.print("\n[green]No regressions detected.[/]")

@cli.command()
@click.option("--store-url", default="http://localhost:8100", help="Episode store URL")
def status(store_url):
    """Check episode store connection."""
    async def _check():
        client = EpisodeClient(store_url)
        try:
            health = await client.health()
            console.print(f"[green]Connected to episode store[/]")
            console.print(f"  Version: {health.get('version', 'unknown')}")
            episodes = await client.list_episodes(limit=1)
            total = len(episodes)
            console.print(f"  Episodes available: {total}+")
        except Exception as e:
            console.print(f"[red]Cannot connect to episode store: {e}[/]")
            sys.exit(1)

    asyncio.run(_check())


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
def report(results_file):
    """Generate a report from saved eval results."""
    from pkg.reports.generator import ReportGenerator

    with open(results_file) as f:
        data = json.load(f)

    eval_run = EvalRun(**data)
    generator = ReportGenerator()
    report_path = generator.generate_json(eval_run)
    console.print(f"[green]Report saved to {report_path}[/]")


if __name__ == "__main__":
    cli()