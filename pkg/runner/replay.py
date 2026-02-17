"""
EV-3: Replay runner.

Takes an episode from the store, replays it through the gateway,
and captures the new episode for comparison. If no gateway is available,
falls back to "dry replay" — just fetches the episode and its replay view.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from pkg.client.episode_client import EpisodeClient


class ReplayRunner:
    """Replays episodes through the AIR Blackbox Gateway.

    In full mode, sends each step's input back through the gateway
    and captures a new episode. In dry mode, just fetches the replay
    view from the episode store (no actual LLM calls).
    """

    def __init__(
        self,
        episode_client: EpisodeClient,
        gateway_url: str | None = None,
    ) -> None:
        self.client = episode_client
        self.gateway_url = gateway_url  # e.g. http://localhost:8080

    async def dry_replay(self, episode_id: str) -> dict:
        """Fetch the replay view without making real LLM calls.

        Returns the episode replay data from the store. Used when
        no gateway is configured or for cost-free regression checks.
        """
        return await self.client.get_replay(episode_id)

    async def replay_episode(self, episode_id: str) -> dict[str, Any]:
        """Replay an episode and return comparison data.

        If gateway_url is set, sends LLM steps through the gateway
        to get fresh responses. Otherwise falls back to dry_replay.
        Returns a dict with original episode, replay data, and timing.
        """
        start = time.time()

        original = await self.client.get_episode(episode_id)
        replay = await self.dry_replay(episode_id)

        result: dict[str, Any] = {
            "episode_id": episode_id,
            "original": original,
            "replay": replay,
            "replayed_steps": [],
            "duration_ms": 0,
            "mode": "dry",
        }

        if self.gateway_url:
            result["mode"] = "live"
            replayed = await self._live_replay(original)
            result["replayed_steps"] = replayed

        result["duration_ms"] = int((time.time() - start) * 1000)
        return result


    async def _live_replay(self, episode: dict) -> list[dict]:
        """Send LLM steps through the gateway for live replay."""
        replayed: list[dict] = []
        steps = episode.get("steps", [])

        async with httpx.AsyncClient(timeout=60.0) as http:
            for step in steps:
                if step.get("step_type") != "llm_call":
                    replayed.append({"step_index": step["step_index"], "skipped": True})
                    continue

                model = step.get("model", "gpt-4")
                input_text = step.get("input_summary", "")

                try:
                    resp = await http.post(
                        f"{self.gateway_url}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": input_text}],
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    replayed.append({
                        "step_index": step["step_index"],
                        "model": model,
                        "response": data,
                        "tokens": data.get("usage", {}).get("total_tokens", 0),
                    })
                except httpx.HTTPError as e:
                    replayed.append({
                        "step_index": step["step_index"],
                        "error": str(e),
                    })

        return replayed
