"""
EV-2: Episode fetcher client.

Async HTTP client that pulls episodes from the agent-episode-store.
Used by the eval runner to fetch episodes for replay and scoring.
"""

from __future__ import annotations

from typing import Any

import httpx


class EpisodeClient:
    """HTTP client for the agent-episode-store API."""

    def __init__(self, base_url: str = "http://localhost:8100") -> None:
        self.base_url = base_url.rstrip("/")

    async def health(self) -> dict:
        """Check episode store health."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/v1/health")
            resp.raise_for_status()
            return resp.json()

    async def list_episodes(
        self,
        agent_id: str | None = None,
        status: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        tool: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List episodes with optional filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status
        if model:
            params["model"] = model
        if provider:
            params["provider"] = provider
        if tool:
            params["tool"] = tool

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/v1/episodes", params=params)
            resp.raise_for_status()
            return resp.json()


    async def get_episode(self, episode_id: str) -> dict:
        """Get a single episode with all steps."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/v1/episodes/{episode_id}")
            resp.raise_for_status()
            return resp.json()

    async def get_replay(self, episode_id: str) -> dict:
        """Get replay-ready view of an episode."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/v1/episodes/{episode_id}/replay")
            resp.raise_for_status()
            return resp.json()

    async def diff_episodes(self, left_id: str, right_id: str) -> dict:
        """Diff two episodes."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/v1/episodes/diff",
                params={"left": left_id, "right": right_id},
            )
            resp.raise_for_status()
            return resp.json()

    async def export_jsonl(
        self,
        agent_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Export episodes as list of dicts (parsed from JSONL stream)."""
        import json

        params: dict[str, str] = {}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/v1/episodes/export",
                params=params,
            )
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            return [json.loads(line) for line in lines if line.strip()]
