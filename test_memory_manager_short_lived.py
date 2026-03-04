from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from types import SimpleNamespace
import sys
import types

import pytest

# Keep tests dependency-light: stub agent_memory_client before importing
# memory_manager_agent so optional runtime dependencies are not required.
if "agent_memory_client" not in sys.modules:
    stub_module = types.ModuleType("agent_memory_client")

    @dataclass
    class _MemoryTarget:
        kind: str
        name: str

    class _AgentMemoryClient:
        pass

    stub_module.MemoryTarget = _MemoryTarget
    stub_module.AgentMemoryClient = _AgentMemoryClient
    sys.modules["agent_memory_client"] = stub_module

from memory_manager_agent import MemoryManagerAgent


class StubClient:
    def __init__(self):
        self.add_calls = []
        self.retrieve_payload = {"results": []}

    async def add_memory(self, **kwargs):
        self.add_calls.append(kwargs)
        return SimpleNamespace(memory_id="m1")

    async def retrieve_memory(self, **kwargs):
        return self.retrieve_payload


def make_agent(client: StubClient) -> MemoryManagerAgent:
    return MemoryManagerAgent(client=client, root_target_name="test")


@pytest.mark.asyncio
async def test_store_short_lived_adds_ttl_metadata():
    client = StubClient()
    agent = make_agent(client)

    await agent.store_short_lived(
        content="Temporary event",
        when_to_use="During current issue triage",
        ttl_hours=24,
        score=0.6,
    )

    assert len(client.add_calls) == 1
    payload = client.add_calls[0]
    assert payload["memory_content"] == "Temporary event"
    assert payload["short_lived"] is True
    assert payload["memory_policy"] == "short_lived"
    assert payload["ttl_seconds"] == 24 * 3600
    assert payload["expires_at"]


@pytest.mark.asyncio
async def test_retrieve_short_lived_filters_expired_and_non_short_lived():
    client = StubClient()
    agent = make_agent(client)

    now = datetime.now(tz=timezone.utc)
    active_expiry = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    expired_expiry = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    client.retrieve_payload = {
        "results": [
            {
                "memory_id": "active",
                "content": "active short-lived",
                "score": 0.7,
                "message_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {"short_lived": True, "expires_at": active_expiry},
            },
            {
                "memory_id": "expired",
                "content": "expired short-lived",
                "score": 0.9,
                "message_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {"short_lived": True, "expires_at": expired_expiry},
            },
            {
                "memory_id": "regular",
                "content": "normal episodic memory",
                "score": 0.8,
                "message_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {"memory_layer": "episodic"},
            },
        ],
    }

    result = await agent.retrieve_short_lived(query="issue triage", top_k=10)

    ids = [item["memory_id"] for item in result["results"]]
    assert ids == ["active"]
    assert result["filtered"]["expired"] == 1
    assert result["filtered"]["non_short_lived"] == 1
