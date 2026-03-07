from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from types import SimpleNamespace
import sys
import types

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - only used for non-pytest script execution
    class _NoPytestMark:
        @staticmethod
        def asyncio(func):
            return func

    class _NoPytest:
        mark = _NoPytestMark()

    pytest = _NoPytest()

# Keep tests dependency-light: stub agent_memory_client before importing
# memory_manager_agent so optional runtime dependencies are not required.
if __name__ != "__main__" and "agent_memory_client" not in sys.modules:
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

class StubClient:
    def __init__(self):
        self.add_calls = []
        self.retrieve_payload = {"results": []}

    async def add_memory(self, **kwargs):
        self.add_calls.append(kwargs)
        return SimpleNamespace(memory_id="m1")

    async def retrieve_memory(self, **kwargs):
        return self.retrieve_payload


def _memory_manager_cls():
    from memory_manager_agent import MemoryManagerAgent

    return MemoryManagerAgent


def make_agent(client: StubClient):
    return _memory_manager_cls()(client=client, root_target_name="test")


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


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    return float(raw)


def _prompt_bool(label: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    raw = input(f"{label} (y/n) [{default_text}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "true", "1"}


async def _run_manual_short_lived_loop() -> None:
    from agent_memory_client import AgentMemoryClient
    MemoryManagerAgent = _memory_manager_cls()

    env_file = input("Env file path (optional): ").strip() or None
    target_kind = input("Target kind [user]: ").strip() or "user"
    target_name = input("Target name [manual_test_user]: ").strip() or "manual_test_user"

    client = AgentMemoryClient.from_env(env_path=env_file)
    agent = MemoryManagerAgent(
        client=client,
        target_kind=target_kind,
        root_target_name=target_name,
    )
    await agent.start()
    try:
        print("\nInteractive short-lived memory runner")
        print("1) Add memory")
        print("2) Retrieve memory by query")
        print("3) Exit\n")

        while True:
            choice = input("Choose option (1/2/3): ").strip()

            if choice == "1":
                content = input("Content: ").strip()
                when_to_use = input("When to use (optional): ").strip()
                score = _prompt_float("Score", 0.7)
                ttl_hours = _prompt_int("TTL hours", 24)

                stored = await agent.store_short_lived(
                    content=content,
                    when_to_use=when_to_use,
                    score=score,
                    ttl_hours=ttl_hours,
                )
                print(json.dumps({"status": "ok", "memory_id": stored.memory_id}, ensure_ascii=False, indent=2))

            elif choice == "2":
                query = input("Query: ").strip()
                top_k = _prompt_int("Top K", 10)
                vector_weight = _prompt_float("Vector weight", 0.7)
                include_expired = _prompt_bool("Include expired", False)

                retrieved = await agent.retrieve_short_lived(
                    query=query,
                    top_k=top_k,
                    vector_weight=vector_weight,
                    include_expired=include_expired,
                )
                print(json.dumps(retrieved, ensure_ascii=False, indent=2))

            elif choice == "3":
                print("Exiting.")
                break

            else:
                print("Invalid option. Use 1, 2, or 3.")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(_run_manual_short_lived_loop())
