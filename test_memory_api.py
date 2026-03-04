from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import memory_api


class StubAgent:
    async def start(self):
        return self

    async def close(self):
        return None

    async def add(self, **kwargs):
        return SimpleNamespace(memory_id="m_add_1")

    async def query_nuanced(self, **kwargs):
        return {"query": kwargs["query"], "strategy": "nuanced_hybrid", "results": [{"memory_id": "m1"}]}

    async def store_short_lived(self, **kwargs):
        return SimpleNamespace(memory_id="m_short_1")

    async def retrieve_short_lived(self, **kwargs):
        return {"query": kwargs["query"], "strategy": "short_lived_hybrid", "results": [{"memory_id": "m2"}]}


def test_memory_api_endpoints(monkeypatch):
    stub = StubAgent()
    monkeypatch.setattr(memory_api.MemoryManagerAgent, "from_env", classmethod(lambda cls, env_path=None: stub))

    with TestClient(memory_api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        add_resp = client.post(
            "/memories",
            json={
                "layer": "semantic",
                "content": "user likes concise replies",
                "when_to_use": "writing responses",
                "score": 0.8,
                "metadata": {"source": "test"},
            },
        )
        assert add_resp.status_code == 200
        assert add_resp.json() == {"status": "ok", "memory_id": "m_add_1"}

        query_resp = client.post(
            "/memories/query",
            json={"query": "concise response style", "top_k": 5, "vector_weight": 0.7},
        )
        assert query_resp.status_code == 200
        assert query_resp.json()["strategy"] == "nuanced_hybrid"

        short_add_resp = client.post(
            "/memories/short-lived",
            json={
                "content": "session token valid for 20 min",
                "when_to_use": "active debugging session",
                "score": 0.6,
                "ttl_hours": 1,
                "metadata": {"ticket": "INC-101"},
            },
        )
        assert short_add_resp.status_code == 200
        assert short_add_resp.json() == {"status": "ok", "memory_id": "m_short_1"}

        short_query_resp = client.post(
            "/memories/short-lived/query",
            json={"query": "session token", "top_k": 5, "vector_weight": 0.7, "include_expired": False},
        )
        assert short_query_resp.status_code == 200
        assert short_query_resp.json()["strategy"] == "short_lived_hybrid"
