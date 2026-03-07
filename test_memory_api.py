from __future__ import annotations

import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

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


@pytest.fixture(autouse=True)
def _reset_auth_state():
    memory_api._auth_state._nonce_expiry_by_key.clear()
    memory_api._auth_state._request_times_by_key.clear()


def _sign_headers(
    *,
    method: str,
    path: str,
    body_obj: dict,
    secret: str,
    key_id: str = "test-key",
    timestamp: int | None = None,
    nonce: str | None = None,
) -> tuple[dict[str, str], str]:
    body_raw = json.dumps(body_obj, separators=(",", ":")).encode("utf-8")
    ts = str(int(time.time()) if timestamp is None else int(timestamp))
    request_nonce = nonce or uuid4().hex
    body_hash = hashlib.sha256(body_raw).hexdigest()
    signing_payload = f"{method.upper()}\n{path}\n{ts}\n{request_nonce}\n{body_hash}"
    signature = hmac.new(secret.encode("utf-8"), signing_payload.encode("utf-8"), hashlib.sha256).hexdigest()

    return (
        {
            "X-Key-Id": key_id,
            "X-Timestamp": ts,
            "X-Nonce": request_nonce,
            "X-Signature": signature,
            "Content-Type": "application/json",
        },
        body_raw.decode("utf-8"),
    )


def _post_signed(client: TestClient, path: str, body: dict, secret: str, **kwargs):
    headers, raw_body = _sign_headers(method="POST", path=path, body_obj=body, secret=secret, **kwargs)
    return client.post(path, data=raw_body, headers=headers)


def test_memory_api_endpoints_with_auth(monkeypatch):
    stub = StubAgent()
    monkeypatch.setenv("API_SECRET", "test-secret")
    monkeypatch.setenv("AUTH_MAX_SKEW_SECONDS", "300")
    monkeypatch.setenv("AUTH_NONCE_TTL_SECONDS", "600")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "60")
    monkeypatch.setattr(memory_api.MemoryManagerAgent, "from_env", classmethod(lambda cls, env_path=None: stub))

    with TestClient(memory_api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        add_resp = _post_signed(
            client,
            "/memories",
            {
                "layer": "semantic",
                "content": "user likes concise replies",
                "when_to_use": "writing responses",
                "score": 0.8,
                "metadata": {"source": "test"},
            },
            "test-secret",
        )
        assert add_resp.status_code == 200
        assert add_resp.json() == {"status": "ok", "memory_id": "m_add_1"}

        query_resp = _post_signed(
            client,
            "/memories/query",
            {"query": "concise response style", "top_k": 5, "vector_weight": 0.7},
            "test-secret",
        )
        assert query_resp.status_code == 200
        assert query_resp.json()["strategy"] == "nuanced_hybrid"

        short_add_resp = _post_signed(
            client,
            "/memories/short-lived",
            {
                "content": "session token valid for 20 min",
                "when_to_use": "active debugging session",
                "score": 0.6,
                "ttl_hours": 1,
                "metadata": {"ticket": "INC-101"},
            },
            "test-secret",
        )
        assert short_add_resp.status_code == 200
        assert short_add_resp.json() == {"status": "ok", "memory_id": "m_short_1"}

        short_query_resp = _post_signed(
            client,
            "/memories/short-lived/query",
            {"query": "session token", "top_k": 5, "vector_weight": 0.7, "include_expired": False},
            "test-secret",
        )
        assert short_query_resp.status_code == 200
        assert short_query_resp.json()["strategy"] == "short_lived_hybrid"


def test_auth_rejects_invalid_signature(monkeypatch):
    stub = StubAgent()
    monkeypatch.setenv("API_SECRET", "test-secret")
    monkeypatch.setattr(memory_api.MemoryManagerAgent, "from_env", classmethod(lambda cls, env_path=None: stub))

    with TestClient(memory_api.app) as client:
        resp = _post_signed(
            client,
            "/memories/query",
            {"query": "x"},
            "wrong-secret",
        )
        assert resp.status_code == 401
        assert "signature" in resp.json()["detail"].lower()


def test_auth_rejects_stale_timestamp(monkeypatch):
    stub = StubAgent()
    monkeypatch.setenv("API_SECRET", "test-secret")
    monkeypatch.setenv("AUTH_MAX_SKEW_SECONDS", "300")
    monkeypatch.setattr(memory_api.MemoryManagerAgent, "from_env", classmethod(lambda cls, env_path=None: stub))

    with TestClient(memory_api.app) as client:
        stale_ts = int(time.time()) - 301
        resp = _post_signed(
            client,
            "/memories/query",
            {"query": "x"},
            "test-secret",
            timestamp=stale_ts,
        )
        assert resp.status_code == 401
        assert "timestamp" in resp.json()["detail"].lower()


def test_auth_rejects_nonce_replay(monkeypatch):
    stub = StubAgent()
    monkeypatch.setenv("API_SECRET", "test-secret")
    monkeypatch.setenv("AUTH_NONCE_TTL_SECONDS", "600")
    monkeypatch.setattr(memory_api.MemoryManagerAgent, "from_env", classmethod(lambda cls, env_path=None: stub))

    with TestClient(memory_api.app) as client:
        shared_nonce = "fixed-nonce"
        payload = {"query": "x"}

        first = _post_signed(
            client,
            "/memories/query",
            payload,
            "test-secret",
            nonce=shared_nonce,
        )
        assert first.status_code == 200

        second = _post_signed(
            client,
            "/memories/query",
            payload,
            "test-secret",
            nonce=shared_nonce,
        )
        assert second.status_code == 401
        assert "nonce" in second.json()["detail"].lower()


def test_rate_limit_per_key(monkeypatch):
    stub = StubAgent()
    monkeypatch.setenv("API_SECRET", "test-secret")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    monkeypatch.setattr(memory_api.MemoryManagerAgent, "from_env", classmethod(lambda cls, env_path=None: stub))

    with TestClient(memory_api.app) as client:
        ok1 = _post_signed(client, "/memories/query", {"query": "x"}, "test-secret", key_id="k1")
        ok2 = _post_signed(client, "/memories/query", {"query": "x"}, "test-secret", key_id="k1")
        limited = _post_signed(client, "/memories/query", {"query": "x"}, "test-secret", key_id="k1")

        assert ok1.status_code == 200
        assert ok2.status_code == 200
        assert limited.status_code == 429


def test_reserved_metadata_keys_are_rejected(monkeypatch):
    stub = StubAgent()
    monkeypatch.setenv("API_SECRET", "test-secret")
    monkeypatch.setattr(memory_api.MemoryManagerAgent, "from_env", classmethod(lambda cls, env_path=None: stub))

    with TestClient(memory_api.app) as client:
        resp = _post_signed(
            client,
            "/memories",
            {
                "layer": "semantic",
                "content": "hello",
                "metadata": {"score": 0.3},
            },
            "test-secret",
        )
        assert resp.status_code == 400
        assert "reserved" in resp.json()["detail"].lower()
