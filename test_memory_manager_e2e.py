"""End-to-end integration test for layered memory manager.

Covers:
1. Add memories into episodic / semantic / long_term layers
2. Verify each layer can show the inserted memory
3. Verify nuanced retrieval returns expected layer-specific memory
4. Cleanup inserted memories

Run:
  pytest -q tests/test_memory_manager_e2e.py -s

Required env vars (via .env or shell):
- AGENT_MEMORY_DB_URI (pgvector connection URI)
- AGENT_MEMORY_OPENAI_API_KEY or OPENAI_API_KEY
"""

from __future__ import annotations

import time
import uuid

import pytest

import memory_manager_agent as mma

try:
    from memory_manager_agent import MemoryManagerAgent
except Exception as import_exc:  # pragma: no cover - defensive collection guard
    MemoryManagerAgent = None
    _IMPORT_ERROR = import_exc
else:
    _IMPORT_ERROR = None


@pytest.mark.asyncio
async def test_memory_manager_e2e_add_and_retrieve_layers():
    """Validate add/retrieve flows across episodic, semantic, and long-term layers."""
    if MemoryManagerAgent is None:
        pytest.skip(f"memory_manager_agent import failed: {_IMPORT_ERROR}")
    if not hasattr(mma.AgentMemoryClient, "from_env"):
        pytest.skip("agent_memory_client runtime is stubbed; skipping e2e integration test.")

    agent = MemoryManagerAgent.from_env()

    if not agent.client.db_uri:
        pytest.skip("AGENT_MEMORY_DB_URI is not set; skipping integration test.")

    if not agent.client.openai_api_key:
        pytest.skip("OpenAI API key is not set; skipping integration test.")

    run_id = f"e2e-{uuid.uuid4().hex[:10]}"

    episodic_text = f"{run_id} episodic: user hit onboarding DB timeout at 10:41"
    semantic_text = f"{run_id} semantic: if onboarding timeout occurs, retry with exponential backoff"
    long_term_text = f"{run_id} long_term: user prefers concise, actionable diagnostics"

    created_ids: list[str] = []
    store_latencies_ms: dict[str, float] = {}
    retrieve_latencies_ms: dict[str, float] = {}

    await agent.start()
    try:
        # 1) Add one memory per layer
        t0 = time.perf_counter()
        e_node = await agent.add(
            layer="episodic",
            content=episodic_text,
            when_to_use=f"{run_id} when debugging a specific failed session",
            score=0.6,
            test_run_id=run_id,
        )
        store_latencies_ms["episodic"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        s_node = await agent.add(
            layer="semantic",
            content=semantic_text,
            when_to_use=f"{run_id} when selecting robust retry strategy",
            score=0.8,
            test_run_id=run_id,
        )
        store_latencies_ms["semantic"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        l_node = await agent.add(
            layer="long_term",
            content=long_term_text,
            when_to_use=f"{run_id} when formatting responses for this user",
            score=0.95,
            test_run_id=run_id,
        )
        store_latencies_ms["long_term"] = (time.perf_counter() - t0) * 1000

        created_ids.extend([e_node.memory_id, s_node.memory_id, l_node.memory_id])

        # 2) Verify each layer contains the inserted memory
        episodic_rows = await agent.show_layer("episodic", limit=500)
        semantic_rows = await agent.show_layer("semantic", limit=500)
        long_term_rows = await agent.show_layer("long_term", limit=500)

        assert any(run_id in row.get("content", "") for row in episodic_rows), "Missing episodic inserted memory"
        assert any(run_id in row.get("content", "") for row in semantic_rows), "Missing semantic inserted memory"
        assert any(run_id in row.get("content", "") for row in long_term_rows), "Missing long-term inserted memory"

        # 3) Nuanced retrieval should surface layer-specific item for matching query intent
        t0 = time.perf_counter()
        q1 = await agent.query_nuanced(query=f"{run_id} onboarding timeout session", top_k=5)
        retrieve_latencies_ms["episodic_query"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        q2 = await agent.query_nuanced(query=f"{run_id} retry strategy backoff", top_k=5)
        retrieve_latencies_ms["semantic_query"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        q3 = await agent.query_nuanced(query=f"{run_id} user response preference", top_k=5)
        retrieve_latencies_ms["long_term_query"] = (time.perf_counter() - t0) * 1000

        assert q1["results"], "No results for episodic-style query"
        assert q2["results"], "No results for semantic-style query"
        assert q3["results"], "No results for long-term-style query"

        # Nuanced retrieval is cross-layer reranked, so top-1 layer is not strictly guaranteed.
        # Validate that the expected layer-specific memory appears in the returned top-k results.
        assert any(
            (item.get("layer") == "episodic" and run_id in item.get("content", ""))
            for item in q1["results"]
        ), "Expected episodic memory not found in episodic-style query results"

        assert any(
            (item.get("layer") == "semantic" and run_id in item.get("content", ""))
            for item in q2["results"]
        ), "Expected semantic memory not found in semantic-style query results"

        assert any(
            (item.get("layer") == "long_term" and run_id in item.get("content", ""))
            for item in q3["results"]
        ), "Expected long-term memory not found in long-term-style query results"

        avg_store_ms = sum(store_latencies_ms.values()) / len(store_latencies_ms)
        avg_retrieve_ms = sum(retrieve_latencies_ms.values()) / len(retrieve_latencies_ms)
        print(
            f"\n[latency] store_ms={store_latencies_ms} avg_store_ms={avg_store_ms:.2f} "
            f"retrieve_ms={retrieve_latencies_ms} avg_retrieve_ms={avg_retrieve_ms:.2f}"
        )

    finally:
        # 4) Cleanup created memories so repeated test runs do not accumulate data
        for memory_id in created_ids:
            try:
                await agent.client.delete_memory(memory_id)
            except Exception:
                # Best-effort cleanup in integration environments
                pass
        await agent.close()
