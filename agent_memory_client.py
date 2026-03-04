"""Standalone wrapper for initializing and using ReMe vector memory.

This module provides a thin, reusable class for:
- initializing ReMe with OpenAI LLM + embedding models
- adding memory records
- retrieving memory records

It supports multiple vector backends (local/chroma/pgvector/qdrant/es).
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import asyncpg
from reme import ReMe
from reme.core.utils.env_utils import load_env
from reme.core.enumeration import MemoryType
from reme.core.schema import MemoryNode, VectorNode

MemoryKind = Literal["user", "task", "tool"]


@dataclass
class MemoryTarget:
    """Identifies which memory namespace to use."""

    kind: MemoryKind
    name: str


class AgentMemoryClient:
    """Simple async client for ReMe vector memory operations."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        embedding_use_dimensions: bool | None = None,
        llm_base_url: str | None = None,
        embedding_base_url: str | None = None,
        vector_backend: str | None = None,
        db_uri: str | None = None,
        working_dir: str | None = None,
        collection_name: str | None = None,
        vector_store_extra: dict[str, Any] | None = None,
        enable_profile: bool = False,
    ):
        """Create a memory client.

        Args:
            openai_api_key: API key for both LLM and embeddings. Falls back to env vars.
            llm_model: LLM model name.
            embedding_model: Embedding model name.
            llm_base_url: Optional custom OpenAI-compatible base URL.
            embedding_base_url: Optional custom embedding base URL.
            vector_backend: One of local/chroma/pgvector/qdrant/es.
            db_uri: Optional URI used by remote backends.
            working_dir: ReMe working directory.
            collection_name: Vector collection/index/table name.
            vector_store_extra: Extra backend-specific vector store settings.
            enable_profile: Enable profile file operations in ReMe.
        """
        self.openai_api_key = (
            openai_api_key
            or os.getenv("AGENT_MEMORY_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("REME_LLM_API_KEY")
            or ""
        )
        self.llm_model = llm_model or os.getenv("AGENT_MEMORY_LLM_MODEL", "gpt-5.2")
        self.embedding_model = embedding_model or os.getenv(
            "AGENT_MEMORY_EMBEDDING_MODEL",
            "text-embedding-3-large",
        )
        env_dims = os.getenv("AGENT_MEMORY_EMBEDDING_DIMENSIONS")
        self.embedding_dimensions = (
            embedding_dimensions
            if embedding_dimensions is not None
            else (int(env_dims) if env_dims else 3072)
        )
        env_use_dims = os.getenv("AGENT_MEMORY_EMBEDDING_USE_DIMENSIONS")
        self.embedding_use_dimensions = (
            embedding_use_dimensions
            if embedding_use_dimensions is not None
            else (str(env_use_dims).lower() == "true" if env_use_dims is not None else False)
        )
        self.llm_base_url = llm_base_url or os.getenv("AGENT_MEMORY_LLM_BASE_URL") or os.getenv("REME_LLM_BASE_URL")
        self.embedding_base_url = (
            embedding_base_url
            or os.getenv("AGENT_MEMORY_EMBEDDING_BASE_URL")
            or os.getenv("REME_EMBEDDING_BASE_URL")
        )
        self.vector_backend = vector_backend or os.getenv("AGENT_MEMORY_VECTOR_BACKEND", "pgvector")
        self.db_uri = db_uri or os.getenv("AGENT_MEMORY_DB_URI") or os.getenv("DATABASE_URL")
        self.working_dir = working_dir or os.getenv("AGENT_MEMORY_WORKING_DIR", ".agent_memory")
        self.collection_name = collection_name or os.getenv("AGENT_MEMORY_COLLECTION_NAME", "agent_memory")
        self.vector_store_extra = vector_store_extra or {}
        self.enable_profile = enable_profile

        self._reme: ReMe | None = None
        self._pg_pool: asyncpg.Pool | None = None

    @classmethod
    def from_env(
        cls,
        env_path: str | None = None,
        **overrides: Any,
    ) -> "AgentMemoryClient":
        """Build a client from .env/environment variables.

        Supported environment variables:
        - AGENT_MEMORY_OPENAI_API_KEY / OPENAI_API_KEY
        - AGENT_MEMORY_LLM_MODEL
        - AGENT_MEMORY_EMBEDDING_MODEL
        - AGENT_MEMORY_EMBEDDING_DIMENSIONS
        - AGENT_MEMORY_EMBEDDING_USE_DIMENSIONS
        - AGENT_MEMORY_LLM_BASE_URL
        - AGENT_MEMORY_EMBEDDING_BASE_URL
        - AGENT_MEMORY_VECTOR_BACKEND
        - AGENT_MEMORY_DB_URI (or DATABASE_URL)
        - AGENT_MEMORY_WORKING_DIR
        - AGENT_MEMORY_COLLECTION_NAME
        """
        load_env(env_path, enable_log=False)
        return cls(**overrides)

    @property
    def reme(self) -> ReMe:
        """Return initialized ReMe instance."""
        if self._reme is None:
            raise RuntimeError("Memory client is not initialized. Call `await start()` first.")
        return self._reme

    def _build_vector_store_config(self) -> dict[str, Any]:
        """Build backend-specific vector store config for ReMe."""
        config: dict[str, Any] = {
            "backend": self.vector_backend,
            "collection_name": self.collection_name,
        }

        if self.db_uri:
            if self.vector_backend == "pgvector":
                # Example: postgresql://user:password@host:5432/dbname
                config["dsn"] = self.db_uri
            elif self.vector_backend == "qdrant":
                # Example: http://localhost:6333 or https://<cluster>.cloud.qdrant.io
                config["url"] = self.db_uri
            elif self.vector_backend == "es":
                # Example: http://localhost:9200
                config["hosts"] = [self.db_uri]
            elif self.vector_backend == "chroma":
                parsed = urlparse(self.db_uri)
                if parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port:
                    config["host"] = parsed.hostname
                    config["port"] = parsed.port
                # For local Chroma persistence, omit db_uri and rely on working_dir.

        config.update(self.vector_store_extra)
        return config

    async def start(self) -> "AgentMemoryClient":
        """Initialize and start ReMe."""
        if self.vector_backend == "pgvector" and not self.db_uri:
            raise ValueError("`db_uri` is required when `vector_backend='pgvector'`.")

        self._reme = ReMe(
            llm_api_key=self.openai_api_key or None,
            embedding_api_key=self.openai_api_key or None,
            llm_base_url=self.llm_base_url,
            embedding_base_url=self.embedding_base_url,
            working_dir=self.working_dir,
            enable_profile=self.enable_profile,
            default_llm_config={
                "backend": "openai",
                "model_name": self.llm_model,
            },
            default_embedding_model_config={
                "backend": "openai",
                "model_name": self.embedding_model,
                "dimensions": self.embedding_dimensions,
                "use_dimensions": self.embedding_use_dimensions,
                "enable_cache": False,
            },
            default_vector_store_config=self._build_vector_store_config(),
            # Keep logs quiet by default in standalone usage.
            log_to_console=False,
            enable_logo=False,
        )
        await self._reme.start()
        return self

    async def close(self) -> None:
        """Gracefully close ReMe resources."""
        if self._pg_pool is not None:
            await self._pg_pool.close()
            self._pg_pool = None
        if self._reme is not None:
            await self._reme.close()
            self._reme = None

    def _target_kwargs(self, target: MemoryTarget) -> dict[str, str]:
        if target.kind == "user":
            return {"user_name": target.name}
        if target.kind == "task":
            return {"task_name": target.name}
        if target.kind == "tool":
            return {"tool_name": target.name}
        raise ValueError(f"Unsupported target kind: {target.kind}")

    @staticmethod
    def default_target_from_env() -> MemoryTarget:
        """Read default memory target from environment variables."""
        return MemoryTarget(
            kind=os.getenv("AGENT_MEMORY_TARGET_KIND", "user"),
            name=os.getenv("AGENT_MEMORY_TARGET_NAME", "default_user"),
        )

    async def add_memory(
        self,
        target: MemoryTarget,
        memory_content: str,
        when_to_use: str = "",
        score: float = 0.0,
        **metadata: Any,
    ) -> Any:
        """Add a memory item to the selected target namespace."""
        return await self.reme.add_memory(
            memory_content=memory_content,
            when_to_use=when_to_use,
            score=score,
            **self._target_kwargs(target),
            **metadata,
        )

    async def retrieve_memory(
        self,
        target: MemoryTarget,
        query: str,
        top_k: int = 10,
        enable_time_filter: bool = True,
        strategy: Literal["semantic", "hybrid"] = "hybrid",
        vector_weight: float = 0.7,
    ) -> dict[str, Any]:
        """Retrieve memory for a target namespace.

        - `semantic`: ReMe default vector retrieval pipeline.
        - `hybrid`: pgvector + PostgreSQL full-text hybrid retrieval.
        """
        if strategy == "hybrid":
            if self.vector_backend != "pgvector":
                raise ValueError("Hybrid strategy in this client currently requires `vector_backend='pgvector'`.")
            return await self.retrieve_memory_hybrid(
                target=target,
                query=query,
                top_k=top_k,
                vector_weight=vector_weight,
            )

        result = await self.reme.retrieve_memory(
            query=query,
            retrieve_top_k=top_k,
            enable_time_filter=enable_time_filter,
            return_dict=True,
            **self._target_kwargs(target),
        )
        return result

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        max_v = max(values)
        min_v = min(values)
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    @staticmethod
    def _metadata_to_dict(value: Any) -> dict[str, Any]:
        """Normalize DB metadata payload to a dictionary."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _memory_type_for_target(target: MemoryTarget) -> str:
        if target.kind == "user":
            return MemoryType.PERSONAL.value
        if target.kind == "task":
            return MemoryType.PROCEDURAL.value
        if target.kind == "tool":
            return MemoryType.TOOL.value
        raise ValueError(f"Unsupported target kind: {target.kind}")

    async def _get_pg_pool(self) -> asyncpg.Pool:
        if self.db_uri is None:
            raise ValueError("`db_uri` is required for pgvector hybrid retrieval.")
        if self._pg_pool is None:
            self._pg_pool = await asyncpg.create_pool(dsn=self.db_uri, min_size=1, max_size=4)
        return self._pg_pool

    async def retrieve_memory_hybrid(
        self,
        target: MemoryTarget,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
    ) -> dict[str, Any]:
        """Hybrid retrieval using pgvector distance + PostgreSQL full-text ranking."""
        if not (0.0 <= vector_weight <= 1.0):
            raise ValueError("`vector_weight` must be between 0 and 1.")
        keyword_weight = 1.0 - vector_weight

        vector_store = self.reme.default_vector_store
        table_name = vector_store.collection_name
        memory_type = self._memory_type_for_target(target)
        memory_target = target.name
        candidate_k = min(max(20, top_k * 3), 200)

        query_vector = await vector_store.get_embedding(query)
        query_vector_str = f"[{','.join(map(str, query_vector))}]"

        pool = await self._get_pg_pool()
        async with pool.acquire() as conn:
            vector_rows = await conn.fetch(
                f"""
                SELECT id, content, metadata, (1 - (vector <=> $1::vector)) AS vector_score
                FROM {table_name}
                WHERE metadata->>'memory_type' = $2
                  AND metadata->>'memory_target' = $3
                ORDER BY vector <=> $1::vector
                LIMIT $4
                """,
                query_vector_str,
                memory_type,
                memory_target,
                candidate_k,
            )

            keyword_rows = await conn.fetch(
                f"""
                SELECT id, content, metadata,
                       ts_rank_cd(to_tsvector('simple', content), websearch_to_tsquery('simple', $1)) AS keyword_score
                FROM {table_name}
                WHERE metadata->>'memory_type' = $2
                  AND metadata->>'memory_target' = $3
                  AND to_tsvector('simple', content) @@ websearch_to_tsquery('simple', $1)
                ORDER BY keyword_score DESC
                LIMIT $4
                """,
                query,
                memory_type,
                memory_target,
                candidate_k,
            )

        vector_scores = {r["id"]: float(r["vector_score"] or 0.0) for r in vector_rows}
        keyword_scores = {r["id"]: float(r["keyword_score"] or 0.0) for r in keyword_rows}
        vector_scores_n = self._normalize_scores(vector_scores)
        keyword_scores_n = self._normalize_scores(keyword_scores)

        merged: dict[str, dict[str, Any]] = {}
        for row in vector_rows:
            mid = row["id"]
            merged[mid] = {
                "id": mid,
                "content": row["content"] or "",
                "metadata": self._metadata_to_dict(row["metadata"]),
                "vector_score": vector_scores_n.get(mid, 0.0),
                "keyword_score": 0.0,
            }
        for row in keyword_rows:
            mid = row["id"]
            if mid not in merged:
                merged[mid] = {
                    "id": mid,
                    "content": row["content"] or "",
                    "metadata": self._metadata_to_dict(row["metadata"]),
                    "vector_score": 0.0,
                    "keyword_score": keyword_scores_n.get(mid, 0.0),
                }
            else:
                merged[mid]["keyword_score"] = keyword_scores_n.get(mid, 0.0)

        ranked = []
        for item in merged.values():
            hybrid_score = item["vector_score"] * vector_weight + item["keyword_score"] * keyword_weight
            item["hybrid_score"] = hybrid_score

            vector_node = VectorNode(
                vector_id=item["id"],
                content=item["content"],
                metadata=item["metadata"],
            )
            memory_node = MemoryNode.from_vector_node(vector_node)
            ranked.append(
                {
                    "memory_id": memory_node.memory_id,
                    "memory_type": memory_node.memory_type.value,
                    "memory_target": memory_node.memory_target,
                    "when_to_use": memory_node.when_to_use,
                    "content": memory_node.content,
                    "score": hybrid_score,
                    "vector_score": item["vector_score"],
                    "keyword_score": item["keyword_score"],
                    "message_time": memory_node.message_time,
                    "author": memory_node.author,
                    "metadata": memory_node.metadata,
                },
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        ranked = ranked[:top_k]

        return {
            "query": query,
            "strategy": "hybrid",
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "results": ranked,
        }

    async def list_memories_raw(
        self,
        limit: int = 200,
        memory_type: str | None = None,
        memory_target: str | None = None,
    ) -> list[dict[str, Any]]:
        """List memory rows directly from pgvector table.

        Useful for inspection/debugging regardless of retrieval strategy.
        """
        if self.vector_backend != "pgvector":
            raise ValueError("`list_memories_raw` is only supported with `vector_backend='pgvector'`.")

        pool = await self._get_pg_pool()
        vector_store = self.reme.default_vector_store
        table_name = vector_store.collection_name

        clauses = []
        params: list[Any] = []
        idx = 1

        if memory_type:
            clauses.append(f"metadata->>'memory_type' = ${idx}")
            params.append(memory_type)
            idx += 1
        if memory_target:
            clauses.append(f"metadata->>'memory_target' = ${idx}")
            params.append(memory_target)
            idx += 1

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_placeholder = f"${idx}"
        params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata
                FROM {table_name}
                {where_clause}
                ORDER BY COALESCE(metadata->>'time_modified', metadata->>'time_created', '') DESC
                LIMIT {limit_placeholder}
                """,
                *params,
            )

        results: list[dict[str, Any]] = []
        for row in rows:
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                # Defensive handling if driver returns serialized JSON in some environments.
                import json

                metadata = json.loads(metadata)

            vector_node = VectorNode(
                vector_id=row["id"],
                content=row["content"] or "",
                metadata=metadata,
            )
            memory_node = MemoryNode.from_vector_node(vector_node)
            results.append(
                {
                    "memory_id": memory_node.memory_id,
                    "memory_type": memory_node.memory_type.value,
                    "memory_target": memory_node.memory_target,
                    "when_to_use": memory_node.when_to_use,
                    "content": memory_node.content,
                    "message_time": memory_node.message_time,
                    "time_created": memory_node.time_created,
                    "time_modified": memory_node.time_modified,
                    "author": memory_node.author,
                    "score": memory_node.score,
                    "metadata": memory_node.metadata,
                },
            )
        return results

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory by memory_id."""
        await self.reme.delete_memory(memory_id)

    async def update_memory(
        self,
        target: MemoryTarget,
        memory_id: str,
        memory_content: str | None = None,
        when_to_use: str | None = None,
        score: float | None = None,
        **metadata: Any,
    ) -> Any:
        """Update an existing memory."""
        return await self.reme.update_memory(
            memory_id=memory_id,
            memory_content=memory_content,
            when_to_use=when_to_use,
            score=score,
            **self._target_kwargs(target),
            **metadata,
        )


async def _demo() -> None:
    """Minimal + full-options runnable example."""
    # Minimal: everything read from `.env`
    client = AgentMemoryClient.from_env()

    # Full options (optional): uncomment and edit only what you need.
    # client = AgentMemoryClient.from_env(
    #     env_path=".env",  # Custom env file path; None -> auto-discover `.env`.
    #     openai_api_key=None,  # OpenAI API key; defaults to env vars.
    #     llm_model=None,  # LLM model name, e.g. "gpt-5.2".
    #     embedding_model=None,  # Embedding model name, e.g. "text-embedding-3-large".
    #     llm_base_url=None,  # Optional OpenAI-compatible endpoint for chat.
    #     embedding_base_url=None,  # Optional OpenAI-compatible endpoint for embeddings.
    #     vector_backend="pgvector",  # Backend: pgvector/local/chroma/qdrant/es.
    #     db_uri=None,  # DB URI (required for pgvector).
    #     working_dir=".agent_memory",  # Runtime directory in current project.
    #     collection_name="agent_memory",  # PG table / collection name.
    #     vector_store_extra=None,  # Backend-specific extra settings.
    #     enable_profile=False,  # Keep False for vector-only memory use.
    # )

    await client.start()

    # Minimal target: read from env (AGENT_MEMORY_TARGET_KIND / AGENT_MEMORY_TARGET_NAME).
    target = AgentMemoryClient.default_target_from_env()

    # Full target (optional):
    # target = MemoryTarget(kind="user", name="mridul")

    # Minimal add.
    await client.add_memory(
        target=target,
        memory_content="User prefers concise answers and concrete steps.",
    )

    # Full add options (optional):
    # await client.add_memory(
    #     target=target,  # Memory namespace (user/task/tool + name).
    #     memory_content="User prefers concise answers and concrete steps.",  # Memory text.
    #     when_to_use="When giving implementation guidance.",  # Retrieval-facing cue text.
    #     score=0.9,  # Importance score.
    #     message_time="2026-03-03 12:00:00",  # Event time.
    #     ref_memory_id="",  # Link to a related memory.
    #     author="agent",  # Source/author tag.
    #     episode_id="ep-001",  # Custom metadata: episode/session grouping.
    # )

    # Minimal retrieve.
    result = await client.retrieve_memory(
        target=target,
        query="How should I format responses for this user?",
    )

    # Full retrieve options (optional):
    # result = await client.retrieve_memory(
    #     target=target,  # Which namespace to retrieve from.
    #     query="How should I format responses for this user?",  # Search query.
    #     top_k=5,  # Number of results to return.
    #     enable_time_filter=True,  # Used by semantic ReMe retriever path.
    #     strategy="hybrid",  # "semantic" or "hybrid" (hybrid requires pgvector).
    #     vector_weight=0.7,  # Hybrid: weight for vector score; keyword = 1 - vector_weight.
    # )

    # Optional: inspect stored rows directly.
    # rows = await client.list_memories_raw(
    #     limit=100,  # Max rows.
    #     memory_type="personal",  # identity/personal/procedural/tool/summary/history.
    #     memory_target=target.name,  # Filter by target name.
    # )

    print(result)

    # Optional: hybrid API directly.
    # result = await client.retrieve_memory_hybrid(
    #     target=target,  # Namespace.
    #     query="How should I format responses for this user?",  # Query.
    #     top_k=5,  # Number of results.
    #     vector_weight=0.7,  # Vector-vs-keyword mix.
    # )

    # Optional: raw listing API.
    # print(rows)

    # Optional: full add from previous example.
    # await client.add_memory(
    #     target=target,
    #     memory_content="User prefers concise answers and concrete steps.",
    #     when_to_use="When giving implementation guidance.",
    #     score=0.9,
    # )

    await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo())
