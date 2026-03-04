"""Memory Manager Agent for layered memory control.

Layers:
- episodic: short-lived events and session-specific details
- semantic: generalized reusable facts/patterns
- long_term: durable preferences, policies, stable identity

This agent uses AgentMemoryClient (pgvector + hybrid retrieval) and provides:
- add memory by layer
- nuanced retrieval across layers
- pruning/deletion of low-value or stale memories
- optional CLI for basic operations
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from agent_memory_client import AgentMemoryClient, MemoryTarget

Layer = Literal["episodic", "semantic", "long_term"]


@dataclass
class LayerConfig:
    target: MemoryTarget
    retrieve_top_k: int
    layer_weight: float


class MemoryManagerAgent:
    """Agent that manages episodic/semantic/long-term memory layers."""

    def __init__(
        self,
        client: AgentMemoryClient,
        target_kind: str = "user",
        root_target_name: str = "default_user",
        episodic_suffix: str = "episodic",
        semantic_suffix: str = "semantic",
        long_term_suffix: str = "long_term",
    ):
        self.client = client
        self.target_kind = target_kind
        self.root_target_name = root_target_name

        self.layers: dict[Layer, LayerConfig] = {
            "episodic": LayerConfig(
                target=MemoryTarget(kind=target_kind, name=f"{root_target_name}:{episodic_suffix}"),
                retrieve_top_k=10,
                layer_weight=0.9,
            ),
            "semantic": LayerConfig(
                target=MemoryTarget(kind=target_kind, name=f"{root_target_name}:{semantic_suffix}"),
                retrieve_top_k=8,
                layer_weight=1.1,
            ),
            "long_term": LayerConfig(
                target=MemoryTarget(kind=target_kind, name=f"{root_target_name}:{long_term_suffix}"),
                retrieve_top_k=6,
                layer_weight=1.25,
            ),
        }

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "MemoryManagerAgent":
        """Build manager from .env + defaults."""
        client = AgentMemoryClient.from_env(env_path=env_path)
        target_kind = os.getenv("AGENT_MEMORY_TARGET_KIND", "user")
        root_target_name = os.getenv("AGENT_MEMORY_TARGET_NAME", "default_user")

        episodic_suffix = os.getenv("AGENT_MEMORY_EPISODIC_SUFFIX", "episodic")
        semantic_suffix = os.getenv("AGENT_MEMORY_SEMANTIC_SUFFIX", "semantic")
        long_term_suffix = os.getenv("AGENT_MEMORY_LONG_TERM_SUFFIX", "long_term")

        return cls(
            client=client,
            target_kind=target_kind,
            root_target_name=root_target_name,
            episodic_suffix=episodic_suffix,
            semantic_suffix=semantic_suffix,
            long_term_suffix=long_term_suffix,
        )

    async def start(self) -> "MemoryManagerAgent":
        await self.client.start()
        return self

    async def close(self) -> None:
        await self.client.close()

    async def add(
        self,
        layer: Layer,
        content: str,
        when_to_use: str = "",
        score: float = 0.7,
        **metadata: Any,
    ) -> Any:
        """Add memory to a specific layer."""
        cfg = self.layers[layer]
        metadata = {
            "memory_layer": layer,
            **metadata,
        }
        return await self.client.add_memory(
            target=cfg.target,
            memory_content=content,
            when_to_use=when_to_use,
            score=score,
            **metadata,
        )

    @staticmethod
    def _format_utc_time(dt: datetime) -> str:
        """Format datetime into stable UTC string used by this module."""
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _parse_ttl_seconds(value: Any) -> int | None:
        """Normalize metadata TTL values into seconds."""
        if value is None:
            return None
        try:
            ttl = int(value)
        except (TypeError, ValueError):
            return None
        return ttl if ttl >= 0 else None

    @staticmethod
    def _resolve_expiry(row: dict[str, Any]) -> datetime | None:
        """Resolve expiration time from metadata fields, if any."""
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            return None

        expires_at = metadata.get("expires_at")
        dt = MemoryManagerAgent._parse_time(expires_at)
        if dt is not None:
            return dt

        ttl_seconds = MemoryManagerAgent._parse_ttl_seconds(metadata.get("ttl_seconds"))
        if ttl_seconds is None:
            return None

        base_time = MemoryManagerAgent._parse_time(
            row.get("time_modified") or row.get("time_created") or row.get("message_time"),
        )
        if base_time is None:
            return None
        return base_time + timedelta(seconds=ttl_seconds)

    @staticmethod
    def _is_short_lived_memory(row: dict[str, Any]) -> bool:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            return False
        return bool(metadata.get("short_lived")) or metadata.get("memory_policy") == "short_lived"

    @staticmethod
    def _is_expired(row: dict[str, Any], now: datetime | None = None) -> bool:
        expiry = MemoryManagerAgent._resolve_expiry(row)
        if expiry is None:
            return False
        now = now or datetime.now(tz=timezone.utc)
        return expiry <= now

    async def store_short_lived(
        self,
        content: str,
        when_to_use: str = "",
        score: float = 0.7,
        ttl_hours: int = 24,
        **metadata: Any,
    ) -> Any:
        """Store short-lived episodic memory with explicit TTL metadata."""
        now = datetime.now(tz=timezone.utc)
        ttl_seconds = max(int(ttl_hours * 3600), 0)
        expires_at = now + timedelta(seconds=ttl_seconds)

        payload = {
            "short_lived": True,
            "memory_policy": "short_lived",
            "ttl_seconds": ttl_seconds,
            "expires_at": self._format_utc_time(expires_at),
            "stored_at": self._format_utc_time(now),
            **metadata,
        }
        return await self.add(
            layer="episodic",
            content=content,
            when_to_use=when_to_use,
            score=score,
            **payload,
        )

    async def retrieve_short_lived(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        include_expired: bool = False,
    ) -> dict[str, Any]:
        """Retrieve non-expired short-lived episodic memories."""
        cfg = self.layers["episodic"]
        # Over-fetch so post-filtering on expiry still returns useful results.
        candidate_k = min(max(top_k * 5, top_k), 100)
        resp = await self.client.retrieve_memory(
            target=cfg.target,
            query=query,
            top_k=candidate_k,
            strategy="hybrid",
            vector_weight=vector_weight,
        )

        now = datetime.now(tz=timezone.utc)
        results: list[dict[str, Any]] = []
        expired_filtered = 0
        non_short_lived_filtered = 0

        for item in resp.get("results", []):
            if not self._is_short_lived_memory(item):
                non_short_lived_filtered += 1
                continue
            if not include_expired and self._is_expired(item, now=now):
                expired_filtered += 1
                continue
            results.append(item)

        # Apply same recency-aware rerank style as nuanced query, but scoped to short-lived episodic memories.
        for item in results:
            base_score = float(item.get("score", 0.0))
            recency = self._recency_boost(item.get("time_modified") or item.get("message_time"))
            item["layer"] = "episodic"
            item["recency_boost"] = recency
            item["final_score"] = base_score * cfg.layer_weight + recency * 0.25

        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        results = results[:top_k]

        return {
            "query": query,
            "strategy": "short_lived_hybrid",
            "layer": "episodic",
            "include_expired": include_expired,
            "results": results,
            "filtered": {
                "expired": expired_filtered,
                "non_short_lived": non_short_lived_filtered,
            },
        }

    @staticmethod
    def _parse_time(value: str | None) -> datetime | None:
        if not value:
            return None

        value = value.strip()
        if not value:
            return None

        # ReMe default format: YYYY-MM-DD HH:MM:SS
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # ISO fallback
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None

    @staticmethod
    def _normalize_content(text: str) -> str:
        return " ".join((text or "").lower().split())

    @staticmethod
    def _recency_boost(time_str: str | None, half_life_days: float = 14.0) -> float:
        dt = MemoryManagerAgent._parse_time(time_str)
        if dt is None:
            return 0.0

        now = datetime.now(tz=timezone.utc)
        age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
        # Exponential-like decay in [0, 1]
        return 1.0 / (1.0 + (age_days / half_life_days))

    async def query_nuanced(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
    ) -> dict[str, Any]:
        """Retrieve and rerank memories across all layers with layer/recency-aware scoring."""
        collected: list[dict[str, Any]] = []

        for layer, cfg in self.layers.items():
            resp = await self.client.retrieve_memory(
                target=cfg.target,
                query=query,
                top_k=cfg.retrieve_top_k,
                strategy="hybrid",
                vector_weight=vector_weight,
            )

            for item in resp.get("results", []):
                base_score = float(item.get("score", 0.0))
                recency = self._recency_boost(item.get("time_modified") or item.get("message_time"))

                # Episodic benefits more from recency.
                recency_weight = 0.25 if layer == "episodic" else 0.10
                final_score = base_score * cfg.layer_weight + recency * recency_weight

                collected.append(
                    {
                        **item,
                        "layer": layer,
                        "layer_weight": cfg.layer_weight,
                        "recency_boost": recency,
                        "final_score": final_score,
                    },
                )

        # Deduplicate by normalized content, keep highest final score.
        dedup: dict[str, dict[str, Any]] = {}
        for item in collected:
            key = self._normalize_content(item.get("content", ""))
            if not key:
                continue
            if key not in dedup or item["final_score"] > dedup[key]["final_score"]:
                dedup[key] = item

        ranked = list(dedup.values())
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        ranked = ranked[:top_k]

        return {
            "query": query,
            "strategy": "nuanced_hybrid",
            "results": ranked,
        }

    async def show_layer(self, layer: Layer, limit: int = 200) -> list[dict[str, Any]]:
        """Show stored memories for one layer."""
        cfg = self.layers[layer]
        rows = await self.client.list_memories_raw(
            limit=limit,
            memory_target=cfg.target.name,
        )
        return rows

    async def prune(
        self,
        episodic_ttl_days: int = 21,
        min_score: float = 0.15,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Remove low-value/stale memories and duplicates.

        Rules:
        - episodic: delete if older than `episodic_ttl_days` and score < `min_score`
        - all layers: deduplicate identical normalized content (keep best-scored latest)
        """
        now = datetime.now(tz=timezone.utc)
        plan_delete: list[dict[str, Any]] = []

        for layer, cfg in self.layers.items():
            rows = await self.client.list_memories_raw(limit=5000, memory_target=cfg.target.name)

            # 1) Staleness rule for episodic layer.
            if layer == "episodic":
                for row in rows:
                    score = float(row.get("score", 0.0))
                    ts = self._parse_time(row.get("time_modified") or row.get("time_created") or row.get("message_time"))
                    if ts is None:
                        continue
                    age_days = (now - ts).total_seconds() / 86400.0
                    if age_days > episodic_ttl_days and score < min_score:
                        plan_delete.append(
                            {
                                "memory_id": row["memory_id"],
                                "layer": layer,
                                "reason": f"episodic_stale(age_days={age_days:.1f},score={score:.2f})",
                            },
                        )

            # 2) Duplicate rule for all layers.
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                key = self._normalize_content(row.get("content", ""))
                if not key:
                    continue
                grouped.setdefault(key, []).append(row)

            for key_rows in grouped.values():
                if len(key_rows) <= 1:
                    continue

                def _rank_key(r: dict[str, Any]) -> tuple[float, float]:
                    score = float(r.get("score", 0.0))
                    ts = self._parse_time(r.get("time_modified") or r.get("time_created") or r.get("message_time"))
                    ts_epoch = ts.timestamp() if ts else 0.0
                    return (score, ts_epoch)

                sorted_rows = sorted(key_rows, key=_rank_key, reverse=True)
                keep_id = sorted_rows[0]["memory_id"]
                for row in sorted_rows[1:]:
                    if row["memory_id"] == keep_id:
                        continue
                    plan_delete.append(
                        {
                            "memory_id": row["memory_id"],
                            "layer": layer,
                            "reason": "duplicate_content",
                        },
                    )

        # Deduplicate delete list itself.
        delete_map = {item["memory_id"]: item for item in plan_delete}
        final_delete = list(delete_map.values())

        if not dry_run:
            for item in final_delete:
                await self.client.delete_memory(item["memory_id"])

        return {
            "dry_run": dry_run,
            "delete_count": len(final_delete),
            "deletions": final_delete,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory Manager Agent")
    parser.add_argument("--env-file", default=None, help="Path to .env (optional)")

    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add memory to a layer")
    add_p.add_argument("--layer", choices=["episodic", "semantic", "long_term"], required=True)
    add_p.add_argument("--content", required=True)
    add_p.add_argument("--when-to-use", default="")
    add_p.add_argument("--score", type=float, default=0.7)

    add_short_p = sub.add_parser("add-short-lived", help="Add short-lived episodic memory with TTL")
    add_short_p.add_argument("--content", required=True)
    add_short_p.add_argument("--when-to-use", default="")
    add_short_p.add_argument("--score", type=float, default=0.7)
    add_short_p.add_argument("--ttl-hours", type=int, default=24, help="TTL in hours (default: 24)")

    q_p = sub.add_parser("query", help="Nuanced retrieval across layers")
    q_p.add_argument("--query", required=True)
    q_p.add_argument("--top-k", type=int, default=10)
    q_p.add_argument("--vector-weight", type=float, default=0.7)

    q_short_p = sub.add_parser("query-short-lived", help="Retrieve short-lived episodic memories")
    q_short_p.add_argument("--query", required=True)
    q_short_p.add_argument("--top-k", type=int, default=10)
    q_short_p.add_argument("--vector-weight", type=float, default=0.7)
    q_short_p.add_argument("--include-expired", action="store_true")

    s_p = sub.add_parser("show", help="Show one layer")
    s_p.add_argument("--layer", choices=["episodic", "semantic", "long_term"], required=True)
    s_p.add_argument("--limit", type=int, default=200)

    p_p = sub.add_parser("prune", help="Prune stale/duplicate memories")
    p_p.add_argument("--episodic-ttl-days", type=int, default=21)
    p_p.add_argument("--min-score", type=float, default=0.15)
    p_p.add_argument("--apply", action="store_true", help="Apply deletions (default is dry-run)")

    return parser


async def _main_async(args: argparse.Namespace) -> None:
    agent = MemoryManagerAgent.from_env(env_path=args.env_file)
    await agent.start()
    try:
        if args.command == "add":
            result = await agent.add(
                layer=args.layer,
                content=args.content,
                when_to_use=args.when_to_use,
                score=args.score,
            )
            print(json.dumps({"status": "ok", "memory_id": result.memory_id}, ensure_ascii=False, indent=2))

        elif args.command == "add-short-lived":
            result = await agent.store_short_lived(
                content=args.content,
                when_to_use=args.when_to_use,
                score=args.score,
                ttl_hours=args.ttl_hours,
            )
            print(json.dumps({"status": "ok", "memory_id": result.memory_id}, ensure_ascii=False, indent=2))

        elif args.command == "query":
            result = await agent.query_nuanced(
                query=args.query,
                top_k=args.top_k,
                vector_weight=args.vector_weight,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif args.command == "query-short-lived":
            result = await agent.retrieve_short_lived(
                query=args.query,
                top_k=args.top_k,
                vector_weight=args.vector_weight,
                include_expired=args.include_expired,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif args.command == "show":
            rows = await agent.show_layer(layer=args.layer, limit=args.limit)
            print(json.dumps(rows, ensure_ascii=False, indent=2))

        elif args.command == "prune":
            result = await agent.prune(
                episodic_ttl_days=args.episodic_ttl_days,
                min_score=args.min_score,
                dry_run=not args.apply,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

        else:
            raise ValueError(f"Unknown command: {args.command}")
    finally:
        await agent.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
