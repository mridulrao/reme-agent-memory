from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from memory_manager_agent import MemoryManagerAgent


async def seed_memories(
    payload_path: Path,
    env_path: str | None,
    batch_tag: str,
    stop_on_error: bool,
) -> int:
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    memories = data.get("memories")
    if not isinstance(memories, list) or not memories:
        raise ValueError("Payload must include a non-empty 'memories' array")

    agent = MemoryManagerAgent.from_env(env_path=env_path)

    if not agent.client.db_uri:
        raise ValueError("AGENT_MEMORY_DB_URI is not set.")
    if not agent.client.openai_api_key:
        raise ValueError("AGENT_MEMORY_OPENAI_API_KEY or OPENAI_API_KEY is not set.")

    added = 0
    failed = 0
    created_ids: list[str] = []
    latency_ms: list[float] = []
    layer_counts: dict[str, int] = defaultdict(int)
    layer_latency_ms: dict[str, list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []

    await agent.start()
    try:
        for idx, mem in enumerate(memories, start=1):
            layer = mem.get("layer")
            content = mem.get("content", "")
            when_to_use = mem.get("when_to_use", "")
            score = float(mem.get("score", 0.7))

            if layer not in {"episodic", "semantic", "long_term"}:
                raise ValueError(f"Invalid layer at index {idx}: {layer}")
            if not content:
                raise ValueError(f"Missing content at index {idx}")

            t0 = time.perf_counter()
            try:
                node = await agent.add(
                    layer=layer,
                    content=content,
                    when_to_use=when_to_use,
                    score=score,
                    seed_batch=batch_tag,
                    seed_index=idx,
                    seed_file=str(payload_path),
                )
                elapsed = (time.perf_counter() - t0) * 1000
                created_ids.append(node.memory_id)
                latency_ms.append(elapsed)
                layer_counts[layer] += 1
                layer_latency_ms[layer].append(elapsed)
                added += 1
                print(
                    f"[{idx}/{len(memories)}] inserted layer={layer} "
                    f"id={node.memory_id} latency_ms={elapsed:.2f}"
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = (time.perf_counter() - t0) * 1000
                failed += 1
                errors.append(
                    {
                        "index": idx,
                        "layer": layer,
                        "latency_ms": round(elapsed, 2),
                        "error": str(exc),
                    }
                )
                print(f"[{idx}/{len(memories)}] FAILED layer={layer} latency_ms={elapsed:.2f} error={exc}")
                if stop_on_error:
                    break

    finally:
        await agent.close()

    avg_ms = (sum(latency_ms) / len(latency_ms)) if latency_ms else 0.0
    print("\n=== Seed Summary ===")
    print(f"payload={payload_path}")
    print(f"batch_tag={batch_tag}")
    print(f"attempted={added + failed} inserted={added} failed={failed}")
    print(f"avg_insert_latency_ms={avg_ms:.2f}")

    for layer in ("episodic", "semantic", "long_term"):
        count = layer_counts[layer]
        vals = layer_latency_ms[layer]
        layer_avg = (sum(vals) / len(vals)) if vals else 0.0
        print(f"layer={layer} inserted={count} avg_latency_ms={layer_avg:.2f}")

    print("\ncreated_memory_ids=")
    print(json.dumps(created_ids, indent=2))

    if errors:
        print("\nerrors=")
        print(json.dumps(errors, indent=2))

    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk insert memories into layered memory manager")
    parser.add_argument(
        "--payload",
        default="memories_seed.json",
        help="Path to JSON payload with top-level {'memories': [...]}.",
    )
    parser.add_argument("--env-path", default=None, help="Optional .env path")
    parser.add_argument(
        "--batch-tag",
        default=f"seed-{int(time.time())}",
        help="Metadata tag to group this seed run.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at first failed insert instead of continuing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = asyncio.run(
        seed_memories(
            payload_path=Path(args.payload),
            env_path=args.env_path,
            batch_tag=args.batch_tag,
            stop_on_error=args.stop_on_error,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
