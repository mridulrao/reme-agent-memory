from __future__ import annotations

import argparse
import asyncio
import json
import time

from memory_manager_agent import MemoryManagerAgent


async def run_query(query: str, top_k: int, vector_weight: float, env_file: str | None) -> None:
    agent = MemoryManagerAgent.from_env(env_path=env_file)

    if not agent.client.db_uri:
        raise ValueError("AGENT_MEMORY_DB_URI is not set.")
    if not agent.client.openai_api_key:
        raise ValueError("AGENT_MEMORY_OPENAI_API_KEY or OPENAI_API_KEY is not set.")

    await agent.start()
    try:
        t0 = time.perf_counter()
        result = await agent.query_nuanced(
            query=query,
            top_k=top_k,
            vector_weight=vector_weight,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        response = {
            "query": query,
            "top_k": top_k,
            "returned": len(result.get("results", [])),
            "latency_ms": round(elapsed_ms, 2),
            "results": result.get("results", []),
        }
        print(json.dumps(response, ensure_ascii=False, indent=2))
    finally:
        await agent.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve related memories for a user query")
    parser.add_argument("query", help="User query/message to search relevant memories for")
    parser.add_argument("--top-k", type=int, default=5, help="Max results to return (default: 5)")
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.7,
        help="Hybrid retrieval vector weight (default: 0.7)",
    )
    parser.add_argument("--env-file", default=None, help="Optional .env file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_query(
            query=args.query,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            env_file=args.env_file,
        )
    )


if __name__ == "__main__":
    main()
