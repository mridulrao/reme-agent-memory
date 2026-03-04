"""Show current memories stored in pgvector-backed ReMe memory.

Usage:
  python show_memory.py \
    --db-uri postgresql://user:password@host:5432/dbname \
    --openai-api-key sk-... \
    [--limit 100] [--memory-type personal] [--memory-target mridul]
"""

from __future__ import annotations

import argparse
import asyncio
import json

from agent_memory_client import AgentMemoryClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show current stored memories")
    parser.add_argument("--env-file", default=None, help="Path to .env file (optional)")
    parser.add_argument("--db-uri", default=None, help="PostgreSQL connection URI")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--memory-type", default=None, help="identity/personal/procedural/tool/summary/history")
    parser.add_argument("--memory-target", default=None, help="Target name, e.g. user/task/tool name")
    return parser


async def main_async(args: argparse.Namespace) -> None:
    client = AgentMemoryClient.from_env(
        env_path=args.env_file,
        openai_api_key=args.openai_api_key,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        db_uri=args.db_uri,
        vector_backend="pgvector",
    )
    if not client.db_uri:
        raise ValueError(
            "Missing db URI. Set AGENT_MEMORY_DB_URI (or DATABASE_URL) in .env, or pass --db-uri.",
        )

    await client.start()
    try:
        rows = await client.list_memories_raw(
            limit=args.limit,
            memory_type=args.memory_type or None,
            memory_target=args.memory_target or None,
        )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    finally:
        await client.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
