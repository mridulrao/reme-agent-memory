from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory_manager_agent import MemoryManagerAgent

Layer = Literal["episodic", "semantic", "long_term"]


class StoreMemoryRequest(BaseModel):
    layer: Layer = Field(description="Memory layer/type to store in.")
    content: str = Field(min_length=1, description="Memory content.")
    when_to_use: str = Field(default="", description="Optional retrieval hint.")
    score: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryMemoryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)


class StoreShortLivedRequest(BaseModel):
    content: str = Field(min_length=1)
    when_to_use: str = Field(default="")
    score: float = Field(default=0.7, ge=0.0, le=1.0)
    ttl_hours: int = Field(default=24, ge=0, le=24 * 365)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryShortLivedRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    include_expired: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = MemoryManagerAgent.from_env()
    await agent.start()
    app.state.memory_agent = agent
    try:
        yield
    finally:
        await agent.close()


app = FastAPI(title="Memory API", version="1.0.0", lifespan=lifespan)


def _agent_from_app() -> MemoryManagerAgent:
    agent = getattr(app.state, "memory_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Memory agent is not initialized.")
    return agent


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/memories")
async def store_memory(payload: StoreMemoryRequest) -> dict[str, str]:
    agent = _agent_from_app()
    try:
        node = await agent.add(
            layer=payload.layer,
            content=payload.content,
            when_to_use=payload.when_to_use,
            score=payload.score,
            **payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "memory_id": node.memory_id}


@app.post("/memories/query")
async def retrieve_memories(payload: QueryMemoryRequest) -> dict[str, Any]:
    agent = _agent_from_app()
    try:
        return await agent.query_nuanced(
            query=payload.query,
            top_k=payload.top_k,
            vector_weight=payload.vector_weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/memories/short-lived")
async def store_short_lived_memory(payload: StoreShortLivedRequest) -> dict[str, str]:
    agent = _agent_from_app()
    try:
        node = await agent.store_short_lived(
            content=payload.content,
            when_to_use=payload.when_to_use,
            score=payload.score,
            ttl_hours=payload.ttl_hours,
            **payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "memory_id": node.memory_id}


@app.post("/memories/short-lived/query")
async def retrieve_short_lived_memories(payload: QueryShortLivedRequest) -> dict[str, Any]:
    agent = _agent_from_app()
    try:
        return await agent.retrieve_short_lived(
            query=payload.query,
            top_k=payload.top_k,
            vector_weight=payload.vector_weight,
            include_expired=payload.include_expired,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
