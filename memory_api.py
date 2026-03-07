from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from memory_manager_agent import MemoryManagerAgent

Layer = Literal["episodic", "semantic", "long_term"]

_RESERVED_MEMORY_METADATA_KEYS = {"layer", "content", "when_to_use", "score"}
_RESERVED_SHORT_LIVED_METADATA_KEYS = {"content", "when_to_use", "score", "ttl_hours"}


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


class _AuthState:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._nonce_expiry_by_key: dict[str, dict[str, float]] = defaultdict(dict)
        self._request_times_by_key: dict[str, deque[float]] = defaultdict(deque)

    async def verify_and_track(self, request: Request) -> None:
        secret = os.getenv("API_SECRET")
        if not secret:
            raise HTTPException(status_code=503, detail="API auth is not configured.")

        key_id = request.headers.get("X-Key-Id", "").strip()
        timestamp = request.headers.get("X-Timestamp", "").strip()
        nonce = request.headers.get("X-Nonce", "").strip()
        signature = request.headers.get("X-Signature", "").strip()

        if not key_id:
            raise HTTPException(status_code=401, detail="Missing X-Key-Id header.")
        if not timestamp:
            raise HTTPException(status_code=401, detail="Missing X-Timestamp header.")
        if not nonce:
            raise HTTPException(status_code=401, detail="Missing X-Nonce header.")
        if not signature:
            raise HTTPException(status_code=401, detail="Missing X-Signature header.")

        try:
            ts_value = int(timestamp)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail="Invalid X-Timestamp header.") from exc

        now = time.time()
        max_skew = int(os.getenv("AUTH_MAX_SKEW_SECONDS", "300"))
        if abs(now - ts_value) > max_skew:
            raise HTTPException(status_code=401, detail="Request timestamp is outside allowed skew.")

        raw_body = await request.body()
        body_hash = hashlib.sha256(raw_body).hexdigest()

        # Canonical signing string required by this API:
        # METHOD + PATH + TIMESTAMP + NONCE + sha256(raw_body)
        signing_payload = f"{request.method.upper()}\n{request.url.path}\n{timestamp}\n{nonce}\n{body_hash}"
        expected = hmac.new(secret.encode("utf-8"), signing_payload.encode("utf-8"), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected, signature):
            raise HTTPException(status_code=401, detail="Invalid request signature.")

        nonce_ttl = int(os.getenv("AUTH_NONCE_TTL_SECONDS", "600"))
        rate_limit_per_min = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

        async with self._lock:
            self._evict_expired_nonces(now)
            self._evict_old_requests(now)

            key_nonces = self._nonce_expiry_by_key[key_id]
            if nonce in key_nonces:
                raise HTTPException(status_code=401, detail="Nonce has already been used.")
            key_nonces[nonce] = now + nonce_ttl

            key_requests = self._request_times_by_key[key_id]
            if len(key_requests) >= rate_limit_per_min:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for key.")
            key_requests.append(now)

    def _evict_expired_nonces(self, now: float) -> None:
        for key_id in list(self._nonce_expiry_by_key.keys()):
            nonces = self._nonce_expiry_by_key[key_id]
            expired = [nonce for nonce, expiry in nonces.items() if expiry <= now]
            for nonce in expired:
                del nonces[nonce]
            if not nonces:
                del self._nonce_expiry_by_key[key_id]

    def _evict_old_requests(self, now: float) -> None:
        window_start = now - 60.0
        for key_id in list(self._request_times_by_key.keys()):
            requests = self._request_times_by_key[key_id]
            while requests and requests[0] <= window_start:
                requests.popleft()
            if not requests:
                del self._request_times_by_key[key_id]

    async def reset(self) -> None:
        async with self._lock:
            self._nonce_expiry_by_key.clear()
            self._request_times_by_key.clear()


_auth_state = _AuthState()


async def require_auth(request: Request) -> None:
    await _auth_state.verify_and_track(request)


def _validate_metadata_keys(metadata: dict[str, Any], reserved: set[str]) -> None:
    overlap = sorted(reserved.intersection(metadata.keys()))
    if overlap:
        joined = ", ".join(overlap)
        raise HTTPException(status_code=400, detail=f"metadata contains reserved keys: {joined}")


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


def _agent_from_app(request: Request) -> MemoryManagerAgent:
    agent = getattr(request.app.state, "memory_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Memory agent is not initialized.")
    return agent


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    agent = getattr(request.app.state, "memory_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Memory agent is not initialized.")
    return {"status": "ok"}


@app.post("/memories")
async def store_memory(request: Request, payload: StoreMemoryRequest) -> dict[str, str]:
    await require_auth(request)
    _validate_metadata_keys(payload.metadata, _RESERVED_MEMORY_METADATA_KEYS)
    agent = _agent_from_app(request)
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
async def retrieve_memories(request: Request, payload: QueryMemoryRequest) -> dict[str, Any]:
    await require_auth(request)
    agent = _agent_from_app(request)
    try:
        return await agent.query_nuanced(
            query=payload.query,
            top_k=payload.top_k,
            vector_weight=payload.vector_weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/memories/short-lived")
async def store_short_lived_memory(request: Request, payload: StoreShortLivedRequest) -> dict[str, str]:
    await require_auth(request)
    _validate_metadata_keys(payload.metadata, _RESERVED_SHORT_LIVED_METADATA_KEYS)
    agent = _agent_from_app(request)
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
async def retrieve_short_lived_memories(request: Request, payload: QueryShortLivedRequest) -> dict[str, Any]:
    await require_auth(request)
    agent = _agent_from_app(request)
    try:
        return await agent.retrieve_short_lived(
            query=payload.query,
            top_k=payload.top_k,
            vector_weight=payload.vector_weight,
            include_expired=payload.include_expired,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
