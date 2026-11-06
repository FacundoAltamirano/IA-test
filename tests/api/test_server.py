from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from ia_test.api.server import create_app
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Response, Usage


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "api_test.db"))
    await database.connect()
    yield database
    await database.close()


def _make_provider(text: str = "Hello from IA-test!") -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content=text,
        model="ia-test-local",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))
    return provider


@pytest.fixture
async def client(db):
    app = create_app(provider=_make_provider(), db=db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── health ────────────────────────────────────────────────────────────────

async def test_health_returns_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "memory_entries" in data


# ── models ────────────────────────────────────────────────────────────────

async def test_list_models_returns_local_model(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "ia-test-local" in ids


# ── messages (non-streaming) ──────────────────────────────────────────────

async def test_messages_returns_anthropic_shape(client):
    payload = {
        "model": "ia-test-local",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Say hi"}],
    }
    r = await client.post("/v1/messages", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["stop_reason"] == "end_turn"
    assert isinstance(data["content"], list)
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hello from IA-test!"
    assert "usage" in data


async def test_messages_passes_system_prompt_to_provider(db):
    provider = _make_provider()
    app = create_app(provider=provider, db=db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        await c.post("/v1/messages", json={
            "model": "ia-test-local",
            "max_tokens": 64,
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "hi"}],
        })
    call_args = provider.complete.call_args
    messages = call_args[0][0]  # positional first arg
    system_msgs = [m for m in messages if m.role == "system"]
    assert len(system_msgs) == 1
    assert "Be concise." in system_msgs[0].content


async def test_messages_rate_limit_raises_429(db):
    from ia_test.safety.limits import RateLimiter
    limiter = RateLimiter(max_per_hour=0, max_per_day=1000)
    app = create_app(provider=_make_provider(), db=db, rate_limiter=limiter)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/v1/messages", json={
            "model": "ia-test-local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 429


# ── messages (streaming) ──────────────────────────────────────────────────

async def test_messages_stream_returns_sse(db):
    app = create_app(provider=_make_provider("Streaming!"), db=db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/v1/messages", json={
            "model": "ia-test-local",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events = []
    for line in r.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    assert "message_start" in types
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "message_stop" in types

    delta_texts = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
    ]
    assert any("Streaming!" in t for t in delta_texts)


# ── context injection ─────────────────────────────────────────────────────

async def test_context_injection_adds_memory_to_system(db):
    await db.store(
        type="knowledge",
        content="Python is a dynamically typed programming language",
        confidence=0.9,
        provenance="human",
        tags=["python"],
    )

    provider = _make_provider()
    app = create_app(provider=provider, db=db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        await c.post("/v1/messages", json={
            "model": "ia-test-local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Tell me about Python language"}],
        })

    call_args = provider.complete.call_args
    messages = call_args[0][0]
    system_msgs = [m for m in messages if m.role == "system"]
    assert len(system_msgs) == 1
    assert "Memory context" in system_msgs[0].content
    assert "Python" in system_msgs[0].content


async def test_context_injection_no_match_gives_empty_system(db):
    """When memory has nothing relevant, system prompt stays None/empty."""
    provider = _make_provider()
    app = create_app(provider=provider, db=db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        await c.post("/v1/messages", json={
            "model": "ia-test-local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        })

    call_args = provider.complete.call_args
    messages = call_args[0][0]
    system_msgs = [m for m in messages if m.role == "system"]
    # No matching memory + no system in request → no system message
    assert len(system_msgs) == 0
