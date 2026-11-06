"""Tests for MemoryToolExecutor — each test hits a real SQLite DB."""
from __future__ import annotations
import json
import pytest

from ia_test.api.memory_tools import MemoryToolExecutor
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import ToolCall


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="test-id-001", name=name, arguments=arguments)


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "tools_test.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def executor(db):
    return MemoryToolExecutor(db)


# ── search_memory ─────────────────────────────────────────────────────────

async def test_search_memory_finds_matching_entry(db, executor):
    await db.store(type="knowledge", content="Python is a dynamically typed language", confidence=0.8)

    result = await executor.execute(_call("search_memory", {"query": "Python language"}))
    data = json.loads(result.content)

    assert data["total_found"] >= 1
    assert any("Python" in e["content"] for e in data["entries"])


async def test_search_memory_empty_db_returns_zero(db, executor):
    result = await executor.execute(_call("search_memory", {"query": "anything"}))
    data = json.loads(result.content)
    assert data["total_found"] == 0
    assert data["entries"] == []


async def test_search_memory_no_match_returns_zero(db, executor):
    await db.store(type="knowledge", content="Machine learning concepts", confidence=0.7)

    result = await executor.execute(_call("search_memory", {"query": "quantum physics"}))
    data = json.loads(result.content)
    assert data["total_found"] == 0


async def test_search_memory_respects_limit(db, executor):
    for i in range(10):
        await db.store(type="knowledge", content=f"Python fact number {i}", confidence=0.6)

    result = await executor.execute(_call("search_memory", {"query": "Python fact", "limit": 3}))
    data = json.loads(result.content)
    assert len(data["entries"]) <= 3


# ── get_memory ────────────────────────────────────────────────────────────

async def test_get_memory_returns_full_entry(db, executor):
    entry = await db.store(type="hypothesis", content="Gravity might be emergent", confidence=0.4)

    result = await executor.execute(_call("get_memory", {"entry_id": entry.id}))
    data = json.loads(result.content)

    assert data["id"] == entry.id
    assert data["content"] == "Gravity might be emergent"
    assert data["type"] == "hypothesis"
    assert data["confidence"] == pytest.approx(0.4)


async def test_get_memory_nonexistent_id_returns_error(db, executor):
    result = await executor.execute(_call("get_memory", {"entry_id": "does-not-exist"}))
    data = json.loads(result.content)
    assert "error" in data


# ── explore_relations ─────────────────────────────────────────────────────

async def test_explore_relations_finds_connected_entries(db, executor):
    source = await db.store(type="knowledge", content="A causes B", confidence=0.7)
    target = await db.store(type="knowledge", content="B leads to C", confidence=0.6)
    await db.add_relation(source.id, target.id, relation_type="supports", weight=1.0)

    result = await executor.execute(_call("explore_relations", {"entry_id": source.id}))
    data = json.loads(result.content)

    assert data["relations_count"] >= 1
    assert data["source_entry"]["id"] == source.id
    connected_ids = [r["connected_entry_id"] for r in data["relations"]]
    assert target.id in connected_ids


async def test_explore_relations_no_relations_returns_empty(db, executor):
    entry = await db.store(type="knowledge", content="Isolated idea", confidence=0.5)

    result = await executor.execute(_call("explore_relations", {"entry_id": entry.id}))
    data = json.loads(result.content)

    assert data["relations_count"] == 0
    assert data["relations"] == []


async def test_explore_relations_nonexistent_id_returns_error(db, executor):
    result = await executor.execute(_call("explore_relations", {"entry_id": "ghost-id"}))
    data = json.loads(result.content)
    assert "error" in data


# ── get_recent_experiments ────────────────────────────────────────────────

async def test_get_recent_experiments_returns_exploration_entries(db, executor):
    await db.store(type="knowledge", content="Human added fact", confidence=0.8, provenance="human")
    await db.store(type="knowledge", content="Autonomous discovery", confidence=0.75, provenance="exploration:curiosity")

    result = await executor.execute(_call("get_recent_experiments", {}))
    data = json.loads(result.content)

    assert data["total_found"] >= 1
    provenances = [e["provenance"] for e in data["entries"]]
    assert all("exploration" in p or p == "experiment" for p in provenances)


async def test_get_recent_experiments_excludes_non_exploration(db, executor):
    await db.store(type="knowledge", content="Human fact only", confidence=0.9, provenance="human")

    result = await executor.execute(_call("get_recent_experiments", {}))
    data = json.loads(result.content)
    # Human entries must not appear
    assert data["total_found"] == 0


async def test_unknown_tool_returns_error(db, executor):
    result = await executor.execute(_call("nonexistent_tool", {}))
    data = json.loads(result.content)
    assert "error" in data
    assert "Unknown tool" in data["error"]
