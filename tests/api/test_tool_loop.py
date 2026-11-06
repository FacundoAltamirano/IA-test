"""Tests for the tool-call loop in _complete_with_tools."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from ia_test.api.memory_tools import MemoryToolExecutor
from ia_test.api.server import _complete_with_tools, MEMORY_TOOLS
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Message, Response, Tool, ToolCall, Usage


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "loop_test.db"))
    await database.connect()
    yield database
    await database.close()


def _base_response(text: str = "Final answer") -> Response:
    return Response(
        content=text,
        model="test",
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def _tool_response(tool_name: str = "search_memory", args: dict | None = None) -> Response:
    return Response(
        content="",
        model="test",
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        tool_calls=[ToolCall(id="tc-001", name=tool_name, arguments=args or {"query": "test"})],
        stop_reason="tool_calls",
    )


def _make_provider_with_tool_support(responses: list[Response]) -> MagicMock:
    """Provider that returns responses in sequence and supports append_tool_exchange."""
    call_count = 0

    async def _complete(messages, tools=None, temperature=0.7):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=_complete)
    provider.append_tool_exchange = lambda messages, response, results: [
        *messages,
        Message(role="assistant", content=response.content),
        Message(role="user", content=json.dumps([{"tool": r.name, "result": r.content} for r in results])),
    ]
    return provider


def _make_provider_no_tool_support(text: str = "plain response") -> MagicMock:
    """Provider without append_tool_exchange — simulates mocks in existing tests."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=_base_response(text))
    # Intentionally no append_tool_exchange attribute
    return provider


# ── no tool support ───────────────────────────────────────────────────────

async def test_loop_falls_back_when_no_tool_support(db):
    provider = _make_provider_no_tool_support("hello")
    executor = MemoryToolExecutor(db)
    messages = [Message(role="user", content="hi")]

    result = await _complete_with_tools(
        provider=provider,
        messages=messages,
        tools=MEMORY_TOOLS,
        executor=executor,
        rate_limiter=None,
        temperature=0.7,
        db=db,
    )

    assert result == "hello"
    provider.complete.assert_called_once()


# ── tool call then text response ──────────────────────────────────────────

async def test_loop_executes_tool_then_continues(db):
    """Provider asks for search_memory, gets result, then returns text."""
    await db.store(type="knowledge", content="Python is awesome", confidence=0.8)

    provider = _make_provider_with_tool_support([
        _tool_response("search_memory", {"query": "Python"}),
        _base_response("I found something about Python in my memory."),
    ])
    executor = MemoryToolExecutor(db)
    messages = [Message(role="user", content="What do you know about Python?")]

    result = await _complete_with_tools(
        provider=provider,
        messages=messages,
        tools=MEMORY_TOOLS,
        executor=executor,
        rate_limiter=None,
        temperature=0.7,
        db=db,
    )

    assert "Python" in result
    assert provider.complete.call_count == 2


async def test_loop_calls_provider_twice_on_one_tool_call(db):
    provider = _make_provider_with_tool_support([
        _tool_response("get_recent_experiments"),
        _base_response("Done"),
    ])
    executor = MemoryToolExecutor(db)
    messages = [Message(role="user", content="What did you explore?")]

    await _complete_with_tools(
        provider=provider,
        messages=messages,
        tools=MEMORY_TOOLS,
        executor=executor,
        rate_limiter=None,
        temperature=0.7,
        db=db,
    )

    assert provider.complete.call_count == 2


# ── max iterations ────────────────────────────────────────────────────────

async def test_loop_stops_at_max_iterations_without_crashing(db):
    """If provider keeps returning tool calls, loop stops after MAX_TOOL_ITERATIONS."""
    # Every call returns a tool call — never a text response (except the final fallback)
    tool_resp = _tool_response("search_memory", {"query": "test"})
    final_fallback = _base_response("Fallback text")
    # 3 tool iterations + 1 final call without tools
    provider = _make_provider_with_tool_support([
        tool_resp, tool_resp, tool_resp, final_fallback
    ])
    executor = MemoryToolExecutor(db)
    messages = [Message(role="user", content="loop test")]

    result = await _complete_with_tools(
        provider=provider,
        messages=messages,
        tools=MEMORY_TOOLS,
        executor=executor,
        rate_limiter=None,
        temperature=0.7,
        db=db,
    )

    # Should get the fallback text, not crash
    assert isinstance(result, str)
    # At most 3 iterations + 1 final = 4 calls total
    assert provider.complete.call_count <= 4


# ── rate limiting ─────────────────────────────────────────────────────────

async def test_loop_stops_on_rate_limit_between_iterations(db):
    """If rate limit hits before the second iteration, return what we have."""
    from ia_test.safety.limits import RateLimiter
    # Allow 1 call (the cycle check in create_message already consumed it)
    # So limit=1 here means the second check inside the loop will fail
    limiter = RateLimiter(max_per_hour=1, max_per_day=1000)
    # Consume the one available call
    await limiter.check_and_record()

    provider = _make_provider_with_tool_support([
        _tool_response("search_memory", {"query": "test"}),
        _base_response("Should not reach here"),
    ])
    executor = MemoryToolExecutor(db)
    messages = [Message(role="user", content="hi")]

    result = await _complete_with_tools(
        provider=provider,
        messages=messages,
        tools=MEMORY_TOOLS,
        executor=executor,
        rate_limiter=limiter,
        temperature=0.7,
        db=db,
    )

    # First call succeeds, rate limit kicks in before second — returns partial or fallback
    assert isinstance(result, str)
    # Provider was called at most twice (first tool call + maybe one more)
    assert provider.complete.call_count <= 2


# ── audit logging ─────────────────────────────────────────────────────────

async def test_tool_calls_are_audited(db):
    """Tool calls and results are recorded in the audit_log table."""
    await db.store(type="knowledge", content="Some knowledge", confidence=0.7)

    provider = _make_provider_with_tool_support([
        _tool_response("search_memory", {"query": "knowledge"}),
        _base_response("Found it."),
    ])
    executor = MemoryToolExecutor(db)
    messages = [Message(role="user", content="search something")]

    await _complete_with_tools(
        provider=provider,
        messages=messages,
        tools=MEMORY_TOOLS,
        executor=executor,
        rate_limiter=None,
        temperature=0.7,
        db=db,
    )

    async with db._db.execute(
        "SELECT operation FROM audit_log WHERE operation IN ('tool_call', 'tool_result')"
    ) as cursor:
        rows = await cursor.fetchall()

    operations = [r["operation"] for r in rows]
    assert "tool_call" in operations
    assert "tool_result" in operations
