import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from ia_test.exploration.engine import ExplorationEngine
from ia_test.core.event_bus import EventBus
from ia_test.core.events import ExplorationStarted, ExplorationCompleted, DiscoveryRecorded
from ia_test.core.settings import Settings
from ia_test.memory.database import MemoryDatabase
from ia_test.curiosity.models import Question
from ia_test.providers.base import Response, Usage


def _make_provider(findings: list[dict] | None = None) -> MagicMock:
    provider = MagicMock()
    payload = {"findings": findings or [], "search_queries": []}
    provider.complete = AsyncMock(return_value=Response(
        content=json.dumps(payload),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=50, completion_tokens=80, total_tokens=130),
    ))
    return provider


def _make_settings() -> Settings:
    return Settings(
        model_provider="openai",
        model_name="gpt-4o-mini",
        openai_api_key="test-key",
    )


def _make_question(content: str = "Why is the sky blue?") -> Question:
    return Question(content=content, priority_score=0.8, trigger="isolation")


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "engine_test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_engine_explore_stores_discovery_in_memory(db):
    findings = [{"content": "Sky appears blue due to Rayleigh scattering", "confidence": 0.9, "source": "synthesis", "tags": ["physics"]}]
    engine = ExplorationEngine(db=db, provider=_make_provider(findings), event_bus=EventBus(), settings=_make_settings())
    question = _make_question()

    result = await engine.explore(question)

    assert result.status == "completed"
    assert len(result.discoveries) == 1
    entries = await db.query(limit=10)
    assert any("Rayleigh scattering" in e.content for e in entries)


async def test_engine_publishes_exploration_started(db):
    bus = EventBus()
    published = []

    async def capture(e):
        published.append(e)

    bus.subscribe(ExplorationStarted, capture)
    engine = ExplorationEngine(db=db, provider=_make_provider(), event_bus=bus, settings=_make_settings())

    await engine.explore(_make_question())

    started = [e for e in published if isinstance(e, ExplorationStarted)]
    assert len(started) == 1
    assert started[0].content == "Why is the sky blue?"


async def test_engine_publishes_exploration_completed(db):
    bus = EventBus()
    published = []

    async def capture(e):
        published.append(e)

    bus.subscribe(ExplorationCompleted, capture)
    engine = ExplorationEngine(db=db, provider=_make_provider(), event_bus=bus, settings=_make_settings())

    await engine.explore(_make_question())

    completed = [e for e in published if isinstance(e, ExplorationCompleted)]
    assert len(completed) == 1


async def test_engine_publishes_discovery_recorded_per_finding(db):
    findings = [
        {"content": "Finding A about light", "confidence": 0.8, "source": "synthesis", "tags": []},
        {"content": "Finding B about optics", "confidence": 0.7, "source": "synthesis", "tags": []},
    ]
    bus = EventBus()
    published = []

    async def capture(e):
        published.append(e)

    bus.subscribe(DiscoveryRecorded, capture)
    engine = ExplorationEngine(db=db, provider=_make_provider(findings), event_bus=bus, settings=_make_settings())

    await engine.explore(_make_question())

    recorded = [e for e in published if isinstance(e, DiscoveryRecorded)]
    assert len(recorded) == 2


async def test_engine_records_error_entry_on_provider_failure(db):
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("API error"))
    engine = ExplorationEngine(db=db, provider=provider, event_bus=EventBus(), settings=_make_settings())

    result = await engine.explore(_make_question())

    assert result.status == "failed"
    assert result.error == "API error"
    error_entries = await db.query(type="error", limit=10)
    assert len(error_entries) == 1


async def test_engine_returns_partial_on_empty_findings(db):
    engine = ExplorationEngine(db=db, provider=_make_provider([]), event_bus=EventBus(), settings=_make_settings())

    result = await engine.explore(_make_question())

    assert result.status == "partial"
    assert result.discoveries == []


async def test_engine_handles_malformed_llm_response(db):
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content="This is not JSON",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))
    engine = ExplorationEngine(db=db, provider=provider, event_bus=EventBus(), settings=_make_settings())

    result = await engine.explore(_make_question())

    assert result.status == "failed"
    assert result.error is not None
