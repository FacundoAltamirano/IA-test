import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from ia_test.curiosity.loop import CuriosityLoop
from ia_test.core.event_bus import EventBus
from ia_test.core.events import QuestionGenerated, ExplorationTriggered
from ia_test.core.settings import Settings
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Response, Usage


def _make_provider(question: str = "What is X?") -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content=json.dumps({"question": question, "source_entry_id": None, "trigger": "isolation"}),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    ))
    return provider


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        model_provider="openai",
        model_name="gpt-4o-mini",
        openai_api_key="test-key",
        max_autonomous_cycles=5,
        max_api_calls_per_hour=100,
        max_api_calls_per_day=1000,
        curiosity_loop_interval_seconds=1,
        curiosity_questions_per_cycle=2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "curiosity_test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_loop_does_not_generate_on_empty_memory(db):
    bus = EventBus()
    loop = CuriosityLoop(
        db=db, provider=_make_provider(),
        event_bus=bus, settings=_make_settings(),
    )

    questions = await loop.run_once()

    assert questions == []
    assert loop.cycles_run == 1


async def test_loop_generates_questions_from_memory(db):
    await db.store(type="knowledge", content="Isolated concept A", confidence=0.5)
    await db.store(type="knowledge", content="Isolated concept B", confidence=0.3)

    bus = EventBus()
    loop = CuriosityLoop(
        db=db, provider=_make_provider("What is concept A?"),
        event_bus=bus, settings=_make_settings(),
    )

    questions = await loop.run_once()

    assert len(questions) > 0
    assert loop.cycles_run == 1


async def test_loop_publishes_question_generated_event(db):
    await db.store(type="knowledge", content="Some isolated fact", confidence=0.5)

    bus = EventBus()
    published_events = []

    async def capture(event): published_events.append(event)
    bus.subscribe(QuestionGenerated, capture)

    loop = CuriosityLoop(
        db=db, provider=_make_provider("Why does X happen?"),
        event_bus=bus, settings=_make_settings(),
    )
    await loop.run_once()

    assert any(isinstance(e, QuestionGenerated) for e in published_events)


async def test_loop_respects_max_autonomous_cycles(db):
    await db.store(type="knowledge", content="Isolated", confidence=0.5)

    bus = EventBus()
    loop = CuriosityLoop(
        db=db, provider=_make_provider(),
        event_bus=bus, settings=_make_settings(max_autonomous_cycles=2),
    )

    for _ in range(3):
        await loop.run_once()

    assert loop.cycles_run == 2


async def test_loop_cycles_increment_correctly(db):
    bus = EventBus()
    loop = CuriosityLoop(
        db=db, provider=_make_provider(),
        event_bus=bus, settings=_make_settings(),
    )

    assert loop.cycles_run == 0
    await loop.run_once()
    assert loop.cycles_run == 1
    await loop.run_once()
    assert loop.cycles_run == 2
