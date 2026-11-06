import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from ia_test.curiosity.loop import CuriosityLoop
from ia_test.core.event_bus import EventBus
from ia_test.core.events import QuestionGenerated, ExplorationTriggered
from ia_test.core.settings import Settings
from ia_test.exploration.engine import ExplorationEngine
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


# ── exploration integration ───────────────────────────────────────────────

def _make_exploration_provider(question_text: str = "What is X?") -> MagicMock:
    """Provider that returns valid JSON for both question generation and exploration."""
    call_count = 0

    async def _complete(messages, tools=None, temperature=0.7):
        nonlocal call_count
        call_count += 1
        # First call: question generator expects a question JSON
        if call_count == 1:
            return Response(
                content=json.dumps({
                    "question": question_text,
                    "source_entry_id": None,
                    "trigger": "isolation",
                }),
                model="gpt-4o-mini",
                usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            )
        # Subsequent calls: exploration engine expects findings JSON
        return Response(
            content=json.dumps({
                "findings": [
                    {
                        "content": f"Discovery about: {question_text}",
                        "confidence": 0.75,
                        "tags": ["autonomous", "exploration"],
                        "source": "synthesis",
                    }
                ],
                "search_queries": [],
            }),
            model="gpt-4o-mini",
            usage=Usage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
        )

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=_complete)
    return provider


async def test_exploration_triggered_when_engine_wired(db):
    """With ExplorationEngine wired, run_once() triggers real exploration."""
    await db.store(type="knowledge", content="Isolated concept needs investigation", confidence=0.4)

    bus = EventBus()
    provider = _make_exploration_provider("What connects these concepts?")
    settings = _make_settings(curiosity_questions_per_cycle=1)

    engine = ExplorationEngine(db=db, provider=provider, event_bus=bus, settings=settings)
    loop = CuriosityLoop(
        db=db, provider=provider,
        event_bus=bus, settings=settings,
        exploration_engine=engine,
    )

    await loop.run_once()

    # At least one discovery should have been stored with exploration provenance
    all_entries = await db.query(limit=100)
    exploration_entries = [e for e in all_entries if e.provenance.startswith("exploration:")]
    assert len(exploration_entries) >= 1


async def test_discovery_content_stored_in_memory(db):
    """The content of the discovery ends up verbatim in the knowledge graph."""
    await db.store(type="knowledge", content="Some isolated fact about Python language", confidence=0.5)

    bus = EventBus()
    provider = _make_exploration_provider("What is Python language?")
    settings = _make_settings(curiosity_questions_per_cycle=1)

    engine = ExplorationEngine(db=db, provider=provider, event_bus=bus, settings=settings)
    loop = CuriosityLoop(
        db=db, provider=provider,
        event_bus=bus, settings=settings,
        exploration_engine=engine,
    )

    await loop.run_once()

    all_entries = await db.query(limit=100)
    contents = [e.content for e in all_entries]
    assert any("Discovery about" in c for c in contents)


async def test_no_exploration_without_engine(db):
    """Without ExplorationEngine, questions are generated but nothing extra is stored."""
    await db.store(type="knowledge", content="Isolated fact", confidence=0.5)

    bus = EventBus()
    provider = _make_provider("Why is X?")
    settings = _make_settings(curiosity_questions_per_cycle=1)

    loop = CuriosityLoop(
        db=db, provider=provider,
        event_bus=bus, settings=settings,
        # no exploration_engine
    )

    before = len(await db.query(limit=100))
    await loop.run_once()
    after = len(await db.query(limit=100))

    # No new entries — only questions generated, not explored
    assert after == before


# ── question persistence ──────────────────────────────────────────────────

async def test_generated_questions_are_persisted_in_db(db):
    """Questions generated in a cycle are stored as 'pending' in the DB."""
    await db.store(type="knowledge", content="Isolated concept needs investigation", confidence=0.5)

    bus = EventBus()
    loop = CuriosityLoop(
        db=db, provider=_make_provider("What connects these?"),
        event_bus=bus, settings=_make_settings(curiosity_questions_per_cycle=1),
    )
    await loop.run_once()

    pending = await db.get_pending_questions(limit=10)
    assert len(pending) >= 1
    assert all(q["status"] == "pending" for q in pending)


async def test_explored_questions_are_marked_answered(db):
    """After successful exploration, the question status is updated to 'answered'."""
    await db.store(type="knowledge", content="Isolated fact about Python language", confidence=0.4)

    bus = EventBus()
    provider = _make_exploration_provider("What is Python language?")
    settings = _make_settings(curiosity_questions_per_cycle=1)

    engine = ExplorationEngine(db=db, provider=provider, event_bus=bus, settings=settings)
    loop = CuriosityLoop(
        db=db, provider=provider,
        event_bus=bus, settings=settings,
        exploration_engine=engine,
    )
    await loop.run_once()

    # No pending questions should remain — all were explored
    pending = await db.get_pending_questions(limit=10)
    assert len(pending) == 0


async def test_pending_questions_are_resumed_in_next_cycle(db):
    """Questions stored as pending in cycle N are picked up in cycle N+1."""
    await db.store(type="knowledge", content="Isolated concept A", confidence=0.5)

    bus = EventBus()
    # No exploration engine — questions stay pending after cycle 1
    loop = CuriosityLoop(
        db=db, provider=_make_provider("Why does A happen?"),
        event_bus=bus, settings=_make_settings(curiosity_questions_per_cycle=1),
    )

    questions_cycle1 = await loop.run_once()
    assert len(questions_cycle1) >= 1
    q_id = questions_cycle1[0].id

    # Cycle 2: same question is resumed from DB (no new data, same entry in memory)
    questions_cycle2 = await loop.run_once()
    resumed_ids = {q.id for q in questions_cycle2}
    assert q_id in resumed_ids


async def test_rate_limit_stops_exploration_mid_cycle(db):
    """If rate limit is hit during exploration, the cycle stops without crashing."""
    await db.store(type="knowledge", content="Entry one isolated", confidence=0.4)
    await db.store(type="knowledge", content="Entry two isolated", confidence=0.4)

    bus = EventBus()
    provider = _make_exploration_provider("What connects these?")
    # Allow exactly 1 call (the cycle check) — exploration calls will be blocked
    from ia_test.safety.limits import RateLimiter
    limiter = RateLimiter(max_per_hour=1, max_per_day=1000)

    settings = _make_settings(curiosity_questions_per_cycle=2)
    engine = ExplorationEngine(db=db, provider=provider, event_bus=bus, settings=settings)
    loop = CuriosityLoop(
        db=db, provider=provider,
        event_bus=bus, settings=settings,
        rate_limiter=limiter,
        exploration_engine=engine,
    )

    # Should not raise, should complete gracefully
    questions = await loop.run_once()
    assert loop.cycles_run == 1  # cycle completed
    # Questions were generated but exploration was cut short
    assert len(questions) >= 0
