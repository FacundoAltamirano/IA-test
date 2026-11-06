import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from ia_test.experiments.engine import ExperimentEngine
from ia_test.experiments.models import Experiment
from ia_test.core.event_bus import EventBus
from ia_test.core.events import ExperimentDesigned, ExperimentCompleted
from ia_test.core.settings import Settings
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Response, Usage


def _make_provider(payload: dict) -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content=json.dumps(payload),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=60, completion_tokens=80, total_tokens=140),
    ))
    return provider


def _make_settings() -> Settings:
    return Settings(
        model_provider="openai",
        model_name="gpt-4o-mini",
        openai_api_key="test-key",
    )


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "experiment_test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_design_returns_experiment_with_procedure(db):
    provider = _make_provider({
        "procedure": ["Step A: check X", "Step B: verify Y"],
        "expected_outcome": "X and Y are consistent",
    })
    engine = ExperimentEngine(db=db, provider=provider, event_bus=EventBus(), settings=_make_settings())

    experiment = await engine.design("Hypothesis: X implies Y")

    assert experiment.hypothesis == "Hypothesis: X implies Y"
    assert len(experiment.procedure) == 2
    assert experiment.expected_outcome == "X and Y are consistent"
    assert experiment.status == "designed"


async def test_design_publishes_experiment_designed_event(db):
    provider = _make_provider({"procedure": ["step"], "expected_outcome": "outcome"})
    bus = EventBus()
    published = []

    async def capture(e):
        published.append(e)

    bus.subscribe(ExperimentDesigned, capture)
    engine = ExperimentEngine(db=db, provider=provider, event_bus=bus, settings=_make_settings())

    experiment = await engine.design("Some hypothesis")

    designed = [e for e in published if isinstance(e, ExperimentDesigned)]
    assert len(designed) == 1
    assert designed[0].experiment_id == experiment.id


async def test_run_fills_actual_outcome(db):
    provider = _make_provider({
        "step_results": ["observed A", "observed B"],
        "actual_outcome": "X and Y are consistent as expected",
    })
    engine = ExperimentEngine(db=db, provider=provider, event_bus=EventBus(), settings=_make_settings())

    experiment = Experiment(
        hypothesis="X implies Y",
        procedure=["Step A", "Step B"],
        expected_outcome="X and Y are consistent",
    )
    result = await engine.run(experiment)

    assert result.actual_outcome == "X and Y are consistent as expected"
    assert result.status != "failed"


async def test_evaluate_sets_conclusion_supports(db):
    provider = _make_provider({
        "conclusion": "supports",
        "reasoning": "Actual matches expected",
        "confidence_delta": 0.15,
    })
    engine = ExperimentEngine(db=db, provider=provider, event_bus=EventBus(), settings=_make_settings())

    experiment = Experiment(
        hypothesis="X implies Y",
        procedure=["step"],
        expected_outcome="Y follows X",
        actual_outcome="Y does follow X",
    )
    result = await engine.evaluate(experiment)

    assert result.conclusion == "supports"
    assert result.confidence_delta == pytest.approx(0.15)
    assert result.status == "completed"


async def test_evaluate_sets_conclusion_refutes(db):
    provider = _make_provider({
        "conclusion": "refutes",
        "reasoning": "Actual contradicts expected",
        "confidence_delta": -0.2,
    })
    engine = ExperimentEngine(db=db, provider=provider, event_bus=EventBus(), settings=_make_settings())

    experiment = Experiment(
        hypothesis="X implies Y",
        procedure=["step"],
        expected_outcome="Y follows X",
        actual_outcome="Y does NOT follow X",
    )
    result = await engine.evaluate(experiment)

    assert result.conclusion == "refutes"
    assert result.confidence_delta == pytest.approx(-0.2)


async def test_store_creates_experiment_memory_entry(db):
    engine = ExperimentEngine(db=db, provider=MagicMock(), event_bus=EventBus(), settings=_make_settings())

    experiment = Experiment(
        hypothesis="Light travels faster than sound",
        procedure=["step"],
        expected_outcome="sound arrives later",
        actual_outcome="sound arrives later",
        conclusion="supports",
        confidence_delta=0.1,
        status="completed",
    )
    await engine.store(experiment)

    entries = await db.query(type="experiment", limit=10)
    assert len(entries) == 1
    assert "Light travels faster than sound" in entries[0].content
    assert "supports" in entries[0].content


async def test_store_adjusts_source_entry_confidence_on_refutes(db):
    source = await db.store(
        type="hypothesis",
        content="X implies Y",
        confidence=0.7,
    )
    engine = ExperimentEngine(db=db, provider=MagicMock(), event_bus=EventBus(), settings=_make_settings())

    experiment = Experiment(
        hypothesis="X implies Y",
        source_entry_id=source.id,
        procedure=["step"],
        expected_outcome="Y",
        actual_outcome="not Y",
        conclusion="refutes",
        confidence_delta=-0.2,
        status="completed",
    )
    await engine.store(experiment)

    updated = await db.get(source.id)
    assert updated.confidence == pytest.approx(0.5)


async def test_store_increases_confidence_on_supports(db):
    source = await db.store(
        type="hypothesis",
        content="X implies Y",
        confidence=0.6,
    )
    engine = ExperimentEngine(db=db, provider=MagicMock(), event_bus=EventBus(), settings=_make_settings())

    experiment = Experiment(
        hypothesis="X implies Y",
        source_entry_id=source.id,
        procedure=["step"],
        expected_outcome="Y",
        actual_outcome="Y",
        conclusion="supports",
        confidence_delta=0.15,
        status="completed",
    )
    await engine.store(experiment)

    updated = await db.get(source.id)
    assert updated.confidence == pytest.approx(0.75)


async def test_store_publishes_experiment_completed(db):
    bus = EventBus()
    published = []

    async def capture(e):
        published.append(e)

    bus.subscribe(ExperimentCompleted, capture)
    engine = ExperimentEngine(db=db, provider=MagicMock(), event_bus=bus, settings=_make_settings())

    experiment = Experiment(
        hypothesis="Some hypothesis",
        conclusion="inconclusive",
        confidence_delta=0.0,
        status="completed",
    )
    await engine.store(experiment)

    completed = [e for e in published if isinstance(e, ExperimentCompleted)]
    assert len(completed) == 1
    assert completed[0].conclusion == "inconclusive"


async def test_full_lifecycle_end_to_end(db):
    """design → run → evaluate → store with mocked provider per call."""
    call_count = 0
    responses = [
        {"procedure": ["Reason through A", "Check if B follows"], "expected_outcome": "B follows from A"},
        {"step_results": ["A is established", "B does follow"], "actual_outcome": "B follows from A as predicted"},
        {"conclusion": "supports", "reasoning": "Exact match", "confidence_delta": 0.1},
    ]

    async def multi_response(messages, **kwargs):
        nonlocal call_count
        payload = responses[call_count % len(responses)]
        call_count += 1
        return Response(
            content=json.dumps(payload),
            model="gpt-4o-mini",
            usage=Usage(prompt_tokens=50, completion_tokens=80, total_tokens=130),
        )

    provider = MagicMock()
    provider.complete = multi_response

    engine = ExperimentEngine(db=db, provider=provider, event_bus=EventBus(), settings=_make_settings())

    hypothesis = await db.store(type="hypothesis", content="A implies B", confidence=0.5)
    experiment = await engine.design("A implies B", source_entry_id=hypothesis.id)
    experiment = await engine.run(experiment)
    experiment = await engine.evaluate(experiment)
    await engine.store(experiment)

    assert experiment.status == "completed"
    assert experiment.conclusion == "supports"
    updated = await db.get(hypothesis.id)
    assert updated.confidence == pytest.approx(0.6)
    exp_entries = await db.query(type="experiment", limit=5)
    assert len(exp_entries) == 1
