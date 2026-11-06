import pytest
from ia_test.memory.database import MemoryDatabase
from ia_test.core.event_bus import EventBus
from ia_test.core.events import EvaluationCompleted
from ia_test.evaluation.engine import EvaluationEngine
from ia_test.evaluation.models import MemorySnapshot


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "eval_engine_test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_snapshot_reflects_current_state(db):
    await db.store(type="knowledge", content="Fact A", confidence=0.8)
    await db.store(type="knowledge", content="Fact B", confidence=0.6)

    engine = EvaluationEngine(db=db, event_bus=EventBus())
    snap = await engine.snapshot()

    assert snap.total_entries == 2
    assert snap.avg_confidence == pytest.approx(0.7)
    assert snap.contradiction_count == 0


async def test_evaluate_learning_improved_verdict(db):
    engine = EvaluationEngine(db=db, event_bus=EventBus())

    before = MemorySnapshot(total_entries=2, avg_confidence=0.6, contradiction_count=0, isolated_count=0)
    after = MemorySnapshot(total_entries=5, avg_confidence=0.65, contradiction_count=0, isolated_count=0)

    evaluation = engine.evaluate_learning(before, after)

    assert evaluation.verdict == "improved"
    assert evaluation.knowledge_gained == 3
    assert evaluation.contradictions_introduced == 0


async def test_evaluate_learning_regressed_on_contradictions(db):
    engine = EvaluationEngine(db=db, event_bus=EventBus())

    before = MemorySnapshot(total_entries=5, avg_confidence=0.7, contradiction_count=0, isolated_count=0)
    after = MemorySnapshot(total_entries=8, avg_confidence=0.65, contradiction_count=4, isolated_count=0)

    evaluation = engine.evaluate_learning(before, after)

    assert evaluation.verdict == "regressed"
    assert evaluation.contradictions_introduced == 4


async def test_evaluate_learning_neutral_verdict(db):
    engine = EvaluationEngine(db=db, event_bus=EventBus())

    before = MemorySnapshot(total_entries=5, avg_confidence=0.6, contradiction_count=1, isolated_count=0)
    after = MemorySnapshot(total_entries=6, avg_confidence=0.59, contradiction_count=2, isolated_count=0)

    evaluation = engine.evaluate_learning(before, after)

    assert evaluation.verdict == "neutral"


async def test_export_metrics_includes_snapshot(db):
    engine = EvaluationEngine(db=db, event_bus=EventBus())
    snap = await engine.snapshot()

    metrics = engine.export_metrics(snap)

    assert "snapshot" in metrics
    assert metrics["snapshot"]["total_entries"] == 0
    assert "taken_at" in metrics["snapshot"]


async def test_export_metrics_includes_evaluation_when_provided(db):
    engine = EvaluationEngine(db=db, event_bus=EventBus())

    before = MemorySnapshot(total_entries=0, avg_confidence=0.0, contradiction_count=0, isolated_count=0)
    after = MemorySnapshot(total_entries=3, avg_confidence=0.7, contradiction_count=0, isolated_count=0)
    evaluation = engine.evaluate_learning(before, after)
    snap = await engine.snapshot()

    metrics = engine.export_metrics(snap, evaluation)

    assert "evaluation" in metrics
    assert metrics["evaluation"]["verdict"] == "improved"
    assert metrics["evaluation"]["knowledge_gained"] == 3


async def test_publish_evaluation_emits_event(db):
    bus = EventBus()
    published = []

    async def capture(e):
        published.append(e)

    bus.subscribe(EvaluationCompleted, capture)
    engine = EvaluationEngine(db=db, event_bus=bus)

    before = MemorySnapshot(total_entries=0, avg_confidence=0.0, contradiction_count=0, isolated_count=0)
    after = MemorySnapshot(total_entries=2, avg_confidence=0.7, contradiction_count=0, isolated_count=0)
    evaluation = engine.evaluate_learning(before, after)
    await engine.publish_evaluation(evaluation)

    completed = [e for e in published if isinstance(e, EvaluationCompleted)]
    assert len(completed) == 1
    assert completed[0].verdict == "improved"
    assert completed[0].knowledge_gained == 2


async def test_full_snapshot_before_after_cycle(db):
    engine = EvaluationEngine(db=db, event_bus=EventBus())
    before = await engine.snapshot()

    await db.store(type="knowledge", content="New discovery about topic X", confidence=0.75)
    await db.store(type="knowledge", content="Another fact about topic X domain", confidence=0.8)

    after = await engine.snapshot()
    evaluation = engine.evaluate_learning(before, after)

    assert evaluation.knowledge_gained == 2
    assert evaluation.verdict == "improved"
