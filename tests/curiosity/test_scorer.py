import pytest
from ia_test.memory.database import MemoryDatabase
from ia_test.curiosity.scorer import UncertaintyScorer, _overall_uncertainty


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_scorer_returns_zero_uncertainty_on_empty_db(db):
    scorer = UncertaintyScorer(db)
    report = await scorer.compute()

    assert report.total_entries == 0
    assert report.overall_uncertainty == 0.0
    assert report.isolated_entries == []
    assert report.contradictions == []


async def test_scorer_detects_isolated_entries(db):
    await db.store(type="knowledge", content="Orphan fact", confidence=0.5)

    scorer = UncertaintyScorer(db)
    report = await scorer.compute()

    assert len(report.isolated_entries) == 1
    assert report.isolated_entries[0]["content"] == "Orphan fact"
    assert report.overall_uncertainty > 0.0


async def test_scorer_detects_contradictions(db):
    a = await db.store(type="knowledge", content="A is true", confidence=0.9)
    b = await db.store(type="knowledge", content="A is false", confidence=0.8)
    await db.add_relation(a.id, b.id, "contradicts")

    scorer = UncertaintyScorer(db)
    report = await scorer.compute()

    assert len(report.contradictions) == 1
    assert report.overall_uncertainty > 0.0


async def test_scorer_detects_low_confidence_connected_entries(db):
    low = await db.store(type="knowledge", content="Uncertain claim", confidence=0.2)
    anchor = await db.store(type="knowledge", content="Known fact", confidence=0.9)
    await db.add_relation(low.id, anchor.id, "related_to")

    scorer = UncertaintyScorer(db)
    report = await scorer.compute()

    low_conf_ids = [e["id"] for e in report.low_confidence_entries]
    assert low.id in low_conf_ids


async def test_scorer_uncertainty_increases_with_more_gaps(db):
    scorer = UncertaintyScorer(db)

    report_empty = await scorer.compute()

    await db.store(type="knowledge", content="Isolated 1", confidence=0.3)
    await db.store(type="knowledge", content="Isolated 2", confidence=0.2)
    report_with_gaps = await scorer.compute()

    assert report_with_gaps.overall_uncertainty > report_empty.overall_uncertainty


def test_overall_uncertainty_pure_function():
    score = _overall_uncertainty(
        total=10,
        isolated_count=5,
        contradiction_count=2,
        confidences=[0.2, 0.3, 0.9, 0.8, 0.5, 0.1, 0.4, 0.7, 0.6, 0.3],
    )
    assert 0.0 < score <= 1.0


def test_overall_uncertainty_zero_on_empty():
    assert _overall_uncertainty(total=0, isolated_count=0, contradiction_count=0, confidences=[]) == 0.0
