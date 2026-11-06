import pytest
from ia_test.memory.database import MemoryDatabase
from ia_test.evaluation.calibrator import ConfidenceCalibrator


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "calibrator_test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_calibrator_lowers_confidence_of_contradicted_entries(db):
    a = await db.store(type="knowledge", content="A is definitely true", confidence=0.9)
    b = await db.store(type="knowledge", content="A is definitely false", confidence=0.85)
    await db.add_relation(a.id, b.id, "contradicts")

    calibrator = ConfidenceCalibrator(db)
    adjusted = await calibrator.calibrate()

    assert adjusted >= 1
    updated_a = await db.get(a.id)
    updated_b = await db.get(b.id)
    assert updated_a.confidence <= 0.55
    assert updated_b.confidence <= 0.55


async def test_calibrator_caps_isolated_high_confidence_entries(db):
    isolated = await db.store(
        type="knowledge", content="Some isolated claim", confidence=0.95
    )

    calibrator = ConfidenceCalibrator(db)
    adjusted = await calibrator.calibrate()

    assert adjusted >= 1
    updated = await db.get(isolated.id)
    assert updated.confidence <= 0.65


async def test_calibrator_does_not_touch_connected_entries(db):
    a = await db.store(type="knowledge", content="Connected fact A", confidence=0.6)
    b = await db.store(type="knowledge", content="Connected fact B", confidence=0.6)
    await db.add_relation(a.id, b.id, "related_to")

    calibrator = ConfidenceCalibrator(db)
    await calibrator.calibrate()

    updated_a = await db.get(a.id)
    assert updated_a.confidence == pytest.approx(0.6)


async def test_calibrator_returns_zero_on_clean_db(db):
    await db.store(type="knowledge", content="Normal grounded entry", confidence=0.5)

    calibrator = ConfidenceCalibrator(db)
    adjusted = await calibrator.calibrate()

    # 0.5 confidence isolated entry — below the 0.7 cap, no adjustment needed
    assert adjusted == 0
