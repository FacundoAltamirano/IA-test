import pytest
from ia_test.memory.database import MemoryDatabase
from ia_test.evaluation.detector import HallucinationDetector


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "detector_test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_ungrounded_claim_scores_high(db):
    # Empty DB — no memory to ground the claim
    detector = HallucinationDetector(db)
    risk = await detector.score("Quantum entanglement enables faster-than-light communication")
    assert risk.score == 1.0
    assert risk.verdict == "ungrounded"
    assert risk.matched_entry_ids == []


async def test_grounded_claim_scores_low(db):
    await db.store(
        type="knowledge",
        content="Quantum entanglement is a phenomenon where particles share quantum states",
        confidence=0.9,
    )
    await db.store(
        type="knowledge",
        content="Entanglement does not enable faster-than-light communication",
        confidence=0.95,
    )
    detector = HallucinationDetector(db)
    risk = await detector.score("Quantum entanglement links particles across distances")
    assert risk.score < 0.5
    assert risk.verdict in ("grounded", "partial")
    assert len(risk.matched_entry_ids) > 0


async def test_partial_overlap_gives_partial_verdict(db):
    await db.store(
        type="knowledge",
        content="Photosynthesis uses chlorophyll to absorb sunlight",
        confidence=0.9,
    )
    detector = HallucinationDetector(db)
    # Shares some words but not enough for full grounding
    risk = await detector.score("Chlorophyll molecules convert light energy into glucose")
    assert risk.verdict in ("grounded", "partial")


async def test_empty_claim_returns_partial(db):
    detector = HallucinationDetector(db)
    risk = await detector.score("the of and")  # only short words, stripped out
    assert risk.verdict == "partial"
