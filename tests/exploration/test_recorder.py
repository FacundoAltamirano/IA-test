import pytest
from ia_test.memory.database import MemoryDatabase
from ia_test.exploration.recorder import DiscoveryRecorder
from ia_test.exploration.models import Discovery


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "recorder_test.db"))
    await database.connect()
    yield database
    await database.close()


async def test_recorder_stores_discovery_as_knowledge_entry(db):
    recorder = DiscoveryRecorder(db)
    discovery = Discovery(
        content="Photosynthesis converts light into chemical energy",
        source="synthesis",
        confidence=0.85,
        tags=["biology", "energy"],
    )

    entry = await recorder.record(discovery, question_id="q-001")

    assert entry.type == "knowledge"
    assert entry.content == discovery.content
    assert entry.confidence == discovery.confidence
    assert "question:q-001" in entry.tags
    assert "source:synthesis" in entry.tags


async def test_recorder_provenance_includes_source(db):
    recorder = DiscoveryRecorder(db)
    discovery = Discovery(
        content="Some web finding",
        source="web_search",
        confidence=0.6,
    )

    entry = await recorder.record(discovery, question_id="q-002")

    assert entry.provenance == "exploration:web_search"


async def test_recorder_links_entries_with_keyword_overlap(db):
    anchor = await db.store(
        type="knowledge",
        content="Photosynthesis requires chlorophyll molecules found in plants",
        confidence=0.9,
    )

    recorder = DiscoveryRecorder(db)
    discovery = Discovery(
        content="Chlorophyll absorbs sunlight during photosynthesis",
        source="synthesis",
        confidence=0.8,
    )
    new_entry = await recorder.record(discovery, question_id="q-003")
    linked = await recorder.link_to_related(new_entry)

    assert linked >= 1
    relations = await db.find_contradictions()  # just to verify DB is live
    # Verify via query that the relation was actually created
    gaps = await db.get_knowledge_gaps()
    gap_ids = [g.entry_id for g in gaps]
    assert new_entry.id not in gap_ids  # no longer isolated


async def test_recorder_creates_derived_from_relation_to_source(db):
    source = await db.store(
        type="knowledge",
        content="Neural networks learn from data through backpropagation",
        confidence=0.9,
    )

    recorder = DiscoveryRecorder(db)
    discovery = Discovery(
        content="Gradient descent optimizes weights during neural network training",
        source="synthesis",
        confidence=0.75,
    )
    new_entry = await recorder.record(discovery, question_id="q-src-01")
    linked = await recorder.link_to_related(new_entry, source_entry_id=source.id)

    assert linked >= 1
    # Verify derived_from relation exists in the DB
    async with db._db.execute(
        "SELECT relation_type FROM memory_relations WHERE source_id = ? AND target_id = ?",
        (new_entry.id, source.id),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row["relation_type"] == "derived_from"


async def test_recorder_does_not_link_unrelated_entries(db):
    await db.store(
        type="knowledge",
        content="Jazz music originated in New Orleans",
        confidence=0.9,
    )

    recorder = DiscoveryRecorder(db)
    discovery = Discovery(
        content="Quantum tunneling allows particles to pass through barriers",
        source="synthesis",
        confidence=0.7,
    )
    new_entry = await recorder.record(discovery, question_id="q-004")
    linked = await recorder.link_to_related(new_entry)

    assert linked == 0
