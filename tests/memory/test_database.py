import pytest
from ia_test.memory.database import MemoryDatabase


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "test_memory.db"))
    await database.connect()
    yield database
    await database.close()


async def test_store_and_retrieve_entry(db):
    entry = await db.store(
        type="knowledge",
        content="Python uses GIL for thread safety.",
        confidence=0.9,
        provenance="human",
        tags=["python", "concurrency"],
    )

    retrieved = await db.get(entry.id)

    assert retrieved is not None
    assert retrieved.id == entry.id
    assert retrieved.content == "Python uses GIL for thread safety."
    assert retrieved.confidence == 0.9
    assert retrieved.type == "knowledge"
    assert retrieved.tags == ["python", "concurrency"]


async def test_persistence_across_reconnect(tmp_path):
    path = str(tmp_path / "persist.db")

    db1 = MemoryDatabase(path)
    await db1.connect()
    entry = await db1.store(type="knowledge", content="Persistent fact", confidence=0.8)
    await db1.close()

    db2 = MemoryDatabase(path)
    await db2.connect()
    retrieved = await db2.get(entry.id)
    await db2.close()

    assert retrieved is not None
    assert retrieved.content == "Persistent fact"


async def test_query_by_type(db):
    await db.store(type="knowledge", content="A fact", confidence=0.8)
    await db.store(type="hypothesis", content="A hypothesis", confidence=0.5)
    await db.store(type="knowledge", content="Another fact", confidence=0.7)

    knowledge = await db.query(type="knowledge")
    hypotheses = await db.query(type="hypothesis")

    assert len(knowledge) == 2
    assert len(hypotheses) == 1


async def test_query_min_confidence_filter(db):
    await db.store(type="knowledge", content="High confidence", confidence=0.9)
    await db.store(type="knowledge", content="Low confidence", confidence=0.2)

    high = await db.query(min_confidence=0.5)

    assert len(high) == 1
    assert high[0].content == "High confidence"


async def test_update_confidence(db):
    entry = await db.store(type="hypothesis", content="Test hypothesis", confidence=0.3)
    await db.update_confidence(entry.id, new_confidence=0.75, reason="verified by experiment")

    updated = await db.get(entry.id)
    assert updated.confidence == 0.75


async def test_confidence_clamped_to_valid_range(db):
    entry = await db.store(type="knowledge", content="Some fact", confidence=0.5)

    await db.update_confidence(entry.id, new_confidence=1.5)
    assert (await db.get(entry.id)).confidence == 1.0

    await db.update_confidence(entry.id, new_confidence=-0.3)
    assert (await db.get(entry.id)).confidence == 0.0


async def test_add_relation_between_entries(db):
    a = await db.store(type="knowledge", content="A")
    b = await db.store(type="knowledge", content="B")

    relation = await db.add_relation(a.id, b.id, "supports")

    assert relation.source_id == a.id
    assert relation.target_id == b.id
    assert relation.relation_type == "supports"


async def test_get_knowledge_gaps_returns_isolated_entries(db):
    # Isolated: no outgoing relations
    isolated = await db.store(type="knowledge", content="Orphan knowledge", confidence=0.5)
    # Connected: has a relation
    connected = await db.store(type="knowledge", content="Connected knowledge", confidence=0.5)
    other = await db.store(type="knowledge", content="Other", confidence=0.5)
    await db.add_relation(connected.id, other.id, "related_to")

    gaps = await db.get_knowledge_gaps()
    gap_ids = [g.entry_id for g in gaps]

    assert isolated.id in gap_ids
    assert connected.id not in gap_ids


async def test_get_knowledge_gaps_includes_low_confidence(db):
    low = await db.store(type="knowledge", content="Uncertain", confidence=0.2)
    anchor = await db.store(type="knowledge", content="Anchor", confidence=0.9)
    await db.add_relation(low.id, anchor.id, "related_to")

    gaps = await db.get_knowledge_gaps()
    gap_ids = [g.entry_id for g in gaps]

    assert low.id in gap_ids


async def test_find_contradictions(db):
    a = await db.store(type="knowledge", content="The sky is blue")
    b = await db.store(type="knowledge", content="The sky is red")
    await db.add_relation(a.id, b.id, "contradicts")

    contradictions = await db.find_contradictions()

    assert len(contradictions) == 1
    assert contradictions[0].source.id == a.id
    assert contradictions[0].target.id == b.id


async def test_find_contradictions_empty_when_none(db):
    a = await db.store(type="knowledge", content="A")
    b = await db.store(type="knowledge", content="B")
    await db.add_relation(a.id, b.id, "supports")

    assert await db.find_contradictions() == []
