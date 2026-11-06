from ia_test.memory.database import MemoryDatabase
from ia_test.memory.models import MemoryEntry
from .models import Discovery


class DiscoveryRecorder:
    def __init__(self, db: MemoryDatabase) -> None:
        self._db = db

    async def record(self, discovery: Discovery, question_id: str) -> MemoryEntry:
        tags = list(discovery.tags) + [
            f"question:{question_id}",
            f"source:{discovery.source}",
        ]
        return await self._db.store(
            type="knowledge",
            content=discovery.content,
            confidence=discovery.confidence,
            provenance=f"exploration:{discovery.source}",
            tags=tags,
        )

    async def link_to_related(
        self,
        new_entry: MemoryEntry,
        limit: int = 200,
        source_entry_id: str | None = None,
    ) -> int:
        """Link new_entry to related entries by keyword overlap.

        Also creates a derived_from relation back to source_entry_id when the
        discovery originated from a specific knowledge graph entry.
        """
        linked = 0

        if source_entry_id and source_entry_id != new_entry.id:
            await self._db.add_relation(new_entry.id, source_entry_id, "derived_from")
            linked += 1

        existing = await self._db.query(limit=limit)
        new_words = {w for w in new_entry.content.lower().split() if len(w) > 4}

        for entry in existing:
            if entry.id == new_entry.id or entry.id == source_entry_id:
                continue
            existing_words = {w for w in entry.content.lower().split() if len(w) > 4}
            if len(new_words & existing_words) >= 2:
                await self._db.add_relation(new_entry.id, entry.id, "related_to")
                linked += 1

        return linked
