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

    async def link_to_related(self, new_entry: MemoryEntry, limit: int = 20) -> int:
        """Link new_entry to existing entries that share meaningful keyword overlap."""
        existing = await self._db.query(limit=limit)
        linked = 0
        new_words = {w for w in new_entry.content.lower().split() if len(w) > 4}

        for entry in existing:
            if entry.id == new_entry.id:
                continue
            existing_words = {w for w in entry.content.lower().split() if len(w) > 4}
            if len(new_words & existing_words) >= 2:
                await self._db.add_relation(new_entry.id, entry.id, "related_to")
                linked += 1

        return linked
