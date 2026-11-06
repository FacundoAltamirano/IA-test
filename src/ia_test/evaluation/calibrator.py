from ia_test.memory.database import MemoryDatabase


class ConfidenceCalibrator:
    """
    Adjusts entry confidence based on graph structure — not on what the LLM claims.
    Entries in contradiction relations lose confidence; isolated high-confidence entries are penalized.
    """

    def __init__(self, db: MemoryDatabase) -> None:
        self._db = db

    async def calibrate(self) -> int:
        """Return the number of entries whose confidence was adjusted."""
        entries = await self._db.query(limit=500)
        contradictions = await self._db.find_contradictions()
        gaps = await self._db.get_knowledge_gaps()

        contradiction_ids: set[str] = set()
        for c in contradictions:
            contradiction_ids.add(c.source.id)
            contradiction_ids.add(c.target.id)

        isolated_ids = {g.entry_id for g in gaps}

        adjusted = 0
        for entry in entries:
            new_confidence = entry.confidence

            if entry.id in contradiction_ids:
                # Contradicted entries should not carry high confidence
                new_confidence = min(entry.confidence, 0.55)
            elif entry.id in isolated_ids and entry.confidence > 0.7:
                # Isolated entries have no corroboration; cap their confidence
                new_confidence = min(entry.confidence, 0.65)

            if abs(new_confidence - entry.confidence) > 0.005:
                await self._db.update_confidence(
                    entry.id,
                    new_confidence,
                    reason="graph_calibration",
                )
                adjusted += 1

        return adjusted
