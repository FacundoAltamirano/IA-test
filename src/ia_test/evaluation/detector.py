from ia_test.memory.database import MemoryDatabase
from .models import HallucinationRisk


class HallucinationDetector:
    """
    Traces a claim back to stored memory entries.
    A claim with no supporting evidence in the knowledge graph is flagged as high risk.
    """

    def __init__(self, db: MemoryDatabase) -> None:
        self._db = db

    async def score(self, claim: str, limit: int = 100) -> HallucinationRisk:
        entries = await self._db.query(limit=limit)
        claim_words = {w for w in claim.lower().split() if len(w) > 4}

        if not claim_words:
            return HallucinationRisk(score=0.5, verdict="partial")

        matched_ids: list[str] = []
        for entry in entries:
            entry_words = {w for w in entry.content.lower().split() if len(w) > 4}
            # A single meaningful keyword (> 4 chars) is enough to establish a link
            if len(claim_words & entry_words) >= 1:
                matched_ids.append(entry.id)

        if not matched_ids:
            return HallucinationRisk(score=1.0, verdict="ungrounded")

        # 2+ matching entries = grounded; 1 = partial
        if len(matched_ids) >= 2:
            return HallucinationRisk(score=0.2, verdict="grounded", matched_entry_ids=matched_ids)

        return HallucinationRisk(score=0.6, verdict="partial", matched_entry_ids=matched_ids)
