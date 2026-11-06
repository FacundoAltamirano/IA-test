"""
UncertaintyScorer — pure computation, no LLM.

This is where the curiosity mechanism lives: not in a prompt, but in
measurable properties of the knowledge graph. The scorer tells the system
WHAT to investigate. The LLM only writes HOW to ask it.
"""

import statistics
from ..memory.database import MemoryDatabase
from .models import UncertaintyReport


class UncertaintyScorer:
    def __init__(self, db: MemoryDatabase) -> None:
        self._db = db

    async def compute(self, limit: int = 10) -> UncertaintyReport:
        gaps = await self._db.get_knowledge_gaps(limit=limit)
        contradictions = await self._db.find_contradictions()
        all_entries = await self._db.query(limit=1000)

        isolated = [
            {"id": g.entry_id, "content": g.content, "score": g.score}
            for g in gaps
            if g.reason == "isolated"
        ]

        low_conf = [
            {"id": g.entry_id, "content": g.content, "score": g.score}
            for g in gaps
            if g.reason == "low_confidence"
        ]

        contradiction_data = [
            {
                "source_id": c.source.id,
                "source_content": c.source.content,
                "target_content": c.target.content,
                "score": _contradiction_score(c.source.confidence, c.target.confidence),
            }
            for c in contradictions
        ]

        overall = _overall_uncertainty(
            total=len(all_entries),
            isolated_count=len(isolated),
            contradiction_count=len(contradictions),
            confidences=[e.confidence for e in all_entries],
        )

        return UncertaintyReport(
            isolated_entries=isolated,
            low_confidence_entries=low_conf,
            contradictions=contradiction_data,
            total_entries=len(all_entries),
            overall_uncertainty=overall,
        )


def _contradiction_score(conf_a: float, conf_b: float) -> float:
    # Contradictions between high-confidence entries are more urgent
    return round((conf_a + conf_b) / 2, 4)


def _overall_uncertainty(
    total: int,
    isolated_count: int,
    contradiction_count: int,
    confidences: list[float],
) -> float:
    if total == 0:
        return 0.0

    isolation_ratio = isolated_count / total
    contradiction_ratio = min(contradiction_count / max(total, 1), 1.0)
    conf_variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
    low_conf_ratio = sum(1 for c in confidences if c < 0.5) / total

    score = (
        0.35 * isolation_ratio
        + 0.25 * contradiction_ratio
        + 0.20 * min(conf_variance * 4, 1.0)
        + 0.20 * low_conf_ratio
    )
    return round(min(score, 1.0), 4)
