from ia_test.core.event_bus import EventBus
from ia_test.core.events import EvaluationCompleted
from ia_test.memory.database import MemoryDatabase

from .calibrator import ConfidenceCalibrator
from .detector import HallucinationDetector
from .models import HallucinationRisk, LearningEvaluation, MemorySnapshot


class EvaluationEngine:
    """
    Distinguishes real improvement from the illusion of improvement.
    Uses memory graph structure — not LLM opinion — as evidence.
    """

    def __init__(self, db: MemoryDatabase, event_bus: EventBus) -> None:
        self._db = db
        self._event_bus = event_bus
        self._detector = HallucinationDetector(db)
        self._calibrator = ConfidenceCalibrator(db)

    async def snapshot(self) -> MemorySnapshot:
        entries = await self._db.query(limit=1000)
        contradictions = await self._db.find_contradictions()
        gaps = await self._db.get_knowledge_gaps()
        avg_conf = (
            sum(e.confidence for e in entries) / len(entries) if entries else 0.0
        )
        return MemorySnapshot(
            total_entries=len(entries),
            avg_confidence=round(avg_conf, 4),
            contradiction_count=len(contradictions),
            isolated_count=len(gaps),
            entry_ids={e.id for e in entries},
        )

    def evaluate_learning(
        self, before: MemorySnapshot, after: MemorySnapshot
    ) -> LearningEvaluation:
        knowledge_gained = after.total_entries - before.total_entries
        confidence_delta = round(after.avg_confidence - before.avg_confidence, 4)
        contradictions_introduced = max(
            0, after.contradiction_count - before.contradiction_count
        )

        evidence: list[str] = []
        if knowledge_gained > 0:
            evidence.append(f"{knowledge_gained} new entries added")
        if confidence_delta > 0.01:
            evidence.append(f"avg confidence +{confidence_delta:.3f}")
        elif confidence_delta < -0.01:
            evidence.append(f"avg confidence {confidence_delta:.3f}")
        if contradictions_introduced > 0:
            evidence.append(f"{contradictions_introduced} new contradictions introduced")

        if contradictions_introduced > 2:
            verdict = "regressed"
        elif knowledge_gained > 0 and confidence_delta >= 0 and contradictions_introduced == 0:
            verdict = "improved"
        elif knowledge_gained == 0 and confidence_delta < -0.01:
            verdict = "regressed"
        else:
            verdict = "neutral"

        return LearningEvaluation(
            verdict=verdict,  # type: ignore[arg-type]
            knowledge_gained=knowledge_gained,
            confidence_delta=confidence_delta,
            contradictions_introduced=contradictions_introduced,
            evidence=evidence,
        )

    async def check_hallucination(self, claim: str) -> HallucinationRisk:
        return await self._detector.score(claim)

    async def calibrate(self) -> int:
        return await self._calibrator.calibrate()

    def export_metrics(
        self,
        snapshot: MemorySnapshot,
        evaluation: LearningEvaluation | None = None,
    ) -> dict:
        data: dict = {
            "snapshot": {
                "total_entries": snapshot.total_entries,
                "avg_confidence": snapshot.avg_confidence,
                "contradiction_count": snapshot.contradiction_count,
                "isolated_count": snapshot.isolated_count,
                "taken_at": snapshot.taken_at,
            }
        }
        if evaluation:
            data["evaluation"] = {
                "verdict": evaluation.verdict,
                "knowledge_gained": evaluation.knowledge_gained,
                "confidence_delta": evaluation.confidence_delta,
                "contradictions_introduced": evaluation.contradictions_introduced,
                "evidence": evaluation.evidence,
            }
        return data

    async def publish_evaluation(self, evaluation: LearningEvaluation) -> None:
        await self._event_bus.publish(
            EvaluationCompleted(
                verdict=evaluation.verdict,
                knowledge_gained=evaluation.knowledge_gained,
                contradictions_introduced=evaluation.contradictions_introduced,
            )
        )
