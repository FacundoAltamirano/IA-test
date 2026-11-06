from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


EvaluationVerdict = Literal["improved", "regressed", "neutral"]
HallucinationVerdict = Literal["grounded", "partial", "ungrounded"]


@dataclass
class MemorySnapshot:
    total_entries: int
    avg_confidence: float
    contradiction_count: int
    isolated_count: int
    entry_ids: set[str] = field(default_factory=set)
    taken_at: str = field(default_factory=_now)


@dataclass
class LearningEvaluation:
    verdict: EvaluationVerdict
    knowledge_gained: int
    confidence_delta: float
    contradictions_introduced: int
    evidence: list[str] = field(default_factory=list)


@dataclass
class HallucinationRisk:
    score: float  # 0.0 = fully grounded, 1.0 = fully ungrounded
    verdict: HallucinationVerdict
    matched_entry_ids: list[str] = field(default_factory=list)
