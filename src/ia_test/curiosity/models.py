from dataclasses import dataclass, field
from typing import Literal
import uuid
from datetime import datetime, timezone


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


QuestionStatus = Literal["pending", "exploring", "answered", "discarded"]
QuestionTrigger = Literal["isolation", "low_confidence", "contradiction", "staleness"]


@dataclass
class Question:
    content: str
    priority_score: float
    trigger: QuestionTrigger
    id: str = field(default_factory=_uid)
    source_entry_id: str | None = None
    status: QuestionStatus = "pending"
    created_at: str = field(default_factory=_now)

    def __lt__(self, other: "Question") -> bool:
        # Higher score = higher priority
        return self.priority_score > other.priority_score


@dataclass
class UncertaintyReport:
    """Output of UncertaintyScorer — input to QuestionGenerator."""
    isolated_entries: list[dict]       # [{id, content, score}]
    low_confidence_entries: list[dict] # [{id, content, confidence, score}]
    contradictions: list[dict]         # [{source_id, source_content, target_content, score}]
    total_entries: int
    overall_uncertainty: float         # 0.0 = fully known, 1.0 = completely uncertain
