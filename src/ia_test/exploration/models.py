from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


ExplorationStatus = Literal["completed", "partial", "failed"]
DiscoverySource = Literal["synthesis", "web_search", "inference"]


@dataclass
class Discovery:
    content: str
    source: DiscoverySource
    confidence: float
    tags: list[str] = field(default_factory=list)
    related_entry_ids: list[str] = field(default_factory=list)


@dataclass
class ExplorationResult:
    question_id: str
    question_content: str
    status: ExplorationStatus
    discoveries: list[Discovery]
    error: str | None = None
    created_at: str = field(default_factory=_now)
