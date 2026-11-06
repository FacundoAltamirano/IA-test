from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


@dataclass
class BaseEvent:
    event_id: str = field(default_factory=_uid)
    timestamp: str = field(default_factory=_now)


@dataclass
class SystemStarted(BaseEvent):
    pass


@dataclass
class SystemStopped(BaseEvent):
    pass


@dataclass
class KnowledgeAcquired(BaseEvent):
    entry_id: str = ""
    entry_type: str = ""
    confidence: float = 0.5


@dataclass
class QuestionGenerated(BaseEvent):
    question_id: str = ""
    content: str = ""
    priority_score: float = 0.0
    # What metric triggered this question
    trigger: str = ""  # isolation | contradiction | novelty | staleness


@dataclass
class ExplorationTriggered(BaseEvent):
    question_id: str = ""


@dataclass
class ExplorationCompleted(BaseEvent):
    question_id: str = ""
    discoveries: list = field(default_factory=list)  # list of entry IDs


@dataclass
class ProposalGenerated(BaseEvent):
    proposal_id: str = ""
    description: str = ""
    # Status is always PROPOSED — human approval required before any action
