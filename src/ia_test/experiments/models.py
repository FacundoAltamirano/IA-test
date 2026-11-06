from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


ExperimentStatus = Literal["designed", "running", "completed", "failed"]
ExperimentConclusion = Literal["supports", "refutes", "inconclusive"]


@dataclass
class Experiment:
    hypothesis: str
    id: str = field(default_factory=_uid)
    source_entry_id: str | None = None  # hypothesis memory entry being tested
    procedure: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    actual_outcome: str | None = None
    conclusion: ExperimentConclusion | None = None
    # How much to adjust source entry confidence after evaluation
    confidence_delta: float = 0.0
    status: ExperimentStatus = "designed"
    created_at: str = field(default_factory=_now)
    completed_at: str | None = None
