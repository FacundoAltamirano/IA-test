from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


# The only status the system can assign. APPLIED is a human action.
ProposalStatus = Literal["PROPOSED"]


@dataclass
class ModificationTarget:
    file_path: str   # repo-relative path
    description: str  # which aspect needs improvement


@dataclass
class ModuleSummary:
    path: str
    line_count: int
    function_count: int
    class_count: int


@dataclass
class TestResult:
    passed: int
    failed: int
    errors: int
    total: int
    duration_seconds: float
    output: str

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0


@dataclass
class Proposal:
    target_path: str
    description: str
    rationale: str
    suggested_changes: str  # human-readable description of the change
    branch_name: str
    id: str = field(default_factory=_uid)
    # The system can only ever set this to PROPOSED — never APPLIED
    status: ProposalStatus = "PROPOSED"
    test_result: TestResult | None = None
    created_at: str = field(default_factory=_now)
