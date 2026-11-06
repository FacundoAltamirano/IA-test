import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from ia_test.self_modification.proposer import ImprovementProposer
from ia_test.self_modification.inspector import ArchitectureInspector
from ia_test.self_modification.models import ModificationTarget
from ia_test.providers.base import Response, Usage


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def inspector(repo_root) -> ArchitectureInspector:
    return ArchitectureInspector(repo_root)


def _make_provider(description="Improve error handling", rationale="Better UX") -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content=json.dumps({
            "description": description,
            "rationale": rationale,
            "suggested_changes": "Add more specific exception types instead of bare Exception catches.",
        }),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=100, completion_tokens=80, total_tokens=180),
    ))
    return provider


async def test_proposal_status_is_always_proposed(inspector):
    provider = _make_provider()
    proposer = ImprovementProposer(provider, inspector)
    target = ModificationTarget(
        file_path="src/ia_test/core/event_bus.py",
        description="Improve exception handling",
    )
    proposal = await proposer.propose(target, branch_name="experiment/auto-20240101-120000")
    assert proposal.status == "PROPOSED"


async def test_proposal_cannot_target_forbidden_path(inspector):
    provider = _make_provider()
    proposer = ImprovementProposer(provider, inspector)
    target = ModificationTarget(
        file_path="src/ia_test/safety/limits.py",
        description="Relax rate limits",
    )
    with pytest.raises(PermissionError):
        await proposer.propose(target, branch_name="experiment/auto-20240101-120000")


async def test_proposal_contains_target_path(inspector):
    provider = _make_provider()
    proposer = ImprovementProposer(provider, inspector)
    target = ModificationTarget(
        file_path="src/ia_test/core/events.py",
        description="Add type hints",
    )
    proposal = await proposer.propose(target, branch_name="experiment/auto-test")
    assert proposal.target_path == "src/ia_test/core/events.py"


async def test_proposal_has_branch_name(inspector):
    provider = _make_provider()
    proposer = ImprovementProposer(provider, inspector)
    target = ModificationTarget(
        file_path="src/ia_test/core/events.py",
        description="Improve documentation",
    )
    proposal = await proposer.propose(target, branch_name="experiment/auto-20240101-130000")
    assert proposal.branch_name == "experiment/auto-20240101-130000"
