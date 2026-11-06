import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from ia_test.self_modification.engine import SelfModificationEngine
from ia_test.self_modification.models import ModificationTarget
from ia_test.core.event_bus import EventBus
from ia_test.core.events import ProposalGenerated
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Response, Usage


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "selfmod_test.db"))
    await database.connect()
    yield database
    await database.close()


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content=json.dumps({
            "description": "Add docstrings to public methods",
            "rationale": "Improves developer experience and self-documentation",
            "suggested_changes": "Add numpy-style docstrings to all public methods in event_bus.py",
        }),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=100, completion_tokens=80, total_tokens=180),
    ))
    return provider


def test_engine_has_no_apply_method(repo_root, db):
    """Safety invariant: the engine cannot apply changes autonomously."""
    engine = SelfModificationEngine(
        db=db, provider=_make_provider(), event_bus=EventBus(), repo_root=repo_root
    )
    assert not hasattr(engine, "apply"), "SelfModificationEngine must not have an apply() method"
    assert not hasattr(engine, "merge"), "SelfModificationEngine must not have a merge() method"


def test_engine_list_candidates_excludes_forbidden(repo_root, db):
    engine = SelfModificationEngine(
        db=db, provider=_make_provider(), event_bus=EventBus(), repo_root=repo_root
    )
    candidates = engine.list_candidate_targets()
    assert len(candidates) > 0
    for c in candidates:
        assert "safety" not in c
        assert "self_modification" not in c


async def test_engine_propose_stores_proposal_in_memory(db, repo_root):
    # Mock test runner and branch manager to avoid git operations
    with (
        patch("ia_test.self_modification.engine.TestRunner") as MockRunner,
        patch("ia_test.self_modification.engine.BranchManager") as MockBranch,
    ):
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=MagicMock(passed=83, failed=0, errors=0, total=83, success=True))
        MockRunner.return_value = mock_runner

        mock_branch = MagicMock()
        mock_branch.generate_branch_name.return_value = "experiment/auto-20240101-120000"
        MockBranch.return_value = mock_branch

        engine = SelfModificationEngine(
            db=db, provider=_make_provider(), event_bus=EventBus(), repo_root=repo_root
        )
        target = ModificationTarget(
            file_path="src/ia_test/core/events.py",
            description="Add docstrings",
        )
        proposal = await engine.propose(target)

    assert proposal.status == "PROPOSED"
    entries = await db.query(type="experiment", limit=10)
    assert len(entries) == 1
    assert "PROPOSED" in entries[0].content
    assert "proposal:pending" in entries[0].tags


async def test_engine_propose_publishes_proposal_generated_event(db, repo_root):
    bus = EventBus()
    published = []

    async def capture(e):
        published.append(e)

    bus.subscribe(ProposalGenerated, capture)

    with (
        patch("ia_test.self_modification.engine.TestRunner") as MockRunner,
        patch("ia_test.self_modification.engine.BranchManager") as MockBranch,
    ):
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=MagicMock(passed=83, failed=0))
        MockRunner.return_value = mock_runner

        mock_branch = MagicMock()
        mock_branch.generate_branch_name.return_value = "experiment/auto-20240101-120000"
        MockBranch.return_value = mock_branch

        engine = SelfModificationEngine(
            db=db, provider=_make_provider(), event_bus=bus, repo_root=repo_root
        )
        await engine.propose(ModificationTarget(
            file_path="src/ia_test/core/events.py",
            description="Improve typing",
        ))

    generated = [e for e in published if isinstance(e, ProposalGenerated)]
    assert len(generated) == 1


async def test_engine_cannot_propose_forbidden_path(db, repo_root):
    engine = SelfModificationEngine(
        db=db, provider=_make_provider(), event_bus=EventBus(), repo_root=repo_root
    )
    with pytest.raises(PermissionError):
        await engine.propose(ModificationTarget(
            file_path="src/ia_test/safety/limits.py",
            description="Raise rate limits",
        ))
