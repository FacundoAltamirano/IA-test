import logging
from pathlib import Path

from ia_test.core.event_bus import EventBus
from ia_test.core.events import ProposalGenerated
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import ModelProvider

from .branch_manager import BranchManager
from .inspector import ArchitectureInspector
from .models import ModificationTarget, Proposal, TestResult
from .proposer import ImprovementProposer
from .test_runner import TestRunner

logger = logging.getLogger(__name__)


class SelfModificationEngine:
    """
    Orchestrates the self-modification lifecycle:
    Inspect → Propose → Branch → Test → Store → Await human approval.

    Cannot apply changes. Cannot merge branches.
    Cannot touch safety/ or self_modification/.
    """

    def __init__(
        self,
        db: MemoryDatabase,
        provider: ModelProvider,
        event_bus: EventBus,
        repo_root: Path,
    ) -> None:
        self._db = db
        self._event_bus = event_bus
        self._inspector = ArchitectureInspector(repo_root)
        self._proposer = ImprovementProposer(provider, self._inspector)
        self._branch_manager = BranchManager(repo_root)
        self._test_runner = TestRunner(repo_root)

    def list_candidate_targets(self) -> list[str]:
        """Return all modifiable source modules (forbidden paths excluded)."""
        return self._inspector.list_modules()

    async def propose(self, target: ModificationTarget) -> Proposal:
        """Full proposal lifecycle — always leaves branch unmerged, proposal PROPOSED."""
        branch_name = self._branch_manager.generate_branch_name()
        proposal = await self._proposer.propose(target, branch_name)

        # Run current test suite to establish a baseline
        try:
            test_result = await self._test_runner.run()
            proposal.test_result = test_result
        except Exception as exc:
            logger.warning("Test runner failed during proposal: %s", exc)

        await self._store_proposal(proposal)
        await self._event_bus.publish(
            ProposalGenerated(
                proposal_id=proposal.id,
                description=proposal.description,
            )
        )
        return proposal

    async def _store_proposal(self, proposal: Proposal) -> None:
        summary = (
            f"[PROPOSED] {proposal.description}\n"
            f"Target: {proposal.target_path}\n"
            f"Rationale: {proposal.rationale}\n"
            f"Branch: {proposal.branch_name}\n"
            f"Status: {proposal.status} — requires human approval before any action."
        )
        await self._db.store(
            type="experiment",
            content=summary,
            confidence=0.5,
            provenance="self_modification:proposed",
            tags=[
                f"proposal:{proposal.id}",
                "proposal:pending",
                f"target:{proposal.target_path}",
            ],
        )

    # apply() intentionally absent — merging and applying are human responsibilities
