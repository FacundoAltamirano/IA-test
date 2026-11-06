import subprocess
from datetime import datetime, timezone
from pathlib import Path


BRANCH_PREFIX = "experiment/auto-"


class BranchManager:
    """
    Creates and lists experiment branches.
    Has NO merge() method — merging requires explicit human action.
    """

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def generate_branch_name(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{BRANCH_PREFIX}{stamp}"

    def create_branch(self, branch_name: str) -> str:
        """Create a new Git branch from the current HEAD. Returns the branch name."""
        if not branch_name.startswith(BRANCH_PREFIX):
            raise ValueError(
                f"Branch name must start with '{BRANCH_PREFIX}'. Got: {branch_name}"
            )
        result = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create branch '{branch_name}': {result.stderr.strip()}"
            )
        # Return to the previous branch so we don't stay on the experiment branch
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=self._root,
            capture_output=True,
        )
        return branch_name

    def list_experiment_branches(self) -> list[str]:
        """Return all experiment/auto-* branches."""
        result = subprocess.run(
            ["git", "branch", "--list", f"{BRANCH_PREFIX}*"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        branches = [b.strip().lstrip("* ") for b in result.stdout.splitlines()]
        return [b for b in branches if b]

    def branch_exists(self, branch_name: str) -> bool:
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    # merge() intentionally absent — human approval required
