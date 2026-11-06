import ast
from pathlib import Path

from .models import ModuleSummary

# Paths the system can NEVER read or suggest modifying.
# This list is hardcoded — the AI cannot change it.
FORBIDDEN_PATHS: frozenset[str] = frozenset(
    [
        "src/ia_test/safety/",
        "src/ia_test/self_modification/",  # cannot modify its own engine
    ]
)


def _is_forbidden(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    return any(normalized.startswith(fp) for fp in FORBIDDEN_PATHS)


class ArchitectureInspector:
    """
    Read-only view of the project's source tree.
    Raises PermissionError on any access to FORBIDDEN_PATHS.
    """

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def list_modules(self) -> list[str]:
        """Return all .py source paths (repo-relative), excluding forbidden ones."""
        modules: list[str] = []
        for f in self._root.rglob("src/**/*.py"):
            rel = f.relative_to(self._root).as_posix()
            if not _is_forbidden(rel):
                modules.append(rel)
        return sorted(modules)

    def read_source(self, path: str) -> str:
        """Read a source file. Raises PermissionError for protected paths."""
        if _is_forbidden(path):
            raise PermissionError(
                f"Access denied: '{path}' is a protected path and cannot be read or modified."
            )
        full = self._root / path
        if not full.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        return full.read_text(encoding="utf-8")

    def summarize(self, path: str) -> ModuleSummary:
        """Return basic structural metrics for a source file."""
        source = self.read_source(path)
        try:
            tree = ast.parse(source)
            functions = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        except SyntaxError:
            functions = classes = 0

        return ModuleSummary(
            path=path,
            line_count=len(source.splitlines()),
            function_count=functions,
            class_count=classes,
        )
