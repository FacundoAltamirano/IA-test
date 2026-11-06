import pytest
from pathlib import Path
from ia_test.self_modification.inspector import ArchitectureInspector, FORBIDDEN_PATHS


@pytest.fixture
def repo_root() -> Path:
    # Points to the actual repo root so the inspector can read real source files
    return Path(__file__).parent.parent.parent


def test_forbidden_paths_includes_safety(repo_root):
    forbidden = FORBIDDEN_PATHS
    assert any("safety" in p for p in forbidden)


def test_inspector_raises_on_safety_path(repo_root):
    inspector = ArchitectureInspector(repo_root)
    with pytest.raises(PermissionError):
        inspector.read_source("src/ia_test/safety/limits.py")


def test_inspector_raises_on_self_modification_path(repo_root):
    inspector = ArchitectureInspector(repo_root)
    with pytest.raises(PermissionError):
        inspector.read_source("src/ia_test/self_modification/engine.py")


def test_inspector_reads_allowed_source(repo_root):
    inspector = ArchitectureInspector(repo_root)
    source = inspector.read_source("src/ia_test/core/events.py")
    assert len(source) > 0
    assert "class" in source or "def" in source


def test_inspector_list_modules_excludes_forbidden(repo_root):
    inspector = ArchitectureInspector(repo_root)
    modules = inspector.list_modules()
    assert len(modules) > 0
    for m in modules:
        assert "safety" not in m
        assert "self_modification" not in m


def test_inspector_summarize_returns_metrics(repo_root):
    inspector = ArchitectureInspector(repo_root)
    summary = inspector.summarize("src/ia_test/core/events.py")
    assert summary.line_count > 0
    assert summary.path == "src/ia_test/core/events.py"


def test_inspector_summarize_counts_classes(repo_root):
    inspector = ArchitectureInspector(repo_root)
    summary = inspector.summarize("src/ia_test/core/event_bus.py")
    assert summary.class_count >= 1
