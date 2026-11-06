import asyncio
import time
from pathlib import Path

from .models import TestResult


class TestRunner:
    """Runs pytest in a subprocess and captures structured results."""

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    async def run(self) -> TestResult:
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", "tests/", "--tb=no", "-q",
            cwd=self._root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        duration = time.monotonic() - start
        output = stdout.decode("utf-8", errors="replace")

        return self._parse_output(output, duration)

    def _parse_output(self, output: str, duration: float) -> TestResult:
        passed = failed = errors = 0
        for line in output.splitlines():
            # pytest summary line: "5 passed, 1 failed, 2 errors in 1.23s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed" and i > 0:
                    try:
                        passed = int(parts[i - 1])
                    except ValueError:
                        pass
                elif part in ("failed", "error") and i > 0:
                    try:
                        n = int(parts[i - 1])
                        if part == "failed":
                            failed += n
                        else:
                            errors += n
                    except ValueError:
                        pass

        total = passed + failed + errors
        return TestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            total=total,
            duration_seconds=round(duration, 2),
            output=output[:2000],
        )
