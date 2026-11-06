"""ProposalAnalyzer — converts a raw self-modification proposal into a concrete diff.

Takes RON's natural-language idea (description + rationale + suggested_changes)
and the actual file contents, then asks GPT-4o to produce:
  - a unified diff that could be applied with `patch`
  - a 2-3 line plain-language summary the human reads in the UI
"""
from __future__ import annotations

import difflib
import json
import logging
import pathlib

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a precise software engineer reviewing an AI agent's self-improvement proposal.
Your job: given the CURRENT FILE and the PROPOSAL, produce:
1. A unified diff (--- original / +++ modified) with real, correct Python changes.
2. A 2-3 sentence plain-language summary (Spanish, warm and clear) of exactly what changes
   and why it matters, written for the human who will approve or reject it.

Respond ONLY with valid JSON in this exact shape:
{
  "diff": "<unified diff string, or empty string if no safe change is possible>",
  "summary": "<2-3 sentence explanation in Spanish>",
  "is_safe": true
}

Rules for the diff:
- Must be a valid unified diff (--- a/file / +++ b/file header, then hunks).
- Only change what the proposal requests. No unrelated refactors.
- If the proposal is vague or would break the system, set diff to "" and explain in summary.
- Preserve all existing imports, indentation, and coding style.
"""


class ProposalAnalyzer:
    """Refines a raw proposal into a reviewable diff using GPT-4o."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def analyze(
        self,
        proposal_id: str,
        target_path: str,
        description: str,
        rationale: str,
        suggested_changes: str,
        repo_root: pathlib.Path,
    ) -> tuple[str, str]:
        """Return (diff, summary). diff may be empty if change is unsafe."""
        abs_path = repo_root / target_path
        try:
            current_source = abs_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current_source = "(file not found)"

        user_msg = f"""\
CURRENT FILE ({target_path}):
```python
{current_source[:8000]}
```

PROPOSAL:
Description: {description}
Rationale: {rationale}
Suggested changes: {suggested_changes}

Produce the unified diff and summary now."""

        logger.info("Analyzing proposal %s with %s", proposal_id, self._model)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            diff = str(data.get("diff", ""))
            summary = str(data.get("summary", "Sin resumen disponible."))
            return diff, summary
        except Exception as exc:
            logger.error("ProposalAnalyzer failed: %s", exc)
            return "", f"Error al analizar la propuesta: {exc}"
