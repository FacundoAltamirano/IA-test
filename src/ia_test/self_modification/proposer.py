import json
import re

from ia_test.providers.base import Message, ModelProvider

from .inspector import ArchitectureInspector, _is_forbidden
from .models import ModificationTarget, Proposal

_SYSTEM = """\
You are a code improvement analyst. You inspect source code and propose ONE specific,
concrete improvement. You never write the actual code — you describe what should change.

Respond ONLY with valid JSON:
{
  "description": "One sentence: what this improvement achieves",
  "rationale": "Why this change would improve the system (1-2 sentences)",
  "suggested_changes": "Concrete description of what to change and how (3-5 sentences)"
}

Constraints:
- Do not suggest changes to safety/, audit logs, or permission systems
- Keep suggestions specific and implementable
- Focus on correctness, clarity, or testability — not premature optimization
"""


class ImprovementProposer:
    def __init__(self, provider: ModelProvider, inspector: ArchitectureInspector) -> None:
        self._provider = provider
        self._inspector = inspector

    async def propose(self, target: ModificationTarget, branch_name: str) -> Proposal:
        if _is_forbidden(target.file_path):
            raise PermissionError(
                f"Cannot propose modifications to protected path: {target.file_path}"
            )

        source = self._inspector.read_source(target.file_path)
        messages = [
            Message(role="system", content=_SYSTEM),
            Message(
                role="user",
                content=(
                    f"File: {target.file_path}\n"
                    f"Improvement focus: {target.description}\n\n"
                    f"Source:\n```python\n{source[:3000]}\n```"
                ),
            ),
        ]
        response = await self._provider.complete(messages=messages, temperature=0.4)
        parsed = self._parse(response.content)

        return Proposal(
            target_path=target.file_path,
            description=parsed.get("description", "Improvement proposal") if parsed else "Improvement proposal",
            rationale=parsed.get("rationale", "") if parsed else "",
            suggested_changes=parsed.get("suggested_changes", response.content[:500]) if parsed else response.content[:500],
            branch_name=branch_name,
            status="PROPOSED",  # always — the system cannot set this to anything else
        )

    def _parse(self, content: str) -> dict | None:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            return None
