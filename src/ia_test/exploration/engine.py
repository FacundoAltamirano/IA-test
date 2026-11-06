import json
import logging
import re

from ia_test.core.event_bus import EventBus
from ia_test.core.events import (
    DiscoveryRecorded,
    ExplorationCompleted,
    ExplorationStarted,
)
from ia_test.core.settings import Settings
from ia_test.curiosity.models import Question
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Message, ModelProvider

from .models import Discovery, ExplorationResult
from .recorder import DiscoveryRecorder
from .tools import ExplorationTool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an exploration agent. Research the given question and return structured findings.

Respond ONLY with valid JSON in this exact format:
{
  "findings": [
    {
      "content": "A specific, concrete factual statement or insight",
      "confidence": 0.7,
      "tags": ["tag1", "tag2"],
      "source": "synthesis"
    }
  ],
  "search_queries": ["optional query string if external data would help"]
}

Rules:
- Each finding must be a specific, non-vague statement
- confidence is 0.0–1.0 (reflect genuine uncertainty)
- source must be "synthesis" (from your training) or "web_search"
- Include 1–5 findings; include no more than 2 search_queries
- If the question is unanswerable, return one finding explaining why with confidence 0.2
"""


class ExplorationEngine:
    def __init__(
        self,
        db: MemoryDatabase,
        provider: ModelProvider,
        event_bus: EventBus,
        settings: Settings,
        tools: list[ExplorationTool] | None = None,
    ) -> None:
        self._db = db
        self._provider = provider
        self._event_bus = event_bus
        self._settings = settings
        self._recorder = DiscoveryRecorder(db)
        self._tools: dict[str, ExplorationTool] = {t.name: t for t in (tools or [])}

    async def explore(self, question: Question) -> ExplorationResult:
        await self._event_bus.publish(
            ExplorationStarted(question_id=question.id, content=question.content)
        )

        try:
            result = await self._run_exploration(question)
        except Exception as exc:
            logger.exception("Exploration failed for question %s", question.id)
            await self._db.store(
                type="error",
                content=f"Exploration failed: {question.content}. Error: {exc}",
                confidence=1.0,
                provenance="exploration:engine",
                tags=[f"question:{question.id}", "error:exploration"],
            )
            result = ExplorationResult(
                question_id=question.id,
                question_content=question.content,
                status="failed",
                discoveries=[],
                error=str(exc),
            )

        await self._event_bus.publish(
            ExplorationCompleted(
                question_id=question.id,
                discoveries=[d.content for d in result.discoveries],
            )
        )
        return result

    async def _run_exploration(self, question: Question) -> ExplorationResult:
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=f"Research this question: {question.content}"),
        ]
        response = await self._provider.complete(messages=messages, temperature=0.5)
        parsed = self._parse_response(response.content)

        if parsed is None:
            return ExplorationResult(
                question_id=question.id,
                question_content=question.content,
                status="failed",
                discoveries=[],
                error="Could not parse LLM response as JSON",
            )

        raw_findings: list[dict] = parsed.get("findings", [])
        search_queries: list[str] = parsed.get("search_queries", [])

        # Execute optional web searches
        for query in search_queries[:2]:
            if "web_search" in self._tools:
                search_result = await self._tools["web_search"].execute(query)
                if not search_result.startswith("[web_search not configured"):
                    raw_findings.append({
                        "content": search_result,
                        "confidence": 0.6,
                        "tags": ["web_search"],
                        "source": "web_search",
                    })

        discoveries: list[Discovery] = []
        for f in raw_findings:
            if not isinstance(f, dict) or not f.get("content"):
                continue

            discovery = Discovery(
                content=str(f["content"]),
                source=f.get("source", "synthesis"),
                confidence=max(0.0, min(1.0, float(f.get("confidence", 0.5)))),
                tags=list(f.get("tags", [])),
            )

            entry = await self._recorder.record(discovery, question.id)
            await self._recorder.link_to_related(
                entry, source_entry_id=question.source_entry_id
            )
            discoveries.append(discovery)

            await self._event_bus.publish(
                DiscoveryRecorded(
                    entry_id=entry.id,
                    question_id=question.id,
                    source=discovery.source,
                )
            )

        status: str = "completed" if discoveries else "partial"
        return ExplorationResult(
            question_id=question.id,
            question_content=question.content,
            status=status,  # type: ignore[arg-type]
            discoveries=discoveries,
        )

    def _parse_response(self, content: str) -> dict | None:
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
