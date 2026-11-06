"""
QuestionGenerator — the LLM generates the TEXT of the question.
The scorer already decided WHAT to investigate and WHY.
The LLM's only job here is to phrase the question well.
"""

import json
import logging

from ..providers.base import Message, ModelProvider
from .models import Question, QuestionTrigger, UncertaintyReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You analyze knowledge graph gaps and generate precise research questions.
You will receive structured data about uncertain, isolated, or contradictory knowledge.
Your job is to generate ONE specific, investigable question that targets the most important gap.
Respond ONLY with valid JSON in this exact format:
{"question": "<the question>", "source_entry_id": "<id or null>", "trigger": "<isolation|low_confidence|contradiction>"}
Do not add explanation, markdown, or any other text."""


class QuestionGenerator:
    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    async def generate(
        self, report: UncertaintyReport, max_questions: int = 3
    ) -> list[Question]:
        if report.total_entries == 0:
            return []

        candidates = _build_candidates(report, max_questions)
        if not candidates:
            return []

        questions = []
        for candidate in candidates:
            try:
                question = await self._generate_one(candidate, report.overall_uncertainty)
                if question:
                    questions.append(question)
            except Exception as exc:
                logger.warning("Question generation failed for candidate: %s", exc)

        return sorted(questions)  # sorted by priority_score desc via __lt__

    async def _generate_one(self, candidate: dict, overall_uncertainty: float) -> Question | None:
        user_content = _format_candidate(candidate)
        response = await self._provider.complete(
            messages=[
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ],
            temperature=0.4,
        )

        parsed = _parse_response(response.content)
        if not parsed:
            return None

        return Question(
            content=parsed["question"],
            priority_score=round(candidate["score"] * (1 + overall_uncertainty), 4),
            trigger=candidate["trigger"],
            source_entry_id=parsed.get("source_entry_id") or candidate.get("id"),
        )


def _build_candidates(report: UncertaintyReport, limit: int) -> list[dict]:
    candidates = []

    for entry in report.isolated_entries:
        candidates.append({**entry, "trigger": "isolation"})

    for entry in report.low_confidence_entries:
        candidates.append({**entry, "trigger": "low_confidence"})

    for c in report.contradictions:
        candidates.append({
            "id": c["source_id"],
            "content": f"{c['source_content']} ↔ {c['target_content']}",
            "score": c["score"],
            "trigger": "contradiction",
        })

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]


def _format_candidate(candidate: dict) -> str:
    trigger = candidate["trigger"]
    content = candidate["content"]
    score = candidate["score"]

    if trigger == "isolation":
        return (
            f"Knowledge gap — ISOLATED ENTRY (no connections in the graph, score={score}):\n"
            f"Content: {content}\n\n"
            f"Generate a question that would help connect or verify this knowledge."
        )
    if trigger == "low_confidence":
        return (
            f"Knowledge gap — LOW CONFIDENCE ENTRY (score={score}):\n"
            f"Content: {content}\n\n"
            f"Generate a question that would help verify or refute this."
        )
    if trigger == "contradiction":
        return (
            f"Knowledge gap — CONTRADICTION (score={score}):\n"
            f"Conflicting entries: {content}\n\n"
            f"Generate a question that would help resolve this contradiction."
        )
    return f"Knowledge gap: {content}\nGenerate a research question."


def _parse_response(content: str) -> dict | None:
    try:
        data = json.loads(content.strip())
        if "question" not in data or not data["question"].strip():
            return None
        trigger = data.get("trigger", "isolation")
        if trigger not in ("isolation", "low_confidence", "contradiction", "staleness"):
            trigger = "isolation"
        data["trigger"] = trigger
        return data
    except (json.JSONDecodeError, KeyError):
        logger.warning("Could not parse question generator response: %r", content[:200])
        return None
