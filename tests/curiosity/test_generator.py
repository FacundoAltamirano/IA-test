import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from ia_test.curiosity.generator import QuestionGenerator
from ia_test.curiosity.models import UncertaintyReport
from ia_test.providers.base import Response, Usage


def _mock_provider(question_text: str = "What is the nature of X?") -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content=json.dumps({
            "question": question_text,
            "source_entry_id": "entry-123",
            "trigger": "isolation",
        }),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    ))
    return provider


def _report_with_isolated() -> UncertaintyReport:
    return UncertaintyReport(
        isolated_entries=[{"id": "entry-123", "content": "Some knowledge", "score": 0.7}],
        low_confidence_entries=[],
        contradictions=[],
        total_entries=5,
        overall_uncertainty=0.4,
    )


async def test_generator_returns_question_from_llm():
    provider = _mock_provider("Why is the sky blue?")
    generator = QuestionGenerator(provider)

    questions = await generator.generate(_report_with_isolated(), max_questions=1)

    assert len(questions) == 1
    assert questions[0].content == "Why is the sky blue?"
    assert questions[0].trigger == "isolation"
    assert questions[0].source_entry_id == "entry-123"


async def test_generator_returns_empty_on_empty_report():
    provider = _mock_provider()
    generator = QuestionGenerator(provider)

    empty_report = UncertaintyReport(
        isolated_entries=[], low_confidence_entries=[],
        contradictions=[], total_entries=0, overall_uncertainty=0.0
    )
    questions = await generator.generate(empty_report)

    assert questions == []
    provider.complete.assert_not_called()


async def test_generator_handles_malformed_llm_response():
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=Response(
        content="This is not JSON at all",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))
    generator = QuestionGenerator(provider)

    questions = await generator.generate(_report_with_isolated())

    assert questions == []


async def test_generator_priority_score_incorporates_overall_uncertainty():
    provider = _mock_provider("Test question")
    generator = QuestionGenerator(provider)

    report_low_uncertainty = UncertaintyReport(
        isolated_entries=[{"id": "e1", "content": "X", "score": 0.5}],
        low_confidence_entries=[], contradictions=[],
        total_entries=10, overall_uncertainty=0.1,
    )
    report_high_uncertainty = UncertaintyReport(
        isolated_entries=[{"id": "e1", "content": "X", "score": 0.5}],
        low_confidence_entries=[], contradictions=[],
        total_entries=10, overall_uncertainty=0.9,
    )

    q_low = await generator.generate(report_low_uncertainty, max_questions=1)
    q_high = await generator.generate(report_high_uncertainty, max_questions=1)

    assert q_high[0].priority_score > q_low[0].priority_score
