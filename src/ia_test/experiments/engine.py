import json
import logging
import re
from datetime import datetime, timezone

from ia_test.core.event_bus import EventBus
from ia_test.core.events import ExperimentCompleted, ExperimentDesigned
from ia_test.core.settings import Settings
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Message, ModelProvider

from .models import Experiment, ExperimentConclusion

logger = logging.getLogger(__name__)

_DESIGN_SYSTEM = """\
You are a scientific experiment designer for an AI reasoning system.
Given a hypothesis, design a simple experiment to test it using only language and reasoning.

Respond ONLY with valid JSON:
{
  "procedure": ["step 1", "step 2", "step 3"],
  "expected_outcome": "What we expect to observe if the hypothesis is true"
}

Rules:
- 2–5 concrete, executable steps (reasoning steps, not code)
- expected_outcome must be specific and verifiable through reasoning
- Keep it focused: one hypothesis, one test
"""

_RUN_SYSTEM = """\
You are executing an experiment step by step. Follow the procedure exactly and report what you observe.

Respond ONLY with valid JSON:
{
  "step_results": ["observation for step 1", "observation for step 2"],
  "actual_outcome": "What was actually observed after all steps"
}
"""

_EVALUATE_SYSTEM = """\
You are evaluating whether an experiment's actual outcome supports, refutes, or is inconclusive
about the original hypothesis. Be rigorous — do not confirm a hypothesis unless the evidence clearly supports it.

Respond ONLY with valid JSON:
{
  "conclusion": "supports | refutes | inconclusive",
  "reasoning": "One sentence explaining the verdict",
  "confidence_delta": 0.1
}

Rules:
- conclusion must be exactly one of: supports, refutes, inconclusive
- confidence_delta is between -0.3 and +0.3 (how much to adjust the hypothesis confidence)
- Positive delta = more confident, negative = less confident
- Use 0.0 delta for inconclusive
"""


class ExperimentEngine:
    def __init__(
        self,
        db: MemoryDatabase,
        provider: ModelProvider,
        event_bus: EventBus,
        settings: Settings,
    ) -> None:
        self._db = db
        self._provider = provider
        self._event_bus = event_bus
        self._settings = settings

    async def design(
        self, hypothesis: str, source_entry_id: str | None = None
    ) -> Experiment:
        """Ask the LLM to design a procedure to test the hypothesis."""
        messages = [
            Message(role="system", content=_DESIGN_SYSTEM),
            Message(role="user", content=f"Hypothesis to test: {hypothesis}"),
        ]
        response = await self._provider.complete(messages=messages, temperature=0.4)
        parsed = self._parse(response.content)

        experiment = Experiment(
            hypothesis=hypothesis,
            source_entry_id=source_entry_id,
        )

        if parsed:
            experiment.procedure = list(parsed.get("procedure", []))
            experiment.expected_outcome = str(parsed.get("expected_outcome", ""))

        await self._event_bus.publish(
            ExperimentDesigned(
                experiment_id=experiment.id,
                hypothesis=hypothesis,
            )
        )
        return experiment

    async def run(self, experiment: Experiment) -> Experiment:
        """Execute the procedure and collect the actual outcome."""
        experiment.status = "running"
        procedure_text = "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(experiment.procedure)
        )
        messages = [
            Message(role="system", content=_RUN_SYSTEM),
            Message(
                role="user",
                content=(
                    f"Hypothesis: {experiment.hypothesis}\n\n"
                    f"Procedure:\n{procedure_text}\n\n"
                    f"Expected outcome: {experiment.expected_outcome}\n\n"
                    "Execute each step and report your observations."
                ),
            ),
        ]

        try:
            response = await self._provider.complete(messages=messages, temperature=0.3)
            parsed = self._parse(response.content)
            if parsed:
                experiment.actual_outcome = str(parsed.get("actual_outcome", ""))
            else:
                experiment.actual_outcome = response.content[:500]
        except Exception as exc:
            logger.exception("Experiment run failed: %s", experiment.id)
            experiment.status = "failed"
            experiment.actual_outcome = f"Run failed: {exc}"

        return experiment

    async def evaluate(self, experiment: Experiment) -> Experiment:
        """Compare expected vs actual outcome and produce a conclusion."""
        messages = [
            Message(role="system", content=_EVALUATE_SYSTEM),
            Message(
                role="user",
                content=(
                    f"Hypothesis: {experiment.hypothesis}\n"
                    f"Expected outcome: {experiment.expected_outcome}\n"
                    f"Actual outcome: {experiment.actual_outcome}"
                ),
            ),
        ]

        try:
            response = await self._provider.complete(messages=messages, temperature=0.2)
            parsed = self._parse(response.content)
            if parsed:
                raw_conclusion = parsed.get("conclusion", "inconclusive")
                if raw_conclusion in ("supports", "refutes", "inconclusive"):
                    experiment.conclusion = raw_conclusion  # type: ignore[assignment]
                else:
                    experiment.conclusion = "inconclusive"
                experiment.confidence_delta = float(
                    max(-0.3, min(0.3, parsed.get("confidence_delta", 0.0)))
                )
            else:
                experiment.conclusion = "inconclusive"
        except Exception as exc:
            logger.exception("Experiment evaluation failed: %s", experiment.id)
            experiment.conclusion = "inconclusive"

        experiment.status = "completed"
        experiment.completed_at = datetime.now(timezone.utc).isoformat()
        return experiment

    async def store(self, experiment: Experiment) -> None:
        """Persist the experiment record and update source hypothesis confidence."""
        summary = (
            f"Experiment on hypothesis: '{experiment.hypothesis}'. "
            f"Conclusion: {experiment.conclusion}. "
            f"Expected: {experiment.expected_outcome}. "
            f"Actual: {experiment.actual_outcome}."
        )
        await self._db.store(
            type="experiment",
            content=summary,
            confidence=0.9,
            provenance="experiment:engine",
            tags=[
                f"conclusion:{experiment.conclusion}",
                f"experiment:{experiment.id}",
            ],
        )

        # Adjust the source hypothesis confidence based on conclusion
        if experiment.source_entry_id and experiment.conclusion != "inconclusive":
            source = await self._db.get(experiment.source_entry_id)
            if source:
                new_confidence = source.confidence + experiment.confidence_delta
                await self._db.update_confidence(
                    experiment.source_entry_id,
                    new_confidence,
                    reason=f"experiment:{experiment.id} concluded {experiment.conclusion}",
                )

        await self._event_bus.publish(
            ExperimentCompleted(
                experiment_id=experiment.id,
                conclusion=experiment.conclusion or "inconclusive",
                confidence_delta=experiment.confidence_delta,
            )
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
