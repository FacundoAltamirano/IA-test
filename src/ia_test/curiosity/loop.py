"""
CuriosityLoop — the autonomous background task.

Runs independently of human interaction. No human needs to ask it to explore.
The loop itself decides when to generate questions based on memory state.
Safety: bounded by MAX_AUTONOMOUS_CYCLES from settings.
"""

import asyncio
import logging

from ..core.event_bus import EventBus
from ..core.events import ExplorationTriggered, QuestionGenerated
from ..core.settings import Settings
from ..memory.database import MemoryDatabase
from ..providers.base import ModelProvider
from ..safety.limits import RateLimiter, RateLimitExceeded
from .generator import QuestionGenerator
from .models import Question
from .scorer import UncertaintyScorer

logger = logging.getLogger(__name__)


class CuriosityLoop:
    def __init__(
        self,
        db: MemoryDatabase,
        provider: ModelProvider,
        event_bus: EventBus,
        settings: Settings,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._db = db
        self._scorer = UncertaintyScorer(db)
        self._generator = QuestionGenerator(provider)
        self._event_bus = event_bus
        self._settings = settings
        self._rate_limiter = rate_limiter or RateLimiter(
            max_per_hour=settings.max_api_calls_per_hour,
            max_per_day=settings.max_api_calls_per_day,
        )
        self._cycles_run = 0
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="curiosity-loop")
        logger.info(
            "Curiosity loop started — interval=%ds, max_cycles=%d",
            self._settings.curiosity_loop_interval_seconds,
            self._settings.max_autonomous_cycles,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Curiosity loop stopped after %d cycles", self._cycles_run)

    async def run_once(self) -> list[Question]:
        """Execute a single curiosity cycle. Used directly in tests."""
        if self._cycles_run >= self._settings.max_autonomous_cycles:
            logger.warning("Max autonomous cycles (%d) reached", self._settings.max_autonomous_cycles)
            return []

        try:
            await self._rate_limiter.check_and_record()
        except RateLimitExceeded as exc:
            logger.warning("Curiosity loop rate-limited: %s", exc)
            return []

        report = await self._scorer.compute(
            limit=self._settings.curiosity_questions_per_cycle * 3
        )

        if report.total_entries == 0:
            logger.debug("Memory empty — no questions to generate yet")
            self._cycles_run += 1
            return []

        questions = await self._generator.generate(
            report, max_questions=self._settings.curiosity_questions_per_cycle
        )

        self._cycles_run += 1
        logger.info(
            "Curiosity cycle %d — uncertainty=%.3f, questions=%d",
            self._cycles_run,
            report.overall_uncertainty,
            len(questions),
        )

        for question in questions:
            await self._event_bus.publish(
                QuestionGenerated(
                    question_id=question.id,
                    content=question.content,
                    priority_score=question.priority_score,
                    trigger=question.trigger,
                )
            )
            if question.priority_score > 0.5:
                await self._event_bus.publish(
                    ExplorationTriggered(question_id=question.id)
                )

        return questions

    async def _loop(self) -> None:
        while self._running:
            if self._cycles_run >= self._settings.max_autonomous_cycles:
                logger.info("Curiosity loop reached max cycles — stopping autonomously")
                self._running = False
                break

            await self.run_once()
            await asyncio.sleep(self._settings.curiosity_loop_interval_seconds)

    @property
    def cycles_run(self) -> int:
        return self._cycles_run
