"""
CuriosityLoop — the autonomous background task.

Runs independently of human interaction. No human needs to ask it to explore.
The loop itself decides when to generate questions based on memory state.
Safety: bounded by MAX_AUTONOMOUS_CYCLES from settings.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..core.event_bus import EventBus
from ..core.events import ExplorationTriggered, QuestionGenerated
from ..core.settings import Settings
from ..memory.database import MemoryDatabase
from ..providers.base import ModelProvider
from ..safety.limits import RateLimiter, RateLimitExceeded
from .generator import QuestionGenerator
from .models import Question
from .scorer import UncertaintyScorer

if TYPE_CHECKING:
    from ..exploration.engine import ExplorationEngine

logger = logging.getLogger(__name__)


class CuriosityLoop:
    def __init__(
        self,
        db: MemoryDatabase,
        provider: ModelProvider,
        event_bus: EventBus,
        settings: Settings,
        rate_limiter: RateLimiter | None = None,
        exploration_engine: ExplorationEngine | None = None,
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
        self._exploration_engine = exploration_engine
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

        quota = self._settings.curiosity_questions_per_cycle

        # 1. Resume pending questions from previous cycles
        pending_rows = await self._db.get_pending_questions(limit=quota)
        questions: list[Question] = [
            Question(
                id=r["id"],
                content=r["content"],
                priority_score=r["priority_score"],
                trigger=r["trigger"],  # type: ignore[arg-type]
                source_entry_id=r["source_entry_id"],
                status="pending",
                created_at=r["created_at"],
            )
            for r in pending_rows
        ]

        # 2. Fill remaining quota with freshly generated questions
        still_needed = quota - len(questions)
        if still_needed > 0:
            report = await self._scorer.compute(limit=still_needed * 3)
            if report.total_entries > 0:
                new_questions = await self._generator.generate(report, max_questions=still_needed)
                for q in new_questions:
                    await self._db.store_question(
                        question_id=q.id,
                        content=q.content,
                        priority_score=q.priority_score,
                        trigger=q.trigger,
                        source_entry_id=q.source_entry_id,
                    )
                questions.extend(new_questions)

        if not questions:
            logger.debug("Memory empty — no questions to generate yet")
            self._cycles_run += 1
            return []

        self._cycles_run += 1
        logger.info(
            "Curiosity cycle %d — %d questions (%d resumed, %d new)",
            self._cycles_run,
            len(questions),
            len(pending_rows),
            len(questions) - len(pending_rows),
        )

        discoveries_total = 0
        for question in questions:
            await self._event_bus.publish(
                QuestionGenerated(
                    question_id=question.id,
                    content=question.content,
                    priority_score=question.priority_score,
                    trigger=question.trigger,
                )
            )

            if self._exploration_engine is not None:
                # Each exploration call is a real provider call — count it against limits
                try:
                    await self._rate_limiter.check_and_record()
                except RateLimitExceeded as exc:
                    logger.warning("Rate limit hit during exploration — stopping cycle early: %s", exc)
                    break

                await self._event_bus.publish(ExplorationTriggered(question_id=question.id))
                try:
                    result = await self._exploration_engine.explore(question)
                    discoveries_total += len(result.discoveries)
                    new_status = "answered" if result.status == "completed" else "failed"
                    await self._db.update_question_status(question.id, new_status)
                    logger.info(
                        "Explored question %s — status=%s, discoveries=%d",
                        question.id[:8],
                        result.status,
                        len(result.discoveries),
                    )
                except Exception as exc:
                    logger.error("Exploration error for question %s: %s", question.id[:8], exc)
                    await self._db.update_question_status(question.id, "failed")
            else:
                # No engine wired — publish the trigger so external subscribers can handle it
                if question.priority_score > 0.5:
                    await self._event_bus.publish(ExplorationTriggered(question_id=question.id))

        if discoveries_total:
            logger.info(
                "Cycle %d complete — %d questions, %d discoveries added to memory",
                self._cycles_run,
                len(questions),
                discoveries_total,
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
