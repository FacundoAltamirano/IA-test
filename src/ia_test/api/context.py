from __future__ import annotations
from ia_test.api.models import RequestMessage
from ia_test.memory.database import MemoryDatabase


class ContextInjector:
    """Enriches a system prompt with relevant entries from the memory graph."""

    _MAX_ENTRIES = 5

    def __init__(self, db: MemoryDatabase) -> None:
        self._db = db

    async def enrich(
        self,
        messages: list[RequestMessage],
        existing_system: str | None,
    ) -> str:
        last_user = next(
            (m for m in reversed(messages) if m.role == "user"), None
        )
        if last_user is None:
            return existing_system or ""

        query_words = {
            w for w in last_user.as_text().lower().split() if len(w) > 4
        }
        if not query_words:
            return existing_system or ""

        all_entries = await self._db.query(limit=100)
        relevant = [
            e
            for e in all_entries
            if e.confidence >= 0.5
            and bool(
                query_words
                & {w for w in e.content.lower().split() if len(w) > 4}
            )
        ]
        relevant.sort(key=lambda e: e.confidence, reverse=True)
        relevant = relevant[: self._MAX_ENTRIES]

        if not relevant:
            return existing_system or ""

        lines = ["[Memory context — relevant knowledge:]"]
        for entry in relevant:
            lines.append(f"- {entry.content}  (confidence {entry.confidence:.2f})")
        lines.append("[End memory context]")
        block = "\n".join(lines)

        return f"{existing_system}\n\n{block}" if existing_system else block
