from __future__ import annotations
from ia_test.api.models import RequestMessage
from ia_test.memory.database import MemoryDatabase


class ContextInjector:
    """Enriches a system prompt with relevant entries from the memory graph.

    Two injection sources are merged and deduplicated:
    1. Recent entries — always included regardless of keyword match, so RON sees
       its own autonomous discoveries even when the user's words don't match.
    2. Keyword-matched entries — entries whose content overlaps with the last
       user message, ranked by confidence.
    """

    _MAX_RECENT = 3
    _MAX_KEYWORD = 5
    _MAX_TOTAL = 8
    _RECENT_MIN_CONFIDENCE = 0.3
    _KEYWORD_MIN_CONFIDENCE = 0.5

    def __init__(self, db: MemoryDatabase) -> None:
        self._db = db

    async def enrich(
        self,
        messages: list[RequestMessage],
        existing_system: str | None,
    ) -> str:
        all_entries = await self._db.query(limit=200)
        if not all_entries:
            return existing_system or ""

        seen_ids: set[str] = set()
        relevant: list = []

        # 1. Most recent entries (by updated_at) above the low confidence threshold
        recent_sorted = sorted(
            (e for e in all_entries if e.confidence >= self._RECENT_MIN_CONFIDENCE),
            key=lambda e: e.updated_at,
            reverse=True,
        )
        for entry in recent_sorted[: self._MAX_RECENT]:
            seen_ids.add(entry.id)
            relevant.append(entry)

        # 2. Keyword-matched entries from the last user message
        last_user = next(
            (m for m in reversed(messages) if m.role == "user"), None
        )
        if last_user:
            query_words = {
                w for w in last_user.as_text().lower().split() if len(w) > 4
            }
            if query_words:
                keyword_matches = [
                    e
                    for e in all_entries
                    if e.id not in seen_ids
                    and e.confidence >= self._KEYWORD_MIN_CONFIDENCE
                    and bool(
                        query_words
                        & {w for w in e.content.lower().split() if len(w) > 4}
                    )
                ]
                keyword_matches.sort(key=lambda e: e.confidence, reverse=True)
                for entry in keyword_matches[: self._MAX_KEYWORD]:
                    seen_ids.add(entry.id)
                    relevant.append(entry)

        relevant = relevant[: self._MAX_TOTAL]

        if not relevant:
            return existing_system or ""

        lines = ["[Memory context — relevant knowledge:]"]
        for entry in relevant:
            lines.append(
                f"- [{entry.type}] {entry.content}"
                f"  (confidence {entry.confidence:.2f}, source: {entry.provenance})"
            )
        lines.append("[End memory context]")
        block = "\n".join(lines)

        return f"{existing_system}\n\n{block}" if existing_system else block
