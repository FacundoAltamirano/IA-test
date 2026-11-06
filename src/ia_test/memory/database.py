import aiosqlite
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    Contradiction,
    EntryType,
    KnowledgeGap,
    MemoryEntry,
    MemoryRelation,
    Provenance,
    RelationType,
)

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

LOW_CONFIDENCE_THRESHOLD = 0.4
ISOLATION_SCORE_WEIGHT = 0.5
LOW_CONFIDENCE_WEIGHT = 0.3
CONTRADICTION_WEIGHT = 0.2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


class MemoryDatabase:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._apply_schema()
        logger.info("Memory database connected: %s", self._path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _apply_schema(self) -> None:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        await self._db.executescript(schema)
        await self._db.commit()

    # ── Entries ──────────────────────────────────────────────────────────────

    async def store(
        self,
        type: EntryType,
        content: str,
        confidence: float = 0.5,
        provenance: Provenance = "unknown",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=_uid(),
            type=type,
            content=content,
            confidence=confidence,
            provenance=provenance,
            tags=tags or [],
            created_at=_now(),
            updated_at=_now(),
        )
        await self._db.execute(
            """
            INSERT INTO memory_entries
                (id, type, content, confidence, provenance, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.type,
                entry.content,
                entry.confidence,
                entry.provenance,
                json.dumps(entry.tags),
                entry.created_at,
                entry.updated_at,
            ),
        )
        await self._db.commit()
        await self._audit("store_entry", {"entry_id": entry.id, "type": type})
        return entry

    async def get(self, entry_id: str) -> MemoryEntry | None:
        async with self._db.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _row_to_entry(row) if row else None

    async def query(
        self,
        type: EntryType | None = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        if type is not None:
            sql = """
                SELECT * FROM memory_entries
                WHERE type = ? AND confidence >= ?
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
            """
            params = (type, min_confidence, limit)
        else:
            sql = """
                SELECT * FROM memory_entries
                WHERE confidence >= ?
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
            """
            params = (min_confidence, limit)

        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_entry(r) for r in rows]

    async def update_confidence(
        self, entry_id: str, new_confidence: float, reason: str = ""
    ) -> None:
        new_confidence = max(0.0, min(1.0, new_confidence))
        await self._db.execute(
            "UPDATE memory_entries SET confidence = ?, updated_at = ? WHERE id = ?",
            (new_confidence, _now(), entry_id),
        )
        await self._db.commit()
        await self._audit(
            "update_confidence",
            {"entry_id": entry_id, "confidence": new_confidence, "reason": reason},
        )

    # ── Relations ─────────────────────────────────────────────────────────────

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
    ) -> MemoryRelation:
        relation = MemoryRelation(
            id=_uid(),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            created_at=_now(),
        )
        await self._db.execute(
            """
            INSERT INTO memory_relations
                (id, source_id, target_id, relation_type, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                relation.id,
                relation.source_id,
                relation.target_id,
                relation.relation_type,
                relation.weight,
                relation.created_at,
            ),
        )
        await self._db.commit()
        return relation

    # ── Curiosity feed ────────────────────────────────────────────────────────

    async def get_knowledge_gaps(self, limit: int = 10) -> list[KnowledgeGap]:
        """
        Identify entries the curiosity engine should investigate.
        Scoring: isolation (no outgoing relations) + low confidence + contradictions.
        """
        gaps: list[KnowledgeGap] = []

        # Isolated: entries with zero outgoing relations
        async with self._db.execute(
            """
            SELECT e.id, e.content, e.confidence
            FROM memory_entries e
            LEFT JOIN memory_relations r ON e.id = r.source_id
            WHERE r.id IS NULL
            ORDER BY e.confidence ASC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            for row in await cursor.fetchall():
                score = ISOLATION_SCORE_WEIGHT + (
                    LOW_CONFIDENCE_WEIGHT * (1.0 - row["confidence"])
                )
                gaps.append(
                    KnowledgeGap(
                        entry_id=row["id"],
                        content=row["content"],
                        reason="isolated",
                        score=round(score, 4),
                    )
                )

        # Low confidence: entries below threshold that have some relations
        async with self._db.execute(
            """
            SELECT e.id, e.content, e.confidence
            FROM memory_entries e
            INNER JOIN memory_relations r ON e.id = r.source_id
            WHERE e.confidence < ?
            GROUP BY e.id
            ORDER BY e.confidence ASC
            LIMIT ?
            """,
            (LOW_CONFIDENCE_THRESHOLD, limit),
        ) as cursor:
            for row in await cursor.fetchall():
                score = LOW_CONFIDENCE_WEIGHT * (1.0 - row["confidence"])
                gaps.append(
                    KnowledgeGap(
                        entry_id=row["id"],
                        content=row["content"],
                        reason="low_confidence",
                        score=round(score, 4),
                    )
                )

        seen = set()
        unique = []
        for g in sorted(gaps, key=lambda x: x.score, reverse=True):
            if g.entry_id not in seen:
                seen.add(g.entry_id)
                unique.append(g)
        return unique[:limit]

    async def find_contradictions(self) -> list[Contradiction]:
        async with self._db.execute(
            """
            SELECT r.id as rel_id,
                   e1.id as s_id, e1.type as s_type, e1.content as s_content,
                   e1.confidence as s_conf, e1.provenance as s_prov,
                   e1.tags as s_tags, e1.created_at as s_created, e1.updated_at as s_updated,
                   e2.id as t_id, e2.type as t_type, e2.content as t_content,
                   e2.confidence as t_conf, e2.provenance as t_prov,
                   e2.tags as t_tags, e2.created_at as t_created, e2.updated_at as t_updated
            FROM memory_relations r
            JOIN memory_entries e1 ON r.source_id = e1.id
            JOIN memory_entries e2 ON r.target_id = e2.id
            WHERE r.relation_type = 'contradicts'
            """
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            Contradiction(
                source=MemoryEntry(
                    id=r["s_id"], type=r["s_type"], content=r["s_content"],
                    confidence=r["s_conf"], provenance=r["s_prov"],
                    tags=json.loads(r["s_tags"]), created_at=r["s_created"],
                    updated_at=r["s_updated"],
                ),
                target=MemoryEntry(
                    id=r["t_id"], type=r["t_type"], content=r["t_content"],
                    confidence=r["t_conf"], provenance=r["t_prov"],
                    tags=json.loads(r["t_tags"]), created_at=r["t_created"],
                    updated_at=r["t_updated"],
                ),
                relation_id=r["rel_id"],
            )
            for r in rows
        ]

    # ── Audit ─────────────────────────────────────────────────────────────────

    async def _audit(self, operation: str, details: dict) -> None:
        await self._db.execute(
            "INSERT INTO audit_log (id, operation, actor, details, timestamp) VALUES (?, ?, ?, ?, ?)",
            (_uid(), operation, "system", json.dumps(details), _now()),
        )
        await self._db.commit()


def _row_to_entry(row: aiosqlite.Row) -> MemoryEntry:
    return MemoryEntry(
        id=row["id"],
        type=row["type"],
        content=row["content"],
        confidence=row["confidence"],
        provenance=row["provenance"],
        tags=json.loads(row["tags"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
