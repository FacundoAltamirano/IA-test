"""Memory tools exposed to the model for active knowledge-graph querying.

Each tool executes a real SQLite query — no simulated or fabricated results.
"""
from __future__ import annotations

import json
import logging

from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Tool, ToolCall, ToolResult
from ia_test.voice.models import VoiceStyle

logger = logging.getLogger(__name__)

_VOICE_STYLE_TAG = "voice:style"
_VALID_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}

MEMORY_TOOLS: list[Tool] = [
    Tool(
        name="update_voice_style",
        description=(
            "Read or update your own voice narration style. "
            "Call with no arguments to read current style. "
            "Call with any subset of the fields to update them. "
            "Changes take effect on the next audio synthesis request. "
            "Available voices: alloy, echo, fable, onyx, nova, shimmer. "
            "speed is a float from 0.25 to 4.0 (1.0 = normal). "
            "pause_marker and long_pause_marker control how [pausa] and [pausa:larga] render. "
            "laugh_soft and laugh_loud control how [risa:suave] and [risa:fuerte] render. "
            "style_notes is free text guidance about your narration character."
        ),
        parameters={
            "type": "object",
            "properties": {
                "voice": {
                    "type": "string",
                    "description": "OpenAI TTS voice name",
                    "enum": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                },
                "speed": {
                    "type": "number",
                    "description": "Narration speed, 0.25 to 4.0",
                },
                "pause_marker": {
                    "type": "string",
                    "description": "Text substituted for [pausa] annotation",
                },
                "long_pause_marker": {
                    "type": "string",
                    "description": "Text substituted for [pausa:larga] annotation",
                },
                "laugh_soft": {
                    "type": "string",
                    "description": "Text substituted for [risa:suave] annotation",
                },
                "laugh_loud": {
                    "type": "string",
                    "description": "Text substituted for [risa:fuerte] annotation",
                },
                "style_notes": {
                    "type": "string",
                    "description": "Free-form narration character description",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="search_memory",
        description=(
            "Search your persistent knowledge graph by keyword. "
            "Returns entries whose content matches the query. "
            "Use this to recall what you know about a topic before answering, "
            "or to check whether you have explored something before."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for in memory entries",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10, max 50)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_memory",
        description=(
            "Retrieve a specific memory entry by its ID. "
            "Use this when you have an ID from a previous search and need the full content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The UUID of the memory entry to retrieve",
                },
            },
            "required": ["entry_id"],
        },
    ),
    Tool(
        name="explore_relations",
        description=(
            "Find all memory entries connected to a given entry. "
            "Returns the connected entries and the relationship type "
            "(supports, contradicts, related_to, derived_from). "
            "Use this to navigate your knowledge graph and find connected ideas."
        ),
        parameters={
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The UUID of the entry to explore relations for",
                },
            },
            "required": ["entry_id"],
        },
    ),
    Tool(
        name="get_recent_experiments",
        description=(
            "Retrieve your most recent autonomous explorations and experiments from memory. "
            "Use this to recall what you have been investigating and what you discovered "
            "without being prompted by the user."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent entries to return (default 10)",
                    "default": 10,
                },
            },
            "required": [],
        },
    ),
]


class MemoryToolExecutor:
    """Executes tool calls against the real SQLite knowledge graph."""

    def __init__(self, db: MemoryDatabase) -> None:
        self._db = db

    async def execute(self, call: ToolCall) -> ToolResult:
        logger.info("Tool call: %s args=%s", call.name, call.arguments)
        try:
            match call.name:
                case "update_voice_style":
                    result = await self._update_voice_style(call.arguments)
                case "search_memory":
                    result = await self._search_memory(call.arguments)
                case "get_memory":
                    result = await self._get_memory(call.arguments)
                case "explore_relations":
                    result = await self._explore_relations(call.arguments)
                case "get_recent_experiments":
                    result = await self._get_recent_experiments(call.arguments)
                case _:
                    result = {"error": f"Unknown tool: {call.name}"}
        except Exception as exc:
            logger.error("Tool %s failed: %s", call.name, exc)
            result = {"error": str(exc)}

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=json.dumps(result, ensure_ascii=False),
        )

    async def _update_voice_style(self, args: dict) -> dict:
        # Load current style from memory
        all_entries = await self._db.query(limit=500)
        existing_entry = None
        current_style = VoiceStyle()

        for entry in all_entries:
            if _VOICE_STYLE_TAG in entry.tags:
                existing_entry = entry
                try:
                    current_style = VoiceStyle.from_dict(json.loads(entry.content))
                except (json.JSONDecodeError, KeyError):
                    pass
                break

        # Read-only call — no args provided
        if not args:
            return {
                "action": "read",
                "current_style": current_style.to_dict(),
                "annotations": sorted(VoiceStyle.KNOWN_ANNOTATIONS),
            }

        # Validate and apply updates
        updated = current_style.to_dict()
        errors = []

        if "voice" in args:
            v = str(args["voice"])
            if v not in _VALID_VOICES:
                errors.append(f"Invalid voice '{v}'. Must be one of: {sorted(_VALID_VOICES)}")
            else:
                updated["voice"] = v

        if "speed" in args:
            speed = float(args["speed"])
            if not 0.25 <= speed <= 4.0:
                errors.append("speed must be between 0.25 and 4.0")
            else:
                updated["speed"] = round(speed, 2)

        for field in ("pause_marker", "long_pause_marker", "laugh_soft", "laugh_loud", "style_notes"):
            if field in args:
                updated[field] = str(args[field])

        if errors:
            return {"error": "; ".join(errors), "current_style": current_style.to_dict()}

        # Persist the new style in memory
        new_content = json.dumps(updated, ensure_ascii=False)
        tags = [_VOICE_STYLE_TAG, "voice:config"]

        if existing_entry:
            await self._db._db.execute(
                "UPDATE memory_entries SET content = ?, updated_at = datetime('now') WHERE id = ?",
                (new_content, existing_entry.id),
            )
            await self._db._db.commit()
        else:
            await self._db.store(
                type="experience",
                content=new_content,
                confidence=1.0,
                provenance="voice:self_modification",
                tags=tags,
            )

        return {
            "action": "updated",
            "previous_style": current_style.to_dict(),
            "new_style": updated,
        }

    async def _search_memory(self, args: dict) -> dict:
        query = str(args.get("query", ""))
        limit = min(int(args.get("limit", 10)), 50)

        all_entries = await self._db.query(limit=500)
        query_words = {w.lower() for w in query.split() if len(w) > 2}

        matches = []
        for entry in all_entries:
            entry_words = {w.lower() for w in entry.content.split() if len(w) > 2}
            if query_words & entry_words:
                matches.append(entry)

        matches.sort(key=lambda e: e.confidence, reverse=True)
        matches = matches[:limit]

        return {
            "query": query,
            "total_found": len(matches),
            "entries": [
                {
                    "id": e.id,
                    "type": e.type,
                    "content": e.content,
                    "confidence": e.confidence,
                    "provenance": e.provenance,
                    "tags": e.tags,
                }
                for e in matches
            ],
        }

    async def _get_memory(self, args: dict) -> dict:
        entry_id = str(args.get("entry_id", ""))
        entry = await self._db.get(entry_id)
        if entry is None:
            return {"error": f"No entry found with id: {entry_id}"}
        return {
            "id": entry.id,
            "type": entry.type,
            "content": entry.content,
            "confidence": entry.confidence,
            "provenance": entry.provenance,
            "tags": entry.tags,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }

    async def _explore_relations(self, args: dict) -> dict:
        entry_id = str(args.get("entry_id", ""))
        source = await self._db.get(entry_id)
        if source is None:
            return {"error": f"No entry found with id: {entry_id}"}

        relations: list[dict] = []
        async with self._db._db.execute(
            """
            SELECT r.id, r.relation_type, r.weight,
                   e.id as connected_id, e.type as connected_type,
                   e.content as connected_content, e.confidence as connected_confidence
            FROM memory_relations r
            JOIN memory_entries e ON r.target_id = e.id
            WHERE r.source_id = ?
            UNION ALL
            SELECT r.id, r.relation_type, r.weight,
                   e.id as connected_id, e.type as connected_type,
                   e.content as connected_content, e.confidence as connected_confidence
            FROM memory_relations r
            JOIN memory_entries e ON r.source_id = e.id
            WHERE r.target_id = ?
            """,
            (entry_id, entry_id),
        ) as cursor:
            for row in await cursor.fetchall():
                relations.append({
                    "relation_id": row["id"],
                    "relation_type": row["relation_type"],
                    "weight": row["weight"],
                    "connected_entry_id": row["connected_id"],
                    "connected_entry_type": row["connected_type"],
                    "connected_entry_content": row["connected_content"],
                    "connected_entry_confidence": row["connected_confidence"],
                })

        return {
            "source_entry": {
                "id": source.id,
                "type": source.type,
                "content": source.content,
                "confidence": source.confidence,
            },
            "relations_count": len(relations),
            "relations": relations,
        }

    async def _get_recent_experiments(self, args: dict) -> dict:
        limit = min(int(args.get("limit", 10)), 50)
        async with self._db._db.execute(
            """
            SELECT id, type, content, confidence, provenance, tags, created_at
            FROM memory_entries
            WHERE provenance LIKE 'exploration:%'
               OR provenance = 'experiment'
               OR type = 'experiment'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

        return {
            "total_found": len(rows),
            "entries": [
                {
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "provenance": row["provenance"],
                    "tags": json.loads(row["tags"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
        }
