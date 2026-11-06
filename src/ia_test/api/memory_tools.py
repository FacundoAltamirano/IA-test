"""Memory tools exposed to the model for active knowledge-graph querying.

Each tool executes a real SQLite query — no simulated or fabricated results.
"""
from __future__ import annotations

import json
import logging

from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)

MEMORY_TOOLS: list[Tool] = [
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
