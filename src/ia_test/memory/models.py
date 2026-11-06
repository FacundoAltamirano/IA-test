from dataclasses import dataclass, field
from typing import Literal

EntryType = Literal["knowledge", "hypothesis", "experience", "error", "experiment"]
RelationType = Literal["supports", "contradicts", "related_to", "derived_from"]
Provenance = Literal["exploration", "experiment", "human", "inference", "unknown"]


@dataclass
class MemoryEntry:
    id: str
    type: EntryType
    content: str
    confidence: float
    provenance: Provenance
    tags: list[str]
    created_at: str
    updated_at: str


@dataclass
class MemoryRelation:
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float
    created_at: str


@dataclass
class KnowledgeGap:
    """An entry that the curiosity engine should investigate."""
    entry_id: str
    content: str
    reason: str   # isolated | low_confidence | contradicted | stale
    score: float  # higher = more urgent


@dataclass
class Contradiction:
    source: MemoryEntry
    target: MemoryEntry
    relation_id: str
