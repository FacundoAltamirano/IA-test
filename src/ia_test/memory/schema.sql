-- Memory System schema
-- Raw SQL — no ORM. Migrations are explicit and auditable.

CREATE TABLE IF NOT EXISTS memory_entries (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL CHECK (type IN ('knowledge', 'hypothesis', 'experience', 'error', 'experiment')),
    content     TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    provenance  TEXT NOT NULL DEFAULT 'unknown',  -- exploration | experiment | human | inference
    tags        TEXT NOT NULL DEFAULT '[]',        -- JSON array of strings
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_relations (
    id            TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    target_id     TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL CHECK (relation_type IN ('supports', 'contradicts', 'related_to', 'derived_from')),
    weight        REAL NOT NULL DEFAULT 1.0 CHECK (weight > 0.0),
    created_at    TEXT NOT NULL
);

-- Append-only audit log. Never modified, never deleted.
CREATE TABLE IF NOT EXISTS audit_log (
    id        TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    actor     TEXT NOT NULL,
    details   TEXT NOT NULL DEFAULT '{}',  -- JSON
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_type       ON memory_entries(type);
CREATE INDEX IF NOT EXISTS idx_entries_confidence ON memory_entries(confidence);
CREATE INDEX IF NOT EXISTS idx_relations_source   ON memory_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target   ON memory_relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type     ON memory_relations(relation_type);
