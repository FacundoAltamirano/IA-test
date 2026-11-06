# RON — Autonomous Cognition & Self-Awareness

Branch: main
Delivery: single-pr (feature incrementally committed to main)
Test runner: `python -m pytest`

## Specs

S1 — Active memory via function calling
RON can call 4 real SQLite tools (search_memory, get_memory, explore_relations,
get_recent_experiments) during a conversation via OpenAI/Anthropic function calling.
Server runs a 3-iteration tool loop with audit logging and rate-limit awareness.

S2 — Context injection always shows recent autonomous discoveries
ContextInjector always injects the 3 most recent entries (by updated_at, confidence >= 0.3)
plus up to 5 keyword-matched entries, deduped by ID, max 8 total.

S3 — Knowledge graph connectivity (relations between entries)
DiscoveryRecorder.link_to_related evaluates up to 200 entries (not 20) so the graph
actually builds edges. Each discovery also creates a derived_from relation back to the
source entry that originated the question (question.source_entry_id).

S4 — Question persistence (pending)
Questions survive between CuriosityLoop cycles. Stored in SQLite, picked up by the scorer
in subsequent cycles. Prevents re-exploring the same ground.

S5 — Voice engine with SSML self-modification (pending)
RON generates SSML instead of plain text. Can read and propose changes to its own narration
style templates stored in memory. Supports pauses, pitch, rate, and sarcasm/laughter markers.

## Tasks

T1 [S1, S2] inline — Active memory + context injection — DONE — commit 805d5dd
T2 [S3] inline — Knowledge graph connectivity fix (limit 20→200 + derived_from) — DONE — commit pending
T3 [S4] inline — Question persistence in SQLite — DONE — commit pending
T4 [S5] inline — SSML voice engine with self-modification — DONE — commit pending

## Log

L1 (original): "quiero que RON tenga memoria activa, pueda usar sus descubrimientos como origen de nuevas
investigaciones, y eventualmente tenga un motor de voz automodificable"

L2: Diagnostic audit completed (2026-11-06). Confirmed: graph has no edges due to limit=20 bug;
no derived_from relation from discovery to source entry; questions are ephemeral (never persisted).

L3: T2 fix implemented — link_to_related limit 20→200, derived_from relation added when
source_entry_id is provided, engine passes question.source_entry_id. 130/130 tests pass.
