# Architecture — IA-test

## Guiding principle

The model (GPT, Claude, etc.) is the **execution engine**, not the intelligence.
The intelligence lives in the architecture we build around it:
the event bus, the memory graph, the uncertainty metrics, the curiosity loop.

If we swap the model tomorrow, the system keeps its memory, its behavior,
and its identity. The model is a replaceable brain; our architecture is the mind.

---

## Component map

```
┌─────────────────────────────────────────────────────────────┐
│                          CORE                                │
│                    (Event Bus + Orchestrator)                │
└──────────┬──────────────────────────────────────┬───────────┘
           │                                      │
    ┌──────▼──────┐                      ┌────────▼────────┐
    │  PROVIDERS  │                      │   SAFETY        │
    │  (OpenAI,   │                      │  (Limits,       │
    │  Anthropic, │                      │   Audit log,    │
    │  Local...)  │                      │   Permissions)  │
    └──────┬──────┘                      └─────────────────┘
           │
    ┌──────▼──────────────────────────────────────────┐
    │                   MEMORY SYSTEM                  │
    │   (Knowledge graph, Confidence, Provenance)      │
    └──┬────────────────┬──────────────┬──────────────┘
       │                │              │
┌──────▼──────┐  ┌──────▼──────┐  ┌───▼──────────────┐
│  CURIOSITY  │  │ EXPLORATION │  │    EXPERIMENTS    │
│  ENGINE     │  │  ENGINE     │  │    ENGINE         │
│  (Metrics → │  │  (Research, │  │  (Hypothesis →    │
│   Questions)│  │   Tools)    │  │   Test → Conclude)│
└──────┬──────┘  └──────┬──────┘  └───┬───────────────┘
       │                │              │
       └────────────────▼──────────────┘
                        │
              ┌─────────▼─────────┐
              │   EVALUATION      │
              │  (Real learning   │
              │   vs illusion)    │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │ SELF-MODIFICATION │
              │  (Inspect →       │
              │   Propose →       │
              │   Branch → Test)  │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │       API         │
              │  (Anthropic-      │
              │   compatible)     │
              └───────────────────┘
```

---

## Event Bus

**Decision:** Internal async pub/sub. No external infrastructure (no Redis, no Kafka).

**Why:** Components need to react to each other without tight coupling.
When Memory learns something, Curiosity should know. When Curiosity generates
a question, Exploration should react. Direct calls would create a dependency
graph impossible to maintain and test.

**Implementation:** Simple `asyncio`-based dispatcher. Events are typed dataclasses.

```python
# Example events
@dataclass
class KnowledgeAcquired:
    entry_id: str
    entry_type: str
    confidence: float

@dataclass
class QuestionGenerated:
    question_id: str
    content: str
    priority_score: float
    trigger: str  # isolation | contradiction | novelty | staleness

@dataclass
class ExplorationCompleted:
    question_id: str
    discoveries: list[str]  # entry IDs

@dataclass
class ProposalGenerated:
    proposal_id: str
    description: str
    # Requires human approval — never auto-applied
```

---

## Model Provider

**Decision:** Protocol-based abstraction. Zero coupling to any specific provider.

```python
class ModelProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
    ) -> Response: ...

    async def stream(
        self,
        messages: list[Message],
    ) -> AsyncIterator[Chunk]: ...
```

**Configuration:** `MODEL_PROVIDER=openai`, `MODEL_NAME=gpt-4o-mini` via env vars.
Switching providers requires zero code changes in business logic.

---

## Memory System

**Decision:** SQLite with raw SQL (no ORM). Transparent, inspectable, portable.

**Why no ORM:** The memory schema evolves as we learn what we need. Raw SQL
keeps migrations explicit and auditable. SQLAlchemy would add abstraction
we don't need yet.

**Key design:** Memory is a typed knowledge graph, not a flat list.

- `memory_entries`: typed facts with confidence and provenance
- `memory_relations`: directed edges between entries (supports, contradicts, related_to)
- `audit_log`: append-only operation log — never modified, never deleted

**Confidence model:**
- Starts at 0.5 for inferred knowledge, 0.8 for experimentally verified
- Increases when corroborated by independent sources
- Decreases when contradicted by new evidence
- Never reaches 1.0 — no claim is treated as absolute truth

---

## Curiosity Engine

**Decision:** Metric-driven curiosity, not prompt-driven.

This is the most important architectural decision in the project.

The difference between simulating curiosity and implementing it:

| Simulation | Computational mechanism |
|------------|------------------------|
| Prompt says "find something interesting" | Metrics detect uncertainty autonomously |
| Human triggers exploration | Background loop runs independently |
| Questions come from instructions | Questions come from memory gap analysis |

**Five uncertainty metrics (computed on memory graph):**

1. `isolation_score`: nodes with no outgoing relations → unexplored territory
2. `confidence_variance`: high variance in a cluster → contested knowledge
3. `contradiction_density`: entries connected by `contradicts` edges
4. `novelty_gradient`: cosine distance from existing knowledge centroid
5. `staleness`: entries not accessed in N curiosity cycles

The LLM generates the actual question text, but the **decision to ask**
and the **priority** are determined by these metrics — not by the model's
inclination.

**Curiosity loop:**
```
Every N seconds (configurable):
  1. Compute uncertainty metrics on memory state
  2. Identify top-K high-scoring knowledge gaps
  3. Generate questions via LLM (metrics as context)
  4. Add to priority queue
  5. Trigger ExplorationEngine for top priority
  6. Wait for result, update memory
  7. Emit events, repeat
```

Respects `MAX_AUTONOMOUS_CYCLES` from safety config.

---

## Self-Modification Engine

**Decision:** Git branch isolation + mandatory human approval gate.

The system can propose and implement changes to itself, but cannot apply them.

```
inspect (read-only)
  → propose (stored as PROPOSED in memory)
    → human approves
      → create branch (experiment/auto-YYYYMMDD-HHMMSS)
        → implement
          → run tests
            → compare metrics
              → store result (PASSED or FAILED)
                → await merge decision (human)
```

**Hard constraints enforced by Safety module:**
- Cannot write to `src/ia_test/safety/`
- Cannot modify `audit_log` table
- Cannot modify its own permission configuration
- Cannot auto-merge any branch
- Branch names must match `experiment/auto-*` pattern

---

## Safety Module

**Decision:** Safety is a separate, hardened module. No other module imports
from it in a way that could allow modification. It is the only module
that imports nothing from the rest of the system.

**Components:**
- `RateLimiter`: enforces API call limits per minute/hour/day
- `CycleGuard`: stops autonomous loops after `MAX_AUTONOMOUS_CYCLES`
- `PermissionChecker`: validates every file write against an allowlist
- `AuditLogger`: append-only operation recorder
- `SandboxManager`: isolated environment for experiments

**Configuration (env vars, read-only after startup):**
```
MAX_AUTONOMOUS_CYCLES=50
MAX_API_CALLS_PER_HOUR=100
MAX_API_CALLS_PER_DAY=500
ALLOWED_WRITE_PATHS=src/,tests/,docs/
FORBIDDEN_WRITE_PATHS=src/ia_test/safety/
```

---

## Evaluation Engine

**Key problem:** A system can appear to learn while actually hallucinating.

**Strategy:**
- Every claim must be traceable to a memory entry
- Confidence is calibrated against experimental results, not self-assessment
- Before/after snapshots measure whether uncertainty actually decreased
- The system cannot evaluate its own self-modification — that uses an independent evaluation run

**Metric:** `learning_delta = uncertainty_before - uncertainty_after`
If `learning_delta <= 0`, the exploration cycle produced no verifiable improvement.

---

## Technology decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Language | Python 3.11+ | Async-native, AI ecosystem |
| Web framework | FastAPI | Async, OpenAPI, easy SSE |
| Database | SQLite | Local, portable, inspectable |
| ORM | None (raw SQL) | Transparent migrations |
| Testing | Pytest + pytest-asyncio | Standard, composable |
| Config | pydantic-settings | Typed env var validation |
| Async | asyncio (stdlib) | No extra dependencies |
| Event bus | Custom (asyncio) | No external infra needed |
