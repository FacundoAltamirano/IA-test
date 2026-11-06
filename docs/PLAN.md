# Development Plan — IA-test

## Objective

Build an experimental AI system with **persistent exploratory motivation**:
a computational mechanism that drives the system to detect uncertainties,
formulate hypotheses, and explore knowledge autonomously — not because a
human told it to, but because that drive is embedded in its architecture.

**Fundamental distinction:**
- Simulating curiosity via instructions → NOT what we want
- Persistent computational motivations → YES
- Verifiable and measurable learning → YES

---

## Phase 1 — Core + Model Provider

**Goal:** A working system skeleton that can communicate with an external model.

### Tasks
- [ ] Define core event bus (`EventBus`, `Event`, `Subscriber`)
- [ ] Define `ModelProvider` protocol (abstract interface)
- [ ] Implement `OpenAIProvider` (GPT-4o-mini to start)
- [ ] Implement `Core` orchestrator that initializes and wires components
- [ ] Environment configuration (`.env`, `Settings` with pydantic-settings)
- [ ] Basic pytest setup
- [ ] Tests: provider returns a valid response, event bus delivers events

### Acceptance criteria
- `Core` starts without errors
- `OpenAIProvider.complete()` returns a structured response
- Event published on bus reaches all subscribers
- All tests pass

### Notes
- Model must be configurable via `MODEL_PROVIDER` and `MODEL_NAME` env vars
- No business logic here — only plumbing

---

## Phase 2 — Persistent Memory

**Goal:** A typed memory system that stores knowledge, experiences, hypotheses,
and errors with confidence scores and provenance tracking.

### Schema (SQLite)

```sql
-- Typed knowledge store
memory_entries (
  id TEXT PRIMARY KEY,
  type TEXT,          -- knowledge | hypothesis | experience | error | experiment
  content TEXT,
  confidence REAL,    -- 0.0 to 1.0
  provenance TEXT,    -- source: exploration | experiment | human | inference
  tags TEXT,          -- JSON array
  created_at TEXT,
  updated_at TEXT
)

-- Relationships between entries
memory_relations (
  id TEXT PRIMARY KEY,
  source_id TEXT,
  target_id TEXT,
  relation_type TEXT, -- supports | contradicts | related_to | derived_from
  weight REAL,
  created_at TEXT
)

-- Audit log (append-only, never modified)
audit_log (
  id TEXT PRIMARY KEY,
  operation TEXT,
  actor TEXT,
  details TEXT,       -- JSON
  timestamp TEXT
)
```

### Tasks
- [ ] Implement SQLite migrations (no ORM — raw SQL for transparency)
- [ ] Implement `MemorySystem` with `store()`, `query()`, `update_confidence()`
- [ ] Implement `get_knowledge_gaps()` — identifies isolated nodes and low-confidence entries
- [ ] Implement `find_contradictions()` — entries that conflict with each other
- [ ] Tests: store, query by type, confidence update, gap detection

### Acceptance criteria
- Memory persists across process restarts
- `get_knowledge_gaps()` returns entries with no outgoing relations or confidence < 0.4
- `find_contradictions()` finds entries with `contradicts` relations

---

## Phase 3 — Curiosity Engine

**Goal:** A persistent background mechanism that autonomously detects
uncertainty, generates questions, and prioritizes exploration — without human
input.

### How it works (not via prompt, via metrics)

```
Memory state → Uncertainty metrics → Question generation → Priority queue
     ↑                                                            ↓
     └──────────────── Exploration result ──────────────────────→┘
```

**Uncertainty metrics (computed, not asked to the model):**
- `isolation_score`: nodes with no relations in the knowledge graph
- `confidence_variance`: high variance in a topic cluster = contested knowledge
- `contradiction_density`: entries that contradict each other
- `novelty_gradient`: how different a potential question is from existing knowledge
- `staleness`: entries not revisited in N cycles

These metrics DRIVE the LLM queries. The model generates the actual questions
but the **decision to ask** comes from the computed metrics.

### Tasks
- [ ] Implement `UncertaintyScorer` — computes the five metrics above
- [ ] Implement `QuestionGenerator` — uses metrics + memory state to generate questions via LLM
- [ ] Implement `QuestionPrioritizer` — ranks questions by `novelty × uncertainty × relevance`
- [ ] Implement `CuriosityLoop` — background async task that runs every N seconds
- [ ] Events emitted: `QuestionGenerated`, `ExplorationTriggered`
- [ ] Tests: metrics compute correctly, loop generates questions without human input

### Acceptance criteria
- System generates questions with no human interaction after startup
- Priority scores are deterministic given the same memory state
- Loop respects `MAX_AUTONOMOUS_CYCLES` safety limit

---

## Phase 4 — Exploration Engine

**Goal:** The system can research a question using available tools, record
discoveries, and feed results back into memory.

### Tasks
- [ ] Implement `ExplorationEngine` with `explore(question)` method
- [ ] Implement basic tool: `web_search` (via API, sandboxed)
- [ ] Implement `DiscoveryRecorder` — formats and stores exploration results
- [ ] Connect discoveries to existing memory via relation detection
- [ ] Events: `ExplorationStarted`, `ExplorationCompleted`, `DiscoveryRecorded`
- [ ] Tests: exploration stores results, relations are created correctly

### Acceptance criteria
- An exploration run produces at least one new `MemoryEntry`
- New entries are connected to existing related entries
- Failed explorations are recorded as `error` type entries

---

## Phase 5 — Experiment Engine

**Goal:** The system can design, run, and evaluate controlled experiments
to test hypotheses.

### Experiment lifecycle

```
Hypothesis → Design → Execute → Observe → Conclude → Store
```

### Tasks
- [ ] Implement `Experiment` dataclass (hypothesis, procedure, expected_outcome, actual_outcome, conclusion)
- [ ] Implement `ExperimentEngine.design(hypothesis)` — LLM designs the experiment
- [ ] Implement `ExperimentEngine.run(experiment)` — executes in sandbox
- [ ] Implement `ExperimentEngine.evaluate(result)` — compares expected vs actual
- [ ] Store complete experiment records in memory
- [ ] Tests: experiment lifecycle end-to-end

---

## Phase 6 — Evaluation Engine

**Goal:** Distinguish real improvement from the illusion of improvement.
This is the hardest and most critical component.

### Evaluation strategies
- **Consistency check**: does the system give the same answer to the same question before and after a learning event?
- **Contradiction detection**: did new knowledge introduce contradictions?
- **Confidence calibration**: is confidence correlated with actual accuracy?
- **Hallucination risk scoring**: flag answers that cannot be traced to stored evidence

### Tasks
- [ ] Implement `EvaluationEngine.evaluate_learning(before_snapshot, after_snapshot)`
- [ ] Implement `HallucinationDetector` — traces claims to memory entries
- [ ] Implement `ConfidenceCalibrator` — adjusts confidence based on verification results
- [ ] Implement metrics dashboard (simple JSON export for now)
- [ ] Tests: hallucination detector flags unsourced claims

---

## Phase 7 — Self-Modification Engine

**Goal:** The system can inspect its own architecture, propose improvements,
implement them in isolated Git branches, run tests, and compare results.

### Safety constraints (non-negotiable)
- Read-only access to source by default
- Writes only to dedicated Git branches (`experiment/auto-YYYYMMDD-HHMMSS`)
- Never auto-merge — human approval required
- Cannot modify the `safety/` module or the audit log
- Cannot grant itself additional permissions

### Lifecycle

```
Inspect → Identify weakness → Propose → Create branch →
Implement → Run tests → Compare → Store result → Await approval
```

### Tasks
- [ ] Implement `ArchitectureInspector` — read-only source analysis
- [ ] Implement `ImprovementProposer` — LLM generates a structured proposal
- [ ] Implement `BranchManager` — creates and manages experiment branches
- [ ] Implement `TestRunner` — runs pytest in isolation, captures results
- [ ] Implement `ResultComparator` — before/after metrics
- [ ] Safety gate: proposals stored as `PROPOSED`, never `APPLIED` without human action
- [ ] Tests: proposals are created, branch exists, auto-apply is impossible

---

## Phase 8 — Local API (Anthropic-compatible)

**Goal:** Expose the system as a local server that Claude Code can use as
a model provider.

### Endpoints
- `POST /v1/messages` — Anthropic Messages API compatible
- `GET /v1/models` — returns available models
- `GET /health` — system status

### Request pipeline

```
Claude Code → POST /v1/messages →
  Safety check →
  Context injection (memory + curiosity state) →
  ModelProvider →
  Response
```

### Notes
- Streaming compatibility must be tested against the real Claude Code client
- Do not claim full compatibility until verified
- Tool use support is a stretch goal

### Tasks
- [ ] FastAPI server with `/v1/messages`
- [ ] Request/response schema matching Anthropic API
- [ ] Streaming support (SSE)
- [ ] Integration test with actual Claude Code client
- [ ] Document verified compatibility and known gaps

---

## Development rules

1. **No auto-commits or auto-pushes.** Human decides what goes to main.
2. **Every experiment in its own branch.** Merge only if it passes tests.
3. **Don't confuse appearing to learn with actually improving.** Every claim needs evidence.
4. **Safety module is read-only for the AI.** It cannot modify its own limits.
5. **Audit log is append-only.** No entry is ever deleted or modified.
6. **Model is always interchangeable.** Business logic never imports from a specific provider.
