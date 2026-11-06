# IA-test

An experimental AI system with persistent exploratory motivation.

This is not a chatbot. The goal is to investigate how to build a system
that generates its own questions, explores knowledge, formulates hypotheses,
experiments, and proposes improvements to its own architecture — without
depending on human instructions to do so.

## Core idea

The curiosity is not a skill or a prompt. It is a persistent computational
mechanism embedded in the architecture that drives the system to detect
uncertainties, seek novelty, and explore — even when no one is talking to it.

## Architecture overview

```
src/ia_test/
├── core/             # Execution nucleus and event bus
├── api/              # Local server compatible with Anthropic Messages API
├── providers/        # Interchangeable model connectors (OpenAI, Anthropic, ...)
├── curiosity/        # Persistent curiosity engine
├── memory/           # Persistent memory (SQLite)
├── exploration/      # Autonomous research and tool use
├── experiments/      # Hypothesis design and controlled experiments
├── self_modification/ # Controlled self-modification via Git branches
├── evaluation/       # Learning verification and anti-hallucination
└── safety/           # Hard limits, audit log, permission system
```

## Development phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core + OpenAI provider | Planned |
| 2 | Persistent memory (SQLite) | Planned |
| 3 | Curiosity engine | Planned |
| 4 | Autonomous exploration | Planned |
| 5 | Experiment engine | Planned |
| 6 | Evaluation engine | Planned |
| 7 | Self-modification engine | Planned |
| 8 | Local API (Anthropic-compatible) | Planned |

## Stack

- Python 3.11+
- FastAPI
- SQLite
- OpenAI API (model-agnostic by design)
- Pytest

## Key rule

Do not confuse the system *appearing to learn* with it *actually improving*.
Every claimed improvement must be measurable and verifiable.

## Setup

```bash
cp .env.example .env
# Add your API keys
pip install -e ".[dev]"
pytest
```
