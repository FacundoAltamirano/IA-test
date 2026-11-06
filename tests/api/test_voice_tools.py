"""Tests for update_voice_style tool and VoiceEngine annotation processing."""
from __future__ import annotations
import json
import pytest

from ia_test.api.memory_tools import MemoryToolExecutor
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import ToolCall
from ia_test.voice.engine import VoiceEngine
from ia_test.voice.models import VoiceStyle


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="test-voice-001", name=name, arguments=arguments)


@pytest.fixture
async def db(tmp_path):
    database = MemoryDatabase(str(tmp_path / "voice_test.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def executor(db):
    return MemoryToolExecutor(db)


# ── update_voice_style read ───────────────────────────────────────────────

async def test_read_style_returns_defaults_when_no_style_stored(db, executor):
    result = await executor.execute(_call("update_voice_style", {}))
    data = json.loads(result.content)

    assert data["action"] == "read"
    assert data["current_style"]["voice"] == "nova"
    assert data["current_style"]["speed"] == 1.0
    assert "[pausa]" in data["annotations"]


# ── update_voice_style write ──────────────────────────────────────────────

async def test_update_voice_changes_voice_and_speed(db, executor):
    result = await executor.execute(_call("update_voice_style", {"voice": "onyx", "speed": 0.9}))
    data = json.loads(result.content)

    assert data["action"] == "updated"
    assert data["new_style"]["voice"] == "onyx"
    assert data["new_style"]["speed"] == 0.9


async def test_update_voice_persists_across_reads(db, executor):
    await executor.execute(_call("update_voice_style", {"voice": "shimmer", "style_notes": "Hablar lento y con calma"}))

    result = await executor.execute(_call("update_voice_style", {}))
    data = json.loads(result.content)
    assert data["current_style"]["voice"] == "shimmer"
    assert "calma" in data["current_style"]["style_notes"]


async def test_update_voice_rejects_invalid_voice(db, executor):
    result = await executor.execute(_call("update_voice_style", {"voice": "invalid_voice"}))
    data = json.loads(result.content)
    assert "error" in data


async def test_update_voice_rejects_out_of_range_speed(db, executor):
    result = await executor.execute(_call("update_voice_style", {"speed": 10.0}))
    data = json.loads(result.content)
    assert "error" in data


async def test_update_voice_allows_partial_update(db, executor):
    await executor.execute(_call("update_voice_style", {"voice": "echo", "speed": 1.2}))
    await executor.execute(_call("update_voice_style", {"pause_marker": "— "}))

    result = await executor.execute(_call("update_voice_style", {}))
    data = json.loads(result.content)
    # Voice and speed from first call, pause_marker from second
    assert data["current_style"]["voice"] == "echo"
    assert data["current_style"]["pause_marker"] == "— "


# ── VoiceEngine annotation processing ────────────────────────────────────

def _engine(db) -> VoiceEngine:
    return VoiceEngine(api_key="test-key", db=db)


async def test_process_annotations_replaces_pausa(db):
    engine = _engine(db)
    style = VoiceStyle(pause_marker="...", long_pause_marker="\n\n...")
    result = engine.process_annotations("Hola [pausa] mundo", style)
    assert result == "Hola ... mundo"


async def test_process_annotations_replaces_risa_suave(db):
    engine = _engine(db)
    style = VoiceStyle(laugh_soft="*ríe*")
    result = engine.process_annotations("Qué gracioso [risa:suave]", style)
    assert "*ríe*" in result


async def test_process_annotations_strips_sarcasmo_marker(db):
    engine = _engine(db)
    style = VoiceStyle()
    result = engine.process_annotations("Claro [sarcasmo] que sí", style)
    assert "[sarcasmo]" not in result
    assert "Claro" in result


async def test_process_annotations_handles_multiple_markers(db):
    engine = _engine(db)
    style = VoiceStyle(pause_marker="...", laugh_soft="*ríe*")
    text = "Interesante [pausa] muy interesante [risa:suave] de verdad"
    result = engine.process_annotations(text, style)
    assert "..." in result
    assert "*ríe*" in result
    assert "[pausa]" not in result


async def test_process_annotations_text_without_markers_unchanged(db):
    engine = _engine(db)
    style = VoiceStyle()
    text = "Hola, esto es un texto sin anotaciones."
    assert engine.process_annotations(text, style) == text


async def test_load_style_uses_stored_config(db):
    engine = _engine(db)
    # Store a style directly in memory
    style_data = VoiceStyle(voice="alloy", speed=1.5).to_dict()
    await db.store(
        type="experience",
        content=json.dumps(style_data),
        confidence=1.0,
        provenance="voice:self_modification",
        tags=["voice:style", "voice:config"],
    )

    loaded = await engine.load_style()
    assert loaded.voice == "alloy"
    assert loaded.speed == 1.5
