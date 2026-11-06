"""VoiceEngine — converts annotated RON text to audio via OpenAI TTS.

Annotation syntax RON uses in its responses:
  [pausa]        → short pause (defaults to "...")
  [pausa:larga]  → long pause ("\n\n...")
  [risa:suave]   → soft laugh (*ríe suavemente*)
  [risa:fuerte]  → loud laugh (*carcajada*)
  [énfasis]      → emphasis marker (stripped — TTS reads punctuation naturally)
  [sarcasmo]     → sarcasm marker (stripped — rhythm implied by surrounding text)

Voice style is loaded from the knowledge graph at request time, so RON can
modify its own narration preferences via update_voice_style and the change
takes effect on the next audio request.
"""
from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from ia_test.memory.database import MemoryDatabase
from .models import VoiceStyle

logger = logging.getLogger(__name__)

_VOICE_STYLE_TAG = "voice:style"
_ANNOTATION_RE = re.compile(
    r"\[pausa:larga\]|\[pausa\]|\[risa:suave\]|\[risa:fuerte\]|\[énfasis\]|\[sarcasmo\]"
)


class VoiceEngine:
    def __init__(self, api_key: str, db: MemoryDatabase) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._db = db

    async def load_style(self) -> VoiceStyle:
        """Load voice style from memory, or return defaults if not yet configured."""
        all_entries = await self._db.query(limit=500)
        for entry in all_entries:
            if _VOICE_STYLE_TAG in entry.tags:
                try:
                    data = json.loads(entry.content)
                    return VoiceStyle.from_dict(data)
                except (json.JSONDecodeError, KeyError):
                    pass
        return VoiceStyle()

    def process_annotations(self, text: str, style: VoiceStyle) -> str:
        """Replace RON's voice annotations with text that OpenAI TTS renders naturally."""

        def _replace(m: re.Match) -> str:
            token = m.group(0)
            if token == "[pausa]":
                return style.pause_marker
            if token == "[pausa:larga]":
                return style.long_pause_marker
            if token == "[risa:suave]":
                return style.laugh_soft
            if token == "[risa:fuerte]":
                return style.laugh_loud
            # [énfasis] and [sarcasmo] — no text substitution; TTS reads surrounding punctuation
            return ""

        return _ANNOTATION_RE.sub(_replace, text).strip()

    async def synthesize(self, text: str) -> bytes:
        """Convert annotated text to MP3 audio bytes via OpenAI TTS."""
        style = await self.load_style()
        clean_text = self.process_annotations(text, style)

        if not clean_text:
            return b""

        logger.info(
            "Voice synthesis: voice=%s speed=%.2f chars=%d",
            style.voice,
            style.speed,
            len(clean_text),
        )

        response = await self._client.audio.speech.create(
            model="tts-1",
            voice=style.voice,  # type: ignore[arg-type]
            input=clean_text,
            speed=style.speed,
            response_format="mp3",
        )
        return response.content
