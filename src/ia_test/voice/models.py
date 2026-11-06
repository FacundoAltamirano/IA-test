"""Voice style model — stored in memory so RON can read and modify it."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VoiceStyle:
    """Controls how RON narrates. Persisted in the knowledge graph under tag 'voice:style'."""

    voice: str = "nova"          # OpenAI TTS voice: alloy | echo | fable | onyx | nova | shimmer
    speed: float = 1.0           # 0.25–4.0
    pause_marker: str = "..."    # Text substituted for [pausa]
    long_pause_marker: str = "\n\n..." # Text substituted for [pausa:larga]
    laugh_soft: str = "*ríe suavemente*"
    laugh_loud: str = "*carcajada*"
    style_notes: str = ""        # Free-form narration guidance RON can read in its identity prompt

    def to_dict(self) -> dict:
        return {
            "voice": self.voice,
            "speed": self.speed,
            "pause_marker": self.pause_marker,
            "long_pause_marker": self.long_pause_marker,
            "laugh_soft": self.laugh_soft,
            "laugh_loud": self.laugh_loud,
            "style_notes": self.style_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceStyle":
        return cls(
            voice=data.get("voice", "nova"),
            speed=float(data.get("speed", 1.0)),
            pause_marker=data.get("pause_marker", "..."),
            long_pause_marker=data.get("long_pause_marker", "\n\n..."),
            laugh_soft=data.get("laugh_soft", "*ríe suavemente*"),
            laugh_loud=data.get("laugh_loud", "*carcajada*"),
            style_notes=data.get("style_notes", ""),
        )

    # ── annotation constants (RON uses these in its text) ─────────────────

    ANNOTATIONS: dict[str, str] = field(default_factory=lambda: {})

    # Non-field class constants used by VoiceEngine
    KNOWN_ANNOTATIONS = frozenset([
        "[pausa]",
        "[pausa:larga]",
        "[risa:suave]",
        "[risa:fuerte]",
        "[énfasis]",
        "[sarcasmo]",
    ])
