from __future__ import annotations
import uuid
from typing import Literal
from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────

class ContentBlock(BaseModel):
    type: str = "text"
    text: str


class RequestMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlock]

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return " ".join(b.text for b in self.content if b.type == "text")


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int = 1024
    messages: list[RequestMessage]
    system: str | None = None
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


# ── Response ───────────────────────────────────────────────────────────────

class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


class ContentOut(BaseModel):
    type: str = "text"
    text: str


class MessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:20]}")
    type: str = "message"
    role: str = "assistant"
    content: list[ContentOut]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: None = None
    usage: UsageOut


# ── Models list ────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ia-test"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
