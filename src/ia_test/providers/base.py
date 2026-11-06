from __future__ import annotations
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class Message:
    role: str
    content: str | list  # list for tool-use content blocks (native provider format)
    extra: dict = field(default_factory=dict)  # provider-specific extra fields (tool_calls, tool_call_id)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # parsed JSON


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str  # JSON-serialized result


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Response:
    content: str
    model: str
    usage: Usage
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


@dataclass
class Chunk:
    content: str
    is_final: bool = False


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
    ) -> Response: ...

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> AsyncIterator[Chunk]: ...
