from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)


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
