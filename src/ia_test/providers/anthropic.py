from __future__ import annotations
from typing import AsyncIterator

import anthropic

from .base import Chunk, Message, Response, Tool, Usage


class AnthropicProvider:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        base_url: str | None = None,
    ) -> None:
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        """Separate the leading system message from the rest."""
        system: str | None = None
        chat: list[dict] = []
        for m in messages:
            if m.role == "system" and not chat:
                system = m.content
            else:
                chat.append({"role": m.role, "content": m.content})
        return system, chat

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
    ) -> Response:
        system, chat = self._split_messages(messages)
        kwargs: dict = dict(
            model=self._model,
            max_tokens=4096,
            messages=chat,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return Response(
            content=text,
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
        )

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> AsyncIterator[Chunk]:
        system, chat = self._split_messages(messages)
        kwargs: dict = dict(
            model=self._model,
            max_tokens=4096,
            messages=chat,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield Chunk(content=text, is_final=False)
        yield Chunk(content="", is_final=True)
