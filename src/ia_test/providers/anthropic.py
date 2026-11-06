from __future__ import annotations
from typing import AsyncIterator

import anthropic

from .base import Chunk, Message, Response, Tool, ToolCall, ToolResult, Usage


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
        system: str | None = None
        chat: list[dict] = []
        for m in messages:
            if m.role == "system" and not chat:
                system = m.content if isinstance(m.content, str) else None
            else:
                d = {"role": m.role, "content": m.content}
                d.update(m.extra)
                chat.append(d)
        return system, chat

    def _tools_to_anthropic(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

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
        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return Response(
            content="".join(text_parts),
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
        )

    def append_tool_exchange(
        self,
        messages: list[Message],
        response: Response,
        results: list[ToolResult],
    ) -> list[Message]:
        """Append assistant tool call + user tool results in Anthropic native format."""
        assistant_content: list = []
        if response.content:
            assistant_content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })

        tool_result_content = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
            }
            for r in results
        ]

        return [
            *messages,
            Message(role="assistant", content=assistant_content),
            Message(role="user", content=tool_result_content),
        ]

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
