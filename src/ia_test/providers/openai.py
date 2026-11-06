from __future__ import annotations
import json
from typing import AsyncIterator

from openai import AsyncOpenAI

from .base import Chunk, Message, Response, Tool, ToolCall, ToolResult, Usage


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def _build_chat_messages(self, messages: list[Message]) -> list[dict]:
        result: list[dict] = []
        for m in messages:
            d: dict = {"role": m.role, "content": m.content or None}
            d.update(m.extra)
            result.append(d)
        return result

    def _tools_to_openai(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
    ) -> Response:
        kwargs: dict = dict(
            model=self._model,
            messages=self._build_chat_messages(messages),
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)
            kwargs["parallel_tool_calls"] = False  # one tool call at a time, simplifies loop

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Safe against MagicMock in tests: only iterate if it's a real list
        raw_tool_calls = getattr(choice.message, "tool_calls", None)
        tool_calls: list[ToolCall] = []
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            for tc in raw_tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return Response(
            content=choice.message.content or "",
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            tool_calls=tool_calls,
            stop_reason="tool_calls" if tool_calls else "end_turn",
        )

    def append_tool_exchange(
        self,
        messages: list[Message],
        response: Response,
        results: list[ToolResult],
    ) -> list[Message]:
        """Append assistant tool call + tool results in OpenAI native format."""
        tc_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in response.tool_calls
        ]

        new_messages = list(messages)
        # OpenAI assistant message: content="" when only tool calls, tool_calls in extra
        new_messages.append(Message(
            role="assistant",
            content=response.content,
            extra={"tool_calls": tc_dicts} if tc_dicts else {},
        ))
        # One "tool" message per result
        for r in results:
            new_messages.append(Message(
                role="tool",
                content=r.content,
                extra={"tool_call_id": r.tool_call_id},
            ))
        return new_messages

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> AsyncIterator[Chunk]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_chat_messages(messages),
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            choice = chunk.choices[0]
            yield Chunk(
                content=choice.delta.content or "",
                is_final=choice.finish_reason is not None,
            )
