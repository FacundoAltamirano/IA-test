from typing import AsyncIterator

from openai import AsyncOpenAI

from .base import Chunk, Message, Response, Tool, Usage


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
    ) -> Response:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
        )
        choice = response.choices[0]
        return Response(
            content=choice.message.content or "",
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
    ) -> AsyncIterator[Chunk]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            choice = chunk.choices[0]
            yield Chunk(
                content=choice.delta.content or "",
                is_final=choice.finish_reason is not None,
            )
