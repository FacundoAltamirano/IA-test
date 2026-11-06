from __future__ import annotations
import json
import time
import uuid
import asyncio
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from ia_test.api.models import (
    ContentOut,
    MessagesRequest,
    MessagesResponse,
    ModelInfo,
    ModelsResponse,
    UsageOut,
)
from ia_test.api.context import ContextInjector
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Message, ModelProvider
from ia_test.safety.limits import RateLimiter

_LOCAL_MODEL_ID = "ia-test-local"
_LOCAL_CREATED = int(time.time())


def create_app(
    provider: ModelProvider,
    db: MemoryDatabase,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    app = FastAPI(title="IA-test Local API", version="0.1.0")
    injector = ContextInjector(db)

    # ── helpers ───────────────────────────────────────────────────────────

    async def _check_rate(limiter: RateLimiter | None) -> None:
        if limiter is None:
            return
        from ia_test.safety.limits import RateLimitExceeded
        try:
            await limiter.check_and_record()
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    def _build_messages(req: MessagesRequest, system_text: str) -> list[Message]:
        """Convert the Anthropic-style request into provider Message objects."""
        msgs: list[Message] = []
        if system_text:
            msgs.append(Message(role="system", content=system_text))
        for m in req.messages:
            msgs.append(Message(role=m.role, content=m.as_text()))
        return msgs

    async def _stream_sse(
        req: MessagesRequest,
        messages: list[Message],
        msg_id: str,
    ) -> AsyncIterator[str]:
        """Yield Server-Sent Events compatible with the Anthropic streaming format."""
        # message_start
        start = {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": req.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        yield f"data: {json.dumps(start)}\n\n"

        # content_block_start
        yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Run the actual provider call (non-streaming internally; wrap into one delta)
        try:
            response = await provider.complete(messages, temperature=req.temperature)
            text = response.content
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            output_tokens = response.usage.completion_tokens if response.usage else len(text.split())
            input_tokens = response.usage.prompt_tokens if response.usage else 0
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': f'[Error: {exc}]'}})}\n\n"
            output_tokens = 0
            input_tokens = 0

        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield "data: {\"type\": \"message_stop\"}\n\n"

    # ── routes ────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict:
        entries = await db.query(limit=1000)
        return {
            "status": "ok",
            "memory_entries": len(entries),
            "model": _LOCAL_MODEL_ID,
        }

    @app.get("/v1/models")
    async def list_models() -> ModelsResponse:
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=_LOCAL_MODEL_ID,
                    created=_LOCAL_CREATED,
                )
            ]
        )

    @app.post("/v1/messages", response_model=None)
    async def create_message(req: MessagesRequest, request: Request) -> JSONResponse | StreamingResponse:
        await _check_rate(rate_limiter)

        system_text = await injector.enrich(req.messages, req.system)
        messages = _build_messages(req, system_text)
        msg_id = f"msg_{uuid.uuid4().hex[:20]}"

        if req.stream:
            return StreamingResponse(
                _stream_sse(req, messages, msg_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            response = await provider.complete(messages, temperature=req.temperature)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        usage = response.usage
        resp = MessagesResponse(
            id=msg_id,
            content=[ContentOut(text=response.content)],
            model=req.model,
            usage=UsageOut(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )
        return JSONResponse(resp.model_dump())

    return app
