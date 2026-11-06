from __future__ import annotations
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

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


# ── provider factory ───────────────────────────────────────────────────────

def _build_provider(settings) -> ModelProvider:  # type: ignore[return]
    name = settings.model_provider.lower()
    if name == "openai":
        from ia_test.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.model_name)
    if name == "anthropic":
        from ia_test.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.model_name,
            base_url=settings.anthropic_base_url or None,
        )
    raise ValueError(
        f"Unknown model_provider '{name}'. "
        "Set MODEL_PROVIDER=openai or MODEL_PROVIDER=anthropic."
    )


# ── chat UI ───────────────────────────────────────────────────────────────

_CHAT_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IA-test</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0d0d0d;--surface:#1a1a1a;--border:#2a2a2a;
    --accent:#7c6af7;--accent2:#a78bfa;
    --text:#e8e8e8;--muted:#666;--user-bg:#1e1a35;--bot-bg:#1a1a1a;
  }
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
       display:flex;flex-direction:column;height:100vh;overflow:hidden}

  /* header */
  header{padding:14px 20px;border-bottom:1px solid var(--border);
         display:flex;align-items:center;gap:12px;background:var(--surface)}
  .logo{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--accent2));
        display:flex;align-items:center;justify-content:center;font-size:16px}
  header h1{font-size:1rem;font-weight:600;letter-spacing:.03em}
  header span{font-size:.75rem;color:var(--muted);margin-left:auto}
  #mem-count{color:var(--accent2);font-weight:600}

  /* messages */
  #messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
  .msg{display:flex;gap:10px;max-width:820px}
  .msg.user{align-self:flex-end;flex-direction:row-reverse}
  .avatar{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;
          align-items:center;justify-content:center;font-size:13px;margin-top:2px}
  .msg.user .avatar{background:var(--accent);color:#fff}
  .msg.bot  .avatar{background:var(--border);color:var(--accent2)}
  .bubble{padding:12px 16px;border-radius:14px;line-height:1.6;font-size:.9rem;max-width:680px}
  .msg.user .bubble{background:var(--user-bg);border:1px solid var(--accent);border-top-right-radius:4px}
  .msg.bot  .bubble{background:var(--bot-bg);border:1px solid var(--border);border-top-left-radius:4px}
  .meta{font-size:.7rem;color:var(--muted);margin-top:5px}
  .msg.user .meta{text-align:right}

  /* typing indicator */
  .typing span{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent2);
               animation:bounce .9s infinite;margin:0 2px}
  .typing span:nth-child(2){animation-delay:.15s}
  .typing span:nth-child(3){animation-delay:.3s}
  @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}

  /* input */
  #input-area{padding:14px 20px;border-top:1px solid var(--border);background:var(--surface);
              display:flex;gap:10px;align-items:flex-end}
  #input{flex:1;background:#111;border:1px solid var(--border);border-radius:10px;
         color:var(--text);padding:10px 14px;font-size:.9rem;resize:none;
         min-height:42px;max-height:140px;line-height:1.5;outline:none;font-family:inherit}
  #input:focus{border-color:var(--accent)}
  #send{background:var(--accent);color:#fff;border:none;border-radius:10px;
        width:42px;height:42px;cursor:pointer;font-size:18px;flex-shrink:0;
        display:flex;align-items:center;justify-content:center;transition:opacity .15s}
  #send:disabled{opacity:.4;cursor:not-allowed}

  /* memory badge */
  .mem-badge{display:inline-block;background:#1e2a1e;border:1px solid #2a3f2a;
             color:#6fcf97;border-radius:4px;font-size:.7rem;padding:1px 6px;margin-left:6px}

  /* scrollbar */
  #messages::-webkit-scrollbar{width:5px}
  #messages::-webkit-scrollbar-track{background:transparent}
  #messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<header>
  <div class="logo">🧠</div>
  <h1>IA-test</h1>
  <span>Memoria: <span id="mem-count">0</span> entradas &nbsp;·&nbsp; gpt-4o-mini</span>
</header>

<div id="messages">
  <div class="msg bot">
    <div class="avatar">🤖</div>
    <div>
      <div class="bubble">
        Hola — soy IA-test, un sistema experimental con memoria persistente.
        Todo lo que hablemos va quedando en mi grafo de conocimiento.
        ¿En qué pensamos hoy?
      </div>
      <div class="meta">IA-test · ahora</div>
    </div>
  </div>
</div>

<div id="input-area">
  <textarea id="input" placeholder="Escribí algo..." rows="1"></textarea>
  <button id="send">↑</button>
</div>

<script>
const msgs = document.getElementById('messages');
const input = document.getElementById('input');
const send  = document.getElementById('send');
const memCount = document.getElementById('mem-count');

const history = [];

function ts() {
  return new Date().toLocaleTimeString('es', {hour:'2-digit',minute:'2-digit'});
}

function appendMsg(role, text) {
  const isUser = role === 'user';
  const div = document.createElement('div');
  div.className = 'msg ' + (isUser ? 'user' : 'bot');
  div.innerHTML = `
    <div class="avatar">${isUser ? '👤' : '🤖'}</div>
    <div>
      <div class="bubble">${text.replace(/\\n/g,'<br>')}</div>
      <div class="meta">${isUser ? 'Vos' : 'IA-test'} · ${ts()}</div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function appendTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = `
    <div class="avatar">🤖</div>
    <div><div class="bubble typing"><span></span><span></span><span></span></div></div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

async function refreshMem() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    memCount.textContent = d.memory_entries;
  } catch {}
}

async function sendMsg() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';
  send.disabled = true;

  appendMsg('user', text);
  history.push({role:'user', content: text});

  const typing = appendTyping();

  try {
    const res = await fetch('/v1/messages', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        max_tokens: 1024,
        messages: history.slice(-20),
      }),
    });
    const data = await res.json();
    const reply = data.content?.[0]?.text ?? '(sin respuesta)';

    typing.remove();
    appendMsg('bot', reply);
    history.push({role:'assistant', content: reply});
    await refreshMem();
  } catch(e) {
    typing.remove();
    appendMsg('bot', '⚠️ Error: ' + e.message);
  }

  send.disabled = false;
  input.focus();
}

send.addEventListener('click', sendMsg);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});

refreshMem();
</script>
</body>
</html>"""


# ── app factory ────────────────────────────────────────────────────────────

def create_app(
    provider: ModelProvider,
    db: MemoryDatabase,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await db.connect()
        yield
        await db.close()

    _app = FastAPI(title="IA-test Local API", version="0.1.0", lifespan=lifespan)
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
        yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        try:
            response = await provider.complete(messages, temperature=req.temperature)
            text = response.content
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            output_tokens = response.usage.completion_tokens if response.usage else len(text.split())
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': f'[Error: {exc}]'}})}\n\n"
            output_tokens = 0

        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield "data: {\"type\": \"message_stop\"}\n\n"

    # ── routes ────────────────────────────────────────────────────────────

    @_app.get("/health")
    async def health() -> dict:
        entries = await db.query(limit=1000)
        return {
            "status": "ok",
            "memory_entries": len(entries),
            "model": _LOCAL_MODEL_ID,
        }

    @_app.get("/v1/models")
    async def list_models() -> ModelsResponse:
        return ModelsResponse(
            data=[ModelInfo(id=_LOCAL_MODEL_ID, created=_LOCAL_CREATED)]
        )

    @_app.get("/", response_class=HTMLResponse)
    async def chat_ui() -> str:
        return _CHAT_HTML

    @_app.post("/v1/messages", response_model=None)
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

    return _app


# ── module-level app factory for uvicorn --factory ─────────────────────────
# Usage:
#   python -m uvicorn ia_test.api.server:app_factory --factory --host 127.0.0.1 --port 8000
#
# Using --factory ensures this runs only when uvicorn starts, not at import time,
# so pytest collection is unaffected.

def app_factory() -> FastAPI:
    import pathlib
    from ia_test.core.settings import Settings

    settings = Settings()
    provider = _build_provider(settings)

    pathlib.Path(settings.memory_db_path).parent.mkdir(parents=True, exist_ok=True)
    db = MemoryDatabase(settings.memory_db_path)
    # DB connection happens in the lifespan handler inside create_app,
    # so no asyncio.run() needed here.

    rate_limiter = RateLimiter(
        max_per_hour=settings.max_api_calls_per_hour,
        max_per_day=settings.max_api_calls_per_day,
    )
    return create_app(provider=provider, db=db, rate_limiter=rate_limiter)
