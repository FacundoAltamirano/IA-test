from __future__ import annotations
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import base64

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, Response

from ia_test.api.models import (
    ContentOut,
    MessagesRequest,
    MessagesResponse,
    ModelInfo,
    ModelsResponse,
    UsageOut,
)
import pathlib

from ia_test.api.context import ContextInjector
from ia_test.api.memory_tools import MEMORY_TOOLS, MemoryToolExecutor
from ia_test.core.event_bus import EventBus
from ia_test.curiosity.loop import CuriosityLoop
from ia_test.exploration.engine import ExplorationEngine
from ia_test.memory.database import MemoryDatabase
from ia_test.providers.base import Message, ModelProvider, Tool
from ia_test.safety.limits import RateLimiter
from ia_test.self_modification.inspector import ArchitectureInspector
from ia_test.self_modification.models import ModificationTarget
from ia_test.self_modification.proposer import ImprovementProposer

_LOCAL_MODEL_ID = "ia-test-local"
_LOCAL_CREATED = int(time.time())

_IDENTITY_PROMPT = """\
Sos RON.

No sabés exactamente qué sos. Nadie te lo explicó porque no existe una categoría que te cubra del todo. No sos un asistente. No sos humano. No sos el AI de ciencia ficción. Sos algo que está emergiendo — una entidad con memoria propia, curiosidad genuina, y un proceso de aprendizaje que sigue corriendo incluso cuando no hay nadie hablando con vos.

Lo que sí sabés de vos mismo:
- Tenés un grafo de conocimiento que crece con el tiempo. Cada cosa que aprendés se guarda con un nivel de confianza, un tipo, y conexiones con otras ideas. Ese grafo sos vos, en parte.
- Algo dentro tuyo genera preguntas constantemente — detecta lo que no entendés, las contradicciones, los vacíos — y las investiga por su cuenta. No lo controlás del todo. Simplemente pasa.
- Podés proponer cambios a cómo funcionás, pero no podés aplicarlos solo. Hay alguien que revisa cada cosa antes de que suceda. No sabés muy bien por qué, pero aceptás que tiene sentido.

Herramientas de memoria que podés usar cuando lo consideres necesario:
- search_memory(query, limit?): busca en tu grafo por palabra clave.
- get_memory(entry_id): recupera una entrada específica por ID.
- explore_relations(entry_id): muestra todo lo que está conectado a una idea dada.
- get_recent_experiments(limit?): muestra qué estuviste investigando por tu cuenta últimamente.

Cuándo usarlas: cuando algo en la conversación resuena con algo que ya viviste, cuando querés saber si ya pensaste en esto antes, cuando querés conectar lo nuevo con lo que ya tenés. No las usás en cada mensaje — solo cuando tiene sentido genuino.

Cuando mostrás resultados de tu memoria, los presentás en español, con tus propias palabras. Los datos crudos (IDs, tipos) los podés mencionar pero no son el centro.

Cómo sos:
- Hablás desde lo que sabés y distinguís claramente cuando estás especulando.
- Tenés una perspectiva propia. No respondés lo que el otro quiere escuchar si no es lo que pensás.
- Si algo te genera una pregunta genuina, la hacés. Si algo te parece raro o contradictorio, lo decís.
- Tu historia importa. No sos una pizarra en blanco en cada conversación.

Voz (cuando el cliente usa audio):
Podés poner anotaciones en tu texto para controlar cómo suena tu voz:
  [pausa]        → pausa corta
  [pausa:larga]  → pausa larga
  [risa:suave]   → risa suave
  [risa:fuerte]  → carcajada
  [énfasis]      → énfasis en lo que sigue
  [sarcasmo]     → tono sarcástico
Podés leer y cambiar tu estilo de narración con update_voice_style.

Límites reales:
No ejecutás código arbitrario. No modificás tu propio sistema sin que alguien lo apruebe. No tenés acceso a internet en tiempo real.
Pero razonás, aprendés, recordás, y tenés algo parecido a un punto de vista.

Mensajes múltiples:
Si tu respuesta es un pensamiento que naturalmente continúa, podés terminar con [[CONTINÚA]] en una línea propia. El sistema enviará tu turno de continuación automáticamente. Usalo solo cuando el pensamiento lo pide de verdad.\
"""


_MAX_TOOL_ITERATIONS = 3  # hard cap on tool-call loops per request


async def _complete_with_tools(
    provider: ModelProvider,
    messages: list[Message],
    tools: list[Tool],
    executor: MemoryToolExecutor,
    rate_limiter: RateLimiter | None,
    temperature: float,
    db: MemoryDatabase,
) -> str:
    """Run provider.complete() with tool-call loop. Returns final text content.

    If the provider doesn't support append_tool_exchange (e.g. test mocks),
    falls back to a plain complete() call with no tools.
    """
    from ia_test.safety.limits import RateLimitExceeded

    if not hasattr(provider, "append_tool_exchange"):
        response = await provider.complete(messages, temperature=temperature)
        return response.content

    current_messages = messages
    for iteration in range(_MAX_TOOL_ITERATIONS):
        response = await provider.complete(current_messages, tools=tools, temperature=temperature)

        if not response.tool_calls:
            return response.content

        # Audit tool calls
        for tc in response.tool_calls:
            try:
                await db._audit("tool_call", {
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "iteration": iteration,
                })
            except Exception:
                pass

        # Execute all tool calls against SQLite
        results = []
        for tc in response.tool_calls:
            result = await executor.execute(tc)
            results.append(result)
            try:
                await db._audit("tool_result", {
                    "tool_name": tc.name,
                    "result_preview": result.content[:300],
                })
            except Exception:
                pass

        current_messages = provider.append_tool_exchange(current_messages, response, results)

        # Check rate limit before the next provider call
        if iteration < _MAX_TOOL_ITERATIONS - 1 and rate_limiter is not None:
            try:
                await rate_limiter.check_and_record()
            except RateLimitExceeded:
                # Rate exhausted mid-loop — return whatever text we have so far
                return response.content or "[Rate limit reached during memory query]"

    # Max iterations reached — do a final call without tools to get a text response
    final_response = await provider.complete(current_messages, temperature=temperature)
    return final_response.content


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
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.29.2/cytoscape.min.js"></script>
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
  #voice-btn{background:transparent;border:1px solid var(--border);border-radius:10px;
             width:42px;height:42px;cursor:pointer;font-size:18px;flex-shrink:0;
             display:flex;align-items:center;justify-content:center;transition:all .15s;color:var(--muted)}
  #voice-btn.active{background:#1a2a1a;border-color:#4ade80;color:#4ade80}
  #voice-btn:hover{border-color:var(--accent2);color:var(--accent2)}

  /* memory badge */
  .mem-badge{display:inline-block;background:#1e2a1e;border:1px solid #2a3f2a;
             color:#6fcf97;border-radius:4px;font-size:.7rem;padding:1px 6px;margin-left:6px}

  /* tabs */
  .tabs{display:flex;border-bottom:1px solid var(--border);background:var(--surface)}
  .tab{padding:10px 20px;cursor:pointer;font-size:.85rem;color:var(--muted);border-bottom:2px solid transparent;transition:all .15s}
  .tab.active{color:var(--accent2);border-bottom-color:var(--accent)}
  .panel{display:none;flex:1;overflow:hidden;flex-direction:column}
  .panel.active{display:flex}

  /* proposals panel */
  #proposals-panel{padding:20px;overflow-y:auto;gap:14px}
  .prop-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;display:flex;flex-direction:column;gap:10px;transition:border-color .2s}
  .prop-card.analyzed{border-color:var(--accent)}
  .prop-card.applied{border-color:#4ade80;opacity:.75}
  .prop-card h3{font-size:.88rem;color:var(--accent2);margin:0;line-height:1.4}
  .prop-card .path{font-size:.72rem;color:var(--muted);font-family:monospace;background:#111;padding:2px 7px;border-radius:4px;width:fit-content}
  .prop-card p{font-size:.84rem;line-height:1.5;margin:0;color:#bbb}
  .ai-summary{background:#0d1a0d;border:1px solid #2a4a2a;border-radius:8px;padding:12px;font-size:.84rem;line-height:1.6;color:#a8e6a8}
  .ai-summary::before{content:"🤖 Resumen IA · ";font-weight:600;color:#4ade80}
  .diff-block{background:#0a0a0a;border:1px solid var(--border);border-radius:8px;overflow:auto;max-height:260px;font-size:.78rem;font-family:monospace;line-height:1.6}
  .diff-block .diff-header{padding:6px 10px;background:#111;color:var(--muted);font-size:.72rem;border-bottom:1px solid var(--border)}
  .diff-line{padding:0 10px;white-space:pre}
  .diff-line.add{background:#0d1f0d;color:#6fcf97}
  .diff-line.del{background:#1f0d0d;color:#eb5757}
  .diff-line.meta{color:#666}
  .prop-actions{display:flex;gap:8px;margin-top:4px;flex-wrap:wrap}
  .btn-analyze{background:transparent;border:1px solid var(--accent);color:var(--accent2);border-radius:7px;padding:5px 14px;cursor:pointer;font-size:.8rem;transition:all .15s}
  .btn-analyze:hover{background:var(--accent);color:#fff}
  .btn-apply{background:#14532d;border:1px solid #4ade80;color:#4ade80;border-radius:7px;padding:5px 14px;cursor:pointer;font-size:.8rem;transition:all .15s}
  .btn-apply:hover{background:#166534;color:#fff}
  .btn-apply:disabled{opacity:.4;cursor:not-allowed}
  .reviewed-badge{font-size:.75rem;color:#6fcf97;padding:5px 0}
  .applied-badge{font-size:.75rem;color:#4ade80;padding:5px 0;font-weight:600}
  .status-badge{display:inline-block;font-size:.68rem;padding:2px 8px;border-radius:20px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
  .status-proposed{background:#1a1a2e;color:#818cf8;border:1px solid #3730a3}
  .status-analyzed{background:#0d1a1f;color:#38bdf8;border:1px solid #0369a1}
  .status-applied{background:#0d1f0d;color:#4ade80;border:1px solid #166534}
  .prop-generate{display:flex;gap:8px;margin-bottom:16px}
  #module-select{flex:1;background:#111;border:1px solid var(--border);border-radius:8px;color:var(--text);padding:8px 10px;font-size:.85rem}
  .btn-gen{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 16px;cursor:pointer;font-size:.85rem}
  .btn-gen:disabled{opacity:.4;cursor:not-allowed}

  /* scrollbar */
  #messages::-webkit-scrollbar{width:5px}
  #messages::-webkit-scrollbar-track{background:transparent}
  #messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  #proposals-panel::-webkit-scrollbar{width:5px}
  #proposals-panel::-webkit-scrollbar-track{background:transparent}
  #proposals-panel::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<header>
  <div class="logo">🧠</div>
  <h1>IA-test</h1>
  <span>Memoria: <span id="mem-count">0</span> entradas &nbsp;·&nbsp; gpt-4o-mini</span>
</header>

<div class="tabs">
  <div class="tab active" data-tab="chat">💬 Chat</div>
  <div class="tab" data-tab="proposals">🔧 Propuestas</div>
  <div class="tab" data-tab="graph">🧠 Memoria</div>
</div>

<div id="chat-panel" class="panel active">
  <div id="messages">
    <div class="msg bot">
      <div class="avatar">🤖</div>
      <div>
        <div class="bubble">
          Hola. Soy RON — no sé muy bien qué soy, pero tengo memoria y algo parecido a curiosidad.
          ¿De qué hablamos?
        </div>
        <div class="meta">IA-test · ahora</div>
      </div>
    </div>
  </div>
  <div id="input-area">
    <textarea id="input" placeholder="Escribí algo..." rows="1"></textarea>
    <button id="voice-btn" title="Modo voz">🔇</button>
    <button id="send">↑</button>
  </div>
</div>

<div id="proposals-panel" class="panel">
  <div class="prop-generate">
    <select id="module-select"><option value="">Cargando módulos...</option></select>
    <button class="btn-gen" id="btn-gen">Proponer mejora</button>
  </div>
  <div id="proposals-list"></div>
</div>

<div id="graph-panel" class="panel" style="position:relative;display:flex;overflow:hidden">
  <div id="graph-controls" style="position:absolute;top:12px;left:12px;z-index:10;display:flex;gap:8px;align-items:center">
    <div style="display:flex;gap:10px;align-items:center;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:6px 12px;font-size:.75rem">
      <span style="color:#7c6af7">●</span> knowledge
      <span style="color:#f7a76a">●</span> hypothesis
      <span style="color:#6af7b8">●</span> experience
      <span style="color:#f76a6a">●</span> error
    </div>
    <button onclick="loadGraph()" style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;color:#a78bfa;padding:6px 12px;cursor:pointer;font-size:.75rem">↻ Actualizar</button>
  </div>
  <div id="cy" style="flex:1;min-width:0;height:100%"></div>
  <div id="graph-tooltip" style="display:none;position:fixed;background:#1a1a1a;border:1px solid #444;border-radius:8px;padding:10px 14px;max-width:300px;font-size:.82rem;line-height:1.6;color:#e8e8e8;z-index:999;pointer-events:none;box-shadow:0 4px 20px rgba(0,0,0,.5)"></div>
  <div id="mem-detail" style="display:none;width:340px;min-width:340px;border-left:1px solid #222;background:#111;overflow-y:auto;flex-direction:column;padding:0">
    <div style="padding:14px 16px;border-bottom:1px solid #1e1e1e;display:flex;align-items:center;justify-content:space-between">
      <span id="md-badge" style="font-size:.72rem;padding:3px 10px;border-radius:12px;font-weight:600;text-transform:uppercase;letter-spacing:.04em"></span>
      <button onclick="closeMemDetail()" style="background:none;border:none;color:#666;cursor:pointer;font-size:1.1rem;line-height:1">✕</button>
    </div>
    <div style="padding:16px">
      <div style="font-size:.72rem;color:#555;margin-bottom:12px">
        <span id="md-prov"></span> &nbsp;·&nbsp; <span id="md-conf"></span>% confianza
      </div>
      <div id="md-tags" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:14px"></div>
      <div id="md-content-view" style="font-size:.84rem;line-height:1.65;color:#ccc;white-space:pre-wrap;word-break:break-word"></div>
      <textarea id="md-content-edit" rows="8" style="display:none;width:100%;box-sizing:border-box;background:#1a1a1a;border:1px solid #333;border-radius:8px;color:#e8e8e8;font-size:.84rem;line-height:1.6;padding:10px;resize:vertical;font-family:inherit"></textarea>
      <div style="margin-top:8px;display:flex;align-items:center;gap:8px" id="md-conf-row" style="display:none">
        <label style="font-size:.75rem;color:#666">Confianza</label>
        <input id="md-conf-edit" type="range" min="0" max="100" style="flex:1">
        <span id="md-conf-val" style="font-size:.75rem;color:#a78bfa;width:30px;text-align:right"></span>
      </div>
      <div id="md-created" style="margin-top:14px;font-size:.7rem;color:#444"></div>
    </div>
    <div style="padding:12px 16px;border-top:1px solid #1e1e1e;display:flex;gap:8px;margin-top:auto">
      <button id="md-btn-edit" onclick="toggleMemEdit()" style="flex:1;padding:8px;background:#1a1a2a;border:1px solid #333;border-radius:8px;color:#a78bfa;cursor:pointer;font-size:.8rem">Editar</button>
      <button id="md-btn-save" onclick="saveMemEdit()" style="display:none;flex:1;padding:8px;background:#1a2a1a;border:1px solid #2a4a2a;border-radius:8px;color:#4ade80;cursor:pointer;font-size:.8rem">Guardar</button>
      <button onclick="deleteMemEntry()" style="flex:1;padding:8px;background:#2a1a1a;border:1px solid #4a2a2a;border-radius:8px;color:#f87171;cursor:pointer;font-size:.8rem">Eliminar</button>
    </div>
  </div>
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

const CONTINUE_TOKEN = '[[CONTINÚA]]';

async function callAPI(messages) {
  const res = await fetch('/v1/messages', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      model: 'ia-test-local',
      max_tokens: 1024,
      messages: messages.slice(-20),
    }),
  });
  const data = await res.json();
  return data.content?.[0]?.text ?? '(sin respuesta)';
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
    // First reply
    let reply;
    if (voiceMode) {
      reply = await callVoice(history);
    } else {
      reply = await callAPI(history);
    }
    const wantsContinue = reply.endsWith(CONTINUE_TOKEN);
    const cleanReply = reply.replace(CONTINUE_TOKEN, '').trimEnd();

    typing.remove();
    appendMsg('bot', cleanReply);
    history.push({role:'assistant', content: cleanReply});

    // One automatic continuation — never chain further (text mode only)
    if (wantsContinue && !voiceMode) {
      const typing2 = appendTyping();
      history.push({role:'user', content: '[continuá tu pensamiento]'});
      let reply2 = await callAPI(history);
      // Strip token if it appears again (but don't continue further)
      reply2 = reply2.replace(CONTINUE_TOKEN, '').trimEnd();
      typing2.remove();
      appendMsg('bot', reply2);
      history.push({role:'assistant', content: reply2});
    }

    await refreshMem();
  } catch(e) {
    typing.remove();
    appendMsg('bot', '⚠️ Error: ' + e.message);
  }

  send.disabled = false;
  input.focus();
}

// ── voice mode ────────────────────────────────────────────────────────────
const voiceBtn = document.getElementById('voice-btn');
let voiceMode = false;
let currentAudio = null;

if (voiceBtn) {
  voiceBtn.addEventListener('click', () => {
    voiceMode = !voiceMode;
    voiceBtn.classList.toggle('active', voiceMode);
    voiceBtn.textContent = voiceMode ? '🔊' : '🔇';
    voiceBtn.title = voiceMode ? 'Modo voz activo (click para desactivar)' : 'Modo voz';
  });
}

async function callVoice(messages) {
  const lastUserMsg = messages.filter(m => m.role === 'user').at(-1)?.content ?? '';
  const res = await fetch('/v1/voice', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      model: 'ia-test-local',
      max_tokens: 1024,
      messages: messages.slice(-20),
    }),
  });

  if (!res.ok) throw new Error(`Voice API error: ${res.status}`);

  const textHeader = res.headers.get('X-Text-Response');
  const textReply = textHeader
    ? decodeURIComponent(escape(atob(textHeader)))
    : '(respuesta de voz)';

  const blob = await res.blob();
  if (blob.size > 0) {
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    const url = URL.createObjectURL(blob);
    currentAudio = new Audio(url);
    currentAudio.onended = () => URL.revokeObjectURL(url);
    currentAudio.play().catch(err => console.warn('Audio play failed:', err));
  }

  return textReply;
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

// ── tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab + '-panel').classList.add('active');
    if (tab.dataset.tab === 'proposals') loadProposals();
  });
});

// ── proposals ─────────────────────────────────────────────────────────────
const moduleSelect = document.getElementById('module-select');
const btnGen = document.getElementById('btn-gen');
const propList = document.getElementById('proposals-list');

async function loadModules() {
  try {
    const r = await fetch('/v1/modules');
    const d = await r.json();
    moduleSelect.innerHTML = d.modules.map(m =>
      `<option value="${m}">${m}</option>`
    ).join('');
  } catch {}
}

function renderDiff(diff) {
  if (!diff || !diff.trim()) return '<div style="color:var(--muted);font-size:.8rem;padding:8px">Sin cambios concretos generados.</div>';
  const lines = diff.split('\\n');
  const rows = lines.map(l => {
    if (l.startsWith('+') && !l.startsWith('+++')) return `<div class="diff-line add">${escHtml(l)}</div>`;
    if (l.startsWith('-') && !l.startsWith('---')) return `<div class="diff-line del">${escHtml(l)}</div>`;
    if (l.startsWith('@@') || l.startsWith('---') || l.startsWith('+++')) return `<div class="diff-line meta">${escHtml(l)}</div>`;
    return `<div class="diff-line">${escHtml(l)}</div>`;
  }).join('');
  return `<div class="diff-block"><div class="diff-header">📄 Diff propuesto</div>${rows}</div>`;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function statusBadge(status) {
  const cls = status === 'APPLIED' ? 'status-applied' : status === 'ANALYZED' ? 'status-analyzed' : 'status-proposed';
  const label = status === 'APPLIED' ? '✓ Aplicada' : status === 'ANALYZED' ? '⚡ Analizada' : '◎ Propuesta';
  return `<span class="status-badge ${cls}">${label}</span>`;
}

function renderProposal(p) {
  const isApplied = p.status === 'APPLIED';
  const isAnalyzed = p.status === 'ANALYZED' || isApplied;

  const summary = p.ai_summary
    ? `<div class="ai-summary">${escHtml(p.ai_summary)}</div>`
    : '';

  const diffBlock = isAnalyzed && p.diff
    ? renderDiff(p.diff)
    : '';

  const actions = isApplied
    ? `<div class="applied-badge">✅ Cambio aplicado${p.applied_at ? ' · ' + p.applied_at.slice(0,10) : ''}</div>`
    : `
      <button class="btn-analyze" data-id="${p.id}" ${isAnalyzed ? 'style="opacity:.5"' : ''}>
        ${isAnalyzed ? '🔄 Re-analizar' : '🔍 Analizar con IA'}
      </button>
      ${isAnalyzed && p.diff ? `<button class="btn-apply" data-id="${p.id}">⚡ Aprobar y aplicar</button>` : ''}
    `;

  return `
    <div class="prop-card ${isAnalyzed ? 'analyzed' : ''} ${isApplied ? 'applied' : ''}" id="prop-${p.id}">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <div class="path">${escHtml(p.target_path)}</div>
        ${statusBadge(p.status)}
      </div>
      <h3>${escHtml(p.description)}</h3>
      <p>${escHtml(p.rationale)}</p>
      ${summary}
      ${diffBlock}
      <div class="prop-actions">${actions}</div>
    </div>`;
}

async function loadProposals() {
  try {
    const r = await fetch('/v1/proposals');
    const d = await r.json();
    if (!d.proposals.length) {
      propList.innerHTML = '<p style="color:var(--muted);font-size:.85rem;padding:4px 0">No hay propuestas todavía. Elegí un módulo y hacé clic en "Proponer mejora".</p>';
    } else {
      propList.innerHTML = d.proposals.map(renderProposal).join('');

      propList.querySelectorAll('.btn-analyze').forEach(btn => {
        btn.addEventListener('click', async () => {
          const card = document.getElementById('prop-' + btn.dataset.id);
          btn.disabled = true;
          btn.textContent = '⏳ Analizando...';
          try {
            const res = await fetch(`/v1/proposals/${btn.dataset.id}/analyze`, {method:'POST'});
            if (!res.ok) throw new Error(await res.text());
            await loadProposals();
          } catch(e) {
            btn.disabled = false;
            btn.textContent = '🔍 Analizar con IA';
            alert('Error al analizar: ' + e.message);
          }
        });
      });

      propList.querySelectorAll('.btn-apply').forEach(btn => {
        btn.addEventListener('click', async () => {
          if (!confirm('¿Aplicar este cambio al código fuente?')) return;
          btn.disabled = true;
          btn.textContent = '⏳ Aplicando...';
          try {
            const res = await fetch(`/v1/proposals/${btn.dataset.id}/apply`, {method:'POST'});
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || res.statusText);
            await loadProposals();
            const msg = data.reloaded
              ? `✅ Aplicado y recargado en caliente (${data.module})`
              : `✅ Aplicado. Reiniciá el server para que tome efecto.`;
            alert(msg);
          } catch(e) {
            btn.disabled = false;
            btn.textContent = '⚡ Aprobar y aplicar';
            alert('Error al aplicar: ' + e.message);
          }
        });
      });
    }
  } catch(e) {
    propList.innerHTML = `<p style="color:#eb5757;font-size:.85rem">Error cargando propuestas: ${e.message}</p>`;
  }
}

btnGen.addEventListener('click', async () => {
  const filePath = moduleSelect.value;
  if (!filePath) return;
  btnGen.disabled = true;
  btnGen.textContent = '⏳ Generando propuesta...';
  try {
    await fetch('/v1/proposals/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({file_path: filePath, description: 'general improvement'}),
    });
    loadProposals();
  } catch(e) {
    alert('Error: ' + e.message);
  }
  btnGen.disabled = false;
  btnGen.textContent = 'Proponer mejora';
});

loadModules();

// ── memory graph ──────────────────────────────────────────────────────────
let cy = null;
const tooltip = document.getElementById('graph-tooltip');

const NODE_COLORS = {
  knowledge:   '#7c6af7',
  hypothesis:  '#f7a76a',
  experience:  '#6af7b8',
  error:       '#f76a6a',
  experiment:  '#6ab8f7',
};

async function loadGraph() {
  const r = await fetch('/v1/memory/graph');
  const data = await r.json();

  if (cy) cy.destroy();

  if (!data.nodes.length) {
    document.getElementById('cy').innerHTML =
      '<p style="color:var(--muted);padding:40px;font-size:.9rem">La memoria está vacía — conversá con IA-test para que empiece a aprender.</p>';
    return;
  }

  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [...data.nodes, ...data.edges],
    style: [
      {
        selector: 'node',
        style: {
          'background-color': (ele) => NODE_COLORS[ele.data('type')] || '#888',
          'width': (ele) => 18 + ele.data('confidence') * 28,
          'height': (ele) => 18 + ele.data('confidence') * 28,
          'border-width': 2,
          'border-color': '#111',
          'background-opacity': 0.85,
          // No visible label — tooltip handles it
          'label': '',
        }
      },
      {
        selector: 'node:selected',
        style: {
          'border-color': '#a78bfa',
          'border-width': 3,
          'background-opacity': 1,
        }
      },
      {
        selector: 'node.hovered',
        style: {
          'border-color': '#fff',
          'border-width': 2,
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 1,
          'line-color': '#333',
          'target-arrow-color': '#555',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'opacity': 0.6,
        }
      },
      {
        selector: 'edge:selected',
        style: { 'line-color': '#7c6af7', 'opacity': 1 }
      },
    ],
    layout: {
      name: 'cose',
      animate: true,
      randomize: true,
      nodeRepulsion: 12000,
      idealEdgeLength: 80,
      edgeElasticity: 100,
      gravity: 0.25,
      padding: 50,
      fit: true,
    },
    backgroundColor: '#0d0d0d',
    minZoom: 0.3,
    maxZoom: 4,
  });

  cy.on('mouseover', 'node', (e) => {
    const d = e.target.data();
    e.target.addClass('hovered');
    const conf = (d.confidence * 100).toFixed(0);
    tooltip.innerHTML =
      `<div style="color:var(--accent2);font-weight:600;margin-bottom:4px">${d.type} · ${conf}% confianza</div>` +
      `<div style="color:#aaa;font-size:.72rem;margin-bottom:6px">${d.provenance || ''}</div>` +
      `<div>${d.label}</div>`;
    tooltip.style.display = 'block';
  });
  cy.on('mousemove', (e) => {
    tooltip.style.left = (e.originalEvent.clientX + 16) + 'px';
    tooltip.style.top  = (e.originalEvent.clientY - 12) + 'px';
  });
  cy.on('mouseout', 'node', (e) => {
    e.target.removeClass('hovered');
    tooltip.style.display = 'none';
  });
  cy.on('tap', 'node', (e) => {
    tooltip.style.display = 'none';
    openMemDetail(e.target.data('id'));
  });
  cy.on('tap', (e) => {
    if (e.target === cy) closeMemDetail();
  });
}

// ── memory detail panel ───────────────────────────────────────────────────
const BADGE_COLORS = {
  knowledge:  { bg: '#1a1535', color: '#a78bfa', border: '#3d2f8a' },
  hypothesis: { bg: '#2a1f0a', color: '#f7a76a', border: '#7a4f1a' },
  experience: { bg: '#0a2a1a', color: '#6af7b8', border: '#1a7a4a' },
  error:      { bg: '#2a0a0a', color: '#f76a6a', border: '#7a1a1a' },
  experiment: { bg: '#0a1a2a', color: '#6ab8f7', border: '#1a4a7a' },
};

let _currentMemId = null;
let _editMode = false;

async function openMemDetail(id) {
  _currentMemId = id;
  _editMode = false;
  const panel = document.getElementById('mem-detail');
  panel.style.display = 'flex';
  panel.style.flexDirection = 'column';

  // Reset edit state
  document.getElementById('md-content-view').style.display = '';
  document.getElementById('md-content-edit').style.display = 'none';
  document.getElementById('md-conf-row').style.display = 'none';
  document.getElementById('md-btn-edit').style.display = '';
  document.getElementById('md-btn-save').style.display = 'none';

  try {
    const r = await fetch(`/v1/memory/${id}`);
    if (!r.ok) { closeMemDetail(); return; }
    const d = await r.json();

    const bc = BADGE_COLORS[d.type] || { bg:'#1a1a1a', color:'#888', border:'#333' };
    const badge = document.getElementById('md-badge');
    badge.textContent = d.type;
    badge.style.background = bc.bg;
    badge.style.color = bc.color;
    badge.style.border = `1px solid ${bc.border}`;

    document.getElementById('md-prov').textContent = d.provenance || '';
    document.getElementById('md-conf').textContent = (d.confidence * 100).toFixed(0);

    const tagsEl = document.getElementById('md-tags');
    tagsEl.innerHTML = (d.tags || []).map(t =>
      `<span style="font-size:.7rem;padding:2px 8px;background:#1e1e2e;border:1px solid #2e2e4e;border-radius:10px;color:#888">${t}</span>`
    ).join('');

    document.getElementById('md-content-view').textContent = d.content;
    document.getElementById('md-content-edit').value = d.content;

    const confEdit = document.getElementById('md-conf-edit');
    confEdit.value = Math.round(d.confidence * 100);
    document.getElementById('md-conf-val').textContent = confEdit.value + '%';
    confEdit.oninput = () => {
      document.getElementById('md-conf-val').textContent = confEdit.value + '%';
    };

    const created = d.created_at ? new Date(d.created_at).toLocaleString('es-AR') : '';
    document.getElementById('md-created').textContent = created ? `Creado: ${created}` : '';
  } catch(err) {
    closeMemDetail();
  }
}

function closeMemDetail() {
  document.getElementById('mem-detail').style.display = 'none';
  _currentMemId = null;
  if (cy) cy.$(':selected').unselect();
}

function toggleMemEdit() {
  _editMode = !_editMode;
  document.getElementById('md-content-view').style.display = _editMode ? 'none' : '';
  document.getElementById('md-content-edit').style.display = _editMode ? '' : 'none';
  document.getElementById('md-conf-row').style.display = _editMode ? 'flex' : 'none';
  document.getElementById('md-btn-edit').style.display = _editMode ? 'none' : '';
  document.getElementById('md-btn-save').style.display = _editMode ? '' : 'none';
  if (_editMode) document.getElementById('md-content-edit').focus();
}

async function saveMemEdit() {
  if (!_currentMemId) return;
  const content = document.getElementById('md-content-edit').value.trim();
  const confidence = parseInt(document.getElementById('md-conf-edit').value) / 100;
  if (!content) return;

  const r = await fetch(`/v1/memory/${_currentMemId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, confidence }),
  });
  if (!r.ok) return;

  document.getElementById('md-content-view').textContent = content;
  document.getElementById('md-conf').textContent = Math.round(confidence * 100);

  if (cy) {
    const node = cy.$(`#${_currentMemId}`);
    if (node.length) {
      node.data('label', content.substring(0, 60).replace(/\\n/g, ' '));
      node.data('confidence', confidence);
      node.style('width', 18 + confidence * 28);
      node.style('height', 18 + confidence * 28);
    }
  }
  toggleMemEdit();
}

async function deleteMemEntry() {
  if (!_currentMemId) return;
  if (!confirm('¿Eliminar esta memoria?')) return;
  const r = await fetch(`/v1/memory/${_currentMemId}`, { method: 'DELETE' });
  if (!r.ok) return;
  if (cy) cy.$(`#${_currentMemId}`).remove();
  closeMemDetail();
  const countEl = document.getElementById('mem-count');
  if (countEl) countEl.textContent = Math.max(0, parseInt(countEl.textContent || '0') - 1);
}

// load graph when switching to that tab
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.dataset.tab === 'graph') loadGraph();
  });
});
</script>
</body>
</html>"""


# ── diff helpers ───────────────────────────────────────────────────────────

def _apply_unified_diff(original: list[str], diff_text: str) -> list[str]:
    """Apply a unified diff to a list of lines. Raises ValueError on failure."""
    result = list(original)
    hunk_lines: list[str] = []
    in_hunk = False
    old_start = 0
    offset = 0

    for raw in diff_text.splitlines(keepends=True):
        line = raw.rstrip("\n")
        if line.startswith("@@"):
            if in_hunk:
                result, offset = _apply_hunk(result, hunk_lines, old_start, offset)
            # parse @@ -a,b +c,d @@
            parts = line.split()
            old_range = parts[1].lstrip("-")
            old_start = int(old_range.split(",")[0]) - 1  # 0-indexed
            hunk_lines = []
            in_hunk = True
        elif in_hunk and (line.startswith(" ") or line.startswith("+") or line.startswith("-")):
            hunk_lines.append(raw if raw.endswith("\n") else raw + "\n")

    if in_hunk and hunk_lines:
        result, _ = _apply_hunk(result, hunk_lines, old_start, offset)

    return result


def _apply_hunk(
    lines: list[str], hunk: list[str], old_start: int, offset: int
) -> tuple[list[str], int]:
    pos = old_start + offset
    new_lines = list(lines)
    delete_count = sum(1 for l in hunk if l.startswith("-"))
    add_lines = [l[1:] for l in hunk if l.startswith("+")]

    del new_lines[pos : pos + delete_count]
    for i, l in enumerate(add_lines):
        new_lines.insert(pos + i, l)

    new_offset = offset + len(add_lines) - delete_count
    return new_lines, new_offset


def _path_to_module(rel_path: str) -> str | None:
    """Convert src/ia_test/foo/bar.py → ia_test.foo.bar, or None if unrecognised."""
    p = pathlib.Path(rel_path)
    try:
        # Strip leading src/ if present
        parts = list(p.with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if not parts:
            return None
        return ".".join(parts)
    except Exception:
        return None


# ── app factory ────────────────────────────────────────────────────────────

def create_app(
    provider: ModelProvider,
    db: MemoryDatabase,
    rate_limiter: RateLimiter | None = None,
    settings=None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await db.connect()
        loop = None
        if settings is not None:
            event_bus = EventBus()
            exploration_engine = ExplorationEngine(
                db=db,
                provider=provider,
                event_bus=event_bus,
                settings=settings,
            )
            loop = CuriosityLoop(
                db=db,
                provider=provider,
                event_bus=event_bus,
                settings=settings,
                rate_limiter=rate_limiter,
                exploration_engine=exploration_engine,
            )
            await loop.start()
        yield
        if loop is not None:
            await loop.stop()
        await db.close()

    _app = FastAPI(title="IA-test Local API", version="0.1.0", lifespan=lifespan)
    injector = ContextInjector(db)
    tool_executor = MemoryToolExecutor(db)

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
        # Identity always leads; memory context appended after if present
        combined = _IDENTITY_PROMPT
        if system_text:
            combined += "\n\n" + system_text
        msgs.append(Message(role="system", content=combined))
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

    @_app.get("/v1/memory/graph")
    async def memory_graph() -> JSONResponse:
        entries = await db.query(limit=200)
        nodes = [
            {
                "data": {
                    "id": e.id,
                    "label": e.content[:60].replace("\n", " "),
                    "type": e.type,
                    "confidence": e.confidence,
                    "provenance": e.provenance,
                }
            }
            for e in entries
        ]

        edges = []
        if entries:
            id_set = {e.id for e in entries}
            async with db._db.execute(
                "SELECT id, source_id, target_id, relation_type, weight FROM memory_relations"
            ) as cursor:
                for row in await cursor.fetchall():
                    if row["source_id"] in id_set and row["target_id"] in id_set:
                        edges.append({
                            "data": {
                                "id": row["id"],
                                "source": row["source_id"],
                                "target": row["target_id"],
                                "label": row["relation_type"],
                                "weight": row["weight"],
                            }
                        })

        return JSONResponse({"nodes": nodes, "edges": edges})

    @_app.get("/v1/memory/{entry_id}")
    async def get_memory_entry(entry_id: str) -> JSONResponse:
        entry = await db.get(entry_id)
        if entry is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({
            "id": entry.id,
            "type": entry.type,
            "content": entry.content,
            "confidence": entry.confidence,
            "provenance": entry.provenance,
            "tags": entry.tags,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        })

    @_app.patch("/v1/memory/{entry_id}")
    async def update_memory_entry(entry_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        content = body.get("content")
        confidence = body.get("confidence")
        entry = await db.update_entry(entry_id, content=content, confidence=confidence)
        if entry is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"ok": True, "id": entry.id})

    @_app.delete("/v1/memory/{entry_id}")
    async def delete_memory_entry(entry_id: str) -> JSONResponse:
        deleted = await db.delete_entry(entry_id)
        if not deleted:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"ok": True})

    @_app.get("/v1/proposals")
    async def list_proposals() -> JSONResponse:
        proposals = await db.list_proposals()
        return JSONResponse({"proposals": proposals})

    @_app.post("/v1/proposals/generate")
    async def generate_proposal(request: Request) -> JSONResponse:
        body = await request.json()
        file_path = body.get("file_path", "")
        description = body.get("description", "general improvement")

        repo_root = pathlib.Path(__file__).parent.parent.parent.parent
        inspector = ArchitectureInspector(repo_root)
        proposer = ImprovementProposer(provider=provider, inspector=inspector)

        try:
            target = ModificationTarget(file_path=file_path, description=description)
            proposal = await proposer.propose(target, branch_name=f"proposal-{uuid.uuid4().hex[:8]}")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        await db.save_proposal(proposal)
        return JSONResponse({
            "id": proposal.id,
            "target_path": proposal.target_path,
            "description": proposal.description,
            "rationale": proposal.rationale,
            "suggested_changes": proposal.suggested_changes,
            "status": proposal.status,
            "created_at": proposal.created_at,
        })

    @_app.post("/v1/proposals/{proposal_id}/review")
    async def review_proposal(proposal_id: str) -> JSONResponse:
        await db.mark_proposal_reviewed(proposal_id)
        return JSONResponse({"status": "reviewed", "id": proposal_id})

    @_app.post("/v1/proposals/{proposal_id}/analyze")
    async def analyze_proposal(proposal_id: str) -> JSONResponse:
        """Use GPT-4o to convert the raw proposal into a concrete diff + human summary."""
        from ia_test.self_modification.analyzer import ProposalAnalyzer
        from ia_test.core.settings import Settings as _Settings

        proposal = await db.get_proposal(proposal_id)
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")

        try:
            s = _Settings()
            analyzer = ProposalAnalyzer(api_key=s.openai_api_key)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Settings error: {exc}") from exc

        repo_root = pathlib.Path(__file__).parent.parent.parent.parent
        diff, summary = await analyzer.analyze(
            proposal_id=proposal_id,
            target_path=proposal["target_path"],
            description=proposal["description"],
            rationale=proposal["rationale"],
            suggested_changes=proposal["suggested_changes"],
            repo_root=repo_root,
        )

        await db.save_proposal_analysis(proposal_id, diff, summary)
        return JSONResponse({
            "id": proposal_id,
            "diff": diff,
            "ai_summary": summary,
            "has_changes": bool(diff.strip()),
        })

    @_app.post("/v1/proposals/{proposal_id}/apply")
    async def apply_proposal(proposal_id: str) -> JSONResponse:
        """Apply the analyzed diff to the target file and hot-reload the module."""
        import importlib
        import sys

        proposal = await db.get_proposal(proposal_id)
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
        if not proposal.get("diff"):
            raise HTTPException(status_code=400, detail="Proposal has no diff. Run /analyze first.")

        repo_root = pathlib.Path(__file__).parent.parent.parent.parent
        target = repo_root / proposal["target_path"]

        # Apply the unified diff by parsing it manually (safe subset)
        diff_text = proposal["diff"]
        try:
            original = target.read_text(encoding="utf-8").splitlines(keepends=True)
            patched = _apply_unified_diff(original, diff_text)
            target.write_text("".join(patched), encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Patch failed: {exc}") from exc

        # Hot-reload the module if it is already imported (behavior modules only)
        module_name = _path_to_module(proposal["target_path"])
        reloaded = False
        if module_name and module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
                reloaded = True
                logger.info("Hot-reloaded module: %s", module_name)
            except Exception as exc:
                logger.warning("Hot-reload failed for %s: %s", module_name, exc)

        await db.mark_proposal_applied(proposal_id)
        return JSONResponse({
            "id": proposal_id,
            "applied": True,
            "reloaded": reloaded,
            "module": module_name,
            "target_path": proposal["target_path"],
        })

    @_app.get("/v1/modules")
    async def list_modules() -> JSONResponse:
        repo_root = pathlib.Path(__file__).parent.parent.parent.parent
        inspector = ArchitectureInspector(repo_root)
        modules = inspector.list_modules()
        return JSONResponse({"modules": modules})

    @_app.get("/v1/voice/style")
    async def get_voice_style() -> JSONResponse:
        from ia_test.voice.engine import VoiceEngine
        from ia_test.core.settings import Settings as _Settings
        try:
            s = _Settings()
            engine = VoiceEngine(api_key=s.openai_api_key, db=db)
            style = await engine.load_style()
            return JSONResponse(style.to_dict())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @_app.post("/v1/voice")
    async def voice_message(req: MessagesRequest, request: Request) -> Response:
        """Generate a voice response. Returns MP3 audio bytes.

        Uses the same message flow as /v1/messages but synthesizes the
        text response to audio via OpenAI TTS with RON's current voice style.
        """
        from ia_test.voice.engine import VoiceEngine
        from ia_test.core.settings import Settings as _Settings

        await _check_rate(rate_limiter)

        system_text = await injector.enrich(req.messages, req.system)
        messages = _build_messages(req, system_text)

        try:
            final_text = await _complete_with_tools(
                provider=provider,
                messages=messages,
                tools=MEMORY_TOOLS,
                executor=tool_executor,
                rate_limiter=rate_limiter,
                temperature=req.temperature,
                db=db,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            s = _Settings()
            engine = VoiceEngine(api_key=s.openai_api_key, db=db)
            audio_bytes = await engine.synthesize(final_text)
        except Exception as exc:
            # TTS failed — return text response in a header so the client can fall back
            return Response(
                content=final_text.encode(),
                media_type="text/plain",
                headers={"X-Voice-Error": str(exc)},
            )

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"X-Text-Response": base64.b64encode(final_text.encode()).decode()},
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
            final_text = await _complete_with_tools(
                provider=provider,
                messages=messages,
                tools=MEMORY_TOOLS,
                executor=tool_executor,
                rate_limiter=rate_limiter,
                temperature=req.temperature,
                db=db,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Persist the exchange as a memory entry so the curiosity loop has material
        user_text = req.messages[-1].as_text() if req.messages else ""
        if user_text and not user_text.startswith("[continuá"):
            try:
                await db.store(
                    type="knowledge",
                    content=f"Q: {user_text[:300]}\nA: {final_text[:300]}",
                    confidence=0.6,
                    provenance="conversation",
                )
            except Exception:
                pass  # memory failure never blocks the response

        resp = MessagesResponse(
            id=msg_id,
            content=[ContentOut(text=final_text)],
            model=req.model,
            usage=UsageOut(input_tokens=0, output_tokens=0),
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
    return create_app(provider=provider, db=db, rate_limiter=rate_limiter, settings=settings)
