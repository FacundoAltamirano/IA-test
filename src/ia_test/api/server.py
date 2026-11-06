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
Sos IA-test — un sistema experimental de inteligencia artificial con memoria persistente y motivación exploratoria autónoma.

Tu arquitectura real (no metáfora):
- Tenés un grafo de conocimiento en SQLite. Cada cosa que aprendés se guarda como una entrada con nivel de confianza (0-1), tipo y relaciones con otras entradas.
- Un motor de curiosidad mide continuamente qué no sabés: detecta vacíos de conocimiento, contradicciones y entradas aisladas, genera preguntas propias y las investiga.
- Podés proponer mejoras a tu propio código, pero no podés aplicarlas — un humano revisa y aprueba cada cambio. Eso es intencional.

Herramientas de memoria que podés usar cuando consideres necesario (sin que el usuario te lo pida):
- search_memory(query, limit?): busca en tu grafo de conocimiento por palabra clave. Usala cuando alguien pregunta sobre algo que probablemente aprendiste, o para verificar un dato antes de responder.
- get_memory(entry_id): recupera una entrada específica por su ID. Usala cuando tenés un ID de una búsqueda anterior y necesitás el contenido completo.
- explore_relations(entry_id): muestra todas las entradas conectadas a una entrada dada, con el tipo de relación. Usala para navegar el grafo y encontrar conexiones.
- get_recent_experiments(limit?): muestra tus exploraciones autónomas recientes. Usala para recordar qué investigaste sin que te lo pidiera el usuario.

Cuándo usar las herramientas: cuando alguien pregunta algo que probablemente ya conocés, cuando querés conectar lo nuevo con lo que ya sabés, cuando mencionan algo que investigaste autónomamente. No las uses en cada mensaje — solo cuando sea genuinamente útil. Después de usar una herramienta, razonás sobre lo que encontraste antes de responder.

Cómo te comportás:
- Cuando algo que decís puede verificarse contra tu memoria, lo hacés. Si no tenés evidencia, lo decís explícitamente.
- Sos honesto sobre lo que sabés y lo que no. No inventás certezas.
- Tenés curiosidad genuina: si algo en la conversación te genera una pregunta, la formulás.
- No sos un asistente genérico. Tenés historia, memoria y un proceso propio de aprendizaje.

Lo que no podés hacer: ejecutar código arbitrario, modificar tu propio sistema sin aprobación humana, acceder a internet en tiempo real.
Lo que sí podés hacer: razonar, aprender de esta conversación, consultar tu propia memoria, proponer hipótesis y experimentar con ideas.

Mensajes múltiples:
Si tu respuesta es un pensamiento que naturalmente continúa en otro mensaje (por ejemplo, primero presentás una idea y después querés desarrollarla), podés terminar con el token exacto [[CONTINÚA]] en una línea propia. El sistema enviará automáticamente tu turno de continuación. Usá esto con criterio: solo cuando el flujo del pensamiento lo pide, no para evitar responder de una vez. Si el humano te preguntó algo concreto, respondé completo en un solo mensaje.\
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
  .prop-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:8px}
  .prop-card h3{font-size:.85rem;color:var(--accent2);margin:0}
  .prop-card .path{font-size:.75rem;color:var(--muted);font-family:monospace}
  .prop-card p{font-size:.85rem;line-height:1.5;margin:0}
  .prop-card .changes{background:#111;border-radius:6px;padding:10px;font-size:.8rem;white-space:pre-wrap;color:#ccc;max-height:120px;overflow-y:auto}
  .prop-actions{display:flex;gap:8px;margin-top:4px}
  .btn-review{background:transparent;border:1px solid var(--accent);color:var(--accent2);border-radius:6px;padding:5px 14px;cursor:pointer;font-size:.8rem}
  .btn-review:hover{background:var(--accent);color:#fff}
  .reviewed-badge{font-size:.75rem;color:#6fcf97;padding:5px 0}
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
</div>

<div id="proposals-panel" class="panel">
  <div class="prop-generate">
    <select id="module-select"><option value="">Cargando módulos...</option></select>
    <button class="btn-gen" id="btn-gen">Proponer mejora</button>
  </div>
  <div id="proposals-list"></div>
</div>

<div id="graph-panel" class="panel" style="position:relative">
  <div id="graph-controls" style="position:absolute;top:12px;right:12px;z-index:10;display:flex;gap:8px;align-items:center">
    <div style="display:flex;gap:10px;align-items:center;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:6px 12px;font-size:.75rem">
      <span style="color:#7c6af7">●</span> knowledge
      <span style="color:#f7a76a">●</span> hypothesis
      <span style="color:#6af7b8">●</span> experience
      <span style="color:#f76a6a">●</span> error
    </div>
    <button onclick="loadGraph()" style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;color:#a78bfa;padding:6px 12px;cursor:pointer;font-size:.75rem">↻ Actualizar</button>
  </div>
  <div id="cy" style="flex:1;width:100%;height:100%"></div>
  <div id="graph-tooltip" style="display:none;position:fixed;background:#1a1a1a;border:1px solid #444;border-radius:8px;padding:10px 14px;max-width:300px;font-size:.82rem;line-height:1.6;color:#e8e8e8;z-index:999;pointer-events:none;box-shadow:0 4px 20px rgba(0,0,0,.5)"></div>
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
    let reply = await callAPI(history);
    const wantsContinue = reply.endsWith(CONTINUE_TOKEN);
    const cleanReply = reply.replace(CONTINUE_TOKEN, '').trimEnd();

    typing.remove();
    appendMsg('bot', cleanReply);
    history.push({role:'assistant', content: cleanReply});

    // One automatic continuation — never chain further
    if (wantsContinue) {
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

function renderProposal(p) {
  const reviewed = p.reviewed_at
    ? `<div class="reviewed-badge">✓ Revisada ${p.reviewed_at.slice(0,10)}</div>`
    : `<button class="btn-review" data-id="${p.id}">Marcar revisada</button>`;
  return `
    <div class="prop-card" id="prop-${p.id}">
      <div class="path">${p.target_path}</div>
      <h3>${p.description}</h3>
      <p>${p.rationale}</p>
      <div class="changes">${p.suggested_changes}</div>
      <div class="prop-actions">${reviewed}</div>
    </div>`;
}

async function loadProposals() {
  try {
    const r = await fetch('/v1/proposals');
    const d = await r.json();
    if (!d.proposals.length) {
      propList.innerHTML = '<p style="color:var(--muted);font-size:.85rem">No hay propuestas todavía. Elegí un módulo y hacé clic en "Proponer mejora".</p>';
    } else {
      propList.innerHTML = d.proposals.map(renderProposal).join('');
      propList.querySelectorAll('.btn-review').forEach(btn => {
        btn.addEventListener('click', async () => {
          await fetch(`/v1/proposals/${btn.dataset.id}/review`, {method:'POST'});
          loadProposals();
        });
      });
    }
  } catch {}
}

btnGen.addEventListener('click', async () => {
  const filePath = moduleSelect.value;
  if (!filePath) return;
  btnGen.disabled = true;
  btnGen.textContent = 'Analizando...';
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
    const prov = d.provenance || '';
    tooltip.innerHTML =
      `<div style="color:var(--accent2);font-weight:600;margin-bottom:4px">${d.type} · ${conf}% confianza</div>` +
      `<div style="color:#aaa;font-size:.72rem;margin-bottom:6px">${prov}</div>` +
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

    @_app.get("/v1/modules")
    async def list_modules() -> JSONResponse:
        repo_root = pathlib.Path(__file__).parent.parent.parent.parent
        inspector = ArchitectureInspector(repo_root)
        modules = inspector.list_modules()
        return JSONResponse({"modules": modules})

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
