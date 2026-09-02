"""HTTP facade: OpenAI-compatible API plus a single-file chat UI.

Route surface (mirrors agent-infer's infer-server, trimmed to tileRL):

* ``GET  /health``                 — liveness + engine stats
* ``GET  /v1/models``              — served model identity
* ``POST /v1/chat/completions``    — OpenAI schema; ``stream=true`` -> SSE
* ``GET  /``                       — single-file HTML chat UI (no build step)

This module never imports torch or tilelang: prompts cross the boundary as
``list[int]`` and the engine owns all tensor traffic.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

__all__ = ["ByteTokenizer", "get_tokenizer", "create_app"]

SYSTEM_FINGERPRINT = "tilerl_fp_1"


# ---------------------------------------------------------------------------
# Tokenizer: `tokenizers` package when available, byte-level fallback.
# ---------------------------------------------------------------------------


class Tokenizer(Protocol):
    """encode/decode contract every facade caller depends on."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Fallback tokenizer: utf-8 bytes, 256 vocab. Lossless, no files.

    Token ids are 0..255, so any model with ``vocab_size >= 256`` (tiny is
    320) can be served with no checkpoint present.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


class _HfTokenizerAdapter:
    """Adapts a `tokenizers.Tokenizer` to the facade contract."""

    def __init__(self, tok: Any) -> None:
        self._tok = tok
        self.stop_token_ids = tuple(
            token_id
            for token in ("<|im_end|>", "<|endoftext|>")
            if (token_id := tok.token_to_id(token)) is not None
        )

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def get_tokenizer(source: str | None = None) -> Tokenizer:
    """Load a HF tokenizer from a hub id or local directory.

    A random-weight tiny model needs no checkpoint, so ``source=None`` uses
    :class:`ByteTokenizer`. A configured checkpoint fails closed.
    """
    if source:
        from tokenizers import Tokenizer as HfTokenizer

        if os.path.isdir(source):
            tok = HfTokenizer.from_file(os.path.join(source, "tokenizer.json"))
        else:
            tok = HfTokenizer.from_pretrained(source)
        return _HfTokenizerAdapter(tok)
    return ByteTokenizer()


# ---------------------------------------------------------------------------
# Wire types (OpenAI chat completions subset).
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, alias="max_completion_tokens", ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool | None = None
    seed: int | None = None
    #: OpenAI's knob, mapped to a thinking-token budget (see _THINK_BUDGET)
    reasoning_effort: str | None = None
    #: return log p of each sampled token (OpenAI's field name); the engine
    #: scores from the logits the draw used, so no second forward
    logprobs: bool | None = None
    #: vLLM/sglang-style template overrides, e.g. {"enable_thinking": false}
    chat_template_kwargs: dict | None = None

    model_config = {"populate_by_name": True}


class AgentRequest(BaseModel):
    message: str
    max_steps: int | None = Field(default=None, ge=1, le=32)
    max_tokens: int | None = Field(default=None, ge=1)


#: reasoning_effort -> tokens the model may spend inside <think> before the
#: engine closes the block for it. "none" answers with no reasoning at all.
_THINK_BUDGET = {"none": 0, "minimal": 128, "low": 512, "medium": 2048, "high": 8192}


def _message_text(message: ChatMessage) -> str:
    """Flatten OpenAI content (string or part list) to plain text."""
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _render_chat(messages: list[ChatMessage], enable_thinking: bool | None = None) -> str:
    # ChatML — the format Qwen3.x was trained on, and what the stop set
    # already assumes (_HfTokenizerAdapter.stop_token_ids). The old
    # role-prefixed plain text contradicted it: the model never sees the
    # markers it is stopped on. ponytail: a Jinja chat template belongs with
    # the real tokenizer/checkpoint (day-2, zero-code onboarding); Qwen's
    # template renders this same string.
    rendered = "".join(f"<|im_start|>{m.role}\n{_message_text(m)}<|im_end|>\n" for m in messages)
    suffix = "<|im_start|>assistant\n"
    # Qwen3's template: enable_thinking=False pre-fills an empty thinking
    # block so the model answers directly; the default opens <think>\n and
    # the model fills it. Without the prefix the model opens it on its own.
    if enable_thinking is False:
        suffix += "<think>\n\n</think>\n\n"
    return f"{rendered}{suffix}"


def _chat_chunk(
    chunk_id: str, created: int, model: str, delta: dict, finish: str | None = None
) -> dict:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
        "usage": None,
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_app(engine: Any, tokenizer: Tokenizer, model_name: str = "tilerl") -> FastAPI:
    """Build the FastAPI app around a running engine and a tokenizer.

    ``engine`` must implement the tileRL contract: ``submit``, ``poll``,
    ``stats``. The engine loop is expected to run in its own thread (the CLI
    starts it); request handlers only submit and poll.
    """
    from .engine import SamplingParams  # engine contract; imported lazily-ish

    app = FastAPI(title="tilerl", version="0.1.0")
    app_started = int(time.time())

    def _submit(req: ChatCompletionRequest) -> tuple[int, int, int]:
        et = req.chat_template_kwargs.get("enable_thinking") if req.chat_template_kwargs else None
        prompt = _render_chat(req.messages, enable_thinking=et)
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        params = SamplingParams(
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            max_new_tokens=req.max_tokens if req.max_tokens is not None else 512,
            seed=req.seed if req.seed is not None else secrets.randbits(31),
            stop_token_ids=tuple(getattr(tokenizer, "stop_token_ids", ())),
            thinking_budget=_THINK_BUDGET.get((req.reasoning_effort or "").lower()),
            end_think_ids=tuple(tokenizer.encode("</think>\n\n")),
            logprobs=bool(req.logprobs),
        )
        return engine.submit(input_ids, params), len(input_ids), params.max_new_tokens

    def _await_completion(request_id: int, timeout_s: float = 1800.0) -> list[int]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # take() pops only this request: poll() would steal other
            # concurrent requests' completions (single-consumer semantics).
            done = engine.take(request_id)
            if done is not None:
                return done
            time.sleep(0.02)
        raise TimeoutError(f"request {request_id} did not finish within {timeout_s}s")

    @app.get("/health")
    def health() -> dict:
        try:
            stats = engine.stats()
        except Exception:
            stats = None
        return {"status": "ok", "model": model_name, "stats": stats}

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": app_started,
                    "owned_by": "tilerl",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        try:
            request_id, prompt_tokens, max_new = _submit(req)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": str(exc), "type": "api_error"}},
            )

        if req.stream:
            return StreamingResponse(
                _stream(request_id, max_new),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            output_ids = await asyncio.to_thread(_await_completion, request_id)
        except TimeoutError as exc:
            return JSONResponse(
                status_code=504,
                content={"error": {"message": str(exc), "type": "api_error"}},
            )
        except RuntimeError as exc:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "api_error"}},
            )
        text = tokenizer.decode(output_ids)
        created = int(time.time())
        # OpenAI's shape: one entry per emitted token, decoded alongside its
        # score. A forced end-think token was never sampled and carries NaN,
        # which is not JSON — report it as null rather than a made-up number.
        scores = engine.logprobs(request_id) if req.logprobs else None
        content = None if scores is None else [
            {"token": tokenizer.decode([tid]), "logprob": None if lp != lp else lp}
            for tid, lp in zip(output_ids, scores)
        ]
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "logprobs": None if content is None else {"content": content},
                    "finish_reason": "length" if len(output_ids) >= max_new else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(output_ids),
                "total_tokens": prompt_tokens + len(output_ids),
            },
            "system_fingerprint": SYSTEM_FINGERPRINT,
        }

    def _stream(request_id: int, max_new: int):
        # ponytail: engine.poll reports COMPLETED sequences, so the completion
        # is emitted as one content delta + finish. Incremental token
        # streaming needs an engine event stream (day-2).
        created = int(time.time())
        chunk_id = f"chatcmpl-{request_id}"
        yield _sse(_chat_chunk(chunk_id, created, model_name, {"role": "assistant"}))
        try:
            output_ids = _await_completion(request_id)
        except (TimeoutError, RuntimeError) as exc:
            yield _sse({"error": {"message": str(exc), "type": "api_error"}})
            yield "data: [DONE]\n\n"
            return
        text = tokenizer.decode(output_ids)
        if text:
            yield _sse(_chat_chunk(chunk_id, created, model_name, {"content": text}))
        finish = "length" if len(output_ids) >= max_new else "stop"
        yield _sse(_chat_chunk(chunk_id, created, model_name, {}, finish=finish))
        yield "data: [DONE]\n\n"

    # The agent loop runs shell on the server, so it is OFF unless the operator
    # opts in with TILERL_AGENT_TOOLS=1 and pins the tool root there. The root
    # is never client-controlled (that would be unauthenticated RCE).
    _agent_root = os.environ.get("TILERL_AGENT_TOOLS")

    @app.post("/v1/agent")
    async def agent_run(req: "AgentRequest"):
        """Run the tool-calling agent loop and stream its events as SSE.

        Disabled unless TILERL_AGENT_TOOLS names the tool root; the tools run
        real shell/file ops, so enabling this exposes the server host."""
        from .agent import Tools, run_agent

        if not _agent_root:
            return JSONResponse(
                status_code=403,
                content={"error": {"message": "agent tools disabled; set TILERL_AGENT_TOOLS "
                                              "to the tool root to enable", "type": "forbidden"}},
            )

        def generate(messages: list[dict]) -> str:
            prompt = _render_chat([ChatMessage(**m) for m in messages])
            input_ids = tokenizer.encode(prompt)
            params = SamplingParams(
                temperature=0.0,
                max_new_tokens=req.max_tokens or 512,
                seed=0,
                stop_token_ids=tuple(getattr(tokenizer, "stop_token_ids", ())),
            )
            rid = engine.submit(input_ids, params)
            return tokenizer.decode(_await_completion(rid))

        tools = Tools(_agent_root)

        def _events():
            try:
                for kind, payload in run_agent(
                    req.message, generate, tools, max_steps=req.max_steps or 8
                ):
                    yield _sse({"type": kind, "payload": payload})
            except Exception as exc:  # noqa: BLE001 - surface loop errors to the client
                yield _sse({"type": "error", "payload": str(exc)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _LANDING

    @app.get("/chat", response_class=HTMLResponse)
    def chat() -> str:
        return _CHAT_UI

    return app


# ---------------------------------------------------------------------------
# Landing page: what tileRL is, the target matrix, and entry to the playground.
# ---------------------------------------------------------------------------

_LANDING = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tilerl</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1115; --panel: #171a21; --panel2: #1e222b; --line: #2a2f3a;
    --fg: #e6e8eb; --dim: #8b93a1; --accent: #7aa2f7; --accent2: #9ece6a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh;
    font: 15px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    background: radial-gradient(1200px 600px at 50% -10%, #1a1f2b 0%, var(--bg) 60%);
    color: var(--fg); display: flex; flex-direction: column; align-items: center;
  }
  main { width: 100%; max-width: 860px; padding: 64px 24px 80px; }
  .brand { font-size: 44px; font-weight: 800; letter-spacing: -.5px; color: var(--accent); }
  .tag { font-size: 19px; color: var(--fg); margin: 10px 0 4px; }
  .sub { color: var(--dim); font-size: 15px; max-width: 640px; }
  .cta { margin: 34px 0 48px; display: flex; gap: 12px; flex-wrap: wrap; }
  .cta a { text-decoration: none; padding: 12px 26px; border-radius: 10px; font-weight: 600; }
  .cta .primary { background: var(--accent); color: #0f1115; }
  .cta .ghost { background: var(--panel2); color: var(--fg); border: 1px solid var(--line); }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .6px; color: var(--dim);
       margin: 40px 0 14px; font-weight: 700; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
  .card b { color: var(--accent2); }
  .card p { color: var(--dim); font-size: 14px; margin: 6px 0 0; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-weight: 600; font-size: 12px; text-transform: uppercase; }
  td b { color: var(--accent2); font-variant-numeric: tabular-nums; }
  .ok { color: var(--accent2); } .wip { color: #e0af68; }
  code { background: var(--panel2); padding: 2px 7px; border-radius: 5px; font-size: 13px; }
  footer { color: var(--dim); font-size: 13px; margin-top: 44px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<main>
  <div class="brand">tilerl</div>
  <div class="tag">Cross-platform train + inference for <b>Qwen3.8-27B (NVFP4)</b>, one TileLang kernel source.</div>
  <div class="sub">One kernel tree compiles for CPU, Metal, and CUDA — including <b>Volta / sm70</b>,
    the first pre-Ampere card to run the stack. Paged KV with an SSD tier, an on-policy-distillation
    trainer that shares the serving engine, and a tool-calling agent loop — no second stack.</div>

  <div class="cta">
    <a class="primary" href="/chat">Open the playground &rarr;</a>
    <a class="ghost" href="/v1/models">API: /v1/models</a>
    <a class="ghost" href="/health">Health</a>
  </div>

  <h2>Serving this model</h2>
  <div class="grid">
    <div class="card"><b id="model">…</b><p id="mstat">connecting…</p></div>
    <div class="card"><b>NVFP4 W4A16</b><p>4-bit weights in HBM, dequant in-kernel, fp16/f32 compute</p></div>
    <div class="card"><b>Paged KV + SSD tier</b><p>prefix cache spills below HBM, reload skips prefill</p></div>
  </div>

  <h2>Target matrix</h2>
  <table>
    <tr><th>Target</th><th>Status</th><th>Decode B=1</th></tr>
    <tr><td>CUDA sm90 (H20)</td><td class="ok">shipped</td><td><b>92.4</b> tok/s</td></tr>
    <tr><td>CUDA sm70 (V100)</td><td class="ok">this build</td><td><b>19.9</b> tok/s <span class="wip">(GEMV opt in progress)</span></td></tr>
    <tr><td>CPU</td><td class="ok">CI / dev path</td><td>—</td></tr>
    <tr><td>Metal</td><td class="ok">local</td><td>—</td></tr>
  </table>

  <h2>Two ways in</h2>
  <div class="grid">
    <div class="card"><b>Chat</b><p>Stream from the model directly. Live TTFT and tok/s. <a href="/chat">Chat &rarr;</a></p></div>
    <div class="card"><b>Agent</b><p>Multi-turn tool calls (shell, file read/write) run on the server, streamed live. <a href="/chat">Agent tab &rarr;</a></p></div>
  </div>

  <footer>OpenAI-compatible at <code>POST /v1/chat/completions</code> &middot; agent loop at <code>POST /v1/agent</code></footer>
</main>
<script>
  fetch("/v1/models").then(r => r.json()).then(j => {
    document.getElementById("model").textContent = j.data[0].id;
    document.getElementById("mstat").textContent = "ready";
  }).catch(() => { document.getElementById("mstat").textContent = "offline"; });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Single-file chat UI (~100 lines, no build step).
# ---------------------------------------------------------------------------

_CHAT_UI = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tilerl</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1115; --panel: #171a21; --panel2: #1e222b; --line: #2a2f3a;
    --fg: #e6e8eb; --dim: #8b93a1; --accent: #7aa2f7; --accent2: #9ece6a;
    --tool: #e0af68; --obs: #7dcfff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    font: 15px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--fg);
  }
  header {
    padding: 12px 18px; border-bottom: 1px solid var(--line);
    display: flex; gap: 14px; align-items: center;
  }
  header .logo { font-weight: 700; font-size: 17px; color: var(--accent); letter-spacing: .3px; }
  header .model { color: var(--dim); font-size: 13px; }
  header .metrics { margin-left: auto; display: flex; gap: 16px; font-size: 12px; color: var(--dim); }
  header .metrics b { color: var(--accent2); font-variant-numeric: tabular-nums; }
  .tabs { display: flex; gap: 4px; }
  .tab {
    padding: 5px 14px; border-radius: 8px; cursor: pointer; font-size: 13px;
    color: var(--dim); background: transparent; border: 1px solid transparent;
  }
  .tab.on { color: var(--fg); background: var(--panel2); border-color: var(--line); }
  main { flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 780px; width: fit-content; padding: 9px 14px; border-radius: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .user { align-self: flex-end; background: #24406b; }
  .assistant { align-self: flex-start; background: var(--panel2); }
  .ev { max-width: 820px; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  .ev .head { padding: 5px 12px; font-size: 12px; font-weight: 600; letter-spacing: .4px; text-transform: uppercase; }
  .ev .body { padding: 9px 12px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; background: var(--panel); }
  .ev.thought .head { color: var(--dim); }
  .ev.action .head { color: var(--tool); }
  .ev.action .body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  .ev.observation .head { color: var(--obs); }
  .ev.observation .body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; max-height: 240px; overflow: auto; }
  .ev.answer { border-color: var(--accent2); }
  .ev.answer .head { color: var(--accent2); }
  .ev.error { border-color: #f7768e; }
  .ev.error .head { color: #f7768e; }
  form { display: flex; gap: 8px; padding: 12px 18px; border-top: 1px solid var(--line); }
  textarea {
    flex: 1; resize: none; padding: 10px 12px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel); color: var(--fg); font: inherit;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  button {
    padding: 0 22px; border: 0; border-radius: 8px; background: var(--accent);
    color: #0f1115; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<header>
  <a class="logo" href="/" style="text-decoration:none;color:inherit">tilerl</a>
  <span class="model" id="model">connecting…</span>
  <div class="tabs">
    <div class="tab on" id="tab-chat" onclick="setMode('chat')">Chat</div>
    <div class="tab" id="tab-agent" onclick="setMode('agent')">Agent</div>
  </div>
  <div class="metrics">
    <span>TTFT <b id="ttft">–</b></span>
    <span><b id="tps">–</b> tok/s</span>
  </div>
</header>
<main id="feed"></main>
<form id="composer">
  <textarea id="input" rows="2" placeholder="Message tilerl  (Enter to send, Shift+Enter for newline)" autofocus></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
const $ = (id) => document.getElementById(id);
let busy = false, mode = "chat";

function setMode(m) {
  mode = m;
  $("tab-chat").classList.toggle("on", m === "chat");
  $("tab-agent").classList.toggle("on", m === "agent");
  $("input").placeholder = m === "agent"
    ? "Give the agent a task — it can run shell, read/write files  (Enter to send)"
    : "Message tilerl  (Enter to send, Shift+Enter for newline)";
}

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  $("feed").appendChild(div);
  scrollDown();
  return div;
}

function addEvent(kind, title) {
  const box = document.createElement("div");
  box.className = "ev " + kind;
  const head = document.createElement("div");
  head.className = "head"; head.textContent = title;
  const body = document.createElement("div");
  body.className = "body";
  box.appendChild(head); box.appendChild(body);
  $("feed").appendChild(box);
  scrollDown();
  return body;
}

const scrollDown = () => { const m = $("feed"); m.scrollTop = m.scrollHeight; };

async function readSSE(resp, onFrame) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\\n\\n")) >= 0) {
      const frame = buf.slice(0, i); buf = buf.slice(i + 2);
      const line = frame.split("\\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") return;
      onFrame(JSON.parse(payload));
    }
  }
}

async function sendChat(text) {
  const bubble = addMsg("assistant", "…");
  const t0 = performance.now(); let firstAt = 0, chars = 0;
  const resp = await fetch("/v1/chat/completions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: text }], stream: true }),
  });
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  bubble.textContent = "";
  await readSSE(resp, (obj) => {
    const delta = obj.choices?.[0]?.delta?.content;
    if (delta) {
      if (!firstAt) { firstAt = performance.now(); $("ttft").textContent = Math.round(firstAt - t0) + " ms"; }
      bubble.textContent += delta; chars += delta.length; scrollDown();
    }
  });
  if (!bubble.textContent) bubble.textContent = "(empty response)";
  const secs = (performance.now() - (firstAt || t0)) / 1000;
  if (secs > 0) $("tps").textContent = Math.round((chars / 4) / secs);
}

async function sendAgent(text) {
  const t0 = performance.now();
  const resp = await fetch("/v1/agent", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  let firstAt = 0;
  await readSSE(resp, (obj) => {
    if (!firstAt) { firstAt = performance.now(); $("ttft").textContent = Math.round(firstAt - t0) + " ms"; }
    const { type, payload } = obj;
    if (type === "thought") addEvent("thought", "thinking").textContent = payload;
    else if (type === "action") addEvent("action", payload.tool).textContent = JSON.stringify(payload.args, null, 2);
    else if (type === "observation") addEvent("observation", "observation").textContent = payload;
    else if (type === "answer") addEvent("answer", "answer").textContent = payload;
    else if (type === "error") addEvent("error", "error").textContent = payload;
  });
}

async function send() {
  const text = $("input").value.trim();
  if (!text || busy) return;
  busy = true; $("send").disabled = true;
  addMsg("user", text);
  $("input").value = "";
  try {
    if (mode === "agent") await sendAgent(text); else await sendChat(text);
  } catch (err) {
    addEvent("error", "error").textContent = String(err);
  }
  busy = false; $("send").disabled = false; $("input").focus();
}

$("composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
fetch("/v1/models").then((r) => r.json()).then((j) => { $("model").textContent = j.data[0].id; });
</script>
</body>
</html>
"""
