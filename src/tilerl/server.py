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
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .prompt import render_prompt, sampling, strip_think
from .tokenizer import ByteTokenizer, Tokenizer, get_tokenizer  # noqa: F401

__all__ = ["ByteTokenizer", "get_tokenizer", "create_app"]

SYSTEM_FINGERPRINT = "tilerl_fp_1"


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
    #: OpenAI's {"include_usage": true} -- adds a final choices-less usage chunk. Opt-in
    #: because a client that indexes choices[0] on every frame breaks on it.
    stream_options: dict | None = None
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


#: reasoning_effort -> cap on <think> tokens; "none" switches thinking off in the prompt.
_MAX_THINK = {"none": 0, "minimal": 128, "low": 512, "medium": 2048, "high": 8192}


def _render_chat(messages: list[ChatMessage], thinking: bool | None = None,
                 reasoning_effort: str | None = None) -> str:
    return render_prompt([m.model_dump() for m in messages], thinking=thinking,
                         effort=reasoning_effort)


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
    app = FastAPI(title="tilerl", version="0.1.0")
    app_started = int(time.time())

    def _submit(req: ChatCompletionRequest) -> tuple[int, int, int]:
        cap = _MAX_THINK.get((req.reasoning_effort or "").lower())
        kw = req.chat_template_kwargs or {}
        thinking = kw.get("enable_thinking")
        if thinking is None:
            # The checkpoint's template treats an undefined enable_thinking as TRUE and
            # always emits <think> one way or the other, so leaving it unset made the model
            # open the tag in its own output. Default to the template's answer, but only for
            # a tokenizer that HAS the tag: ByteTokenizer spells it as 7 raw bytes, and its
            # bare turn is the tiny/dev path the None state exists for.
            thinking = (len(tokenizer.encode("<think>")) == 1 or None) if cap != 0 else False
        input_ids = tokenizer.encode(_render_chat(
            req.messages, thinking, kw.get("reasoning_effort") or req.reasoning_effort
        ))
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        params = sampling(tokenizer, thinking, req.max_tokens if req.max_tokens is not None else 512,
                          temperature=req.temperature, top_p=req.top_p, max_think_tokens=cap,
                          seed=req.seed, logprobs=bool(req.logprobs))
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
                _stream(request_id, max_new, prompt_tokens, bool(
                    (req.stream_options or {}).get("include_usage")
                )),
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
        text = strip_think(tokenizer.decode(output_ids))  # reasoning is the model's, not the reply
        created = int(time.time())
        # OpenAI's shape: one entry per emitted token, decoded alongside its
        # score. A forced end-think token was never sampled and carries NaN,
        # which is not JSON — report it as null rather than a made-up number.
        # The scores cover every SAMPLED token, reasoning included, while
        # message.content above is the stripped display text: the two are
        # deliberately different lengths. Truncating this list to match the
        # text would break the RL join, which scores what was sampled.
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

    def _stream(request_id: int, max_new: int, prompt_tokens: int, include_usage: bool):
        created = int(time.time())
        chunk_id = f"chatcmpl-{request_id}"
        yield _sse(_chat_chunk(chunk_id, created, model_name, {"role": "assistant"}))
        deadline = time.monotonic() + 1800.0
        sent = 0  # characters of the STRIPPED reply already emitted
        seen = 0  # tokens already decoded, so a quiet poll costs nothing
        try:
            while True:
                # peek() is lock-free; take() blocks on the engine lock for a whole
                # forward (325 ms of a 335 ms run measured), so calling it each poll
                # would starve this loop back to one delta. Poll peek, take once it
                # reports the request has left the queues.
                # ponytail: one delta per ~21 tokens, narrow step()'s lock for per-token
                live = engine.peek(request_id)
                if live is None:
                    break
                if len(live) > seen:
                    seen = len(live)
                    # Decode the whole prefix, not the new ids: one token can be a partial
                    # UTF-8 sequence. Only the TAIL can be incomplete, so strip trailing
                    # replacement chars and hold them until the bytes arrive -- cutting at
                    # the FIRST one drops the whole reply whenever the text legitimately
                    # contains an unmappable byte, which is every prefix on the tiny model.
                    text = strip_think(tokenizer.decode(live).rstrip("�"))
                    if len(text) > sent:
                        yield _sse(
                            _chat_chunk(chunk_id, created, model_name, {"content": text[sent:]})
                        )
                        sent = len(text)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"request {request_id} did not finish within 1800.0s")
                time.sleep(0.02)
            output_ids = _await_completion(request_id)
        except (TimeoutError, RuntimeError) as exc:
            yield _sse({"error": {"message": str(exc), "type": "api_error"}})
            yield "data: [DONE]\n\n"
            return
        # reasoning is the model's, not the reply; sent counts stripped characters, so
        # this is the remainder of the same string the deltas were cut from
        tail = strip_think(tokenizer.decode(output_ids))[sent:]
        if tail:
            yield _sse(_chat_chunk(chunk_id, created, model_name, {"content": tail}))
        finish = "length" if len(output_ids) >= max_new else "stop"
        yield _sse(_chat_chunk(chunk_id, created, model_name, {}, finish=finish))
        # A final usage-only chunk, OpenAI's include_usage shape. Without it a client can
        # only guess the token count from characters, and chars/4 is ~4x low for Chinese
        # (roughly one token per character) -- a fabricated rate on the page's own meter.
        # Opt-in: it carries no choices, so a client that indexes choices[0] every frame
        # would raise on it.
        if include_usage:
            usage = _chat_chunk(chunk_id, created, model_name, {})
            usage["choices"] = []
            usage["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(output_ids),
                "total_tokens": prompt_tokens + len(output_ids),
            }
            yield _sse(usage)
        yield "data: [DONE]\n\n"

    # Anthropic Messages: what Claude Code speaks. Same engine, same tokenizer;
    # it records token ids per request, which the OpenAI route does not.
    from .messages import mount_messages

    mount_messages(app, engine, tokenizer, model_name)

    # The agent loop runs shell on the server, so it is OFF unless the operator
    # opts in with TILERL_AGENT_TOOLS=1 and pins the tool root there. The root
    # is never client-controlled (that would be unauthenticated RCE).
    _agent_root = os.environ.get("TILERL_AGENT_TOOLS")

    @app.post("/v1/agent")
    async def agent_run(req: AgentRequest):
        """Run the tool-calling agent loop and stream its events as SSE.

        Disabled unless TILERL_AGENT_TOOLS names the tool root; the tools run
        real shell/file ops, so enabling this exposes the server host."""
        from .agent import Tools, run_agent
        from .engine import SamplingParams

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
# Single-file chat UI: inline CSS/JS, system fonts, no network fetch beyond this origin
# (the server is often reached through a tunnel with no egress).
# ---------------------------------------------------------------------------

_CHAT_UI = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tilerl</title>
<style>
  :root {
    color-scheme: light;
    --paper: #f3f2ef; --surface: #fbfaf8; --sunken: #eae8e3;
    --line: #dcd9d3; --line-firm: #c5c1b8;
    --ink: #22211e; --ink-2: #57534c; --ink-3: #726d64;
    --accent: #4f746e; --accent-ink: #3d5b56; --accent-fg: #f8f8f6; --accent-wash: #e6ebe9;
    --danger: #91574f; --danger-wash: #f1e6e4;
    --shadow: 0 1px 2px rgba(34,33,30,.05), 0 10px 28px -18px rgba(34,33,30,.22);
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --r: 3px; --r2: 7px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --paper: #15161a; --surface: #1b1d21; --sunken: #212429;
      --line: #2b2e34; --line-firm: #3d4148;
      --ink: #e3e1dd; --ink-2: #a3a19b; --ink-3: #8d8c86;
      --accent: #8fb0aa; --accent-ink: #a8c5c0; --accent-fg: #14161a; --accent-wash: #1e2625;
      --danger: #c08d85; --danger-wash: #2a2020;
      --shadow: 0 1px 2px rgba(0,0,0,.32), 0 12px 32px -20px rgba(0,0,0,.7);
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    font: 15px/1.6 var(--sans); background: var(--paper); color: var(--ink);
    -webkit-font-smoothing: antialiased;
  }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }

  /* Header rail: identity left, mode switch middle, instrument cluster right. */
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 10px 20px; border-bottom: 1px solid var(--line); background: var(--surface);
  }
  .mark { font: 600 14px/1 var(--mono); letter-spacing: -.2px; color: var(--ink); text-decoration: none; }
  .mark:hover { color: var(--accent-ink); }
  .id { display: flex; align-items: center; gap: 7px; min-width: 0; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--line-firm); flex: none; }
  .dot.live { background: var(--accent); animation: pulse 1.6s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: .3; } }
  .model {
    font: 11.5px/1 var(--mono); color: var(--ink-3); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 30ch;
  }
  .tabs { display: flex; gap: 2px; padding: 2px; background: var(--sunken); border-radius: var(--r2); }
  .tab {
    padding: 5px 13px; border: 0; border-radius: 5px; cursor: pointer; background: transparent;
    font: 500 12.5px/1.3 var(--sans); color: var(--ink-3);
  }
  .tab:hover { color: var(--ink-2); }
  .tab.on {
    color: var(--ink); background: var(--surface); box-shadow: 0 1px 2px rgba(0,0,0,.07);
  }

  /* Instruments: the point of the demo, so tabular numerals and a real unit treatment. */
  .gauges {
    margin-left: auto; display: flex; border: 1px solid var(--line);
    border-radius: var(--r2); background: var(--paper); overflow: hidden;
  }
  .gauge { padding: 4px 14px 5px; min-width: 96px; }
  .gauge + .gauge { border-left: 1px solid var(--line); }
  .gk {
    display: block; font: 600 9.5px/1.4 var(--sans); text-transform: uppercase;
    letter-spacing: .1em; color: var(--ink-3);
  }
  .gv { display: flex; align-items: baseline; gap: 3px; }
  .gv b {
    font: 500 17px/1.2 var(--mono); font-variant-numeric: tabular-nums;
    color: var(--ink); letter-spacing: -.4px;
  }
  .gv i { font: 400 10px/1 var(--sans); font-style: normal; color: var(--ink-3); }

  /* Transcript: a document with a role gutter, not two colored bubbles. */
  main { flex: 1; overflow-y: auto; padding: 4px 20px 28px; }
  .col { max-width: 760px; margin: 0 auto; }
  .turn { display: grid; grid-template-columns: 62px minmax(0, 1fr); gap: 18px; padding: 20px 0; }
  .turn + .turn, .turn + .ev, .ev + .turn { border-top: 1px solid var(--line); }
  .who {
    font: 600 9.5px/2.2 var(--sans); text-transform: uppercase; letter-spacing: .1em;
    color: var(--ink-3); text-align: right;
  }
  .turn.user .who { color: var(--ink-2); }
  .content { white-space: pre-wrap; overflow-wrap: anywhere; }
  .turn.user .content {
    font-size: 14.5px; color: var(--ink-2);
    border-left: 2px solid var(--line-firm); padding-left: 13px;
  }
  .turn.assistant .content { font-size: 15px; line-height: 1.68; }
  .content.streaming::after {
    content: ""; display: inline-block; width: 2px; height: 1.05em; margin-left: 1px;
    background: var(--accent); vertical-align: text-bottom; animation: blink 1.1s steps(2) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }
  .note { font: 11.5px/1.6 var(--sans); color: var(--ink-3); padding: 4px 0 0 80px; }

  /* Reasoning: present but subordinate — no box, hairline rule, smaller dim type. */
  .think { margin: 0 0 14px; border-left: 2px solid var(--line); }
  .think > summary {
    display: flex; align-items: center; gap: 7px; padding: 1px 0 1px 13px; cursor: pointer;
    list-style: none; font: 600 9.5px/1.8 var(--sans); text-transform: uppercase;
    letter-spacing: .1em; color: var(--ink-3);
  }
  .think > summary::-webkit-details-marker { display: none; }
  .think > summary:hover { color: var(--ink-2); }
  .chev { font-size: 7px; line-height: 1; transition: transform .18s ease; }
  .think[open] .chev { transform: rotate(90deg); }
  .n { font: 400 9.5px/1 var(--mono); letter-spacing: 0; text-transform: none; color: var(--ink-3); }
  .thinkbody {
    padding: 7px 0 3px 13px; max-height: 300px; overflow: auto;
    font-size: 13.5px; line-height: 1.62; color: var(--ink-2);
    white-space: pre-wrap; overflow-wrap: anywhere;
  }

  /* Agent stream: the five kinds differ by weight and indent, not by hue. */
  .ev { margin: 0 0 10px 80px; }
  .ev:first-child { margin-top: 20px; }
  .ev .head {
    display: flex; align-items: center; gap: 7px;
    font: 600 9.5px/1.8 var(--sans); text-transform: uppercase; letter-spacing: .1em;
    color: var(--ink-3);
  }
  .mk { font: 400 11px/1 var(--mono); color: var(--ink-3); }
  .ev .body { white-space: pre-wrap; overflow-wrap: anywhere; }

  .ev.thought { border-left: 2px solid var(--line); padding-left: 13px; }
  .ev.thought .head { padding-bottom: 2px; }
  .ev.thought .body { font-size: 13.5px; line-height: 1.62; color: var(--ink-2); }

  .ev.action, .ev.observation {
    margin-left: 100px; border: 1px solid var(--line); border-radius: var(--r);
    background: var(--surface); overflow: hidden;
  }
  .ev.action .head, .ev.observation .head {
    padding: 5px 11px; background: var(--sunken); border-bottom: 1px solid var(--line);
  }
  .ev.action .head span:last-child { font-family: var(--mono); text-transform: none; letter-spacing: 0; font-size: 11px; color: var(--ink-2); }
  .ev.action .body, .ev.observation .body {
    padding: 9px 11px; font: 12.5px/1.6 var(--mono); color: var(--ink-2);
  }
  /* A call and its return read as one unit. */
  .ev.action + .ev.observation {
    margin-top: -10px; border-top: 0;
    border-top-left-radius: 0; border-top-right-radius: 0;
  }
  .ev.observation .body { background: var(--paper); max-height: 240px; overflow: auto; font-size: 12px; }

  .ev.answer, .ev.error {
    margin-left: 80px; padding: 13px 16px; border: 1px solid var(--line);
    border-left: 3px solid var(--accent); border-radius: var(--r);
    background: var(--surface); box-shadow: var(--shadow);
  }
  .ev.answer .body { font-size: 15px; line-height: 1.68; color: var(--ink); padding-top: 3px; }
  .ev.error { border-left-color: var(--danger); background: var(--danger-wash); box-shadow: none; }
  .ev.error .head { color: var(--danger); }
  .ev.error .mk { color: var(--danger); }
  .ev.error .body { font: 12.5px/1.6 var(--mono); color: var(--ink-2); padding-top: 3px; }

  /* Empty state */
  .empty { padding: 40px 0 0; max-width: 560px; }
  .empty h1 { margin: 0 0 7px; font: 600 19px/1.3 var(--sans); letter-spacing: -.2px; }
  .empty p { margin: 0; font-size: 14px; color: var(--ink-2); }
  .seeds { display: flex; flex-wrap: wrap; gap: 7px; margin: 22px 0 0; padding: 0; list-style: none; }
  .seed {
    padding: 6px 12px; border: 1px solid var(--line); border-radius: 14px; cursor: pointer;
    background: var(--surface); font: 12.5px/1.4 var(--sans); color: var(--ink-2); text-align: left;
  }
  .seed:hover { border-color: var(--line-firm); color: var(--ink); }
  .legend {
    margin: 30px 0 0; padding-top: 18px; border-top: 1px solid var(--line);
    display: grid; grid-template-columns: 62px 1fr; gap: 6px 18px;
    font-size: 12.5px; color: var(--ink-2);
  }
  .legend dt {
    font: 600 9.5px/1.9 var(--sans); text-transform: uppercase; letter-spacing: .1em;
    color: var(--ink-3); text-align: right;
  }
  .legend dd { margin: 0; }

  /* Composer */
  form { padding: 0 20px 18px; }
  .dock {
    max-width: 760px; margin: 0 auto; background: var(--surface);
    border: 1px solid var(--line); border-radius: var(--r2); box-shadow: var(--shadow);
  }
  .dock:focus-within { border-color: var(--line-firm); }
  textarea {
    display: block; width: 100%; resize: none; border: 0; background: transparent;
    padding: 12px 14px 2px; color: var(--ink); font: 15px/1.6 var(--sans); max-height: 200px;
  }
  textarea::placeholder { color: var(--ink-3); }
  textarea:focus { outline: none; }
  .dockbar { display: flex; align-items: center; gap: 10px; padding: 4px 10px 9px 14px; }
  .hint { font: 11.5px/1.5 var(--sans); color: var(--ink-3); }
  kbd {
    font: 10px/1.5 var(--mono); border: 1px solid var(--line); border-bottom-width: 2px;
    border-radius: var(--r); padding: 0 4px; color: var(--ink-2); background: var(--sunken);
  }
  .btn {
    margin-left: auto; padding: 6px 16px; border: 1px solid transparent; border-radius: var(--r);
    font: 500 13px/1.5 var(--sans); cursor: pointer;
    background: var(--accent); color: var(--accent-fg);
  }
  .btn:hover { background: var(--accent-ink); }
  .btn:disabled { opacity: .42; cursor: default; background: var(--accent); }
  .btn.ghost {
    margin-left: 0; background: transparent; color: var(--ink-2); border-color: var(--line-firm);
  }
  .btn.ghost:hover { background: var(--sunken); color: var(--ink); }

  @media (max-width: 640px) {
    header { padding: 10px 14px; gap: 10px; }
    .gauges { margin-left: 0; width: 100%; }
    .gauge { flex: 1; min-width: 0; }
    main { padding: 4px 14px 24px; }
    form { padding: 0 14px 14px; }
    .turn { grid-template-columns: 1fr; gap: 5px; }
    .who { text-align: left; }
    .ev, .ev.answer, .ev.error { margin-left: 0; }
    .ev.action, .ev.observation { margin-left: 14px; }
    .note { padding-left: 0; }
    .legend { grid-template-columns: 1fr; gap: 2px; }
    .legend dt { text-align: left; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::after { animation: none !important; transition: none !important; }
  }
</style>
</head>
<body>
<header>
  <div class="id">
    <a class="mark" href="/">tilerl</a>
    <span class="dot" id="dot"></span>
    <span class="model" id="model">connecting…</span>
  </div>
  <div class="tabs" role="tablist">
    <button class="tab on" id="tab-chat" type="button" role="tab" aria-selected="true" onclick="setMode('chat')">Chat</button>
    <button class="tab" id="tab-agent" type="button" role="tab" aria-selected="false" onclick="setMode('agent')">Agent</button>
  </div>
  <div class="gauges">
    <div class="gauge">
      <span class="gk">TTFT</span>
      <span class="gv"><b id="ttft">–</b><i>ms</i></span>
    </div>
    <div class="gauge">
      <span class="gk">Throughput</span>
      <span class="gv"><b id="tps">–</b><i>tok/s</i></span>
    </div>
  </div>
</header>
<main id="scroll">
  <div class="col" id="feed">
    <div class="empty" id="empty">
      <h1>Playground</h1>
      <p>Streamed straight from the local engine. Time to first token and throughput
         are measured in the browser on every response.</p>
      <ul class="seeds">
        <li><button class="seed" type="button" data-seed="Explain paged KV cache to someone who has written a matmul kernel but never served a model.">Explain paged KV cache</button></li>
        <li><button class="seed" type="button" data-seed="Write a Python function that merges overlapping intervals, then walk through its edge cases.">Write and review a function</button></li>
        <li><button class="seed" type="button" data-seed="What limits decode throughput on a single GPU — memory bandwidth or compute? Reason it out.">Reason about a bottleneck</button></li>
      </ul>
      <dl class="legend">
        <dt>Chat</dt><dd>One streaming turn at a time, with history threaded back each send.</dd>
        <dt>Agent</dt><dd>A tool-calling loop; each thought, call, and return arrives as its own event.</dd>
      </dl>
    </div>
  </div>
</main>
<form id="composer">
  <div class="dock">
    <textarea id="input" rows="1" placeholder="Message tilerl…" autofocus></textarea>
    <div class="dockbar">
      <span class="hint"><kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a newline</span>
      <button class="btn ghost" id="stop" type="button" hidden>Stop</button>
      <button class="btn" id="send" type="submit">Send</button>
    </div>
  </div>
</form>
<script>
const $ = (id) => document.getElementById(id);
let busy = false, mode = "chat", history = [], abort = null;

const PLACEHOLDER = {
  chat: "Message tilerl…",
  agent: "Give the agent a task — it can run shell, read and write files…",
};

function setMode(m) {
  mode = m;
  for (const k of ["chat", "agent"]) {
    const t = $("tab-" + k);
    t.classList.toggle("on", m === k);
    t.setAttribute("aria-selected", String(m === k));
  }
  $("input").placeholder = PLACEHOLDER[m];
}

// The empty state is real content, not a spacer: drop it the moment a turn lands.
function clearEmpty() {
  const e = $("empty");
  if (e) e.remove();
}

function addMsg(role, text) {
  clearEmpty();
  const turn = document.createElement("div");
  turn.className = "turn " + role;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "tilerl";
  const content = document.createElement("div");
  content.className = "content";
  content.textContent = text;
  turn.appendChild(who); turn.appendChild(content);
  $("feed").appendChild(turn);
  scrollDown();
  return content;
}

function addNote(text) {
  const n = document.createElement("div");
  n.className = "note";
  n.textContent = text;
  $("feed").appendChild(n);
  scrollDown();
  return n;
}

// Reasoning sits above the answer inside the same turn, recessive and collapsible.
function addThinking(bubble) {
  const det = document.createElement("details");
  det.className = "think";
  det.open = true;
  const sum = document.createElement("summary");
  const chev = document.createElement("span");
  chev.className = "chev"; chev.textContent = "▶";
  const label = document.createElement("span");
  label.textContent = "Reasoning";
  const n = document.createElement("span");
  n.className = "n";
  sum.appendChild(chev); sum.appendChild(label); sum.appendChild(n);
  const body = document.createElement("div");
  body.className = "thinkbody";
  det.appendChild(sum); det.appendChild(body);
  bubble.parentNode.insertBefore(det, bubble);
  scrollDown();
  return body;
}

const MARKS = { thought: "~", action: "→", observation: "←", answer: "◆", error: "!" };

function addEvent(kind, title) {
  clearEmpty();
  const box = document.createElement("div");
  box.className = "ev " + kind;
  const head = document.createElement("div");
  head.className = "head";
  const mk = document.createElement("span");
  mk.className = "mk"; mk.textContent = MARKS[kind] || "·";
  const label = document.createElement("span");
  label.textContent = title;
  head.appendChild(mk); head.appendChild(label);
  const body = document.createElement("div");
  body.className = "body";
  box.appendChild(head); box.appendChild(body);
  $("feed").appendChild(box);
  scrollDown();
  return body;
}

const scrollDown = () => { const m = $("scroll"); m.scrollTop = m.scrollHeight; };

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
  const bubble = addMsg("assistant", "");
  bubble.classList.add("streaming");
  const t0 = performance.now(); let firstAt = 0;
  history.push({ role: "user", content: text });
  abort = new AbortController();
  let raw = "", think = null, collapsed = false;
  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal: abort.signal,
      body: JSON.stringify({ messages: history, stream: true, stream_options: { include_usage: true } }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    await readSSE(resp, (obj) => {
      // The usage-only chunk carries no choices, so read it before indexing into them.
      if (obj.usage) {
        const secs = (performance.now() - (firstAt || t0)) / 1000;
        if (secs > 0) $("tps").textContent = (obj.usage.completion_tokens / secs).toFixed(1);
        return;
      }
      const delta = obj.choices?.[0]?.delta?.content;
      if (!delta) return;
      if (!firstAt) { firstAt = performance.now(); $("ttft").textContent = Math.round(firstAt - t0); }
      raw += delta;
      // Split the model's reasoning out of the answer. The tags can land mid-delta, so
      // re-partition the whole accumulated text each frame rather than tracking a state
      // machine across chunk boundaries.
      const open = raw.indexOf("<think>");
      if (open < 0) { bubble.textContent = raw; scrollDown(); return; }
      const close = raw.indexOf("</think>", open);
      if (!think) think = addThinking(bubble);
      think.textContent = raw.slice(open + 7, close < 0 ? undefined : close).trim();
      think.parentNode.querySelector(".n").textContent = think.textContent.length + " chars";
      bubble.textContent = (raw.slice(0, open) + (close < 0 ? "" : raw.slice(close + 8))).trim();
      // Fold the reasoning away once the answer proper starts, but only once, so a
      // reader who opened it back up keeps it open.
      if (close >= 0 && !collapsed) { collapsed = true; think.parentNode.open = false; }
      scrollDown();
    });
  } finally {
    bubble.classList.remove("streaming");
    // Keep partial text from a stopped stream, but never thread an empty turn back.
    if (raw) history.push({ role: "assistant", content: raw });
  }
  if (!bubble.textContent && !think) bubble.textContent = "(empty response)";
}

async function sendAgent(text) {
  const t0 = performance.now();
  abort = new AbortController();
  const resp = await fetch("/v1/agent", {
    method: "POST", headers: { "Content-Type": "application/json" },
    signal: abort.signal,
    body: JSON.stringify({ message: text }),
  });
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  let firstAt = 0;
  await readSSE(resp, (obj) => {
    if (!firstAt) { firstAt = performance.now(); $("ttft").textContent = Math.round(firstAt - t0); }
    const { type, payload } = obj;
    if (type === "thought") addEvent("thought", "thinking").textContent = payload;
    else if (type === "action") addEvent("action", payload.tool).textContent = JSON.stringify(payload.args, null, 2);
    else if (type === "observation") addEvent("observation", "observation").textContent = payload;
    else if (type === "answer") addEvent("answer", "answer").textContent = payload;
    else if (type === "error") addEvent("error", "error").textContent = payload;
  });
}

function setBusy(on) {
  busy = on;
  $("send").disabled = on;
  $("stop").hidden = !on;
  $("dot").classList.toggle("live", on);
}

async function send() {
  const text = $("input").value.trim();
  if (!text || busy) return;
  setBusy(true);
  addMsg("user", text);
  $("input").value = "";
  autosize();
  try {
    if (mode === "agent") await sendAgent(text); else await sendChat(text);
  } catch (err) {
    // Stopping is a deliberate act, not a failure worth an error card.
    if (err && err.name === "AbortError") addNote("Stopped.");
    else addEvent("error", "error").textContent = String(err);
  }
  abort = null;
  setBusy(false);
  $("input").focus();
}

function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 200) + "px";
}

$("composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("stop").addEventListener("click", () => { if (abort) abort.abort(); });
$("input").addEventListener("input", autosize);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
for (const b of document.querySelectorAll(".seed")) {
  b.addEventListener("click", () => {
    $("input").value = b.dataset.seed;
    autosize();
    $("input").focus();
  });
}
autosize();
fetch("/v1/models").then((r) => r.json()).then((j) => {
  $("model").textContent = j.data[0].id;
}).catch(() => { $("model").textContent = "offline"; });
</script>
</body>
</html>
"""
