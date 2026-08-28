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

    model_config = {"populate_by_name": True}


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


def _render_chat(messages: list[ChatMessage]) -> str:
    # ChatML — the format Qwen3.x was trained on, and what the stop set
    # already assumes (_HfTokenizerAdapter.stop_token_ids). The old
    # role-prefixed plain text contradicted it: the model never sees the
    # markers it is stopped on. ponytail: a Jinja chat template belongs with
    # the real tokenizer/checkpoint (day-2, zero-code onboarding); Qwen's
    # template renders this same string.
    rendered = "".join(f"<|im_start|>{m.role}\n{_message_text(m)}<|im_end|>\n" for m in messages)
    return f"{rendered}<|im_start|>assistant\n"


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
        prompt = _render_chat(req.messages)
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
        )
        return engine.submit(input_ids, params), len(input_ids), params.max_new_tokens

    def _await_completion(request_id: int, timeout_s: float = 600.0) -> list[int]:
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
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
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

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _CHAT_UI

    return app


# ---------------------------------------------------------------------------
# Single-file chat UI (~100 lines, no build step).
# ---------------------------------------------------------------------------

_CHAT_UI = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tilerl chat</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; flex-direction: column;
    font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    background: #14161a; color: #e6e8eb;
  }
  header {
    padding: 10px 16px; border-bottom: 1px solid #2a2d33;
    display: flex; gap: 10px; align-items: baseline;
  }
  header b { color: #7aa2f7; }
  header span { color: #8b9099; font-size: 13px; }
  main { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 760px; width: fit-content; padding: 8px 14px; border-radius: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .user { align-self: flex-end; background: #1f3a5f; }
  .assistant { align-self: flex-start; background: #22252b; }
  form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #2a2d33; }
  textarea {
    flex: 1; resize: none; padding: 10px 12px; border-radius: 8px;
    border: 1px solid #2a2d33; background: #1a1d22; color: #e6e8eb; font: inherit;
  }
  textarea:focus { outline: none; border-color: #7aa2f7; }
  button {
    padding: 0 20px; border: 0; border-radius: 8px; background: #7aa2f7;
    color: #14161a; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<header><b>tilerl</b><span id="model">connecting…</span></header>
<main id="messages"></main>
<form id="composer">
  <textarea id="input" rows="2" placeholder="Message tilerl  (Enter to send, Shift+Enter for newline)" autofocus></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
const $ = (id) => document.getElementById(id);
let busy = false;

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  $("messages").appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

async function send() {
  const text = $("input").value.trim();
  if (!text || busy) return;
  busy = true;
  $("send").disabled = true;
  addMsg("user", text);
  $("input").value = "";
  const bubble = addMsg("assistant", "…");
  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: text }], stream: true }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    bubble.textContent = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\\n\\n")) >= 0) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const line = frame.split("\\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") continue;
        const delta = JSON.parse(payload).choices?.[0]?.delta?.content;
        if (delta) {
          bubble.textContent += delta;
          window.scrollTo(0, document.body.scrollHeight);
        }
      }
    }
    if (!bubble.textContent) bubble.textContent = "(empty response)";
  } catch (err) {
    bubble.textContent = "[error] " + err;
  }
  busy = false;
  $("send").disabled = false;
  $("input").focus();
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
