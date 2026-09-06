"""HTTP facade: OpenAI-compatible API plus a single-file chat UI.

Route surface (mirrors agent-infer's infer-server, trimmed to tileRL):

* ``GET  /health``                 — liveness + engine stats
* ``GET  /v1/models``              — served model identity
* ``POST /v1/chat/completions``    — OpenAI schema; ``stream=true`` -> SSE
* ``POST /v1/messages``            — Anthropic Messages (messages.py)
* ``POST /v1/responses``           — OpenAI Responses (responses.py)
* ``GET  /``                       — single-file HTML chat UI (no build step)
* ``GET  /about``                  — what tileRL is, target matrix

This module never imports torch or tilelang: prompts cross the boundary as
``list[int]`` and the engine owns all tensor traffic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .messages import _parse_tool_calls, mount_messages
from .prompt import refuse_unsupported, render_prompt, sampling, split_think
from .responses import mount_responses
from .tokenizer import ByteTokenizer, Tokenizer, get_tokenizer  # noqa: F401
from .ui_assets import _CHAT_UI, _LANDING

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
    #: OpenAI nests the schema under "function"; the template and messages.py
    #: both want Anthropic's flat {name, description, input_schema}.
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    #: Declared only so it can be REFUSED: an undeclared field is dropped by
    #: pydantic without a trace, so the client gets a 200 and no sign its stop
    #: never applied. Unimplemented, not unsupported -- see
    #: errors/2026-09-06-stop-sequences-accepted-and-never-applied.md.
    stop: str | list[str] | None = None

    model_config = {"populate_by_name": True}


#: reasoning_effort -> cap on <think> tokens; "none" switches thinking off in the prompt.
_MAX_THINK = {"none": 0, "minimal": 128, "low": 512, "medium": 2048, "high": 8192}


def _unsupported_choice(choice: Any) -> bool:
    """`tool_choice` beyond auto/none, or None when it is honourable as given."""
    if choice is None:
        return None
    name = choice if isinstance(choice, str) else (choice or {}).get("type")
    return name not in ("auto", "none", None)


def _flatten_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """OpenAI's ``{type, function: {name, description, parameters}}`` as the flat
    ``{name, description, input_schema}`` the template renders and
    ``messages._parse_tool_calls`` reads schemas from. One tool vocabulary for
    both routes, so a call parses identically whichever API asked for it."""
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function") or t
        out.append({"name": fn.get("name"), "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or fn.get("input_schema") or {}})
    return out


def _render_chat(messages: list[ChatMessage], thinking: bool | None = None,
                 reasoning_effort: str | None = None,
                 tools: list[dict[str, Any]] | None = None) -> str:
    return render_prompt([m.model_dump() for m in messages], tools=tools,
                         thinking=thinking, effort=reasoning_effort)


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


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


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

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        """A rejected field must speak the envelope of the API it was sent to.

        FastAPI's default is `{"detail": [...]}` with status 422, which no
        OpenAI or Anthropic client can read: both look under `error.message`,
        and the SDKs map 422 to UnprocessableEntityError rather than
        BadRequestError. Each route's hand-written 400s already emit the right
        shape -- pydantic runs before the handler body, so it bypassed them.
        One app serves both APIs, so the envelope is chosen by path.
        """
        msg = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'body'}: {e.get('msg', '')}"
            for e in exc.errors()) or "invalid request"
        if request.url.path.startswith("/v1/messages"):
            body: dict[str, Any] = {"type": "error",
                                    "error": {"type": "invalid_request_error", "message": msg}}
        else:
            body = {"error": {"message": msg, "type": "invalid_request_error",
                              "param": None, "code": None}}
        return JSONResponse(status_code=400, content=body)

    def _submit(req: ChatCompletionRequest) -> tuple[int, int, int, bool, list | None]:
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
        # We render tools into the prompt and cannot force or forbid a call, so a
        # tool_choice stronger than a hint is refused rather than echoed.
        refuse_unsupported(tool_choice=_unsupported_choice(req.tool_choice),
                           stop=req.stop)
        tools = _flatten_tools(req.tools)
        input_ids = tokenizer.encode(_render_chat(
            req.messages, thinking, kw.get("reasoning_effort") or req.reasoning_effort, tools
        ))
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        params = sampling(tokenizer, thinking, req.max_tokens if req.max_tokens is not None else 512,
                          temperature=req.temperature, top_p=req.top_p, max_think_tokens=cap,
                          seed=req.seed, logprobs=bool(req.logprobs))
        # bool(thinking): True when the prompt opened <think>, so the reply carries only
        # the closer and strip_think must be told (None = bare turn, nothing to strip)
        return (engine.submit(input_ids, params), len(input_ids), params.max_new_tokens,
                bool(thinking), tools)

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
        # "ok" was a literal, so an engine whose stats() raises answered the same as a
        # healthy one. Loop liveness is deliberately NOT checked: it would need
        # engine._thread, which DataParallelEngine lacks, so the check would pass
        # vacuously on --devices -- the shape of the peek gap. That wants a liveness
        # method on the engine.
        try:
            stats = engine.stats()
        except Exception as exc:
            return {"status": "degraded", "model": model_name, "stats": None,
                    "error": f"{type(exc).__name__}: {exc}"}
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
            request_id, prompt_tokens, max_new, opened, tools = _submit(req)
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
                _stream(request_id, max_new, prompt_tokens, opened, bool(
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
        # Same split as _stream below, so flipping `stream` does not change which
        # fields a reply has: #159 wired reasoning_content into the streaming path
        # only, and a client that switched got the reasoning on one path and lost
        # it on the other.
        reasoning, text = split_think(tokenizer.decode(output_ids), opened)
        # The template answers a tool request in <tool_call> XML; parse it with the
        # SAME function /v1/messages uses, so one call cannot mean two things
        # depending on which API asked. The prose before the first call is the
        # model's own explanation and stays as content.
        text, calls = _parse_tool_calls(text, tools)
        tool_calls = [
            {"id": f"call_{request_id}_{i}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
            for i, (n, a) in enumerate(calls)
        ] or None
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
                    "message": {"role": "assistant",
                                # null, not "", when a tool call carries no prose:
                                # OpenAI's shape, and "" reads as an empty reply.
                                "content": text or None if tool_calls else text,
                                # None, not "": the field is absent for a bare turn
                                # or thinking off, which is what a client checks.
                                "reasoning_content": reasoning or None,
                                "tool_calls": tool_calls},
                    "logprobs": None if content is None else {"content": content},
                    "finish_reason": ("tool_calls" if tool_calls else
                                      "length" if len(output_ids) >= max_new else "stop"),
                }
            ],
            "usage": _usage(prompt_tokens, len(output_ids)),
            "system_fingerprint": SYSTEM_FINGERPRINT,
        }

    def _stream(request_id: int, max_new: int, prompt_tokens: int, opened: bool,
                include_usage: bool):
        created = int(time.time())
        chunk_id = f"chatcmpl-{request_id}"
        yield _sse(_chat_chunk(chunk_id, created, model_name, {"role": "assistant"}))
        deadline = time.monotonic() + 1800.0
        sent = 0  # characters of the STRIPPED reply already emitted
        sent_r = 0  # characters of the reasoning already emitted
        seen = 0  # tokens already decoded, so a quiet poll costs nothing

        def content_frame(delta: dict, completion: int) -> str:
            # Cumulative tokens on every content frame, vLLM's continuous_usage_stats
            # shape. Without it a live rate gauge can only count frames, and this loop
            # coalesces ~1.8 tokens into each (measured: 109 frames for 200 tokens on the
            # 27B), so the page would show roughly half the real rate until the final
            # usage chunk landed. choices stays populated, so a client that indexes it is
            # unharmed; the usage-ONLY chunk remains the one with an empty choices list.
            chunk = _chat_chunk(chunk_id, created, model_name, delta)
            if include_usage:
                chunk["usage"] = _usage(prompt_tokens, completion)
            return _sse(chunk)

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
                    raw = tokenizer.decode(live).rstrip("�")
                    reasoning, text = split_think(raw, opened)
                    if "</think>" not in raw:
                        # the prefix may end in a partial closer; hold that much back
                        reasoning = reasoning[: -len("</think>")]
                    # reasoning goes out as vLLM's reasoning_content, so the page folds
                    # on the field rather than on a closer the reply no longer carries
                    if len(reasoning) > sent_r:
                        yield content_frame({"reasoning_content": reasoning[sent_r:]}, seen)
                        sent_r = len(reasoning)
                    if len(text) > sent:
                        yield content_frame({"content": text[sent:]}, seen)
                        sent = len(text)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"request {request_id} did not finish within 1800.0s")
                time.sleep(0.02)
            output_ids = _await_completion(request_id)
        except (TimeoutError, RuntimeError) as exc:
            yield _sse({"error": {"message": str(exc), "type": "api_error"}})
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:
            # The 200 header left before this generator ran, so an escaping exception
            # reaches the client as 200 with zero frames and no [DONE] -- a success
            # status over a failed request, which reads as an empty reply. Measured:
            # DataParallelEngine had no `peek`, and `serve --devices` returned exactly
            # that. Logged as well as framed, because a tidy error frame is easier to
            # ignore than silence and this branch means a defect, not a busy engine.
            logging.exception("stream for request %s died", request_id)
            yield _sse({"error": {"message": f"{type(exc).__name__}: {exc}",
                                  "type": "internal_error"}})
            yield "data: [DONE]\n\n"
            return
        # sent counts stripped characters, so these are the remainders of the same
        # strings the deltas were cut from
        reasoning, text = split_think(tokenizer.decode(output_ids), opened)
        if len(reasoning) > sent_r:
            yield content_frame({"reasoning_content": reasoning[sent_r:]}, len(output_ids))
        if len(text) > sent:
            yield content_frame({"content": text[sent:]}, len(output_ids))
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
            usage["usage"] = _usage(prompt_tokens, len(output_ids))
            yield _sse(usage)
        yield "data: [DONE]\n\n"

    # Anthropic Messages: what Claude Code speaks. Same engine, same tokenizer;
    # it records token ids per request, which the OpenAI route does not.
    mount_messages(app, engine, tokenizer, model_name)

    # OpenAI Responses: the same engine again, differing only in wire shape --
    # a flat typed `output` list instead of `choices`.
    mount_responses(app, engine, tokenizer, model_name)

    # The root is the playground: whoever opens the host:port wants to type at the
    # model, not read what tileRL is. The landing page keeps its content at /about.
    @app.get("/", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    def chat() -> str:
        return _CHAT_UI

    @app.get("/about", response_class=HTMLResponse)
    def about() -> str:
        return _LANDING

    return app
