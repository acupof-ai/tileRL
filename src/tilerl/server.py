"""HTTP facade: OpenAI-compatible API plus a single-file chat UI.

Route surface (mirrors agent-infer's infer-server, trimmed to tileRL):

* ``GET  /health``                 — liveness + engine stats
* ``GET  /v1/models``              — served model identity
* ``POST /v1/chat/completions``    — OpenAI schema; ``stream=true`` -> SSE
* ``POST /v1/messages``            — Anthropic Messages (messages.py)
* ``POST /v1/responses``           — OpenAI Responses (responses.py)
* ``WS   /ws/chat``                — the playground's transport, delta/done/error
* ``GET  /``, ``GET /chat``        — the chat UI, built from ``web/`` into ``static/``
* ``GET  /about``                  — what tileRL is, target matrix

This module never imports torch or tilelang: prompts cross the boundary as
``list[int]`` and the engine owns all tensor traffic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .messages import _COMPLETION_TIMEOUT_S, _parse_tool_calls, mount_messages
from .prompt import (
    POLL_INTERVAL_S,
    await_completion,
    bad_effort,
    choice_name,
    cut_at_stop,
    flatten_tools,
    hosted_tool_fields,
    refuse_unsupported,
    render_prompt,
    sampling,
    split_think,
    stop_texts,
    think_cap,
    thinking_enabled,
    tools_for_render,
    unknown_fields,
)
from .responses import mount_responses
from .tokenizer import ByteTokenizer, Tokenizer, get_tokenizer  # noqa: F401
from .ui_assets import _LANDING

__all__ = ["ByteTokenizer", "get_tokenizer", "create_app"]

SYSTEM_FINGERPRINT = "tilerl_fp_1"


# ---------------------------------------------------------------------------
# Wire types (OpenAI chat completions subset).
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    #: OpenAI replays a prior assistant call in the next request as
    #: ``tool_calls`` on the assistant message (nested
    #: ``{function:{name,arguments}}``); the matching result arrives as a
    #: later ``role:"tool"`` message carrying ``tool_call_id``.
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


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
    #: A bare string or a list; both are documented. Honoured by the engine, so
    #: `finish_reason` is "stop" and the sequence is cut from the returned text.
    stop: str | list[str] | None = None

    # One assignment: a second would replace this config, not merge into it.
    model_config = ConfigDict(populate_by_name=True, extra="allow")


def _ws_body(ask: dict) -> dict:
    return _normalize_thinking(dict(ask))


def _normalize_thinking(body: dict) -> dict:
    """A top-level enable_thinking moves into chat_template_kwargs so the one
    renderer sees it. OpenAI/sglang clients send it top-level; without the move
    pydantic (extra=allow) swallows it on the HTTP route and thinking stays on."""
    if "enable_thinking" in body:
        body["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}),
                                        "enable_thinking": body.pop("enable_thinking")}
    return body



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


class ClientDisconnected(Exception):
    """A real http.disconnect arrived while the completion worker still runs."""


def overloaded_body(exc: Exception) -> dict[str, Any] | None:
    """The 503 error body for an EngineOverloaded submit refusal, or None.

    Capacity is the server's responsibility, not the client's: no 429, no
    retry_after. The exception message already carries the cap and the
    in-flight count (running + waiting); the body exposes the same fact in
    machine-readable fields so a caller can choose whether to queue.
    """
    from .engine import EngineOverloaded

    if not isinstance(exc, EngineOverloaded):
        return None
    import re

    m = re.search(r"(\d+) in-flight requests and the cap is (\d+)", str(exc))
    inflight, cap = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return {"message": str(exc), "type": "overloaded_error",
            "inflight": inflight, "cap": cap}


_DISCONNECT_POLL_S = 0.05


async def await_or_cancel(request: Request, engine: Any, rid_box: list,
                          run_fn: Any, *args: Any) -> Any:
    """Poll a blocking completion fn in a thread and watch the ASGI disconnect.

    uvicorn delivers a client hang-up as http.disconnect WITHOUT cancelling the
    request task, so a bare ``await asyncio.to_thread(poll)`` never sees it (the
    httpx path that cancels the task still reaches the CancelledError branch).
    The worker is created once; ``is_disconnected()`` is awaited FRESH each tick
    — Starlette's one-shot peek returns False while connected and never flips, so
    a reused task detects nothing (and busy-waits). The wait timeout is the poll
    interval, not just a timeout. On disconnect the row is cancelled (that frees
    the slot) and ClientDisconnected is raised for a 499; the orphaned worker is
    not awaited.
    """
    worker = asyncio.ensure_future(asyncio.to_thread(run_fn, *args))
    # Attach unconditionally before the worker can finish: the off-loop cancel
    # opens a window where the worker dies with RequestFailed while the
    # disconnect branch is still inside to_thread(cancel), and the route then
    # raises without ever calling worker.result(). A callback attached only to a
    # still-pending worker would skip that case -> "Task exception was never
    # retrieved". On an already-done future the callback is scheduled now.
    worker.add_done_callback(lambda t: t.exception())
    while True:
        done, _ = await asyncio.wait({worker}, timeout=_DISCONNECT_POLL_S)
        if worker in done:
            return worker.result()
        if await request.is_disconnected():
            # Off the loop: cancel takes engine._lock across _release; a
            # slow step tick holding it must not freeze the event loop.
            await asyncio.to_thread(engine.cancel, rid_box[0])
            raise ClientDisconnected()


_STREAM_END = object()


async def stream_or_cancel(request: Request, engine: Any, request_id: int,
                            body: Any):
    """The SSE body under the same disconnect watch as the non-stream routes.

    Iterating the sync generator through ``to_thread(next, ...)`` leaves an
    in-flight worker anyio cannot interrupt: a sync call already running in the
    worker thread is not cancelled, and iterate_in_threadpool never acloses the
    sync generator, so a socket hang-up used to leave the row generating -- the
    _stream GeneratorExit branch only fires at GC/teardown. Each chunk is fetched
    one at a time; a fresh is_disconnected() poll runs while the fetch blocks,
    and disconnect cancels the row and stops the body. A normal completion wins
    the race: it returns before a disconnect tick can cancel the finished row,
    and cancel() on a finished id is a no-op."""
    worker = asyncio.ensure_future(asyncio.to_thread(next, body, _STREAM_END))
    worker.add_done_callback(lambda t: t.exception())
    while True:
        try:
            done, _ = await asyncio.wait({worker}, timeout=_DISCONNECT_POLL_S)
        except asyncio.CancelledError:
            # Response teardown while a fetch is in flight: same outcome as a
            # client hang-up. Off the loop: cancel takes engine._lock, which a
            # slow step tick can hold for seconds; a synchronous call would
            # block the event loop and freeze /health for every connection.
            await asyncio.to_thread(engine.cancel, request_id)
            raise
        if worker in done:
            item = worker.result()
            if item is _STREAM_END:
                return
            yield item
            worker = asyncio.ensure_future(
                asyncio.to_thread(next, body, _STREAM_END))
            worker.add_done_callback(lambda t: t.exception())
        elif await request.is_disconnected():
            await asyncio.to_thread(engine.cancel, request_id)
            return


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
        effort = (req.reasoning_effort or "").lower()
        cap = think_cap(effort or None)
        kw = req.chat_template_kwargs or {}
        thinking = kw.get("enable_thinking")
        if thinking is None:
            # The checkpoint's template treats an undefined enable_thinking as TRUE and
            # always emits <think> one way or the other, so leaving it unset made the model
            # open the tag in its own output. cap == 0 (reasoning_effort none) switches
            # thinking off; otherwise default to the template's answer, but only for a
            # tokenizer that HAS the tag - ByteTokenizer spells it as 7 raw bytes and its
            # bare turn is the tiny/dev path the None state exists for.
            thinking = thinking_enabled(tokenizer, None, cap == 0)
        # We render tools into the prompt and cannot force or forbid a call, so a
        # tool_choice stronger than a hint is refused rather than echoed.
        unknown_fields(req)  # warns; this route has no recorder, so the warn is all there is
        named = choice_name(req.tool_choice)
        refuse_unsupported(reasoning_effort=bad_effort(effort or None),
                          tool_choice=named not in ("auto", "none", None),
                          **hosted_tool_fields(req.tools))
        tools = tools_for_render(flatten_tools(req.tools), req.tool_choice)
        input_ids = tokenizer.encode(_render_chat(
            req.messages, thinking, kw.get("reasoning_effort") or req.reasoning_effort, tools
        ))
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        # Omitted max_tokens means "as much as fits", not 512: a 512 cap ends a long
        # reply at finish_reason=length, which reads to a client as a dropped stream.
        # `room_for` is the engine's own admission arithmetic, so the default is always
        # accepted and a prompt that does not fit still hits submit's refusal.
        max_new = (req.max_tokens if req.max_tokens is not None
                   else engine.room_for(len(input_ids)))
        params = sampling(tokenizer, thinking, max_new,
                          temperature=req.temperature, top_p=req.top_p, max_think_tokens=cap,
                          seed=req.seed, logprobs=bool(req.logprobs), stop=req.stop)
        # bool(thinking): True when the prompt opened <think>, so the reply carries only
        # the closer and strip_think must be told (None = bare turn, nothing to strip)
        return (engine.submit(input_ids, params), len(input_ids), params.max_new_tokens,
                bool(thinking), tools)

    def _await_completion(request_id: int,
                          timeout_s: float = _COMPLETION_TIMEOUT_S) -> list[int]:
        return await_completion(engine, request_id, timeout_s)

    @app.get("/health")
    def health() -> dict:
        # "ok" was a literal, so an engine whose stats() raises answered the same as a
        # healthy one. Loop liveness is deliberately NOT checked via engine._thread:
        # that wants a liveness method on the engine, not a private attribute read.
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
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        req = ChatCompletionRequest.model_validate(_normalize_thinking(req.model_dump()))
        request_id = -1
        try:
            # to_thread: engine.submit takes step()'s lock; on the loop a request arriving
            # during a long prefill freezes every route, /health included.
            request_id, prompt_tokens, max_new, opened, tools = await asyncio.to_thread(_submit, req)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content={"error": overloaded_body(exc) or {
                    "message": str(exc), "type": "api_error"}},
            )

        if req.stream:
            return StreamingResponse(
                stream_or_cancel(request, engine, request_id,
                                 _stream(request_id, max_new, prompt_tokens, opened, bool(
                                     (req.stream_options or {}).get("include_usage")
                                 ), stop_texts(req.stop), tools,
                                 choice_name(req.tool_choice) != "none")),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        rid_box = [request_id]
        try:
            output_ids = await await_or_cancel(
                request, engine, rid_box, _await_completion, request_id)
        except asyncio.CancelledError:
            # Client hung up before the non-stream reply; stop generating for
            # nobody. Off the loop (cancel takes engine._lock across _release).
            await asyncio.to_thread(engine.cancel, request_id)
            raise
        except ClientDisconnected:
            return Response(status_code=499)
        except TimeoutError as exc:
            # The server gave up waiting but the row is still generating: cancel
            # frees the slot; cancel() on a row the engine already failed (the
            # RuntimeError below) is a False-returning no-op.
            await asyncio.to_thread(engine.cancel, request_id)
            return JSONResponse(
                status_code=504,
                content={"error": {"message": str(exc), "type": "api_error"}},
            )
        except RuntimeError as exc:
            await asyncio.to_thread(engine.cancel, request_id)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "api_error"}},
            )
        # Same split as _stream below, so flipping `stream` does not change which
        # fields a reply has: #159 wired reasoning_content into the streaming path
        # only, and a client that switched got the reasoning on one path and lost
        # it on the other.
        reasoning, text = split_think(tokenizer.decode(output_ids), opened)
        # The engine keeps the token that completed the match, so `text` still
        # carries the sequence and all three APIs exclude it.
        stopped = await asyncio.to_thread(engine.stop_text, request_id)
        text = cut_at_stop(text, stopped)
        # The template answers a tool request in <tool_call> XML; parse it with the
        # SAME function /v1/messages uses, so one call cannot mean two things
        # depending on which API asked. The prose before the first call is the
        # model's own explanation and stays as content.
        text, calls = _parse_tool_calls(text, tools)
        # choice:"none" must also discard a call the model emits anyway: the
        # tools block was hidden, but the parser still matches free-form XML.
        if choice_name(req.tool_choice) == "none":
            calls = []
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
        scores = (await asyncio.to_thread(engine.logprobs, request_id)) if req.logprobs else None
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
                    # A stop sequence is OpenAI's "stop" too, and it takes precedence
                    # over length: the cap was not what ended this one.
                    "finish_reason": ("tool_calls" if tool_calls else "stop" if stopped
                                      else "length" if len(output_ids) >= max_new else "stop"),
                }
            ],
            "usage": _usage(prompt_tokens, len(output_ids)),
            "system_fingerprint": SYSTEM_FINGERPRINT,
        }

    def _deltas(request_id: int, max_new: int, opened: bool, stops: tuple[str, ...] = (),
                tools: list | None = None, allow_tool_calls: bool = True):
        """One request's reply, as ``(kind, payload, completion_tokens)`` triples.

        ``kind`` is ``delta`` (payload is a ``reasoning_content``/``content`` dict),
        ``tool_calls`` (payload is the non-stream route's (name, args) pairs),
        ``error`` (an OpenAI error body) or ``done`` (payload is the finish_reason,
        and it is the last item). Shared by the SSE route and ``/ws/chat``: the two
        transports differ only in how they frame these, so they cannot disagree about
        where a ``</think>`` goes, where a stop sequence cuts, or when a reply is
        ``length``.

        Blocking, by design -- it is driven from a thread on both routes.
        """
        # The checkpoint opens a tool call with this literal; once any suffix of it
        # reaches the wire it can never be un-sent, so a forming call is held back.
        call_open = "<tool_call>"

        deadline = time.monotonic() + _COMPLETION_TIMEOUT_S
        sent = 0  # safe content characters already emitted (absolute index)
        sent_r = 0  # characters of the reasoning already emitted
        seen = 0  # tokens already decoded, so a quiet poll costs nothing
        # the most of a stop sequence that can still turn out to be a prefix
        stop_hold = max((len(x) for x in stops), default=1) - 1
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
                        yield "delta", {"reasoning_content": reasoning[sent_r:]}, seen
                        sent_r = len(reasoning)
                    # A COMPLETE stop match ends the text; otherwise hold the
                    # stop-prefix chars. With calls allowed, also hold a complete
                    # <tool_call> opener and any forming opener suffix, so the XML
                    # leaks in neither chunks nor a single frame at a time.
                    safe = text
                    if stops:
                        done = [safe.index(x) for x in stops if x in safe]
                        safe = (safe[:min(done)] if done
                                else safe[:max(0, len(safe) - stop_hold)])
                    # The opener hold runs UNCONDITIONALLY: even when
                    # tool_choice:"none" discards the structured calls
                    # (allow_tool_calls False), the raw XML bytes must not
                    # reach a content delta. allow_tool_calls only decides
                    # whether the held call is emitted as a tool_calls frame.
                    cut = safe.find(call_open)
                    if cut >= 0:
                        safe = safe[:cut]
                    else:
                        for n in range(1, len(call_open)):
                            if safe.endswith(call_open[:n]):
                                safe = safe[:-n]
                                break
                    # Hold trailing whitespace unconditionally too: a call
                    # follows its prose after "\n", and the parser strips that
                    # prose, so the separator cannot reach a content delta on
                    # the choice:none path either.
                    upto = len(safe.rstrip())
                    if upto > sent:
                        yield "delta", {"content": safe[sent:upto]}, seen
                        sent = upto
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"request {request_id} did not finish within {_COMPLETION_TIMEOUT_S}s")
                time.sleep(POLL_INTERVAL_S)
            output_ids = _await_completion(request_id)
        except (TimeoutError, RuntimeError) as exc:
            # Sync generator: already in a to_thread worker, off the loop.
            engine.cancel(request_id)
            yield "error", {"message": str(exc), "type": "api_error"}, seen
            return
        except Exception as exc:
            # The 200 header left before this generator ran, so an escaping exception
            # reaches the client as 200 with zero frames and no [DONE] -- a success
            # status over a failed request, which reads as an empty reply. Measured:
            # an engine without `peek` returned exactly that
            # (errors/2026-09-05-the-sse-handler-covered-two-of-six). Logged as well as
            # framed, because a tidy error frame is easier to ignore than silence and
            # this branch means a defect, not a busy engine.
            logging.exception("stream for request %s died", request_id)
            yield "error", {"message": f"{type(exc).__name__}: {exc}",
                            "type": "internal_error"}, seen
            return
        # sent counts stripped characters, so these are the remainders of the same
        # strings the deltas were cut from
        reasoning, text = split_think(tokenizer.decode(output_ids), opened)
        stopped = engine.stop_text(request_id)
        text = cut_at_stop(text, stopped)
        prose, calls = _parse_tool_calls(text, tools)
        if not allow_tool_calls:
            calls = []
        if len(reasoning) > sent_r:
            yield "delta", {"reasoning_content": reasoning[sent_r:]}, len(output_ids)
        # Tool-call XML never reached the content deltas (the opener was held).
        # _parse_tool_calls strips it for both paths; choice:"none" keeps the
        # parse but discards the calls, so the terminal tail is the prose only.
        tail = prose
        if calls:
            # The non-stream parser strips the prose around the call; the
            # separating newline rides with the XML, not the content delta.
            tail = tail.rstrip()
        if len(tail) > sent:
            yield "delta", {"content": tail[sent:]}, len(output_ids)
        if calls:
            yield "tool_calls", calls, len(output_ids)
        finish = ("tool_calls" if calls else "stop" if stopped
                   else "length" if len(output_ids) >= max_new else "stop")
        yield ("done", finish, len(output_ids))

    def _stream(request_id: int, max_new: int, prompt_tokens: int, opened: bool,
                include_usage: bool, stops: tuple[str, ...] = (),
                tools: list | None = None, allow_tool_calls: bool = True):
        created = int(time.time())
        chunk_id = f"chatcmpl-{request_id}"
        yield _sse(_chat_chunk(chunk_id, created, model_name, {"role": "assistant"}))
        completion = 0
        try:
            for kind, payload, completion in _deltas(request_id, max_new, opened, stops,
                                                      tools, allow_tool_calls):
                if kind == "error":
                    yield _sse({"error": payload})
                    yield "data: [DONE]\n\n"
                    return
                if kind == "tool_calls":
                    for i, (name, args) in enumerate(payload):
                        # One full-arguments delta per call: the SDK appends, and
                        # index/id/function are byte-identical to the non-stream
                        # message.tool_calls element.
                        yield _sse(_chat_chunk(chunk_id, created, model_name, {
                            "tool_calls": [{"index": i,
                                            "id": f"call_{request_id}_{i}",
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(args,
                                                                       ensure_ascii=False)}}]}))
                    continue
                if kind == "delta":
                    # Cumulative tokens on every content frame, vLLM's
                    # continuous_usage_stats shape. Without it a live rate gauge can only
                    # count frames, and this loop coalesces ~1.8 tokens into each
                    # (measured: 109 frames for 200 tokens on the 27B), so the page would
                    # show roughly half the real rate until the final usage chunk landed.
                    # choices stays populated, so a client that indexes it is unharmed;
                    # the usage-ONLY chunk remains the one with an empty choices list.
                    chunk = _chat_chunk(chunk_id, created, model_name, payload)
                    if include_usage:
                        chunk["usage"] = _usage(prompt_tokens, completion)
                    yield _sse(chunk)
                else:
                    yield _sse(_chat_chunk(chunk_id, created, model_name, {},
                                           finish=payload))
        except GeneratorExit:
            # Defense-in-depth only: the live disconnect path is stream_or_cancel
            # above. This fires when the abandoned sync generator is finalized at
            # GC/process teardown (anyio cannot interrupt the in-flight thread
            # call itself), and still must free the row then. Executes inside
            # the to_thread worker, never on the event loop: no to_thread here.
            engine.cancel(request_id)
            raise
        # A final usage-only chunk, OpenAI's include_usage shape. Without it a client can
        # only guess the token count from characters, and chars/4 is ~4x low for Chinese
        # (roughly one token per character) -- a fabricated rate on the page's own meter.
        # Opt-in: it carries no choices, so a client that indexes choices[0] every frame
        # would raise on it.
        if include_usage:
            usage = _chat_chunk(chunk_id, created, model_name, {})
            usage["choices"] = []
            usage["usage"] = _usage(prompt_tokens, completion)
            yield _sse(usage)
        yield "data: [DONE]\n\n"

    # Anthropic Messages: what Claude Code speaks. Same engine, same tokenizer;
    # it records token ids per request, which the OpenAI route does not.
    mount_messages(app, engine, tokenizer, model_name)

    # OpenAI Responses: the same engine again, differing only in wire shape --
    # a flat typed `output` list instead of `choices`.
    mount_responses(app, engine, tokenizer, model_name)

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket) -> None:
        """The playground's transport. Frames: delta / done / error, one request each.

        WebSocket rather than SSE because the page has to SEND a turn as well as read
        one, and EventSource is receive-only -- the old page posted the turn and opened
        a second connection for the reply. One socket per turn, so there are no request
        ids on the wire and a reload cannot leave a stream attached to the wrong bubble.
        """
        await ws.accept()
        try:
            ask = await ws.receive_json()
        except Exception:  # a client that closes before sending has nothing to answer
            return
        try:
            # A picked constructor hides every other field from extra="allow".
            req = ChatCompletionRequest.model_validate(_ws_body(ask))
            request_id, prompt_tokens, max_new, opened, tools = (
                await asyncio.to_thread(_submit, req))
        except Exception as exc:
            await ws.send_json({"t": "error", "message": f"{type(exc).__name__}: {exc}"})
            await ws.close()
            return

        # _deltas blocks on the engine; stepping it in a thread keeps the event loop free
        # to serve the other routes while one page streams.
        gen, end = _deltas(request_id, max_new, opened, stop_texts(req.stop),
                           tools, choice_name(req.tool_choice) != "none"), object()
        calls = None
        try:
            while (item := await asyncio.to_thread(next, gen, end)) is not end:
                kind, payload, completion = item
                if kind == "delta":
                    await ws.send_json({"t": "delta", **payload})
                elif kind == "tool_calls":
                    calls = [{"id": f"call_{request_id}_{i}", "type": "function",
                              "name": n, "arguments": json.dumps(a, ensure_ascii=False)}
                             for i, (n, a) in enumerate(payload)]
                    await ws.send_json({"t": "tool_calls", "tool_calls": calls})
                elif kind == "error":
                    await ws.send_json({"t": "error", "message": payload["message"]})
                    break
                else:
                    await ws.send_json({"t": "done", "finish_reason": payload,
                                        **({"tool_calls": calls} if payload == "tool_calls" else {}),
                                        "usage": _usage(prompt_tokens, completion)})
        except WebSocketDisconnect:
            # gen.close() stops this poll loop; the cancel is what stops the engine,
            # measured at 1891 tokens and 104 KV blocks after one socket closed.
            gen.close()
            await asyncio.to_thread(engine.cancel, request_id)
            return
        await ws.close()

    @app.get("/about", response_class=HTMLResponse)
    def about() -> str:
        return _LANDING

    _STATIC = Path(__file__).parent / "static"

    @app.get("/chat", include_in_schema=False)
    def chat() -> FileResponse:
        # StaticFiles(html=True) answers "/" with index.html but treats "/chat" as a
        # missing file, and /chat is the URL the landing page links to.
        return FileResponse(_STATIC / "index.html")

    # The root is the playground: whoever opens the host:port wants to type at the model,
    # not read what tileRL is; the landing page keeps its content at /about. The bundle is
    # built by `web/` and committed, so serving it needs no node here. Mounted LAST:
    # Starlette matches routes in registration order and a mount at "/" swallows every
    # path declared after it.
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="chat-assets")

    return app
