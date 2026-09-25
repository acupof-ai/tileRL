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
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .messages import (
    _COMPLETION_TIMEOUT_S,
    _parse_tool_calls,
    completion_timeout_from_env,
    mount_messages,
)
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
        body["chat_template_kwargs"] = {
            **(body.get("chat_template_kwargs") or {}),
            "enable_thinking": body.pop("enable_thinking"),
        }
    return body


def _render_chat(
    messages: list[ChatMessage],
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    return render_prompt(
        [m.model_dump() for m in messages], tools=tools, thinking=thinking, effort=reasoning_effort
    )


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
    return {"message": str(exc), "type": "overloaded_error", "inflight": inflight, "cap": cap}


_DISCONNECT_POLL_S = 0.05


def _worker_retrieved(t: Any) -> None:
    """Retrieve a to_thread(next) worker's exception so it is not logged as
    "never retrieved". A worker cancelled while its SSE task is torn down
    (simultaneous hangups) is an expected cancellation, not an error: the
    detached drain owns the body's lifecycle, so swallow CancelledError here."""
    if t.cancelled():
        return
    t.exception()


#: Cap on detaching an SSE body's final drain. A disconnected SSE task must not
#: await its in-flight executor worker from INSIDE its own cancellation: under a
#: Starlette task-group teardown that cancellation is re-delivered on every
#: checkpoint, so a shield await re-raises forever and busy-spins the event loop
#: (the 2026-09-16 wedge). Instead the drain runs as a detached task, OUTSIDE the
#: cancelled scope; this bounds how long it waits for an already-running next()
#: before giving up (the worker thread itself is never killed).
_DRAIN_WAIT_S = 30.0

#: Detached SSE body-drain tasks. A strong module-level set (not a local the
#: generator frame drops on teardown): without a reachable reference the task can
#: be GC'd mid-drain, which is exactly what would orphan the sync body and defer
#: its close() to nondeterministic GC. Discarded when the drain completes.
_draining: set = set()


def _detach_drain(engine: Any, request_id: int, worker: Any, body: Any) -> None:
    """Finalize an SSE sync generator OFF its (possibly being-cancelled) task.

    Runs in a detached task that is not a child of the disconnecting SSE task's
    cancel scope — so repeated cancellation of that task cannot re-raise inside
    this wait and spin the loop. The body is closed on a worker thread (its
    GeneratorExit calls engine.cancel, which takes a lock; doing that on the
    event loop froze /health). Bounded by _DRAIN_WAIT_S; a worker that never
    returns is abandoned (its daemon thread is not killed, but it is no longer
    blocking the loop)."""
    task = asyncio.ensure_future(_drain_body(engine, request_id, worker, body))
    _draining.add(task)
    task.add_done_callback(_draining.discard)


async def _drain_body(engine: Any, request_id: int, worker: Any, body: Any) -> None:
    # Detached from the SSE cancel scope, so these awaits are not re-cancelled by
    # the disconnecting task group.
    # Cancel FIRST: a teardown whose throw lands at the `yield` is a GeneratorExit
    # the loop's `except CancelledError` disconnect branch never sees, so no
    # in-scope cancel ran, and the in-flight next() is parked until exactly this
    # cancel (measured 2026-09-16: 2 of 8 storm hangups leaked this way). This is
    # the same idempotent cancel body.close()'s GeneratorExit would run — moved
    # before the wait, since waiting on a worker blocked on this cancel deadlocks.
    # Idempotent: cancel on a finished/unknown row is a no-op. A cancel that RAISES
    # must not abort the drain before the worker join and body.close() -- a raised
    # cancel used to skip both, leaking the in-flight worker and leaving the
    # generator unclosed. Log it and keep going: the worker is joined bounded below
    # regardless, and body.close()'s GeneratorExit backstop retries the (idempotent)
    # cancel once the generator is suspended.
    try:
        await asyncio.to_thread(engine.cancel, request_id)
    except Exception:
        logging.warning(
            "body drain rid=%s: engine.cancel raised; continuing to worker join "
            "and generator close",
            request_id,
            exc_info=True,
        )
    try:
        await asyncio.wait_for(asyncio.shield(worker), timeout=_DRAIN_WAIT_S)
    except TimeoutError:
        # The row is already cancelled/slot-free, but the sync generator is not
        # closed: closing a still-executing generator raises ValueError. This is
        # the exact event the wedge box must see, so never skip it silently.
        logging.warning(
            "sse drain rid=%s: in-flight worker did not finish in %.0fs; "
            "body.close() skipped (row cancelled, generator unclosed)",
            request_id,
            _DRAIN_WAIT_S,
        )
        return
    except Exception:
        logging.warning(
            "sse drain rid=%s: in-flight worker raised; body.close() skipped "
            "(row cancelled, generator unclosed)",
            request_id,
            exc_info=True,
        )
        return
    await asyncio.to_thread(body.close)


async def _await_drains() -> None:
    """Bounded join of in-flight detached drains at graceful shutdown (SIGTERM).
    asyncio.wait (not gather): the timeout does NOT cancel the drains — each is
    already self-bounded by _DRAIN_WAIT_S — so a late one still closes its body
    after shutdown returns. The supervisor's SIGKILL-on-wedge path needs no join;
    this covers an ordinary restart leaving no unclosed sync generator."""
    if not _draining:
        return
    _, pending = await asyncio.wait(tuple(_draining), timeout=_DRAIN_WAIT_S)
    if pending:
        logging.warning(
            "sse shutdown: %d body drain(s) unfinished after %.0fs; their "
            "generators close late or unclosed",
            len(pending),
            _DRAIN_WAIT_S,
        )


@contextlib.asynccontextmanager
async def _lifespan(app: Any):
    yield
    await _await_drains()


#: /health reports unhealthy when active requests make no step progress for this
#: long. A long but RETURNING prefill (up to a few s) must stay healthy; a
#: wedged device forward (never returns) must not. Overridable for tests/ops.
HEALTH_STUCK_AFTER_S = float(os.environ.get("TILERL_HEALTH_STUCK_S", "60"))


async def await_or_cancel(
    request: Request, engine: Any, rid_box: list, run_fn: Any, *args: Any
) -> Any:
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
    worker.add_done_callback(_worker_retrieved)
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

#: Poll cadence for the websocket disconnect watcher; matches the HTTP routes'
#: _DISCONNECT_POLL_S so a prefill-time close cancels within ~one chunk tick.
_WS_DISCONNECT_POLL_S = 0.05


class _WsClientGone(Exception):
    """The websocket client disconnected while a turn was generating."""


async def _ws_next_or_gone(ws: WebSocket, next_worker: Any) -> Any:
    """Await one in-flight ``to_thread(next(gen, end))`` while watching the
    websocket for a client close, polling at :data:`_WS_DISCONNECT_POLL_S`.

    The old WS loop awaited the blocking ``next`` directly. ``WebSocketDisconnect``
    is raised only by a ``receive``/``send``; prefill emits no frame and the loop
    sends nothing before the first token, so a close before the first frame was
    never seen and ``engine.cancel`` never ran — the slot stayed held for the whole
    prefill (issue #667). A websocket's only close signal is a pending
    ``receive()`` resolving to ``websocket.disconnect``, so a watcher future is
    raced alongside each item fetch (the HTTP routes poll ``is_disconnected``,
    which a websocket does not expose). One turn per socket: any inbound message
    while generating is treated as a close too.
    """
    watcher = asyncio.ensure_future(ws.receive())
    try:
        while True:
            done, _ = await asyncio.wait(
                {next_worker, watcher},
                timeout=_WS_DISCONNECT_POLL_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if next_worker in done:
                # Resolve/cancel the watcher without leaving it orphaned; if it
                # already had an inbound (disconnect) message, retrieve it now.
                if watcher in done:
                    watcher.result()
                else:
                    watcher.cancel()
                return next_worker.result()
            if watcher in done:
                raise _WsClientGone
    except BaseException:
        # The caller owns the generator's in-flight next() (gen.close on teardown);
        # only the watcher is ours to cancel here.
        watcher.cancel()
        raise


async def stream_or_cancel(request: Request, engine: Any, request_id: int, body: Any):
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
    worker.add_done_callback(_worker_retrieved)
    try:
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
                # A fetch can finish in the same window the client hung up. Check
                # BEFORE launching another fetch: otherwise the post-disconnect
                # next() is abandoned in flight and the body is only ever reaped
                # by GC -- which gen_close()s it on the event loop thread.
                if await request.is_disconnected():
                    await asyncio.to_thread(engine.cancel, request_id)
                    return
                yield item
                worker = asyncio.ensure_future(asyncio.to_thread(next, body, _STREAM_END))
                worker.add_done_callback(_worker_retrieved)
            elif await request.is_disconnected():
                await asyncio.to_thread(engine.cancel, request_id)
                return
    finally:
        # Finalize the INNER sync generator OUTSIDE this task's cancellation.
        # Awaiting the in-flight to_thread(next) here used to live inside the SSE
        # task's own cancel scope: a burst of hangups re-delivered cancellation on
        # every checkpoint, so `await shield(worker)` re-raised in a tight loop
        # with no real waiter, spinning the event loop and starving the GIL (the
        # 2026-09-16 wedge). Detach the drain instead — it cancels the row, awaits
        # the worker and runs body.close on a worker thread from a task the cancel
        # cannot reach. The in-scope awaited engine.cancel calls above stay: they
        # release the slot before this frame unwinds; the drain's cancel is the
        # idempotent backstop for a GeneratorExit that skipped them.
        _detach_drain(engine, request_id, worker, body)


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_app(
    engine: Any,
    tokenizer: Tokenizer,
    model_name: str = "tilerl",
    completion_timeout_s: float | None = None,
    stream_pace: bool = False,
    stream_pace_depth: int = 12,
) -> FastAPI:
    """Build the FastAPI app around a running engine and a tokenizer.

    ``engine`` must implement the tileRL contract: ``submit``, ``poll``,
    ``stats``. The engine loop is expected to run in its own thread (the CLI
    starts it); request handlers only submit and poll.

    ``completion_timeout_s`` caps how long the non-stream routes wait on a whole
    reply; None reads ``TILERL_COMPLETION_TIMEOUT_S`` (default 1800, 0 = no
    deadline). Streamed SSE and /ws keep the fixed ``_COMPLETION_TIMEOUT_S``
    frame guard.

    ``stream_pace`` re-times ONLY SSE delta frames through
    :mod:`stream_pacing` to smooth the periodic sparse-refresh stall; the SSE
    envelope, fields, order and usage are unchanged. Opt-in, default off.
    """
    completion_timeout_s = (
        completion_timeout_from_env()
        if completion_timeout_s is None
        else float(completion_timeout_s)
    )
    app = FastAPI(title="tilerl", version=__version__, lifespan=_lifespan)
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
        msg = (
            "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'body'}: {e.get('msg', '')}"
                for e in exc.errors()
            )
            or "invalid request"
        )
        if request.url.path.startswith("/v1/messages"):
            body: dict[str, Any] = {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": msg},
            }
        else:
            body = {
                "error": {
                    "message": msg,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            }
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
        refuse_unsupported(
            reasoning_effort=bad_effort(effort or None),
            tool_choice=named not in ("auto", "none", None),
            **hosted_tool_fields(req.tools),
        )
        tools = tools_for_render(flatten_tools(req.tools), req.tool_choice)
        input_ids = tokenizer.encode(
            _render_chat(
                req.messages, thinking, kw.get("reasoning_effort") or req.reasoning_effort, tools
            )
        )
        if not input_ids:
            raise ValueError("empty prompt after tokenization")
        # Omitted max_tokens means "as much as fits", not 512: a 512 cap ends a long
        # reply at finish_reason=length, which reads to a client as a dropped stream.
        # `room_for` is the engine's own admission arithmetic, so the default is always
        # accepted and a prompt that does not fit still hits submit's refusal.
        max_new = req.max_tokens if req.max_tokens is not None else engine.room_for(len(input_ids))
        params = sampling(
            tokenizer,
            thinking,
            max_new,
            temperature=req.temperature,
            top_p=req.top_p,
            max_think_tokens=cap,
            seed=req.seed,
            logprobs=bool(req.logprobs),
            stop=req.stop,
        )
        # bool(thinking): True when the prompt opened <think>, so the reply carries only
        # the closer and strip_think must be told (None = bare turn, nothing to strip)
        return (
            engine.submit(input_ids, params),
            len(input_ids),
            params.max_new_tokens,
            bool(thinking),
            tools,
        )

    def _await_completion(request_id: int, timeout_s: float = completion_timeout_s) -> list[int]:
        return await_completion(engine, request_id, timeout_s)

    @app.get("/health")
    def health():
        # "ok" was a literal, so an engine whose stats() raises answered the same as a
        # healthy one. Loop liveness is deliberately NOT checked via engine._thread:
        # that wants a liveness method on the engine, not a private attribute read.
        try:
            stats = engine.stats()
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "model": model_name,
                    "stats": None,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        # Step-loop progress: stats() is a lock-free SNAPSHOT, so it keeps
        # returning the last tick while the loop is frozen inside a device
        # forward -- a wedged server answered 200. liveness() reads the last
        # non-idle tick timestamp (idle/quiet engines stay healthy) and flags a
        # stall so LB/ops can detect and restart it.
        liveness = getattr(engine, "liveness", None)
        if liveness is not None:
            live, stuck_s = liveness(HEALTH_STUCK_AFTER_S)
            if not live:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "model": model_name,
                        "stats": stats,
                        "stuck_secs": round(stuck_s, 3),
                    },
                )
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
            request_id, prompt_tokens, max_new, opened, tools = await asyncio.to_thread(
                _submit, req
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "error": overloaded_body(exc) or {"message": str(exc), "type": "api_error"}
                },
            )

        if req.stream:
            from .stream_pacing import pace_deltas

            triples = _deltas(
                request_id,
                max_new,
                opened,
                stop_texts(req.stop),
                tools,
                choice_name(req.tool_choice) != "none",
            )
            triples = pace_deltas(triples, stream_pace, target_depth=stream_pace_depth)
            body = _stream(
                request_id,
                max_new,
                prompt_tokens,
                opened,
                bool((req.stream_options or {}).get("include_usage")),
                stop_texts(req.stop),
                tools,
                choice_name(req.tool_choice) != "none",
                triples,
            )
            return StreamingResponse(
                stream_or_cancel(request, engine, request_id, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        rid_box = [request_id]
        try:
            output_ids = await await_or_cancel(
                request, engine, rid_box, _await_completion, request_id
            )
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
            {
                "id": f"call_{request_id}_{i}",
                "type": "function",
                "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)},
            }
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
        content = (
            None
            if scores is None
            else [
                {"token": tokenizer.decode([tid]), "logprob": None if lp != lp else lp}
                for tid, lp in zip(output_ids, scores)
            ]
        )
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        # null, not "", when a tool call carries no prose:
                        # OpenAI's shape, and "" reads as an empty reply.
                        "content": text or None if tool_calls else text,
                        # None, not "": the field is absent for a bare turn
                        # or thinking off, which is what a client checks.
                        "reasoning_content": reasoning or None,
                        "tool_calls": tool_calls,
                    },
                    "logprobs": None if content is None else {"content": content},
                    # A stop sequence is OpenAI's "stop" too, and it takes precedence
                    # over length: the cap was not what ended this one.
                    "finish_reason": (
                        "tool_calls"
                        if tool_calls
                        else "stop"
                        if stopped
                        else "length"
                        if len(output_ids) >= max_new
                        else "stop"
                    ),
                }
            ],
            "usage": _usage(prompt_tokens, len(output_ids)),
            "system_fingerprint": SYSTEM_FINGERPRINT,
        }

    def _deltas(
        request_id: int,
        max_new: int,
        opened: bool,
        stops: tuple[str, ...] = (),
        tools: list | None = None,
        allow_tool_calls: bool = True,
    ):
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
                        safe = safe[: min(done)] if done else safe[: max(0, len(safe) - stop_hold)]
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
                        f"request {request_id} did not finish within {_COMPLETION_TIMEOUT_S}s"
                    )
                time.sleep(POLL_INTERVAL_S)
            output_ids = _await_completion(request_id)
        except GeneratorExit:
            # Mirrors _stream: the drain closes this generator from a worker
            # thread while it is suspended at a yield, so free the row here. This
            # is the WS path's only cancel backstop when the drain's own
            # engine.cancel raised (the close retries the idempotent cancel).
            # Off the loop on the live path (to_thread(gen.close)); do NOT assume
            # it can never run on the loop (GC of a direct iterator would land here).
            engine.cancel(request_id)
            raise
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
            yield (
                "error",
                {"message": f"{type(exc).__name__}: {exc}", "type": "internal_error"},
                seen,
            )
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
        finish = (
            "tool_calls"
            if calls
            else "stop"
            if stopped
            else "length"
            if len(output_ids) >= max_new
            else "stop"
        )
        yield ("done", finish, len(output_ids))

    def _stream(
        request_id: int,
        max_new: int,
        prompt_tokens: int,
        opened: bool,
        include_usage: bool,
        stops: tuple[str, ...] = (),
        tools: list | None = None,
        allow_tool_calls: bool = True,
        triples=None,
    ):
        created = int(time.time())
        chunk_id = f"chatcmpl-{request_id}"
        yield _sse(_chat_chunk(chunk_id, created, model_name, {"role": "assistant"}))
        completion = 0
        if triples is None:
            triples = _deltas(request_id, max_new, opened, stops, tools, allow_tool_calls)
        try:
            for kind, payload, completion in triples:
                if kind == "error":
                    yield _sse({"error": payload})
                    yield "data: [DONE]\n\n"
                    return
                if kind == "tool_calls":
                    for i, (name, args) in enumerate(payload):
                        # One full-arguments delta per call: the SDK appends, and
                        # index/id/function are byte-identical to the non-stream
                        # message.tool_calls element.
                        yield _sse(
                            _chat_chunk(
                                chunk_id,
                                created,
                                model_name,
                                {
                                    "tool_calls": [
                                        {
                                            "index": i,
                                            "id": f"call_{request_id}_{i}",
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(args, ensure_ascii=False),
                                            },
                                        }
                                    ]
                                },
                            )
                        )
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
                    yield _sse(_chat_chunk(chunk_id, created, model_name, {}, finish=payload))
        except GeneratorExit:
            # Free the row if the sync generator is finalized without going
            # through stream_or_cancel's disconnect branch. Threading caveat:
            # GeneratorExit can run on WHATEVER thread drops the body's last ref
            # -- historically that included the event loop thread when this
            # generator was GC'd as stream_or_cancel's frame tore down, and a
            # synchronous lock-taking cancel there froze /health for the seconds
            # a step tick held engine._lock. stream_or_cancel now closes this
            # body from a worker thread (to_thread(body.close)), so on the live
            # path GeneratorExit runs off the loop; this bare cancel stays for
            # the non-ASGI / direct-iteration cases where there is no loop to
            # offload from. Do NOT re-add an assumption that this never runs on
            # the event loop.
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
    mount_messages(app, engine, tokenizer, model_name, completion_timeout_s)

    # OpenAI Responses: the same engine again, differing only in wire shape --
    # a flat typed `output` list instead of `choices`.
    mount_responses(app, engine, tokenizer, model_name, completion_timeout_s)

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
            request_id, prompt_tokens, max_new, opened, tools = await asyncio.to_thread(
                _submit, req
            )
        except Exception as exc:
            await ws.send_json({"t": "error", "message": f"{type(exc).__name__}: {exc}"})
            await ws.close()
            return

        # _deltas blocks on the engine; stepping it in a thread keeps the event loop free
        # to serve the other routes while one page streams.
        gen, end = (
            _deltas(
                request_id,
                max_new,
                opened,
                stop_texts(req.stop),
                tools,
                choice_name(req.tool_choice) != "none",
            ),
            object(),
        )
        calls = None
        worker = None
        gone = False
        try:
            while True:
                # Race each blocking next() against a client close so a close
                # during the frame-less prefill cancels within a chunk tick (#667).
                worker = asyncio.ensure_future(asyncio.to_thread(next, gen, end))
                worker.add_done_callback(_worker_retrieved)
                item = await _ws_next_or_gone(ws, worker)
                if item is end:
                    break
                kind, payload, completion = item
                if kind == "delta":
                    await ws.send_json({"t": "delta", **payload})
                elif kind == "tool_calls":
                    calls = [
                        {
                            "id": f"call_{request_id}_{i}",
                            "type": "function",
                            "name": n,
                            "arguments": json.dumps(a, ensure_ascii=False),
                        }
                        for i, (n, a) in enumerate(payload)
                    ]
                    await ws.send_json({"t": "tool_calls", "tool_calls": calls})
                elif kind == "error":
                    await ws.send_json({"t": "error", "message": payload["message"]})
                    break
                else:
                    await ws.send_json(
                        {
                            "t": "done",
                            "finish_reason": payload,
                            **({"tool_calls": calls} if payload == "tool_calls" else {}),
                            "usage": _usage(prompt_tokens, completion),
                        }
                    )
        except asyncio.CancelledError:
            # Bare parent-task cancel (shutdown/supervisor): no websocket.disconnect
            # frame ever arrives, so the disconnect watcher cannot see it. Let the
            # finally detach the drain, then re-raise so the cancellation is not
            # swallowed (a bare handler teardown used to leak the slot to fill end).
            raise
        except (_WsClientGone, WebSocketDisconnect):
            # The client went away mid-turn. The finally drains the row/generator;
            # the socket is already closed, so skip the normal close below.
            gone = True
        finally:
            # Detach the teardown OUTSIDE this (possibly being-cancelled) task, the
            # same transport-neutral path SSE uses (stream_or_cancel): cancel the
            # row off the loop, bounded-join the in-flight next() worker, then
            # gen.close() on a worker thread. It must be detached (not awaited
            # in-scope) so a bare parent-task CancelledError still schedules it —
            # the strong module-level _draining set keeps the task alive past this
            # frame's unwind, and _lifespan joins it at shutdown. The parked next()
            # is still executing the generator, so gen.close() runs only after the
            # worker join, inside _drain_body.
            _detach_drain(engine, request_id, worker, gen)
        if not gone:
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
