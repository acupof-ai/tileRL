"""Red/green gate for /health when a request's engine.submit waits on step()'s lock.

The stats snapshot made /health lock-free on 2026-09-07, but the OpenAI chat and
/ws/chat handlers called the lock-taking _submit on the event loop itself.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("TILERL_TARGET", "cpu")

from fastapi.testclient import TestClient
from test_server import _ByteTokenizer, _ScriptedEngine

from tilerl.server import create_app


def test_health_answers_while_a_submit_waits_on_the_engine_lock():
    """`/health` must answer while a chat/ws request's `submit` waits for `step()`'s lock.

    Distinct from `test_a_request_in_flight_does_not_freeze_the_server`: that gate covers
    the post-submit poll on /v1/messages and /v1/responses, both of which already run their
    whole handler in `asyncio.to_thread`. The OpenAI chat route and `/ws/chat` only wrapped
    the WAIT — `_submit` itself (which takes the engine lock inside `engine.submit`) ran on
    the event loop. During a long prefill `step()` holds that lock across the whole forward,
    so a request arriving then parks the single event loop on lock acquisition and every
    lock-free route — `/health` included — stalls for the rest of the prefill. cc's V100
    soak saw /health return no response 4 times in ~50 min under a 4x7.4k burst; the stats
    snapshot fix (2026-09-07) made /health lock-free but could not unblock the loop.

    A double whose `submit` blocks is the right substitute: the property is which THREAD
    the route calls a blocking `submit` from, not engine internals.
    """
    tok = _ByteTokenizer()

    class _SubmitBlockingEngine(_ScriptedEngine):
        #: 1.4 s, between the 1 s /health assertion and the 30 s join: a blocked event
        #: loop answers at ~1.4 s (red), a free one at ms (green).
        HOLD_S = 1.4

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.entered = threading.Event()

        def submit(self, input_ids, params=None):
            self.entered.set()
            time.sleep(self.HOLD_S)  # a step() holding the lock across a prefill
            return super().submit(input_ids, params)

    engine = _SubmitBlockingEngine(tok, ["</think>\n\ndone"])
    app = create_app(engine, tok, model_name="tiny")
    with TestClient(app) as c:
        done: dict[str, object] = {}
        t = threading.Thread(target=lambda: done.update(
            code=c.post("/v1/chat/completions",
                        json={"model": "tiny", "max_tokens": 8, "stream": False,
                              "messages": [{"role": "user", "content": "hi"}]}).status_code))
        t.start()
        try:
            assert engine.entered.wait(10.0), "chat submit never entered; the arm proves nothing"
            t0 = time.monotonic()
            health = c.get("/health")
            elapsed = time.monotonic() - t0
        finally:
            t.join(timeout=30)

    assert health.status_code == 200, health.text
    assert elapsed < 1.0, (
        f"/health took {elapsed:.2f}s while a chat submit waited on the engine lock — "
        f"_submit runs on the event loop instead of via asyncio.to_thread")
    assert done.get("code") == 200, f"the chat request itself failed: {done}"


def test_health_answers_while_a_ws_submit_waits_on_the_engine_lock():
    """Same lock-on-the-loop defect on `/ws/chat`, which called the same blocking `_submit`
    synchronously after `receive_json`."""
    tok = _ByteTokenizer()

    class _SubmitBlockingEngine(_ScriptedEngine):
        HOLD_S = 1.4

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.entered = threading.Event()

        def submit(self, input_ids, params=None):
            self.entered.set()
            time.sleep(self.HOLD_S)
            return super().submit(input_ids, params)

    engine = _SubmitBlockingEngine(tok, ["done"])
    app = create_app(engine, tok, model_name="tiny")
    with TestClient(app) as c:
        frames: list[dict] = []

        def _ws():
            with c.websocket_connect("/ws/chat") as ws:
                ws.send_json({"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 8})
                while True:
                    msg = ws.receive_json()
                    frames.append(msg)
                    if msg.get("t") in ("done", "error"):
                        break

        t = threading.Thread(target=_ws)
        t.start()
        try:
            assert engine.entered.wait(10.0), "ws submit never entered; the arm proves nothing"
            t0 = time.monotonic()
            health = c.get("/health")
            elapsed = time.monotonic() - t0
        finally:
            t.join(timeout=30)

    assert health.status_code == 200, health.text
    assert elapsed < 1.0, (
        f"/health took {elapsed:.2f}s while a ws submit waited on the engine lock — "
        f"ws _submit runs on the event loop instead of via asyncio.to_thread")
    assert frames and frames[-1].get("t") == "done", frames
