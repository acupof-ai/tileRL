"""Deterministic gate for /health when a request's engine.submit is in flight.

The stats snapshot made /health lock-free on 2026-09-07, but the OpenAI chat
and /ws/chat handlers called the lock-taking _submit on the event loop itself.

The gates are event-driven, not wall-clock: the double's ``submit`` sets
``entered`` and then blocks on a release EVENT, so at the assertion point a
submit is provably in flight (not merely "probably slow inside a time window").
/health returning BEFORE the release is set proves the route did not run the
blocking submit on the event loop. A route that did would deadlock /health
until the release, and the join watchdog catches it. The generous margin only
separates "returned" from "hung"; no latency number is asserted, so a loaded
CI cannot turn the gate red.
"""

from __future__ import annotations

import os
import threading

os.environ.setdefault("TILERL_TARGET", "cpu")

from fastapi.testclient import TestClient
from test_server import _ByteTokenizer, _ScriptedEngine

from tilerl.server import create_app


class _SubmitBlockingEngine(_ScriptedEngine):
    """submit parks on a release event: the test drives /health while a submit
    is guaranteed in flight and has not returned."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.entered = threading.Event()
        self.release = threading.Event()

    def submit(self, input_ids, params=None):
        self.entered.set()
        self.release.wait(30.0)  # a step() holding the lock across a prefill
        return super().submit(input_ids, params)


def _app(engine):
    return create_app(engine, _ByteTokenizer(), model_name="tiny")


def test_health_answers_while_a_chat_submit_is_in_flight():
    """`/health` must answer while a chat submit is blocked off the event loop.

    The property is which THREAD the route calls a blocking `submit` from, not
    engine internals: via asyncio.to_thread the event loop keeps serving /health
    while a worker is parked in submit; on the loop itself /health cannot run
    until submit returns.
    """
    engine = _SubmitBlockingEngine(_ByteTokenizer(), ["</think>\n\ndone"])
    with TestClient(_app(engine)) as c:
        done: dict[str, object] = {}

        def _post():
            done["code"] = c.post(
                "/v1/chat/completions",
                json={"model": "tiny", "max_tokens": 8, "stream": False,
                      "messages": [{"role": "user", "content": "hi"}]}).status_code

        t = threading.Thread(target=_post)
        t.start()
        try:
            assert engine.entered.wait(10.0), "chat submit never entered; the arm proves nothing"
            # submit is provably blocked (release unset): /health must still answer.
            reader: dict[str, object] = {}
            ht = threading.Thread(target=lambda: reader.update(resp=c.get("/health")))
            ht.start()
            ht.join(5.0)
            assert not ht.is_alive(), (
                "/health did not return while a chat submit was in flight — _submit "
                "runs on the event loop instead of via asyncio.to_thread")
            assert not engine.release.is_set(), (
                "/health returned only after the blocking submit completed")
            health = reader["resp"]
        finally:
            engine.release.set()
            t.join(timeout=30)

    assert health.status_code == 200, health.text
    assert done.get("code") == 200, f"the chat request itself failed: {done}"


def test_health_answers_while_a_ws_submit_is_in_flight():
    """Same in-flight-submit defect on `/ws/chat`, which called the same blocking
    `_submit` synchronously after `receive_json`."""
    engine = _SubmitBlockingEngine(_ByteTokenizer(), ["done"])
    with TestClient(_app(engine)) as c:
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
            reader: dict[str, object] = {}
            ht = threading.Thread(target=lambda: reader.update(resp=c.get("/health")))
            ht.start()
            ht.join(5.0)
            assert not ht.is_alive(), (
                "/health did not return while a ws submit was in flight — ws _submit "
                "runs on the event loop instead of via asyncio.to_thread")
            assert not engine.release.is_set(), (
                "/health returned only after the blocking ws submit completed")
            health = reader["resp"]
        finally:
            engine.release.set()
            t.join(timeout=30)

    assert health.status_code == 200, health.text
    assert frames and frames[-1].get("t") == "done", frames
