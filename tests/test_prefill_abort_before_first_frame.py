"""Real-uvicorn PRE-first-token abort gate (#667).

The mid-stream gate (test_a_mid_stream_sse_close_...) closes AFTER the first
content frame. UI's bug is the other half: aborting a request while it is still
in PREFILL, before ANY first token frame goes out — the client socket closes
immediately, but ``slots_used`` reportedly stays full until the fill ends.

A stub engine reproduces that phase deterministically: submit allocates a slot,
then peek()/take() return None (no tokens = still prefilling) until either
cancel frees it or ``finish`` is called. No content frame can be emitted while
it blocks, so a client that closes has aborted pre-first-frame by construction.

Drives the SAME real-uvicorn transport as the mid-stream gate (httptools EOF ->
http.disconnect -> the server's disconnect watcher -> engine.cancel), for both
SSE and non-stream. Event-synchronized, not wall-clock:
submit observed -> close -> cancel observed -> slot released.

Outcomes this discriminates (see #667):
  * cancel NOT called  -> server/transport misses pre-frame cancel (red, server.py)
  * cancel called, slot not released -> engine preemption (single-forward seam)
  * both -> green: ui's hold is measurement/another factor.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import json
import socket
import threading
import time

from test_server import _ByteTokenizer, _uvicorn_server  # type: ignore


class _PrefillEngine:
    """A row stuck in prefill (peek/take return None) until cancel or finish.

    Holds the slot/blocks submit allocated for the whole wait, so the assertion
    is on a LIVE row: cancel must call and free, not merely record a dead id.
    """

    def __init__(self, tok) -> None:
        self._ids = tok.encode("late first token text")
        self.submitted = threading.Event()
        self.cancel_finished = threading.Event()
        self.cancelled: list[int] = []
        self.blocks_used = self.slots_used = 0
        self._release = threading.Event()

    # lifecycle used by the routes -------------------------------------------------
    def submit(self, input_ids, params=None) -> int:
        self.blocks_used += 4
        self.slots_used += 1
        self.submitted.set()
        return 7

    def peek(self, request_id: int):
        # Still prefilling until released; bounded so a missing cancel fails fast.
        self._release.wait(2.0)
        return None if request_id in self.cancelled else self._ids

    def take(self, request_id: int):
        self._release.wait(2.0)
        if request_id in self.cancelled:
            raise RuntimeError("cancelled")
        return self._ids

    def cancel(self, request_id: int) -> bool:
        if request_id in self.cancelled:
            return False
        self.cancelled.append(request_id)
        self.blocks_used = self.slots_used = 0
        self._release.set()
        self.cancel_finished.set()
        return True

    def room_for(self, prompt_tokens: int) -> int:
        return 512

    def stop_text(self, request_id: int):
        return None

    def logprobs(self, request_id: int):
        return []

    def stats(self) -> dict:
        return {}


def _wait_true(pred, timeout: float, what: str):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(what)


def _abort_before_first_frame(port: int, stream: bool) -> None:
    eng = _PrefillEngine(_ByteTokenizer())
    server, srv_port = _uvicorn_server(eng, _ByteTokenizer())
    try:
        payload = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                              "stream": stream, "max_tokens": 16})
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
            f"\r\n{payload}").encode()
        s = socket.create_connection(("127.0.0.1", srv_port), timeout=5)
        s.sendall(request)
        # The row is admitted and holding its slot before the client aborts.
        _wait_true(eng.submitted.is_set, 5.0, "submit never observed")
        assert eng.slots_used == 1, f"slot not held pre-frame: {eng.slots_used}"
        # Abort NOW, while peek/take return None: no first token has existed.
        s.shutdown(socket.SHUT_RDWR)
        s.close()
        # The transport must observe the disconnect and cancel the live row,
        # which frees its slot -- event-synchronized.
        _wait_true(lambda: 7 in eng.cancelled, 5.0,
                   "engine.cancel NOT called on a pre-first-frame abort "
                   "(server/transport pre-frame disconnect gap)")
        _wait_true(eng.cancel_finished.is_set, 5.0, "cancel did not finish")
        assert eng.slots_used == 0 and eng.blocks_used == 0, (
            f"cancel called but slot not released: {eng.slots_used} slots, "
            f"{eng.blocks_used} blocks (engine preemption seam)")
    finally:
        server.should_exit = True


def test_sse_abort_before_first_token_cancels_and_releases():
    _abort_before_first_frame(0, stream=True)


def test_nonstream_abort_before_first_token_cancels_and_releases():
    _abort_before_first_frame(0, stream=False)
