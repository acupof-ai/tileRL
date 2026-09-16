"""Real-uvicorn WS pre-first-token abort gate (#667) — the playground transport.

UI aborts through ``/ws/chat``, not SSE/non-stream. Unlike ``stream_or_cancel``
(which actively polls ``is_disconnected`` while a ``to_thread(next)`` blocks),
``ws_chat`` is a plain loop::

    while (item := await asyncio.to_thread(next, gen, end)) is not end:
        await ws.send_json(...)
    except WebSocketDisconnect: engine.cancel(...)

During a long cold PREFILL the coroutine is parked in the first
``to_thread(next)`` waiting for token one. A client ``ws.close()`` does not
interrupt that worker, and the loop does not re-await a ws operation until
``next`` returns, so ``WebSocketDisconnect`` — and therefore
``engine.cancel`` — only fires after the fill naturally emits its first token
(the ~80s ui observed). There is no SSE-style disconnect watcher on the WS path.

This gate connects with the real ``websockets`` client against real uvicorn,
sends an ask against a prefill-blocked engine (no first token can exist), closes
BEFORE any frame, and asserts — event-synced — that ``engine.cancel`` is called
and the slot freed within ~one tick. RED today; the fix is a WS disconnect
watcher that observes the close while the first ``next`` is blocked (mirror of
stream_or_cancel). The SSE/non-stream equivalents stay green controls.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import asyncio
import threading
import time

import websockets  # type: ignore
from test_server import _ByteTokenizer, _uvicorn_server  # type: ignore


class _PrefillHeldEngine:
    """A row that NEVER reaches its first token until cancelled.

    peek/take block on a long bound and return None the whole time (a cold fill
    that has not emitted token one), so ``next(gen)`` cannot return on its own
    inside the assertion window. If cancel only fires after ``next`` returns
    (the WS gap), it never fires here; an active disconnect watcher fires it
    while the row is still blocked. ``emitted`` records any token produced,
    which must stay 0.
    """

    def __init__(self) -> None:
        self.submitted = threading.Event()
        self.cancel_finished = threading.Event()
        self.release = threading.Event()
        self.cancelled: list[int] = []
        self.emitted = 0
        self.blocks_used = self.slots_used = 0

    def submit(self, input_ids, params=None) -> int:
        self.blocks_used += 4
        self.slots_used += 1
        self.submitted.set()
        return 7

    def peek(self, request_id: int):
        self._hold(request_id)
        return None  # always still prefilling

    def take(self, request_id: int):
        self._hold(request_id)
        if request_id in self.cancelled:
            raise RuntimeError("cancelled")
        return None

    def _hold(self, request_id: int) -> None:
        # Bounded and SHORT: enough to outlive the cancel-check window, then the
        # teardown force-cancels so the parked worker ends and the suite exits.
        self.release.wait(2.5)

    def cancel(self, request_id: int) -> bool:
        if request_id in self.cancelled:
            return False
        self.cancelled.append(request_id)
        self.blocks_used = self.slots_used = 0
        self.release.set()
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


def _wait_true(pred, timeout: float, what: str) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(what)


def test_ws_abort_before_first_token_cancels_and_releases():
    async def scenario() -> None:
        eng = _PrefillHeldEngine()
        server, port = _uvicorn_server(eng, _ByteTokenizer())
        ws = None
        failed: AssertionError | None = None
        try:
            ws = await asyncio.wait_for(
                websockets.connect(f"ws://127.0.0.1:{port}/ws/chat"), timeout=5)
            await ws.send('{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}')
            _wait_true(eng.submitted.is_set, 5.0, "submit never observed")
            assert eng.slots_used == 1, f"slot not held pre-frame: {eng.slots_used}"
            # Abort while token one is still pending and next() cannot return.
            await ws.close()
            try:
                _wait_true(lambda: 7 in eng.cancelled, 1.2,
                           "RED: /ws/chat did not call engine.cancel on a "
                           "pre-first-frame close while the first to_thread(next) "
                           "is blocked (no SSE-style disconnect watcher)")
                _wait_true(eng.cancel_finished.is_set, 1.2, "cancel did not finish")
                assert eng.slots_used == 0 and eng.blocks_used == 0, (
                    f"cancel called but slot not released: {eng.slots_used} slots, "
                    f"{eng.blocks_used} blocks")
                assert eng.emitted == 0, "a token was emitted before cancel"
            except AssertionError as exc:
                failed = exc  # red is the expected result; still tear down cleanly
            finally:
                eng.release.set()  # unblock any parked worker
                if 7 not in eng.cancelled:
                    eng.cancel(7)  # free the orphaned row so the server can wind down
        finally:
            if ws is not None:
                with __import__("contextlib").suppress(Exception):
                    await ws.close()
            server.should_exit = True
        if failed is not None:
            raise failed

    asyncio.run(scenario())
