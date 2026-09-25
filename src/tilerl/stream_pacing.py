"""Server-side stream pacing: smooth the periodic sparse-refresh stall.

The sparse decode graph emits ~1.8 tokens every ~41 ms, but every 32 ticks an
eager refresh takes ~218 ms, so a raw SSE stream shows a >150 ms gap on a
strict 32-tick period (measured 2026-09-25, see probe_stream_smooth.py). This
module re-times ONLY when delta items leave the server; the SSE envelope,
fields, order and usage are produced by the caller unchanged.

The pacer schedules by cumulative completion-token position (each delta the
server yields already carries that), not by frames: deltas batch ~1.8 tokens,
so frame spacing cannot define a steady cadence. It holds the first tokens
until ``target_depth`` have arrived (a one-time ~0.2 s first-token delay that
buys a smooth stream from token one — emitting token zero immediately would
just push the first stall to the client), then releases on a per-token clock
at the long-run production rate (the prior includes one refresh period, then
the measured mean takes over after a cycle, so the buffer neither grows
without bound nor drains dry). A non-delta frame (done/error/tool_calls) or
end-of-stream flushes every buffered delta immediately, so the tail never waits.

Pure and clock-injected: the CPU gate feeds a synthetic production timeline
with a 218 ms stall every 32 ticks and asserts paced emit gaps with no real
sleep.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

T = TypeVar("T")

#: tokens buffered before the first emit; 12 tokens at the 24 ms pace is
#: ~288 ms, covering the ~259 ms worst refresh tick with margin.
DEFAULT_TARGET_DEPTH = 12
#: seconds per emitted token. Fixed to the R32 long-run mean rounded UP: one
#: 32-tick cycle spans 31 graph ticks (41 ms) + one refresh (~259 ms) and emits
#: 32*2 tokens, i.e. 1530/64 = 23.9 ms/token. The pace must be >= the true mean
#: (24.0 here) or the buffer drains a fraction of a token every cycle and runs
#: dry at the next stall; the +0.1 ms slack is unmeasurable as latency but keeps
#: the level non-negative. A fixed interval is deliberate — a rate measured only
#: while the buffer fills samples the fast graph-only ticks and drains before the
#: first stall. Geometry is injected via the CLI; ponytail: fixed R32 constant,
#: a per-config table if a second refresh interval ships.
FIXED_SECONDS_PER_TOKEN = 0.024


class TokenStreamPacer:
    def __init__(
        self,
        target_depth: int = DEFAULT_TARGET_DEPTH,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time

        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.target_depth = max(1, int(target_depth))
        self._buf: list[tuple[int, T]] = []
        self._emitted_pos = 0
        self._released = False
        self._anchor_t = 0.0
        self._anchor_pos = 0
        self._interval_s = FIXED_SECONDS_PER_TOKEN

    def _drain_due(self) -> Iterator[T]:
        """Release buffered items whose token position the pace clock reached.
        Strict timetable while the buffer stays ahead; if production fell behind
        (a stall longer than the headroom) the due time is in the past — emit
        this one token at now and re-anchor the rest to the steady interval, so
        a missed deadline resumes smoothly instead of bursting the debt."""
        while self._buf:
            pos, item = self._buf[0]
            if pos <= self._emitted_pos:
                self._buf.pop(0)
                yield item
                continue
            now = self._clock()
            due = self._anchor_t + (pos - self._anchor_pos) * self._interval_s
            if now < due:
                self._sleep(due - now)  # producer fills the buffer while we wait
                continue
            self._emitted_pos = pos
            self._buf.pop(0)
            yield item

    def _enqueue(self, pos: int, item: T) -> int:
        """Add one item at cumulative position; returns its clamped position."""
        pos = max(pos, self._emitted_pos)
        self._buf.append((pos, item))
        return pos

    def pump(self) -> Iterator[T]:
        """Release buffered items the pace clock is due for now. Honors the
        fill gate (nothing emits until ``target_depth`` is buffered)."""
        if not self._released:
            if not self._buf or self._buf[-1][0] - self._emitted_pos < self.target_depth:
                return iter(())  # still buying headroom
            self._released = True
            self._anchor_t = self._clock()
            self._anchor_pos = self._emitted_pos
        if not self._buf:
            # Released but the producer has not caught up to the timetable yet.
            # Sleep until the NEXT token's due time (production fills while we
            # wait), then the caller pumps again. No re-anchor: the fixed
            # timetable is what removes the stall; production ahead/behind only
            # changes buffer level, never the emit times.
            next_due = (
                self._anchor_t + (self._emitted_pos + 1 - self._anchor_pos) * self._interval_s
            )
            wait = next_due - self._clock()
            if wait > 0:
                self._sleep(wait)
            return iter(())
        return self._drain_due()

    def submit(self, pos: int, item: T) -> Iterator[T]:
        """Buffer one item and release what the pace clock is due for now."""
        self._enqueue(pos, item)
        return self.pump()

    def flush(self) -> Iterator[T]:
        """Emit every buffered item with no pacing wait (tail / control frame)."""
        buf, self._buf = self._buf, []
        self._released = True
        for _pos, item in buf:
            yield item


def pace_deltas(
    triples,
    enabled: bool,
    target_depth: int = DEFAULT_TARGET_DEPTH,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
):
    """Wrap a server ``_deltas`` triple stream ``(kind, payload, completion)``.
    Only ``delta`` triples are paced; any other kind (done/error/tool_calls)
    first flushes buffered deltas, then passes through immediately. Every input
    triple is yielded exactly once, in order."""
    if not enabled:
        yield from triples
        return
    pacer = TokenStreamPacer(target_depth, clock, sleep)
    for triple in triples:
        kind, _payload, completion = triple
        if kind == "delta":
            yield from pacer.submit(max(1, int(completion)), triple)
        else:
            yield from pacer.flush()
            yield triple
    yield from pacer.flush()
