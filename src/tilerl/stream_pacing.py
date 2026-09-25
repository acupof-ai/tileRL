"""Server-side stream pacing: smooth the periodic sparse-refresh stall.

The sparse decode graph emits ~1.8 tokens every 41 ms, but every 32 ticks an
eager refresh takes ~218-259 ms, so a raw SSE stream shows a >150 ms gap on a
strict 32-tick period (measured 2026-09-25, see probe_stream_smooth.py). This
module re-times ONLY when delta items leave the server; the SSE envelope,
fields, order and usage are produced by the caller unchanged.

The pacer schedules by cumulative completion-token position (each delta the
server yields already carries that), not by frames: deltas batch ~1.8 tokens,
so frame spacing cannot define a steady cadence. It holds the first tokens
until ``target_depth`` have arrived (a one-time headroom delay that buys a
smooth stream from token one — emitting token zero immediately would just push
the first stall to the client), then emits one token every ``interval``.

``interval`` is ADAPTIVE: ``max(PRIOR_SECONDS_PER_TOKEN, cumulative mean wall
per token since the first token)``. The cumulative mean is what makes this work
across the measured regimes — first-miss decode ~47 ms/token, prefix-hit ~83,
hot steady state ~24:

* production slower than the prior (cold/hit): the measured mean dominates, so
  the client is paced at the real (slow) rate and the buffer never runs dry —
  pacing cannot invent throughput, it only removes the self-inflicted gaps;
* production faster than the prior (the graph-only fill ticks): the prior caps
  the rate so the headroom is not spent before the first refresh;
* one 32-tick refresh adds 259 ms over ~64 tokens; averaged cumulatively that
  moves the mean only a fraction of a millisecond, so a single stall does not
  jerk the cadence, while the headroom absorbs the gap itself.

A non-delta frame (done/error/tool_calls) or end-of-stream flushes every
buffered delta immediately, so the tail never waits.

Pure and clock-injected: the CPU gate drives synthetic timelines (periodic
stall, and a slow cold regime) with a virtual clock and no real sleep.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

T = TypeVar("T")

#: tokens buffered before the first emit; at the ~24 ms hot cadence 12 tokens is
#: ~288 ms, covering the ~259 ms worst refresh tick with margin.
DEFAULT_TARGET_DEPTH = 12
#: floor on the emit interval (seconds/token): the hot steady-state mean rounded
#: up (R32 cycle = (31*41+259)/(32*2) ~= 23.9 ms). Keeps the graph-only fill
#: ticks from spending the headroom before the first refresh.
PRIOR_SECONDS_PER_TOKEN = 0.024


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
        # cumulative production mean: wall from the first observed delta
        self._t0: float | None = None
        self._p0 = 0
        self._interval_s = PRIOR_SECONDS_PER_TOKEN
        self._next_emit_t = 0.0

    def _update_interval(self, pos: int, now: float) -> float:
        if self._t0 is None:
            self._t0, self._p0 = now, 0
        dp = pos - self._p0
        dt = now - self._t0
        measured = dt / dp if dp > 0 and dt > 0 else PRIOR_SECONDS_PER_TOKEN
        self._interval_s = max(PRIOR_SECONDS_PER_TOKEN, measured)
        return self._interval_s

    def _enqueue(self, pos: int, item: T) -> int:
        """Add one production item and update the adaptive interval from its
        arrival time. Returns its clamped cumulative position."""
        pos = max(int(pos), self._emitted_pos)
        self._update_interval(pos, self._clock())
        self._buf.append((pos, item))
        return pos

    def pump(self) -> Iterator[T]:
        """Buffer-aware release. Before release, honors the fill gate. After,
        emits every currently-buffered token that the adaptive clock says is due,
        sleeping on it inside this call (the producer advances on its own thread
        meanwhile, but only what is already buffered is sent this round). When the
        buffer runs dry the caller pulls more production and pumps again."""
        if not self._released:
            if not self._buf or self._buf[-1][0] - self._emitted_pos < self.target_depth:
                return iter(())  # still buying headroom
            self._released = True
            self._next_emit_t = self._clock()
        while self._buf:
            pos, item = self._buf[0]
            if pos <= self._emitted_pos:
                self._buf.pop(0)
                yield item
                continue
            wait = self._next_emit_t - self._clock()
            if wait > 0:
                self._sleep(wait)
                continue
            self._emitted_pos = pos
            self._buf.pop(0)
            self._next_emit_t += self._interval_s
            yield item

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
