"""CPU gate for the SSE refresh-stall pacer (#165469).

Discrete-event simulation of production running CONCURRENTLY with the pacer:
graph ticks every 41 ms emit 2 tokens, every 32nd tick an eager refresh adds
218 ms. The virtual clock only advances inside the pacer's sleep; while it
advances, every production item whose arrival is due is enqueued first (that
is the producer thread filling the buffer while the pacer waits). Asserts paced
inter-emit p99 and max are under 60 ms, order/count preserved, the tail adds no
wait; pacer OFF must keep the 218 ms gap (the negative control).
"""

from __future__ import annotations

from tilerl.stream_pacing import TokenStreamPacer, pace_deltas

GRAPH_S = 0.041
STALL_S = 0.218
REFRESH_EVERY = 32
TOKENS_PER_TICK = 2
N_TICKS = 80
N_TOK = N_TICKS * TOKENS_PER_TICK  # 160 produced tokens


def _timeline():
    """(arrival_s, cumulative_token_pos) for every produced TOKEN: two tokens
    land together at each graph tick (spec depth 1, ~85% accept); the 32nd tick
    every cycle adds the 218 ms eager refresh to BOTH of that tick's tokens."""
    out = []
    t = 0.0
    tok = 0
    for i in range(N_TICKS):
        if i > 0:
            t += GRAPH_S + (STALL_S if i % REFRESH_EVERY == 0 else 0.0)
        for _ in range(TOKENS_PER_TICK):
            tok += 1
            out.append((t, tok))
    return out


def _drive(enabled: bool, target_depth: int = 12):
    timeline = _timeline()
    if not enabled:
        # raw stream: an item is observable exactly at its production arrival
        return [a for a, _p in timeline], list(range(N_TOK)), timeline[-1][0]

    now = {"t": 0.0}
    pi = 0
    p = TokenStreamPacer(
        target_depth,
        clock=lambda: now["t"],
        sleep=lambda s: now.__setitem__("t", now["t"] + max(0.0, s)),
    )

    def drive_clock_to(want: float):
        nonlocal pi
        # producer runs concurrently: move the clock straight to `want` (do NOT
        # snap to a production tick — that would skip the 24 ms due times), and
        # enqueue every item whose arrival is crossed.
        while pi < N_TOK and timeline[pi][0] <= want + 1e-9:
            _a, pos = timeline[pi]
            p._enqueue(pos, ("delta", pi, pos))
            pi += 1
        now["t"] = want

    def sim_sleep(s):
        drive_clock_to(now["t"] + max(0.0, s))

    p._sleep = sim_sleep

    drive_clock_to(0.0)  # items arriving at t=0
    emit_s, order = [], []
    while pi < N_TOK:
        # pace only while production is still running
        t_before = now["t"]
        outs = list(p.pump())  # pump sleeps the virtual clock via sim_sleep
        if not outs and now["t"] <= t_before + 1e-12:
            # fill phase, or a due time the producer has not filled yet: advance
            # production to its next arrival and re-enqueue.
            now["t"] = max(now["t"], timeline[pi][0])
            drive_clock_to(now["t"])
        for triple in outs:
            _kind, idx, _pos = triple
            order.append(idx)
            emit_s.append(now["t"])
    # production ended: the tail flushes immediately at the last arrival, no wait
    end_t = timeline[-1][0]
    for triple in p.flush():
        order.append(triple[1])
        emit_s.append(end_t)
    return emit_s, order, end_t


def _gaps(ms: list[float]) -> list[float]:
    return [b - a for a, b in zip(ms, ms[1:])]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def test_pacer_smooths_the_periodic_refresh_stall():
    raw, _o, _ = _drive(False)
    paced, order, end_ms = _drive(True)

    assert order == list(range(N_TOK)), "pacer dropped or reordered a delta"

    raw_gaps, paced_gaps = _gaps(raw), _gaps(paced)
    # synthetic input really carries the 218 ms stall (negative control is red):
    # two tokens share each tick (so most gaps are 0), but the refresh tick makes
    # the cross-tick gap jump by the full 218 ms.
    assert max(raw_gaps) >= GRAPH_S + STALL_S - 0.001, (
        f"synthetic timeline lost its stall: max raw gap {max(raw_gaps)}"
    )

    # the gate under test
    assert _pct(paced_gaps, 0.99) < 0.060, (
        f"paced p99 {_pct(paced_gaps, 0.99) * 1000:.1f} ms still shows a refresh stall"
    )
    assert max(paced_gaps) < 0.060, f"paced max gap {max(paced_gaps) * 1000:.1f} ms >= 60"

    # no trailing wait: the headroom is bought up front, not added at the tail
    assert end_ms <= _timeline()[-1][0] + 1e-6, (
        f"pacer added {(end_ms - _timeline()[-1][0]) * 1000:.1f} ms of trailing wait"
    )
    assert paced[0] >= 0.24, (
        f"first-token fill {paced[0] * 1000:.0f} ms shorter than the bought headroom"
    )


def test_pacer_off_is_a_passthrough():
    src = [("delta", {"x": 0}, 1), ("done", "stop", 1)]
    assert list(pace_deltas(iter(src), False)) == src
