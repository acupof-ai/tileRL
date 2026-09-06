"""Does a captured CUDA graph pay the eager launch floor? Measured: no, 2.99x less.

Two bounds were published for the 112 remaining f32->f16 casts and called an unresolved
conflict:
  count pro-rate  0.60 ms/token = 112/305 x the profiled 1.64 ms  -> 5.4 us per cast
  floor x count   1.13 ms/token = 112 x the measured 10.1 us floor

The gap was attributed to "floor x count assumes no overlap". Wrong twice: `bk.timeit` is
already back-to-back, so 10.1 us/call IS the overlapped rate; and the 1.64 ms came from a
profile of the CAPTURED GRAPH, which does not pay per-call launch cost. That is the premise
of #21's "the microbench has an eager launch floor the graph path does not pay" -- it was in
the same entry, unapplied.

Result on sm70, the same 112 casts at the real widths (o_proj 16 + out_proj 48 at 6144,
ab 48 at 5120), M=1: eager 9.76 us/cast, captured graph 3.27, 2.99x. The eager arm
reproduces the floor; the graph arm sits below the pro-rate. The bounds were an eager number
and an in-graph number, not a disagreement.

  eager      -- a Python loop of 112 casts, timed back-to-back
  graph      -- the same 112 casts captured once and replayed
  saves      -- the difference, which is the launch cost per cast

Needs ~1 MiB, so it runs beside the endpoint.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: I001 - after the sys.path insert

#: the real shapes, from the checkpoint's text_config: 16 full-attention layers (o_proj)
#: and 48 linear layers (out_proj at hq*d=6144, ab at hidden=5120)
WIDTHS = [6144] * 16 + [6144] * 48 + [5120] * 48


def _time(fn, iters: int) -> float:
    """Wall clock per iteration, one sync at the end. Median of 5 windows."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    wins = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        wins.append((time.perf_counter() - t0) / iters)
    return statistics.median(wins) * 1e3  # ms


def main() -> int:
    free, total = torch.cuda.mem_get_info()
    print(f"resident={(total - free) >> 20} MiB of {total >> 20}; this probe needs ~1 MiB")
    dev = torch.device("cuda")
    xs = [torch.ones(1, w, device=dev, dtype=torch.float32) for w in WIDTHS]
    print(f"{len(xs)} casts: {WIDTHS.count(6144)} at 6144, {WIDTHS.count(5120)} at 5120")

    def eager():
        for x in xs:
            x.to(torch.float16)

    eager_ms = _time(eager, iters=20)

    # Capture the same work. Warmup on a side stream is required before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            eager()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [x.to(torch.float16) for x in xs]
    assert len(outs) == len(xs)

    graph_ms = _time(g.replay, iters=20)

    print(f"\n{'arm':>22} {'ms / 112 casts':>15} {'us per cast':>13}")
    print(f"{'eager loop':>22} {eager_ms:>15.3f} {eager_ms * 1e3 / len(xs):>13.2f}")
    print(f"{'captured graph':>22} {graph_ms:>15.3f} {graph_ms * 1e3 / len(xs):>13.2f}")
    print(f"{'graph saves':>22} {eager_ms - graph_ms:>15.3f} "
          f"{(eager_ms - graph_ms) * 1e3 / len(xs):>13.2f}")
    if graph_ms > 0:
        print(f"\neager / graph = {eager_ms / graph_ms:.2f}x")
    print("Measured 2026-09-06 on sm70: eager 9.76 us/cast (which reproduces the 10.1 us")
    print("floor) against 3.27 in-graph, 2.99x. So the two published bounds for these")
    print("casts -- 0.60 ms/token from an in-graph profile and 1.13 from floor x count --")
    print("were an in-graph number and an eager number, not a conflict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
