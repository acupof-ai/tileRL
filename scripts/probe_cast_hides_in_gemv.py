"""Does a real GEMV hide the f32->f16 cast that feeds it, or add to it?

The one live descendant of task #21. `backend.py:696` casts X inside the chunk loop, once
per GEMV launch, and an isolated graph of the 112 remaining casts measured **3.27 us each,
0.37 ms/token** (wins/2026-09-06-the-launch-floor-is-ten-microseconds.md). That was recorded
as an UPPER bound with the reason stated: the shipped graph interleaves each cast with the
GEMV that consumes it, and a GEMV that is DRAM-bound has idle SMs a cast could occupy.

The OPEN.md row says this is "blocked on capacity, not a job" -- a 19 GB model against
~5 GB free. That is true of `prof_decode_budget.py`, which loads the model. It is NOT true of
the question: a cast and the GEMV it feeds are two kernels, and one weight is 60 MB.

Three arms, all captured graphs so none pays the eager launch floor:

  cast only    112 casts                      -- reproduces the 0.37 ms/token bound
  gemv only    112 GEMVs on pre-cast f16 X    -- the work the cast is hidden behind
  cast + gemv  112 (cast -> gemv) pairs       -- what the shipped path actually runs

fused ~= gemv means the cast is free in place; fused ~= gemv + cast means the bound is the
cost. Anything between is the fraction that hides, which is the quantity #21 asked for.

**Two limitations, stated rather than hidden.** (1) Weights come from a POOL of `--pool`
distinct tensors cycled over the 112 launches, not 112 distinct ones: 112 would be 4.07 GB
against ~5 GB free beside ckl's resident endpoint. A pool still streams from DRAM per launch
-- one 6144x5120 f16 weight is 60 MB against a 6 MB L2, so it cannot be cache-resident even
when reused -- but it does not reproduce the shipped path's total DRAM footprint. (2) This
uses a dense f16 matmul, not the fp4 kernel, whose quantized weights are the capacity
problem. What carries over is DRAM-boundness at M=1 (~2 FLOP/byte), which is the property
that decides whether a launch-bound cast can hide; the fp4 kernel's own occupancy is not
measured here.

  scripts/v100.sh 'python3 scripts/probe_cast_hides_in_gemv.py'
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

#: from the checkpoint's text_config: 16 full-attention layers (o_proj, hq*d=6144 -> 5120)
#: and 48 linear layers (out_proj 6144 -> 5120, ab 5120 -> 96)
SHAPES = [(6144, 5120)] * 16 + [(6144, 5120)] * 48 + [(5120, 96)] * 48

#: V100 L2. The condition that matters is not per-weight size but whether a weight can still
#: be resident when its next launch comes: the whole pool is streamed between two launches of
#: the same tensor, so the pool must exceed L2 by a margin or the reused arm reads no DRAM and
#: the answer biases toward "the cast does not hide". Asserted on the pool total below.
_L2_BYTES = 6 << 20


def _time(fn, iters: int) -> float:
    """ms per iteration, one sync per window, median of 7 windows."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    wins = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        wins.append((time.perf_counter() - t0) / iters)
    return statistics.median(wins) * 1e3


def _graph(fn):
    """Capture `fn` and return its replay. Warmup on a side stream is required."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pool", type=int, default=4,
                    help="distinct weights per shape, cycled over the launches")
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    dev = torch.device("cuda")
    shapes = sorted(set(SHAPES))
    pool_mb = sum(k * n for k, n in shapes) * 2 * args.pool / 2**20
    print(f"free={free >> 20} MiB of {total >> 20}; weight pool ~{pool_mb:.0f} MiB "
          f"({args.pool} per shape)")
    assert (free >> 20) > pool_mb * 2, "not enough free memory; lower --pool"

    assert pool_mb * 2**20 > _L2_BYTES * 4, (
        f"pool is {pool_mb:.0f} MiB against a {_L2_BYTES >> 20} MiB L2: a weight could still "
        "be resident at its next launch, so the GEMV arm would not be DRAM-bound"
    )

    pool = {s: [torch.ones(*s, device=dev, dtype=torch.float16) for _ in range(args.pool)]
            for s in shapes}
    xs32 = [torch.ones(1, k, device=dev, dtype=torch.float32) for k, _ in SHAPES]
    xs16 = [x.to(torch.float16) for x in xs32]
    # one weight per launch, cycled: launch i of shape s takes pool[s][i % pool]
    seen: dict[tuple[int, int], int] = {}
    ws = []
    for s in SHAPES:
        i = seen.get(s, 0)
        ws.append(pool[s][i % args.pool])
        seen[s] = i + 1

    def cast_only():
        for x in xs32:
            x.to(torch.float16)

    def gemv_only():
        for x, w in zip(xs16, ws, strict=True):
            x @ w

    def fused():
        for x, w in zip(xs32, ws, strict=True):
            x.to(torch.float16) @ w

    arms = {name: _time(_graph(fn), args.iters)
            for name, fn in (("cast only", cast_only), ("gemv only", gemv_only),
                             ("cast + gemv", fused))}

    n = len(SHAPES)
    print(f"\n{n} pairs, all in captured graphs\n")
    print(f"{'arm':>14} {'ms':>9} {'us each':>9}")
    for name, ms in arms.items():
        print(f"{name:>14} {ms:>9.3f} {ms * 1e3 / n:>9.2f}")

    cast, gemv, both = arms["cast only"], arms["gemv only"], arms["cast + gemv"]
    added = both - gemv
    print(f"\ncast ADDS to the GEMV  {added:+.3f} ms  ({added * 1e3 / n:+.2f} us each)")
    print(f"cast measured ALONE   {cast:.3f} ms  ({cast * 1e3 / n:.2f} us each)")
    if cast > 0:
        print(f"fraction hidden       {(1.0 - added / cast) * 100:.0f}%  "
              "(100% = free in place, 0% = the isolated bound is the cost)")

    # Which route makes that true? If same-stream kernels in a graph simply serialize, then
    # additivity is a property of the stream and NOT evidence about the GEMV's occupancy --
    # and doubling the kernel count would double the time. It does not: 112 -> 224 casts came
    # back 1.70x, not 2.00x, which says the per-arm cost is `intercept + slope * count` and a
    # graph replay has a fixed cost of its own. So sweep the count and fit, because the
    # intercept is exactly the part that inflated "3.27 us per cast" in the isolated arm.
    print(f"\n{'casts':>7} {'ms':>9} {'us each':>9}")
    fit = []
    for count in (0, 28, 56, 112, 224, 448):
        xs = (xs32 * 4)[:count]

        def many(xs=xs):
            for x in xs:
                x.to(torch.float16)

        ms = _time(_graph(many), args.iters)
        fit.append((count, ms))
        each = f"{ms * 1e3 / count:>9.2f}" if count else f"{'--':>9}"
        print(f"{count:>7} {ms:>9.3f} {each}")

    # slope from the two widest points, intercept from the fit's own zero arm
    (c_lo, ms_lo), (c_hi, ms_hi) = fit[-2], fit[-1]
    slope_us = (ms_hi - ms_lo) * 1e3 / (c_hi - c_lo)
    empty_ms = fit[0][1]
    print(f"\nmarginal cost per cast  {slope_us:.2f} us   (slope, {c_lo} -> {c_hi})")
    print(f"empty graph replay      {empty_ms * 1e3:.1f} us  (measured, the 0-cast arm)")

    # Placement control. The sweep's 112 row and the "cast only" arm above are the SAME 112
    # casts, and they did not agree, so re-measure the first arm last: a difference between
    # two readings of one arm bounds what any difference BETWEEN arms can mean.
    again = _time(_graph(cast_only), args.iters)
    spread = abs(again - cast) / min(again, cast)
    print(f"\n{'cast only, first position':>28} {cast:>8.3f} ms  {cast * 1e3 / n:>6.2f} us each")
    print(f"{'cast only, last position':>28} {again:>8.3f} ms  {again * 1e3 / n:>6.2f} us each")
    print(f"{'same arm, spread':>28} {spread * 100:>8.1f}%")
    print(f"{'cast in the pair, vs alone':>28} {abs(added - cast) / cast * 100:>8.1f}%")
    print("\nThe pair-vs-alone difference is only meaningful if it exceeds the same-arm")
    print("spread. Read those two percentages together before concluding anything about")
    print("hiding: the marginal slope is the figure that does not depend on either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
