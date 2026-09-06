"""Is a same-stream cast serialized after its GEMV, or does occupancy decide?

`probe_cast_hides_in_gemv.py` measured that 112 casts ADD 0.346 ms to a 9.393 ms GEMV
sequence -- they do not hide -- and the entry attributed that to the GEMV having no room. That
attribution was never tested, and it has a cheaper rival: **kernels launched into one stream
execute in issue order**, so a cast placed after a GEMV cannot overlap it at ANY occupancy.
If that is the route, "the GEMV has no idle SMs" is a claim the measurement does not support,
and the reject stands on the 1.33% arithmetic alone.

Discriminating arm: run the same casts on a SECOND stream, concurrent with the GEMVs, inside
one captured graph. A graph records cross-stream parallelism, so:

  serial (one stream)      cast + gemv, as shipped        -> ~ gemv + cast
  parallel (two streams)   same work, cast on side stream -> if this is ALSO ~ gemv + cast,
                           the GPU genuinely has no room; if it approaches gemv alone, the
                           serialization was the stream order and occupancy never bound.

The distinction changes what the entry may claim, not the verdict: either way the shipped
path pays the cast, because the shipped path is one stream.

  scripts/v100.sh 'python3 scripts/probe_cast_stream_order.py'
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

#: the real widths, from the checkpoint's text_config (16 full-attn o_proj + 48 out_proj at
#: 6144->5120, 48 ab at 5120->96), same as probe_cast_hides_in_gemv.py
SHAPES = [(6144, 5120)] * 64 + [(5120, 96)] * 48

_L2_BYTES = 6 << 20


def _time(fn, iters: int) -> float:
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
    ap.add_argument("--pool", type=int, default=4)
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    dev = torch.device("cuda")
    shapes = sorted(set(SHAPES))
    pool_mb = sum(k * n for k, n in shapes) * 2 * args.pool / 2**20
    print(f"free={free >> 20} MiB of {total >> 20}; weight pool ~{pool_mb:.0f} MiB")
    assert (free >> 20) > pool_mb * 2, "not enough free memory; lower --pool"
    assert pool_mb * 2**20 > _L2_BYTES * 4, "pool inside 4x L2; the GEMV would not be DRAM-bound"

    pool = {s: [torch.ones(*s, device=dev, dtype=torch.float16) for _ in range(args.pool)]
            for s in shapes}
    xs32 = [torch.ones(1, k, device=dev, dtype=torch.float32) for k, _ in SHAPES]
    xs16 = [x.to(torch.float16) for x in xs32]
    seen: dict[tuple[int, int], int] = {}
    ws = []
    for s in SHAPES:
        i = seen.get(s, 0)
        ws.append(pool[s][i % args.pool])
        seen[s] = i + 1

    side = torch.cuda.Stream()

    def gemv_only():
        for x, w in zip(xs16, ws, strict=True):
            x @ w

    def serial():
        for x, w in zip(xs32, ws, strict=True):
            x.to(torch.float16) @ w

    def parallel():
        """Casts on a side stream, GEMVs on the current one, joined at the end.

        The GEMVs consume the PRE-cast xs16 so there is no data dependency forcing order --
        the question is whether the hardware can run them at the same time, not whether this
        particular fusion can.
        """
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            for x in xs32:
                x.to(torch.float16)
        for x, w in zip(xs16, ws, strict=True):
            x @ w
        cur.wait_stream(side)

    arms = {name: _time(_graph(fn), args.iters) for name, fn in
            (("gemv only", gemv_only), ("serial cast+gemv", serial),
             ("parallel (2 streams)", parallel))}
    # Placement control: the same arm re-measured last. probe_cast_hides_in_gemv.py found an
    # 18.4% same-arm spread on the cast arm, which was 3.3x the difference it was tempting to
    # report -- so no difference here means anything until this spread is on the page.
    again = _time(_graph(gemv_only), args.iters)

    n = len(SHAPES)
    print(f"\n{n} casts + {n} GEMVs, all captured graphs\n")
    print(f"{'arm':>22} {'ms':>9}")
    for k, v in arms.items():
        print(f"{k:>22} {v:>9.3f}")

    g, s, p = arms["gemv only"], arms["serial cast+gemv"], arms["parallel (2 streams)"]
    spread = abs(again - g) / min(again, g)
    print(f"\n{'gemv only, first position':>26} {g:>8.3f} ms")
    print(f"{'gemv only, last position':>26} {again:>8.3f} ms")
    print(f"{'same-arm spread':>26} {spread * 100:>7.1f}%  ({spread * g:.3f} ms)")
    print(f"\ncast added, serial    {s - g:+.3f} ms")
    print(f"cast added, parallel  {p - g:+.3f} ms")
    print(f"serial - parallel     {s - p:+.3f} ms   <- the effect under test")
    if s - g != 0:
        print(f"parallel recovers     {(1 - (p - g) / (s - g)) * 100:.0f}% of the serial add")
    if s - p != 0:
        print(f"effect / same-arm noise = {abs(s - p) / max(spread * g, 1e-9):.1f}x "
              "(needs to be >1 to mean anything)")
    print("\n~0% recovered  -> the GPU has no room; occupancy is the binding reason.")
    print("~100% recovered -> stream order was the reason and occupancy never bound;")
    print("                   the entry may not attribute the add to the GEMV's occupancy.")
    print("Either way the shipped path pays it: the shipped decode graph is one stream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
