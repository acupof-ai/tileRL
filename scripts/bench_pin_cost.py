"""Where a DRAM demotion's time goes: pinned allocation, the copy, or neither.

`DramSnapshots.demote` calls `torch.empty_like(..., pin_memory=True)` per snapshot.
Page-locking is a kernel operation that scales with the buffer, and unlike the copy it
cannot overlap. This times the two halves separately at the real snapshot size.

  python scripts/bench_pin_cost.py --mib 144 --iters 10
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=float, default=144.0)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no cuda device")
        return 1

    n = int(args.mib * 2**20 // 4)
    src = torch.randn(n, device="cuda")

    def timed(fn, label: str) -> float:
        fn()  # warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / args.iters
        print(f"{label:34s} {ms:8.2f} ms   {args.mib / 1024 / (ms / 1000):7.2f} GiB/s")
        return ms

    alloc = timed(lambda: torch.empty(n, pin_memory=True), "pinned alloc, fresh each time")
    plain = timed(lambda: torch.empty(n), "unpinned alloc")
    reused = torch.empty(n, pin_memory=True)
    copy_in = timed(lambda: reused.copy_(src), "copy into a REUSED pinned buffer")

    def fresh():
        buf = torch.empty(n, pin_memory=True)
        buf.copy_(src)

    both = timed(fresh, "alloc + copy (what demote does now)")
    unpinned = torch.empty(n)
    timed(lambda: unpinned.copy_(src), "copy into an unpinned buffer")

    print()
    print(f"allocation is {alloc / both * 100:.0f}% of what demote pays")
    print(f"a reused pinned buffer would be {both / copy_in:.2f}x faster")
    print(f"pinned alloc is {alloc / plain:.1f}x an unpinned one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
