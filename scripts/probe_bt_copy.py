"""What does the draft's block-table copy cost per tick, after widening it to the pool?

`6c6f6df` sized the draft's table to `num_blocks` so `Mb` stops being a shape axis
(it is compiled into the kernel). That trades a compile per new context length for a
wider H2D copy on EVERY draft tick -- and `spec.py` builds it with plain
`torch.zeros` where `engine.py:667` pins its own, so the copy is pageable.

Prices the copy alone at the widths the tree actually reaches, against the 5.5 ms
draft forward it sits next to. If it is under ~1% the widening is free; if not, the
fix is `pin_memory` plus building the table once per request instead of per tick.

    python3 scripts/probe_bt_copy.py            # cpu: prints the byte counts only
    TILERL_TARGET=cuda python3 scripts/probe_bt_copy.py
"""

from __future__ import annotations

import time

import torch

DRAFT_MS = 5.54  # wins/2026-09-04-depth-4-stalls-...: timed, depth 3, B=1, ctx=1024
REPS = 200


def bench(n: int, nb: int, pin: bool) -> float:
    """ms per (build + H2D) of an [n, nb] int64 block table."""
    dev = torch.device("cuda")
    rows = [list(range(min(nb, 6)))] * n
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        bt = torch.zeros(n, nb, dtype=torch.long, pin_memory=pin)
        for i, blocks in enumerate(rows):
            bt[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.long)
        bt.to(dev, non_blocking=pin)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / REPS


def main() -> None:
    # (n, nb): the served config, the ctx=8192 pool from #66, and the pre-6c6f6df width.
    # Both nb arms at BOTH n, or the delta mixes the widening with the batch: comparing
    # an n=8 row against the n=1 pre-fix row read +101 us of "widening cost" that was
    # mostly just eight rows instead of one.
    cases = [(1, 6), (1, 256), (1, 4146), (8, 6), (8, 256), (8, 4146)]
    print(f"# draft forward = {DRAFT_MS} ms; a table under 1% of it is free")
    print(f"{'n':>3} {'nb':>6} {'KiB':>8}", end="")
    if torch.cuda.is_available():
        print(f" {'pageable':>9} {'pinned':>8} {'% draft':>8}")
    else:
        print("   (no cuda: byte counts only)")
    for n, nb in cases:
        kib = n * nb * 8 / 1024
        print(f"{n:>3} {nb:>6} {kib:>8.1f}", end="")
        if torch.cuda.is_available():
            a, b = bench(n, nb, False), bench(n, nb, True)
            print(f" {a:>8.3f}m {b:>7.3f}m {a / DRAFT_MS * 100:>7.2f}%")
        else:
            print()


if __name__ == "__main__":
    main()
