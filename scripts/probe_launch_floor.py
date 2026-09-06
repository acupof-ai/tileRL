"""What IS the eager-harness launch floor on sm70, and does it depend on the kernel?

Four files assert "~60 us regardless of shape" and none says how it was obtained; three
of the four cite one of the others. The nearest measured relative is ~40 us on H20
(errors/2026-08-27-fp4-gemv-issue-bound-ncu.md), and whether the sm70 figure is a
re-measurement or an adjustment of it is not stated anywhere.

This measures it directly through the same harness the claim is about: `benchkit.timeit`,
CUDA events around a Python loop. A floor means the per-call time stops falling as the
kernel's own work goes to zero, so sweep work down and read where it flattens.

Three arms, cheapest first, all through torch (no TileLang compile, so this needs ~nothing
on the card and can run beside a busy endpoint):

* `empty`   -- a no-op-ish kernel (`torch.empty(1).zero_()`), the smallest launch there is
* `tiny`    -- one 32-elem add, still ~0 GPU time
* `sweep`   -- N-element adds over 5 decades, so the knee between launch-bound and
               work-bound is visible rather than assumed

If the floor is real and shape-independent, arms 1-2 land on the same number and the sweep
is flat until N is large. If they differ, the "regardless of shape" half is wrong even if
the magnitude is right.

**Run this on an IDLE card.** It needs ~130 MB and will start beside a live endpoint, but
the number it produces would then be wrong in the direction that matters: a launch floor is
a CPU-side and issue-path quantity, and a card serving traffic contends for exactly that.
Measured on this pod once already, an idle-card probe put a 144 MiB pinned copy at 11.55 ms
against 161.9 ms in the live path, 14.0x
(errors/2026-09-05-a-two-variable-condition-read-as-a-dead-end.md). A contended floor
reading is worse than no reading, because it looks like a measurement.

  scripts/v100.sh run floor '/usr/bin/python3 -u scripts/probe_launch_floor.py'
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchkit as bk
import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("no cuda; the floor is a property of the launch path and needs the card")
        return 0
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    print(f"device={name} sm{cap[0]}{cap[1]}  free={free >> 20} MiB of {total >> 20}")
    # Refuse a contended card rather than print a contended number. 1 GiB of headroom is
    # far more than this probe needs; the threshold is "is anything else resident".
    if total - free > (1 << 30):
        print(f"REFUSING: {(total - free) >> 20} MiB is already resident, so another "
              "process is on this card. A launch floor is a CPU-side and issue-path "
              "quantity and contention inflates it -- the reading would look like a "
              "measurement. Re-run on an idle card.")
        return 1

    a = torch.ones(1, device=dev)
    print(f"\n{'arm':>12} {'us/call':>9}")
    for label, fn in (
        ("empty", lambda: torch.empty(1, device=dev).zero_()),
        ("tiny_add", lambda: a.add(1.0)),
        ("inplace", lambda: a.add_(0.0)),
    ):
        us = bk.timeit(fn, iters=200, warmup=20) * 1000.0
        print(f"{label:>12} {us:9.2f}")

    # The knee: per-call time against elements. Launch-bound while flat, work-bound after.
    print(f"\n{'elems':>12} {'us/call':>9} {'GB/s':>9}")
    for n in (1, 1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24):
        x = torch.ones(n, device=dev)
        us = bk.timeit(lambda: x.add(1.0), iters=200, warmup=20) * 1000.0
        gbs = 2 * n * 4 / (us * 1e-6) / 1e9  # read + write, f32
        print(f"{n:12d} {us:9.2f} {gbs:9.1f}")

    # And the number the claim is really about: a GEMV-sized kernel's own GPU time versus
    # what the harness reports for it. Same shape class as o_proj, torch matmul so no JIT.
    print(f"\n{'gemv-ish':>12} {'us/call':>9}")
    w = torch.ones(6144, 5120, device=dev, dtype=torch.float16)
    v = torch.ones(1, 5120, device=dev, dtype=torch.float16)
    us = bk.timeit(lambda: v @ w.t(), iters=200, warmup=20) * 1000.0
    gb = (6144 * 5120 * 2) / 1e9
    print(f"{'1x5120x6144':>12} {us:9.2f}   {gb / (us * 1e-6):.1f} GB/s over {gb * 1e3:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
