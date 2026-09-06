"""What IS the eager-harness launch floor on sm70, and does it depend on the kernel?

Five files asserted "~60 us regardless of shape" with no instrument behind any of them;
three cited one of the others. This measured it, and both halves came back: the
shape-independence holds, the magnitude was 6x high.

Measured through the same harness the claim was about (`benchkit.timeit`: CUDA events
around a Python loop, divided by iters) plus two other timer shapes, because the
convention is what the 6x turned out to be about. A floor means per-call time stops
falling as the kernel's own work goes to zero, so sweep work down and read where it
flattens.

Four arms, cheapest first, all through torch (no TileLang compile, so this needs ~nothing
on the card):

* `empty`   -- a no-op-ish kernel (`torch.empty(1).zero_()`), the smallest launch there is
* `tiny`    -- one 1-elem add, still ~0 GPU time
* `sweep`   -- N-element adds over 5 decades, so the knee between launch-bound and
               work-bound is visible rather than assumed
* `shapes`  -- the same two kernels under three timer conventions, which is where the 6x
               against the retired figure lives: amortized is 2.1x below per-call-sync

**Gates on utilization, not residency.** The quantity that inflates a launch floor is
another process *issuing* work; a resident-but-idle endpoint issues nothing. The V100
permanently holds a 27 GB endpoint at 0% util, so the old residency gate refused a card
that was valid. Samples `nvidia-smi` at 200 ms for 30 s (n~150) from a detached process --
a Python thread is starved by the GIL and collected n=2 over a 40 s run, which cannot tell
idle from busy. Measured cost of the sampler on the quantity being measured: 1.006x.

Measured on sm70 2026-09-06 (util n=150, median 0%, max 2%): the floor is **10.1 us
amortized**, **21.2 us per-call-with-sync** and **25.6 us isolated** -- flat across four
decades of element count, so shape-independent as asserted, but 6x below the ~60 us five
files assert. No timer shape here reaches 60. See
wins/2026-09-06-the-launch-floor-is-ten-microseconds.md.

  scripts/v100.sh run floor '/usr/bin/python3 -u scripts/probe_launch_floor.py'
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchkit as bk  # noqa: I001 - after the sys.path insert above, not sortable with it
import torch


_UTIL_LOG = Path("/tmp/probe_launch_floor_util.txt")


def _sampler() -> subprocess.Popen:
    """Detached nvidia-smi at 200ms. A Python thread gets starved by the GIL: an earlier
    version of this sampled n=2 over a 40 s run, and n=2 cannot tell idle from busy."""
    _UTIL_LOG.unlink(missing_ok=True)
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
         "-lms", "200"],
        stdout=_UTIL_LOG.open("w"), stderr=subprocess.DEVNULL,
    )


def _samples() -> list[int]:
    if not _UTIL_LOG.exists():
        return []
    return [int(x) for x in _UTIL_LOG.read_text().split() if x.isdigit()]


def main() -> int:
    if not torch.cuda.is_available():
        print("no cuda; the floor is a property of the launch path and needs the card")
        return 0
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    print(f"device={name} sm{cap[0]}{cap[1]}  free={free >> 20} MiB of {total >> 20}")

    # Gate on utilization over a window, not residency. The quantity that inflates a launch
    # floor is another process ISSUING work; a resident-but-idle endpoint issues nothing.
    # The V100 permanently holds a 27 GB endpoint at 0% util, so a residency gate refused a
    # card that was in fact valid, and the item asking for "an idle V100" was unsatisfiable.
    p = _sampler()
    time.sleep(30)  # 200ms -> n~150; a handful of samples cannot discriminate
    p.terminate()
    p.wait(timeout=5)
    u = sorted(_samples())
    if not u:
        print("REFUSING: no utilization samples, so 'idle' is unverified. Is nvidia-smi on PATH?")
        return 1
    med, p90 = u[len(u) // 2], u[int(len(u) * 0.9)]
    print(f"util over 30s: n={len(u)} min={u[0]} median={med} p90={p90} max={u[-1]}")
    if med > 5:
        print(f"REFUSING: median utilization {med}% means another process is issuing work. "
              "A launch floor is a CPU-side and issue-path quantity and contention inflates "
              "it -- the reading would look like a measurement. Re-run on an idle card. "
              f"({(total - free) >> 20} MiB resident, which on its own is fine.)")
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

    # `bk.timeit` divides an event window by iters, so it reports AMORTIZED per-call cost:
    # the CPU enqueues call k+1 while the GPU runs call k. A harness that syncs per call
    # measures a different, larger quantity. Both get called "the launch floor", so print
    # both rather than let the next reader pick the one that suits the argument.
    print(f"\n{'timer shape':>22} {'tiny_add':>10} {'gemv':>10}")
    rows = {}
    for label, fn in (("tiny_add", lambda: a.add(1.0)), ("gemv", lambda: v @ w.t())):
        amort = bk.timeit(fn, iters=300, warmup=30) * 1000.0
        t0 = time.perf_counter()
        for _ in range(300):
            fn()
            torch.cuda.synchronize()
        per_call = (time.perf_counter() - t0) / 300 * 1e6
        singles = []
        for _ in range(100):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            singles.append(s.elapsed_time(e) * 1000.0)
        singles.sort()
        rows[label] = (amort, per_call, singles[len(singles) // 2])
    for i, shape in enumerate(("amortized (timeit)", "per-call sync (wall)", "isolated (events)")):
        print(f"{shape:>22} {rows['tiny_add'][i]:10.2f} {rows['gemv'][i]:10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
