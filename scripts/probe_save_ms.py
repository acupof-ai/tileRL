"""Time the tier's `torch.save` stage, which five comments describe as "~100 ms"
and no timer has ever measured.

Drives KvTier.spill_state directly rather than a server: the save is a daemon-thread
write of a CPU blob to a path, so it needs no GPU and no model. Runs on this Mac and
on the pod.

The claim under test is a PER-SAVE cost, so the probe reports ms/save, not a total --
a total divided by a save count I did not print is how three earlier rounds went.

Usage:
  uv run python scripts/probe_save_ms.py [--mib 320.6] [--entries 6] [--dir /tmp/x]
"""

import argparse
import os
import shutil
import tempfile
import time

import torch

from tilerl.kv_cache import KvTier


def main() -> int:
    ap = argparse.ArgumentParser()
    # 320.6 MiB is one real entry: 156.9 MB GDN state + 167.8 MB KV, from
    # errors/2026-09-05-the-ssd-benchmark-never-touched-the-ssd.md:28.
    ap.add_argument("--mib", type=float, default=320.6)
    ap.add_argument("--entries", type=int, default=6)
    ap.add_argument("--dir", default=None, help="spill dir; default a fresh temp dir")
    args = ap.parse_args()

    root = args.dir or tempfile.mkdtemp(prefix="save-ms-")
    os.makedirs(root, exist_ok=True)
    # Bytes must never be evicted mid-probe or the mean mixes saves with removals.
    cap = int(args.mib * (1 << 20)) * (args.entries + 2)
    tier = KvTier(root, fingerprint="probe-save-ms", max_bytes=cap)

    n = int(args.mib * (1 << 20) / 4)  # f32
    print(f"# spill dir {root}")
    print(f"# {args.entries} x {args.mib:.1f} MiB = "
          f"{args.entries * args.mib / 1024:.2f} GiB written\n")

    # Distinct contents per entry: one shared tensor re-saved could let the page cache
    # or the allocator serve a repeat in a way a real spill never sees.
    for i in range(args.entries):
        state = torch.full((n,), float(i), dtype=torch.float32)
        tier.spill_state(i, (i * 64,), state, None)

    # The save is off-tick by design, so the probe waits for the daemon rather than
    # timing the enqueue -- timing the caller is what "~100 ms" would have measured.
    t0 = time.perf_counter()
    while tier.stats()["ssd_saves"] < args.entries and time.perf_counter() - t0 < 900:
        time.sleep(0.05)
    wall = time.perf_counter() - t0

    st = tier.stats()
    print(f"keys = {sorted(st)}\n")
    saves = st["ssd_saves"]
    assert saves == args.entries, f"only {saves}/{args.entries} saved after {wall:.1f}s: {st}"

    per = st["ssd_save_ms"] / saves
    mibs = args.mib / (per / 1000)
    print(f"saves            {saves}")
    print(f"save_ms total    {st['ssd_save_ms']}")
    print(f"ms/save          {per:.1f}")
    print(f"implied MiB/s    {mibs:.1f}")
    print(f"daemon wall s    {wall:.2f}")
    print(f"\nvs the ~100 ms asserted in five comments: {per / 100:.1f}x")
    print("A rate far above the host device's speed is the page cache, not the tier:")
    print("this writes to whatever fs --dir names, so compare against THAT device.")

    if not args.dir:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
