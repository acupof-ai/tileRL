"""Is `ssd_save_ms` disk time or page-cache time? The cap depends on which.

`kv_cache.py:518` is a bare `torch.save(blob, dst)` with no `fsync`, and the timer wraps
exactly that call. A buffered write returns when the kernel has ACCEPTED the bytes, not when
the device has them, so `ssd_save_ms` is an upper bound on nothing and a lower bound on the
real cost. The 641.8 ms measured for a 320.6 MiB entry is cited five places and is the number
`max_pending=32` is being re-sized against, so which quantity it is decides the cap.

Discriminator, three arms on the same bytes and the same file:

  save            what the tier does today: torch.save, timed
  save+fsync      the same, plus fsync(fd) before stopping the clock -- the device's cost
  implied rate    MiB/s for each, compared to the volume's measured sequential rate

If save == save+fsync, the writes are already reaching the device inside the call and 641.8
is a real per-save cost. If save << save+fsync, the tier's timer measures memcpy into the
page cache, the daemon returns long before the bytes are durable, and a cap sized on 641.8 is
sized on the wrong number in the direction that makes the cap look safe.

Also reports the volume's own rate from a dd-style write, because a per-save figure above the
device's sequential speed is page cache by arithmetic, whatever the arms say
(errors/2026-09-05-the-ssd-benchmark-never-touched-the-ssd.md).

Never drops caches: this is a shared machine and the pod is a shared host.

  uv run python scripts/probe_save_fsync.py --mib 320.6 --reps 3
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import statistics
import tempfile
import time

import torch


def _write_once(blob, path: str, fsync: bool) -> float:
    """ms for one torch.save, optionally including the fsync that makes it durable."""
    t0 = time.perf_counter()
    torch.save(blob, path)
    if fsync:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return (time.perf_counter() - t0) * 1000


def _volume_rate(root: str, mib: float) -> float:
    """MiB/s for one buffered+fsynced raw write of the same size: the device's own ceiling."""
    path = os.path.join(root, "rate-probe.bin")
    buf = os.urandom(1 << 20)
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        for _ in range(int(mib)):
            f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    dt = time.perf_counter() - t0
    os.unlink(path)
    return mib / dt


def main() -> int:
    ap = argparse.ArgumentParser()
    # 320.6 MiB is one real entry: 156.9 MB GDN state + 167.8 MB KV.
    ap.add_argument("--mib", type=float, default=320.6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    root = args.dir or tempfile.mkdtemp(prefix="save-fsync-")
    os.makedirs(root, exist_ok=True)
    free = shutil.disk_usage(root).free / (1 << 20)
    need = args.mib * 3
    print(f"# {root}: {free:.0f} MiB free, probe needs ~{need:.0f}")
    assert free > need * 2, "not enough free space"

    n = int(args.mib * (1 << 20) / 4)
    rates = [_volume_rate(root, args.mib) for _ in range(3)]
    rate = statistics.median(rates)
    print(f"# volume sequential write (buffered + fsync): median {rate:.1f} MiB/s of "
          f"{[f'{r:.0f}' for r in rates]} -- one sample is not a rate\n")

    arms: dict[str, list[float]] = {"save": [], "save+fsync": [], "raw write+fsync": [], "torch.save to RAM": [], "to RAM then 1 write": []}
    for rep in range(args.reps):
        # Distinct contents per rep so no arm can be served a repeat.
        for name, do_fsync in (("save", False), ("save+fsync", True)):
            blob = torch.full((n,), float(rep), dtype=torch.float32)
            path = os.path.join(root, f"{name}-{rep}.pt")
            arms[name].append(_write_once(blob, path, do_fsync))
            os.unlink(path)
            del blob
        # Third and fourth arms. torch.save at 0.06x the volume's rate says the cost is NOT
        # the device, so the alternative must be measured rather than assumed. `tobytes()` is
        # timed INSIDE the raw arm: excluding it would compare torch.save against a write of
        # bytes nobody produced, and any real replacement still has to leave the tensor.
        blob = torch.full((n,), float(rep), dtype=torch.float32)
        path = os.path.join(root, f"raw-{rep}.bin")
        t0 = time.perf_counter()
        mv = blob.numpy().tobytes()
        with open(path, "wb") as f:
            f.write(mv)
            f.flush()
            os.fsync(f.fileno())
        arms["raw write+fsync"].append((time.perf_counter() - t0) * 1000)
        os.unlink(path)
        del mv
        # torch.save to memory: no disk at all, so this is the serializer's own cost.
        buf = io.BytesIO()
        t0 = time.perf_counter()
        torch.save(blob, buf)
        arms["torch.save to RAM"].append((time.perf_counter() - t0) * 1000)
        del buf
        # The decisive arm: serialize to memory, then ONE write+fsync. If this lands near
        # RAM + raw, torch.save's file writer is the bill and the tier can pay ~5x less for
        # the same bytes and the same format -- no dtype change, no new file layout.
        path = os.path.join(root, f"mem1-{rep}.pt")
        t0 = time.perf_counter()
        buf = io.BytesIO()
        torch.save(blob, buf)
        with open(path, "wb") as f:
            f.write(buf.getbuffer())
            f.flush()
            os.fsync(f.fileno())
        arms["to RAM then 1 write"].append((time.perf_counter() - t0) * 1000)
        os.unlink(path)
        del blob, buf

    print(f"{'arm':>12} {'median ms':>10} {'MiB/s':>9} {'vs volume':>10}")
    med = {}
    for name, xs in arms.items():
        m = statistics.median(xs)
        med[name] = m
        r = args.mib / (m / 1000)
        print(f"{name:>12} {m:>10.1f} {r:>9.1f} {r / rate:>9.2f}x")

    ratio = med["save+fsync"] / med["save"]
    print(f"\nfsync multiplies the timed cost by {ratio:.2f}x")
    print(f"the tier's own figure, for reference: 641.8 ms for {args.mib:.1f} MiB "
          f"= {args.mib / 0.6418:.1f} MiB/s")
    print("\n~1.0x  -> torch.save already pays the device; 641.8 is a real per-save cost.")
    print("  >>1x -> the tier times memcpy into the page cache and the bytes are not")
    print("          durable when `ssd_saves` increments; a cap sized on it is sized low.")
    print("A 'save' rate above 1.0x the volume is page cache by arithmetic, either way.")
    raw = med["raw write+fsync"]
    print(f"\nsave+fsync / raw write+fsync = {med['save+fsync'] / raw:.2f}x")
    print(f"  raw write is {args.mib / (raw / 1000):.1f} MiB/s, "
          f"{args.mib / (raw / 1000) / rate:.2f}x the volume")
    print("If raw ~= save, the cost is the write and the device is simply slower for this")
    print("access pattern than a 1 MiB-loop dd. If raw << save, torch.save's own")
    print("serialization is the bill and the tier is not disk-bound at all.")

    if not args.dir:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
