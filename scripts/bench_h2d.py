"""Pinned H2D bandwidth for a GDN snapshot, and what it means against re-prefill.

The kv_tier design rests on one unmeasured figure: I estimated pinned DRAM->HBM at
12 GB/s. This measures it. A snapshot is what a DRAM tier would move on a hit --
48 linear layers x 48 heads x 128^2 f32 state plus one conv-window plane = 149.6
MiB on sm70.

Reports both directions (a demotion is D2H, a promotion is H2D) and both pinned
and unpinned, because the design claims pinned is required and that claim has not
been tested either.

Runs alongside the resident server: it allocates ~600 MiB against ~5 GiB free, and
the server is idle (running: 0). Co-tenancy can only make these numbers WORSE, so
a good number here is trustworthy and a bad one would need a re-run on an idle card.
"""

import torch

SNAP_MIB = 149.6
ITERS = 20


def bench(nbytes: int, pin: bool) -> tuple[float, float]:
    n = nbytes // 4
    host = torch.empty(n, dtype=torch.float32, pin_memory=pin)
    dev = torch.empty(n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    def timed(dst, src):
        # Warm once: the first copy pays page-table setup on unpinned memory.
        dst.copy_(src, non_blocking=pin)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(ITERS):
            dst.copy_(src, non_blocking=pin)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / ITERS

    h2d = timed(dev, host)
    d2h = timed(host, dev)
    return h2d, d2h


def main() -> None:
    free, total = torch.cuda.mem_get_info()
    print(f"card: {torch.cuda.get_device_name(0)}  free {free / 2**20:.0f} / "
          f"{total / 2**20:.0f} MiB")
    nbytes = int(SNAP_MIB * 2**20)
    print(f"snapshot size: {SNAP_MIB} MiB ({nbytes:,} B), {ITERS} iters\n")
    for pin in (True, False):
        h2d, d2h = bench(nbytes, pin)
        tag = "pinned" if pin else "unpinned"
        print(f"{tag:9s} H2D {h2d:7.2f} ms = {SNAP_MIB / 1024 / (h2d / 1000):6.2f} GiB/s"
              f"   D2H {d2h:7.2f} ms = {SNAP_MIB / 1024 / (d2h / 1000):6.2f} GiB/s")
        if pin:
            # The number the design needs: a promotion's cost against re-prefilling.
            # 163.1 s is the measured warm cost of the 11019-token prompt (#96).
            print(f"          -> a snapshot hit reloads in {h2d:.1f} ms; re-prefilling the "
                  f"11019-token prompt costs 163100 ms = {163100 / h2d:,.0f}x")
    print("\nreminder: measured with the 27B server resident and idle, so these are "
          "lower bounds on an idle card.")


if __name__ == "__main__":
    main()
