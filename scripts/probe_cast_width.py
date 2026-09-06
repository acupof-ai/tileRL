"""Is the remaining-112 cast cost pro-ratable by COUNT, or does width matter?

The 193 deleted casts and the 112 remaining ones are different widths, so
112/305 x 1.64 ms is only valid if a cast is launch-bound rather than byte-bound. This
tick's floor sweep said per-call time is flat to 262,144 elements, and every cast width
here is far below that -- but that was measured on `add`, not on a dtype-converting copy,
which reads f32 and writes f16 and might not share the knee.

The real widths, from the checkpoint's text_config (hidden 5120, hq*d 6144, ffn 17408):

  deleted 193:  qkv/qkvz/gate_up/lm_head <- [M, 5120] ; down <- [M, 17408]
  remaining 112: o_proj 16 + out_proj 48 <- [M, 6144] ; ab 48 <- [M, 5120]

If all widths land on the same per-call time, count is the right unit and the pro-rate
holds. If 17408 is dearer, the deleted set was over-weighted and the remaining 112 are
worth MORE per cast than the average -- the pro-rate would understate them.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchkit as bk  # noqa: I001 - after the sys.path insert
import torch


def main() -> int:
    free, total = torch.cuda.mem_get_info()
    print(f"resident={(total - free) >> 20} MiB of {total >> 20}; this probe needs ~1 MiB")
    dev = torch.device("cuda")

    print(f"\n{'width':>8} {'M':>3} {'amortized us':>13} {'isolated us':>12} {'MB':>7} {'GB/s':>8}")
    rows = {}
    for width in (5120, 6144, 17408):
        for m in (1, 8):
            x = torch.ones(m, width, device=dev, dtype=torch.float32)
            amort = min(bk.timeit(lambda: x.to(torch.float16), iters=300, warmup=30) * 1000.0
                        for _ in range(3))
            singles = []
            for _ in range(60):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                x.to(torch.float16)
                e.record()
                torch.cuda.synchronize()
                singles.append(s.elapsed_time(e) * 1000.0)
            iso = statistics.median(singles)
            mb = m * width * 6 / 1e6  # read 4 + write 2 bytes per element
            print(f"{width:>8} {m:>3} {amort:>13.2f} {iso:>12.2f} {mb:>7.3f} "
                  f"{mb / 1e3 / (amort * 1e-6):>8.1f}")
            rows[(width, m)] = amort

    # The question, answered as a ratio rather than by eye.
    for m in (1, 8):
        a, b = rows[(5120, m)], rows[(17408, m)]
        print(f"\nM={m}: 17408 / 5120 = {b / a:.3f}x  "
              f"(bytes ratio is {17408 / 5120:.2f}x)")
        print("  ~1.0x means launch-bound, so pro-rating the 1.64 ms by COUNT is valid.")
        print("  ~3.4x means byte-bound, and the count pro-rate is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
