"""Can the fp4 block scales be stored f16 without changing a single value?

The scale stream is 3.22 GB of the checkpoint's 20.35 GB (measured), so f16
halves it to 1.61 and takes the per-token weight traffic to 18.7 GB — a 8%
roofline gain, the largest structural item left. It is only worth doing if the
conversion is EXACT: the served scales are e4m3 (4 significant bits) times a
power of two from renorm_fp4_scale, and f16 carries 11, so it should round-trip
— but 'should' is what this checks.

Reports the worst relative error over every scale tensor in the checkpoint and
the count of values that do not survive f32 -> f16 -> f32.

  scripts/v100.sh run f16 '/usr/bin/python3 -u scripts/check_scale_f16.py'
"""

from __future__ import annotations

import glob
import os
import sys

import torch
from safetensors import safe_open


def main() -> None:
    src = os.environ.get("TILERL_QWEN38_SOURCE")
    if not src:
        raise SystemExit("set TILERL_QWEN38_SOURCE")
    worst, bad, total, tensors = 0.0, 0, 0, 0
    subnormal = 0
    lo, hi = float("inf"), 0.0
    for f in sorted(glob.glob(src + "/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in list(h.keys()):
                if "scale" not in k.lower():
                    continue
                t = h.get_tensor(k).float()
                r = t.half().float()
                d = (t - r).abs()
                nz = t.abs() > 0
                rel = torch.where(nz, d / t.abs().clamp(min=1e-30), d)
                worst = max(worst, float(rel.max()))
                bad += int((d != 0).sum())
                total += t.numel()
                tensors += 1
                v = t[nz].abs()
                if v.numel():
                    lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
                    # f16 normals start at 2^-14; below that precision degrades.
                    subnormal += int((v < 2.0**-14).sum())
    print(f"\nscale tensors {tensors}, values {total/1e6:.1f}M")
    print(f"range [{lo:.3e}, {hi:.3e}]  (f16 normal min 6.104e-05, max 65504)")
    print(f"f16 subnormal values: {subnormal} ({100*subnormal/max(total,1):.4f}%)")
    print(f"values changed by f32->f16->f32: {bad} ({100*bad/max(total,1):.4f}%)")
    print(f"worst relative error: {worst:.3e}")
    print("\nEXACT — f16 scales are a pure win" if bad == 0 else
          f"\nNOT exact: {bad} values move. Compare {worst:.1e} against the "
          "1e-2 parity gate before deciding.")
    if hi > 65504 or lo < 2.0**-24:
        print("!! range exceeds f16 — some scales would overflow or flush to zero")


if __name__ == "__main__":
    sys.exit(main())
