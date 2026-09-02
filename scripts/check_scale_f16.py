"""Can the block-scale plane be narrowed, and by how much?

The scale plane is 3.20 GB of the 16.04 GB streamed per dense decode token —
20% of the whole weight stream, for one f32 per 16 weights. Narrowing it is the
only structural byte reduction left (the nibbles are already 4-bit). This asks
what each candidate width costs in exactness, on the REAL checkpoint, before any
kernel changes: f16 and bf16 halve the plane, e4m3 quarters it.

The parity gate is rtol=1e-2, so a worst-case relerr well inside that is a pass;
"% values that move" is reported because bit-exactness, if a format has it, is
worth more than a gate pass — it means no re-validation of anything downstream.

  scripts/v100.sh 'cd ~/models/Qwen3.8-27B-NVFP4 && /usr/bin/python3 \
      $HOME/tilerl-v100/scripts/check_scale_f16.py'
"""

from __future__ import annotations

import argparse
import glob

import torch
from safetensors import safe_open

#: (label, bytes/value, quantize) — round-trip through the candidate storage type.
CANDS = [
    ("f16", 2, lambda s: s.half().float()),
    ("bf16", 2, lambda s: s.bfloat16().float()),
    ("e4m3", 1, lambda s: s.to(torch.float8_e4m3fn).float()),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--shards", type=int, default=6, help="0 = all")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.dir}/model-*.safetensors"))
    files = files if args.shards == 0 else files[: args.shards]
    n = 0
    moved = dict.fromkeys((c[0] for c in CANDS), 0)
    worst = dict.fromkeys((c[0] for c in CANDS), 0.0)
    lo = hi = None
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():  # noqa: SIM118 — safe_open exposes only .keys()
                if not k.endswith(".scale"):
                    continue
                s = h.get_tensor(k).float()
                nz = s[s != 0]
                if nz.numel():
                    a, b = nz.abs().min().item(), nz.abs().max().item()
                    lo = a if lo is None else min(lo, a)
                    hi = b if hi is None else max(hi, b)
                for label, _, q in CANDS:
                    r = q(s)
                    moved[label] += int((r != s).sum())
                    # Relative, not absolute: the scale multiplies a whole block,
                    # so its error passes straight through to every product.
                    worst[label] = max(
                        worst[label],
                        ((r - s).abs() / s.abs().clamp(min=1e-30)).max().item(),
                    )
                n += s.numel()

    print(f"# {len(files)} shards, {n / 1e6:.1f}M scale values, "
          f"nonzero magnitude {lo:.3e} .. {hi:.3e}")
    print(f"# plane is 3.20 GB of the 16.04 GB streamed per token (20%)\n")
    print(f"{'store':>6} {'plane GB':>9} {'stream GB':>10} {'roofline':>9} "
          f"{'% moved':>9} {'worst relerr':>13}")
    print(f"{'f32':>6} {3.20:>9.2f} {16.04:>10.2f} {900 / 16.04:>8.1f}/s "
          f"{0.0:>8.1f}% {'exact':>13}")
    for label, nb, _ in CANDS:
        plane = 3.20 * nb / 4
        stream = 16.04 - 3.20 + plane
        print(f"{label:>6} {plane:>9.2f} {stream:>10.2f} {900 / stream:>8.1f}/s "
              f"{100 * moved[label] / n:>8.1f}% {worst[label]:>13.2e}")
    print("\nrtol gate is 1e-2. A format with 0% moved is bit-exact — no downstream")
    print("re-validation; one with a small relerr still has to clear parity.")


if __name__ == "__main__":
    main()
