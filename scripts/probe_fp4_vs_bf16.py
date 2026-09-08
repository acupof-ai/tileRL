#!/usr/bin/env python3
"""Is fp4's 28.6% of HBM a property of the fp4 path, or of the M=8 shape?

The fp4 GEMM reaches 1144.7 GB/s at M=8 on H20 -- 28.6% of the 4 TB/s nameplate -- and
that shortfall is the 3.5x half of the decode forward's 5.23x headroom
(errors/2026-09-08-async-launches-billed-the-forward-to-the-sampler.md). Two readings
point at opposite work:

  small-M ceiling  -> any kernel is bandwidth-starved at M=8, the 3.5x is mostly
                      unreachable, and the lever is batch size, not the kernel
  fp4 path cost    -> bf16 reaches 60-80% at the same shape and fp4 does not, so the
                      dequantization path is the cost and it is a kernel-side target

Same shape, same M sweep, two dtypes, is the discriminator. It has decided this question
in this tree before: #240 read `vs_bf16` falling from 5.47-5.62x to 1.96-2.05x.

**Criterion, fixed before the run** (tilerl-27): bf16 utilization > 1.5x fp4's at M=8
means the fp4 path is the cost and is worth digging into. Utilizations close together
mean the ceiling is the shape, and the architecture advice shifts from "fix the kernel"
toward "raise the batch".

**Each arm is priced in its own bytes**, summed from the tensors that kernel is handed --
dividing both by one byte count would report the dtype ratio as a bandwidth difference.
Measured: bf16 356.5 MB against fp4 133.8 MB, 2.66x. (Predicted 111.6 MB / 3.20x while
writing this, from a block size of 32; the checkpoint's scale plane is `[N, K/16]`, so the
scales are 44.6 MB, not 22.3. The printed numbers come from the tensors, so the table was
unaffected -- but the estimate was wrong in the paragraph warning about wrong estimates.)

Utilization, not milliseconds, is the comparison: bf16 moving 2.66x the bytes SHOULD take
longer, and the question is which arm gets closer to the memory system's limit.

**What this discriminator cannot settle, found by running it.** The bf16 arm is FLAT from
M=2 to M=32 -- 0.270 / 0.270 / 0.270 / 0.265 / 0.273 ms, a 16x change in M moving 3% --
because it pads M the way mma8 does. So it is not a control that varies with the
condition; over that range it is a constant that happens to cross fp4 near M=8. The ratio
reads 0.97x at M=2, 1.14x at M=8 and 2.20x at M=32, which means the criterion returns a
different verdict depending on where it is read. It licenses "the gap at M=8 is too small
to blame the fp4 path", not "both dtypes hit the same physical ceiling".

M=1 is the one clean point: both paths take the GEMV there (bf16's 0.101 ms breaks its own
constant), and bf16 reaches 88.4% against fp4's 45.8%. Evidence for an fp4 path cost lives
at M=1, which is not the rollout's shape.

Weights must be resident before timing -- `_build_model` returns a CPU model and
`linear_fp4` migrates at the tilelang boundary, which once turned this measurement into a
133.8 MB PCIe copy reported as 3.7 GB/s.

Run:
  scripts/pod_run.sh --wait bf16cmp <card> -- python3 scripts/probe_fp4_vs_bf16.py
"""
import argparse
import json
import os
import sys
import time

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import torch  # noqa: E402

_NAMEPLATE_TBS = 4.0


def _time(fn, n=100) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--out", default="/work/fp4_vs_bf16.json")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model

    backend = get_backend()
    _, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    dev = torch.device("cuda")
    for k, v in list(model.params.items()):
        if v.device.type != "cuda":
            model.params[k] = v.to(dev)

    keys = [k[: -len(".wq")] for k in model.params if k.endswith(".wq")]
    key = max(keys, key=lambda k: model.params[f"{k}.wq"].numel())
    wq, sc = model.params[f"{key}.wq"], model.params[f"{key}.scale"]
    osc = model.params.get(f"{key}.oscale")
    N, K = wq.shape[0], wq.shape[1] * 2

    # The bf16 twin of the same weight, dequantized so the two arms multiply the same
    # numbers -- a randn weight would differ in distribution, and an fp4 kernel's cost can
    # depend on its scales.
    from tilerl_kernels import reference
    # Before any linear_fp4 call: `_served_fp4` (backend.py:652) rewrites wq in place to
    # the sm90-twiddled layout and tags it, so unpacking afterwards would decode a
    # different byte order into a silently wrong bf16 weight. Asserted, not sequenced --
    # the ordering here is invisible at the call site.
    assert getattr(wq, "_tl_layout", "natural") == "natural", (
        f"wq is already {wq._tl_layout}: unpack must run before the fp4 kernel twiddles it"
    )
    w16 = reference.unpack_fp4(wq, sc, osc).to(torch.bfloat16).contiguous()
    assert w16.shape == (N, K), (w16.shape, (N, K))

    b4 = sum(t.numel() * t.element_size() for t in (wq, sc, osc) if t is not None)
    b16 = w16.numel() * w16.element_size()
    print(f"{key}  N={N} K={K}   fp4 {b4 / 1e6:.1f} MB   bf16 {b16 / 1e6:.1f} MB"
          f"   ({b16 / b4:.2f}x the bytes)")

    emb = model.params["embed_tokens"]
    print(f"\n{'M':>3} {'fp4 ms':>8} {'fp4 GB/s':>9} {'fp4 %':>7} "
          f"{'bf16 ms':>8} {'bf16 GB/s':>10} {'bf16 %':>7} {'bf16/fp4':>9}")
    rows = []
    for M in [int(x) for x in args.batches.split(",")]:
        x = emb[:M].to(torch.float32).contiguous()
        s4 = _time(lambda x=x: backend.linear_fp4(x, wq, sc, oscale=osc))
        s16 = _time(lambda x=x: backend.linear(x, w16))
        u4 = b4 / s4 / 1e9 / (_NAMEPLATE_TBS * 1000)
        u16 = b16 / s16 / 1e9 / (_NAMEPLATE_TBS * 1000)
        rows.append({"M": M, "fp4_ms": s4 * 1000, "bf16_ms": s16 * 1000,
                     "fp4_GB_s": b4 / s4 / 1e9, "bf16_GB_s": b16 / s16 / 1e9,
                     "fp4_util": u4, "bf16_util": u16, "ratio": u16 / u4})
        print(f"{M:>3} {s4 * 1000:8.3f} {b4 / s4 / 1e9:9.1f} {100 * u4:6.1f}% "
              f"{s16 * 1000:8.3f} {b16 / s16 / 1e9:10.1f} {100 * u16:6.1f}% "
              f"{u16 / u4:9.2f}x")

    m8 = next(r for r in rows if r["M"] == 8)
    print(f"\nat M=8: fp4 {100 * m8['fp4_util']:.1f}%  bf16 {100 * m8['bf16_util']:.1f}%"
          f"  ratio {m8['ratio']:.2f}x")
    if m8["ratio"] > 1.5:
        print("-> bf16 reaches materially more of the bus at the same shape: the fp4 path "
              "is the cost, and the headroom is a kernel-side target.")
    else:
        print("-> both dtypes sit near the same fraction of the bus: the ceiling is the "
              "M=8 shape, not fp4. Raising the batch is the lever; the kernel is not.")

    # Neither arm may exceed the bus, and neither may sit at PCIe speed -- the two ways
    # this measurement has already been wrong today, one per side of the ratio.
    for r in rows:
        for d in ("fp4", "bf16"):
            assert r[f"{d}_util"] < 1.0, f"M={r['M']} {d} above nameplate: {r[f'{d}_GB_s']:.0f} GB/s"
            assert r[f"{d}_GB_s"] > 100.0, (
                f"M={r['M']} {d} at {r[f'{d}_GB_s']:.1f} GB/s, PCIe rather than HBM speed: "
                "the weights are not resident"
            )

    with open(args.out, "w") as f:
        json.dump({"key": key, "N": N, "K": K, "fp4_bytes": b4, "bf16_bytes": b16,
                   "rows": rows}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
