#!/usr/bin/env python3
"""What fraction of HBM bandwidth does a B=8 decode tick's fp4 weight stream actually reach?

The rollout's cost is the decode forward: 28.64 of 29.71 ms/tick, against a 6.11 ms
weight-stream floor -- 21.3% of the bandwidth bound, 4.7x of headroom
(errors/2026-09-08-async-launches-billed-the-forward-to-the-sampler.md). That figure is
a whole-tick average over every layer, launch and norm. This times the fp4 GEMMs alone,
at the decode shape, so the utilization is attributed to the kernel rather than to the
tick.

**Operands of the bound, spelled out, because each of the four was wrong once today:**

  * bytes  = the weight bytes of THIS GEMM (nibbles + f32 block scales + oscale), summed
             from the tensors handed to the kernel, not from a config estimate.
  * rate   = bytes / seconds. Reported against both 4.0 TB/s (H20 nameplate) and
             3.35 TB/s (a realistic achieved rate), because a nameplate ratio flatters.
  * PER TICK, not per token: a decode tick streams each weight once and serves all M
             rows from it.
  * AGGREGATE, not per row: dividing by M gives the B=1 figure and understates 8-fold.

**Routing matters here and a synthetic GEMM would miss it.** `backend.linear_fp4`
(backend.py:671) sends 2 <= M <= _MGEMV to an M-row GEMV plan, and `_MGEMV` defaults to 3
(backend.py:108) -- so B=8 decode does NOT take the GEMV path, it takes the tile ladder
below. This probe calls `linear_fp4` itself at each M so the routing is the shipped one,
and sweeps M across the boundary so the switch is visible rather than assumed.

**Fixture: real checkpoint weights, not randn.** A random-normal fixture put top-p's
nucleus at 162301/248320 when the real one is 43, hiding the property under test
(25, 2026-09-08). An fp4 block scale is `block_max / 6`, so a distribution with no
outliers gives every block the same scale and can flatter the dequantization path.
`--randn` exists only to price that difference; it is not the measurement.

**No bare `torch.cuda.synchronize()` near a captured region** -- it raises
`cudaErrorStreamCaptureInvalidated` and silently drops the run to eager, 3.14x. This is
a standalone process that captures nothing, so its syncs are safe by construction.

**The weights must be resident before timing.** `_build_model` returns a CPU model, and
`backend.linear_fp4` migrates at the tilelang boundary (`_dev`, backend.py:432), so
timing it on CPU tensors measures a 133.8 MB host-to-device copy every call. The first
run of this probe did exactly that and reported 3.7 GB/s -- 0.1% of HBM, and 3061 ms/tick
for a stream the tick measurably does in 28.64 ms. Both bounds are asserted below, since
the too-fast assert alone let a 100x-too-slow number through.

Also dumps the resident weight bytes per key. Two independent shape derivations
(18.23 GB and 18.58 GB) fall 5.9 GB short of the 24.44 GB summed off the loaded model,
and the model is loaded here anyway, so the dump is free.

Run (ask tilerl-27 for a card first; 0 and 6 are the tileRL allocation):
  scripts/pod_run.sh --wait fp4gemm <card> -- python3 scripts/probe_fp4_decode_gemm.py
"""
import argparse
import json
import os
import sys
import time

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import torch  # noqa: E402

_HBM = {"nameplate 4.0": 4.0, "achieved 3.35": 3.35}


def _bytes(*ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def _time(fn, n=50) -> float:
    """Seconds per call. Syncs around the loop, not inside it, so launch overhead is
    amortized the way a real tick amortizes it."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,3,4,8,16")
    ap.add_argument("--randn", action="store_true",
                    help="price a synthetic fixture against the real weights; not the "
                         "measurement -- a distribution can hide the property under test")
    ap.add_argument("--out", default="/work/fp4_decode_gemm.json")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    print(f"arch {backend.arch}")

    dev = torch.device("cuda")
    off = [k for k, v in model.params.items() if v.device.type != "cuda"]
    for k in off:
        model.params[k] = model.params[k].to(dev)
    print(f"moved {len(off)} of {len(model.params)} params to {dev}")

    # The byte dump: which keys the 24.44 GB is summed over.
    per_key = {k: v.numel() * v.element_size() for k, v in model.params.items()}
    total = sum(per_key.values())
    by_suffix: dict[str, int] = {}
    for k, v in per_key.items():
        by_suffix[k.split(".")[-1]] = by_suffix.get(k.split(".")[-1], 0) + v
    print(f"\nresident params {total} B = {total / 1e9:.3f} GB ({total / 1024 ** 3:.2f} GiB)"
          f"  over {len(per_key)} keys")
    for s, v in sorted(by_suffix.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {s:<14} {v / 1e9:7.3f} GB")

    # One real fp4 linear at the decode shape: the largest fp4 weight in the model, so
    # the timing is dominated by the stream rather than by launch overhead. Chosen by
    # measured size, not by name -- `sorted(params)[0]` would take whatever key sorts
    # first, which is not the one this comment would be describing.
    fp4_keys = [k[: -len(".wq")] for k in model.params if k.endswith(".wq")]
    key = max(fp4_keys, key=lambda k: model.params[f"{k}.wq"].numel())
    wq = model.params[f"{key}.wq"]
    scale = model.params[f"{key}.scale"]
    oscale = model.params.get(f"{key}.oscale")
    N, Kp = wq.shape[0], wq.shape[1] * 2
    wb = _bytes(wq, scale, oscale)
    print(f"\nGEMM under test: {key}  N={N} K={Kp}  weights {wb / 1e6:.1f} MB")

    rows = []
    emb = model.params["embed_tokens"]
    if emb.shape[1] != Kp:
        raise SystemExit(
            f"hidden {emb.shape[1]} != this GEMM's K={Kp}: {key} is not fed by the "
            "embedding, so pick a GEMM on the residual stream instead of falling back "
            "to randn -- a silent synthetic fixture is how a distribution hides the "
            "property under test"
        )
    for M in [int(x) for x in args.batches.split(",")]:
        x = emb[:M].to(torch.float32).contiguous()
        if args.randn:
            x = torch.randn_like(x)
        s = _time(lambda x=x: backend.linear_fp4(x, wq, scale, oscale=oscale))
        r = {"M": M, "ms": s * 1000, "GB_s": wb / s / 1e9,
             "util": {n: wb / s / 1e9 / (tb * 1000) for n, tb in _HBM.items()}}
        rows.append(r)
        u = "  ".join(f"{n} {100 * v:5.1f}%" for n, v in r["util"].items())
        print(f"  M={M:<3} {r['ms']:7.3f} ms  {r['GB_s']:8.1f} GB/s   {u}")

    # The whole tick's fp4 stream at the measured per-GEMM rate: bytes / rate, which is
    # what the 21.3% whole-tick figure should be compared against.
    fp4_bytes = sum(v for k, v in per_key.items()
                    if k.endswith((".wq", ".scale", ".oscale")))
    m8 = next(r for r in rows if r["M"] == 8)
    print(f"\nfp4 weight bytes over the model: {fp4_bytes / 1e9:.3f} GB")
    print(f"at M=8's measured {m8['GB_s']:.1f} GB/s -> {1000 * fp4_bytes / m8['GB_s'] / 1e9:.2f} "
          f"ms/tick for the fp4 stream alone")
    # Named explicitly so nobody adds these bytes to the line above. This rate is
    # linear_fp4's; the fp8 weights go through linear_fp8, a different kernel whose
    # achieved rate is not measured here. Dividing all quantized bytes by this rate put
    # "the weight stream is 67% of the forward" into a survey before it was retracted.
    fp8_bytes = sum(v for k, v in per_key.items() if k.endswith((".w8", ".wscale")))
    print(f"NOT covered: {fp8_bytes / 1e9:.3f} GB of fp8 weights "
          f"({100 * fp8_bytes / (fp8_bytes + fp4_bytes):.0f}% of quantized bytes) take "
          "linear_fp8, whose rate this probe never measures")
    print("whole-tick measurement for comparison: 28.64 ms/tick "
          "(errors/2026-09-08-async-launches-billed-the-forward-to-the-sampler.md)")

    # A rate above the nameplate means the weights are not being re-read -- a cache
    # effect or a wrong byte count -- not a kernel that beat physics.
    assert m8["util"]["nameplate 4.0"] < 1.0, (
        f"M=8 reads {m8['GB_s']:.1f} GB/s, above the 4000 GB/s nameplate: the byte count "
        "or the timing is wrong, since no kernel exceeds its own bandwidth"
    )
    # And the floor. A GEMM reading HBM cannot be slower than the PCIe link, so a rate
    # down here means the weights are not resident and each call is paying a host copy --
    # which is what the first run measured. The too-fast assert could not see it.
    assert m8["GB_s"] > 100.0, (
        f"M=8 reads {m8['GB_s']:.1f} GB/s, near PCIe rather than HBM speed: the weights "
        "are being copied to the device per call instead of read from it"
    )

    with open(args.out, "w") as f:
        json.dump({"arch": backend.arch, "key": key, "N": N, "K": Kp, "weight_bytes": wb,
                   "rows": rows, "resident_bytes": total, "fp4_bytes": fp4_bytes,
                   "by_suffix": by_suffix}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
