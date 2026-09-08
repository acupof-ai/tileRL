#!/usr/bin/env python3
"""The fp8 half of the weight stream: what bandwidth does `linear_fp8` reach at decode?

The forward's cost is 28.64 ms/tick and the fp4 GEMMs explain 9.84 ms of it at their
measured 1144.7 GB/s. The other 10.628 GB of quantized weights -- 49% of them -- take
`linear_fp8` (backend.py:820), whose rate has never been measured. That leaves ~8.25 ms
of the forward with no attributed kernel, and this half is the first candidate for it.

Dividing all quantized bytes by the fp4 rate is what put "the weight stream is 67% of the
forward" into a survey before it was retracted: the numerator spanned two quantization
paths and the denominator only one. So this probe times fp8 on its own tensors and reports
the two halves separately, never their sum over one rate.

25's derived expectation, recorded before the run so the result cannot be read into it:
w8 is 10.625 GB at 1 B/param against fp4's 7.499 GB of nibbles at 0.5, so fp8 covers about
0.71x the parameters but moves 1.42x the bytes. At equal utilization the fp8 half would
therefore take LONGER than the fp4 half -- the opposite of treating fp4 as the hot path.
That is derived, not measured; it is a prediction this run can falsify.

**Size-matched pair, and one dtype per process.** The first run took the largest weight of
each dtype: `lm_head` at 1272.7 MB for fp8 against `layers.0.gate_up` at 133.8 MB for fp4,
9.5x apart. A larger weight amortizes launch over more bytes and reaches a higher rate on
that alone, so that comparison measured size at least as much as dtype. Worse, the fp4 arm
read 769.2 GB/s at M=1 where two earlier standalone runs of the identical shape read 1830
and 1855 -- 2.4x slower purely from sharing a process with the fp8 arm. So the arms are now
picked to match in bytes, and `--only` runs one dtype per process with the other's number
taken from a separate run.

**Neither an ideal ratio nor this probe's ratio bounds a whole-decode measurement.** A
pure-occupancy argument gives 0.500x per token for B=8->16, assuming per-call time does
not change with M; measured here it does (fp4 1.575x, fp8 1.255x), which puts the
GEMM-side figure at 0.710x. 25 then measured decode/token at 0.618x -- between the two,
and 115% of this probe's number. Above 100% means this is not a bound: it comes from one
~70 MB shape per dtype, the model spans ~1 MB to 1272 MB, larger weights amortize better,
and decode carries attention, GDN state and launches that scale differently again. Both
"ideals" are point estimates dressed as bounds, wrong in opposite directions. Report the
pair as an interval and say what each end assumes.

**A "GB/s" here is assumed bytes over measured time.** The numerator is
`numel * itemsize`, never measured: if a kernel's tiling keeps a weight resident across
output blocks, it moves fewer bytes than that and the shortfall is reported as a lower
utilization. So these numbers bound "time per assumed byte", and a dtype difference in
them has two indistinguishable readings -- a different achieved rate, or a different
actual byte count (25, 2026-09-08). Separating them needs DRAM traffic from outside the
process (`ncu --metrics dram__bytes_read.sum`), which this probe does not collect.

**Compile gate** (25, #318): `len(backend._kernels)` counts JIT entries, keyed on
`(name, args, kw)`, so a compile is exactly one new key. Any compile during the timed
region means the number includes codegen, and the probe REFUSES rather than reporting it.
This matters more here than in a single-dtype sweep: fp4 and fp8 are different kernels
with separate cache entries, so a warmup count sufficient for one is not evidence for the
other.

Weights must be resident -- a CPU model makes `_dev` copy over PCIe per call and the
result is 3.7 GB/s, which happened.

Run:
  scripts/pod_run.sh --wait fp8rate <card> -- python3 scripts/probe_fp8_decode_gemm.py
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


def _bytes(*ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--only", choices=("fp8", "fp4"), default=None,
                    help="time one dtype in this process. Interleaving both in one process "
                         "slowed the fp4 arm 2.4x against its standalone number.")
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="max |fp8 bytes / fp4 bytes - 1| for the pair to be comparable")
    ap.add_argument("--out", default="/work/fp8_decode_gemm.json")
    args = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model

    backend = get_backend()
    _, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    dev = torch.device("cuda")
    for k, v in list(model.params.items()):
        if v.device.type != "cuda":
            model.params[k] = v.to(dev)

    def _timed(fn):
        """(seconds per call, compiles during the timed region). A compile inside the
        loop makes the mean include codegen, so the count is returned, not assumed."""
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        before = len(backend._kernels)
        t = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / args.iters
        return dt, len(backend._kernels) - before

    fp8 = sorted(k[: -len(".w8")] for k in model.params if k.endswith(".w8"))
    fp4 = sorted(k[: -len(".wq")] for k in model.params if k.endswith(".wq"))
    if not fp8:
        raise SystemExit("no .w8 params: this checkpoint has no fp8 half to measure")
    # Closest pair by bytes, not the largest of each: rate rises with weight size, so a
    # 9.5x size gap would be read as a dtype difference.
    def _sz(k, sfx):
        return _bytes(*(model.params.get(k + s) for s in sfx))

    s8 = {k: _sz(k, (".w8", ".wscale", ".oscale")) for k in fp8}
    s4 = {k: _sz(k, (".wq", ".scale", ".oscale")) for k in fp4}
    k8, k4 = min(((a, b) for a in fp8 for b in fp4),
                 key=lambda ab: abs(s8[ab[0]] / s4[ab[1]] - 1.0))
    w8, ws = model.params[f"{k8}.w8"], model.params[f"{k8}.wscale"]
    o8 = model.params.get(f"{k8}.oscale")
    wq, sc = model.params[f"{k4}.wq"], model.params[f"{k4}.scale"]
    o4 = model.params.get(f"{k4}.oscale")
    b8, b4 = _bytes(w8, ws, o8), _bytes(wq, sc, o4)
    if abs(b8 / b4 - 1.0) > args.tolerance:
        raise SystemExit(
            f"closest pair still differs {b8 / b4:.2f}x in bytes ({k8} {b8 / 1e6:.1f} MB vs "
            f"{k4} {b4 / 1e6:.1f} MB), beyond --tolerance {args.tolerance}: a rate gap this "
            "would produce is size, not dtype"
        )
    print(f"fp8 {k8}  N={w8.shape[0]} K={w8.shape[1]}  {b8 / 1e6:.1f} MB")
    print(f"fp4 {k4}  N={wq.shape[0]} K={wq.shape[1] * 2}  {b4 / 1e6:.1f} MB")

    emb = model.params["embed_tokens"]
    hid = emb.shape[1]
    print(f"\n{'M':>3} {'fp8 ms':>8} {'fp8 GB/s':>9} {'fp8 %':>7} "
          f"{'fp4 ms':>8} {'fp4 GB/s':>9} {'fp4 %':>7} {'fp8/fp4 util':>13}")
    rows, refused = [], []
    for M in [int(x) for x in args.batches.split(",")]:
        x8 = (emb[:M] if w8.shape[1] == hid else
              torch.randn(M, w8.shape[1], device=dev)).to(torch.float32).contiguous()
        x4 = (emb[:M] if wq.shape[1] * 2 == hid else
              torch.randn(M, wq.shape[1] * 2, device=dev)).to(torch.float32).contiguous()
        s8, c8 = ((float("nan"), 0) if args.only == "fp4" else
                  _timed(lambda x=x8: backend.linear_fp8(x, w8, ws, oscale=o8)))
        s4, c4 = ((float("nan"), 0) if args.only == "fp8" else
                  _timed(lambda x=x4: backend.linear_fp4(x, wq, sc, oscale=o4)))
        if c8 or c4:
            refused.append((M, c8, c4))
        u8, u4 = b8 / s8 / 1e9 / (_NAMEPLATE_TBS * 1000), b4 / s4 / 1e9 / (_NAMEPLATE_TBS * 1000)
        rows.append({"M": M, "fp8_ms": s8 * 1000, "fp4_ms": s4 * 1000, "fp8_util": u8,
                     "fp4_util": u4, "fp8_GB_s": b8 / s8 / 1e9, "fp4_GB_s": b4 / s4 / 1e9,
                     "compiles": [c8, c4]})
        print(f"{M:>3} {s8 * 1000:8.3f} {b8 / s8 / 1e9:9.1f} {100 * u8:6.1f}% "
              f"{s4 * 1000:8.3f} {b4 / s4 / 1e9:9.1f} {100 * u4:6.1f}% {u8 / u4:12.2f}x")

    if refused:
        print(f"\nREFUSED: kernels compiled inside the timed region at M={refused} "
              f"(fp8, fp4 counts). Those means include codegen, so they are not rates. "
              f"Raise --warmup and rerun.")
        return 1

    # The two halves, each over its own path's bytes and its own measured rate.
    tot8 = sum(v.numel() * v.element_size() for k, v in model.params.items()
               if k.endswith((".w8", ".wscale")))
    tot4 = sum(v.numel() * v.element_size() for k, v in model.params.items()
               if k.endswith((".wq", ".scale", ".oscale")))
    m8 = next(r for r in rows if r["M"] == 8)
    ms8 = 1000 * tot8 / m8["fp8_GB_s"] / 1e9
    ms4 = 1000 * tot4 / m8["fp4_GB_s"] / 1e9
    print("\nmodel-wide, each half at its own measured M=8 rate:")
    print(f"  fp8  {tot8 / 1e9:6.3f} GB at {m8['fp8_GB_s']:7.1f} GB/s = {ms8:6.2f} ms/tick")
    print(f"  fp4  {tot4 / 1e9:6.3f} GB at {m8['fp4_GB_s']:7.1f} GB/s = {ms4:6.2f} ms/tick")
    if args.only:
        # NaN compares false, so `ms8 > ms4` would silently print "did not hold" for a
        # run that measured one arm. A verdict needs both arms.
        print(f"  --only {args.only}: the other half is not measured here, so neither the "
              "sum nor the fp8-vs-fp4 verdict can be formed from this run alone")
    else:
        print(f"  sum {ms8 + ms4:.2f} ms of the measured 28.64 ms forward "
              f"({100 * (ms8 + ms4) / 28.64:.0f}%)")
        print(f"25's prediction was fp8 slower than fp4: "
              f"{'held' if ms8 > ms4 else 'did not hold'} ({ms8:.2f} vs {ms4:.2f} ms)")

    for r in rows:
        for d in ("fp8", "fp4"):
            if r[f"{d}_util"] != r[f"{d}_util"]:  # NaN: this dtype was not run
                continue
            assert r[f"{d}_util"] < 1.0, f"M={r['M']} {d} above nameplate"
            assert r[f"{d}_GB_s"] > 100.0, f"M={r['M']} {d} at PCIe speed, not resident"

    with open(args.out, "w") as f:
        json.dump({"fp8_key": k8, "fp4_key": k4, "fp8_bytes": b8, "fp4_bytes": b4,
                   "model_fp8_bytes": tot8, "model_fp4_bytes": tot4, "rows": rows}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
