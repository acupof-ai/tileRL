"""Does a 128-thread consumer warpgroup let `linear_fp4_bwd` emit wgmma?

Reproduces the four-cell table in wins/2026-09-07-fp4-backward-warpgroup.md, which carries the
mechanism, the per-call-count pricing and why two earlier probes missed it. Two knobs, both at
the shipped call site: the M-tile (64 pins 20 tiles at M=1280, 128 halves the dequant
redundancy) and the thread count (`_THREADS=64` gives a 2-warp consumer; Hopper's wgmma issues
from 4).

Each cell reads the CUDA its own kernel compiled to, so the MMA class is measured rather than
inferred from a ratio, and carries `verified`: the first version of this probe could not read
the source and printed "still mma.sync, so this is a tile effect" for every cell -- a failed
read rendered as a verdict, and the opposite of the truth.

Real checkpoint weights: synthetic `randint(0,255)` nibbles are uniform over the e2m1 grid where
a checkpoint concentrates near zero. Same FLOPs, same bytes, but it would hide or invent a
data-dependent stall. Built through the shipped factory -- a copied kernel body is a second
kernel sharing a name.

  scripts/pod_run.sh fp4thr 6 -- python3 -u scripts/probe_fp4_bwd_tiles.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src")
)

import torch  # noqa: E402

#: (N, K) of the two shapes that carry 306.7 of the fp4 row's 307.1 TFLOP -- gate/up and down.
SHAPES = ((17408, 5120), (5120, 17408))
M = 1280  # micro=1 splits the group, so one backward sees one row's tokens


def _median_ms(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def _emitted(kern, *args) -> dict:
    """Which MMA the compiler picked for this cell, or `verified: False`.

    `*args` must be the same arguments the cell was called with: the factory returns a `JITImpl`
    that compiles per shape, so `get_kernel_source()` with no arguments raises
    `TypeError: missing a required argument: 'block_M'` -- there is no single source to ask for.
    The first version of this probe swallowed that raise and printed "still mma.sync, so this is
    a tile effect" for every cell, the opposite of the truth; the second blamed a property/method
    mismatch and still read nothing. Hence `verified`: a False refuses to name a cause rather
    than defaulting to one.
    """
    try:
        src = kern.get_kernel_source(*args)
    except Exception:
        return {"verified": False, "wgmma": None}
    if not (isinstance(src, str) and "__global__" in src):
        return {"verified": False, "wgmma": None}
    guard = src.split("threadIdx.x) < ")[1].split(")")[0] if "threadIdx.x) < " in src else ""
    lb = src.split("__launch_bounds__(")[1].split(",")[0].split(")")[0] \
        if "__launch_bounds__(" in src else ""
    out = {"verified": True, "wgmma": "wgmma" in src, "mma_sync": "mma_sync" in src,
           "launch_bounds": int(lb) if lb.isdigit() else None}
    if out["launch_bounds"] and guard.isdigit():
        # the consumer partition, not launch_bounds, is what correlates: 192 appears on both
        # sides of the wgmma line, which is how the first read of this got reported wrong
        out["consumer"] = out["launch_bounds"] - int(guard)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    a = ap.parse_args()

    from tilerl_kernels import kernels_linear
    from tilerl_kernels.backend import _MMA_RED, _pad2d, _round_up, get_backend

    from tilerl.cli import _build_model

    backend = get_backend()
    assert backend.device.type == "cuda", "this sweep needs the card"
    _, model = _build_model("qwen38-27b", seed=0, keep_master=False)
    rows = []

    for n, k in SHAPES:
        hit = next(((key, wq) for key, wq in model.params.items()
                    if key.endswith(".wq") and tuple(wq.shape) == (n, k // 2)), None)
        if hit is None:
            print(json.dumps({"N": n, "K": k, "skipped": "no packed tensor of this shape"}))
            continue
        key, wq = hit
        wq = backend._served_fp4(wq)  # the layout the shipped kernel reads
        scale = backend._const_f32(model.params[key[: -len(".wq")] + ".scale"])
        g = torch.randn(M, n, dtype=torch.bfloat16, device=backend.device)

        # bf16 reference at the same shape: the ratio every cell is judged against
        w_ref = torch.randn(n, k, dtype=torch.bfloat16, device=backend.device)
        bf16_ms, _, _ = _median_ms(lambda: g @ w_ref, a.reps)
        w_ref = None
        torch.cuda.empty_cache()
        print(json.dumps({"N": n, "K": k, "tensor": key, "bf16_ms": round(bf16_ms, 4)}),
              flush=True)

        for bm in (64, 128):
            gp = _pad2d(backend._c(g), _round_up(M, bm), _round_up(n, _MMA_RED))
            wqp = _pad2d(wq, _round_up(n, _MMA_RED), _round_up(k, 64) // 2)
            sp = _pad2d(scale, _round_up(n, _MMA_RED), _round_up(k, 64) // 16)
            for threads in (64, 128):
                kern = kernels_linear.make_linear_fp4_bwd_mma(backend.target)
                call = (gp, wqp, sp, bm, 64, threads)
                med, lo, hi = _median_ms(lambda: kern(*call), a.reps)
                rows.append({"N": n, "K": k, "bM": bm, "threads": threads,
                             "median_ms": round(med, 4), "min_ms": round(lo, 4),
                             "max_ms": round(hi, 4), "vs_bf16": round(med / bf16_ms, 2),
                             **_emitted(kern, *call)})
                print(json.dumps(rows[-1], sort_keys=True), flush=True)
        g = None
        torch.cuda.empty_cache()

    # the mechanism, before any ratio: did the wide consumer change the MMA at all?
    for t in (64, 128):
        seen = [r for r in rows if r.get("threads") == t]
        read = [r for r in seen if r["verified"]]
        if read:
            print(f"# threads={t}: wgmma in {sum(bool(r['wgmma']) for r in read)}/{len(read)}"
                  f" builds, consumer width {sorted({r.get('consumer') for r in read})}")
        if len(read) < len(seen):
            print(f"# threads={t}: {len(seen) - len(read)} builds unread -- no MMA claim below "
                  f"is evidence for them")
    for n, _k in SHAPES:
        by = {(r["bM"], r["threads"]): r for r in rows if r["N"] == n and "median_ms" in r}
        for bm in (64, 128):
            lo, hi = by.get((bm, 64)), by.get((bm, 128))
            if not (lo and hi):
                continue
            if not (lo["verified"] and hi["verified"]):
                tag = "MMA class unread, so the cause is not established here"
            elif hi["wgmma"] and not lo["wgmma"]:
                tag = "wgmma appeared"
            else:
                tag = "same MMA class both sides, so this is a tile effect"
            print(f"# N={n} bM={bm}: threads 64->128 is "
                  f"{lo['median_ms'] / hi['median_ms']:.3f}x -- {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
