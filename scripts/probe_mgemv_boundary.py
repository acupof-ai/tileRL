"""Does the M-row GEMV still beat the mma8 ladder at every M it is used for?

`_MGEMV=3` (backend.py:108) routes 2 <= M <= 3 to the GEMV. The ruling behind it
(wins/2026-08-29-m-row-gemv.md) is that mma8 pads M to 8 rows unconditionally, so it
costs the same at M=2 as at M=8 -- which makes a GEMV cheaper at small M. That ruling was
measured on a full decode replay, not per GEMM.

The single-GEMM sweep on 2026-09-08 suggests the boundary is off by one: GEMV at M=2 took
0.099 ms while the padded ladder runs ~0.117, but GEMV at M=3 took 0.155 ms -- slower than
the ladder it is preferred over. This tests each M directly by flipping TILERL_MGEMV,
since a cross-M comparison (M=3 GEMV vs M=4 ladder) changes two things at once.

**Criterion, fixed before the run** (tilerl-27): a per-M verdict, never an average. The
boundary belongs at the largest M where the GEMV still wins.

**This must be reported as a search, not a yes/no.** The first version asked "can the
boundary drop to 1", got "no" because M=2 prefers the GEMV, and printed `keep _MGEMV=3` --
a value it had never evaluated, off data showing the boundary belongs at 2. A criterion
containing "largest" cannot be answered by testing one candidate and negating it, and a
probe must not print the incumbent as a conclusion unless the incumbent was an arm.
See errors/2026-09-08-a-yes-no-probe-for-a-where-question.md.

M=1 is reported but excluded: it is outside `2 <= M <= _MGEMV` and has its own ruling.
"""
import os
import sys
import time

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import torch  # noqa: E402


def _time(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    import tilerl_kernels.backend as B
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
    emb = model.params["embed_tokens"]
    B_default = B._MGEMV
    print(f"{key}  N={wq.shape[0]} K={wq.shape[1] * 2}   _MGEMV default {B_default}")

    print(f"\n{'M':>3} {'GEMV ms':>9} {'ladder ms':>10} {'ladder/GEMV':>12}  verdict")
    verdicts, verdict_ratio = {}, {}
    for M in (1, 2, 3, 4):
        x = emb[:M].to(torch.float32).contiguous()
        out = {}
        for arm, mg in (("gemv", 8), ("ladder", 0)):
            B._MGEMV = mg
            out[arm] = _time(lambda x=x: backend.linear_fp4(x, wq, sc, oscale=osc))
        r = out["ladder"] / out["gemv"]
        # Below 1.0 the ladder is faster, i.e. the GEMV is the wrong choice at this M.
        v = "ladder wins" if r < 1.0 else "GEMV wins"
        verdicts[M], verdict_ratio[M] = v, 1 / r
        print(f"{M:>3} {out['gemv']:9.3f} {out['ladder']:10.3f} {r:12.2f}  {v}")
    B._MGEMV = 3

    # Report the boundary the sweep FOUND, not a yes/no on one candidate. Asking "can it
    # drop to 1" and negating that says nothing about whether the incumbent is right: the
    # first version printed `keep _MGEMV=3` off data that says 2.
    won = [M for M in (2, 3, 4) if verdicts[M] == "GEMV wins"]
    found = max(won) if won else 1
    print(f"\nGEMV wins at M={won or 'none'}, so the measured boundary is _MGEMV={found}"
          f" (current {B_default}).")
    if found != B_default:
        print(f"-> the boundary is off by {B_default - found}: at M={found + 1} the ladder "
              f"is {verdict_ratio[found + 1]:.2f}x faster and is not being used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
