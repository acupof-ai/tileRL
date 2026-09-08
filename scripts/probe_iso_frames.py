"""Which matrix fails ISO's orthonormality guard, and does the failure track shape?

The ISO arm on the 27B raised at iso.py:74 -- `frames in torch.float32 are not orthonormal
(max|UᵀU−I| = 4.0e-03)` against a 1e-3 tolerance. Three things are unknown and each changes
the fix:

  1. WHICH matrix. If it is one shape class the answer is different from all of them.
  2. Whether f64 clears it. If yes the guard is a precision choice, not an impossibility.
  3. What the error is as a function of k = min(m, n), since orthonormality error in a
     backward-stable SVD grows with the dimension it is summing over.

Frames are host-resident by construction for CUDA params (iso.py:82), so this runs the SVDs
on the host and needs no card. Reports per shape class, largest first, and stops early unless
--all is given: one 248320x5120 SVD is ~11 s and there are 20 classes.
"""
import argparse
import os
import time

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl.cli import _build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=6, help="shape classes to test, largest first")
ap.add_argument("--all", action="store_true")
ap.add_argument("--host-only", action="store_true",
                help="skip materialize; the first version of this probe did this by accident "
                     "and reported f32 as passing")
ap.add_argument("--cost", action="store_true",
                help="also time host f32 svd over EVERY shape class, for C in the break-even")
a = ap.parse_args()

cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
from tilerl.model import drop_quantized  # noqa: E402

drop_quantized(model)

# `_build_model` leaves params on the HOST; `build_engine` -> `materialize` is what puts them
# on the card, and that is the state ISO's SVD runs in. Materializing directly rather than
# building an engine, since this probe needs no KV.
if not a.host_only:
    from tilerl_kernels.backend import get_backend

    model.params = get_backend().materialize(model.params)

two_d = {k: v for k, v in model.params.items()
         if v.dim() == 2 and v.dtype in (torch.bfloat16, torch.float32)}
by_shape = {}
for k, v in sorted(two_d.items()):
    by_shape.setdefault(tuple(v.shape), []).append(k)
order = sorted(by_shape, key=lambda s: -min(s))
if not a.all:
    order = order[:a.limit]

print(f"{len(two_d)} master 2D params in {len(by_shape)} shape classes; testing {len(order)}")
print(f"params live on {two_d[next(iter(two_d))].device}\n")
print(f"{'shape':>20} {'k':>7} {'host f32':>10} {'card f32':>10} {'host f64':>10} "
      f"{'h32 s':>7} {'c32 s':>7} {'h64 s':>7}  card passes 1e-3")
worst = (0.0, None)
for shape in order:
    k = min(shape)
    name = by_shape[shape][0]
    p = two_d[name].detach()

    # ISO calls `torch.linalg.svd(p.to(frame_dtype))` on the param WHERE IT LIVES (iso.py:69);
    # the `.cpu()` only happens afterwards, to the frames. So the card's SVD is the one whose
    # orthonormality the guard reads. A host-only probe answers an adjacent question -- the
    # first version of this probe did exactly that and reported f32 as passing.
    def err_on(t):
        u, _, _ = torch.linalg.svd(t, full_matrices=False)
        e = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
        out = float((u.T @ u - e).abs().max())
        del u
        return out

    t = time.perf_counter()
    e32 = err_on(p.cpu().float())
    t32 = time.perf_counter() - t

    t = time.perf_counter()
    e64 = err_on(p.cpu().double())
    t64 = time.perf_counter() - t

    if p.device.type == "cuda":
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            e32c = err_on(p.float())
            torch.cuda.synchronize()
        except Exception as exc:
            e32c = float("nan")
            print(f"  card f32 svd raised {type(exc).__name__}", flush=True)
        tc = time.perf_counter() - t
        torch.cuda.empty_cache()
    else:
        e32c, tc = float("nan"), 0.0

    if e32c > worst[0] or worst[1] is None:
        worst = (e32c, shape)
    print(f"{str(shape):>20} {k:>7} {e32:10.2e} {e32c:10.2e} {e64:10.2e} "
          f"{t32:7.2f} {tc:7.2f} {t64:7.2f}  {e32c <= 1e-3}", flush=True)

print(f"\nworst CARD f32 error {worst[0]:.2e} on {worst[1]}, guard is 1e-3 (iso.py:73)")
print("The arm raised at 4.0e-03. If the card column reproduces that and the host column")
print("does not, the guard is reading cuSOLVER's accuracy at this width, not f32's -- and the")
print("fix is where the SVD runs, not what dtype it runs in.")

# C, the one-time frame cost, over EVERY class rather than the largest few: the break-even
# against ISO's 2.7x step claim is N > C / (85.617 * (1 - 1/2.7)) = C / 53.9 steps, so C
# decides whether host-side frames are affordable inside a 100-step run. Extrapolating from a
# subset is what this probe exists to avoid.
if a.cost:
    print(f"\nC: host f32 svd over all {len(by_shape)} classes")
    print(f"{'shape':>20} {'n':>4} {'per s':>8} {'class s':>10}")
    total = 0.0
    for shape in sorted(by_shape, key=lambda s: -min(s)):
        n = len(by_shape[shape])
        p = two_d[by_shape[shape][0]].detach().cpu().float()
        t = time.perf_counter()
        torch.linalg.svd(p, full_matrices=False)
        per = time.perf_counter() - t
        total += per * n
        print(f"{str(shape):>20} {n:>4} {per:8.2f} {per*n:10.1f}", flush=True)
    print(f"\nC = {total:.0f} s = {total/60:.1f} min over {len(two_d)} matrices")
    print(f"break-even against 2.7x at 85.617 s/step: N > {total/53.9:.0f} steps")
    print(f"break-even against a weaker 2.0x:         N > {total/(85.617*0.5):.0f} steps")
    print("P1 plans 100 steps, so host-side frames are affordable iff the first number < 100.")
