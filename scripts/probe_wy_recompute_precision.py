"""Two separable arms of a C=64 backward recompute, priced before either is written.

The C=64 recompute has two halves that are usually said in one breath:

  arm 1  raise the backward's chunk 16 -> 64, still all-f32 torch. 4x fewer python
         iterations in BOTH loops (`_gdn_chunk_fwd`'s recompute and `_gdn_chunk_bwd`'s
         adjoint), no kernel involved.
  arm 2  additionally take M, h, W and d from `Backend._gdn_wy_core`'s stage outputs, so
         the recompute skips the triangular solve and the state scan.

Arm 2's outputs are bf16 -- `gdn_solve_tril` returns `Ai` bf16 (kernels_gdn.py:220) and
`gdn_state_scan` returns h / V_new bf16 (:368, :370). bf16 carries ~3 decimal digits, so
arm 2 cannot be assumed to land where arm 1 does. This measures both against the same
estimator the C=16 verdict was priced on
(wins/2026-08-29-chunked-gdn-backward.md: gdn_backward at chunk C vs chunk 1, worst relative
error over all eleven gradients, 3 seeds).

Arm 2 here is a SIMULATION, not a kernel measurement: the four tensors the kernels would
supply are rounded to bf16 and back in the f32 reference. It bounds the dtype effect and
nothing else -- tf32 block products inside `gdn_solve_tril` round further, so the real
kernel path is no better than this.

  TILERL_TARGET=cpu python3 scripts/probe_wy_recompute_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src")
)

import torch  # noqa: E402
from tilerl_kernels import reference as ref  # noqa: E402

#: What arm 2 would inherit from the kernels: M <- gdn_solve_tril's bf16 Ai, s <- the bf16
#: entry state h, W/d <- the bf16 w / V_new the state scan wrote.
_BF16_KEYS = ("M", "s", "W", "d")


def _bf16_fwd(real, keys=_BF16_KEYS):
    """`_gdn_chunk_fwd` with arm 2's tensors rounded to bf16, chained state included."""
    def fwd(qc, kc, vc, bc, gtc, s):
        out, s_next, c = real(qc, kc, vc, bc, gtc, s)
        for k in keys:
            c[k] = c[k].bfloat16().float()
        # the scan carries the state in bf16 too, so the chain rounds every chunk
        return out, (s_next.bfloat16().float() if "s" in keys else s_next), c
    return fwd


def _inputs(seed: int):
    torch.manual_seed(seed)
    b, t, nkh, nvh, kd, vd, kern = 1, 128, 2, 4, 16, 16, 4
    rnd = lambda *s: torch.randn(*s, dtype=torch.float32)
    q, k = rnd(b, t, nkh * kd), rnd(b, t, nkh * kd)
    v, z = rnd(b, t, nvh * vd), rnd(b, t, nvh * vd)
    g, beta = rnd(b, t, nvh), rnd(b, t, nvh)
    state, grad = torch.zeros(b, nvh, kd, vd), rnd(b, t, nvh * vd)
    kw = dict(z=z, conv1d_weight=rnd(nkh * kd * 2 + nvh * vd, kern),
              dt_bias=rnd(nvh), a_log=rnd(nvh), norm_weight=rnd(vd))
    return (grad, q, k, v, g, beta, state), kw


def worst_rel(chunk: int, seed: int, bf16: bool, keys=_BF16_KEYS) -> float:
    args, kw = _inputs(seed)
    keep_c, keep_f = ref._GDN_CHUNK, ref._gdn_chunk_fwd
    try:
        ref._GDN_CHUNK = 1  # the serial scan: the reduction order 51e965e measured against
        base = [x.double() for x in ref.gdn_backward(*args, **kw)]
        ref._GDN_CHUNK = chunk
        if bf16:
            ref._gdn_chunk_fwd = _bf16_fwd(keep_f, keys)
        got = ref.gdn_backward(*args, **kw)
    finally:
        ref._GDN_CHUNK, ref._gdn_chunk_fwd = keep_c, keep_f
    return max((a.double() - r).abs().max().item() / max(r.abs().max().item(), 1e-30)
               for a, r in zip(got, base))


def main() -> int:
    seeds = (0, 1, 2)
    arms = (("C=16 f32   (shipped)", 16, False),
            ("C=64 f32   (arm 1)", 64, False),
            ("C=64 bf16  (arm 2)", 64, True))
    print(f"{'arm':22}  {'worst rel over seeds 0/1/2':>26}   vs 1e-4 bar")
    out = {}
    for label, chunk, bf16 in arms:
        e = max(worst_rel(chunk, s, bf16) for s in seeds)
        out[label] = e
        print(f"{label:22}  {e:26.2e}   {'PASS' if e < 1e-4 else 'FAIL'} ({1e-4 / e:.0f}x)")

    e16, e64, ebf = (out[a[0]] for a in arms)
    # a control: without it "arm 2 is worse" could be the estimator, not the dtype
    assert e64 > e16, f"C=64 ({e64:.2e}) is not coarser than C=16 ({e16:.2e})"
    assert ebf > e64, (
        f"bf16 stage outputs ({ebf:.2e}) are not coarser than f32 at the same chunk "
        f"({e64:.2e}) -- the simulation is not rounding anything")
    print()
    print("08-29's table, same estimator, for reference: C=16 4-12e-7, C=64 1.9-4.9e-6.")
    print(f"arm 2 / arm 1 = {ebf / e64:.0f}x, which is the bf16 cost alone at equal chunk.")

    # Which tensor carries it: a reduced arm 2 is worth writing only if some subset passes.
    # One key at a time, because the four are independent kernel outputs and a joint number
    # cannot say which one to keep in f32.
    print()
    print("per-tensor, C=64, one key in bf16 at a time (bar 1e-4):")
    for k in _BF16_KEYS:
        e = max(worst_rel(64, s, True, (k,)) for s in seeds)
        print(f"  {k:4} bf16{'':11}  {e:26.2e}   {'PASS' if e < 1e-4 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
