"""Minimum known-answer cases for the ported state scan, to locate the NaN.

The full-shape check returned NaN with the right timing (34.5 us against fla's
34.0), so the kernel runs and the arithmetic is wrong. These cases have答案 by
inspection: with W=0 the recurrence is S' = e_last*S and V_new = U.
"""

from __future__ import annotations

import torch

from tilerl_kernels import kernels_gdn
from tilerl_kernels.backend import get_backend


def main() -> None:
    b = get_backend()
    dev = b.device
    torch.manual_seed(0)
    B, S, H, DK, C = 1, 64, 1, 128, 64  # ONE chunk: no cross-chunk carry at all
    kern = kernels_gdn.make_gdn_state_scan(b.target, block_DV=32)

    z = lambda *sh, dt=torch.bfloat16: torch.zeros(*sh, device=dev, dtype=dt)
    k, w, u = z(B, S, H, DK), z(B, S, H, DK), z(B, S, H, DK)
    g = torch.zeros(B, S, H, device=dev)
    st = torch.zeros(B, H, DK, DK, device=dev)
    vn, out = kern(k, w, u, g, st, C)
    print(f"all zero     -> vnew finite {bool(torch.isfinite(vn).all())} "
          f"state finite {bool(torch.isfinite(out).all())} "
          f"|vnew| {vn.float().abs().max():.3f} |state| {out.abs().max():.3f}")

    # W=0, K=0, U=1, S=1, g=0 -> V_new = U = 1 ; S' = exp2(0)*S = 1
    u1 = torch.ones(B, S, H, DK, device=dev, dtype=torch.bfloat16)
    st1 = torch.ones(B, H, DK, DK, device=dev)
    vn, out = kern(k, w, u1, g, st1, C)
    print(f"W=K=0,U=1,S=1-> vnew[:4] {vn[0, 0, 0, :4].float().tolist()} "
          f"state[:4] {out[0, 0, 0, :4].tolist()}   (both should be 1.0)")

    # add K=1: S' = S + K^T V_new = 1 + 64*1 = 65 per entry
    k1 = torch.ones(B, S, H, DK, device=dev, dtype=torch.bfloat16)
    vn, out = kern(k1, w, u1, g, st1, C)
    print(f"K=1          -> state[:4] {out[0, 0, 0, :4].tolist()}   (should be 65.0)")

    # G must be the chunk-local INCLUSIVE CUMSUM — a constant g is not a legal
    # input, and feeding one is what made the first version of this probe read
    # 64 / 23.5 / 3.18 where 1 / 0.5 / 0.25 was expected.
    # With K=U=0 the gemm contributes nothing, so state = exp2(G[last]) * S.
    for gv in (0.0, -0.25, -0.5):
        gg = torch.full((B, S, H), gv, device=dev).cumsum(1)  # legal: cumsum
        glast = float(gg[0, C - 1, 0])
        want = 2.0 ** glast
        _, out = kern(k, w, u, gg, st1, C)
        print(f"per-step g={gv:+.2f} (G_last={glast:+.1f}) -> state[0] "
              f"{out[0, 0, 0, 0]:.4g}   (should be {want:.4g})")


if __name__ == "__main__":
    main()
