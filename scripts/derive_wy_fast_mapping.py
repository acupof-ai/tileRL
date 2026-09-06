"""Does upstream's (W, U) equal our (W, U)? Derive the mapping, then assert it.

The wy_fast port's parity gate needs a reference, and `_gdn_chunk_fwd`'s cache is 16 tensors
in our layout while upstream returns 2. This script answers what the correspondence is, in
torch, on CPU, with no kernel and no card -- so the answer is a definition rather than a
measurement.

Read off `examples/gdn/example_wy_fast.py`'s kernel body (:107-129), which is the only
definition in that checkout -- its torch side calls `fla.ops.gated_delta_rule.wy_fast`, which
is not vendored here:

    U = A @ (V * Beta)                      # :113-117
    W = A @ (K * Beta * exp(G))             # :123-127

and ours, `reference.py:613-615`:

    bV  = bp * v                            # bp = beta broadcast
    beK = (beta * e) * k                    # e = exp(cumsum(gt)) within the chunk
    U, W = M @ bV, M @ beK

So the two agree iff A == M, upstream's Beta is our post-sigmoid beta, and upstream's G is
already the chunk-local cumsum. The third holds in the pipeline: `example_wy_fast.py` itself
does NOT cumsum (it takes G raw), but `example_chunk_delta_bwd.py:55` runs
`chunk_local_cumsum(G, chunk_size)` before its kernels, so G is cumsummed upstream of both.

  python3 scripts/derive_wy_fast_mapping.py
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


def upstream_w_u(K, V, Beta, G, A):
    """`example_wy_fast.py`'s kernel, in torch. Shapes [B,S,H,*], A is [B,S,H,BS]."""
    b = Beta.unsqueeze(-1)
    eg = torch.exp(G).unsqueeze(-1)
    # A @ x contracts A's last dim (the in-chunk index) with x's S -- per (batch, head)
    Am = A.permute(0, 2, 1, 3)                    # [B,H,S,BS]
    U = Am @ (V * b).permute(0, 2, 1, 3)          # [B,H,S,DV]
    W = Am @ (K * b * eg).permute(0, 2, 1, 3)     # [B,H,S,DK]
    return W, U


def _controls(k, v, beta, A, ours_W, ours_U) -> None:
    """Each fold gets its own arm: an allclose that passes says nothing until the mutation
    that should break it does. Dropping exp(G) from W moved max|diff| 3.0e-08 -> 6.5e-01;
    dropping beta from U moved it 0.0 -> 2.6e+00. W and U fold beta separately, so one arm
    cannot cover both."""
    b = beta.unsqueeze(-1)
    no_gate = A @ (k * b).permute(0, 2, 1, 3)
    no_beta = A @ v.permute(0, 2, 1, 3)
    assert not torch.allclose(ours_W, no_gate, rtol=1e-5, atol=1e-6), (
        "W matches without the gate fold -- the check cannot see exp(G)")
    assert not torch.allclose(ours_U, no_beta, rtol=1e-5, atol=1e-6), (
        "U matches without the beta fold -- the check cannot see beta")


def main() -> int:
    torch.manual_seed(0)
    B, n, HV, DK, DV = 2, ref._GDN_CHUNK, 3, 8, 8
    q = torch.randn(B, n, HV, DK, dtype=torch.float32)
    k = torch.nn.functional.normalize(torch.randn(B, n, HV, DK), dim=-1)
    v = torch.randn(B, n, HV, DV, dtype=torch.float32)
    beta = torch.rand(B, n, HV, dtype=torch.float32)          # already post-sigmoid, in (0,1)
    gt = -torch.rand(B, n, HV, dtype=torch.float32)           # log-space, negative
    s = torch.randn(B, HV, DK, DV, dtype=torch.float32)

    _, _, c = ref._gdn_chunk_fwd(q, k, v, beta, gt, s)
    ours_W = c["W"]                                            # [B,HV,n,DK]
    ours_U = c["M"] @ c["bV"]                                   # U is not cached by name
    A, e = c["M"], c["e"]

    # A is ours by construction here; the point of the check is the beta/gate folding.
    G = torch.log(e)                                           # e = exp(cumsum(gt)) -> G = cumsum
    up_W, up_U = upstream_w_u(k, v, beta, G, A.permute(0, 2, 1, 3))

    for name, mine, theirs in (("W", ours_W, up_W), ("U", ours_U, up_U)):
        ok = torch.allclose(mine, theirs, rtol=1e-5, atol=1e-6)
        d = (mine - theirs).abs().max().item()
        print(f"{name}: allclose={ok}  max|diff|={d:.3e}  shape={tuple(mine.shape)}")
        assert ok, f"{name} does not match: max|diff|={d:.3e}"

    _controls(k, v, beta, A, ours_W, ours_U)

    # The transform, stated once the numbers agree.
    print()
    print("mapping (upstream <- ours), verified numerically above:")
    print("  A     <- M           = (I + tril(beta_i <k_i,k_j> D_ij, -1))^-1, reference.py:611")
    print("  Beta  <- bt          = sigmoid(beta), POST-sigmoid (reference.py:913)")
    print("  G     <- log(e)      = cumsum(gt) within the chunk; upstream cumsums via")
    print("                         chunk_local_cumsum before its kernels, not inside wy_fast")
    print("  K     <- kn          = L2-normalized k, NOT scaled by 1/sqrt(DK) (that is q's)")
    print("  W     -> c['W']      = M @ ((beta*e) * k)")
    print("  U     -> M @ c['bV'] = M @ (beta * v)   -- not cached under its own name")
    print()
    print("NOT established here: that A == M for upstream's own A. A is an INPUT to wy_fast")
    print("(prepare_input:30), produced by a solve upstream of it, so the wy_fast port's gate")
    print("compares W/U given the same A -- the M solve itself is a separate kernel's parity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
