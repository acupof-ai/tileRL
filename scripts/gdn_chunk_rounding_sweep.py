"""Measure which GDN chunk-size pairs push the state out of parity tolerance.

The chunkwise-WY parallel scan rounds differently per chunk length. The consumer
is the e2e prefix-store test, which compares two engines' states with
``torch.allclose(rtol=1e-2, atol=1e-5)`` (tests/test_e2e.py). On CUDA a 64-vs-128
chunk difference broke that gate on a near-zero element (ratio ~2.0,
errors/2026-09-10-gdn-state-chunk-size-rounding). This script measures, for every
chunk-size pair, the worst per-element violation ratio
``|delta| / (atol + rtol * |ref|)`` over a seq_len grid: ratio > 1 means the pair
breaks parity. The bound it establishes is watched by test_gdn_chunk_rounding_bound.

Dev-only: no runtime path imports this. TILERL_TARGET=metal|cpu picks the torch
device; the reference is pure float32 torch, so the rounding magnitude is
device-dependent and the bound must hold on the CI target (cpu).
"""

import os
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "packages", "tilerl-kernels", "src")
)
from tilerl_kernels import reference  # noqa: E402

# the consumer's tolerance: torch.allclose(rtol=1e-2, atol=1e-5) in tests/test_e2e.py
RTOL, ATOL = 1e-2, 1e-5


def _inputs(b, t, nkh, nvh, kd, vd, ker, seed, scale=10.0):
    torch.manual_seed(seed)
    sc = 0.1 * scale
    qkv = 2 * nkh * kd + nvh * vd
    q = torch.randn(b, t, nkh * kd) * sc
    k = torch.randn(b, t, nkh * kd) * sc
    v = torch.randn(b, t, nvh * vd) * sc
    g = torch.randn(b, t, nvh)
    beta = torch.randn(b, t, nvh)
    z = torch.randn(b, t, nvh * vd) * sc
    state = torch.randn(b, nvh, kd, vd) * 0.1 * sc
    window = torch.randn(b, ker - 1, qkv) * sc
    kw = dict(
        conv1d_weight=torch.randn(qkv, ker) * 0.1,
        dt_bias=torch.randn(nvh),
        a_log=torch.randn(nvh) * 0.1,
        norm_weight=torch.ones(vd),
        conv_window=window,
    )
    return q, k, v, g, beta, z, state, kw


def _state(t, chunk, q, k, v, g, beta, z, state, kw):
    """End-of-span state after t tokens, cut into chunks of `chunk` (0 = serial)."""
    _, s, _ = reference.gdn_forward(
        q[:, :t],
        k[:, :t],
        v[:, :t],
        g[:, :t],
        beta[:, :t],
        state,
        z=z[:, :t],
        chunkwise=chunk,
        **kw,
    )
    return s


def _worst_ratio(sa, sb):
    """Worst per-element parity-violation ratio and the element that produced it."""
    tol = ATOL + RTOL * sb.abs()
    ratio = (sa - sb).abs() / tol
    flat = ratio.flatten()
    idx = int(flat.argmax().item())
    return (
        float(flat[idx].item()),
        float(sb.flatten()[idx].item()),
        float((sa - sb).abs().flatten()[idx].item()),
        float(tol.flatten()[idx].item()),
    )


def main():
    target = os.environ.get("TILERL_TARGET", "metal")
    dev = "cpu"
    if target == "metal" and torch.backends.mps.is_available():
        dev = "mps"
    print(f"device: {dev}  (parity tolerance rtol={RTOL}, atol={ATOL})")

    b, nkh, nvh, kd, vd, ker, seed = 2, 2, 6, 16, 16, 4, 31
    t_max = 256
    q, k, v, g, beta, z, state, kw = _inputs(b, t_max, nkh, nvh, kd, vd, ker, seed)
    q, k, v, g, beta, z, state = (x.to(dev) for x in (q, k, v, g, beta, z, state))
    kw = {kk: vv.to(dev) for kk, vv in kw.items()}

    chunks = [16, 32, 64, 128]
    seqs = [64, 100, 128, 164, 256]

    # states[t][chunk]
    states = {t: {c: _state(t, c, q, k, v, g, beta, z, state, kw) for c in chunks} for t in seqs}

    print("\nchunk-vs-chunk: worst parity-violation ratio over seq_lens (ratio > 1 breaks parity)")
    header = "      " + "".join(f"{f'{cB:>4}':>12}" for cB in chunks)
    print(header)
    worst_overall = 0.0
    worst_detail = None
    for cA in chunks:
        cells = []
        for cB in chunks:
            if cA == cB:
                cells.append(f"{'-':>12}")
                continue
            r_max, ref_v, delta, tol = 0.0, 0.0, 0.0, 0.0
            for t in seqs:
                r, rv, d, tv = _worst_ratio(states[t][cA], states[t][cB])
                if r > r_max:
                    r_max, ref_v, delta, tol = r, rv, d, tv
            if r_max > worst_overall:
                worst_overall = r_max
                worst_detail = (cA, cB, ref_v, delta, tol)
            cells.append(f"{r_max:>12.3e}")
        print(f"{cA:>4}  " + "".join(cells))

    cA, cB, ref_v, delta, tol = worst_detail
    print(
        f"\nworst: c={cA} vs c={cB}  ratio={worst_overall:.3e}  "
        f"|ref|={abs(ref_v):.3e}  delta={delta:.3e}  tol={tol:.3e}"
    )

    print("\nchunkwise vs serial: worst parity-violation ratio over seq_lens")
    for c in chunks:
        r_max = 0.0
        for t in seqs:
            s_ref = _state(t, 0, q, k, v, g, beta, z, state, kw)
            r, *_ = _worst_ratio(states[t][c], s_ref)
            r_max = max(r_max, r)
        print(f"  c={c:>3} vs serial: {r_max:.3e}")


if __name__ == "__main__":
    main()
