"""benchkit smoke: shipped GDN chunk kernel vs torch-eager at slice4 prefill-512.
    scripts/_pod_bench.sh 'PYTHONPATH=src python3 scripts/bench_smoke.py'
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import benchkit
import torch
from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import gdn_forward

B, TT, QD, NVH, K, V, KER = 1, 512, 2048, 48, 128, 128, 4
VD = NVH * V
QKVD = 2 * QD + VD


def main():
    torch.manual_seed(0)
    q = torch.randn(B, TT, QD, device="cuda") * 0.1
    k = torch.randn(B, TT, QD, device="cuda") * 0.1
    v = torch.randn(B, TT, VD, device="cuda") * 0.1
    z = torch.randn(B, TT, VD, device="cuda") * 0.1
    g = torch.randn(B, TT, NVH, device="cuda")
    beta = torch.randn(B, TT, NVH, device="cuda")
    state = torch.randn(B, NVH, K, V, device="cuda") * 0.01
    window = torch.randn(B, KER - 1, QKVD, device="cuda") * 0.1
    kw = dict(
        conv1d_weight=torch.randn(QKVD, KER, device="cuda") * 0.1,
        dt_bias=torch.randn(NVH, device="cuda"),
        a_log=torch.randn(NVH, device="cuda") * 0.1,
        norm_weight=torch.ones(V, device="cuda"),
        conv_window=window,
        z=z,
    )
    ref = gdn_forward(q, k, v, g, beta, state, **kw)
    print("reference done", flush=True)
    backend = get_backend()

    def arm():
        return backend.linear_attn_chunk(q, k, v, g, beta, state, **kw)

    benchkit.ab("gdn-chunk-smoke", [("gdn_chunk_fused", arm)], ref)


if __name__ == "__main__":
    main()
