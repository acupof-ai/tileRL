"""A/B native-fp8 in_proj_qkv+z fusion at prefill shapes (H20, sm90).

Arm A: two linear_fp8 launches (qkv then z, same input) — the shipped GDN
forward. Arm B: one linear_fp8 on the concat'd w8/wscale + the split back at
the qkv boundary. Same math (the per-128-block wscale concats losslessly:
10240 = 80 blocks, 6144 = 48), so relerr ~1e-6; the question is launch count.

    REMOTE_DIR=/work/tilerl_qkvz BENCH_GPUS=7 scripts/_pod_bench.sh \\
        'PYTHONPATH=src python3 scripts/bench_fp8_qkvz.py'
"""

from __future__ import annotations

import torch

import benchkit
from tilerl.ops.backend import get_backend


def _quant_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-128-block e4m3 weight quant (the checkpoint format load_hf keeps
    native): w8 [N,K] f8_e4m3, wscale [ceil(N/128), ceil(K/128)] f32."""
    n, k = w.shape
    ns, ks = (n + 127) // 128, (k + 127) // 128
    wp = w.float().new_zeros((ns * 128, ks * 128))
    wp[:n, :k] = w.float()
    scale = (
        wp.reshape(ns, 128, ks, 128).permute(0, 2, 1, 3).reshape(ns, ks, -1).abs().amax(-1) / 448.0
    ).clamp_min(1e-12)
    w8 = (wp / scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).clamp(-448, 448)
    return w8[:n, :k].to(torch.float8_e4m3fn), scale.contiguous()


def main() -> None:
    backend = get_backend()
    torch.manual_seed(0)
    p = benchkit.GDN_PREFILL
    M = p["B"] * p["T"]  # 512
    K = 2048  # hidden size of the bench input
    qd = kd = p["QD"]  # 2048
    vd = p["nvh"] * p["V"]  # 6144
    n_qkv, n_z = qd + kd + vd, vd  # 10240, 6144

    x = backend._dev(torch.randn(M, K) * 0.5, torch.bfloat16)
    w8_qkv, ws_qkv = _quant_fp8_block(torch.randn(n_qkv, K) * 0.1)
    w8_z, ws_z = _quant_fp8_block(torch.randn(n_z, K) * 0.1)
    w8_qkv, ws_qkv = backend._dev(w8_qkv, w8_qkv.dtype), backend._dev(ws_qkv, ws_qkv.dtype)
    w8_z, ws_z = backend._dev(w8_z, w8_z.dtype), backend._dev(ws_z, ws_z.dtype)
    w8_qkvz = torch.cat([w8_qkv, w8_z], dim=0).contiguous()
    ws_qkvz = torch.cat([ws_qkv, ws_z], dim=0).contiguous()

    def arm_a() -> tuple[torch.Tensor, torch.Tensor]:
        qkv = backend.linear_fp8(x, w8_qkv, ws_qkv)
        z = backend.linear_fp8(x, w8_z, ws_z)
        return qkv, z

    def arm_b() -> tuple[torch.Tensor, torch.Tensor]:
        qkvz = backend.linear_fp8(x, w8_qkvz, ws_qkvz)
        return qkvz[:, :n_qkv], qkvz[:, n_qkv:]

    ref = arm_a()
    torch.cuda.synchronize()
    benchkit.ab(
        f"fp8 qkvz fusion prefill (M={M}, K={K}, N={n_qkv}+{n_z})",
        [("two-launch", arm_a), ("fused", arm_b)],
        ref,
    )


if __name__ == "__main__":
    main()
