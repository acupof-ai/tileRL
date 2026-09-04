"""Test bf16 linear kernels: backend.linear vs torch reference.

Tests the GEMV path (M=1) and WGMMA path (M=8) that the MTP head uses,
with random bf16 weights. No model load needed — runs in seconds.

Usage:
    PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=7 \
        python3 scripts/debug_bf16_linear.py
"""

from __future__ import annotations

import torch

from tilerl.ops.backend import get_backend


def main() -> None:
    backend = get_backend()
    device = backend.device
    print(f"backend: target={backend.target} device={device}", flush=True)

    # (name, K, N) matching the MTP head's projection shapes.
    shapes = [
        ("fc", 10240, 5120),
        ("q_proj", 5120, 12288),
        ("k_proj", 5120, 1024),
        ("o_proj", 6144, 5120),
        ("gate_proj", 5120, 17408),
        ("down_proj", 17408, 5120),
    ]

    for M in (1, 8):
        print(f"\n=== M={M} ({'GEMV' if M == 1 else 'WGMMA'}) ===", flush=True)
        for name, K, N in shapes:
            torch.manual_seed(42)
            x = torch.randn(M, K, device=device, dtype=torch.float32)
            w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

            y_backend = backend.linear(x, w)
            y_torch = x @ w.float().T

            diff = (y_backend - y_torch).abs()
            rel = diff / (y_torch.abs() + 1e-6)
            print(
                f"{name:12s} K={K:5d} N={N:5d}  "
                f"max_diff={diff.max().item():.6f}  "
                f"mean_diff={diff.mean().item():.6f}  "
                f"max_rel={rel.max().item():.4f}  "
                f"be_norm={y_backend.norm().item():.2f}  "
                f"torch_norm={y_torch.norm().item():.2f}",
                flush=True,
            )
            if diff.max().item() > 0.1:
                print(f"  FAIL: backend[:3]={y_backend[0, :3].tolist()}", flush=True)
                print(f"        torch[:3]={y_torch[0, :3].tolist()}", flush=True)


if __name__ == "__main__":
    main()
