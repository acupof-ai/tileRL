"""Measure the fp4 GEMV gap: direct kernel vs through-backend, SAME shapes.

The final bench saw 31% roof through the backend on per-layer shapes vs 46%
for the direct kernel — but on different shapes. This runs both on the same
per-layer shapes in one process to decide whether the gap is backend fat
(recoverable -> ~73 tok/s decode) or shape amortization (single-stream is
capped ~55).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src TILERL_TARGET=cuda \\
        python3 scripts/bench_gemv_gap.py /host/tc27-nvfp4-slice4 --layers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, "scripts")
import bench_fp4_gemv as bg  # noqa: E402 — _measure_bw_gbs, _time_calls

from tilerl.config import qwen36_27b  # noqa: E402
from tilerl.model import fp4_param_keys, load_hf  # noqa: E402
from tilerl.ops import kernels_mma  # noqa: E402
from tilerl.ops.backend import _round_up, get_backend  # noqa: E402
from tilerl.ops.reference import linear_fp4  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    args = p.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = replace(
        qwen36_27b(),
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in qwen36_27b().full_attn_layers if i < args.layers),
    )
    model = load_hf(cfg, args.source)
    model.params = {k: v.to(backend.device) for k, v in model.params.items()}
    direct = kernels_mma.make_linear_fp4_gemv(backend.target)

    bw = bg._measure_bw_gbs()
    print(f"\n=== fp4 GEMV: backend vs direct, same shapes (BW {bw:.1f} GB/s) ===")
    print(
        f"  {'shape (N,K)':<22} {'bytes MB':>9} {'roof ms':>8} "
        f"{'backend ms':>10} {'direct ms':>9} {'backend %':>9} {'direct %':>9} {'ratio':>6}"
    )
    keys = sorted(k for k in fp4_param_keys(cfg) if k + ".wq" in model.params)
    for key in keys:
        wq = model.params[key + ".wq"]
        scale = model.params[key + ".scale"]
        N, K = wq.shape[0], wq.shape[1] * 2
        Kp, Np = _round_up(K, 256), _round_up(N, 4)
        x = torch.randn(1, K, device=backend.device, dtype=torch.bfloat16)
        # Identical args to what backend.linear_fp4 passes the kernel.
        xp = torch.nn.functional.pad(x, (0, Kp - K))
        wqp = torch.nn.functional.pad(wq, (0, Kp // 2 - wq.shape[1], 0, Np - N))
        sp = torch.nn.functional.pad(scale, (0, Kp // 32 - scale.shape[1], 0, Np - N))

        yb = backend.linear_fp4(x, wq, scale)
        yd = direct(xp, wqp, sp, 32, 4)[:1, :N]
        ref = linear_fp4(x.cpu(), wq.cpu(), scale.cpu())
        assert (yb.cpu() - ref).abs().max() < 1e-2 * ref.abs().max() + 1e-3, (
            f"{key}: backend parity"
        )
        assert (yd.cpu() - ref).abs().max() < 1e-2 * ref.abs().max() + 1e-3, f"{key}: direct parity"

        for _ in range(5):
            backend.linear_fp4(x, wq, scale)
            direct(xp, wqp, sp, 32, 4)
        backend_ms = bg._time_calls(lambda: backend.linear_fp4(x, wq, scale), 50)
        direct_ms = bg._time_calls(lambda: direct(xp, wqp, sp, 32, 4), 50)

        bytes_ = N * K * 0.75 + 2 * K
        roof_ms = bytes_ / (bw * 1e9) * 1e3
        print(
            f"  {f'{N},{K}':<22} {bytes_ / 2**20:>9.2f} {roof_ms:>8.4f} "
            f"{backend_ms:>10.4f} {direct_ms:>9.4f} "
            f"{100 * roof_ms / backend_ms:>8.1f}% {100 * roof_ms / direct_ms:>8.1f}% "
            f"{backend_ms / direct_ms:>5.2f}x"
        )


if __name__ == "__main__":
    main()
