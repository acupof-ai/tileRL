"""Isolated prefill bench for the GDN projections: fp4 path (pack the master,
linear_fp4) vs native fp8 (linear_fp8), same process, back-to-back — the ratio
is contention-independent (both arms see the same GPU phase).

Usage:
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=3 \\
        PYTHONPATH=src python3 scripts/bench_fp8_gdn.py /host/tc27-nvfp4-slice4
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from tilerl.config import qwen36_27b
from tilerl.model import load_hf
from tilerl.ops.backend import get_backend
from tilerl.ops.reference import pack_fp4


def _time(fn, iters=20):
    fn()  # warmup
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--m", type=int, default=512)
    args = p.parse_args()

    backend = get_backend()
    cfg = qwen36_27b()
    cfg = replace(
        cfg,
        num_layers=args.layers,
        full_attn_layers=tuple(i for i in cfg.full_attn_layers if i < args.layers),
    )
    model = load_hf(cfg, args.source, keep_master=True)  # fp4-repacks the masters
    keys = [
        f"layers.{i}.{k}"
        for i in range(args.layers)
        for k in ("in_proj_qkv", "in_proj_z", "out_proj")
    ]
    keys = [k for k in keys if k + ".w8" in model.params]
    print(f"GDN fp8 linears: {len(keys)}", flush=True)

    total_flop = 0.0
    fp4_ms = fp8_ms = 0.0
    for key in keys:
        master = model.params[key]
        w8 = model.params[key + ".w8"]
        wscale = model.params[key + ".wscale"]
        n, k = master.shape
        wq, scale = pack_fp4(master)
        x = torch.randn(args.m, k, device=backend.device, dtype=torch.bfloat16) * 0.5
        # migrate params
        w8 = w8.to(backend.device)
        wscale = wscale.to(backend.device)
        wq = wq.to(backend.device)
        scale = scale.to(backend.device)
        master = master.to(backend.device)
        flop = 2 * args.m * n * k
        total_flop += flop
        t4 = _time(lambda: backend.linear_fp4(x, wq, scale, master=master))
        t8 = _time(lambda: backend.linear_fp8(x, w8, wscale, master=master))
        fp4_ms += t4
        fp8_ms += t8
        print(
            f"  {key.split('.')[-1]:<14} {n}x{k:<6} fp4 {t4:7.3f} ms ({flop / t4 / 1e9:6.1f} TFLOPS)  "
            f"fp8 {t8:7.3f} ms ({flop / t8 / 1e9:6.1f} TFLOPS)  speedup {t4 / t8:.2f}x",
            flush=True,
        )
    print(
        f"  {'TOTAL':<14} {'':<13} fp4 {fp4_ms:7.3f} ms ({total_flop / fp4_ms / 1e9:6.1f} TFLOPS)  "
        f"fp8 {fp8_ms:7.3f} ms ({total_flop / fp8_ms / 1e9:6.1f} TFLOPS)  speedup {fp4_ms / fp8_ms:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
