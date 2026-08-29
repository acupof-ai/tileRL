"""Bench paged_attention on the H20 pod: prefill (M=512) and decode (M=1)
at the 27B full-attn shapes (H=24, Hkv=4, D=256, block=16) vs roofline.

Kernel-level: calls the kernel factory directly (naive f32 vs sm90 MMA),
not the backend method, so the numbers are raw kernel time.

Usage:
    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src TILERL_TARGET=cuda \\
        python3 scripts/bench_paged_attn.py [--kernel naive|mma]
"""

from __future__ import annotations

import argparse

import torch

from tilerl_kernels import kernels, kernels_mma

#: 27B full-attn layer geometry (config.qwen36_27b).
H, HKV, D, BLOCK = 24, 4, 256, 16
#: H20 dense tensor-core throughput (docs/experience/wins/2026-08-24-fp8-prefill-wgmma.md).
BF16_TFLOPS = 148.0


def measure_bw_gbs() -> float:
    """Achievable HBM BW from a device-to-device copy (read+write = 2N bytes)."""
    n = 256 * 2**20 // 4
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    iters = 50
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        dst.copy_(src)
    e.record()
    torch.cuda.synchronize()
    return 2 * n * 4 / (s.elapsed_time(e) / iters / 1e3) / 1e9


def time_calls(fn, iters: int) -> float:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def make_kv(kv_len: int, dtype: torch.dtype):
    nb = (kv_len + BLOCK - 1) // BLOCK
    pool = torch.randn(nb, HKV, BLOCK, D, device="cuda", dtype=dtype)
    bt = torch.arange(nb, dtype=torch.int32, device="cuda").unsqueeze(0)
    return pool, bt


def bench(kernel: str, bw_gbs: float) -> None:
    target = "cuda"
    if kernel == "naive":
        make = kernels.make_paged_attention
        dtype = torch.float32
    else:
        make = kernels_mma.make_paged_attention_mma
        dtype = torch.bfloat16
    k = make(target)
    scale = 1.0 / (D**0.5)
    print(f"=== paged_attention ({kernel}) vs roofline, BW {bw_gbs:.0f} GB/s ===")
    print(f"  {'case':<22} {'ms':>10} {'roof ms':>10} {'%roof':>8}")

    def run_mma(q, kc, vc, bt, sl, block_m=None):
        s = q.shape[1]
        if block_m is None:
            block_m = 64 if s >= 64 else 16
        pad = -s % block_m
        if pad:
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
        k(q, kc, vc, bt, sl, scale, BLOCK, block_m, s, 128)

    # decode: M=1, KV=4096 (memory-bound: K+V stream, f32 naive / bf16 mma)
    kv_len = 4096
    kc, bt = make_kv(kv_len, dtype)
    vc = kc.clone()
    q = torch.randn(1, 1, H, D, device="cuda", dtype=dtype)
    sl = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
    if kernel == "mma":
        run = lambda: run_mma(q, kc, vc, bt, sl)  # noqa: E731
        run()  # JIT warmup
        ms = time_calls(run, 50)
        bytes_kv = 2 * HKV * kv_len * D * 2
        roof = bytes_kv / (bw_gbs * 1e9) * 1e3
        print(f"  {'decode M=1 KV=4096':<22} {ms:>10.4f} {roof:>10.4f} {roof / ms * 100:>7.1f}%")
    else:
        run = lambda: k(q, kc, vc, bt, sl, scale, block_size=BLOCK, threads=64)  # noqa: E731
        run()  # JIT warmup (30-120s on first compile)
        ms = time_calls(run, 50)
        bytes_kv = 2 * HKV * kv_len * D * 4
        roof = bytes_kv / (bw_gbs * 1e9) * 1e3
        print(f"  {'decode M=1 KV=4096':<22} {ms:>10.4f} {roof:>10.4f} {roof / ms * 100:>7.1f}%")
    # prefill: M=512 causal chunk (compute/tiling-bound)
    m = 512
    kc, bt = make_kv(m, dtype)
    vc = kc.clone()
    q = torch.randn(1, m, H, D, device="cuda", dtype=dtype)
    sl = torch.tensor([m], dtype=torch.int32, device="cuda")
    if kernel == "mma":
        run = lambda: run_mma(q, kc, vc, bt, sl)  # noqa: E731
    else:
        run = lambda: k(q, kc, vc, bt, sl, scale, block_size=BLOCK, threads=64)  # noqa: E731
    run()  # JIT warmup
    ms = time_calls(run, 20)
    flops = 2 * H * (m * (m + 1) // 2) * D * 2  # QK + PV, causal
    roof = flops / (BF16_TFLOPS * 1e12) * 1e3
    print(f"  {'prefill M=512 KV=512':<22} {ms:>10.4f} {roof:>10.4f} {roof / ms * 100:>7.1f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=["naive", "mma"], default="naive")
    args = p.parse_args()
    bw = measure_bw_gbs()
    bench(args.kernel, bw)


if __name__ == "__main__":
    main()
