"""Micro-benchmark the sm70 fp4 GEMV alone under ncu — no full model, no engine.
Runs the kernel on the 27B's real GEMV shapes so ncu can read occupancy / DRAM
throughput / issue efficiency on the one kernel that is 86.7% of the tick.

  # timing (no ncu):
  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70e \
    TILERL_TARGET=cuda PYTHONPATH=packages/tilerl-kernels/src:src \
    python scripts/_gemv_micro.py
  # under ncu (profiles only the marked region):
  ncu --profile-from-start off --set full --target-processes all \
    -o /tmp/gemv_ncu python scripts/_gemv_micro.py
"""
import sys
import time

import torch

from tilerl_kernels import kernels_linear, reference

RT, NP = 32, 4  # reduce_thread, n_partition — the backend's gemv plan


def build(N, K, block=32):
    # Random packed bytes — speed benchmark only, no accuracy check, so skip
    # pack_fp4 (its argmin OOMs on lm_head-sized N). Layout matches the kernel.
    dev = "cuda"
    wq = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    sc = torch.rand(N, K // block, dtype=torch.float32, device=dev) + 0.5
    osc = torch.ones(N, dtype=torch.float32, device=dev)
    x = torch.randn(1, K, dtype=torch.float32, device=dev) * 0.5
    return (x, wq, sc, osc, torch.zeros(1, N, dtype=torch.float32, device=dev))


def main():
    k = kernels_linear.make_linear_fp4_gemv_sm70("cuda")
    # 27B decode GEMV shapes (N, K): qkv/o/gate_up/down/lm_head
    shapes = [(5120, 5120), (17408, 5120), (5120, 17408), (248320, 5120)]
    args = {s: build(*s) for s in shapes}
    # warm compile
    for s, (x, wq, sc, osc, res) in args.items():
        k(x, wq, sc, osc, res, RT, NP, 32)
    torch.cuda.synchronize()

    ncu = "--profile-from-start" in " ".join(sys.argv) or __import__("os").environ.get("NCU")
    for s, (x, wq, sc, osc, res) in args.items():
        N, K = s
        bytes_w = N * K // 2 + N * (K // 32) * 4 + N * 4  # wq + scale + oscale
        iters = 50
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            k(x, wq, sc, osc, res, RT, NP, 32)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        gbs = bytes_w / (us * 1e-6) / 1e9
        print(f"N={N:6d} K={K:5d}: {us:8.1f} us  {gbs:6.1f} GB/s  ({100*gbs/900:.0f}% roofline)",
              flush=True)
    print("MICRO OK", flush=True)


if __name__ == "__main__":
    torch.cuda.profiler.start() if __import__("os").environ.get("NCU") else None
    main()
    torch.cuda.profiler.stop() if __import__("os").environ.get("NCU") else None
