"""Feasibility: does T.gemm on f16 shared tiles lower to sm70 mma.m8n8k4?

Tiny square f16 GEMM accumulating f32 on sm70, checked against torch. If this
hits the mma.sync.f32.f16 path and matches, the w4a16 prefill kernel is the
sm90 w4a8 schedule with X/W dequantized to f16 instead of e4m3 and this same
T.gemm. Run on the V100:

  TILERL_TARGET=cuda /usr/bin/python3 scripts/probe_sm70_f16_mma.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src"))

import tilelang  # noqa: E402
import torch  # noqa: E402
from tilelang import language as T  # noqa: E402


def make_f16_gemm(M, N, K, bM, bN, bK):
    @tilelang.jit(target="cuda")
    def gemm(A, B, C):
        M_, N_, K_ = T.const("M, N, K")
        A: T.Tensor((M_, K_), "float16")
        B: T.Tensor((K_, N_), "float16")
        C: T.Tensor((M_, N_), "float32")
        with T.Kernel(T.ceildiv(N_, bN), T.ceildiv(M_, bM), threads=128) as (bx, by):
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bK, bN), "float16")
            Cc = T.alloc_fragment((bM, bN), "float32")
            T.clear(Cc)
            for k in T.Pipelined(K_ // bK, num_stages=2):
                T.copy(A[by * bM, k * bK], As)
                T.copy(B[k * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cc)
            T.copy(Cc, C[by * bM, bx * bN])
        return C

    return gemm


def main():
    M = N = K = 1024
    bM = bN = 64
    bK = 32
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    ref = (a.float() @ b.float())
    fn = make_f16_gemm(M, N, K, bM, bN, bK)
    out = fn(a, b)
    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"sm70 f16 T.gemm -> f32: max abs {err:.4f} rel {rel:.2e} out0 {out[0,0]:.3f}")
    assert rel < 2e-2, rel
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        fn(a, b)
    torch.cuda.synchronize()
    s.record()
    for _ in range(20):
        fn(a, b)
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e) / 20
    tflops = 2 * M * N * K / ms / 1e9
    print(f"1024^3 f16 gemm: {ms:.3f} ms, {tflops:.1f} TFLOP/s")
    print("PROBE_OK")


if __name__ == "__main__":
    main()
