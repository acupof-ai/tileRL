"""fp8 prefill GEMM sweep: dequant / tile / K-split variants of
make_linear_fp4_fp8_mma.

Diagnostic only — not shipped. Each variant is a standalone tilelang kernel;
the winner lands in src/tilerl/ops/kernels_mma.py (make_linear_fp4_fp8_mma).

Hypothesis: the shipped kernel's speedup vs bf16 is monotonic in N (grid
blocks): 1.46x @ 544 blocks (2.3 waves) -> 1.01x @ 160 blocks (0.68 waves).
Warp specialization is off, so dequant and WGMMA share threads; with <1 wave
the SMs stall when resident blocks align in dequant. Variants attack the
dequant cost (v_int32: integer-only e2m1->e4m3, scale on the accumulator),
the hiding (v_ws), and the grid size (v_split2, v_m64).

Variants:
  baseline  shipped kernel (128x128, block_K=64, stages=3, 128 threads,
            scale-in-dequant)
  v_int32   block_K=32, integer e2m1fn->e4m3 dequant (exact subset, no FP,
            no requant), per-N-row scale on the accumulator per K-tile
            (deepgemm 2xAcc)
  v_ws      baseline + warp specialization enabled
  v_split2  baseline + 2-way K-split, f32 atomic add into a zeroed output
  v_sota    SOTA example defaults: 256x128, block_K=128, stages=2, 256 threads
  v_m64     baseline with block_M=64 (2x M-tiles -> 2x blocks)

Usage (pod):
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=N PYTHONPATH=src \\
        python3 scripts/_sweep_fp8_prefill.py [baseline v_int32 ...]
"""

from __future__ import annotations

import sys
import time

import torch
import tilelang
import tilelang.language as T

sys.path.insert(0, "src")
from tilerl_kernels.kernels_mma import make_quant_fp8_e4m3
from tilerl_kernels.reference import linear_fp4, pack_fp4

SHAPES = [
    (512, 5120, 17408),  # gate/up
    (512, 17408, 5120),  # down
    (512, 5120, 10240),  # in_proj_qkv
    (512, 5120, 6144),  # in_proj_z
    (512, 6144, 5120),  # out_proj
]


def _pass_configs(ws_disabled: bool = True):
    cfg = {"tl.disable_data_race_check": True}
    if ws_disabled:
        cfg[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True
    return cfg


# ---------------------------------------------------------- dequant macros


def _dequant_fp4_macro(out_dtype, local_size):
    """Shipped dequant: packed WQ tile -> dequant W tile in shared, one
    per-32-block scale per chunk (copied verbatim from kernels_mma.py)."""
    local_compress = local_size // 2

    @T.macro
    def dequant(WQ_shared, Scale_shared, W_shared, block_N, block_K):
        for i in T.Parallel(block_N * block_K // local_size):
            WQ_local = T.alloc_local((local_compress,), "uint8")
            W_local = T.alloc_local((local_size,), out_dtype)
            cbase = i * local_compress
            nbase = i * local_size
            for v in T.vectorized(local_compress):
                WQ_local[v] = WQ_shared[(cbase + v) // (block_K // 2), (cbase + v) % (block_K // 2)]
            s = Scale_shared[nbase // block_K, (nbase % block_K) // 32]
            for v in T.serial(local_size):
                byte = WQ_local[v // 2]
                nib = (byte >> ((v % 2) * 4)) & 15
                ni32 = T.cast(nib, "int32")
                bits = ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
                w = T.reinterpret(bits, "float32")
                W_local[v] = T.cast(w * s, out_dtype)
            for v in T.vectorized(local_size):
                W_shared[(nbase + v) // block_K, (nbase + v) % block_K] = W_local[v]

    return dequant


def _dequant_fp4_int_macro(block_N, block_K):
    """Integer e2m1fn -> e4m3 dequant: the e2m1fn grid is an exact subset of
    e4m3, so each nibble maps to an e4m3 byte by bit manipulation
    (sign<<7 | (e2+6)<<3 | m<<2) — no FP, no requant rounding. The per-32
    weight scale moves to the accumulator (per K-tile) in the kernel."""
    local_size = 16  # e4m3: 16 elems per 128-bit transaction
    local_compress = local_size // 2

    @T.macro
    def dequant(WQ_shared, W_shared):
        for i in T.Parallel(block_N * block_K // local_size):
            WQ_local = T.alloc_local((local_compress,), "uint8")
            W_local = T.alloc_local((local_size,), "float8_e4m3fn")
            cbase = i * local_compress
            nbase = i * local_size
            for v in T.vectorized(local_compress):
                WQ_local[v] = WQ_shared[(cbase + v) // (block_K // 2), (cbase + v) % (block_K // 2)]
            for v in T.serial(local_size):
                byte = WQ_local[v // 2]
                nib = (byte >> ((v % 2) * 4)) & 15
                b = ((nib & 8) << 4) | ((((nib >> 1) & 3) + 6) << 3) | ((nib & 1) << 2)
                W_local[v] = T.reinterpret(T.cast(b, "uint8"), "float8_e4m3fn")
            for v in T.vectorized(local_size):
                W_shared[(nbase + v) // block_K, (nbase + v) % block_K] = W_local[v]

    return dequant


# ------------------------------------------------------------- variants


def _make_scale_in_dequant(target, block_K, stages, threads, ws_off):
    """Common scale-in-dequant kernel (baseline / v_ws / v_sota / v_m64).
    block_M/block_N are launch-time specializations (Python ints from the
    caller, like backend.linear_fp4)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs(ws_off))
    def ker(XQ, WQ, WScale, AScale, block_M, block_N):
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // 32), "float32")
        AScale: T.Tensor((M,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            X_shared = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, block_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
            Scale_shared = T.alloc_shared((block_N, block_K // 32), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // block_K, num_stages=stages):
                T.copy(XQ[by * block_M, k * block_K], X_shared)
                T.copy(WQ[bx * block_N, k * block_K // 2], WQ_shared)
                T.copy(WScale[bx * block_N, k * block_K // 32], Scale_shared)
                _dequant_fp4_macro("float8_e4m3fn", 16)(
                    WQ_shared, Scale_shared, W_shared, block_N, block_K
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] / AScale[by * block_M + i]
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    return ker


def make_baseline(target):
    return _make_scale_in_dequant(target, 64, 3, 128, ws_off=True)


def make_v_ws(target):
    return _make_scale_in_dequant(target, 64, 3, 128, ws_off=False)


def make_v_sota(target):
    return _make_scale_in_dequant(target, 128, 2, 256, ws_off=True)


def make_v_m64(target):
    return _make_scale_in_dequant(target, 64, 3, 128, ws_off=True)


def make_v_int32(target):
    """block_K=32, integer dequant (no FP), per-N-row scale on the
    accumulator per K-tile (deepgemm 2xAcc)."""
    BK = 32

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def ker(XQ, WQ, WScale, AScale, block_M, block_N):
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // 32), "float32")
        AScale: T.Tensor((M,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            X_shared = T.alloc_shared((block_M, BK), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, BK // 2), "uint8")
            W_shared = T.alloc_shared((block_N, BK), "float8_e4m3fn")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_accum = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_accum)
            T.clear(C_local)
            for k in T.Pipelined(K // BK, num_stages=3):
                T.copy(XQ[by * block_M, k * BK], X_shared)
                T.copy(WQ[bx * block_N, k * BK // 2], WQ_shared)
                _dequant_fp4_int_macro(block_N, BK)(WQ_shared, W_shared)
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_accum[i, j] += C_local[i, j] * WScale[bx * block_N + j, k]
                T.clear(C_local)
            for i, j in T.Parallel(block_M, block_N):
                C_accum[i, j] = C_accum[i, j] / AScale[by * block_M + i]
            T.copy(C_accum, Y[by * block_M, bx * block_N])
        return Y

    return ker


def _make_split(target, split, block_N):
    """K-split kernel: each block sums K/split, then f32 atomic-adds its
    partial into the zeroed output. The AScale divide distributes over the
    split sum, so it stays per-partial."""
    BK = 64

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def ker(XQ, WQ, WScale, AScale, Y, block_M, block_N):
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // 32), "float32")
        AScale: T.Tensor((M,), "float32")
        Y: T.Tensor((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split, threads=128) as (
            bx,
            by,
            bk,
        ):
            X_shared = T.alloc_shared((block_M, BK), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, BK // 2), "uint8")
            W_shared = T.alloc_shared((block_N, BK), "float8_e4m3fn")
            Scale_shared = T.alloc_shared((block_N, BK // 32), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            k0 = bk * (K // split // BK)
            k1 = (bk + 1) * (K // split // BK)
            for k in T.Pipelined(k1 - k0, num_stages=3):
                kk = k0 + k
                T.copy(XQ[by * block_M, kk * BK], X_shared)
                T.copy(WQ[bx * block_N, kk * BK // 2], WQ_shared)
                T.copy(WScale[bx * block_N, kk * BK // 32], Scale_shared)
                _dequant_fp4_macro("float8_e4m3fn", 16)(
                    WQ_shared, Scale_shared, W_shared, block_N, BK
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] / AScale[by * block_M + i]
            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(Y[by * block_M + i, bx * block_N + j], C_local[i, j])

    return ker


def make_v_split2(target):
    return _make_split(target, 2, 128)


def make_v_split4(target):
    return _make_split(target, 4, 128)


def make_v_n64(target):
    return _make_scale_in_dequant(target, 64, 3, 128, ws_off=True)


def make_v_n64_split2(target):
    return _make_split(target, 2, 64)


# name -> (make, block_M, block_N, block_K, takes_zeroed_Y)
VARIANTS = {
    "baseline": (make_baseline, 128, 128, 64, False),
    "v_int32": (make_v_int32, 128, 128, 32, False),
    "v_ws": (make_v_ws, 128, 128, 64, False),
    "v_split2": (make_v_split2, 128, 128, 64, True),
    "v_split4": (make_v_split4, 128, 128, 64, True),
    "v_sota": (make_v_sota, 256, 128, 128, False),
    "v_m64": (make_v_m64, 64, 128, 64, False),
    "v_n64": (make_v_n64, 128, 64, 64, False),
    "v_n64_split2": (make_v_n64_split2, 128, 64, 64, True),
}

_QUANT = None


def _quant():
    global _QUANT
    if _QUANT is None:
        _QUANT = make_quant_fp8_e4m3("cuda")
    return _QUANT


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def _make_inputs(dev, M, K, N, bK):
    """Pack the weights and quantize the padded activation (same construction
    as backend.linear_fp4's fp8 path). Padded to the variant's block_K."""
    torch.manual_seed(0)
    w_master = torch.randn(N, K) * 0.1
    wq, scale = pack_fp4(w_master)
    x = torch.randn(M, K) * 0.5
    Mp, Np, Kp = _round_up(M, 64), _round_up(N, 128), _round_up(K, bK)
    x2 = torch.zeros(Mp, Kp, dtype=torch.bfloat16)
    x2[:M, :K] = x
    x2 = x2.to(dev)
    xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=dev)
    ascale = torch.empty((Mp,), dtype=torch.float32, device=dev)
    _quant()(x2, xq, ascale, 256)
    wq_p = torch.zeros(Np, Kp // 2, dtype=torch.uint8, device=dev)
    wq_p[:N, : K // 2] = wq.to(dev)
    scale_p = torch.zeros(Np, Kp // 32, dtype=torch.float32, device=dev)
    scale_p[:N, : K // 32] = scale.to(dev)
    ref = linear_fp4(x, wq, scale).to(dev)
    return xq, wq_p, scale_p, ascale, ref


def _time(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


def _run_variant(name, make, bM, bN, bK, takes_y, xq, wq, wscale, ascale, M, N):
    """Compile + parity + time one variant. Returns (ms, rel_ref, rel_base, ok, jit)."""
    Kp = xq.shape[1]
    xq_v = xq[:, : _round_up(Kp, bK)]
    wq_v = wq[:, : _round_up(Kp, bK) // 2]
    wscale_v = wscale[:, : _round_up(Kp, bK) // 32]
    t0 = time.perf_counter()
    try:
        ker = make("cuda")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:<10}: COMPILE-FAIL {type(exc).__name__}: {exc}", flush=True)
        return None
    jit = time.perf_counter() - t0
    if takes_y:
        Y = torch.zeros(xq.shape[0], wq.shape[0], dtype=torch.float32, device=xq.device)
        ker(xq_v, wq_v, wscale_v, ascale, Y, bM, bN)
        torch.cuda.synchronize()
        out = Y[:M, :N]
        Yt = torch.zeros_like(Y)
        ms = _time(lambda: ker(xq_v, wq_v, wscale_v, ascale, Yt, bM, bN))
    else:
        out = ker(xq_v, wq_v, wscale_v, ascale, bM, bN)[:M, :N]
        torch.cuda.synchronize()
        ms = _time(lambda: ker(xq_v, wq_v, wscale_v, ascale, bM, bN))
    return ms, out, jit


def main():
    names = sys.argv[1:] or list(VARIANTS)
    dev = "cuda"
    for M, K, N in SHAPES:
        print(f"=== M={M} K={K} N={N} ===", flush=True)
        xq, wq, wscale, ascale, ref = _make_inputs(dev, M, K, N, 128)
        # baseline first: it is the parity reference (the shipped kernel's
        # math). The fp32 reference carries the fp8 quant floor (~4% rel-err:
        # 2% activation + 1.7% weight requant), so variants gate against
        # baseline, not against the fp32 reference. v_int32 drops the weight
        # requant (exact e2m1->e4m3), so it is MORE accurate than baseline.
        b_ms, b_out, b_jit = _run_variant(
            "baseline", *VARIANTS["baseline"], xq, wq, wscale, ascale, M, N
        )
        b_rel = (b_out.float() - ref).abs().max().item() / ref.abs().max().item()
        flops = 2 * M * N * K
        print(
            f"{'baseline':<10}: {b_ms:7.3f} ms  {flops / b_ms / 1e9:6.1f} TFLOP/s  "
            f"rel-ref {b_rel:.2e}  (jit {b_jit:.0f}s)",
            flush=True,
        )
        for name in names:
            if name == "baseline":
                continue
            make, bM, bN, bK, takes_y = VARIANTS[name]
            res = _run_variant(name, make, bM, bN, bK, takes_y, xq, wq, wscale, ascale, M, N)
            if res is None:
                continue
            ms, out, jit = res
            rel_ref = (out.float() - ref).abs().max().item() / ref.abs().max().item()
            rel_base = (out.float() - b_out.float()).abs().max().item() / b_out.abs().max().item()
            # gate: same math as shipped (rel_base < 1e-2), or strictly more
            # accurate (v_int32: rel_ref below the shipped floor).
            ok = rel_base < 1e-2 or rel_ref < b_rel
            print(
                f"{name:<10}: {ms:7.3f} ms  {flops / ms / 1e9:6.1f} TFLOP/s  "
                f"rel-ref {rel_ref:.2e}  rel-base {rel_base:.2e}  "
                f"{'OK' if ok else 'FAIL'}  (jit {jit:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
