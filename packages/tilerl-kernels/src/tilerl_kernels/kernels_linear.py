"""Linear + quant MMA (tensor-core) TileLang kernels for sm90 — SOTA
schedules ported from the tilelang examples. Registered in the sm90 cell
of the dispatch matrix (registry.py); kernels.py keeps the portable floor
(CPU T.gemm + naive FMA) for cpu/metal. The MMA schedules do not lower on
CPU (T.gemm -> WGMMA only on sm90).

All kernels are bf16-IO on sm90 (the backend casts f32 at the boundary on
CPU/metal; eager JIT does not specialize on dtype) and lower to bf16 WGMMA
with f32 accumulation. The fp4 kernels decode the OCP e2m1 grid with the
lop3-style integer bit-pattern fast decode (no exp2 in the loop).
"""

from __future__ import annotations

import os

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs

__all__ = [
    "make_gemm_nt_mma",
    "make_gemm_nn_mma",
    "make_gemm_tn_mma",
    "make_linear_fp4_mma",
    "make_linear_fp4_gemv",
    "make_linear_bf16_gemv",
    "make_linear_fp8_gemv",
    "make_quant_fp8_e4m3",
    "make_linear_fp4_fp8_mma",
    "make_linear_fp8_mma",
]

#: Reduction-tile size (K for gemm_nt/nn, M for gemm_tn, K for linear_fp4).
#: WGMMA K on sm90 is 16 (bf16); 32 is 2 K-steps, divides every model K dim
#: (all are multiples of 32), and matches examples/gemm/example_gemm.py.
#: The backend pads the reduction dim to a multiple of this on CUDA and imports
#: THIS name to do it — a second copy of the number would silently mis-pad.
#: ``TILERL_RED_TILE`` A/Bs it in one run: 32 makes the fp4 backward's inner
#: gemm [64,32] @ [32,64], which is WGMMA's minimum K, and that kernel measures
#: 15% of peak (docs/experience/errors/2026-08-29-mma8-is-register-bound.md).
_RED_TILE = int(os.environ.get("TILERL_RED_TILE", "32"))
if _RED_TILE % 16 or _RED_TILE <= 0:
    raise ValueError(f"TILERL_RED_TILE must be a positive multiple of 16, got {_RED_TILE}")

# ---------------------------------------------------------------- gemm (MMA)


def make_gemm_nt_mma(target: str):
    """C = A @ B.T + Bias. A [M,K], B [N,K] -> C [M,N].

    # SOTA copy: examples/gemm/example_gemm.py @ tilelang main
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate) instead of fp16; Bias
    # fused into the epilogue; reduction tile fixed at _RED_TILE (backend
    # pads K).
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_nt(A, B, Bias, block_M, block_N, threads):
        # 128 threads (warp group) for WGMMA on large tiles; small tiles
        # (block_M < 32) cannot be evenly partitioned across 4 warps, so
        # keep the caller's 64 (mma.sync per-warp, still tensor cores).
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "bfloat16")
        B: T.Tensor((N, K), "bfloat16")
        Bias: T.Tensor((N,), "float32")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, _RED_TILE), "bfloat16")
            B_shared = T.alloc_shared((block_N, _RED_TILE), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _RED_TILE, num_stages=3):
                T.copy(A[by * block_M, k * _RED_TILE], A_shared)
                T.copy(B[bx * block_N, k * _RED_TILE], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += Bias[bx * block_N + j]
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return gemm_nt


def make_gemm_nn_mma(target: str):
    """C = A @ B. A [M,K], B [K,N] -> C [M,N].

    # SOTA copy: examples/gemm/example_gemm.py @ tilelang main
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); B is [K,N] (loaded
    # as-is, no transpose); reduction tile fixed at _RED_TILE.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_nn(A, B, block_M, block_N, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), "bfloat16")
        B: T.Tensor((K, N), "bfloat16")
        C = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, _RED_TILE), "bfloat16")
            B_shared = T.alloc_shared((_RED_TILE, block_N), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _RED_TILE, num_stages=3):
                T.copy(A[by * block_M, k * _RED_TILE], A_shared)
                T.copy(B[k * _RED_TILE, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return gemm_nn


def make_gemm_tn_mma(target: str):
    """C = A.T @ B. A [M,N], B [M,K] -> C [N,K] (C_ij = sum_m A_mi B_mj).

    # SOTA copy: examples/gemm/example_gemm.py @ tilelang main
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); transpose_A=True with
    # the reduction over M tiled at _RED_TILE; output tiles are
    # (block_N, block_K) per the naive signature.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_tn(A, B, block_N, block_K, threads):
        threads = 128 if block_N >= 32 else threads
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, N), "bfloat16")
        B: T.Tensor((M, K), "bfloat16")
        C = T.empty((N, K), "float32")
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((_RED_TILE, block_N), "bfloat16")
            B_shared = T.alloc_shared((_RED_TILE, block_K), "bfloat16")
            C_local = T.alloc_fragment((block_N, block_K), "float32")
            T.clear(C_local)
            for m in T.Pipelined(M // _RED_TILE, num_stages=3):
                T.copy(A[m * _RED_TILE, by * block_N], A_shared)
                T.copy(B[m * _RED_TILE, bx * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_A=True)
            T.copy(C_local, C[by * block_N, bx * block_K])
        return C

    return gemm_tn


# ---------------------------------------------------------------- linear fp4 (MMA)


def _dequant_fp4_macro(out_dtype, local_size, block):
    """Vectorized e2m1 dequant: a packed WQ tile -> a dequantized W tile in
    shared memory, 128-bit transactions (local_size elems out / local_size//2
    packed bytes in per chunk), one per-``block`` scale per chunk (``block``
    must be a multiple of local_size, so a chunk never crosses a scale block).

    The chunk loop is T.Parallel (not T.serial): a serial chunk loop obstructs
    the K-loop pipeliner on long K-loops (the dequant can't hide behind the
    WGMMA), costing ~20% on K=17408. T.Parallel lets the software pipeliner
    overlap dequant(k+1) with WGMMA(k).

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (fast_dequant path's per-thread vectorized macro)
    # Adapted: the SOTA's twiddling extern decodes our twiddled e2m1 bytes to
    #   bf16 (_FP4_TWIDDLE_SRC, natural order), then the float block scale;
    #   tileRL's float block scale is staged to shared (Scale_shared)
    #   and applied once per chunk (the chunk is aligned and never crosses a
    #   scale block); the chunk loop is T.Parallel (the SOTA's T.serial obstructs
    #   the K-loop pipeliner on long K). block_K must be a Python int (the
    #   vectorizer needs the literal divisor, like the SOTA's Block_QK).
    """
    local_compress = local_size // 2  # 2 nibbles per byte
    assert local_size % 8 == 0 and block % local_size == 0, (local_size, block)

    @T.macro
    def dequant(WQ_shared, Scale_shared, W_shared, block_N, block_K):
        T.import_source(_FP4_TWIDDLE_SRC)
        for i in T.Parallel(block_N * block_K // local_size):
            WQ_local = T.alloc_local((local_compress,), "uint8")
            W_local = T.alloc_local((local_size,), out_dtype)
            cbase = i * local_compress
            nbase = i * local_size
            for v in T.vectorized(local_compress):
                WQ_local[v] = WQ_shared[(cbase + v) // (block_K // 2), (cbase + v) % (block_K // 2)]
            s = Scale_shared[nbase // block_K, (nbase % block_K) // block]
            # twiddle decode: 4 bytes -> 8 bf16 in natural order (2.25 ops/elem)
            D_local = T.alloc_local((local_size,), "bfloat16")
            for wi in T.unroll(local_compress // 4):
                T.call_extern(
                    "tl_fp4_decode8_p", T.access_ptr(WQ_local[4 * wi], "r"),
                    T.access_ptr(D_local[8 * wi], "w"), dtype="void",
                )
            for v in T.unroll(local_size):
                W_local[v] = T.cast(T.cast(D_local[v], "float32") * s, out_dtype)
            for v in T.vectorized(local_size):
                W_shared[(nbase + v) // block_K, (nbase + v) % block_K] = W_local[v]

    return dequant


#: K-tile for the fp4 MMA paths: bf16 WGMMA K=16 (4 steps), fp8 WGMMA K=32
#: (2 steps). 64 amortizes the dequant over multiple WGMMA steps; the backend
#: pads K to a multiple of this on CUDA. Not the scale block: the scale block
#: (16 or 32) must divide this.
_FP4_BLOCK_K = 64

#: N-tile for the fp4 fp8 prefill path. 64 (not the caller's 128): the 128
#: tile left the small-N grids (down/out, N=5120 -> 40 N-tiles) under 1 wave,
#: so the dequant and WGMMA phases aligned across resident blocks and the
#: tensor cores idled. 64 doubles the N-tile count (2+ waves on every prefill
#: shape) for +33% geo-mean TFLOP/s (sweep 2026-08-25,
#: scripts/_sweep_fp8_prefill.py). The caller still pads N to a multiple of
#: 128, which is a multiple of 64.
_FP4_BLOCK_N = 64


def make_linear_fp4_mma(target: str):
    """Fused e2m1 dequant + matmul (sm90 MMA), bf16-IO.

    X [M,K] bf16, WQ uint8 [N,K//2] twiddled (reference.twiddle_fp4), Scale
    [N,K//block] f32. Y[m,n] = sum_k X[m,k] * w[n,k] * Scale[n,k//block].

    The dequant runs ahead of the WGMMA in the pipelined K-loop: each stage
    copies the X/WQ/Scale tiles to shared, vectorizes the dequant into W_shared
    (128-bit transactions, one scale per 8-elem chunk), then WGMMA reads
    W_shared. With num_stages=3 the dequant of stage k+1 issues while the
    WGMMA of stage k is in flight (the async WGMMA + double-buffered shared),
    so the dequant hides behind the MMA.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (fast_dequant path: vectorized shared dequant before
    #   the WGMMA, pipelined)
    # Adapted: bf16 IO (bf16 WGMMA, f32 accumulate); tileRL's float per-block
    #   scale (block_max/6) staged to shared and applied per chunk
    #   instead of the example's integer-exponent scale; OCP e2m1 grid
    #   (matches pack_fp4; a zero nibble is e2m1 0.0, so padded WQ bytes
    #   decode to 0.0).
    """

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4(X, WQ, Scale, block_M, block_N, block, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        X: T.Tensor((M, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _FP4_BLOCK_K), "bfloat16")
            WQ_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _FP4_BLOCK_K), "bfloat16")
            Scale_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // block), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _FP4_BLOCK_K, num_stages=3):
                T.copy(X[by * block_M, k * _FP4_BLOCK_K], X_shared)
                T.copy(WQ[bx * block_N, k * _FP4_BLOCK_K // 2], WQ_shared)
                T.copy(Scale[bx * block_N, k * _FP4_BLOCK_K // block], Scale_shared)
                _dequant_fp4_macro("bfloat16", 8, block)(
                    WQ_shared, Scale_shared, W_shared, block_N, _FP4_BLOCK_K
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp4


# ---------------------------------------------------------------- linear fp4 (GEMV)


#: e2m1 x8 -> 4 x bf16x2 in 18 PTX ops, from the twiddled byte layout
#: (reference.twiddle_fp4). prmt selector 0x0123 keeps the upper 16 bits zero,
#: which sidesteps the CUDA 12.9 prmt.b32 immediate-truncation bug
#: (docs/experience/errors/2026-08-26-fp4-gemv-dequant-issue-rejected.md).
_FP4_TWIDDLE_SRC = r"""
// SOTA copy: tilelang examples/dequantize_gemm/quantize/mxfp.py
//   decode_fp4_to_bf16_twiddling (the asm block) — e2m1 x8 -> 4 x bf16x2 in
//   18 ops. Wrapped here as a 16-elem GEMV tile: decode 2 words, 8 packed
//   bf16x2 FMAs against X, unpack the bf16x2 partial, scale, f32-accumulate.
__device__ __forceinline__ void tl_fp4_decode8(unsigned w, unsigned *out) {
  unsigned tmp, bias, d0, d1, d2, d3, d4, d5, d6;
  asm volatile(
      // To handle the endianness issue
      "prmt.b32 %13, %4, 0, 0x0123;"
      "mov.b32 %12, 0x7e807e80;"
      "and.b32 %0, %13, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %0, %0, %12;"
      "shl.b32 %1, %13, 3;"
      "and.b32 %1, %1, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %1, %1, %12;"
      "shl.b32 %2, %13, 6;"
      "and.b32 %2, %2, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %2, %2, %12;"
      "shl.b32 %5, %13, 1;"
      "and.b32 %6, %5, 0b10000000000000001000000000000000;"
      "shr.b32 %7, %13, 3;"
      "and.b32 %8, %7, 0b00000001100000000000000110000000;"
      "or.b32 %9, %6, %8;"
      "shr.b32 %10, %13, 7;"
      "and.b32 %11, %10, 0b00000000010000000000000001000000;"
      "or.b32 %3, %9, %11;"
      "mul.bf16x2 %3, %3, %12;"
      :"=r"(out[0])
      ,"=r"(out[1])
      ,"=r"(out[2])
      ,"=r"(out[3])
      :"r"(w), "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(d4), "r"(d5), "r"(d6), "r"(bias), "r"(tmp)
    );
}
__device__ __forceinline__ void tl_fp4_decode8_p(const void *wq, void *out) {
  tl_fp4_decode8(*(const unsigned *)wq, (unsigned *)out);
}
// e4m3 x2 (a 16-bit lane pair in t: byte0 = elem0, byte2 = elem1) -> bf16x2:
// bf16 = s<<15 | e<<7 | m<<4 == (b&0x80)<<8 | (b&0x7f)<<4, then * 2^120
// (0x7B80) rebiases the 4-bit exponent (bias 7 -> 127); e4m3 subnormals land
// on bf16 subnormals and rebias exactly, like the e2m1 twiddle above.
__device__ __forceinline__ unsigned tl_e4m3x2_to_bf16x2(unsigned t) {
  unsigned r = ((t << 8) & 0x80008000u) | ((t << 4) & 0x07F007F0u);  // nvcc emits lop3
  asm("mul.bf16x2 %0, %0, %1;" : "+r"(r) : "r"(0x7B807B80u));
  return r;
}
__device__ __forceinline__ unsigned tl_bf16x2_to_f16x2(unsigned v) {
  float lo = __uint_as_float(v << 16), hi = __uint_as_float(v & 0xffff0000u);
  unsigned r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ unsigned tl_e4m3x2_to_f16x2(unsigned short p) {
  unsigned r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(p));
  return r;
}
// GROUP tiles, loaded straight from global (uint4) into registers first —
// tilelang locals handed to an extern by pointer land in local memory.
// Tiles are block_K elements apart: W stride = block_K bytes, X stride =
// 2*block_K bytes. sc[g] = the tile's block scale.
template <int G>
__device__ __forceinline__ void tl_fp8_gemv_tiles(const void *w8v, const void *xv, int block_K,
                                                  const float *sc, float *acc) {
  const unsigned char *w8 = (const unsigned char *)w8v;
  const unsigned short *x = (const unsigned short *)xv;
  uint4 w[G], x0[G], x1[G];
  // volatile asm keeps all G*3 loads issued before any math (nvcc sinks plain
  // loads next to their use, which is exactly the latency serialization we
  // are trying to avoid).
#pragma unroll
  for (int g = 0; g < G; ++g) {
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(w[g].x), "=r"(w[g].y), "=r"(w[g].z), "=r"(w[g].w) : "l"(w8 + g * block_K));
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x0[g].x), "=r"(x0[g].y), "=r"(x0[g].z), "=r"(x0[g].w) : "l"(x + g * block_K));
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x1[g].x), "=r"(x1[g].y), "=r"(x1[g].z), "=r"(x1[g].w) : "l"(x + g * block_K + 8));
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const unsigned ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
    const unsigned xw[8] = {x0[g].x, x0[g].y, x0[g].z, x0[g].w, x1[g].x, x1[g].y, x1[g].z, x1[g].w};
    unsigned a = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      unsigned lo, hi;
      asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo) : "r"(ww[i]));
      asm("prmt.b32 %0, %1, 0, 0x4342;" : "=r"(hi) : "r"(ww[i]));
      asm volatile("fma.rn.bf16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(tl_e4m3x2_to_bf16x2(lo)), "r"(xw[2 * i]));
      asm volatile("fma.rn.bf16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(tl_e4m3x2_to_bf16x2(hi)), "r"(xw[2 * i + 1]));
    }
    float l = __uint_as_float(a << 16), h = __uint_as_float(a & 0xffff0000u);
    *acc = fmaf(sc[g], l + h, *acc);
  }
}
// Tensor-core decode GEMM for M <= 8 (Marlin-style). One warp, one k32 chunk,
// NG groups of 8 output rows. The natural twiddled layout is already a valid
// mma B fragment under a consistent k permutation: lane (g = l/4, q = l%4)
// loads 4 bytes = 8 consecutive k of row 8*grp + g, decoded to 4 bf16x2 pairs
// d0..d3; d0/d1 are b0/b1 of k16 tile 0 and d2/d3 of tile 1 with virtual
// k {2q, 2q+1, 2q+8, 2q+9} standing for actual k 8q+{0,1,2,3} (+4 for tile
// 1). The A fragment uses the same map: one LDG.128 of X row g at actual
// k0+8q gives a0/a2 of both tiles; rows 8..15 (a1/a3) are zero. All 8 of a
// lane's elements sit in one 16-block, so one scale per lane per chunk,
// applied on the B fragment (bf16 mul, exact: block scales are e4m3 values).
// acc[grp*4 + {0,1}] = C rows g, cols 8*grp + 2q + {0,1}; {2,3} are junk rows.
// GROUP fp4 tiles (16 elems = 8 twiddled bytes each), loaded straight from
// global into registers first (same rationale as tl_fp8_gemv_tiles). Tiles
// are block_K elements apart: W stride block_K/2 bytes, X stride 2*block_K.
// The warp's whole K range in one call: accumulators and scales stay in
// registers (a TIR local handed to the extern by pointer lives in local
// memory — every mma then round-trips its 16 accumulators through L1; ncu
// showed L1 traffic at 2x DRAM and no gain from 128 -> 64 registers).
// chunk in [c0, c1): W at wq + chunk*16 bytes (fp4) / *32 (fp8), X at
// x + chunk*32 elements, scale = sc[row_off + chunk*32/block] per row.
template <int NG, int G>
__device__ __forceinline__ void tl_fp4_mma_rows(const void *wqv, int w_grp_stride, const void *xv,
                                                const float *sc, int sc_grp_stride,
                                                int sc_per_chunk, int c0, int c1, float *out) {
  const unsigned char *wq = (const unsigned char *)wqv;
  const unsigned short *x = (const unsigned short *)xv;
  float acc[NG * 4];
#pragma unroll
  for (int i = 0; i < NG * 4; ++i) acc[i] = 0.f;
  const unsigned zero = 0u;
  int c = c0;
  for (; c + G <= c1; c += G) {
    uint4 xa[G];
    unsigned w[G][NG];
    float s[G][NG];
#pragma unroll
    for (int k = 0; k < G; ++k) {
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(xa[k].x), "=r"(xa[k].y), "=r"(xa[k].z), "=r"(xa[k].w) : "l"(x + (c + k) * 32));
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(w[k][g]) : "l"(wq + g * w_grp_stride + (c + k) * 16));
        s[k][g] = __ldg(sc + g * sc_grp_stride + (c + k) * sc_per_chunk);
      }
    }
#pragma unroll
    for (int k = 0; k < G; ++k) {
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        unsigned d[4];
        tl_fp4_decode8(w[k][g], d);
        unsigned s2;
        asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(s2) : "f"(s[k][g]));
#pragma unroll
        for (int i = 0; i < 4; ++i) asm("mul.bf16x2 %0, %0, %1;" : "+r"(d[i]) : "r"(s2));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(acc[g * 4]), "+f"(acc[g * 4 + 1]), "+f"(acc[g * 4 + 2]), "+f"(acc[g * 4 + 3])
                     : "r"(xa[k].x), "r"(zero), "r"(xa[k].y), "r"(zero), "r"(d[0]), "r"(d[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(acc[g * 4]), "+f"(acc[g * 4 + 1]), "+f"(acc[g * 4 + 2]), "+f"(acc[g * 4 + 3])
                     : "r"(xa[k].z), "r"(zero), "r"(xa[k].w), "r"(zero), "r"(d[2]), "r"(d[3]));
      }
    }
  }
  for (; c < c1; ++c) {  // tail chunks
    uint4 xa;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(xa.x), "=r"(xa.y), "=r"(xa.z), "=r"(xa.w) : "l"(x + c * 32));
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      unsigned w;
      asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(w) : "l"(wq + g * w_grp_stride + c * 16));
      const float sg = __ldg(sc + g * sc_grp_stride + c * sc_per_chunk);
      unsigned d[4];
      tl_fp4_decode8(w, d);
      unsigned s2;
      asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(s2) : "f"(sg));
#pragma unroll
      for (int i = 0; i < 4; ++i) asm("mul.bf16x2 %0, %0, %1;" : "+r"(d[i]) : "r"(s2));
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(acc[g * 4]), "+f"(acc[g * 4 + 1]), "+f"(acc[g * 4 + 2]), "+f"(acc[g * 4 + 3])
                   : "r"(xa.x), "r"(zero), "r"(xa.y), "r"(zero), "r"(d[0]), "r"(d[1]));
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(acc[g * 4]), "+f"(acc[g * 4 + 1]), "+f"(acc[g * 4 + 2]), "+f"(acc[g * 4 + 3])
                   : "r"(xa.z), "r"(zero), "r"(xa.w), "r"(zero), "r"(d[2]), "r"(d[3]));
    }
  }
#pragma unroll
  for (int i = 0; i < NG * 4; ++i) out[i] = acc[i];
}
// Wide-load fp4 twin of tl_fp4_mma_rows: the lane loads v2.u32 (8 bytes = 16
// fp4) where the narrow one loads u32 (4 bytes = 8), halving the WEIGHT load
// instructions. mma8 issues 1.93x the GEMV's load requests for identical DRAM
// traffic (errors/2026-08-29-mma8-is-register-bound.md), and the fp8 twin
// below already loads v2.u32 — its mma8/gemv ratio is 2.07x against fp4's 2.64x.
//
// The lane -> k map widens with it: lane q owns k in [64*cp + 16q, +16) of a
// 64-k chunk PAIR, and X is read at the same element offset, which is what
// keeps the mma's virtual-k permutation consistent between the A and B
// fragments. One scale still covers a lane's 16 values: 16q is aligned to both
// the 16- and 32-element scale blocks.
template <int NG, int G>
__device__ __forceinline__ void tl_fp4_mma_rows_w8(const void *wqv, int w_grp_stride,
                                                   const void *xv, const float *sc,
                                                   int sc_grp_stride, int sc_per_pair,
                                                   int p0, int p1, float *out) {
  const unsigned char *wq = (const unsigned char *)wqv;
  const unsigned short *x = (const unsigned short *)xv;
  float acc[NG * 4];
#pragma unroll
  for (int i = 0; i < NG * 4; ++i) acc[i] = 0.f;
  const unsigned zero = 0u;
  for (int p = p0; p < p1; ++p) {
    uint4 xa0, xa1;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(xa0.x), "=r"(xa0.y), "=r"(xa0.z), "=r"(xa0.w) : "l"(x + p * 64));
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(xa1.x), "=r"(xa1.y), "=r"(xa1.z), "=r"(xa1.w) : "l"(x + p * 64 + 8));
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      uint2 w;
      asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                   : "=r"(w.x), "=r"(w.y) : "l"(wq + g * w_grp_stride + p * 32));
      const float sg = __ldg(sc + g * sc_grp_stride + p * sc_per_pair);
      unsigned d[8];
      tl_fp4_decode8(w.x, d);
      tl_fp4_decode8(w.y, d + 4);
      unsigned s2;
      asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(s2) : "f"(sg));
#pragma unroll
      for (int i = 0; i < 8; ++i) asm("mul.bf16x2 %0, %0, %1;" : "+r"(d[i]) : "r"(s2));
      const unsigned xw[8] = {xa0.x, xa0.y, xa0.z, xa0.w, xa1.x, xa1.y, xa1.z, xa1.w};
#pragma unroll
      for (int t = 0; t < 4; ++t)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(acc[g * 4]), "+f"(acc[g * 4 + 1]), "+f"(acc[g * 4 + 2]),
                       "+f"(acc[g * 4 + 3])
                     : "r"(xw[2 * t]), "r"(zero), "r"(xw[2 * t + 1]), "r"(zero),
                       "r"(d[2 * t]), "r"(d[2 * t + 1]));
    }
  }
#pragma unroll
  for (int i = 0; i < NG * 4; ++i) out[i] = acc[i];
}
template <int NG, int G>
__device__ __forceinline__ void tl_fp8_mma_rows(const void *w8v, int w_grp_stride, const void *xv,
                                                const float *sc, int sc_shift, int c0,
                                                int c1, float *out) {
  // 128-block scale: all 32 rows of a block share one scale row, chunk c's
  // column is (c*32 + 8q) / 128 = c >> 2 for every lane.
  const unsigned char *w8 = (const unsigned char *)w8v;
  const unsigned short *x = (const unsigned short *)xv;
  float acc[NG * 4];
#pragma unroll
  for (int i = 0; i < NG * 4; ++i) acc[i] = 0.f;
  const unsigned zero = 0u;
  for (int c = c0; c < c1; c += G) {  // G chunks of loads in flight, then the math
    const int n = (c1 - c < G) ? (c1 - c) : G;
    uint4 xa[G];
    uint2 w[G][NG];
    float s[G][NG];
#pragma unroll
    for (int k = 0; k < G; ++k) {
      if (k < n) {
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(xa[k].x), "=r"(xa[k].y), "=r"(xa[k].z), "=r"(xa[k].w) : "l"(x + (c + k) * 32));
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                       : "=r"(w[k][g].x), "=r"(w[k][g].y) : "l"(w8 + g * w_grp_stride + (c + k) * 32));
          s[k][g] = __ldg(sc + ((c + k) >> sc_shift));
        }
      }
    }
#pragma unroll
    for (int k = 0; k < G; ++k) {
      if (k < n) {
        const unsigned a0 = tl_bf16x2_to_f16x2(xa[k].x), a2 = tl_bf16x2_to_f16x2(xa[k].y);
        const unsigned a0b = tl_bf16x2_to_f16x2(xa[k].z), a2b = tl_bf16x2_to_f16x2(xa[k].w);
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          unsigned d[4] = {tl_e4m3x2_to_f16x2((unsigned short)(w[k][g].x)),
                           tl_e4m3x2_to_f16x2((unsigned short)(w[k][g].x >> 16)),
                           tl_e4m3x2_to_f16x2((unsigned short)(w[k][g].y)),
                           tl_e4m3x2_to_f16x2((unsigned short)(w[k][g].y >> 16))};
          unsigned s2;
          asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(s2) : "f"(s[k][g]));
#pragma unroll
          for (int i = 0; i < 4; ++i) asm("mul.f16x2 %0, %0, %1;" : "+r"(d[i]) : "r"(s2));
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+f"(acc[g * 4]), "+f"(acc[g * 4 + 1]), "+f"(acc[g * 4 + 2]), "+f"(acc[g * 4 + 3])
                       : "r"(a0), "r"(zero), "r"(a2), "r"(zero), "r"(d[0]), "r"(d[1]));
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+f"(acc[g * 4]), "+f"(acc[g * 4 + 1]), "+f"(acc[g * 4 + 2]), "+f"(acc[g * 4 + 3])
                       : "r"(a0b), "r"(zero), "r"(a2b), "r"(zero), "r"(d[2]), "r"(d[3]));
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < NG * 4; ++i) out[i] = acc[i];
}
// Warp reduce of M accumulators across the 32 lanes of threadIdx.x (the GEMV's
// reduce_thread dim is exactly one warp). Replaces tvm_thread_allreduce, whose
// output must be a zero-indexed buffer — M rows would need M of them, and the
// eager builder only accepts range/T.* loops, not a Python one over them.
template <int M>
__device__ __forceinline__ void tl_warp_reduce_m(float *acc) {
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
  }
}
// M-row GEMV tiles: W is streamed and decoded ONCE and reused across M rows of
// X, so the weight bytes (the decode bottleneck) do not scale with M. X rows
// are xrow elements apart. M=1 is the plain GEMV; the mma8 path pads M to 8 and
// pays the full 8-row cost at M=2.
template <int G, int M>
__device__ __forceinline__ void tl_fp8_gemv_tiles_m(const void *w8v, const void *xv, int block_K,
                                                    int xrow, const float *sc, float *acc) {
  const unsigned char *w8 = (const unsigned char *)w8v;
  const unsigned short *x = (const unsigned short *)xv;
  uint4 w[G];
#pragma unroll
  for (int g = 0; g < G; ++g)
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(w[g].x), "=r"(w[g].y), "=r"(w[g].z), "=r"(w[g].w) : "l"(w8 + g * block_K));
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const unsigned ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
    unsigned d[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      unsigned lo, hi;
      asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo) : "r"(ww[i]));
      asm("prmt.b32 %0, %1, 0, 0x4342;" : "=r"(hi) : "r"(ww[i]));
      d[2 * i] = tl_e4m3x2_to_bf16x2(lo);
      d[2 * i + 1] = tl_e4m3x2_to_bf16x2(hi);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const unsigned short *xp = x + m * xrow + g * block_K;
      uint4 x0, x1;
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x0.x), "=r"(x0.y), "=r"(x0.z), "=r"(x0.w) : "l"(xp));
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x1.x), "=r"(x1.y), "=r"(x1.z), "=r"(x1.w) : "l"(xp + 8));
      const unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      unsigned a = 0u;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        asm volatile("fma.rn.bf16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(d[j]), "r"(xw[j]));
      float l = __uint_as_float(a << 16), h = __uint_as_float(a & 0xffff0000u);
      acc[m] = fmaf(sc[g], l + h, acc[m]);
    }
  }
}
template <int G, int M>
__device__ __forceinline__ void tl_fp4_gemv_tiles_m(const void *wqv, const void *xv, int block_K,
                                                    int xrow, const float *sc, float *acc) {
  const unsigned char *wq = (const unsigned char *)wqv;
  const unsigned short *x = (const unsigned short *)xv;
  uint2 w[G];
#pragma unroll
  for (int g = 0; g < G; ++g)
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                 : "=r"(w[g].x), "=r"(w[g].y) : "l"(wq + g * (block_K / 2)));
#pragma unroll
  for (int g = 0; g < G; ++g) {
    unsigned d[8];
    tl_fp4_decode8(w[g].x, d);
    tl_fp4_decode8(w[g].y, d + 4);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const unsigned short *xp = x + m * xrow + g * block_K;
      uint4 x0, x1;
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x0.x), "=r"(x0.y), "=r"(x0.z), "=r"(x0.w) : "l"(xp));
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x1.x), "=r"(x1.y), "=r"(x1.z), "=r"(x1.w) : "l"(xp + 8));
      const unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      unsigned a = 0u;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        asm volatile("fma.rn.bf16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(d[j]), "r"(xw[j]));
      float l = __uint_as_float(a << 16), h = __uint_as_float(a & 0xffff0000u);
      acc[m] = fmaf(sc[g], l + h, acc[m]);
    }
  }
}
template <int G>
__device__ __forceinline__ void tl_fp4_gemv_tiles(const void *wqv, const void *xv, int block_K,
                                                  const float *sc, float *acc) {
  const unsigned char *wq = (const unsigned char *)wqv;
  const unsigned short *x = (const unsigned short *)xv;
  uint2 w[G];
  uint4 x0[G], x1[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(w[g].x), "=r"(w[g].y) : "l"(wq + g * (block_K / 2)));
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x0[g].x), "=r"(x0[g].y), "=r"(x0[g].z), "=r"(x0[g].w) : "l"(x + g * block_K));
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x1[g].x), "=r"(x1[g].y), "=r"(x1[g].z), "=r"(x1[g].w) : "l"(x + g * block_K + 8));
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    unsigned d[8];
    tl_fp4_decode8(w[g].x, d);
    tl_fp4_decode8(w[g].y, d + 4);
    const unsigned xw[8] = {x0[g].x, x0[g].y, x0[g].z, x0[g].w, x1[g].x, x1[g].y, x1[g].z, x1[g].w};
    unsigned a = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      asm volatile("fma.rn.bf16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(d[j]), "r"(xw[j]));
    float l = __uint_as_float(a << 16), h = __uint_as_float(a & 0xffff0000u);
    *acc = fmaf(sc[g], l + h, *acc);
  }
}
"""

# sm70 (Volta) twin of the fp4 twiddle source. Kept separate: _FP4_TWIDDLE_SRC
# carries sm80+/sm89+ instructions (mul.bf16x2, cvt.rn.f16x2.e4m3x2) that do
# not assemble for sm70, and this cell's only fp4 kernel is the GEMV below.
# Everything here is sm_53+ (prmt, mul.f16x2, fma.rn.f16x2, cvt.rn.f16x2.f32).
_FP4_TWIDDLE_SRC_F16 = r"""
#include <cuda_fp16.h>  // __half2 / __low2float for the f16x2 -> f32 horizontal sum
// sm70 twin of tl_fp4_decode8 for the fp16-twiddled layout (reference.twiddle_fp4_f16):
// e2m1 x8 -> 4 x f16x2. Same prmt + slot structure, but the nibble bits rest at
// fp16 field positions (15/11/10/9) and the rebias is mul.f16x2 by 2^14 (0x7400).
// Validated bit-exact against the e2m1 LUT (tests/test_fp4_twiddle.py).
__device__ __forceinline__ void tl_fp4_decode8_f16(unsigned w, unsigned *out) {
  unsigned t, a, b, c, d;
  asm volatile("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(t) : "r"(w));
  asm("and.b32 %0, %1, 0x8E008E00;" : "=r"(a) : "r"(t));
  asm("shl.b32 %0, %1, 7;" : "=r"(b) : "r"(t));
  asm("and.b32 %0, %0, 0x8E008E00;" : "+r"(b));
  unsigned cs, cf;
  asm("shl.b32 %0, %1, 14;" : "=r"(cs) : "r"(t));
  asm("and.b32 %0, %0, 0x80008000;" : "+r"(cs));
  asm("shr.b32 %0, %1, 3;" : "=r"(cf) : "r"(t));
  asm("and.b32 %0, %0, 0x0E000E00;" : "+r"(cf));
  asm("or.b32 %0, %1, %2;" : "=r"(c) : "r"(cs), "r"(cf));
  unsigned ds, df;
  asm("shl.b32 %0, %1, 15;" : "=r"(ds) : "r"(t));
  asm("and.b32 %0, %0, 0x80008000;" : "+r"(ds));
  asm("shl.b32 %0, %1, 4;" : "=r"(df) : "r"(t));
  asm("and.b32 %0, %0, 0x0E000E00;" : "+r"(df));
  asm("or.b32 %0, %1, %2;" : "=r"(d) : "r"(ds), "r"(df));
  const unsigned bias = 0x74007400u;  // 2^14 per fp16 lane: rebias e2m1 exp (bias 1 -> 15)
  asm("mul.f16x2 %0, %0, %1;" : "+r"(a) : "r"(bias));
  asm("mul.f16x2 %0, %0, %1;" : "+r"(b) : "r"(bias));
  asm("mul.f16x2 %0, %0, %1;" : "+r"(c) : "r"(bias));
  asm("mul.f16x2 %0, %0, %1;" : "+r"(d) : "r"(bias));
  out[0] = a; out[1] = b; out[2] = c; out[3] = d;
}
// GROUP 16-elem fp4 tiles for the sm70 GEMV. WQ is fp16-twiddled, X is f32
// (sm70 has no bf16 IO); loads go straight from global (tilelang locals handed
// to an extern by pointer land in local memory). Per tile: decode 2 words ->
// 8 f16x2, cvt X f32->f16x2, 8 fma.rn.f16x2 into one fp16x2 accumulator (fp16
// accumulation stays inside the 16-elem scale block, like the sm90 bf16x2 GEMV),
// one f32 scale-accumulate. Tiles are block_K elements apart.
template <int G>
__device__ __forceinline__ void tl_fp4_gemv_tiles_f16(const void *wqv, const void *xv,
                                                      int block_K, const float *sc, float *acc) {
  const unsigned char *wq = (const unsigned char *)wqv;
  const float *x = (const float *)xv;
  unsigned w[G][2];
  float4 xb[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                 : "=r"(w[g][0]), "=r"(w[g][1]) : "l"(wq + g * block_K / 2));
#pragma unroll
    for (int i = 0; i < 4; ++i)
      asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                   : "=f"(xb[g][i].x), "=f"(xb[g][i].y), "=f"(xb[g][i].z), "=f"(xb[g][i].w)
                   : "l"(x + g * block_K + 4 * i));
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    unsigned d0[4], d1[4];
    tl_fp4_decode8_f16(w[g][0], d0);
    tl_fp4_decode8_f16(w[g][1], d1);
    unsigned xw[8];
    const float *xf = (const float *)xb[g];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // cvt.rn.f16x2.f32 is sm80+; sm70 converts scalar (cvt.rn.f16.f32, sm_53+)
      unsigned lo = __half_as_ushort(__float2half_rn(xf[2 * i]));
      unsigned hi = __half_as_ushort(__float2half_rn(xf[2 * i + 1]));
      xw[i] = lo | (hi << 16);
    }
    unsigned a = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[i]), "r"(d0[i]));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[4 + i]), "r"(d1[i]));
    }
    __half2 ah = __halves2half2(__ushort_as_half(a & 0xffff), __ushort_as_half(a >> 16));
    *acc = fmaf(sc[g], __low2float(ah) + __high2float(ah), *acc);
  }
}
// M-row warp reduce: each of the M accumulators gets a full warp reduction.
// __shfl_down_sync is sm70+ (Volta) — fine, this cell is sm70-only.
template <int M>
__device__ __forceinline__ void tl_warp_reduce_m_f16(float *acc) {
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
  }
}
// X-as-f16 twin of tl_fp4_gemv_tiles_f16_m. The f32 version spends 32 of its
// ~49 per-row instructions turning X into f16 (16 __float2half_rn + 8 shift +
// 8 or) and re-reads X in every block: at M=8, N=17408 that is 4352 blocks x
// 8 rows x 5120 f32 = 0.71 GB, 78% of the kernel's measured time. Pre-packed
// f16 X halves the traffic and drops the row body to ~17 instructions, which
// is what makes an M-row verify cheaper than M separate decodes.
template <int G, int M>
__device__ __forceinline__ void tl_fp4_gemv_tiles_f16_m_xh(
    const void *wqv, const void *xv, int K, int block_K,
    const float *sc, float *acc) {
  const unsigned char *wq = (const unsigned char *)wqv;
  const unsigned *x = (const unsigned *)xv;  // 2 halves per word
#pragma unroll
  for (int g = 0; g < G; ++g) {
    unsigned w0, w1;
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                 : "=r"(w0), "=r"(w1) : "l"(wq + g * block_K / 2));
    unsigned d0[4], d1[4];
    tl_fp4_decode8_f16(w0, d0);
    tl_fp4_decode8_f16(w1, d1);
    // Same no-unroll rule as the f32 twin: one row's xw live at a time.
    for (int m = 0; m < M; ++m) {
      unsigned xw[8];
      // 16 halves = 8 words = two v4.u32 loads, already in fp16.
      const unsigned *xg = x + (size_t)m * (K / 2) + g * block_K / 2;
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(xw[0]), "=r"(xw[1]), "=r"(xw[2]), "=r"(xw[3]) : "l"(xg));
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(xw[4]), "=r"(xw[5]), "=r"(xw[6]), "=r"(xw[7]) : "l"(xg + 4));
      unsigned a = 0u;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[i]), "r"(d0[i]));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[4 + i]), "r"(d1[i]));
      }
      __half2 ah = __halves2half2(__ushort_as_half(a & 0xffff), __ushort_as_half(a >> 16));
      acc[m] = fmaf(sc[g], __low2float(ah) + __high2float(ah), acc[m]);
    }
  }
}
// M-row twin: WQ is loaded + decoded ONCE per tile and reused across all M
// rows, so the weight bytes (the bottleneck — W is ~900x X for M=1) do not
// scale with M. M is a compile-time template arg (the factory bakes it); the
// backend pads M up and slices the output. Mirrors the sm90 fp8 GEMV's
// tl_fp8_gemv_tiles_m<G,M> structure.
template <int G, int M>
__device__ __forceinline__ void tl_fp4_gemv_tiles_f16_m(
    const void *wqv, const void *xv, int K, int block_K,
    const float *sc, float *acc) {
  const unsigned char *wq = (const unsigned char *)wqv;
  const float *x = (const float *)xv;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    unsigned w0, w1;
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                 : "=r"(w0), "=r"(w1) : "l"(wq + g * block_K / 2));
    unsigned d0[4], d1[4];
    tl_fp4_decode8_f16(w0, d0);
    tl_fp4_decode8_f16(w1, d1);
    // No #pragma unroll on the M loop: unrolling 8 rows inside the G-unroll
    // (4x) spills registers (32 bodies x ~25 regs >> 256/thread) and was
    // 150x slower (5.7 ms/tick -> 5.7 s/tick). One row's xb live at a time.
    for (int m = 0; m < M; ++m) {
      float4 xb[4];
      const float *xg = x + m * K + g * block_K;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(xb[i].x), "=f"(xb[i].y), "=f"(xb[i].z), "=f"(xb[i].w)
                     : "l"(xg + 4 * i));
      unsigned xw[8];
      const float *xf = (const float *)xb;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        unsigned lo = __half_as_ushort(__float2half_rn(xf[2 * i]));
        unsigned hi = __half_as_ushort(__float2half_rn(xf[2 * i + 1]));
        xw[i] = lo | (hi << 16);
      }
      unsigned a = 0u;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[i]), "r"(d0[i]));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(a) : "r"(xw[4 + i]), "r"(d1[i]));
      }
      __half2 ah = __halves2half2(__ushort_as_half(a & 0xffff), __ushort_as_half(a >> 16));
      acc[m] = fmaf(sc[g], __low2float(ah) + __high2float(ah), acc[m]);
    }
  }
}
"""


def make_linear_fp4_gemv(target: str, M: int = 1, GROUP: int = 4):
    """Fused e2m1 dequant + GEMV (sm90), the decode path of linear_fp4.

    X [M,K] bf16, WQ uint8 [N,K//2] TWIDDLED (reference.twiddle_fp4), Scale
    [N,K//block] f32. Y[m,n] = sum_k X[m,k] * w[n,k] * Scale[n,k//block].
    ``M`` is a compile-time row count sharing one W stream.

    One warp group per 4 output rows streams WQ once; each thread owns a
    16-elem slice per K-chunk; ``tl_fp4_gemv_tiles<GROUP>`` (C) loads GROUP
    tiles straight from global with program-ordered vector loads, then per
    tile: twiddle decode (2.25 ops/elem, packed bf16x2 out), 8
    ``fma.rn.bf16x2`` against the natural X words, one f32 scale-accumulate
    per 16-elem scale block (bf16 accumulation stays inside the block:
    relerr 4e-3, under the 1e-2 gate). ~3 instr/elem — the shuffle-LUT GEMV
    it replaced issued 6.8/elem at 82% issue-busy / 40% DRAM (ncu
    2026-08-27): gate_up 83.5 -> 56.8 us. See wins/2026-08-28-fp8-gemv-bf16x2
    for why the loads are in C and volatile.

    # SOTA copy: tilelang examples/dequantize_gemm/quantize/mxfp.py
    #   decode_fp4_to_bf16_twiddling + example_dequant_gemv_fp16xint4.py
    #   (split-K + tvm_thread_allreduce GEMV schedule)
    # Adapted: OCP e2m1 grid with tileRL's float block scale on the tile
    #   partial; bf16x2 FMA tile in C (T.call_extern); M rows share one W stream.
    # Constraint: block % 16 == 0 (a tile never straddles a scale); the backend pads
    #   K to 256 so every tile is full.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp4_gemv(X, WQ, Scale, OScale, Res, reduce_thread, n_partition, block):
        N, K = T.const("N, K")
        assert block % 16 == 0  # one scale per 16-elem tile (block 16) or per two (32)
        micro = 16
        block_K = reduce_thread * micro
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        X: T.Tensor((M, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        OScale: T.Tensor((N,), "float32")  # per-row epilogue scale, folded (was a torch mul)
        Res: T.Tensor((M, N), "float32")  # residual stream (zeros when none): Y = Res + y
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            T.import_source(_FP4_TWIDDLE_SRC)
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            acc = T.alloc_local((M,), "float32")
            for m in T.unroll(M):
                acc[m] = 0.0
            sc = T.alloc_local((GROUP,), "float32")
            for kg in T.serial(num_g):
                base = kg * GROUP * block_K + kr * micro
                for g in T.unroll(GROUP):
                    sc[g] = Scale[n, (base + g * block_K) // block]
                T.call_extern(
                    f"tl_fp4_gemv_tiles_m<{GROUP}, {M}>", T.access_ptr(WQ[n, base // 2], "r"),
                    T.access_ptr(X[0, base], "r"), block_K, K, T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"), dtype="void",
                )
            for kt in T.serial(num_ko - num_g * GROUP):  # K-tail, one tile at a time
                base = (num_g * GROUP + kt) * block_K + kr * micro
                sc[0] = Scale[n, base // block]
                T.call_extern(
                    f"tl_fp4_gemv_tiles_m<1, {M}>", T.access_ptr(WQ[n, base // 2], "r"),
                    T.access_ptr(X[0, base], "r"), block_K, K, T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"), dtype="void",
                )
            T.call_extern(f"tl_warp_reduce_m<{M}>", T.access_ptr(acc, "rw"), dtype="void")
            if kr == 0:
                for m in T.unroll(M):
                    Y[m, n] = Res[m, n] + acc[m] * OScale[n]
        return Y

    return linear_fp4_gemv


def make_linear_fp4_mma8(target: str, NG: int = 4, KW: int = 4, G: int = 4, W8: int = 1):
    """Decode GEMM for M <= 8 on the tensor cores (sm90): X [8, K] bf16 (rows
    >= M zero) @ twiddled fp4 -> Y [8, N] f32 (+ Res, * OScale). A block owns
    NG*8 output rows; its KW warps split K in k32 chunks (``tl_fp4_mma_rows``)
    and reduce through shared memory. ~3 instr per weight element for ANY
    M <= 8 — the M=1 GEMV's cost at B=8 (the scalar batched GEMV was
    register-bound at 204 regs/lane, errors/2026-08-28-batched-scalar-gemv).

    # SOTA copy: Marlin (stream W once, A fragment from registers, group
    #   scale on the B fragment); fragment layouts per PTX ISA m16n8k16.
    """
    NB = NG * 8

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp4_mma8(X, WQ, Scale, OScale, Res, block):
        N, K = T.const("N, K")
        assert block % 16 == 0
        nchunk = K // 32
        per = T.ceildiv(nchunk, KW)
        X: T.Tensor((8, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        OScale: T.Tensor((N,), "float32")
        Res: T.Tensor((8, N), "float32")
        Y = T.empty((8, N), "float32")
        with T.Kernel(T.ceildiv(N, NB), threads=(32, KW)) as bx:
            T.import_source(_FP4_TWIDDLE_SRC)
            lane = T.thread_binding(0, 32, thread="threadIdx.x")
            kw = T.thread_binding(0, KW, thread="threadIdx.y")
            n0 = bx * NB
            g = lane // 4
            q = lane % 4
            acc = T.alloc_local((NG * 4,), "float32")
            red = T.alloc_shared((KW, 8, NB), "float32")
            # whole per-warp K range in one extern call: accumulators live in
            # registers there (see tl_fp4_mma_rows); the scale for chunk c of
            # row n is Scale[n, (c*32 + 8q)//block] = row base + c*(32//block)
            if W8:
                # Chunk PAIRS (64 k): the lane owns 16 fp4 at byte 8q of the
                # pair, and X at element 16q, which is the same k range.
                npair = nchunk // 2
                pper = T.ceildiv(npair, KW)
                T.call_extern(
                    f"tl_fp4_mma_rows_w8<{NG}, {G}>",
                    T.access_ptr(WQ[n0 + g, 8 * q], "r"), 8 * (K // 2),
                    T.access_ptr(X[g, 16 * q], "r"),
                    T.access_ptr(Scale[n0 + g, (16 * q) // block], "r"), 8 * (K // block),
                    64 // block, kw * pper, T.min(npair, (kw + 1) * pper),
                    T.access_ptr(acc, "w"), dtype="void",
                )
            else:
                T.call_extern(
                    f"tl_fp4_mma_rows<{NG}, {G}>",
                    T.access_ptr(WQ[n0 + g, 4 * q], "r"), 8 * (K // 2),
                    T.access_ptr(X[g, 8 * q], "r"),
                    T.access_ptr(Scale[n0 + g, (8 * q) // block], "r"), 8 * (K // block),
                    32 // block, kw * per, T.min(nchunk, (kw + 1) * per),
                    T.access_ptr(acc, "w"), dtype="void",
                )
            for grp in T.unroll(NG):
                red[kw, g, 8 * grp + 2 * q] = acc[4 * grp]
                red[kw, g, 8 * grp + 2 * q + 1] = acc[4 * grp + 1]
            T.tvm_storage_sync("shared")
            for i in T.Parallel(8 * NB):
                r = i // NB
                col = i % NB
                tot = T.alloc_fragment((1,), "float32")
                tot[0] = 0.0
                for w in T.serial(KW):
                    tot[0] += red[w, r, col]
                Y[r, n0 + col] = Res[r, n0 + col] + tot[0] * OScale[n0 + col]
        return Y

    return linear_fp4_mma8


# ---------------------------------------------------------------- linear bf16 (GEMV)


def make_linear_bf16_gemv(target: str):
    """GEMV (sm90), the decode (M=1) path of linear: X[1,K] bf16 @ W[N,K]
    bf16 -> Y[1,N] f32.

    Decode is memory-bound: one warp group per n_partition output rows streams
    W once (2 bytes/elem). Each thread owns a K-slice of block_K =
    reduce_thread * micro_size_k elems (micro_size_k = 128-bit transaction /
    16-bit bf16 = 8); partials reduce across the warp. f32 accumulate of the
    bf16 products is exactly what WGMMA does (bf16*bf16 is exact in f32).
    Roofline = (N*K*2 + 2K) bytes / HBM BW.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py
    #   @ tilelang main (dequantize_gemv, split-K + tvm_thread_allreduce path)
    # Adapted: bf16 W streamed directly (the dequant stage disappears), f32
    #   output (matches the WGMMA path's dtype); M fixed at 1 (decode) so the
    #   grid has no M dim.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_bf16_gemv(X, W, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro_size_k = 8  # 128-bit transaction / 16-bit bf16
        block_K = reduce_thread * micro_size_k
        X: T.Tensor((1, K), "bfloat16")
        W: T.Tensor((N, K), "bfloat16")
        Y = T.empty((1, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            X_local = T.alloc_local((micro_size_k,), "bfloat16")
            W_local = T.alloc_local((micro_size_k,), "bfloat16")
            acc = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            for ko in T.serial(T.ceildiv(K, block_K)):
                base = ko * block_K + kr * micro_size_k
                for v in T.vectorized(micro_size_k):
                    X_local[v] = X[0, base + v]
                for v in T.vectorized(micro_size_k):
                    W_local[v] = W[n, base + v]
                for ki in T.serial(micro_size_k):
                    acc[0] += T.cast(X_local[ki], "float32") * T.cast(W_local[ki], "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), acc[0], True, reduced[0], kr, dtype="handle"
                    )
                )
            if kr == 0:
                Y[0, n] = reduced[0]
        return Y

    return linear_bf16_gemv


# ---------------------------------------------------------------- linear fp4 (GEMV, sm70/Volta)


def make_linear_fp4_gemv_sm70(target: str, GROUP: int = 4):
    """Fused e2m1 dequant + GEMV for Volta (sm70), the decode (M=1) path.

    X[1,K] f32, WQ uint8 [N,K//2] fp16-TWIDDLED (reference.twiddle_fp4_f16),
    Scale[N,K//block] f32. Y[0,n] = Res[0,n] + OScale[n] * sum_k X[0,k] *
    e2m1(WQ nibble) * Scale[n,k//block].

    sm70 has no bf16x2 math (sm_80+), so the sm90 tl_fp4_decode8 is dead here;
    the fp16 twin tl_fp4_decode8_f16 (prmt + shift/mask + mul.f16x2 2^14, all
    sm_53+) decodes 8 elems in ~15 ops vs the branch-free bit-synth's ~10/elem
    it replaces. Each thread's 16-elem slice stays in one scale block (block_K =
    reduce_thread*16), so one Scale lookup per tile; fp16 accumulation stays
    inside the block (peer to the sm90 bf16x2 GEMV), then one f32 add per tile.
    Loads run in C straight from global — TIR locals handed to an extern by
    pointer land in local memory.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp4_gemv_sm70(X, WQ, Scale, OScale, Res, reduce_thread, n_partition, block):
        N, K = T.const("N, K")
        micro = 16  # one scale block (NVFP4 block=16); 8 twiddled bytes = 1 decode pair
        block_K = reduce_thread * micro
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        X: T.Tensor((1, K), "float32")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        OScale: T.Tensor((N,), "float32")
        Res: T.Tensor((1, N), "float32")
        Y = T.empty((1, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            T.import_source(_FP4_TWIDDLE_SRC_F16)
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            acc = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            sc = T.alloc_local((GROUP,), "float32")
            for kg in T.serial(num_g):
                base = kg * GROUP * block_K + kr * micro
                for g in T.unroll(GROUP):
                    sc[g] = Scale[n, (base + g * block_K) // block]
                T.call_extern(
                    f"tl_fp4_gemv_tiles_f16<{GROUP}>",
                    T.access_ptr(WQ[n, base // 2], "r"),
                    T.access_ptr(X[0, base], "r"),
                    block_K,
                    T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"),
                    dtype="void",
                )
            for kt in T.serial(num_ko - num_g * GROUP):  # K-tail, one tile at a time
                base = (num_g * GROUP + kt) * block_K + kr * micro
                sc[0] = Scale[n, base // block]
                T.call_extern(
                    "tl_fp4_gemv_tiles_f16<1>",
                    T.access_ptr(WQ[n, base // 2], "r"),
                    T.access_ptr(X[0, base], "r"),
                    block_K,
                    T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"),
                    dtype="void",
                )
            reduced = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), acc[0], True, reduced[0], kr, dtype="handle"
                    )
                )
            if kr == 0:
                Y[0, n] = Res[0, n] + reduced[0] * OScale[n]
        return Y

    return linear_fp4_gemv_sm70


def make_linear_fp4_gemv_sm70_m(
    target: str, M: int = 8, GROUP: int = 4, xh: bool = False, sh: bool = False
):
    """M-row (decode-batch) twin of make_linear_fp4_gemv_sm70.

    X[M,K] f32 (f16 when ``xh``), WQ[N,K//2] fp16-TWIDDLED, Scale[N,K//block]
    f32 (f16 when ``sh``), OScale[N] f32, Res[M,N] f32 -> Y[M,N] f32. WQ is
    loaded + decoded ONCE per tile and reused across all M rows, so the weight
    bytes do not scale with M. This is the sm70 decode-batch path (M=2..16),
    replacing the per-row GEMV loop (M launches/layer, OOM-prone). M is a
    compile-time factory arg; the backend pads M up and slices. ``xh`` takes X
    pre-packed as f16: same numerics (both round to nearest f16), half the X
    traffic, and 32 fewer instructions per row than converting inside the tile
    loop.

    ``sh`` narrows the Scale plane to f16 in global memory only — the tile loop
    still hands the extern f32, so the dequant math is untouched.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp4_gemv_sm70_m(X, WQ, Scale, OScale, Res, reduce_thread, n_partition, block):
        N, K = T.const("N, K")
        micro = 16  # 8 twiddled bytes = 1 decode pair; block is 16 or 32 (>= micro)
        block_K = reduce_thread * micro
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        tiles = "tl_fp4_gemv_tiles_f16_m_xh" if xh else "tl_fp4_gemv_tiles_f16_m"
        # sh must be read in plain Python before the annotations, or tilelang's
        # builder cannot resolve it (errors/2026-09-02-tilelang-closure-must-be-
        # read-before-annotation.md).
        s_dtype = "float16" if sh else "float32"
        X: T.Tensor((M, K), "float16" if xh else "float32")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // block), s_dtype)
        OScale: T.Tensor((N,), "float32")
        Res: T.Tensor((M, N), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            T.import_source(_FP4_TWIDDLE_SRC_F16)
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            acc = T.alloc_local((M,), "float32")
            for m in T.unroll(M):
                acc[m] = 0.0
            sc = T.alloc_local((GROUP,), "float32")
            for kg in T.serial(num_g):
                base = kg * GROUP * block_K + kr * micro
                for g in T.unroll(GROUP):
                    sc[g] = Scale[n, (base + g * block_K) // block]
                T.call_extern(
                    f"{tiles}<{GROUP},{M}>",
                    T.access_ptr(WQ[n, base // 2], "r"),
                    T.access_ptr(X[0, base], "r"),
                    K,
                    block_K,
                    T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"),
                    dtype="void",
                )
            for kt in T.serial(num_ko - num_g * GROUP):  # K-tail, one tile at a time
                base = (num_g * GROUP + kt) * block_K + kr * micro
                sc[0] = Scale[n, base // block]
                T.call_extern(
                    f"{tiles}<1,{M}>",
                    T.access_ptr(WQ[n, base // 2], "r"),
                    T.access_ptr(X[0, base], "r"),
                    K,
                    block_K,
                    T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"),
                    dtype="void",
                )
            T.call_extern(f"tl_warp_reduce_m_f16<{M}>", T.access_ptr(acc, "rw"), dtype="void")
            if kr == 0:
                for m in T.unroll(M):
                    Y[m, n] = Res[m, n] + acc[m] * OScale[n]
        return Y

    return linear_fp4_gemv_sm70_m


# ---------------------------------------------------------------- linear fp8 (GEMV)


def make_linear_fp8_gemv(target: str, M: int = 1, GROUP: int = 4):
    """GEMV (sm90), the decode path of linear_fp8: X[M,K] bf16 @ W8[N,K]
    e4m3 with per-128-block scale -> Y[M,N] f32. ``M`` is a compile-time row
    count: W is streamed and decoded once and reused across all M rows, so the
    weight bytes (the bottleneck) do not scale with M.

    Same split-K + warp-reduce schedule as make_linear_bf16_gemv, but W is
    e4m3 (micro_size_k=16, 128-bit/8-bit) and each thread's 16-elem slice
    stays within one 128-block scale (block_K=512=4 scale blocks), so one
    WScale lookup per chunk. Roofline = (N*K*1.03 + 2K) bytes / HBM BW.

    GROUP chunks are loaded before any math; each 16-elem tile is
    ``tl_fp8_gemv_tile16`` (C): e4m3 -> bf16x2 by bit placement (prmt + lop3,
    times 2^120), 8 ``fma.rn.bf16x2``, one f32 scale-accumulate per tile. The
    scalar cvt+FMA loop it replaces issued 7.3 instr/elem at 76-85% SM busy
    with DRAM at 54% (ncu 2026-08-28, in the real model); this is ~3.2.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py
    #   @ tilelang main (dequantize_gemv, split-K + tvm_thread_allreduce path)
    # Adapted: e4m3 W streamed directly (1 byte/elem vs the bf16 GEMV's 2),
    #   per-128-block f32 scale applied per chunk; M rows share one W stream.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp8_gemv(X, W8, WScale, OScale, Res, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro_size_k = 16  # 128-bit transaction / 8-bit e4m3
        block_K = reduce_thread * micro_size_k  # 512 = 4 scale blocks of 128
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        X: T.Tensor((M, K), "bfloat16")
        W8: T.Tensor((N, K), "float8_e4m3fn")
        WScale: T.Tensor((T.ceildiv(N, 128), T.ceildiv(K, 128)), "float32")
        OScale: T.Tensor((N,), "float32")  # per-row epilogue scale, folded (was a torch mul)
        Res: T.Tensor((M, N), "float32")  # residual stream (zeros when none): Y = Res + y
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            T.import_source(_FP4_TWIDDLE_SRC)
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            acc = T.alloc_local((M,), "float32")
            for m in T.unroll(M):
                acc[m] = 0.0
            sc = T.alloc_local((GROUP,), "float32")
            for kg in T.serial(num_g):
                base = kg * GROUP * block_K + kr * micro_size_k
                for g in T.unroll(GROUP):
                    # the 16-elem slice never crosses a 128-block
                    sc[g] = WScale[n // 128, (base + g * block_K) // 128]
                T.call_extern(
                    f"tl_fp8_gemv_tiles_m<{GROUP}, {M}>", T.access_ptr(W8[n, base], "r"),
                    T.access_ptr(X[0, base], "r"), block_K, K, T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"), dtype="void",
                )
            for kt in T.serial(num_ko - num_g * GROUP):  # K-tail, one tile at a time
                base = (num_g * GROUP + kt) * block_K + kr * micro_size_k
                sc[0] = WScale[n // 128, base // 128]
                T.call_extern(
                    f"tl_fp8_gemv_tiles_m<1, {M}>", T.access_ptr(W8[n, base], "r"),
                    T.access_ptr(X[0, base], "r"), block_K, K, T.access_ptr(sc, "r"),
                    T.access_ptr(acc, "rw"), dtype="void",
                )
            T.call_extern(f"tl_warp_reduce_m<{M}>", T.access_ptr(acc, "rw"), dtype="void")
            if kr == 0:
                for m in T.unroll(M):
                    Y[m, n] = Res[m, n] + acc[m] * OScale[n]
        return Y

    return linear_fp8_gemv


def make_linear_fp8_mma8(target: str, NG: int = 4, KW: int = 4, G: int = 4):
    """fp8 twin of make_linear_fp4_mma8 (per-128-block scale)."""
    NB = NG * 8

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp8_mma8(X, W8, WScale, OScale, Res):
        N, K = T.const("N, K")
        nchunk = K // 32
        per = T.ceildiv(nchunk, KW)
        X: T.Tensor((8, K), "bfloat16")
        W8: T.Tensor((N, K), "float8_e4m3fn")
        WScale: T.Tensor((T.ceildiv(N, 128), T.ceildiv(K, 128)), "float32")
        OScale: T.Tensor((N,), "float32")
        Res: T.Tensor((8, N), "float32")
        Y = T.empty((8, N), "float32")
        with T.Kernel(T.ceildiv(N, NB), threads=(32, KW)) as bx:
            T.import_source(_FP4_TWIDDLE_SRC)
            lane = T.thread_binding(0, 32, thread="threadIdx.x")
            kw = T.thread_binding(0, KW, thread="threadIdx.y")
            n0 = bx * NB
            g = lane // 4
            q = lane % 4
            acc = T.alloc_local((NG * 4,), "float32")
            red = T.alloc_shared((KW, 8, NB), "float32")
            # whole per-warp K range in one extern call (register accumulators).
            # A 32-row block never crosses a 128-row scale block, so one scale
            # row serves all NG groups; chunk c reads column c >> 2.
            T.call_extern(
                f"tl_fp8_mma_rows<{NG}, {G}>",
                T.access_ptr(W8[n0 + g, 8 * q], "r"), 8 * K,
                T.access_ptr(X[g, 8 * q], "r"),
                T.access_ptr(WScale[(n0 + g) // 128, 0], "r"), 2,
                kw * per, T.min(nchunk, (kw + 1) * per),
                T.access_ptr(acc, "w"), dtype="void",
            )
            for grp in T.unroll(NG):
                red[kw, g, 8 * grp + 2 * q] = acc[4 * grp]
                red[kw, g, 8 * grp + 2 * q + 1] = acc[4 * grp + 1]
            T.tvm_storage_sync("shared")
            for i in T.Parallel(8 * NB):
                r = i // NB
                col = i % NB
                tot = T.alloc_fragment((1,), "float32")
                tot[0] = 0.0
                for w in T.serial(KW):
                    tot[0] += red[w, r, col]
                Y[r, n0 + col] = Res[r, n0 + col] + tot[0] * OScale[n0 + col]
        return Y

    return linear_fp8_mma8


# ---------------------------------------------------------------- quant fp8 (per-token)


def make_quant_fp8_e4m3(target: str):
    """Per-token dynamic quant: X [M,K] bf16 -> XQ [M,K] e4m3 + Scale [M] f32.

    ``Scale[m] = FP8_MAX / max(|X[m,:]|)``; ``XQ = (X * Scale).to(e4m3)``. One
    block per row; threads reduce the row absmax via shared memory, then each
    thread quantizes its strided K-slice. The activation dequant
    (``Y = XQ @ W / Scale``) lives in the linear_fp4_fp8 epilogue.

    Per-token (not per-32-block) scale: the e4m3 multiplicative quant error
    (~2% RMS) does not average down over K either way, and a per-token scale
    lets the gemm epilogue be a single per-row divide (no per-K-tile temp
    fragment, which breaks the WGMMA pipeline and costs ~2x).

    # Original: the per-token W8A8 quant from vLLM/sglang
    #   (examples/cast/example_per_token_cast_to_fp8.py is the triton form),
    #   written as a tilelang block-per-row kernel with a shared-memory max
    #   reduction (same idiom as the softmax/gdn kernels in this file). e4m3fn
    #   finite max is 448; the per-token scale maps each row's absmax there.
    """
    FP8_MAX = 448.0  # e4m3fn finite max

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def quant_fp8(X, XQ, Scale, threads):
        M, K = T.const("M, K")
        X: T.Tensor((M, K), "bfloat16")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        Scale: T.Tensor((M,), "float32")
        with T.Kernel(M, threads=threads) as bx:
            tid = T.get_thread_binding(0)
            red = T.alloc_shared((threads,), "float32")
            amax = T.alloc_fragment((1,), "float32")
            amax[0] = 0.0
            for k in T.serial(T.ceildiv(K, threads)):
                idx = tid + k * threads
                if idx < K:
                    v = T.cast(X[bx, idx], "float32")
                    amax[0] = T.max(amax[0], T.max(v, 0.0 - v))
            red[tid] = amax[0]
            T.tvm_storage_sync("shared")
            if tid == 0:
                m = T.alloc_fragment((1,), "float32")
                m[0] = 0.0
                for i in T.serial(threads):
                    m[0] = T.max(m[0], red[i])
                red[0] = FP8_MAX / T.max(m[0], 1e-12)
            T.tvm_storage_sync("shared")
            s = red[0]
            for k in T.serial(T.ceildiv(K, threads)):
                idx = tid + k * threads
                if idx < K:
                    XQ[bx, idx] = T.cast(T.cast(X[bx, idx], "float32") * s, "float8_e4m3fn")
            if tid == 0:
                Scale[bx] = s

    return quant_fp8


# ---------------------------------------------------------------- linear fp4 (fp8 MMA)


def make_linear_fp4_fp8_mma(target: str, k_split: int = 1):
    """Fused e2m1 dequant-to-e4m3 + fp8 WGMMA (sm90 prefill path).

    XQ [M,K] e4m3 (per-token quantized activation, from make_quant_fp8_e4m3),
    WQ uint8 [N,K//2] twiddled (reference.twiddle_fp4), WScale [N,K//block] f32 (the
    checkpoint's scale block, 16 or 32), AScale [M] f32 (per-token scale).
    ``Y[m,n] = (sum_k XQ[m,k] * e2m1(WQ) * WScale[n,k//block]) / AScale[m]``.

    Same dequant schedule as make_linear_fp4_mma (the vectorized shared-memory
    macro, SOTA: examples/dequantize_gemm/..._bf16_fp4_hopper.py), with the
    dequant target dtype e4m3 (16 elems per 128-bit transaction vs 8 for
    bf16): the K-loop copies XQ/WQ/Scale tiles to shared, vectorizes the
    dequant into W_shared (one scale per 16-elem chunk), then fp8 WGMMA reads
    W_shared — with num_stages=3 the dequant of stage k+1 issues while the
    WGMMA of stage k is in flight. The e2m1 grid ({0,±.5,±1,±1.5,±2,±3,±4,±6})
    is an exact subset of e4m3, so the dequant is: nibble -> fp32 grid
    (integer fast decode) -> *WScale -> cast to e4m3. The cast is a requant, so
    WScale must arrive renormalized (`reference.renorm_fp4_scale`) with
    ``6 * WScale`` inside e4m3's normal range: 2.3% weight error there, 50%
    when checkpoint-native magnitudes saturate. The per-token activation scale
    is one divide in the epilogue.

    k_split > 1 adds a K-split grid dim: each (bx, by, bk) block sums
    K/k_split K-tiles and f32 atomic-adds its partial into a caller-zeroed Y
    (the AScale divide distributes over the split sum). The caller must zero
    Y before launch and pad K to a multiple of _FP4_BLOCK_K * k_split.
    k_split=1 is the shipped kernel below, unchanged.

    # SOTA copy: examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py
    #   @ tilelang main (fast_dequant path: vectorized shared dequant before
    #   the WGMMA, pipelined)
    # Adapted: e4m3 operands (fp8 WGMMA, f32 accumulate), the dequant target
    #   dtype is e4m3 (requant) instead of bf16; per-token activation dequant
    #   (1/AScale[m]) in the epilogue. Padded WQ bytes (0x00) decode to 0.0.
    # SOTA copy (k_split > 1): examples/gemm_streamk/example_tilelang_gemm_streamk.py
    #   @ tilelang main (split-K grid + atomic-add reduction family)
    # Adapted: fixed 2-way equal K-split (not stream-K scheduling) on the
    #   dequant+WGMMA body above; f32 atomic add into a zeroed output.
    """

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4_fp8(XQ, WQ, WScale, AScale, block_M, block_N, block, threads):
        threads = 128 if block_M >= 32 else threads
        block_N = _FP4_BLOCK_N  # 64-tile: doubles the N-grid vs the caller's
        # 128, putting every prefill shape at 2+ waves (the dequant/WGMMA
        # phases no longer align across resident blocks). See _FP4_BLOCK_N.
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // block), "float32")
        AScale: T.Tensor((M,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _FP4_BLOCK_K), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _FP4_BLOCK_K), "float8_e4m3fn")
            Scale_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // block), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(K // _FP4_BLOCK_K, num_stages=3):
                T.copy(XQ[by * block_M, k * _FP4_BLOCK_K], X_shared)
                T.copy(WQ[bx * block_N, k * _FP4_BLOCK_K // 2], WQ_shared)
                T.copy(WScale[bx * block_N, k * _FP4_BLOCK_K // block], Scale_shared)
                _dequant_fp4_macro("float8_e4m3fn", 16, block)(
                    WQ_shared, Scale_shared, W_shared, block_N, _FP4_BLOCK_K
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] / AScale[by * block_M + i]
            T.copy(C_local, Y[by * block_M, bx * block_N])
        return Y

    if k_split == 1:
        return linear_fp4_fp8

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4_fp8_split(XQ, WQ, WScale, AScale, Y, block_M, block_N, block, threads):
        threads = 128 if block_M >= 32 else threads
        block_N = _FP4_BLOCK_N
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        WQ: T.Tensor((N, K // 2), "uint8")
        WScale: T.Tensor((N, K // block), "float32")
        AScale: T.Tensor((M,), "float32")
        Y: T.Tensor((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), k_split, threads=threads) as (
            bx,
            by,
            bk,
        ):
            X_shared = T.alloc_shared((block_M, _FP4_BLOCK_K), "float8_e4m3fn")
            WQ_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, _FP4_BLOCK_K), "float8_e4m3fn")
            Scale_shared = T.alloc_shared((block_N, _FP4_BLOCK_K // block), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            k0 = bk * (K // k_split // _FP4_BLOCK_K)
            k1 = (bk + 1) * (K // k_split // _FP4_BLOCK_K)
            for k in T.Pipelined(k1 - k0, num_stages=3):
                kk = k0 + k
                T.copy(XQ[by * block_M, kk * _FP4_BLOCK_K], X_shared)
                T.copy(WQ[bx * block_N, kk * _FP4_BLOCK_K // 2], WQ_shared)
                T.copy(WScale[bx * block_N, kk * _FP4_BLOCK_K // block], Scale_shared)
                _dequant_fp4_macro("float8_e4m3fn", 16, block)(
                    WQ_shared, Scale_shared, W_shared, block_N, _FP4_BLOCK_K
                )
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] / AScale[by * block_M + i]
            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(Y[by * block_M + i, bx * block_N + j], C_local[i, j])

    return linear_fp4_fp8_split


# ---------------------------------------------------------------- linear fp8 (native MMA)


def make_linear_fp8_mma(target: str):
    """Native fp8 WGMMA linear (sm90 prefill path): e4m3 weights + e4m3
    activations, per-128-block weight scale, per-token activation scale.

    XQ [M,K] e4m3 (per-token quantized, from make_quant_fp8_e4m3),
    W8 [N,K] e4m3 (checkpoint-native, no dequant),
    WScale [ceil(N/128), K//128] f32 (per-128-block weight scale),
    AScale [M] f32 (per-token activation scale).
    ``Y[m,n] = (sum_k XQ[m,k] * W8[n,k] * WScale[n//128, k//128]) / AScale[m]``.

    The per-128-block weight scale is applied to the accumulator per K-chunk
    (block_K=128 = the fp8 WGMMA K=32 x 4 steps): C_local holds the chunk's
    unscaled MMA result, C_accum += C_local * WScale — the deepgemm 2xAcc
    pattern. No K-loop dequant: the weight operand is native e4m3, so the
    loop body is copy+copy+gemm+scale with no requant cast (the fp4 prefill
    path's dequant-to-e4m3 is what held it at 16-22% of peak).

    # SOTA copy: examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py
    #   @ tilelang main (per-128-block scale, C_local_accum += C_local *
    #   (scales_a * scales_b) per chunk, T.clear per chunk)
    # Adapted: per-token activation scale (one divide in the epilogue) instead
    #   of per-block scales_a; f32 output; e4m3 IO (fp8 WGMMA, f32 accumulate);
    #   WScale in the checkpoint's native [N//128, K//128] layout (one scalar
    #   per N-tile, broadcast over the fragment).
    """
    _BLOCK_K = 128  # matches the checkpoint's 128-block scale; fp8 WGMMA K=32 x 4

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp8(XQ, W8, WScale, AScale, block_M, block_N, threads):
        threads = 128 if block_M >= 32 else threads
        M, N, K = T.const("M, N, K")
        XQ: T.Tensor((M, K), "float8_e4m3fn")
        W8: T.Tensor((N, K), "float8_e4m3fn")
        WScale: T.Tensor((T.ceildiv(N, 128), K // 128), "float32")
        AScale: T.Tensor((M,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, _BLOCK_K), "float8_e4m3fn")
            W_shared = T.alloc_shared((block_N, _BLOCK_K), "float8_e4m3fn")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_accum = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_accum)
            T.clear(C_local)
            for k in T.Pipelined(K // _BLOCK_K, num_stages=4):
                T.copy(XQ[by * block_M, k * _BLOCK_K], X_shared)
                T.copy(W8[bx * block_N, k * _BLOCK_K], W_shared)
                scale_b = WScale[bx * block_N // 128, k]
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_accum[i, j] += C_local[i, j] * scale_b
                T.clear(C_local)
            for i, j in T.Parallel(block_M, block_N):
                C_accum[i, j] = C_accum[i, j] / AScale[by * block_M + i]
            T.copy(C_accum, Y[by * block_M, bx * block_N])
        return Y

    return linear_fp8


def make_linear_fp4_bwd_mma(target: str, local_size: int = 8, block: int = 16):
    """gx = grad @ W for a frozen packed weight: A [M,N] @ W [N,K] -> [M,K].

    The backward contracts over N — the weight's ROW index — so it reads whole
    rows of W, which is exactly how the packed bytes are stored. The slab drops
    straight into gemm_nn's B tile with no transpose, and the dequant happens in
    shared memory inside the K-loop, so the full bf16 weight is never
    materialized (the two-step version wrote and re-read ~160 GB per training
    step).

    The per-row epilogue scale is folded into ``grad`` by the caller: it scales
    weight row n, and [M,N] is far smaller than [N,K].
    """
    dequant = _dequant_fp4_macro("bfloat16", local_size, block)

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp4_bwd(A, WQ, Scale, block_M, block_N, threads):
        M, N = T.const("M, N")
        K2 = T.const("K2")
        K = K2 * 2
        A: T.Tensor((M, N), "bfloat16")
        WQ: T.Tensor((N, K2), "uint8")
        Scale: T.Tensor((N, K // block), "float32")
        C = T.empty((M, K), "float32")
        with T.Kernel(T.ceildiv(K, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, _RED_TILE), "bfloat16")
            WQ_shared = T.alloc_shared((_RED_TILE, block_N // 2), "uint8")
            Scale_shared = T.alloc_shared((_RED_TILE, block_N // block), "float32")
            W_shared = T.alloc_shared((_RED_TILE, block_N), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for n in T.Pipelined(N // _RED_TILE, num_stages=3):
                T.copy(A[by * block_M, n * _RED_TILE], A_shared)
                T.copy(WQ[n * _RED_TILE, bx * (block_N // 2)], WQ_shared)
                T.copy(Scale[n * _RED_TILE, bx * (block_N // block)], Scale_shared)
                dequant(WQ_shared, Scale_shared, W_shared, _RED_TILE, block_N)
                T.gemm(A_shared, W_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return linear_fp4_bwd
