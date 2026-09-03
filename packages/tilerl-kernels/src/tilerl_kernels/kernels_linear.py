"""Linear + quant tensor-core kernels for sm90 (bf16/fp8 WGMMA, f32
accumulate), ported from the tilelang gemm / dequantize_gemm / deepgemm
examples. bf16-IO; the fp4 kernels decode e2m1 by integer bit patterns."""

from __future__ import annotations

import os

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs

#: Reduction tile (K for gemm_nt/nn, M for gemm_tn): 2 WGMMA K-steps, divides
#: every model dim. The backend imports this name to pad, so it is defined once.
#: TILERL_RED_TILE A/Bs it (errors/2026-08-29-mma8-is-register-bound.md).
_RED_TILE = int(os.environ.get("TILERL_RED_TILE", "32"))
# 128 is the smallest K pad in backend._CUDA_PLAN; a tile that does not divide it
# truncates that kernel's reduction the same silent way.
if _RED_TILE <= 0 or _RED_TILE % 16 or 128 % _RED_TILE:
    raise ValueError(
        f"TILERL_RED_TILE must be a positive multiple of 16 dividing 128, got {_RED_TILE}")

# ---------------------------------------------------------------- gemm (MMA)


def make_gemm_nt_mma(target: str):
    """C = A @ B.T + Bias. A [M,K], B [N,K] -> C [M,N] (example_gemm.py, bf16 WGMMA)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gemm_nt(A, B, Bias, block_M, block_N, threads):
        # tiles under 32 rows cannot be partitioned across a 4-warp group
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
    """C = A @ B. A [M,K], B [K,N] -> C [M,N]."""

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
    """C = A.T @ B. A [M,N], B [M,K] -> C [N,K] (C_ij = sum_m A_mi B_mj)."""

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
    """Vectorized e2m1 dequant of a packed WQ tile into W_shared, one block
    scale per local_size chunk (example_dequant_gemm_bf16_fp4_hopper.py's
    fast_dequant macro over our twiddled bytes). The chunk loop is T.Parallel:
    a serial one blocks the K-loop pipeliner, ~20% on K=17408."""
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


#: fp4 K-tile: amortizes the dequant over several WGMMA steps; the scale block
#: (16 or 32) must divide it and the backend pads K to it.
_FP4_BLOCK_K = 64

#: fp4->fp8 prefill N-tile. 64, not the caller's 128: the 128 tile left the
#: N=5120 grids under one wave, +33% geo-mean TFLOP/s (scripts/_sweep_fp8_prefill.py).
_FP4_BLOCK_N = 64


def make_linear_fp4_mma(target: str):
    """Fused e2m1 dequant + bf16 WGMMA. X [M,K] bf16, WQ uint8 [N,K//2]
    twiddled (reference.twiddle_fp4), Scale [N,K//block] f32; the dequant of
    stage k+1 overlaps the WGMMA of stage k. Padded WQ bytes decode to 0."""

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


#: CUDA helpers behind T.call_extern. Loads are done in C with volatile asm:
#: nvcc sinks plain loads next to their use, and a TIR local handed to an
#: extern by pointer lives in local memory (wins/2026-08-28-fp8-gemv-bf16x2.md).
_FP4_TWIDDLE_SRC = r"""
// e2m1 x8 -> 4 x bf16x2 in 18 ops, from the twiddled byte layout
// (reference.twiddle_fp4; tilelang quantize/mxfp.py decode_fp4_to_bf16_twiddling).
// prmt selector 0x0123 keeps the upper 16 bits zero: CUDA 12.9 prmt.b32 truncates
// immediates (errors/2026-08-26-fp4-gemv-dequant-issue-rejected.md).
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
// e4m3 x2 (bytes 0 and 2 of t) -> bf16x2 by bit placement, then * 2^120
// (0x7B80) rebiases the exponent 7 -> 127; subnormals rebias exactly.
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
// Tensor-core decode GEMM for M <= 8 (Marlin-style): one warp, NG groups of 8
// output rows, the warp's whole K range in one call so the accumulators stay
// in registers. The twiddled layout is already a valid mma B fragment under a
// consistent k permutation: lane (g = l/4, q = l%4) loads 8 consecutive k of
// row 8*grp + g as d0..d3 (b0/b1 of k16 tile 0, then tile 1), virtual k
// {2q, 2q+1, 2q+8, 2q+9} standing for actual 8q+{0..3} (+4 for tile 1); the A
// fragment uses the same map from one LDG.128 of X row g, rows 8..15 zero.
// A lane's 8 elements share one scale block, applied on the B fragment.
// acc[grp*4 + {0,1}] = C rows g, cols 8*grp + 2q + {0,1}; {2,3} are junk rows.
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
// Wide-load twin of tl_fp4_mma_rows: v2.u32 per lane (16 fp4) halves the weight
// load instructions (errors/2026-08-29-mma8-is-register-bound.md). Lane q owns
// k in [64*cp + 16q, +16) of a 64-k chunk pair, X read at the same offset, so
// the virtual-k permutation stays consistent; 16q is aligned to both scale blocks.
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
  // 128-block scale: one scale row per 32-row block, chunk c reads column c >> 2
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
// Warp reduce of M accumulators (tvm_thread_allreduce takes one output buffer, not M)
template <int M>
__device__ __forceinline__ void tl_warp_reduce_m(float *acc) {
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
  }
}
// M-row GEMV tiles: W is decoded once and reused across M rows of X (xrow apart)
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
"""


def make_linear_fp4_gemv(target: str, M: int = 1, GROUP: int = 4):
    """Fused e2m1 dequant + GEMV, the decode path of linear_fp4
    (example_dequant_gemv_fp16xint4.py's split-K schedule, bf16x2 FMA tiles in
    C). X [M,K] bf16, WQ twiddled, Scale [N,K//block] f32, block % 16 == 0 so a
    16-elem tile never straddles a scale; ``M`` rows share one W stream.
    bf16 accumulation stays inside one scale block: relerr 4e-3."""

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
        OScale: T.Tensor((N,), "float32")  # per-row epilogue scale
        Res: T.Tensor((M, N), "float32")  # residual stream (zeros when none)
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
    """Marlin-style decode GEMM for M <= 8: X [8,K] bf16 (rows >= M zero) @
    twiddled fp4 -> Y [8,N] f32 (+ Res, * OScale). A block owns NG*8 rows; KW
    warps split K (``tl_fp4_mma_rows``) and reduce through shared memory.
    The scalar batched GEMV it replaced was register-bound
    (errors/2026-08-28-batched-scalar-gemv)."""
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
            # scale for chunk c of row n: Scale[n, (c*32 + 8q)//block]
            if W8:
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
    """bf16 GEMV, the M=1 path of linear: split-K across reduce_thread lanes,
    warp allreduce (example_dequant_gemv_fp16xint4.py without the dequant)."""

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


# ---------------------------------------------------------------- linear fp8 (GEMV)


def make_linear_fp8_gemv(target: str, M: int = 1, GROUP: int = 4):
    """fp8 twin of make_linear_fp4_gemv: e4m3 W with a per-128-block scale
    (a thread's 16-elem slice never crosses a block), bf16x2 FMA tiles in C."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def linear_fp8_gemv(X, W8, WScale, OScale, Res, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro_size_k = 16  # 128-bit transaction / 8-bit e4m3
        block_K = reduce_thread * micro_size_k
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        X: T.Tensor((M, K), "bfloat16")
        W8: T.Tensor((N, K), "float8_e4m3fn")
        WScale: T.Tensor((T.ceildiv(N, 128), T.ceildiv(K, 128)), "float32")
        OScale: T.Tensor((N,), "float32")
        Res: T.Tensor((M, N), "float32")
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
            # a 32-row block never crosses a 128-row scale block: one scale row for all NG groups
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
    """Per-token e4m3 quant: Scale[m] = 448 / max|X[m,:]|, XQ = (X * Scale).
    Per-token, not per-block, so the gemm epilogue is one per-row divide (a
    per-K-tile fragment breaks the WGMMA pipeline, ~2x)."""
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
    """w4a8: e2m1 dequant into e4m3 + fp8 WGMMA. XQ [M,K] e4m3 and AScale [M]
    from make_quant_fp8_e4m3, WQ twiddled, WScale [N,K//block] f32;
    Y = (XQ @ dequant(WQ).T) / AScale. The e4m3 cast is a requant, so WScale
    must be renormalized (reference.renorm_fp4_scale): 2.3% weight error
    there, 50% when checkpoint magnitudes saturate.
    k_split > 1: each K-slice block f32 atomic-adds into a caller-zeroed Y;
    the caller pads K to a multiple of _FP4_BLOCK_K * k_split."""

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def linear_fp4_fp8(XQ, WQ, WScale, AScale, block_M, block_N, block, threads):
        threads = 128 if block_M >= 32 else threads
        block_N = _FP4_BLOCK_N
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
    """Native fp8 WGMMA linear (example_deepgemm_fp8_2xAcc.py): XQ [M,K] e4m3,
    W8 [N,K] e4m3, WScale [ceil(N/128), K//128] applied per 128-K chunk to
    the accumulator, AScale [M] divided in the epilogue."""
    _BLOCK_K = 128  # the checkpoint's scale block

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
    Contracting over N reads whole packed rows, so the dequant happens in
    shared memory inside the loop and the bf16 weight is never materialized.
    The caller folds the per-row scale into grad."""
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
