"""Matrix A/B for the fp4 decode GEMV: every structural cell in one pod session, relerr + time vs shipped.
Cells (CELLS=a,b selects): gemv_shuffle (control) gemv_prmt mma_bitcast_m16 mma_prmt_ext_m{16,32,64} mma_prmt_buf_m16.
Usage: CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python3 scripts/_matrix_gemv.py [shape_idx]
"""

from __future__ import annotations

import os
import sys
import time

import tilelang
import tilelang.language as T
import torch
from tilerl_kernels.kernels_linear import make_linear_fp4_gemv, make_linear_fp4_mma
from tilerl_kernels.reference import unpack_fp4

DEQUANT_SRC = """
__device__ void dequant_e2m1fn_prmt(const unsigned char* __restrict__ packed,
                                    void* __restrict__ out_v) {
    const unsigned int p = *reinterpret_cast<const unsigned int*>(packed);
    const unsigned int lo = p & 0x0F0F0F0Fu;
    const unsigned int hi = (p >> 4) & 0x0F0F0F0Fu;
    unsigned short* out = reinterpret_cast<unsigned short*>(out_v);
    unsigned long long* out64 = reinterpret_cast<unsigned long long*>(out);
    // lo half: nibbles 0,2,4,6 (low nibble of each byte)
    const unsigned int idx_lo = lo & 0x07070707u;
    const unsigned int low_lo = __byte_perm(0xC0804000u, 0xC0804000u, idx_lo);
    unsigned int high_lo = __byte_perm(0x3F3F3F3Fu, 0x40404040u, idx_lo);
    high_lo |= (lo & 0x08080808u) << 4;
    const unsigned int lo_0 = __byte_perm(low_lo, high_lo, 0x05010400u);  // [n0, n2]
    const unsigned int lo_1 = __byte_perm(low_lo, high_lo, 0x07030602u);  // [n4, n6]
    // hi half: nibbles 1,3,5,7 (high nibble of each byte)
    const unsigned int idx_hi = hi & 0x07070707u;
    const unsigned int low_hi = __byte_perm(0xC0804000u, 0xC0804000u, idx_hi);
    unsigned int high_hi = __byte_perm(0x3F3F3F3Fu, 0x40404040u, idx_hi);
    high_hi |= (hi & 0x08080808u) << 4;
    const unsigned int hi_0 = __byte_perm(low_hi, high_hi, 0x05010400u);  // [n1, n3]
    const unsigned int hi_1 = __byte_perm(low_hi, high_hi, 0x07030602u);  // [n5, n7]
    // interleave to [n0,n1,n2,n3], [n4,n5,n6,n7]
    out64[0] = (unsigned long long)__byte_perm(lo_0, hi_0, 0x05040100u)
             | ((unsigned long long)__byte_perm(lo_0, hi_0, 0x07060302u) << 32);
    out64[1] = (unsigned long long)__byte_perm(lo_1, hi_1, 0x05040100u)
             | ((unsigned long long)__byte_perm(lo_1, hi_1, 0x07060302u) << 32);
}
"""


# ---------------------------------------------------------------- PRMT scalar GEMV

def make_prmt_scalar_gemv(target: str):
    """Shipped GEMV structure, PRMT extern dequant replacing the shuffle LUT."""
    GROUP = 4

    @tilelang.jit(target=target, pass_configs={"tl.disable_data_race_check": True})
    def kernel(X, WQ, Scale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro_size_k = 8
        block_K = reduce_thread * micro_size_k
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        X: T.Tensor((1, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            Xs = T.alloc_local((GROUP, micro_size_k), "bfloat16")
            # 1D so T.access_ptr(offset=) works (2D locals reject 1-index access)
            Ws = T.alloc_local((GROUP * micro_size_k // 2,), "uint8")
            Wb = T.alloc_local((GROUP * micro_size_k,), "bfloat16")
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            T.import_source(DEQUANT_SRC)
            acc[0] = 0.0
            for kg in T.serial(num_g):
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * micro_size_k
                    for v in T.vectorized(micro_size_k):
                        Xs[g, v] = X[0, base + v]
                    for v in T.vectorized(micro_size_k // 2):
                        Ws[g * 4 + v] = WQ[n, base // 2 + v]
                for g in T.unroll(GROUP):
                    T.call_extern(
                        "handle", "dequant_e2m1fn_prmt",
                        T.access_ptr(Ws, "r", offset=g * 4),
                        T.access_ptr(Wb, "w", offset=g * 8),
                    )
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * micro_size_k
                    partial[0] = 0.0
                    for ki in T.unroll(micro_size_k):
                        partial[0] += T.cast(Xs[g, ki], "float32") * T.cast(Wb[g * 8 + ki], "float32")
                    acc[0] += Scale[n, base // 32] * partial[0]
            for kt in T.serial(num_ko - num_g * GROUP):
                base = (num_g * GROUP + kt) * block_K + kr * micro_size_k
                for v in T.vectorized(micro_size_k):
                    Xs[0, v] = X[0, base + v]
                for v in T.vectorized(micro_size_k // 2):
                    Ws[v] = WQ[n, base // 2 + v]
                T.call_extern(
                    "handle", "dequant_e2m1fn_prmt",
                    T.access_ptr(Ws, "r"), T.access_ptr(Wb, "w"),
                )
                partial[0] = 0.0
                for ki in T.unroll(micro_size_k):
                    partial[0] += T.cast(Xs[0, ki], "float32") * T.cast(Wb[ki], "float32")
                acc[0] += Scale[n, base // 32] * partial[0]
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
                Y[0, n] = T.cast(reduced[0], "bfloat16")
        return Y

    return kernel


# ---------------------------------------------------------------- PRMT MMA

def make_prmt_mma(target: str, block_M: int = 16, block_N: int = 64,
                  block_K: int = 64, threads: int = 128, use_buf: bool = False):
    """PRMT extern dequant + mma.sync decode GEMV, M padded to block_M."""
    local_size = 8
    local_compress = local_size // 2

    @T.macro
    def dequant_prmt(WQ_shared, Scale_shared, W_shared):
        T.import_source(DEQUANT_SRC)
        tx = T.get_thread_binding()
        WQ_local = T.alloc_local((local_compress,), "uint8")
        W_local = T.alloc_local((local_size,), "bfloat16")
        for i in T.serial(0, block_N * block_K // threads // local_size):
            cidx = i * threads * local_compress + tx * local_compress
            for v in T.vectorized(local_compress):
                WQ_local[v] = WQ_shared[
                    (cidx + v) // (block_K // 2), (cidx + v) % (block_K // 2)
                ]
            T.call_extern(
                "handle", "dequant_e2m1fn_prmt",
                T.access_ptr(WQ_local, "r"), T.access_ptr(W_local, "w"),
            )
            oidx = i * threads * local_size + tx * local_size
            s = Scale_shared[oidx // block_K, (oidx % block_K) // 32]
            for v in T.vectorized(local_size):
                idx = oidx + v
                W_shared[idx // block_K, idx % block_K] = T.cast(
                    T.cast(W_local[v], "float32") * s, "bfloat16"
                )

    @T.macro
    def dequant_prmt_buf(WQ_shared, Scale_shared, W_buf, W_shared):
        T.import_source(DEQUANT_SRC)
        tx = T.get_thread_binding()
        WQ_local = T.alloc_local((local_compress,), "uint8")
        W_local = T.alloc_local((local_size,), "bfloat16")
        for i in T.serial(0, block_N * block_K // threads // local_size):
            cidx = i * threads * local_compress + tx * local_compress
            for v in T.vectorized(local_compress):
                WQ_local[v] = WQ_shared[
                    (cidx + v) // (block_K // 2), (cidx + v) % (block_K // 2)
                ]
            T.call_extern(
                "handle", "dequant_e2m1fn_prmt",
                T.access_ptr(WQ_local, "r"), T.access_ptr(W_local, "w"),
            )
            oidx = i * threads * local_size + tx * local_size
            s = Scale_shared[oidx // block_K, (oidx % block_K) // 32]
            for v in T.vectorized(local_size):
                idx = oidx + v
                W_buf[idx // block_K, idx % block_K] = T.cast(
                    T.cast(W_local[v], "float32") * s, "bfloat16"
                )
        T.copy(W_buf, W_shared)

    @tilelang.jit(
        target=target,
        pass_configs={
            "tl.disable_data_race_check": True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def kernel(X, WQ, Scale):
        M, N, K = T.const("M, N, K")
        X: T.Tensor((M, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), "bfloat16")
            WQ_shared = T.alloc_shared((block_N, block_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, block_K), "bfloat16")
            Scale_shared = T.alloc_shared((block_N, block_K // 32), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            if use_buf:
                W_buf = T.alloc_shared((block_N, block_K), "bfloat16")
            for k in T.Pipelined(K // block_K, num_stages=3):
                T.copy(X[0, k * block_K], X_shared)
                T.copy(WQ[bx * block_N, k * block_K // 2], WQ_shared)
                T.copy(Scale[bx * block_N, k * block_K // 32], Scale_shared)
                if use_buf:
                    dequant_prmt_buf(WQ_shared, Scale_shared, W_buf, W_shared)
                else:
                    dequant_prmt(WQ_shared, Scale_shared, W_shared)
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)
            for j in T.Parallel(block_N):
                Y[0, bx * block_N + j] = T.cast(C_local[0, j], "bfloat16")
        return Y

    return kernel


# ---------------------------------------------------------------- matrix runner

def timeit(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_matrix(N, K, name):
    print(f"\n{'='*70}\n{name}: N={N} K={K}\n{'='*70}", flush=True)
    dev = "cuda"
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=dev)
    wq = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scale = torch.rand((N, K // 32), device=dev) * 0.1 + 0.01
    x = torch.randn((1, K), dtype=torch.bfloat16, device=dev)
    ref = (x.float() @ unpack_fp4(wq, scale).float().T).squeeze(0)
    roof = N * K * 0.75 / (3.3e12) * 1e3

    def relerr(got):
        return (got.float() - ref).norm().item() / ref.norm().item()

    def pad_mma(bM, bK=64):
        Nm = ((N + 63) // 64) * 64
        Km = ((K + bK - 1) // bK) * bK
        xm = torch.zeros((bM, Km), dtype=torch.bfloat16, device=dev)
        xm[0] = torch.nn.functional.pad(x, (0, Km - K))[0]
        wqm = torch.nn.functional.pad(
            torch.nn.functional.pad(wq, (0, 0, 0, Nm - N)), (0, (Km - K) // 2))
        scm = torch.nn.functional.pad(
            torch.nn.functional.pad(scale, (0, 0, 0, Nm - N)), (0, Km // 32 - K // 32))
        return xm, wqm, scm

    cells = []
    sel = os.environ.get("CELLS", "gemv_shuffle,gemv_prmt,mma_bitcast_m16,mma_prmt_ext_m16,mma_prmt_ext_m32,mma_prmt_ext_m64,mma_prmt_buf_m16").split(",")

    def cell_gemv_shuffle():
        Kp = ((K + 255) // 256) * 256
        Np = ((N + 3) // 4) * 4
        gemv = make_linear_fp4_gemv("cuda")
        xg = torch.nn.functional.pad(x, (0, Kp - K))
        wqg = torch.nn.functional.pad(wq, (0, 0, 0, Np - N))
        scg = torch.nn.functional.pad(scale, (0, 0, 0, Np - N))
        return gemv, (xg, wqg, scg, 32, 4, 32), lambda: gemv(xg, wqg, scg, 32, 4, 32)[0, :N]
    if "gemv_shuffle" in sel:
        cells.append(("gemv_shuffle", cell_gemv_shuffle))

    def cell_gemv_prmt():
        Kp = ((K + 255) // 256) * 256
        Np = ((N + 3) // 4) * 4
        gemv = make_prmt_scalar_gemv("cuda")
        xg = torch.nn.functional.pad(x, (0, Kp - K))
        wqg = torch.nn.functional.pad(wq, (0, 0, 0, Np - N))
        scg = torch.nn.functional.pad(scale, (0, 0, 0, Np - N))
        return gemv, (xg, wqg, scg, 32, 4), lambda: gemv(xg, wqg, scg, 32, 4)[0, :N]
    if "gemv_prmt" in sel:
        cells.append(("gemv_prmt", cell_gemv_prmt))

    def cell_mma_bitcast_m16():
        xm, wqm, scm = pad_mma(16)
        mma = make_linear_fp4_mma("cuda")
        return mma, (xm, wqm, scm, 16, 64, 32, 64), lambda: mma(xm, wqm, scm, 16, 64, 32, 64)[0, :N]
    if "mma_bitcast_m16" in sel:
        cells.append(("mma_bitcast_m16", cell_mma_bitcast_m16))

    for bM in (16, 32, 64):
        def cell_mma_prmt(bM=bM):
            xm, wqm, scm = pad_mma(bM)
            kern = make_prmt_mma("cuda", block_M=bM)
            return kern, (xm, wqm, scm), lambda: kern(xm, wqm, scm)[0, :N]
        if f"mma_prmt_ext_m{bM}" in sel:
            cells.append((f"mma_prmt_ext_m{bM}", cell_mma_prmt))

    def cell_mma_prmt_buf_m16():
        xm, wqm, scm = pad_mma(16)
        kern = make_prmt_mma("cuda", block_M=16, use_buf=True)
        return kern, (xm, wqm, scm), lambda: kern(xm, wqm, scm)[0, :N]
    if "mma_prmt_buf_m16" in sel:
        cells.append(("mma_prmt_buf_m16", cell_mma_prmt_buf_m16))

    results = []
    for cname, cfn in cells:
        print(f"\n  --- {cname} ---", flush=True)
        try:
            kern, args, outfn = cfn()
            y = outfn()
            torch.cuda.synchronize()
            r = relerr(y)
            ms = timeit(outfn)
            pct = 100 * roof / ms
            results.append((cname, ms, r, pct, "OK"))
            print(f"  {cname:22s} {ms:.4f} ms  relerr={r:.2e}  {pct:.1f}% roof", flush=True)
        except Exception as e:
            msg = str(e).split("\n")[-2] if "\n" in str(e) else str(e)
            results.append((cname, 0, 0, 0, f"FAIL: {msg[:100]}"))
            print(f"  {cname:22s} FAIL: {msg[:150]}", flush=True)

    print(f"\n{'='*70}\nSUMMARY ({name})\n{'='*70}", flush=True)
    print(f"{'cell':22s} {'ms':>8s} {'x-vs-shuffle':>12s} {'relerr':>10s} {'%roof':>7s}  verdict")
    base_ms = next((r[1] for r in results if r[0] == "gemv_shuffle"), None)
    for cname, ms, r, pct, verdict in results:
        ratio = f"{base_ms/ms:.3f}x" if ms > 0 and base_ms else "-"
        print(f"{cname:22s} {ms:8.4f} {ratio:>12s} {r:10.2e} {pct:7.1f}  {verdict}")
    return results


def main():
    shapes = [
        (248320, 5120, "lm_head"),
        (34816, 5120, "gate_up(fused)"),
        (5120, 17408, "down_proj"),
    ]
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_matrix(*shapes[idx])


if __name__ == "__main__":
    main()
