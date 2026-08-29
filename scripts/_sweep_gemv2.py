"""Round-6 sweep: group4 without the X buffer. The grouped kernel's Xs[32]
bf16 (16 regs) is the biggest register chunk; X is shared across all rows and
fits in L1 (2 KB), so reloading it during the FMA may recover occupancy at
the cost of L1 loads (which use the idle load pipe). Tests both shape
orientations (N=17408,K=5120 and N=5120,K=17408) since the bench showed
noisy per-linear results.

Diagnostic only — not shipped.
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T

sys.path.insert(0, "src")
from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import linear_fp4, pack_fp4

b = get_backend()

MICRO = 8
GROUP = 4


def _e2m1fn_fp32(nib):
    ni32 = T.cast(nib, "int32")
    bits = ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
    return T.reinterpret(bits, "float32")


def _shfl(lut, byte, ki):
    nib = (byte >> ((ki % 2) * 4)) & 15
    return T.tvm_warp_shuffle(0xFFFFFFFF, lut, T.cast(nib, "int32"), 32, 32)


def make_variant(target, mode):
    """mode='flat': round-1 kernel. mode='group4': grouped with X buffer.
    mode='noxbuf': grouped, X reloaded during FMA (no Xs buffer)."""

    @tilelang.jit(target=target, pass_configs={"tl.disable_data_race_check": True})
    def ker(X, WQ, Scale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        block_K = reduce_thread * MICRO
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
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            red = T.alloc_local((1,), "float32")
            lut = _e2m1fn_fp32(kr & 15)
            acc[0] = 0.0
            if mode == "flat":
                X_local = T.alloc_local((MICRO,), "bfloat16")
                WQ_local = T.alloc_local((MICRO // 2,), "uint8")
                for ko in T.serial(num_ko):
                    base = ko * block_K + kr * MICRO
                    for v in T.vectorized(MICRO):
                        X_local[v] = X[0, base + v]
                    for v in T.vectorized(MICRO // 2):
                        WQ_local[v] = WQ[n, base // 2 + v]
                    partial[0] = 0.0
                    for ki in T.unroll(MICRO):
                        partial[0] += T.cast(X_local[ki], "float32") * _shfl(
                            lut, WQ_local[ki // 2], ki
                        )
                    acc[0] += Scale[n, base // 32] * partial[0]
            else:
                Ws = T.alloc_local((GROUP, MICRO // 2), "uint8")
                ws = T.alloc_local((GROUP, MICRO), "float32")
                if mode == "group4":
                    Xs = T.alloc_local((GROUP, MICRO), "bfloat16")
                for kg in T.serial(num_g):
                    for g in T.unroll(GROUP):
                        base = (kg * GROUP + g) * block_K + kr * MICRO
                        if mode == "group4":
                            for v in T.vectorized(MICRO):
                                Xs[g, v] = X[0, base + v]
                        for v in T.vectorized(MICRO // 2):
                            Ws[g, v] = WQ[n, base // 2 + v]
                    for g in T.unroll(GROUP):
                        for ki in T.unroll(MICRO):
                            ws[g, ki] = _shfl(lut, Ws[g, ki // 2], ki)
                    for g in T.unroll(GROUP):
                        base = (kg * GROUP + g) * block_K + kr * MICRO
                        partial[0] = 0.0
                        for ki in T.unroll(MICRO):
                            if mode == "group4":
                                xv = T.cast(Xs[g, ki], "float32")
                            else:
                                xv = T.cast(X[0, base + ki], "float32")
                            partial[0] += xv * ws[g, ki]
                        acc[0] += Scale[n, base // 32] * partial[0]
                for kt in T.serial(num_ko - num_g * GROUP):
                    base = (num_g * GROUP + kt) * block_K + kr * MICRO
                    for v in T.vectorized(MICRO // 2):
                        Ws[0, v] = WQ[n, base // 2 + v]
                    partial[0] = 0.0
                    for ki in T.unroll(MICRO):
                        if mode == "group4":
                            xv = T.cast(Xs[0, ki], "float32")
                        else:
                            xv = T.cast(X[0, base + ki], "float32")
                        partial[0] += xv * _shfl(lut, Ws[0, ki // 2], ki)
                    acc[0] += Scale[n, base // 32] * partial[0]
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(T.uint32(1), acc[0], True, red[0], kr, dtype="handle")
                )
            if kr == 0:
                Y[0, n] = T.cast(red[0], "bfloat16")
        return Y

    return ker


def time_call(fn, iters=50):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


bw = 3312.0
for N, K in ((17408, 5120), (5120, 17408)):
    torch.manual_seed(0)
    wq, sc = pack_fp4(torch.randn(N, K))
    x = torch.randn(1, K, device=b.device)
    wq, sc = wq.to(b.device), sc.to(b.device)
    Kp = ((K + 255) // 256) * 256
    Np = ((N + 3) // 4) * 4
    xp = torch.nn.functional.pad(x, (0, Kp - K)).bfloat16()
    wqp = torch.nn.functional.pad(wq, (0, (Kp - K) // 2, 0, Np - N))
    sp = torch.nn.functional.pad(sc, (0, (Kp - K) // 32, 0, Np - N))
    ref = linear_fp4(x.cpu(), wq.cpu(), sc.cpu())
    roof = (N * K * 0.75 + 4 * K) / bw / 1e9 * 1e3
    print(f"\n--- N={N} K={K}  roof {roof:.4f} ms ---", flush=True)
    for mode in ("flat", "group4", "noxbuf"):
        ker = make_variant("cuda", mode)
        t0 = time.perf_counter()
        y = ker(xp, wqp, sp, 32, 4)[:, :N].float()
        torch.cuda.synchronize()
        jit = time.perf_counter() - t0
        err = (y.cpu() - ref).abs().max().item() / ref.abs().max().item()
        ms = time_call(lambda: ker(xp, wqp, sp, 32, 4))
        print(
            f"{mode:<10}: {ms:.4f} ms  ({100 * roof / ms:5.1f}% roof, rel-err {err:.2e}, jit {jit:.0f}s)",
            flush=True,
        )
