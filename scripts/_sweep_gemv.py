"""Isolate the fp4 GEMV dequant on the shipped schedule (micro_size_k=8, partial-scale, warp-shuffle LUT).
Diagnostic only; bench_fp4_gemv.py is the committed bench.
H20 N=17408 K=5120: issue-bound. nodecode ~57% roof, lutshfl ~44%, bitcast ~30%; 2 acc / wider tiles /
shared-X / shared-LUT / 256-LUT / f32-X all worse.
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


def _e2m1fn_fp32(nib):
    # e2m1fn grid is power-of-two: sign<<31 | (126+e)<<23 | m<<22, no exp2
    ni32 = T.cast(nib, "int32")
    bits = ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
    return T.reinterpret(bits, "float32")


def make_variant(target, decode):
    @tilelang.jit(target=target, pass_configs={"tl.disable_data_race_check": True})
    def ker(X, WQ, Scale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        block_K = reduce_thread * MICRO
        X: T.Tensor((1, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            X_local = T.alloc_local((MICRO,), "bfloat16")
            WQ_local = T.alloc_local((MICRO // 2,), "uint8")
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            red = T.alloc_local((1,), "float32")
            lut = _e2m1fn_fp32(kr & 15) if decode == "lutshfl" else 0.0
            acc[0] = 0.0
            for ko in T.serial(T.ceildiv(K, block_K)):
                base = ko * block_K + kr * MICRO
                for v in T.vectorized(MICRO):
                    X_local[v] = X[0, base + v]
                for v in T.vectorized(MICRO // 2):
                    WQ_local[v] = WQ[n, base // 2 + v]
                partial[0] = 0.0
                for ki in T.serial(MICRO):
                    byte = WQ_local[ki // 2]
                    nib = (byte >> ((ki % 2) * 4)) & 15
                    if decode == "nodecode":
                        w = 1.0
                    elif decode == "lutshfl":
                        w = T.tvm_warp_shuffle(0xFFFFFFFF, lut, T.cast(nib, "int32"), 32, 32)
                    else:
                        w = _e2m1fn_fp32(nib)
                    partial[0] += T.cast(X_local[ki], "float32") * w
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


N, K = 17408, 5120
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

bw = 3312.0
roof = (N * K * 0.75 + 4 * K) / bw / 1e9 * 1e3
print(f"roof: {roof:.4f} ms (BW {bw:.0f} GB/s), micro={MICRO}", flush=True)

for decode in ("nodecode", "bitcast", "lutshfl"):
    ker = make_variant("cuda", decode)
    t0 = time.perf_counter()
    y = ker(xp, wqp, sp, 32, 4)[:, :N].float()
    torch.cuda.synchronize()
    jit = time.perf_counter() - t0
    err = (y.cpu() - ref).abs().max().item() / ref.abs().max().item()
    ms = time_call(lambda: ker(xp, wqp, sp, 32, 4))
    print(
        f"{decode:<12}: {ms:.4f} ms  ({100 * roof / ms:5.1f}% roof, rel-err {err:.2e}, jit {jit:.0f}s)",
        flush=True,
    )
