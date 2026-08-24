"""Isolate the GEMV bottleneck: full decode vs trivial weight vs LUT decode."""

import sys
import time

import torch
import tilelang
import tilelang.language as T

sys.path.insert(0, "src")
from tilerl.ops.backend import get_backend
from tilerl.ops.reference import linear_fp4, pack_fp4

b = get_backend()


def make_variant(target, mode):
    @tilelang.jit(target=target, pass_configs={"tl.disable_data_race_check": True})
    def ker(X, WQ, Scale, LUT, reduce_thread, n_partition):
        N, K = T.const("N, K")
        micro = 4
        block_K = reduce_thread * micro
        X: T.Tensor((1, K), "float32")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 16), "float32")
        LUT: T.Tensor((8,), "float32")
        Y = T.empty((1, N), "float32")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            X_local = T.alloc_local((micro,), "float32")
            WQ_local = T.alloc_local((micro // 2,), "uint8")
            acc = T.alloc_local((1,), "float32")
            red = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            for ko in T.serial(T.ceildiv(K, block_K)):
                base = ko * block_K + kr * micro
                for v in T.vectorized(micro):
                    X_local[v] = X[0, base + v]
                for v in T.vectorized(micro // 2):
                    WQ_local[v] = WQ[n, base // 2 + v]
                s = Scale[n, base // 16]
                for ki in T.serial(micro):
                    byte = WQ_local[ki // 2]
                    nib = (byte >> ((ki % 2) * 4)) & 15
                    if mode == "nodecode":
                        w = 1.0
                    elif mode == "lut":
                        sign = T.cast(1 - 2 * T.cast(nib >> 3, "int32"), "float32")
                        w = sign * LUT[nib & 7] * s
                    else:
                        sign = T.cast(1 - 2 * T.cast(nib >> 3, "int32"), "float32")
                        e = T.cast((nib >> 1) & 3, "float32")
                        m = T.cast(nib & 1, "float32")
                        w = sign * (0.5 * T.exp2(e)) * (1.0 + 0.5 * m) * s
                    acc[0] += X_local[ki] * w
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(T.uint32(1), acc[0], True, red[0], kr, dtype="handle")
                )
            if kr == 0:
                Y[0, n] = red[0]
        return Y

    return ker


def time_call(fn, iters=30):
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
Kp = ((K + 127) // 128) * 128
Np = ((N + 3) // 4) * 4
xp = torch.nn.functional.pad(x, (0, Kp - K))
wqp = torch.nn.functional.pad(wq, (0, (Kp - K) // 2, 0, Np - N))
sp = torch.nn.functional.pad(sc, (0, (Kp - K) // 16, 0, Np - N))
lut = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=b.device)

for mode in ("full", "lut", "nodecode"):
    ker = make_variant("cuda", mode)
    t0 = time.perf_counter()
    y = ker(xp, wqp, sp, lut, 32, 4)[:, :N]
    torch.cuda.synchronize()
    jit = time.perf_counter() - t0
    ms = time_call(lambda: ker(xp, wqp, sp, lut, 32, 4))
    roof = (N * K * 0.75 + 4 * K) / 3312e9 * 1e3
    print(f"{mode:<9}: {ms:.4f} ms  ({100*roof/ms:.0f}% roof, jit {jit:.1f}s)", flush=True)
