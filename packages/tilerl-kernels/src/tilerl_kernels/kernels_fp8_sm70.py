"""Keep FP8 checkpoint bytes on sm70; reproduce the old dequantized BF16 face."""

import tilelang
import tilelang.language as T

from .kernels_mega import _FP8_SRC


def make_linear_fp8_sm70(target):
    @tilelang.jit(target=target)
    def linear_fp8(X, W8, WScale, OScale):
        M, N, K = T.const("M, N, K")
        X: T.Tensor((M, K), "float32")
        W8: T.Tensor((N, K), "uint8")
        WScale: T.Tensor((T.ceildiv(N, 128), T.ceildiv(K, 128)), "float32")
        OScale: T.Tensor((N,), "float32")
        Y = T.empty((M, N), "float32")
        with T.Kernel(T.ceildiv(N, 4), M, threads=128) as (bx, row):
            T.import_source(_FP8_SRC)
            tx = T.get_thread_binding(0)
            n = bx * 4 + tx // 32
            lane = tx % 32
            acc = T.alloc_local((1,), "float32")
            acc[0] = 0.0
            if n < N:
                for tile in T.serial(T.ceildiv(K, 32)):
                    k = tile * 32 + lane
                    if k < K:
                        scale = WScale[n // 128, k // 128] * OScale[n]
                        w = T.call_extern("tl_mega_fp8", W8[n, k], scale, dtype="float32")
                        acc[0] += X[row, k] * w
                for level in T.unroll(5):
                    acc[0] += T.shfl_down(acc[0], 16 >> level)
                if lane == 0:
                    Y[row, n] = acc[0]
        return Y
    return linear_fp8
