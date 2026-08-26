"""Round-7 sweep: get the fp4 dequant off the FMA critical path.

The grouped kernel (group4, shipped) issues 32 shuffles then 32 FMAs per
group: the shuffle issue slots are serialized with the FMA issue slots
(~64 cyc/group vs the nodecode floor's ~32). This round tests structures
that move the shuffles off the consumer warp's issue path:

- group4:     shipped baseline (registers, grouped decode).
- shared_pp:  same-warp shared-memory ping-pong (decode g+1 -> shared, FMA g
              from shared). Adds 32 LDS + 32 STS on the load pipe per group;
              expected to tie group4 (load-pipe bound at ~64 cyc).
- group8:     GROUP=8 register variant (same issue/elem, more regs).
- producer:   producer/consumer warp split (threadIdx.z role). Producer warps
              do 32 SHFL + 32 STS/group into a RING=3 SPSC shared ring,
              running 2 groups ahead; consumer warps do 32 LDS + 32 FMA with
              ZERO shuffles, the prefetch LDS dual-issuing with the FMA chain.
              Target ~32-40 cyc/group.

Diagnostic only — not shipped.
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T

sys.path.insert(0, "src")
from tilerl.ops.backend import get_backend
from tilerl.ops.reference import linear_fp4, pack_fp4

b = get_backend()

MICRO = 8
GROUP = 4
N_PART = 4
RT = 32  # reduce_thread (backend hardcodes 32)
RING = 3  # producer runs 2 groups ahead of the consumer


def _e2m1fn_fp32(nib):
    ni32 = T.cast(nib, "int32")
    bits = ((ni32 & 8) << 28) | ((126 + ((ni32 >> 1) & 3)) << 23) | ((ni32 & 1) << 22)
    return T.reinterpret(bits, "float32")


def _shfl(lut, byte, ki):
    nib = (byte >> ((ki % 2) * 4)) & 15
    return T.tvm_warp_shuffle(0xFFFFFFFF, lut, T.cast(nib, "int32"), 32, 32)


def _reduce_y(acc, red, kr, Y, n):
    with T.attr(
        T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
        "reduce_scope",
        T.reinterpret(T.uint64(0), dtype="handle"),
    ):
        T.evaluate(T.tvm_thread_allreduce(T.uint32(1), acc[0], True, red[0], kr, dtype="handle"))
    if kr == 0:
        Y[0, n] = T.cast(red[0], "bfloat16")


def _make_group(target, group):
    """group4 (shipped) / group8: grouped decode in registers."""

    @tilelang.jit(target=target, pass_configs={"tl.disable_data_race_check": True})
    def ker(X, WQ, Scale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        block_K = reduce_thread * MICRO
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // group
        X: T.Tensor((1, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            Xs = T.alloc_local((group, MICRO), "bfloat16")
            Ws = T.alloc_local((group, MICRO // 2), "uint8")
            ws = T.alloc_local((group, MICRO), "float32")
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            red = T.alloc_local((1,), "float32")
            lut = _e2m1fn_fp32(kr & 15)
            acc[0] = 0.0
            for kg in T.serial(num_g):
                for g in T.unroll(group):
                    base = (kg * group + g) * block_K + kr * MICRO
                    for v in T.vectorized(MICRO):
                        Xs[g, v] = X[0, base + v]
                    for v in T.vectorized(MICRO // 2):
                        Ws[g, v] = WQ[n, base // 2 + v]
                for g in T.unroll(group):
                    for ki in T.unroll(MICRO):
                        ws[g, ki] = _shfl(lut, Ws[g, ki // 2], ki)
                for g in T.unroll(group):
                    base = (kg * group + g) * block_K + kr * MICRO
                    partial[0] = 0.0
                    for ki in T.unroll(MICRO):
                        partial[0] += T.cast(Xs[g, ki], "float32") * ws[g, ki]
                    acc[0] += Scale[n, base // 32] * partial[0]
            for kt in T.serial(num_ko - num_g * group):
                base = (num_g * group + kt) * block_K + kr * MICRO
                for v in T.vectorized(MICRO):
                    Xs[0, v] = X[0, base + v]
                for v in T.vectorized(MICRO // 2):
                    Ws[0, v] = WQ[n, base // 2 + v]
                partial[0] = 0.0
                for ki in T.unroll(MICRO):
                    partial[0] += T.cast(Xs[0, ki], "float32") * _shfl(lut, Ws[0, ki // 2], ki)
                acc[0] += Scale[n, base // 32] * partial[0]
            _reduce_y(acc, red, kr, Y, n)
        return Y

    return ker


def _make_shared_pp(target):
    """Same-warp shared ping-pong: decode g+1 into shared while FMA consumes
    g from shared (LDS). RING=2."""

    @tilelang.jit(target=target, pass_configs={"tl.disable_data_race_check": True})
    def ker(X, WQ, Scale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        block_K = reduce_thread * MICRO
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        GM = GROUP * MICRO
        X: T.Tensor((1, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, n_partition), threads=(reduce_thread, n_partition)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")
            n = bx * n_partition + ni
            # lane-major: 32 lanes at the same elem hit 32 banks
            Wsh = T.alloc_shared((2, N_PART, GM, RT), "float32")
            Xs = T.alloc_local((GROUP, MICRO), "bfloat16")
            Ws = T.alloc_local((GROUP, MICRO // 2), "uint8")
            wcur = T.alloc_local((GM,), "float32")
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            red = T.alloc_local((1,), "float32")
            lut = _e2m1fn_fp32(kr & 15)
            acc[0] = 0.0
            if num_g >= 1:
                for g in T.unroll(GROUP):
                    base = g * block_K + kr * MICRO
                    for v in T.vectorized(MICRO // 2):
                        Ws[g, v] = WQ[n, base // 2 + v]
                for g in T.unroll(GROUP):
                    for ki in T.unroll(MICRO):
                        Wsh[0, ni, g * MICRO + ki, kr] = _shfl(lut, Ws[g, ki // 2], ki)
            T.sync_threads()
            for kg in T.serial(num_g):
                for i in T.unroll(GM):
                    wcur[i] = Wsh[kg % 2, ni, i, kr]
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * MICRO
                    for v in T.vectorized(MICRO):
                        Xs[g, v] = X[0, base + v]
                for g in T.unroll(GROUP):
                    base = (kg * GROUP + g) * block_K + kr * MICRO
                    partial[0] = 0.0
                    for ki in T.unroll(MICRO):
                        partial[0] += T.cast(Xs[g, ki], "float32") * wcur[g * MICRO + ki]
                    acc[0] += Scale[n, base // 32] * partial[0]
                if kg + 1 < num_g:
                    for g in T.unroll(GROUP):
                        base = ((kg + 1) * GROUP + g) * block_K + kr * MICRO
                        for v in T.vectorized(MICRO // 2):
                            Ws[g, v] = WQ[n, base // 2 + v]
                    for g in T.unroll(GROUP):
                        for ki in T.unroll(MICRO):
                            Wsh[(kg + 1) % 2, ni, g * MICRO + ki, kr] = _shfl(
                                lut, Ws[g, ki // 2], ki
                            )
                T.sync_threads()
            for kt in T.serial(num_ko - num_g * GROUP):
                base = (num_g * GROUP + kt) * block_K + kr * MICRO
                for v in T.vectorized(MICRO // 2):
                    Ws[0, v] = WQ[n, base // 2 + v]
                partial[0] = 0.0
                for ki in T.unroll(MICRO):
                    partial[0] += T.cast(X[0, base + ki], "float32") * _shfl(
                        lut, Ws[0, ki // 2], ki
                    )
                acc[0] += Scale[n, base // 32] * partial[0]
            _reduce_y(acc, red, kr, Y, n)
        return Y

    return ker


def _make_producer(target):
    """Producer/consumer warp split. threadIdx.z=0 warps decode (SHFL+STS)
    into a RING=3 SPSC shared ring, 2 groups ahead; threadIdx.z=1 warps FMA
    (LDS+FMA, no shuffles). The consumer's prefetch LDS of group kg+1
    dual-issues with the FMA chain of group kg."""

    @tilelang.jit(target=target, pass_configs={"tl.disable_data_race_check": True})
    def ker(X, WQ, Scale, reduce_thread, n_partition):
        N, K = T.const("N, K")
        block_K = reduce_thread * MICRO
        num_ko = T.ceildiv(K, block_K)
        num_g = num_ko // GROUP
        GM = GROUP * MICRO
        X: T.Tensor((1, K), "bfloat16")
        WQ: T.Tensor((N, K // 2), "uint8")
        Scale: T.Tensor((N, K // 32), "float32")
        Y = T.empty((1, N), "bfloat16")
        with T.Kernel(T.ceildiv(N, N_PART), threads=(reduce_thread, N_PART, 2)) as bx:
            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, N_PART, thread="threadIdx.y")
            role = T.thread_binding(0, 2, thread="threadIdx.z")
            n = bx * N_PART + ni
            Wsh = T.alloc_shared((RING, N_PART, GM, RT), "float32")
            Ws = T.alloc_local((GROUP, MICRO // 2), "uint8")
            wcur = T.alloc_local((GM,), "float32")
            Xs = T.alloc_local((GROUP, MICRO), "bfloat16")
            acc = T.alloc_local((1,), "float32")
            partial = T.alloc_local((1,), "float32")
            red = T.alloc_local((1,), "float32")
            lut = _e2m1fn_fp32(kr & 15)
            acc[0] = 0.0

            if role == 0:
                if num_g >= 1:
                    for g in T.unroll(GROUP):
                        base = g * block_K + kr * MICRO
                        for v in T.vectorized(MICRO // 2):
                            Ws[g, v] = WQ[n, base // 2 + v]
                    for g in T.unroll(GROUP):
                        for ki in T.unroll(MICRO):
                            Wsh[0, ni, g * MICRO + ki, kr] = _shfl(lut, Ws[g, ki // 2], ki)
                if num_g >= 2:
                    for g in T.unroll(GROUP):
                        base = (GROUP + g) * block_K + kr * MICRO
                        for v in T.vectorized(MICRO // 2):
                            Ws[g, v] = WQ[n, base // 2 + v]
                    for g in T.unroll(GROUP):
                        for ki in T.unroll(MICRO):
                            Wsh[1, ni, g * MICRO + ki, kr] = _shfl(lut, Ws[g, ki // 2], ki)
            T.sync_threads()
            if num_g >= 1:
                if role == 1:
                    for i in T.unroll(GM):
                        wcur[i] = Wsh[0, ni, i, kr]
                for kg in T.serial(num_g):
                    if role == 1:
                        for g in T.unroll(GROUP):
                            base = (kg * GROUP + g) * block_K + kr * MICRO
                            for v in T.vectorized(MICRO):
                                Xs[g, v] = X[0, base + v]
                        for g in T.unroll(GROUP):
                            base = (kg * GROUP + g) * block_K + kr * MICRO
                            partial[0] = 0.0
                            for ki in T.unroll(MICRO):
                                partial[0] += T.cast(Xs[g, ki], "float32") * wcur[g * MICRO + ki]
                            acc[0] += Scale[n, base // 32] * partial[0]
                        if kg + 1 < num_g:
                            for i in T.unroll(GM):
                                wcur[i] = Wsh[(kg + 1) % RING, ni, i, kr]
                    if role == 0:  # noqa: SIM102 (PrimExpr and: nested, not combined)
                        if kg + 2 < num_g:
                            for g in T.unroll(GROUP):
                                base = ((kg + 2) * GROUP + g) * block_K + kr * MICRO
                                for v in T.vectorized(MICRO // 2):
                                    Ws[g, v] = WQ[n, base // 2 + v]
                            for g in T.unroll(GROUP):
                                for ki in T.unroll(MICRO):
                                    Wsh[(kg + 2) % RING, ni, g * MICRO + ki, kr] = _shfl(
                                        lut, Ws[g, ki // 2], ki
                                    )
                    T.sync_threads()
            if role == 1:
                for kt in T.serial(num_ko - num_g * GROUP):
                    base = (num_g * GROUP + kt) * block_K + kr * MICRO
                    for v in T.vectorized(MICRO // 2):
                        Ws[0, v] = WQ[n, base // 2 + v]
                    partial[0] = 0.0
                    for ki in T.unroll(MICRO):
                        partial[0] += T.cast(X[0, base + ki], "float32") * _shfl(
                            lut, Ws[0, ki // 2], ki
                        )
                    acc[0] += Scale[n, base // 32] * partial[0]
                _reduce_y(acc, red, kr, Y, n)
        return Y

    return ker


VARIANTS = {
    "group4": lambda t: _make_group(t, 4),
    "group8": lambda t: _make_group(t, 8),
    "shared_pp": _make_shared_pp,
    "producer": _make_producer,
}


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
    from tilerl.ops import kernels_linear

    y_ship = kernels_linear.make_linear_fp4_gemv("cuda")(xp, wqp, sp, 32, 4, 32)[:, :N].float().cpu()
    e_ship = (y_ship - ref).abs().max().item() / ref.abs().max().item()
    print(f"  [harness check] shipped vs ref: {e_ship:.4e}", flush=True)
    roof = (N * K * 0.75 + 2 * K) / bw / 1e9 * 1e3
    print(f"\n--- N={N} K={K}  roof {roof:.4f} ms (bf16 X) ---", flush=True)
    for name, make in VARIANTS.items():
        ker = make("cuda")
        t0 = time.perf_counter()
        y = ker(xp, wqp, sp, 32, 4)[:, :N].float()
        torch.cuda.synchronize()
        jit = time.perf_counter() - t0
        err = (y.cpu() - ref).abs().max().item() / ref.abs().max().item()
        ms = time_call(lambda: ker(xp, wqp, sp, 32, 4))
        print(
            f"{name:<10}: {ms:.4f} ms  ({100 * roof / ms:5.1f}% roof, rel-err {err:.2e}, "
            f"jit {jit:.0f}s)",
            flush=True,
        )
