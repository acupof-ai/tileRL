"""Gated-delta-net fused kernels for sm90: decode (T tokens, in place) and chunk prefill.
f32 state/weights, bf16 activations; parity oracle is reference.gdn_forward."""

from __future__ import annotations

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs

_NO_WARP_SPEC = tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED.value
_FAST_MATH = tilelang.PassConfigKey.TL_ENABLE_FAST_MATH.value


# ---- chunkwise-WY gated delta, prefill: fla's chunk_gated_delta_rule_fwd stage by stage,
# each kernel a transcription of tilelang examples/gdn/ (names, shapes, dtypes only).
# Layout [B,S,H,D]; G is the chunk-local inclusive cumsum of the log gate. Q/K carry
# the HK key heads and the kernels index bh // (H // HK): no head-repeat copy.


def make_gdn_prep_bf16(target: str):
    """:func:`kernels.make_gdn_prep` on sm90: one thread per head column, the
    q/k L2 sums by block allreduce, bf16 out for the WY gemms. The conv window
    stays f32 at both ends -- it is the state pool's dtype."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_prep(Q, Key, Val, GIn, BIn, DtBias, ALog, ConvW, Window, threads):
        B, TT, HK, DK, NVH, DV, KER, QKVD = T.const("B, TT, HK, DK, NVH, DV, KER, QKVD")
        Q: T.Tensor((B, TT, HK, DK), "bfloat16")
        Key: T.Tensor((B, TT, HK, DK), "bfloat16")
        Val: T.Tensor((B, TT, NVH, DV), "bfloat16")
        GIn: T.Tensor((B, TT, NVH), "float32")
        BIn: T.Tensor((B, TT, NVH), "float32")
        DtBias: T.Tensor((NVH,), "float32")
        ALog: T.Tensor((NVH,), "float32")
        ConvW: T.Tensor((QKVD, KER), "float32")
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        Qo = T.empty((B, TT, HK, DK), "bfloat16")
        Ko = T.empty((B, TT, HK, DK), "bfloat16")
        Vo = T.empty((B, TT, NVH, DV), "bfloat16")
        Go = T.empty((B, TT, NVH), "float32")
        Bo = T.empty((B, TT, NVH), "bfloat16")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        scale = T.rsqrt(T.cast(DK, "float32"))
        with T.Kernel(NVH, TT, B, threads=threads) as (vh, t, bb):
            tv = T.get_thread_binding(0)
            kh = vh * HK // NVH
            qc = kh * DK + tv
            kc = HK * DK + kh * DK + tv
            vc = 2 * HK * DK + vh * DV + tv
            cq = T.alloc_local((1,), "float32")
            ck = T.alloc_local((1,), "float32")
            cv = T.alloc_local((1,), "float32")
            sq = T.alloc_local((1,), "float32")
            sk = T.alloc_local((1,), "float32")
            pq = T.alloc_local((1,), "float32")
            pk = T.alloc_local((1,), "float32")

            cq[0] = 0.0
            ck[0] = 0.0
            cv[0] = 0.0
            for tap in T.serial(KER):
                if t + tap < KER - 1:  # the carried window, else this segment's raw qkv
                    cq[0] += Window[bb, t + tap, qc] * ConvW[qc, tap]
                    ck[0] += Window[bb, t + tap, kc] * ConvW[kc, tap]
                    cv[0] += Window[bb, t + tap, vc] * ConvW[vc, tap]
                else:
                    s = t + tap - (KER - 1)
                    cq[0] += T.cast(Q[bb, s, kh, tv], "float32") * ConvW[qc, tap]
                    ck[0] += T.cast(Key[bb, s, kh, tv], "float32") * ConvW[kc, tap]
                    cv[0] += T.cast(Val[bb, s, vh, tv], "float32") * ConvW[vc, tap]
            cq[0] = cq[0] * T.sigmoid(cq[0])
            ck[0] = ck[0] * T.sigmoid(ck[0])
            Vo[bb, t, vh, tv] = T.cast(cv[0] * T.sigmoid(cv[0]), "bfloat16")

            pq[0] = cq[0] * cq[0]
            pk[0] = ck[0] * ck[0]
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(T.uint32(1), pq[0], True, sq[0], tv, dtype="handle")
                )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(T.uint32(1), pk[0], True, sk[0], tv, dtype="handle")
                )
            if vh % (NVH // HK) == 0:  # one value head per GQA group writes q/k
                Qo[bb, t, kh, tv] = T.cast(cq[0] * T.rsqrt(sq[0] + 1e-12) * scale, "bfloat16")
                Ko[bb, t, kh, tv] = T.cast(ck[0] * T.rsqrt(sk[0] + 1e-12), "bfloat16")
            if tv == 0:
                x = GIn[bb, t, vh] + DtBias[vh]
                sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                Go[bb, t, vh] = -T.exp(ALog[vh]) * sp
                Bo[bb, t, vh] = T.cast(T.sigmoid(BIn[bb, t, vh]), "bfloat16")
            if t == 0:  # next window: the last KER-1 raw tokens of Window ++ qkv
                for tap in T.serial(KER - 1):
                    if TT + tap < KER - 1:
                        NewWindow[bb, tap, qc] = Window[bb, TT + tap, qc]
                        NewWindow[bb, tap, kc] = Window[bb, TT + tap, kc]
                        NewWindow[bb, tap, vc] = Window[bb, TT + tap, vc]
                    else:
                        s = TT + tap - (KER - 1)
                        NewWindow[bb, tap, qc] = T.cast(Q[bb, s, kh, tv], "float32")
                        NewWindow[bb, tap, kc] = T.cast(Key[bb, s, kh, tv], "float32")
                        NewWindow[bb, tap, vc] = T.cast(Val[bb, s, vh, tv], "float32")
        return Qo, Ko, Vo, Go, Bo, NewWindow

    return gdn_prep


def make_gdn_chunk_cumsum(target: str, threads: int = 256):
    """Chunk-local inclusive cumsum of the gate (example_cumsum.py,
    tilelang_chunk_local_cumsum_scalar: head_first=False, use_fragment=False)."""

    @tilelang.jit(target=target, pass_configs={**_pass_configs(), _NO_WARP_SPEC: True})
    def gdn_chunk_cumsum(G, chunk):
        B, S, H = T.const("B, S, H")
        G: T.Tensor((B, S, H), "float32")
        GNew = T.empty((B, S, H), "float32")
        with T.Kernel(T.ceildiv(S, chunk), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            G_shared = T.alloc_shared((1, chunk), "float32", scope="shared")
            T.copy(G[bb, bs * chunk : (bs + 1) * chunk, bh], G_shared)
            T.cumsum(G_shared, dim=1, reverse=False)
            T.copy(G_shared, GNew[bb, bs * chunk : (bs + 1) * chunk, bh])
        return GNew

    return gdn_chunk_cumsum


def make_gdn_chunk_kkt(target: str, block_DK: int = 64, threads: int = 128,
                       num_stages: int = 2):
    """A = tril(beta_i <k_i,k_j> exp(G_i - G_j), -1) per chunk, f32 for the solve
    (example_chunk_scaled_dot_kkt.py, tilelang_chunk_scaled_dot_kkt_fwd, use_g=True)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_chunk_kkt(K, Beta, G, chunk):
        B, S, HK, DK = T.const("B, S, HK, DK")
        H = T.const("H")
        K: T.Tensor((B, S, HK, DK), "bfloat16")
        Beta: T.Tensor((B, S, H), "bfloat16")
        G: T.Tensor((B, S, H), "float32")
        A = T.empty((B, S, H, chunk), "float32")
        block_S = chunk
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            kh = bh // (H // HK)
            Beta_shared = T.alloc_shared((block_S,), "bfloat16", scope="shared")
            K_shared = T.alloc_shared((block_S, block_DK), "bfloat16")
            A_shared = T.alloc_shared((block_S, block_S), "float32")
            Beta_K_fragment = T.alloc_fragment((block_S, block_DK), "bfloat16")
            A_fragment = T.alloc_fragment((block_S, block_S), "float32")
            G_shared = T.alloc_shared((block_S,), "float32", scope="shared")
            G_diff_local = T.alloc_fragment((block_S, block_S), "float32")

            T.fill(A_fragment, 0)
            T.disable_warp_group_reg_alloc()
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(
                    K[bb, bs * block_S : (bs + 1) * block_S, kh,
                      i_k * block_DK : (i_k + 1) * block_DK],
                    K_shared,
                )
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    Beta_K_fragment[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s]
                T.gemm(Beta_K_fragment, K_shared, A_fragment, transpose_B=True)

            for i_s in T.Parallel(block_S):
                G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                G_diff_local[i_s1, i_s2] = G_shared[i_s1] - G_shared[i_s2]
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                A_fragment[i_s1, i_s2] = T.if_then_else(
                    G_diff_local[i_s1, i_s2] <= 0 and i_s1 > i_s2,
                    A_fragment[i_s1, i_s2] * T.exp(G_diff_local[i_s1, i_s2]),
                    0,
                )

            T.copy(A_fragment, A_shared)
            T.copy(A_shared, A[bb, bs * block_S : (bs + 1) * block_S, bh, :])
        return A

    return gdn_chunk_kkt


def make_gdn_solve_tril(target: str, threads: int = 32):
    """Ai = (I + A)^-1 per chunk, A strictly lower from gdn_chunk_kkt, bf16 for the w,u
    gemm. 16x16 blocks: each diagonal block by row-wise forward substitution
    (Y = -A - A Y, then + I), each off-diagonal block Ai_rc = -Ai_rr sum_{c<=j<r} A_rj Ai_jc
    on the tensor cores, nearest the diagonal first (fla/ops/utils/solve_tril.py,
    merge_16x16_to_64x64_inverse_kernel; TileLang form after examples/kda/
    chunk_inter_solve_fused.py sections 3-5: f32 16x16 shared operands, one warp).
    The block products run at tf32 where fla's dots are ieee; the bf16 output rounds
    coarser than either."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_solve_tril(A, chunk):
        B, S, H = T.const("B, S, H")
        A: T.Tensor((B, S, H, chunk), "float32")
        Ai = T.empty((B, S, H, chunk), "bfloat16")
        BC, NB = 16, chunk // 16
        with T.Kernel(T.ceildiv(S, chunk), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            A_s = T.alloc_shared((chunk, chunk), "float32")
            Ai_s = T.alloc_shared((chunk, chunk), "float32")
            X_s = T.alloc_shared((BC, BC), "float32")
            Y_s = T.alloc_shared((BC, BC), "float32")
            P_f = T.alloc_fragment((BC, BC), "float32")

            T.copy(A[bb, bs * chunk : (bs + 1) * chunk, bh, :], A_s)
            # diagonal blocks: rows 0 and 1 of Y are -A already, rows 2.. substitute
            for i, j in T.Parallel(chunk, chunk):
                Ai_s[i, j] = T.if_then_else(i // BC == j // BC and i > j, -A_s[i, j], 0)
            for i in T.serial(2, BC):
                for d, c in T.Parallel(NB, BC):
                    for j in T.serial(i):
                        Ai_s[d * BC + i, d * BC + c] -= (
                            A_s[d * BC + i, d * BC + j] * Ai_s[d * BC + j, d * BC + c]
                        )
                T.tvm_storage_sync("shared")
            for i in T.Parallel(chunk):
                Ai_s[i, i] = 1.0

            # off-diagonal blocks, by distance from the diagonal so every Ai_jc is final
            for dist in range(1, NB):
                for c in range(NB - dist):
                    r = c + dist
                    for j in range(c, r):
                        T.copy(A_s[r * BC : (r + 1) * BC, j * BC : (j + 1) * BC], X_s)
                        T.copy(Ai_s[j * BC : (j + 1) * BC, c * BC : (c + 1) * BC], Y_s)
                        T.gemm(X_s, Y_s, P_f, clear_accum=j == c)
                    T.copy(Ai_s[r * BC : (r + 1) * BC, r * BC : (r + 1) * BC], X_s)
                    T.copy(P_f, Y_s)
                    T.gemm(X_s, Y_s, P_f, clear_accum=True)
                    for i, j in T.Parallel(BC, BC):
                        Ai_s[r * BC + i, c * BC + j] = -P_f[i, j]

            T.copy(Ai_s, Ai[bb, bs * chunk : (bs + 1) * chunk, bh, :])
        return Ai

    return gdn_solve_tril


def make_gdn_chunk_wu(target: str, block_DK: int = 64, block_DV: int = 32,
                      threads: int = 128, num_stages: int = 3):
    """W = A (K beta exp(G)), U = A (V beta) per chunk, A = (I + kkt)^-1
    (example_wy_fast.py, tilelang_recompute_w_u_fwd)."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_chunk_wu(K, V, Beta, G, A, chunk):
        B, S, HK, DK = T.const("B, S, HK, DK")
        H, DV = T.const("H, DV")
        K: T.Tensor((B, S, HK, DK), "bfloat16")
        V: T.Tensor((B, S, H, DV), "bfloat16")
        Beta: T.Tensor((B, S, H), "bfloat16")
        G: T.Tensor((B, S, H), "float32")
        A: T.Tensor((B, S, H, chunk), "bfloat16")
        W = T.empty((B, S, H, DK), "bfloat16")
        U = T.empty((B, S, H, DV), "bfloat16")
        block_S = chunk
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            kh = bh // (H // HK)
            Beta_shared = T.alloc_shared((block_S,), "bfloat16", scope="shared")
            K_shared = T.alloc_shared((block_S, block_DK), "bfloat16")
            V_shared = T.alloc_shared((block_S, block_DV), "bfloat16")
            G_shared = T.alloc_shared((block_S,), "float32", scope="shared")
            A_shared = T.alloc_shared((block_S, block_S), "bfloat16")
            W_fragment = T.alloc_fragment((block_S, block_DK), "float32")
            U_fragment = T.alloc_fragment((block_S, block_DV), "float32")
            W_shared = T.alloc_shared((block_S, block_DK), "bfloat16")
            U_shared = T.alloc_shared((block_S, block_DV), "bfloat16")
            W_Beta_shared = T.alloc_shared((block_S, block_DK), "bfloat16")
            U_Beta_shared = T.alloc_shared((block_S, block_DV), "bfloat16")

            T.annotate_layout({
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
            })

            T.disable_warp_group_reg_alloc()
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]
                G_shared[i_s] = T.exp(G[bb, bs * block_S + i_s, bh])

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(
                    V[bb, bs * block_S : (bs + 1) * block_S, bh,
                      i_v * block_DV : (i_v + 1) * block_DV],
                    V_shared,
                )
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    U_Beta_shared[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                T.gemm(A_shared, U_Beta_shared, U_fragment, clear_accum=True)
                # First copy to smem, then copy to gmem to reduce U2RU instructions
                T.copy(U_fragment, U_shared)
                T.copy(
                    U_shared,
                    U[bb, bs * block_S : (bs + 1) * block_S, bh,
                      i_v * block_DV : (i_v + 1) * block_DV],
                )

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(
                    K[bb, bs * block_S : (bs + 1) * block_S, kh,
                      i_k * block_DK : (i_k + 1) * block_DK],
                    K_shared,
                )
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    W_Beta_shared[i_s, i_k2] = (
                        K_shared[i_s, i_k2] * Beta_shared[i_s] * G_shared[i_s]
                    )
                T.gemm(A_shared, W_Beta_shared, W_fragment, clear_accum=True)
                T.copy(W_fragment, W_shared)
                T.copy(
                    W_shared,
                    W[bb, bs * block_S : (bs + 1) * block_S, bh,
                      i_k * block_DK : (i_k + 1) * block_DK],
                )
        return W, U

    return gdn_chunk_wu


def make_gdn_state_scan(target: str, block_DV: int = 32, threads: int = 128):
    """Inter-chunk state scan, S_next = exp(G_last) S + K^T (U - W S), the state in a
    fragment across the chunk loop; exports every chunk's entry state h for chunk_o
    (example_chunk_delta_h.py, tilelang_chunk_gated_delta_rule_fwd_h with
    use_g, use_initial_state, store_final_state, save_new_value all True).
    The chunk loop cannot be pipelined: the state is loop-carried through b_h_shared,
    and the software pipeliner reorders that copy (NaN at 1 and 2; the earlier port's
    lost e_last * S term was the same fault,
    errors/2026-08-29-gdn-state-scan-port-wip.md)."""

    @tilelang.jit(target=target, pass_configs={**_pass_configs(), _FAST_MATH: True})
    def gdn_state_scan(K, W, U, G, initial_state, chunk):
        B, S, HK, DK = T.const("B, S, HK, DK")
        H, DV = T.const("H, DV")
        block_S = chunk
        BS = S // block_S
        K: T.Tensor((B, S, HK, DK), "bfloat16")
        W: T.Tensor((B, S, H, DK), "bfloat16")
        U: T.Tensor((B, S, H, DV), "bfloat16")
        G: T.Tensor((B, S, H), "float32")
        initial_state: T.Tensor((B, H, DK, DV), "bfloat16")
        h = T.empty((B, BS, H, DK, DV), "bfloat16")
        final_state = T.empty((B, H, DK, DV), "float32")
        V_new = T.empty((B, S, H, DV), "bfloat16")
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H
            kh = bh // (H // HK)

            b_h_shared = T.alloc_shared((DK, block_DV), "bfloat16")
            b_h_fragment = T.alloc_fragment((DK, block_DV), "float32")

            U_shared = T.alloc_shared((block_S, block_DV), "bfloat16")
            U_fragment = T.alloc_fragment((block_S, block_DV), "float32")
            W_shared = T.alloc_shared((block_S, DK), "bfloat16")
            V_new_fragment = T.alloc_fragment((block_S, block_DV), "float32")
            V_new_shared = T.alloc_shared((block_S, block_DV), "bfloat16")
            K_shared = T.alloc_shared((block_S, DK), "bfloat16")
            G_last_local = T.alloc_var(T.float32)
            G_shared = T.alloc_shared((block_S, block_DV), "float32")
            G_fragment = T.alloc_fragment((block_S, block_DV), "float32")

            T.annotate_layout({
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                G_shared: tilelang.layout.make_swizzled_layout(G_shared),
            })

            T.use_swizzle(10)

            T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], b_h_shared)
            T.copy(b_h_shared, b_h_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=0):
                T.copy(b_h_shared, h[bb, i_s, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

                T.copy(W[bb, i_s * block_S : (i_s + 1) * block_S, bh, 0:DK], W_shared)
                T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)

                T.copy(
                    U[bb, i_s * block_S : (i_s + 1) * block_S, bh,
                      bv * block_DV : (bv + 1) * block_DV],
                    U_shared,
                )
                T.copy(U_shared, U_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = -V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                T.copy(V_new_fragment, dst=V_new_shared)
                T.copy(
                    V_new_shared,
                    V_new[bb, i_s * block_S : (i_s + 1) * block_S, bh,
                          bv * block_DV : (bv + 1) * block_DV],
                )

                T.copy(K[bb, i_s * block_S : (i_s + 1) * block_S, kh, 0:DK], K_shared)
                G_last_local = G[bb, (i_s + 1) * block_S - 1, bh]
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    G_shared[i_s2, i_v] = G[bb, i_s * block_S + i_s2, bh]
                T.copy(G_shared, G_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = (
                        V_new_fragment[i_s2, i_v]
                        * T.exp2((G_last_local - G_fragment[i_s2, i_v]) * 1.4426950408889634)
                        if G_last_local - G_fragment[i_s2, i_v] <= 0
                        else 0
                    )
                G_last_local = T.exp2(G_last_local * 1.4426950408889634)
                for i_k, i_v in T.Parallel(DK, block_DV):
                    b_h_fragment[i_k, i_v] *= G_last_local

                T.copy(V_new_fragment, V_new_shared)
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)

                T.copy(b_h_fragment, b_h_shared)

            T.copy(b_h_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])
        return h, final_state, V_new

    return gdn_state_scan


def make_gdn_chunk_o(target: str, block_DK: int = 128, block_DV: int = 128,
                     threads: int = 128, num_stages: int = 1):
    """O = scale (exp(G) Q h + tril(Q K^T exp(G_i - G_j)) V_new) per chunk, h the
    chunk's entry state (example_chunk_o.py, tilelang_chunk_fwd_o, use_g=True).
    O is f32: the gated RMSNorm after it runs in f32, so a bf16 O was one more cast."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_chunk_o(Q, K, V, HIDDEN, G, chunk, scale):
        B, S, HK, DK = T.const("B, S, HK, DK")
        H, DV = T.const("H, DV")
        block_S = chunk
        Q: T.Tensor((B, S, HK, DK), "bfloat16")
        K: T.Tensor((B, S, HK, DK), "bfloat16")
        V: T.Tensor((B, S, H, DV), "bfloat16")
        HIDDEN: T.Tensor((B, S // block_S, H, DK, DV), "bfloat16")
        G: T.Tensor((B, S, H), "float32")
        O = T.empty((B, S, H, DV), "float32")
        with T.Kernel(
            T.ceildiv(DV, block_DV), T.ceildiv(S, block_S), B * H, threads=threads
        ) as (bv, bs, bbh):
            bb, bh = bbh // H, bbh % H
            kh = bh // (H // HK)
            Q_shared = T.alloc_shared((block_S, block_DK), "bfloat16")
            K_shared = T.alloc_shared((block_S, block_DK), "bfloat16")
            V_shared = T.alloc_shared((block_S, block_DV), "bfloat16")
            H_shared = T.alloc_shared((block_DK, block_DV), "bfloat16")
            A_shared = T.alloc_shared((block_S, block_S), "bfloat16")
            O_shared = T.alloc_shared((block_S, block_DV), "float32")
            A_fragment = T.alloc_fragment((block_S, block_S), "float32")
            O_fragment = T.alloc_fragment((block_S, block_DV), "float32")
            G_shared = T.alloc_shared((block_S,), "float32", scope="shared")
            G_diff_local = T.alloc_fragment((block_S, block_S), "float32")

            T.clear(A_fragment)
            T.clear(O_fragment)
            T.disable_warp_group_reg_alloc()
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(
                    Q[bb, bs * block_S : (bs + 1) * block_S, kh,
                      i_k * block_DK : (i_k + 1) * block_DK],
                    Q_shared,
                )
                T.copy(
                    K[bb, bs * block_S : (bs + 1) * block_S, kh,
                      i_k * block_DK : (i_k + 1) * block_DK],
                    K_shared,
                )
                T.copy(
                    HIDDEN[bb, bs, bh, i_k * block_DK : (i_k + 1) * block_DK,
                           bv * block_DV : (bv + 1) * block_DV],
                    H_shared,
                )
                T.gemm(Q_shared, H_shared, O_fragment)
                T.gemm(Q_shared, K_shared, A_fragment, transpose_B=True)

            for i_s in T.Parallel(block_S):
                G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
            for i_s, i_v in T.Parallel(block_S, block_DV):
                O_fragment[i_s, i_v] = O_fragment[i_s, i_v] * T.exp(G_shared[i_s])
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                G_diff_local[i_s1, i_s2] = G_shared[i_s1] - G_shared[i_s2]
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                A_fragment[i_s1, i_s2] = T.if_then_else(
                    G_diff_local[i_s1, i_s2] <= 0,
                    A_fragment[i_s1, i_s2] * T.exp(G_diff_local[i_s1, i_s2]),
                    0,
                )

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                if i_s1 < i_s2:
                    A_fragment[i_s1, i_s2] = 0

            T.copy(
                V[bb, bs * block_S : (bs + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV],
                V_shared,
            )
            T.copy(A_fragment, A_shared)
            T.gemm(A_shared, V_shared, O_fragment)

            for i_s, i_v in T.Parallel(block_S, block_DV):
                O_fragment[i_s, i_v] = O_fragment[i_s, i_v] * scale

            T.copy(O_fragment, O_shared)
            T.copy(
                O_shared,
                O[bb, bs * block_S : (bs + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV],
            )
        return O

    return gdn_chunk_o


def make_gdn_decode_fused(target: str):
    """Fused GDN decode core over TT tokens per row, one launch: conv1d + SiLU +
    q/k L2-norm + decay-first delta recurrence + gated RMSNorm + z-gate. One
    block per (value head, batch); thread tv owns state column S[:, tv], carried
    in registers across the TT steps so the pool is read and written once.
    ``ks`` > 0 (speculative verify) also keeps the state and window after each of
    the first ks tokens, which is what lets a TT>1 tick skip the gather/scatter.
    Ported from tilelang examples/gdn/qwen36_gdr_decode_fused.py
    (branch feat/qwen36-gdn-megakernel), f32 IO, time-major conv window."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_decode_fused(
        Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Windows, Par, States, Slots,
        StepStates, StepWindows, layer: T.int32, ks, threads,
    ):
        # QD is the constexpr (a constexpr must appear directly in a buffer shape)
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
        QKVD = 2 * QD + VD
        scale = T.rsqrt(T.cast(K, "float32"))
        Q: T.Tensor((B, TT, QD), "float32")
        Key: T.Tensor((B, TT, QD), "float32")
        Val: T.Tensor((B, TT, VD), "float32")
        Z: T.Tensor((B, TT, VD), "float32")
        GIn: T.Tensor((B, TT, NVH), "float32")
        BIn: T.Tensor((B, TT, NVH), "float32")
        DtBias: T.Tensor((NVH,), "float32")
        ALog: T.Tensor((NVH,), "float32")
        NormW: T.Tensor((V,), "float32")
        ConvW: T.Tensor((QKVD, KER), "float32")
        # conv windows are double-buffered (read Par[slot], write 1-Par[slot]):
        # q/k columns are shared across the GQA group, so no in-place shift
        S, L, KS, KW = T.const("S, L, KS, KW")
        Windows: T.Tensor((S, L, 2, KER - 1, QKVD), "float32")
        Par: T.Tensor((S,), "int32")
        States: T.Tensor((S, L, NVH, K, V), "float32")  # updated in place
        Slots: T.Tensor((B,), "int32")
        # ks=0 leaves these unwritten; the caller aliases them onto States/Windows
        StepStates: T.Tensor((S, L, KS, NVH, K, V), "float32")
        StepWindows: T.Tensor((S, L, KW, KER - 1, QKVD), "float32")
        Out = T.empty((B, TT, VD), "bfloat16")  # out_proj (fp8 GEMV) reads bf16
        with T.Kernel(NVH, B, threads=threads) as (vh, bb):
            tv = T.get_thread_binding(0)
            slot = Slots[bb]
            par = Par[slot]
            nxt = 1 - par
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv  # Q tensor column == Window/ConvW q column
            kc = QD + kh * K + tv  # Window/ConvW k column (K tensor column == qc)
            vc = 2 * QD + vh * V + tv  # Window/ConvW v column (V tensor column = vh*V+tv)

            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            qn = T.alloc_shared((1,), "float32")
            kn = T.alloc_shared((1,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")
            rms_s = T.alloc_shared((1,), "float32")
            # per-token fragments, hoisted out of the serial scan
            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_mem = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")

            # staged in registers: an in-place global RAW serialized every j (6.5 -> 57 us)
            s_loc = T.alloc_local((K,), "float32")
            accs = T.alloc_local((2 * 4,), "float32")
            for j in T.serial(K):
                s_loc[j] = States[slot, layer, vh, j, tv]

            for t in T.serial(TT):
                # conv1d (KER taps over Windows[par] ++ this tick's qkv) + SiLU
                cq[0] = 0.0
                ck[0] = 0.0
                cv[0] = 0.0
                for tap in T.serial(KER):
                    if t + tap < KER - 1:
                        cq[0] += Windows[slot, layer, par, t + tap, qc] * ConvW[qc, tap]
                        ck[0] += Windows[slot, layer, par, t + tap, kc] * ConvW[kc, tap]
                        cv[0] += Windows[slot, layer, par, t + tap, vc] * ConvW[vc, tap]
                    else:
                        ti = t + tap - (KER - 1)
                        cq[0] += Q[bb, ti, qc] * ConvW[qc, tap]
                        ck[0] += Key[bb, ti, qc] * ConvW[kc, tap]
                        cv[0] += Val[bb, ti, vh * V + tv] * ConvW[vc, tap]
                q_s[tv] = cq[0] * T.sigmoid(cq[0])
                k_s[tv] = ck[0] * T.sigmoid(ck[0])
                v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                # L2-norm + g/beta (thread 0 reduces, broadcasts via shared)
                if tv == 0:
                    acc_q = T.alloc_fragment((1,), "float32")
                    acc_k = T.alloc_fragment((1,), "float32")
                    T.clear(acc_q)
                    T.clear(acc_k)
                    for j in T.serial(K):
                        acc_q[0] += q_s[j] * q_s[j]
                        acc_k[0] += k_s[j] * k_s[j]
                    qn[0] = T.rsqrt(acc_q[0] + 1e-12)
                    kn[0] = T.rsqrt(acc_k[0] + 1e-12)
                    x = GIn[bb, t, vh] + DtBias[vh]
                    sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                    exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                    beta_s[0] = T.sigmoid(BIn[bb, t, vh])
                T.tvm_storage_sync("shared")

                q_s[tv] = q_s[tv] * qn[0] * scale
                k_s[tv] = k_s[tv] * kn[0]
                T.tvm_storage_sync("shared")

                # recurrence: decay + kv_mem, then rank-1 update + out;
                # 4 accumulators per pass break the K-deep FMA chain
                for a in T.serial(2 * 4):
                    accs[a] = 0.0
                for j in T.serial(K // 4):
                    base = j * 4
                    for u in T.unroll(4):
                        sj = s_loc[base + u] * exp_g_s[0]
                        s_loc[base + u] = sj
                        accs[u] += sj * k_s[base + u]
                kv_mem[0] = accs[0] + accs[1] + accs[2] + accs[3]
                delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
                for j in T.serial(K // 4):
                    base = j * 4
                    for u in T.unroll(4):
                        sj = s_loc[base + u] + delta[0] * k_s[base + u]
                        s_loc[base + u] = sj
                        accs[4 + u] += sj * q_s[base + u]
                if t < ks:  # per-chain-step state for a speculative verify
                    for j in T.serial(K):
                        StepStates[slot, layer, t, vh, j, tv] = s_loc[j]
                out_s[tv] = accs[4] + accs[5] + accs[6] + accs[7]
                T.tvm_storage_sync("shared")

                # gated RMSNorm + z-gate
                if tv == 0:
                    acc_sq = T.alloc_fragment((1,), "float32")
                    T.clear(acc_sq)
                    for j in T.serial(V):
                        acc_sq[0] += out_s[j] * out_s[j]
                    rms_s[0] = T.rsqrt(acc_sq[0] / T.cast(V, "float32") + 1e-6)
                # also fences token t's q_s/k_s reads against token t+1's writes
                T.tvm_storage_sync("shared")
                gate = Z[bb, t, vh * V + tv]
                Out[bb, t, vh * V + tv] = T.cast(
                    out_s[tv] * rms_s[0] * NormW[tv] * (gate * T.sigmoid(gate)), "bfloat16"
                )

            for j in T.serial(K):
                States[slot, layer, vh, j, tv] = s_loc[j]

            # new conv window: the last KER-1 raw qkv of (Windows[par] ++ this
            # tick's qkv); only the GQA representative writes the shared q/k
            for tap in T.serial(KER - 1):
                if TT + tap < KER - 1:
                    Windows[slot, layer, nxt, tap, vc] = Windows[slot, layer, par, TT + tap, vc]
                else:
                    Windows[slot, layer, nxt, tap, vc] = Val[bb, TT + tap - (KER - 1), vh * V + tv]
            if is_rep:
                for tap in T.serial(KER - 1):
                    if TT + tap < KER - 1:
                        Windows[slot, layer, nxt, tap, qc] = Windows[slot, layer, par, TT + tap, qc]
                        Windows[slot, layer, nxt, tap, kc] = Windows[slot, layer, par, TT + tap, kc]
                    else:
                        Windows[slot, layer, nxt, tap, qc] = Q[bb, TT + tap - (KER - 1), qc]
                        Windows[slot, layer, nxt, tap, kc] = Key[bb, TT + tap - (KER - 1), qc]

            # the same window after each of the first ks tokens
            for s in T.serial(ks):
                for tap in T.serial(KER - 1):
                    if s + 1 + tap < KER - 1:
                        StepWindows[slot, layer, s, tap, vc] = Windows[
                            slot, layer, par, s + 1 + tap, vc
                        ]
                    else:
                        StepWindows[slot, layer, s, tap, vc] = Val[
                            bb, s + 1 + tap - (KER - 1), vh * V + tv
                        ]
                if is_rep:
                    for tap in T.serial(KER - 1):
                        if s + 1 + tap < KER - 1:
                            StepWindows[slot, layer, s, tap, qc] = Windows[
                                slot, layer, par, s + 1 + tap, qc
                            ]
                            StepWindows[slot, layer, s, tap, kc] = Windows[
                                slot, layer, par, s + 1 + tap, kc
                            ]
                        else:
                            StepWindows[slot, layer, s, tap, qc] = Q[
                                bb, s + 1 + tap - (KER - 1), qc
                            ]
                            StepWindows[slot, layer, s, tap, kc] = Key[
                                bb, s + 1 + tap - (KER - 1), qc
                            ]

        return Out

    return gdn_decode_fused


def make_gdn_chunk_fused(target: str):
    """Prefill form of make_gdn_decode_fused: the same serial scan, but over each
    row's own SeqQLens tokens and over gathered [B, ...] state, so a tick whose
    rows have different widths still runs. Returns the raw recurrence output;
    the caller applies the gated RMSNorm and z-gate. StepStates/StepWindows
    [B, KS, ...] receive the state after each of the first KS tokens (prefill
    passes KS=1).
    Serial scan, not chunkwise-WY: measured slower at our shapes,
    errors/2026-08-25-gdn-chunked-gdr-rejected."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_chunk_fused(
        Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State, SeqQLens,
        StepStates, StepWindows, threads,
    ):
        # TT, not T: T is the tilelang.language alias
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
        QKVD = 2 * QD + VD
        scale = T.rsqrt(T.cast(K, "float32"))
        Q: T.Tensor((B, TT, QD), "bfloat16")
        Key: T.Tensor((B, TT, QD), "bfloat16")
        Val: T.Tensor((B, TT, VD), "bfloat16")
        Z: T.Tensor((B, TT, VD), "bfloat16")
        GIn: T.Tensor((B, TT, NVH), "float32")
        BIn: T.Tensor((B, TT, NVH), "float32")
        DtBias: T.Tensor((NVH,), "float32")
        ALog: T.Tensor((NVH,), "float32")
        NormW: T.Tensor((V,), "float32")
        ConvW: T.Tensor((QKVD, KER), "float32")
        Window: T.Tensor((B, KER - 1, QKVD), "bfloat16")
        State: T.Tensor((B, NVH, K, V), "float32")
        SeqQLens: T.Tensor((B,), "int32")
        KS = T.const("KS")
        StepStates: T.Tensor((B, KS, NVH, K, V), "float32")
        StepWindows: T.Tensor((B, KS, KER - 1, QKVD), "bfloat16")
        Out = T.empty((B, TT, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "bfloat16")
        with T.Kernel(NVH, B, threads=threads) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv  # Q tensor column == Window/ConvW q column
            kc = QD + kh * K + tv  # Window/ConvW k column (K tensor column == qc)
            vc = 2 * QD + vh * V + tv  # Window/ConvW v column (V tensor column = vh*V+tv)

            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            qn = T.alloc_shared((1,), "float32")
            kn = T.alloc_shared((1,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            pq = T.alloc_local((1,), "float32")
            pk = T.alloc_local((1,), "float32")
            sq = T.alloc_local((1,), "float32")
            sk = T.alloc_local((1,), "float32")

            # per-token fragments, hoisted out of the serial scan
            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_mem = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")
            acc_o = T.alloc_fragment((1,), "float32")

            # state column in registers across all T; 4 accumulators per pass
            # break the 128-deep FMA chain (21.6% over the global-state form)
            state_local = T.alloc_local((K,), "float32")
            accs = T.alloc_local((2 * 4,), "float32")
            for j in T.serial(K):
                state_local[j] = State[bb, vh, j, tv]

            for t in T.serial(SeqQLens[bb]):
                # conv1d (KER taps over Window ++ qkv) + SiLU
                cq[0] = 0.0
                ck[0] = 0.0
                cv[0] = 0.0
                for tap in T.serial(KER):
                    if t + tap < KER - 1:
                        cq[0] += T.cast(Window[bb, t + tap, qc], "float32") * ConvW[qc, tap]
                        ck[0] += T.cast(Window[bb, t + tap, kc], "float32") * ConvW[kc, tap]
                        cv[0] += T.cast(Window[bb, t + tap, vc], "float32") * ConvW[vc, tap]
                    else:
                        cq[0] += T.cast(Q[bb, t + tap - (KER - 1), qc], "float32") * ConvW[qc, tap]
                        ck[0] += (
                            T.cast(Key[bb, t + tap - (KER - 1), qc], "float32") * ConvW[kc, tap]
                        )
                        cv[0] += (
                            T.cast(Val[bb, t + tap - (KER - 1), vh * V + tv], "float32")
                            * ConvW[vc, tap]
                        )
                q_s[tv] = cq[0] * T.sigmoid(cq[0])
                k_s[tv] = ck[0] * T.sigmoid(ck[0])
                v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                # L2-norm by block allreduce: a thread-0 sum was half the kernel at T=512
                pq[0] = q_s[tv] * q_s[tv]
                pk[0] = k_s[tv] * k_s[tv]
                with T.attr(
                    T.comm_reducer(lambda a, bb_: a + bb_, [T.cast(0, "float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(T.uint32(1), pq[0], True, sq[0], tv, dtype="handle")
                    )
                with T.attr(
                    T.comm_reducer(lambda a, bb_: a + bb_, [T.cast(0, "float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(T.uint32(1), pk[0], True, sk[0], tv, dtype="handle")
                    )
                if tv == 0:
                    qn[0] = T.rsqrt(sq[0] + 1e-12)
                    kn[0] = T.rsqrt(sk[0] + 1e-12)
                    x = GIn[bb, t, vh] + DtBias[vh]
                    sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                    exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                    beta_s[0] = T.sigmoid(BIn[bb, t, vh])
                T.tvm_storage_sync("shared")

                q_s[tv] = q_s[tv] * qn[0] * scale
                k_s[tv] = k_s[tv] * kn[0]
                T.tvm_storage_sync("shared")

                # recurrence: decay + kv_mem, then rank-1 update + out
                for a in T.serial(2 * 4):
                    accs[a] = 0.0
                for j in T.serial(K // 4):
                    base = j * 4
                    for u in T.unroll(4):
                        sj = state_local[base + u] * exp_g_s[0]
                        state_local[base + u] = sj
                        accs[u] += sj * k_s[base + u]
                kv_mem[0] = accs[0] + accs[1] + accs[2] + accs[3]
                delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
                for j in T.serial(K // 4):
                    base = j * 4
                    for u in T.unroll(4):
                        sj = state_local[base + u] + delta[0] * k_s[base + u]
                        state_local[base + u] = sj
                        accs[4 + u] += sj * q_s[base + u]
                acc_o[0] = accs[4] + accs[5] + accs[6] + accs[7]
                if t < KS:  # per-chain-step state for a speculative verify
                    for j in T.serial(K):
                        StepStates[bb, t, vh, j, tv] = state_local[j]
                # raw core out: an in-loop thread-0 RMSNorm reduce held us/step at 3.1
                Out[bb, t, vh * V + tv] = acc_o[0]

            for j in T.serial(K // 4):
                base = j * 4
                for u in T.unroll(4):
                    NewState[bb, vh, base + u, tv] = state_local[base + u]

            # new conv window: last KER-1 raw qkv tokens of (Window ++ qkv);
            # only the GQA representative writes the shared q/k channels
            for tap in T.serial(KER - 1):
                if SeqQLens[bb] + tap < KER - 1:
                    NewWindow[bb, tap, vc] = Window[bb, SeqQLens[bb] + tap, vc]
                else:
                    NewWindow[bb, tap, vc] = Val[bb, SeqQLens[bb] + tap - (KER - 1), vh * V + tv]
            if is_rep:
                for tap in T.serial(KER - 1):
                    if SeqQLens[bb] + tap < KER - 1:
                        NewWindow[bb, tap, qc] = Window[bb, SeqQLens[bb] + tap, qc]
                        NewWindow[bb, tap, kc] = Window[bb, SeqQLens[bb] + tap, kc]
                    else:
                        NewWindow[bb, tap, qc] = Q[bb, SeqQLens[bb] + tap - (KER - 1), qc]
                        NewWindow[bb, tap, kc] = Key[bb, SeqQLens[bb] + tap - (KER - 1), qc]

            # the same window after each of the first KS tokens
            for s in T.serial(KS):
                for tap in T.serial(KER - 1):
                    if s + 1 + tap < KER - 1:
                        StepWindows[bb, s, tap, vc] = Window[bb, s + 1 + tap, vc]
                    else:
                        StepWindows[bb, s, tap, vc] = Val[bb, s + 1 + tap - (KER - 1), vh * V + tv]
                if is_rep:
                    for tap in T.serial(KER - 1):
                        if s + 1 + tap < KER - 1:
                            StepWindows[bb, s, tap, qc] = Window[bb, s + 1 + tap, qc]
                            StepWindows[bb, s, tap, kc] = Window[bb, s + 1 + tap, kc]
                        else:
                            StepWindows[bb, s, tap, qc] = Q[bb, s + 1 + tap - (KER - 1), qc]
                            StepWindows[bb, s, tap, kc] = Key[bb, s + 1 + tap - (KER - 1), qc]

        return Out, NewState, NewWindow

    return gdn_chunk_fused
