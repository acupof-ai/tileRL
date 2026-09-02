"""Gated-delta-net fused kernels for sm90: decode (T tokens, in place) and chunk prefill.
f32 state/weights, bf16 activations; parity oracle is reference.gdn_forward."""

from __future__ import annotations

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs


def make_gdn_state_scan(target: str, block_DV: int = 32, threads: int = 128,
                        num_stages: int = 1):
    """Inter-chunk state scan of the chunkwise-WY form, S_next = e_last * S +
    K^T (U - W S), state held in registers (example_chunk_delta_h.py).

    UNFINISHED, not registered: the e_last * S term is lost (T.gemm does not
    accumulate into h_fr). Record and next step:
    docs/experience/errors/2026-08-29-gdn-state-scan-port-wip.md.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_state_scan(K, W, U, G, State, chunk):
        B, S, H, DK = T.const("B, S, H, DK")
        DV = T.const("DV")
        K: T.Tensor((B, S, H, DK), "bfloat16")
        W: T.Tensor((B, S, H, DK), "bfloat16")
        U: T.Tensor((B, S, H, DV), "bfloat16")
        G: T.Tensor((B, S, H), "float32")  # chunk-local inclusive cumsum
        State: T.Tensor((B, H, DK, DV), "float32")
        VNew = T.empty((B, S, H, DV), "bfloat16")
        Out = T.empty((B, H, DK, DV), "float32")
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H
            v0 = bv * block_DV
            h_sh = T.alloc_shared((DK, block_DV), "bfloat16")
            h_fr = T.alloc_fragment((DK, block_DV), "float32")
            u_sh = T.alloc_shared((chunk, block_DV), "bfloat16")
            u_fr = T.alloc_fragment((chunk, block_DV), "float32")
            w_sh = T.alloc_shared((chunk, DK), "bfloat16")
            k_sh = T.alloc_shared((chunk, DK), "bfloat16")
            vn_fr = T.alloc_fragment((chunk, block_DV), "float32")
            vn_sh = T.alloc_shared((chunk, block_DV), "bfloat16")
            g_sh = T.alloc_shared((chunk, block_DV), "float32")
            g_fr = T.alloc_fragment((chunk, block_DV), "float32")
            # two vars: rebinding one T.alloc_var swaps in a host-side expression
            g_last = T.alloc_var(T.float32)
            g_decay = T.alloc_var(T.float32)
            T.annotate_layout({
                u_sh: tilelang.layout.make_swizzled_layout(u_sh),
                g_sh: tilelang.layout.make_swizzled_layout(g_sh),
            })
            T.copy(State[bb, bh, 0:DK, v0 : v0 + block_DV], h_sh)
            T.copy(h_sh, h_fr)
            for c in T.Pipelined(T.ceildiv(S, chunk), num_stages=num_stages):
                # V_new = U - W @ S
                T.copy(W[bb, c * chunk : (c + 1) * chunk, bh, 0:DK], w_sh)
                T.gemm(w_sh, h_sh, vn_fr, clear_accum=True)
                T.copy(U[bb, c * chunk : (c + 1) * chunk, bh, v0 : v0 + block_DV], u_sh)
                T.copy(u_sh, u_fr)
                for i, j in T.Parallel(chunk, block_DV):
                    vn_fr[i, j] = u_fr[i, j] - vn_fr[i, j]
                T.copy(vn_fr, vn_sh)
                T.copy(vn_sh, VNew[bb, c * chunk : (c + 1) * chunk, bh, v0 : v0 + block_DV])
                # decay to the chunk end, then S += K^T V_new
                T.copy(K[bb, c * chunk : (c + 1) * chunk, bh, 0:DK], k_sh)
                g_last = G[bb, (c + 1) * chunk - 1, bh]
                for i, j in T.Parallel(chunk, block_DV):
                    g_sh[i, j] = G[bb, c * chunk + i, bh]
                T.copy(g_sh, g_fr)
                for i, j in T.Parallel(chunk, block_DV):
                    vn_fr[i, j] = (
                        vn_fr[i, j] * T.exp2((g_last - g_fr[i, j]) * 1.4426950408889634)
                        if g_last - g_fr[i, j] <= 0
                        else 0
                    )
                g_decay = T.exp2(g_last * 1.4426950408889634)
                for i, j in T.Parallel(DK, block_DV):
                    h_fr[i, j] *= g_decay
                T.copy(vn_fr, vn_sh)
                T.gemm(k_sh, vn_sh, h_fr, transpose_A=True)
                T.copy(h_fr, h_sh)
            T.copy(h_fr, Out[bb, bh, 0:DK, v0 : v0 + block_DV])
        return VNew, Out

    return gdn_state_scan


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
