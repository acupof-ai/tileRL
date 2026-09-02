"""Gated-delta-net fused kernels for sm90 (decode T=1 + chunk prefill) —
SOTA schedules ported from the tilelang examples. Registered in the sm90
cell of the dispatch matrix (registry.py); the torch-eager parity path is
reference.gdn_forward. f32 state/weights, bf16 IO for the activations.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

from .kernels_mma import _pass_configs

__all__ = [
    "make_gdn_decode_fused",
    "make_gdn_chunk_fused",
]


# ---------------------------------------------------------------- gated-delta decode (fused)


def make_gdn_decode_fused(target: str, out_dtype: str = "bfloat16"):
    """Fused gated-delta decode core (sm90): conv1d + SiLU + q/k L2-norm +
    decay-first delta recurrence + gated RMSNorm + z-gate, one launch for
    T=1.

    Replaces reference.gdn_forward's Python head loop (~384 tiny kernel
    launches per layer per decode tick on the 27B slice: 48 value heads x
    ~8 einsums each). One block per (value head, batch); thread tv owns
    state column S[:, tv] (state in HBM, two serial passes over K).

    # SOTA copy: examples/gdn/qwen36_gdr_decode_fused.py @
    #   tilelang branch feat/qwen36-gdn-megakernel (commit 0fb99503, unmerged)
    # Adapted: f32 IO (tileRL convention; the branch is bf16-IO and rounds
    #   preact to bf16 before SiLU for arle parity — skipped, f32 matches
    #   reference.gdn_forward); separate NewState/NewWindow outputs (the
    #   branch mutates state and a conv_state ring in place); time-major
    #   conv window [B, K-1, qkv] (the branch is channel-major); q/k/v
    #   passed as separate [B, QD]/[B, VD] tensors (the branch takes a
    #   catted qkv — separate tensors make QD a direct constexpr).
    # Recurrence: tileRL's decay-first form (S *= g, then p = k @ S) — the
    #   branch matches, verified equation-by-equation against
    #   reference.gdn_forward.
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_decode_fused(
        Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Windows, Par, States, Slots,
        layer: T.int32, threads,
    ):
        # QD (flat q/k dim) is the constexpr, not NKH: tilelang requires each
        # constexpr used directly in a buffer shape, and NKH appears only
        # indirectly (NKH*K). Q/Key are [B, QD] (QD direct); the GQA key head
        # is kh = vh*(QD//K)//NVH. Params are Key/Val (not K/V) to avoid
        # shadowing the K/V constexpr head dims.
        B, QD, NVH, K, V, KER = T.const("B, QD, NVH, K, V, KER")
        VD = NVH * V
        QKVD = 2 * QD + VD
        scale = T.rsqrt(T.cast(K, "float32"))
        Q: T.Tensor((B, QD), "float32")
        Key: T.Tensor((B, QD), "float32")
        Val: T.Tensor((B, VD), "float32")
        Z: T.Tensor((B, VD), "float32")
        GIn: T.Tensor((B, NVH), "float32")
        BIn: T.Tensor((B, NVH), "float32")
        DtBias: T.Tensor((NVH,), "float32")
        ALog: T.Tensor((NVH,), "float32")
        NormW: T.Tensor((V,), "float32")
        ConvW: T.Tensor((QKVD, KER), "float32")
        # conv windows live in the pool, double-buffered: read plane Par[slot],
        # write 1-Par[slot] (q/k columns are shared across the GQA group's
        # blocks, so the shift cannot be done in place); the tick flips Par.
        S, L = T.const("S, L")
        Windows: T.Tensor((S, L, 2, KER - 1, QKVD), "float32")
        Par: T.Tensor((S,), "int32")
        # The state is updated IN PLACE in the pool at [Slots[b], layer]: each
        # thread owns its (j, tv) cells, so no gather/scatter kernels and no
        # NewState buffer (was 2 index launches + 3 MB of traffic per layer).
        States: T.Tensor((S, L, NVH, K, V), "float32")
        Slots: T.Tensor((B,), "int32")
        Out = T.empty((B, VD), out_dtype)  # bf16 on sm90, f32 on sm70 (out_proj IO dtype)
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

            # conv1d (K taps) + SiLU on this head's q/k/v channels
            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            cq[0] = Q[bb, qc] * ConvW[qc, KER - 1]
            ck[0] = Key[bb, qc] * ConvW[kc, KER - 1]
            cv[0] = Val[bb, vh * V + tv] * ConvW[vc, KER - 1]
            for tap in T.serial(KER - 1):
                cq[0] += Windows[slot, layer, par, tap, qc] * ConvW[qc, tap]
                ck[0] += Windows[slot, layer, par, tap, kc] * ConvW[kc, tap]
                cv[0] += Windows[slot, layer, par, tap, vc] * ConvW[vc, tap]
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
                x = GIn[bb, vh] + DtBias[vh]
                sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                beta_s[0] = T.sigmoid(BIn[bb, vh])
            T.tvm_storage_sync("shared")

            q_s[tv] = q_s[tv] * qn[0] * scale
            k_s[tv] = k_s[tv] * kn[0]
            T.tvm_storage_sync("shared")

            # recurrence: decay + kv_mem, then rank-1 update + out
            kv_mem = T.alloc_fragment((1,), "float32")
            T.clear(kv_mem)
            # The column is staged in registers between the two passes: an
            # in-place read-after-write on the same global buffer serialized
            # every j on the previous store (6.5 -> 57 us/call).
            s_loc = T.alloc_local((K,), "float32")
            for j in T.serial(K):
                sj = States[slot, layer, vh, j, tv] * exp_g_s[0]
                s_loc[j] = sj
                kv_mem[0] += sj * k_s[j]
            delta = (v_s[tv] - kv_mem[0]) * beta_s[0]
            acc_o = T.alloc_fragment((1,), "float32")
            T.clear(acc_o)
            for j in T.serial(K):
                sj = s_loc[j] + delta * k_s[j]
                States[slot, layer, vh, j, tv] = sj
                acc_o[0] += sj * q_s[j]
            out_s[tv] = acc_o[0]
            T.tvm_storage_sync("shared")

            # gated RMSNorm + z-gate
            if tv == 0:
                acc_sq = T.alloc_fragment((1,), "float32")
                T.clear(acc_sq)
                for j in T.serial(V):
                    acc_sq[0] += out_s[j] * out_s[j]
                rms_s[0] = T.rsqrt(acc_sq[0] / T.cast(V, "float32") + 1e-6)
            T.tvm_storage_sync("shared")
            gate = Z[bb, vh * V + tv]
            Out[bb, vh * V + tv] = T.cast(
                out_s[tv] * rms_s[0] * NormW[tv] * (gate * T.sigmoid(gate)), out_dtype
            )

            # new conv window: shift left, append current qkv. q/k channels
            # are shared across the GQA group — only the representative writes.
            for tap in T.serial(KER - 2):
                Windows[slot, layer, nxt, tap, vc] = Windows[slot, layer, par, tap + 1, vc]
            Windows[slot, layer, nxt, KER - 2, vc] = Val[bb, vh * V + tv]
            if is_rep:
                for tap in T.serial(KER - 2):
                    Windows[slot, layer, nxt, tap, qc] = Windows[slot, layer, par, tap + 1, qc]
                    Windows[slot, layer, nxt, tap, kc] = Windows[slot, layer, par, tap + 1, kc]
                Windows[slot, layer, nxt, KER - 2, qc] = Q[bb, qc]
                Windows[slot, layer, nxt, KER - 2, kc] = Key[bb, qc]

        return Out

    return gdn_decode_fused


# ---------------------------------------------------------------- gated-delta chunk prefill (fused)


def make_gdn_chunk_fused(target: str):
    """Fused gated-delta chunk prefill core (sm90): the T>1 generalization of
    make_gdn_decode_fused. One block per (value head, batch); thread tv owns
    state column S[:, tv]; a serial scan over T tokens carries the state in a
    per-thread local array (decay-first recurrence, matching
    reference.gdn_forward).

    Replaces reference.gdn_forward's Python head loop on prefill (~150k tiny
    kernel launches per 512-token prefill on the 27B slice: 48 value heads x
    ~8 einsums x T). Same fused ops as decode: conv1d + SiLU + q/k L2-norm +
    decay-first delta recurrence + gated RMSNorm + z-gate. The conv1d history
    (carried Window ++ qkv) is read per tap from HBM like the decode kernel —
    a per-thread sliding window cannot live in shared memory (each thread
    owns a different channel; shared would race) and fragments forbid the
    rq[i]=rq[i+1] shift (uniform-index constraint).

    StepStates/StepWindows (caller-allocated, KS from their shape) receive the
    state and conv window after EACH of the first KS tokens: a speculative
    verify passes KS = chain length and adopts the accepted prefix's state
    afterwards, instead of paying a second forward. Prefill passes KS=1 and
    the guarded write fires once — one plane, off the unrolled inner loops.

    Rows are left-aligned valid tokens with a per-row bound SeqQLens [B]:
    a decode row (seq_q=1, token at t=0) scans one token with the same
    Window++qkv conv semantics as the decode kernel, so mixed batches
    (decode rows + a prefill chunk) run this one kernel.

    # Original: T-loop generalization of make_gdn_decode_fused (itself a SOTA
    # copy of examples/gdn/qwen36_gdr_decode_fused.py @ tilelang branch
    # feat/qwen36-gdn-megakernel). The branch's prefill path is chunkwise-WY
    # (qwen36_prefill_wy.py + qwen36_prefill_scan_o.py); tileRL's decay-first
    # recurrence is serial-within-block instead — within a chunk scan
    # serially over T steps, across chunks carry the state (input State /
    # output NewState are the carry). The chunked-WY reordering of this same
    # recurrence is exact for an affine-in-S0 recurrence — the intra-chunk
    # KKT triangular solve + inter-chunk state carry is the block form of
    # this serial scan, not a different recurrence (measured slower at our
    # shapes, see errors/2026-08-25-gdn-chunked-gdr-rejected).
    # Local state column: the state column (K=128 floats) lives in a
    # per-thread local array (registers/L1), loaded once at the seed and
    # stored once per token in pass 2 — not streamed through global 4x per
    # token (2 loads + 2 stores, strided). 4 accumulators per pass break the
    # 128-deep FMA chain into 4 x 32-deep and issue 4 state loads per
    # iteration (ILP hides the L1/register latency). bf16 IO halves the
    # conv/projection load traffic (Q/Key/Val/Z/Window/NewWindow are bf16;
    # state/out/weights stay f32). Sweep (scripts/_sweep_gdn_prefill2.py,
    # H20 quiet): local+4acc+bf16 is the winner (21.6% faster than the
    # global-state f32 baseline; 8acc and the fused-dot reassociation tested
    # worse). The shared-memory state tile (64KB) was rejected earlier
    # (1.7x slower — LDS bank conflicts, global hits L1 with better
    # pipelining).
    """

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def gdn_chunk_fused(
        Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State, SeqQLens,
        StepStates, StepWindows, threads,
    ):
        # TT (sequence length) is the const, not T: T is the tilelang.language
        # module alias and rebinding it would break T.serial/T.Kernel below.
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

            # per-thread state column: carried in a local array across all T
            # tokens (loaded once at seed, written once after the scan)
            # instead of streamed through global 4x per token. 4 accumulators
            # per pass break the 128-deep FMA chain into 4 x 32-deep and issue
            # 4 state loads per iteration (ILP hides the L1/register latency).
            state_local = T.alloc_local((K,), "float32")
            accs = T.alloc_local((2 * 4,), "float32")
            for j in T.serial(K):
                state_local[j] = State[bb, vh, j, tv]

            for t in T.serial(SeqQLens[bb]):
                # conv1d (KER taps over Window ++ qkv) + SiLU on this head's
                # q/k/v channels — same per-tap global reads as the decode
                # kernel, generalized with the t offset.
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

                # L2-norm by block allreduce (the rmsnorm_fused idiom). Thread
                # 0 alone summing K=128 twice is 256 dependent FMAs on the
                # critical path of EVERY token — at T=512 roughly half this
                # kernel, with the block's other 127 threads idle.
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

                # recurrence: decay + kv_mem, then rank-1 update + out.
                # The state column lives in state_local (registers/L1); the
                # caller only consumes the chunk-end state, so NewState is
                # written once after the scan, not per token.
                # 4 accumulators per pass (chain 128->32 deep, 4 loads/iter).
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
                # Raw core out; the gated RMSNorm and z-gate are the caller's
                # (Backend._gdn_chunk_fused) two kernels now. Done here they
                # were thread 0 summing V=128 serially INSIDE the token loop,
                # with the block's other 127 threads idle between two syncs —
                # on the critical path of every one of T steps, which is why
                # us/step is flat at 3.1 across T=64..512.
                Out[bb, t, vh * V + tv] = acc_o[0]

            # chunk-end state: the only NewState the caller consumes (the
            # next chunk's State seed). Per-token writes were dead stores.
            for j in T.serial(K // 4):
                base = j * 4
                for u in T.unroll(4):
                    NewState[bb, vh, base + u, tv] = state_local[base + u]

            # new conv window: last KER-1 raw qkv tokens of (Window ++ qkv).
            # q/k channels are shared across the GQA group — only the
            # representative writes them.
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

            # same window, but after each of the first KS tokens (s+1 consumed).
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

        return Out, NewState, NewWindow  # StepStates/StepWindows are written in place

    return gdn_chunk_fused
