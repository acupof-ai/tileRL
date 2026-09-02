"""GDN prefill serial-scan sweep: parallel-reduce / conv-ring / bf16-io / K-split variants of
make_gdn_chunk_fused (kernels_mma.py). Diagnostic only, not shipped.
H20 slice4 (B=1 T=512 nkh=16 nvh=48 K=V=128 KER=4): baseline 4.82 ms; 64KB shared state tile 8.15 ms, REJECTED.
Usage (pod): TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=N PYTHONPATH=src python3 scripts/_sweep_gdn_prefill.py [baseline par ...]
"""

from __future__ import annotations

import sys
import time

import torch
import tilelang
import tilelang.language as T

sys.path.insert(0, "src")
from tilerl_kernels.reference import gdn_forward

B, TT, QD, NVH, K, V, KER = 1, 512, 2048, 48, 128, 128, 4
VD = NVH * V
QKVD = 2 * QD + VD


_PASS = {"tl.disable_data_race_check": True}


def _reduce_add(val, red, tv):
    # red must be a fresh alloc_local per call: reusing one crashes the allreduce lowering (tilelang 0.1.13)
    with T.attr(
        T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
        "reduce_scope",
        T.reinterpret(T.uint64(0), dtype="handle"),
    ):
        T.evaluate(T.tvm_thread_allreduce(T.uint32(1), val, True, red[0], tv, dtype="handle"))


# ---------------------------------------------------------------- baseline


def make_baseline(target):
    @tilelang.jit(target=target, pass_configs=_PASS)
    def ker(Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State):
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
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
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        State: T.Tensor((B, NVH, K, V), "float32")
        Out = T.empty((B, TT, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        with T.Kernel(NVH, B, threads=V) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv
            kc = QD + kh * K + tv
            vc = 2 * QD + vh * V + tv

            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            qn = T.alloc_shared((1,), "float32")
            kn = T.alloc_shared((1,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")
            rms_s = T.alloc_shared((1,), "float32")

            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_mem = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")
            acc_o = T.alloc_fragment((1,), "float32")
            acc_q = T.alloc_fragment((1,), "float32")
            acc_k = T.alloc_fragment((1,), "float32")
            acc_sq = T.alloc_fragment((1,), "float32")

            for j in T.serial(K):
                NewState[bb, vh, j, tv] = State[bb, vh, j, tv]

            for t in T.serial(TT):
                cq[0] = 0.0
                ck[0] = 0.0
                cv[0] = 0.0
                for tap in T.serial(KER):
                    if t + tap < KER - 1:
                        cq[0] += Window[bb, t + tap, qc] * ConvW[qc, tap]
                        ck[0] += Window[bb, t + tap, kc] * ConvW[kc, tap]
                        cv[0] += Window[bb, t + tap, vc] * ConvW[vc, tap]
                    else:
                        cq[0] += Q[bb, t + tap - (KER - 1), qc] * ConvW[qc, tap]
                        ck[0] += Key[bb, t + tap - (KER - 1), qc] * ConvW[kc, tap]
                        cv[0] += Val[bb, t + tap - (KER - 1), vh * V + tv] * ConvW[vc, tap]
                q_s[tv] = cq[0] * T.sigmoid(cq[0])
                k_s[tv] = ck[0] * T.sigmoid(ck[0])
                v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                if tv == 0:
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

                T.clear(kv_mem)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] * exp_g_s[0]
                    NewState[bb, vh, j, tv] = sj
                    kv_mem[0] += sj * k_s[j]
                delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
                T.clear(acc_o)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] + delta[0] * k_s[j]
                    NewState[bb, vh, j, tv] = sj
                    acc_o[0] += sj * q_s[j]
                out_s[tv] = acc_o[0]
                T.tvm_storage_sync("shared")

                if tv == 0:
                    T.clear(acc_sq)
                    for j in T.serial(V):
                        acc_sq[0] += out_s[j] * out_s[j]
                    rms_s[0] = T.rsqrt(acc_sq[0] / T.cast(V, "float32") + 1e-6)
                T.tvm_storage_sync("shared")
                gate = Z[bb, t, vh * V + tv]
                Out[bb, t, vh * V + tv] = (
                    out_s[tv] * rms_s[0] * NormW[tv] * (gate * T.sigmoid(gate))
                )

            for tap in T.serial(KER - 1):
                if TT + tap < KER - 1:
                    NewWindow[bb, tap, vc] = Window[bb, TT + tap, vc]
                else:
                    NewWindow[bb, tap, vc] = Val[bb, TT + tap - (KER - 1), vh * V + tv]
            if is_rep:
                for tap in T.serial(KER - 1):
                    if TT + tap < KER - 1:
                        NewWindow[bb, tap, qc] = Window[bb, TT + tap, qc]
                        NewWindow[bb, tap, kc] = Window[bb, TT + tap, kc]
                    else:
                        NewWindow[bb, tap, qc] = Q[bb, TT + tap - (KER - 1), qc]
                        NewWindow[bb, tap, kc] = Key[bb, TT + tap - (KER - 1), qc]

        return Out, NewState, NewWindow

    return ker


# ---------------------------------------------------------------- parallel reduce


def make_par(target):
    """Baseline + block-wide warp-shuffle reduces for q/k norm and rms; state stays in HBM/L2."""

    @tilelang.jit(target=target, pass_configs=_PASS)
    def ker(Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State):
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
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
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        State: T.Tensor((B, NVH, K, V), "float32")
        Out = T.empty((B, TT, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        with T.Kernel(NVH, B, threads=V) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv
            kc = QD + kh * K + tv
            vc = 2 * QD + vh * V + tv

            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")

            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_mem = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")
            acc_o = T.alloc_fragment((1,), "float32")
            red_q = T.alloc_local((1,), "float32")
            red_k = T.alloc_local((1,), "float32")
            red_o = T.alloc_local((1,), "float32")
            qn = T.alloc_fragment((1,), "float32")
            kn = T.alloc_fragment((1,), "float32")
            rms = T.alloc_fragment((1,), "float32")

            for j in T.serial(K):
                NewState[bb, vh, j, tv] = State[bb, vh, j, tv]

            for t in T.serial(TT):
                cq[0] = 0.0
                ck[0] = 0.0
                cv[0] = 0.0
                for tap in T.serial(KER):
                    if t + tap < KER - 1:
                        cq[0] += Window[bb, t + tap, qc] * ConvW[qc, tap]
                        ck[0] += Window[bb, t + tap, kc] * ConvW[kc, tap]
                        cv[0] += Window[bb, t + tap, vc] * ConvW[vc, tap]
                    else:
                        cq[0] += Q[bb, t + tap - (KER - 1), qc] * ConvW[qc, tap]
                        ck[0] += Key[bb, t + tap - (KER - 1), qc] * ConvW[kc, tap]
                        cv[0] += Val[bb, t + tap - (KER - 1), vh * V + tv] * ConvW[vc, tap]
                q_s[tv] = cq[0] * T.sigmoid(cq[0])
                k_s[tv] = ck[0] * T.sigmoid(ck[0])
                v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                _reduce_add(q_s[tv] * q_s[tv], red_q, tv)
                qn[0] = T.rsqrt(red_q[0] + 1e-12)
                _reduce_add(k_s[tv] * k_s[tv], red_k, tv)
                kn[0] = T.rsqrt(red_k[0] + 1e-12)
                if tv == 0:
                    x = GIn[bb, t, vh] + DtBias[vh]
                    sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                    exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                    beta_s[0] = T.sigmoid(BIn[bb, t, vh])
                T.tvm_storage_sync("shared")

                q_s[tv] = q_s[tv] * qn[0] * scale
                k_s[tv] = k_s[tv] * kn[0]
                T.tvm_storage_sync("shared")

                T.clear(kv_mem)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] * exp_g_s[0]
                    NewState[bb, vh, j, tv] = sj
                    kv_mem[0] += sj * k_s[j]
                delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
                T.clear(acc_o)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] + delta[0] * k_s[j]
                    NewState[bb, vh, j, tv] = sj
                    acc_o[0] += sj * q_s[j]
                out_s[tv] = acc_o[0]
                T.tvm_storage_sync("shared")

                _reduce_add(out_s[tv] * out_s[tv], red_o, tv)
                rms[0] = T.rsqrt(red_o[0] / T.cast(V, "float32") + 1e-6)
                gate = Z[bb, t, vh * V + tv]
                Out[bb, t, vh * V + tv] = out_s[tv] * rms[0] * NormW[tv] * (gate * T.sigmoid(gate))

            for tap in T.serial(KER - 1):
                if TT + tap < KER - 1:
                    NewWindow[bb, tap, vc] = Window[bb, TT + tap, vc]
                else:
                    NewWindow[bb, tap, vc] = Val[bb, TT + tap - (KER - 1), vh * V + tv]
            if is_rep:
                for tap in T.serial(KER - 1):
                    if TT + tap < KER - 1:
                        NewWindow[bb, tap, qc] = Window[bb, TT + tap, qc]
                        NewWindow[bb, tap, kc] = Window[bb, TT + tap, kc]
                    else:
                        NewWindow[bb, tap, qc] = Q[bb, TT + tap - (KER - 1), qc]
                        NewWindow[bb, tap, kc] = Key[bb, TT + tap - (KER - 1), qc]

        return Out, NewState, NewWindow

    return ker


# ---------------------------------------------------------------- par + conv ring


def make_par_ring(target):
    """par + conv1d window as a KER-row shared ring; thread tv owns its own channels, no barriers."""

    @tilelang.jit(target=target, pass_configs=_PASS)
    def ker(Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State):
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
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
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        State: T.Tensor((B, NVH, K, V), "float32")
        Out = T.empty((B, TT, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        with T.Kernel(NVH, B, threads=V) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv
            kc = QD + kh * K + tv
            vc = 2 * QD + vh * V + tv

            # window row w -> slot w+1, token t -> slot t%KER, tap i at t reads slot (t+i+1)%KER
            ring = T.alloc_shared((KER, 3 * V), "float32")
            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")

            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_mem = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")
            acc_o = T.alloc_fragment((1,), "float32")
            red_q = T.alloc_local((1,), "float32")
            red_k = T.alloc_local((1,), "float32")
            red_o = T.alloc_local((1,), "float32")
            qn = T.alloc_fragment((1,), "float32")
            kn = T.alloc_fragment((1,), "float32")
            rms = T.alloc_fragment((1,), "float32")

            for j in T.serial(K):
                NewState[bb, vh, j, tv] = State[bb, vh, j, tv]
            for w in T.serial(KER - 1):
                ring[w + 1, tv] = Window[bb, w, qc]
                ring[w + 1, V + tv] = Window[bb, w, kc]
                ring[w + 1, 2 * V + tv] = Window[bb, w, vc]
            T.tvm_storage_sync("shared")

            for t in T.serial(TT):
                slot = t % KER
                ring[slot, tv] = Q[bb, t, qc]
                ring[slot, V + tv] = Key[bb, t, qc]
                ring[slot, 2 * V + tv] = Val[bb, t, vh * V + tv]
                cq[0] = 0.0
                ck[0] = 0.0
                cv[0] = 0.0
                for tap in T.serial(KER):
                    r = (t + tap + 1) % KER
                    cq[0] += ring[r, tv] * ConvW[qc, tap]
                    ck[0] += ring[r, V + tv] * ConvW[kc, tap]
                    cv[0] += ring[r, 2 * V + tv] * ConvW[vc, tap]
                q_s[tv] = cq[0] * T.sigmoid(cq[0])
                k_s[tv] = ck[0] * T.sigmoid(ck[0])
                v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                _reduce_add(q_s[tv] * q_s[tv], red_q, tv)
                qn[0] = T.rsqrt(red_q[0] + 1e-12)
                _reduce_add(k_s[tv] * k_s[tv], red_k, tv)
                kn[0] = T.rsqrt(red_k[0] + 1e-12)
                if tv == 0:
                    x = GIn[bb, t, vh] + DtBias[vh]
                    sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                    exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                    beta_s[0] = T.sigmoid(BIn[bb, t, vh])
                T.tvm_storage_sync("shared")

                q_s[tv] = q_s[tv] * qn[0] * scale
                k_s[tv] = k_s[tv] * kn[0]
                T.tvm_storage_sync("shared")

                T.clear(kv_mem)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] * exp_g_s[0]
                    NewState[bb, vh, j, tv] = sj
                    kv_mem[0] += sj * k_s[j]
                delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
                T.clear(acc_o)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] + delta[0] * k_s[j]
                    NewState[bb, vh, j, tv] = sj
                    acc_o[0] += sj * q_s[j]
                out_s[tv] = acc_o[0]
                T.tvm_storage_sync("shared")

                _reduce_add(out_s[tv] * out_s[tv], red_o, tv)
                rms[0] = T.rsqrt(red_o[0] / T.cast(V, "float32") + 1e-6)
                gate = Z[bb, t, vh * V + tv]
                Out[bb, t, vh * V + tv] = out_s[tv] * rms[0] * NormW[tv] * (gate * T.sigmoid(gate))

            for tap in T.serial(KER - 1):
                r = (TT + tap + 1) % KER
                NewWindow[bb, tap, vc] = ring[r, 2 * V + tv]
            if is_rep:
                for tap in T.serial(KER - 1):
                    r = (TT + tap + 1) % KER
                    NewWindow[bb, tap, qc] = ring[r, tv]
                    NewWindow[bb, tap, kc] = ring[r, V + tv]

        return Out, NewState, NewWindow

    return ker


# ---------------------------------------------------------------- par + ring + bf16 IO


def make_par_ring_bf16(target):
    """par_ring + bf16 global IO for q/k/v/z/window; state and ring stay f32."""

    @tilelang.jit(target=target, pass_configs=_PASS)
    def ker(Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State):
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
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
        Out = T.empty((B, TT, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "bfloat16")
        with T.Kernel(NVH, B, threads=V) as (vh, bb):
            tv = T.get_thread_binding(0)
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv
            kc = QD + kh * K + tv
            vc = 2 * QD + vh * V + tv

            ring = T.alloc_shared((KER, 3 * V), "float32")
            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")

            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_mem = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")
            acc_o = T.alloc_fragment((1,), "float32")
            red_q = T.alloc_local((1,), "float32")
            red_k = T.alloc_local((1,), "float32")
            red_o = T.alloc_local((1,), "float32")
            qn = T.alloc_fragment((1,), "float32")
            kn = T.alloc_fragment((1,), "float32")
            rms = T.alloc_fragment((1,), "float32")

            for j in T.serial(K):
                NewState[bb, vh, j, tv] = State[bb, vh, j, tv]
            for w in T.serial(KER - 1):
                ring[w + 1, tv] = T.cast(Window[bb, w, qc], "float32")
                ring[w + 1, V + tv] = T.cast(Window[bb, w, kc], "float32")
                ring[w + 1, 2 * V + tv] = T.cast(Window[bb, w, vc], "float32")
            T.tvm_storage_sync("shared")

            for t in T.serial(TT):
                slot = t % KER
                ring[slot, tv] = T.cast(Q[bb, t, qc], "float32")
                ring[slot, V + tv] = T.cast(Key[bb, t, qc], "float32")
                ring[slot, 2 * V + tv] = T.cast(Val[bb, t, vh * V + tv], "float32")
                cq[0] = 0.0
                ck[0] = 0.0
                cv[0] = 0.0
                for tap in T.serial(KER):
                    r = (t + tap + 1) % KER
                    cq[0] += ring[r, tv] * ConvW[qc, tap]
                    ck[0] += ring[r, V + tv] * ConvW[kc, tap]
                    cv[0] += ring[r, 2 * V + tv] * ConvW[vc, tap]
                q_s[tv] = cq[0] * T.sigmoid(cq[0])
                k_s[tv] = ck[0] * T.sigmoid(ck[0])
                v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                _reduce_add(q_s[tv] * q_s[tv], red_q, tv)
                qn[0] = T.rsqrt(red_q[0] + 1e-12)
                _reduce_add(k_s[tv] * k_s[tv], red_k, tv)
                kn[0] = T.rsqrt(red_k[0] + 1e-12)
                if tv == 0:
                    x = GIn[bb, t, vh] + DtBias[vh]
                    sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                    exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                    beta_s[0] = T.sigmoid(BIn[bb, t, vh])
                T.tvm_storage_sync("shared")

                q_s[tv] = q_s[tv] * qn[0] * scale
                k_s[tv] = k_s[tv] * kn[0]
                T.tvm_storage_sync("shared")

                T.clear(kv_mem)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] * exp_g_s[0]
                    NewState[bb, vh, j, tv] = sj
                    kv_mem[0] += sj * k_s[j]
                delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
                T.clear(acc_o)
                for j in T.serial(K):
                    sj = NewState[bb, vh, j, tv] + delta[0] * k_s[j]
                    NewState[bb, vh, j, tv] = sj
                    acc_o[0] += sj * q_s[j]
                out_s[tv] = acc_o[0]
                T.tvm_storage_sync("shared")

                _reduce_add(out_s[tv] * out_s[tv], red_o, tv)
                rms[0] = T.rsqrt(red_o[0] / T.cast(V, "float32") + 1e-6)
                gate = T.cast(Z[bb, t, vh * V + tv], "float32")
                Out[bb, t, vh * V + tv] = out_s[tv] * rms[0] * NormW[tv] * (gate * T.sigmoid(gate))

            for tap in T.serial(KER - 1):
                r = (TT + tap + 1) % KER
                NewWindow[bb, tap, vc] = T.cast(ring[r, 2 * V + tv], "bfloat16")
            if is_rep:
                for tap in T.serial(KER - 1):
                    r = (TT + tap + 1) % KER
                    NewWindow[bb, tap, qc] = T.cast(ring[r, tv], "bfloat16")
                    NewWindow[bb, tap, kc] = T.cast(ring[r, V + tv], "bfloat16")

        return Out, NewState, NewWindow

    return ker


# ---------------------------------------------------------------- par + K-split


def make_par_ksplit(target):
    """par + K-split: 2 threads per V column, each K/2 rows; h=0 owners do conv/norm/output."""

    @tilelang.jit(target=target, pass_configs=_PASS)
    def ker(Q, Key, Val, Z, GIn, BIn, DtBias, ALog, NormW, ConvW, Window, State):
        B, TT, QD, NVH, K, V, KER = T.const("B, TT, QD, NVH, K, V, KER")
        VD = NVH * V
        HK = K // 2
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
        Window: T.Tensor((B, KER - 1, QKVD), "float32")
        State: T.Tensor((B, NVH, K, V), "float32")
        Out = T.empty((B, TT, VD), "float32")
        NewState = T.empty((B, NVH, K, V), "float32")
        NewWindow = T.empty((B, KER - 1, QKVD), "float32")
        with T.Kernel(NVH, B, threads=2 * V) as (vh, bb):
            tid = T.get_thread_binding(0)
            tv = tid % V  # value column
            h = tid // V  # 0 = K rows 0..HK-1, 1 = rows HK..K-1
            kh = vh * (QD // K) // NVH
            is_rep = (vh % (NVH // (QD // K))) == 0
            qc = kh * K + tv
            kc = QD + kh * K + tv
            vc = 2 * QD + vh * V + tv

            q_s = T.alloc_shared((K,), "float32")
            k_s = T.alloc_shared((K,), "float32")
            v_s = T.alloc_shared((V,), "float32")
            exp_g_s = T.alloc_shared((1,), "float32")
            beta_s = T.alloc_shared((1,), "float32")
            out_s = T.alloc_shared((V,), "float32")
            partial_s = T.alloc_shared((V,), "float32")
            delta_s = T.alloc_shared((V,), "float32")

            cq = T.alloc_fragment((1,), "float32")
            ck = T.alloc_fragment((1,), "float32")
            cv = T.alloc_fragment((1,), "float32")
            kv_p = T.alloc_fragment((1,), "float32")
            out_p = T.alloc_fragment((1,), "float32")
            delta = T.alloc_fragment((1,), "float32")
            red_q = T.alloc_local((1,), "float32")
            red_k = T.alloc_local((1,), "float32")
            red_o = T.alloc_local((1,), "float32")
            qn = T.alloc_fragment((1,), "float32")
            kn = T.alloc_fragment((1,), "float32")
            rms = T.alloc_fragment((1,), "float32")

            for j in T.serial(K):
                NewState[bb, vh, j, tv] = State[bb, vh, j, tv]

            for t in T.serial(TT):
                if h == 0:
                    cq[0] = 0.0
                    ck[0] = 0.0
                    cv[0] = 0.0
                    for tap in T.serial(KER):
                        if t + tap < KER - 1:
                            cq[0] += Window[bb, t + tap, qc] * ConvW[qc, tap]
                            ck[0] += Window[bb, t + tap, kc] * ConvW[kc, tap]
                            cv[0] += Window[bb, t + tap, vc] * ConvW[vc, tap]
                        else:
                            cq[0] += Q[bb, t + tap - (KER - 1), qc] * ConvW[qc, tap]
                            ck[0] += Key[bb, t + tap - (KER - 1), qc] * ConvW[kc, tap]
                            cv[0] += Val[bb, t + tap - (KER - 1), vh * V + tv] * ConvW[vc, tap]
                    q_s[tv] = cq[0] * T.sigmoid(cq[0])
                    k_s[tv] = ck[0] * T.sigmoid(ck[0])
                    v_s[tv] = cv[0] * T.sigmoid(cv[0])
                T.tvm_storage_sync("shared")

                pq = T.if_then_else(h == 0, q_s[tv] * q_s[tv], 0.0)
                _reduce_add(pq, red_q, tid)
                qn[0] = T.rsqrt(red_q[0] + 1e-12)
                pk = T.if_then_else(h == 0, k_s[tv] * k_s[tv], 0.0)
                _reduce_add(pk, red_k, tid)
                kn[0] = T.rsqrt(red_k[0] + 1e-12)
                if h == 0 and tv == 0:
                    x = GIn[bb, t, vh] + DtBias[vh]
                    sp = T.if_then_else(x > 20.0, x, T.log(1.0 + T.exp(x)))
                    exp_g_s[0] = T.exp(-T.exp(ALog[vh]) * sp)
                    beta_s[0] = T.sigmoid(BIn[bb, t, vh])
                T.tvm_storage_sync("shared")

                if h == 0:
                    q_s[tv] = q_s[tv] * qn[0] * scale
                    k_s[tv] = k_s[tv] * kn[0]
                T.tvm_storage_sync("shared")

                T.clear(kv_p)
                for j in T.serial(HK):
                    jj = h * HK + j
                    sj = NewState[bb, vh, jj, tv] * exp_g_s[0]
                    NewState[bb, vh, jj, tv] = sj
                    kv_p[0] += sj * k_s[jj]
                if h == 1:
                    partial_s[tv] = kv_p[0]
                T.tvm_storage_sync("shared")
                if h == 0:
                    delta[0] = (v_s[tv] - (kv_p[0] + partial_s[tv])) * beta_s[0]
                    delta_s[tv] = delta[0]
                T.tvm_storage_sync("shared")

                T.clear(out_p)
                for j in T.serial(HK):
                    jj = h * HK + j
                    sj = NewState[bb, vh, jj, tv] + delta_s[tv] * k_s[jj]
                    NewState[bb, vh, jj, tv] = sj
                    out_p[0] += sj * q_s[jj]
                if h == 1:
                    partial_s[tv] = out_p[0]
                T.tvm_storage_sync("shared")
                if h == 0:
                    out_s[tv] = out_p[0] + partial_s[tv]
                T.tvm_storage_sync("shared")

                po = T.if_then_else(h == 0, out_s[tv] * out_s[tv], 0.0)
                _reduce_add(po, red_o, tid)
                rms[0] = T.rsqrt(red_o[0] / T.cast(V, "float32") + 1e-6)
                if h == 0:
                    gate = Z[bb, t, vh * V + tv]
                    Out[bb, t, vh * V + tv] = (
                        out_s[tv] * rms[0] * NormW[tv] * (gate * T.sigmoid(gate))
                    )

            if h == 0:
                for tap in T.serial(KER - 1):
                    if TT + tap < KER - 1:
                        NewWindow[bb, tap, vc] = Window[bb, TT + tap, vc]
                    else:
                        NewWindow[bb, tap, vc] = Val[bb, TT + tap - (KER - 1), vh * V + tv]
                if is_rep:
                    for tap in T.serial(KER - 1):
                        if TT + tap < KER - 1:
                            NewWindow[bb, tap, qc] = Window[bb, TT + tap, qc]
                            NewWindow[bb, tap, kc] = Window[bb, TT + tap, kc]
                        else:
                            NewWindow[bb, tap, qc] = Q[bb, TT + tap - (KER - 1), qc]
                            NewWindow[bb, tap, kc] = Key[bb, TT + tap - (KER - 1), qc]

        return Out, NewState, NewWindow

    return ker


VARIANTS = {
    "baseline": make_baseline,
    "par": make_par,
    "par_ring": make_par_ring,
    "par_ring_bf16": make_par_ring_bf16,
    "par_ksplit": make_par_ksplit,
}


def _make_inputs(device, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, TT, QD, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, TT, QD, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, TT, VD, device=device, dtype=dtype) * 0.1
    z = torch.randn(B, TT, VD, device=device, dtype=dtype) * 0.1
    g = torch.randn(B, TT, NVH, device=device)
    beta = torch.randn(B, TT, NVH, device=device)
    state = torch.randn(B, NVH, K, V, device=device) * 0.01
    window = torch.randn(B, KER - 1, QKVD, device=device, dtype=dtype) * 0.1
    kw = dict(
        conv1d_weight=torch.randn(QKVD, KER, device=device) * 0.1,
        dt_bias=torch.randn(NVH, device=device),
        a_log=torch.randn(NVH, device=device) * 0.1,
        norm_weight=torch.ones(V, device=device),
        conv_window=window,
    )
    return q, k, v, z, g, beta, state, kw


def _ker_args(q, k, v, z, g, beta, state, kw):
    return (
        q,
        k,
        v,
        z,
        g,
        beta,
        kw["dt_bias"],
        kw["a_log"],
        kw["norm_weight"],
        kw["conv1d_weight"],
        kw["conv_window"],
        state,
    )


def _time(fn, iters=20):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    names = sys.argv[1:] or list(VARIANTS)
    dev = "cuda"
    q, k, v, z, g, beta, state, kw = _make_inputs(dev, torch.float32)
    ref = gdn_forward(q, k, v, g, beta, state, z=z, **kw)
    print(f"reference done (out {ref[0].shape})", flush=True)

    for name in names:
        make = VARIANTS[name]
        bf16 = name.endswith("bf16")
        t0 = time.perf_counter()
        ker = make("cuda")
        if bf16:
            args = _ker_args(
                q.bfloat16(),
                k.bfloat16(),
                v.bfloat16(),
                z.bfloat16(),
                g,
                beta,
                state,
                {**kw, "conv_window": kw["conv_window"].bfloat16()},
            )
        else:
            args = _ker_args(q, k, v, z, g, beta, state, kw)
        out, ns, nw = ker(*args)
        torch.cuda.synchronize()
        jit = time.perf_counter() - t0

        err_o = (out.float() - ref[0]).abs().max().item() / ref[0].abs().max().item()
        err_s = (ns - ref[1]).abs().max().item() / ref[1].abs().max().item()
        nw_ref = ref[2]
        err_w = (nw.float() - nw_ref).abs().max().item() / nw_ref.abs().max().item()
        ok = max(err_o, err_s, err_w) < 1e-2

        ms = _time(lambda: ker(*args))
        print(
            f"{name:<22}: {ms:.4f} ms  rel-err out {err_o:.2e} state {err_s:.2e} "
            f"win {err_w:.2e}  {'OK' if ok else 'FAIL'}  (jit {jit:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
