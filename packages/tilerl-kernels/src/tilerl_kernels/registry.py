"""Precision x arch dispatch matrix and target resolution. A new arch is one
_register() call; empty sets are pending-remote (docs/support-matrix.md)."""

from __future__ import annotations

import os

import torch

from . import kernels, kernels_attn, kernels_gdn, kernels_linear, kernels_mma

_REGISTRY: dict[tuple[str, str], dict[str, object]] = {}

# sm70 split counts, chosen per tick by query width. 32 splits are 1.20x faster
# at S=1 where PO is 3 MiB; by S=8 they are within 0.5% of 16 while PO reaches
# 1.5 GiB, so a wide tick halves it for free. The threshold clears the widest
# verify the ladder submits (depth 7 is S=8). Exported so the parity gate and the
# guard test read the shipped rule instead of restating it.
SM70_KVSPLIT = 32
SM70_KVSPLIT_WIDE = 16


def sm70_kvsplit(s: int) -> int:
    return SM70_KVSPLIT if s < 8 else SM70_KVSPLIT_WIDE


def _register(precision: str, arch: str, kernels: dict[str, object]) -> None:
    _REGISTRY[(precision, arch)] = kernels


def _resolve(precision: str, arch: str) -> dict[str, object]:
    for key in ((precision, arch), (precision, "any"), ("any", "any")):
        if key in _REGISTRY:
            if not _REGISTRY[key]:
                raise NotImplementedError(
                    f"({precision!r}, {arch!r}) is pending-remote bring-up "
                    "(see docs/support-matrix.md)"
                )
            return _REGISTRY[key]
    raise KeyError(f"no kernel set for ({precision!r}, {arch!r})")


_CPU_KERNELS = {  # f32 kernels, bf16 cast at the boundary
    "rmsnorm_partial": kernels.make_rmsnorm_partial,
    "rmsnorm_apply": kernels.make_rmsnorm_apply,
    "rmsnorm_rstd": kernels.make_rmsnorm_rstd,
    "rmsnorm_bwd_x": kernels.make_rmsnorm_bwd_x,
    "gemm_nt": kernels.make_gemm_nt,
    "gemm_nn": kernels.make_gemm_nn,
    "gemm_tn": kernels.make_gemm_tn,
    "silu_mul": kernels.make_silu_mul,
    "softmax": kernels.make_softmax,
    "rope": kernels.make_rope,
    "embedding": kernels.make_embedding,
    "linear_fp4": kernels.make_linear_fp4,
    "paged_attention": kernels.make_paged_attention,
    "gdn_prep": kernels.make_gdn_prep,
    "gdn_post": kernels.make_gdn_post,
}
_register("bf16", "cpu", _CPU_KERNELS)
_register("fp4", "cpu", _CPU_KERNELS)  # fp4 is a weight format, not a compute dtype
_METAL_KERNELS = {  # Metal's T.gemm rejects global operands
    **_CPU_KERNELS,
    "gemm_nt": kernels.make_gemm_nt_naive,
    "gemm_nn": kernels.make_gemm_nn_naive,
    "gemm_tn": kernels.make_gemm_tn_naive,
}
_register("bf16", "metal", _METAL_KERNELS)
_register("fp4", "metal", _METAL_KERNELS)
_SM90_KERNELS = {  # WGMMA schedules; the backend pads M/N to 16 and K to 32
    **_CPU_KERNELS,
    "rmsnorm_apply": kernels.make_rmsnorm_apply_bf16,
    "rmsnorm_fused": kernels.make_rmsnorm_fused_bf16,
    # q/k norm: the output survives to the bf16 KV pool, so a bf16 store here
    # rounds twice (errors/2026-09-03-unfused-prelude-double-rounds.md)
    "rmsnorm_fused_f32": kernels.make_rmsnorm_fused_f32,
    "silu_mul": kernels.make_silu_mul_bf16,
    "gemm_nt": kernels_linear.make_gemm_nt_mma,
    "gemm_nn": kernels_linear.make_gemm_nn_mma,
    "gemm_tn": kernels_linear.make_gemm_tn_mma,
    "linear_fp4": kernels_linear.make_linear_fp4_mma,
    "linear_fp4_gemv": kernels_linear.make_linear_fp4_gemv,
    "linear_fp4_mma8": kernels_linear.make_linear_fp4_mma8,
    "linear_fp4_bwd": kernels_linear.make_linear_fp4_bwd_mma,
    "linear_bf16_gemv": kernels_linear.make_linear_bf16_gemv,
    "linear_fp4_fp8": lambda target: kernels_linear.make_linear_fp4_fp8_mma(target, k_split=2),
    # decode (M<=16): 8-way K-split buys resident warps, +7.5% at B=8
    # (wins/2026-08-26-batch-decode-h2.md)
    "linear_fp4_fp8_decode": lambda target: kernels_linear.make_linear_fp4_fp8_mma(
        target, k_split=8
    ),
    "linear_fp8": kernels_linear.make_linear_fp8_mma,
    "linear_fp8_gemv": kernels_linear.make_linear_fp8_gemv,
    "linear_fp8_mma8": kernels_linear.make_linear_fp8_mma8,
    "quant_fp8": kernels_linear.make_quant_fp8_e4m3,
    "write_tokens": kernels_mma.make_write_tokens,
    "attn_prep": kernels_mma.make_attn_prep,
    "gdn_decode_fused": kernels_gdn.make_gdn_decode_fused,
    "gdn_chunk_fused": kernels_gdn.make_gdn_chunk_fused,
    "gdn_prep": kernels_gdn.make_gdn_prep_bf16,
    "gdn_post": lambda t: kernels.make_gdn_post(t, "bfloat16"),
    # chunkwise-WY prefill: the default for whole-chunk full-length rows
    "gdn_chunk_cumsum": kernels_gdn.make_gdn_chunk_cumsum,
    "gdn_chunk_kkt": kernels_gdn.make_gdn_chunk_kkt,
    "gdn_solve_tril": kernels_gdn.make_gdn_solve_tril,
    "gdn_chunk_wu": kernels_gdn.make_gdn_chunk_wu,
    "gdn_state_scan": kernels_gdn.make_gdn_state_scan,
    "gdn_chunk_o": kernels_gdn.make_gdn_chunk_o,
    "paged_attention": kernels_attn.make_paged_attention_mma,
    "paged_attention_decode": kernels_attn.make_paged_attention_decode,
    "paged_attention_combine": kernels_attn.make_paged_attention_combine,
    # long context (>64K): 64 splits keep the per-block scan at <= 4K tokens
    "paged_attention_decode_64": lambda t: kernels_attn.make_paged_attention_decode(t, KVSPLIT=64),
    "paged_attention_combine_64": lambda t: kernels_attn.make_paged_attention_combine(t, KVSPLIT=64),
}
_register("bf16", "sm90", _SM90_KERNELS)
_register("fp4", "sm90", _SM90_KERNELS)
# Volta: T.gemm lowers to fp16-only mma.sync.m8n8k4, so the sm90 MMA family is
# dead — the cell is the CPU f32 floor plus the kernels that also run on Volta.
_SM70_KERNELS = {
    **_CPU_KERNELS,
    # Narrow variants for the elementwise ops whose output feeds a GEMV: the sm70
    # GEMV wants X in f16 and used to cast it at dispatch — one cast per launch,
    # over bytes rmsnorm/silu_mul had just written. Producing f16 at the source
    # removes 193 of the 305 casts a dense token pays. Separate keys, not a
    # replacement: q_norm/k_norm feed rope and attention, which are f32.
    "rmsnorm_apply_narrow": lambda t: kernels.make_rmsnorm_apply_bf16(t, out_dtype="float16"),
    "silu_mul": lambda t: kernels.make_silu_mul_bf16(t, out_dtype="float16"),
    "linear_fp4_gemv": kernels_linear.make_linear_fp4_gemv_sm70,
    # M-row ladder (decode/verify M<=8, prefill M=32) as ONE entry: M/xh/sh are
    # factory args and Backend._kernel keys the compile cache on them, so a
    # 2-row verify does not pay for 8. Rounding X to f16 once outside the kernel
    # took 127 us/row flat down to 24-45 us/row.
    "linear_fp4_gemv_sm70_m": kernels_linear.make_linear_fp4_gemv_sm70_m,
    # gdn_decode_fused and write_tokens fix graph capture: their eager fallbacks
    # host-sync on int(device_tensor) per token.
    "gdn_decode_fused": lambda t: kernels_gdn.make_gdn_decode_fused(t, out_dtype="float32"),
    # without it prefill (T>1) falls to reference.gdn_forward, a Python serial
    # scan — ~250k eager ops for 8x64, 62s of the 64s tick 1
    "gdn_chunk_fused": kernels_gdn.make_gdn_chunk_fused,
    # f32 pool: the attention kernel is f32-IO, and a bf16 pool cast the whole
    # plane per call (4.71 ms/token, 14% of a 4096-ctx token)
    "write_tokens": kernels_mma.make_write_tokens_f32,
    # split-KV decode attention, S>=1 (speculative verify too); every sm70
    # attention call takes it, leaving the generic kernel to the other targets.
    # sm70 only: the source is target-neutral but the win is filling 80 SMs, so
    # it loses where T.Kernel lowers to a serial loop (cpu) and is unproven on
    # metal. Bare factories: KVSPLIT comes from the call site (backend.py) so a
    # wide tick can trade splits for footprint, and Backend._kernel keys the
    # compile cache on it.
    "paged_attention_split": kernels.make_paged_attention_split,
    "paged_attention_split_combine": kernels.make_paged_attention_split_combine,
}
_register("bf16", "sm70", _SM70_KERNELS)
_register("fp4", "sm70", _SM70_KERNELS)
for _arch in ("sm100", "sm120"):
    _register("bf16", _arch, {})  # pending-remote slot


def _arch_for(target: str) -> str:
    if target == "c":
        return "cpu"
    if target.startswith("cuda") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    return {"hip": "rocm", "metal": "metal"}.get(target, target)


def resolve_target() -> str:
    """TILERL_TARGET=cpu|cuda|metal|auto. ``auto`` is cuda when visible, else
    ``c`` (tilelang's own auto picks metal on a Mac, not the dev/CI path)."""
    target = os.environ.get("TILERL_TARGET", "auto").strip().lower()
    aliases = {"cpu": "c", "": "auto"}
    target = aliases.get(target, target)
    if target == "auto":
        return "cuda" if torch.cuda.is_available() else "c"
    return target
