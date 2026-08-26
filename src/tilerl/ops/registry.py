"""Precision x arch dispatch matrix: kernel sets keyed by (precision, arch),
target resolution, and the cpu/metal/sm90 cell definitions.

Adding fp8 or a new SM arch is ONE _register() call. Full matrix:
docs/support-matrix.md.
"""

from __future__ import annotations

import os

import torch

from . import kernels
from . import kernels_attn
from . import kernels_gdn
from . import kernels_linear
from . import kernels_mma

__all__ = ["resolve_target"]

# ---------------------------------------------------------------------------
# Kernel sets are keyed by (precision, arch); _resolve walks the fallback
# chain exact -> (precision, "any") -> ("any", "any"). Day-1: bf16/fp4 on CPU
# (f32-compute kernels, bf16 cast at the boundary); GPU arches are
# pending-remote slots — registered so the matrix is honest, NotImplementedError
# on use.
# ---------------------------------------------------------------------------

_REGISTRY: dict[tuple[str, str], dict[str, object]] = {}


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


_CPU_KERNELS = {  # bf16 on CPU: the f32 TileLang kernels (bf16 cast at the boundary)
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
    "linear_attn_chunk": kernels.make_linear_attn_chunk,
}
_register("bf16", "cpu", _CPU_KERNELS)
# fp4 is a weight format, not a compute dtype: its cell reuses the bf16 set
# (linear_fp4 is in it; the rest of the layer is the bf16 path).
_register("fp4", "cpu", _CPU_KERNELS)
# metal: same target-neutral kernel source, except the three gemms — Metal's
# T.gemm lowering rejects global operands, so the metal cell swaps in the
# naive FMA schedules from kernels.py (see "gemm (naive FMA schedule)").
_METAL_KERNELS = {
    **_CPU_KERNELS,
    "gemm_nt": kernels.make_gemm_nt_naive,
    "gemm_nn": kernels.make_gemm_nn_naive,
    "gemm_tn": kernels.make_gemm_tn_naive,
}
_register("bf16", "metal", _METAL_KERNELS)
_register("fp4", "metal", _METAL_KERNELS)
# sm90: the MMA (WGMMA) schedules from kernels_linear.py / kernels_gdn.py /
# kernels_attn.py — shared-memory tiled T.gemm with pipelining, the SOTA
# pattern from examples/gemm/example_gemm.py.
# The naive FMA gemms stay in kernels.py as the metal/other-arch fallback.
# The MMA kernels require block M/N divisible by 16 and the reduction dim
# divisible by 32; the CUDA path of linear/linear_bwd/linear_fp4
# zero-pads tails so the kernel always sees exact tiles.
_SM90_KERNELS = {
    **_CPU_KERNELS,
    "gemm_nt": kernels_linear.make_gemm_nt_mma,
    "gemm_nn": kernels_linear.make_gemm_nn_mma,
    "gemm_tn": kernels_linear.make_gemm_tn_mma,
    "linear_fp4": kernels_linear.make_linear_fp4_mma,
    "linear_fp4_gemv": kernels_linear.make_linear_fp4_gemv,
    "linear_bf16_gemv": kernels_linear.make_linear_bf16_gemv,
    "linear_fp4_fp8": lambda target: kernels_linear.make_linear_fp4_fp8_mma(target, k_split=2),
    # Decode (M<=16): 8-way K-split. At bM=16 a block is 2 warps, so the
    # split buys resident warps for HBM latency hiding — +7.5% at B=8 vs the
    # prefill kernel's 2-way (A/B 2026-08-26, docs/experience/wins/
    # 2026-08-26-batch-decode-h2.md).
    "linear_fp4_fp8_decode": lambda target: kernels_linear.make_linear_fp4_fp8_mma(
        target, k_split=8
    ),
    "linear_fp8": kernels_linear.make_linear_fp8_mma,
    "linear_fp8_gemv": kernels_linear.make_linear_fp8_gemv,
    "quant_fp8": kernels_linear.make_quant_fp8_e4m3,
    "write_tokens": kernels_mma.make_write_tokens,
    "gdn_decode_fused": kernels_gdn.make_gdn_decode_fused,
    "gdn_chunk_fused": kernels_gdn.make_gdn_chunk_fused,
    "paged_attention": kernels_attn.make_paged_attention_mma,
}
_register("bf16", "sm90", _SM90_KERNELS)
_register("fp4", "sm90", _SM90_KERNELS)
for _arch in ("sm100", "sm120", "rocm"):
    _register("bf16", _arch, {})  # pending-remote slot


def _arch_for(target: str) -> str:
    """Arch tag for the matrix: cpu | sm90 | sm100 | ... | rocm | metal."""
    if target == "c":
        return "cpu"
    if target.startswith("cuda") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    return {"hip": "rocm", "metal": "metal"}.get(target, target)


def resolve_target() -> str:
    """Resolve the tilelang target string for this process.

    ``TILERL_TARGET`` overrides; accepts the friendly names ``cpu|cuda|rocm|
    metal|auto`` (cpu -> ``"c"``, the working CPU target; rocm -> ``"hip"``).
    ``auto`` (the default) maps to ``"cuda"`` when a CUDA device is visible and
    ``"c"`` otherwise — tilelang's own ``auto`` would pick metal on this Mac,
    which is not the dev/CI path.
    """
    target = os.environ.get("TILERL_TARGET", "auto").strip().lower()
    aliases = {"cpu": "c", "rocm": "hip", "": "auto"}
    target = aliases.get(target, target)
    if target == "auto":
        return "cuda" if torch.cuda.is_available() else "c"
    return target
