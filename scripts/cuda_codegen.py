"""Emit CUDA C source for a tileRL kernel on a host with no GPU and no nvcc.

The macOS tilelang wheel is built ``USE_CUDA=OFF`` (CMakeLists auto-selects
Metal on APPLE), so ``libtilelang.dylib`` ships **zero** ``tl.cuda.*`` globals:
all 14 CUDA passes and both ``target.build.tilelang_cuda*`` codegen entries are
absent. Nothing about the target string or pass_configs can reach them.

Two substitutions get a GEMV all the way to CUDA C anyway:

1. The 14 CUDA passes become identity. They lower TMA / mbarrier / tcgen05 /
   warp-specialization / persistent-CTA constructs, none of which a decode GEMV
   contains, so dropping them is a no-op *for this kernel class* (see the
   coverage note below).
2. ``target.build.tilelang_cuda_without_compile`` becomes TVM's stock
   ``target.build.cuda``, which is present. On a ``USE_CUDA=OFF`` build it
   deliberately returns a ``CUDAFallbackModuleNode`` carrying the raw source
   for later cross-compile -- exactly the no-nvcc path we want.

Covered: ``linear_fp4_gemv`` / ``linear_bf16_gemv`` / ``linear_fp8_gemv`` --
register-only kernels with no ``T.copy`` / ``T.gemm``. Not covered: every MMA
kernel, which needs ``src/cuda/op/copy.cc`` + ``gemm.cc`` (compiled out of the
wheel) and dies with "tl.copy requires a target-specific implementation".

The emitter is TVM's ``CodeGenCUDA``, not tilelang's ``CodeGenTileLangCUDA``,
so the text is for *inspection*, not a byte-exact preview of a pod build. The
vector-load widths are safe to read off it regardless: they are fixed in the
device TIR by VectorizeLoop, before any codegen runs.
"""

import sys

import tilelang
import tilelang.cuda.codegen  # noqa: F401  registers the real cuda codegen first
import tilelang.cuda.transform as ct
from tilelang.backend.device_codegen import (
    DeviceCodegen,
    global_func_device_codegen,
    register_device_codegen,
)

TARGET = {"kind": "cuda", "arch": "sm_90a"}  # bare "cuda" silently becomes sm_50


def enable():
    for name in ct.__all__:
        setattr(ct, name, lambda *a, **k: (lambda mod: mod))
    register_device_codegen(
        "cuda",
        DeviceCodegen(
            "cuda",
            build=global_func_device_codegen("target.build.cuda"),
            build_without_compile=global_func_device_codegen("target.build.cuda"),
            supports_target=lambda t: t.kind.name == "cuda" and "cutedsl" not in t.keys,
        ),
        override=True,
    )
    tilelang.disable_cache()  # the disk cache tries to load a cubin loader that isn't here


if __name__ == "__main__":
    import torch

    enable()
    from tilerl_kernels import kernels_linear as kl

    N, K = 5120, 17408  # down_proj
    src = kl.make_linear_fp4_gemv(TARGET).get_kernel_source(
        torch.zeros(1, K, dtype=torch.bfloat16),
        torch.zeros(N, K // 2, dtype=torch.uint8),
        torch.zeros(N, K // 16),
        32,
        4,
        16,
    )
    # micro_size_k=8 packs 4 uint8 per thread -> LDG.32 on the weight stream,
    # against LDG.128 (uint4) on the activation. This is the whole lever.
    assert "*(uint*)(WQ" in src and "*(uint4*)(X " in src, "load widths moved"
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(src)
    print(f"{len(src)} bytes", *sys.argv[1:2])
