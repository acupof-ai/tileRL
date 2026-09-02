"""Emit CUDA C for a register-only GEMV kernel on a host with no GPU/nvcc.
The macOS wheel is USE_CUDA=OFF, so the 14 CUDA passes become identity (a GEMV
uses none of them) and TVM's stock target.build.cuda emits the source. MMA
kernels are not covered (copy.cc/gemm.cc are compiled out). The emitter is
TVM's CodeGenCUDA, so read vector-load widths off it, not byte-exact text.
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
    # micro_size_k=8: LDG.32 on the weight stream, LDG.128 on the activation
    assert "*(uint*)(WQ" in src and "*(uint4*)(X " in src, "load widths moved"
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(src)
    print(f"{len(src)} bytes", *sys.argv[1:2])
