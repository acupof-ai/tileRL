"""Exercise the fused instruction kernel before loading the checkpoint."""

import json

import torch
from tilerl_kernels.kernels_mega import ADD, LINEAR, NORM, make_megakernel
from tilerl_kernels.reference import pack_fp4, renorm_fp4_scale, twiddle_fp4_f16


def main():
    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(1, 512, device=device)
    norm = torch.ones(512, device=device)
    h = torch.empty_like(x)
    w = torch.randn(128, 512, device=device) * 0.03
    y = torch.empty(1, 128, device=device)
    residual = torch.randn_like(y)
    out = torch.empty_like(y)
    dummy = torch.zeros(1, device=device)
    xh = torch.empty_like(x, dtype=torch.float16)
    ptrs = [dummy, x, norm, h, w, y, residual, out, xh]
    program = torch.tensor([
        [NORM, 1, 512, 0, 1, 3, 2, 0, 0, 0, 0, 0],
        [LINEAR, 1, 128, 512, 3, 5, 4, 0, 0, 8, 0, 0],
        [ADD, 1, 128, 0, 5, 7, 0, 0, 0, 6, 0, 0],
    ], dtype=torch.int64, device=device)
    pointers = torch.tensor([p.data_ptr() for p in ptrs], device=device, dtype=torch.uint64)
    position = torch.zeros(1, device=device, dtype=torch.int32)
    kernel = make_megakernel(80, len(program), len(ptrs))
    kernel(program, pointers, position)
    expected = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)) @ w.T + residual
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
    print(json.dumps({"gate": "norm_linear_add", "max_abs": (out - expected).abs().max().item()}))
    packed, scale = pack_fp4(w)
    scale, oscale = renorm_fp4_scale(scale)
    packed = twiddle_fp4_f16(packed)
    scale = scale.half()
    ptrs = [dummy, x, norm, h, packed, y, residual, out, xh, scale, oscale]
    pointers = torch.tensor([p.data_ptr() for p in ptrs], device=device, dtype=torch.uint64)
    program[1, 7:9] = torch.tensor([9, 10], device=device)
    program[1, 11] = 32
    kernel = make_megakernel(80, len(program), len(ptrs))
    kernel(program, pointers, position)
    from tilerl_kernels.kernels_linear import make_linear_fp4_gemv_sm70_m
    baseline = make_linear_fp4_gemv_sm70_m("cuda", 1, 4, True, True)
    reference = baseline(h.half(), packed, scale, oscale, torch.zeros_like(y), 32, 4, 32) + residual
    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
    print(json.dumps({"gate": "fp4_linear_add", "max_abs": (out - reference).abs().max().item()}))


if __name__ == "__main__":
    main()
