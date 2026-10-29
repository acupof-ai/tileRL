"""Check an unchanged FP8 checkpoint face on Volta."""

import json

import torch
from tilerl_kernels.kernels_mega import LINEAR, make_megakernel
from tilerl_kernels.reference import dequant_fp8


def main():
    torch.manual_seed(37)
    b, n, k = 4, 256, 512
    x = torch.randn(b, k, device="cuda")
    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    ws = torch.rand(n // 128, k // 128, device="cuda") * 0.05
    oscale = torch.rand(n, device="cuda") * 0.1
    out = torch.empty(b, n, device="cuda")
    ptrs = [x, w, ws, oscale, out]
    pointers = torch.tensor([p.data_ptr() for p in ptrs], device="cuda", dtype=torch.uint64)
    program = torch.tensor([[LINEAR,b,n,k,0,4,1,2,3,0,0,-1]], device="cuda", dtype=torch.int64)
    kernel = make_megakernel(80, 1, len(ptrs))
    kernel(program, pointers, torch.zeros(1, device="cuda", dtype=torch.int32))
    weights = (dequant_fp8(w, ws) * oscale[:, None]).bfloat16().float()
    expected = x @ weights.T
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
    print(json.dumps({"gate":"fp8_face", "max_abs":(out-expected).abs().max().item()}))
    from tilerl_kernels.kernels_fp8_sm70 import make_linear_fp8_sm70
    baseline = make_linear_fp8_sm70("cuda")(x, w.view(torch.uint8), ws, oscale)
    torch.testing.assert_close(baseline, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(baseline, out, rtol=1e-4, atol=1e-5)
    print(json.dumps({"gate":"fp8_baseline", "max_abs":(baseline-expected).abs().max().item()}))


if __name__ == "__main__":
    main()
