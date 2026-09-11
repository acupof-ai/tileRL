"""Standalone parity for the sm70 w4a16 prefill block GEMM.

Checks make_linear_fp4_f16_mma_sm70 against the reference f32 linear on random
weights, feeding f16-TWIDDLED wq (the layout materialize stamps on sm70). A
small M,N,K first (smoke), then a prefill-shaped M=256 tile.

  TILERL_TARGET=cuda PYTHONPATH=src:packages/tilerl-kernels/src \
    /usr/bin/python3 scripts/probe_sm70_fp4_f16_mma.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src"))

import torch  # noqa: E402
from tilerl_kernels import kernels_linear as kl  # noqa: E402
from tilerl_kernels import reference  # noqa: E402


def run_case(M, N, K, block=32, bM=64, bN=64):
    torch.manual_seed(0)
    w_bf16 = torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.2
    wq, scale = reference.pack_fp4(w_bf16, block=block)
    scale, oscale = reference.renorm_fp4_scale(scale)
    wq_tw = reference.twiddle_fp4_f16(wq).cuda()
    scale = scale.cuda()
    oscale = oscale.cuda()
    x = (torch.randn(M, K, dtype=torch.float32, device="cuda") * 0.3).half()
    fn = kl.make_linear_fp4_f16_mma_sm70("cuda")
    y = fn(x, wq_tw, scale, oscale, min(bM, M), bN, block, 128)
    # reference f32
    wref = reference.unpack_fp4(wq.cuda(), scale, oscale).float()
    ref = x.float() @ wref.t()
    err = (y - ref).abs()
    rel = err.max().item() / ref.abs().max().item()
    p99 = torch.quantile(err.flatten().float(), 0.99).item()
    # ms (median 20 after 5 warms)
    for _ in range(5):
        fn(x, wq_tw, scale, oscale, min(bM, M), bN, block, 128)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        fn(x, wq_tw, scale, oscale, min(bM, M), bN, block, 128)
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e) / 20
    print(f"M{M} N{N} K{K}: max abs {err.max():.4f} p99 {p99:.4f} max rel {rel:.3e}  {ms:.3f} ms")
    return rel, ms


def main():
    r1, _ = run_case(64, 128, 64, block=32, bM=64, bN=64)
    # prefill-shaped 27B rows (N,K from the real layers): GDN gate 17408x5120,
    # attn q_proj 12288x5120; M sweeps 32 (old ladder top rung) -> 256.
    for M in (32, 64, 128, 256):
        r, ms = run_case(M, 17408, 5120, block=32, bM=min(M, 64), bN=64)
        print(f"  -> {ms / M:.4f} ms/token-row at M={M}")
        assert r < 5e-2, r
    assert r1 < 5e-2, r1
    print("PROBE_OK")


if __name__ == "__main__":
    main()
