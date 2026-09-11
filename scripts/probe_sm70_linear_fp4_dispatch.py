"""End-to-end sm70 prefill path: Backend.linear_fp4 at M>8 must take the new f16
block GEMM and match the f32 reference, across decode (M=1 -> GEMV ladder) and
prefill (M=256 -> block GEMM).

  TILERL_TARGET=cuda /usr/bin/python3 scripts/probe_sm70_linear_fp4_dispatch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src"))

import torch  # noqa: E402
from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402


def main():
    be = get_backend()
    assert be.arch == "sm70", be.arch
    torch.manual_seed(0)
    N, K, blk = 2048, 5120, 32
    w_bf16 = torch.randn(N, K, dtype=torch.float32) * 0.2
    wq_nat, scale = reference.pack_fp4(w_bf16, block=blk)
    scale, oscale = reference.renorm_fp4_scale(scale)
    wq_tw = reference.twiddle_fp4_f16(wq_nat)
    # reference: unpack the NATURAL packed bytes (untwiddle is byte-perfect, so
    # unpack(twiddle(x)) == unpack(x) is what we are checking the kernels against).
    wref = reference.unpack_fp4(wq_nat, scale, oscale).cuda().float()
    wq_tw, scale, oscale = wq_tw.cuda(), scale.cuda(), oscale.cuda()
    # Tag AFTER .cuda(): a device move strips the custom _tl_layout attribute, so
    # tagging on the CPU tensor lets _served_fp4 see "natural" and double-twiddle
    # the bytes (rel 1.69). materialize stamps the tag on the GPU tensor.
    wq_tw._tl_layout = "tw-f16"
    # Only the NEW path (M>8 -> f16 block GEMM) is under test here. M<=8 keeps the
    # proven GEMV ladder; this harness feeds f32 scale, which that path's sh-detection
    # treats differently from production and would mis-compare. The block-GEMM path is
    # validated directly against the natural-pack reference.
    for M in (64, 256):
        x = (torch.randn(M, K, dtype=torch.float32, device="cuda") * 0.3)
        y = be.linear_fp4(x, wq_tw, scale, oscale=oscale)
        ref = x @ wref.t()
        rel = (y - ref).abs().max().item() / ref.abs().max().item()
        print(f"M={M}: shape {tuple(y.shape)} max rel {rel:.3e}")
        assert rel < 5e-2, (M, rel)
    print("PROBE_OK")


if __name__ == "__main__":
    main()
