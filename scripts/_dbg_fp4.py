import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src"))

import torch  # noqa: E402
from tilerl_kernels import kernels_linear as kl  # noqa: E402
from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend, _pad2d  # noqa: E402

torch.manual_seed(0)
be = get_backend()
N, K, blk = 2048, 5120, 32
M = 64
w = torch.randn(N, K, dtype=torch.float32) * 0.2
wq, scale = reference.pack_fp4(w, block=blk)
scale, oscale = reference.renorm_fp4_scale(scale)
wq_tw = reference.twiddle_fp4_f16(wq)
wref = reference.unpack_fp4(wq, scale, oscale).cuda().float()
x = (torch.randn(M, K, dtype=torch.float32, device="cuda") * 0.3)
ref = x @ wref.t()

# direct kernel (known good path from standalone)
fn = kl.make_linear_fp4_f16_mma_sm70("cuda")
yd = fn(x.half(), wq_tw.cuda(), scale.cuda(), oscale.cuda(), 64, 64, blk, 128)
print("direct rel", (yd - ref).abs().max().item() / ref.abs().max().item())

# backend
wqt = wq_tw.clone(); wqt._tl_layout = "tw-f16"
yb = be.linear_fp4(x, wqt, scale.cuda(), oscale=oscale.cuda())
print("backend rel", (yb - ref).abs().max().item() / ref.abs().max().item())
print("direct vs backend rel", (yb - yd).abs().max().item() / yd.abs().max().item())
print("x dtype into backend rows; backend io", be.io)
