"""Real-weight parity: sm70 fp16-twiddle GEMV vs f32 reference on CHECKPOINT weights.

The micro parity (scripts/parity_sm70_gemv.py) passed on pack_fp4 weights; the 27B
e2e decodes gibberish. This loads layer-0's actual .wq/.scale/.oscale from the NVFP4
checkpoint and checks each projection — isolates whether the kernel is wrong on real
weights (layout/scale) or the bug is upstream (fuse, gated-delta, engine).

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70f16 \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=/data00/.../Qwen3.8-27B-NVFP4 \
    PYTHONPATH=packages/tilerl-kernels/src:src CUDA_VISIBLE_DEVICES=0 \
    python3 scripts/parity_real_weights.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from tilerl_kernels import reference  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

bk = get_backend()
assert bk.arch == "sm70", f"sm70 only, got {bk.arch}"
src = os.environ["TILERL_QWEN38_SOURCE"]

# Every fp4 projection in layer 0 (gated-delta layer) + a few from a full-attn layer.
keys = [
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.linear_attn.in_proj_z",
    "model.language_model.layers.0.linear_attn.out_proj",
    # a full-attention layer's projections (layer index from config: 3 is full-attn in Qwen3.8)
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.3.self_attn.k_proj",
    "model.language_model.layers.3.self_attn.v_proj",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.3.mlp.gate_proj",
    "model.language_model.layers.3.mlp.up_proj",
    "model.language_model.layers.3.mlp.down_proj",
]

# Find which safetensors file holds each key.
idx = os.path.join(src, "model.safetensors.index.json")
import json  # noqa: E402

with open(idx) as f:
    wmap = json.load(f)["weight_map"]

torch.manual_seed(0)
g = torch.Generator().manual_seed(1)
worst = 0.0
for base in keys:
    wk, sk, ok = f"{base}.wq", f"{base}.scale", f"{base}.oscale"
    if wk not in wmap:
        print(f"  {base}: not in checkpoint (skipped)")
        continue
    fn = os.path.join(src, wmap[wk])
    with safe_open(fn, framework="pt") as st:
        wq = st.get_tensor(wk)
        sc = st.get_tensor(sk)
        osc = st.get_tensor(ok) if ok in st.keys() else None  # noqa: SIM118 - pod safetensors lacks __contains__
    N, K2 = wq.shape
    K = K2 * 2
    # M=1 (decode GEMV), M=8 (M-row kernel), M=16/32 (M=32 prefill chunking).
    for M in (1, 8, 16, 32):
        x = torch.randn(M, K, generator=g, dtype=torch.float32)
        wq_d, sc_d, x_d = wq.to(bk.device), sc.to(bk.device), x.to(bk.device)
        osc_d = osc.to(bk.device) if osc is not None else None
        # reference on the NATURAL bytes (the kernel twiddles in place)
        y_ref = reference.linear_fp4(x, wq, sc, oscale=osc)
        y_ker = bk.linear_fp4(x_d, wq_d, sc_d, oscale=osc_d).float().cpu()
        rel = ((y_ker - y_ref).norm() / y_ref.norm()).item()
        worst = max(worst, rel)
        print(f"  {base.split('.')[-1]:>10} M={M} N={N:>6} K={K:>6}  relerr {rel:.4e}  {'PASS' if rel < 1e-2 else 'FAIL'}")

print(f"\nworst {worst:.4e}  gate 1e-2  {'PASS' if worst < 1e-2 else 'FAIL'}")
sys.exit(1 if worst >= 1e-2 else 0)
