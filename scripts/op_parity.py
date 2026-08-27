"""Kernel-vs-reference parity at the REAL 27B dims, layer 0 — the check the
tiny parity gate can't do (hidden 64 vs 5120, head_dim 16 vs 256). check 2
collapses the output to an input-independent token; attention+GDN already run
the torch-eager reference yet it still collapses, and linear_fp4 is verified
(check 3). So the culprit is one of rmsnorm / linear_fp8 / silu_mul / embedding
at scale. This runs each on a random activation and prints kernel-vs-reference
fro-relerr; the one that blows up is the bug.

  PYTHONPATH=src TILERL_TARGET=cuda python3 -u scripts/op_parity.py \
      /data00/Qwen3.8-27B-NVFP4 --gpu 7
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("TILERL_TARGET", "cuda")

    import torch

    from tilerl.config import qwen38_27b
    from tilerl.model import load_hf
    from tilerl.ops import reference
    from tilerl.ops.backend import get_backend

    backend = get_backend()
    cfg = qwen38_27b()
    model = load_hf(cfg, args.source, fuse_projections=False)
    cfg = model.cfg
    P = model.params
    g = torch.Generator(device="cpu").manual_seed(0)
    H = cfg.hidden_size

    def relerr(a, b):
        a, b = a.float(), b.float()
        return float((a - b).norm() / b.norm().clamp_min(1e-30))

    def dev(t):  # params are CPU-resident; move to the backend device for the ref
        return t.to(backend.device)

    print(f"\n{'op':<28} {'fro_relerr':>12}")
    print("-" * 42)

    # rmsnorm at real hidden — the op I already found one (grid) bug in.
    x = torch.randn(1, 8, H, generator=g).to(backend.device)
    w = dev(P["layers.0.input_norm"])
    k = backend.rmsnorm(x, w, cfg.rms_eps)
    r = reference.rmsnorm(x, w, cfg.rms_eps)
    print(f"{'rmsnorm[hidden]':<28} {relerr(k, r):>12.4e}")

    # per-head rmsnorm at head_dim 256 (q_norm) — 3D broadcast path.
    d = cfg.head_dim
    xh = torch.randn(1, 8, cfg.num_attention_heads, d, generator=g).to(backend.device)
    wq = dev(P["layers.3.q_norm"])  # layer 3 = first full-attn
    print(f"{'rmsnorm[head_dim]':<28} {relerr(backend.rmsnorm(xh, wq, cfg.rms_eps), reference.rmsnorm(xh, wq, cfg.rms_eps)):>12.4e}")

    # linear_fp8 (GDN in_proj_qkv is fp8 e4m3) at real dims.
    key = "layers.0.in_proj_qkv"  # layer 0 = GDN, fp8
    if P.get(key + ".w8") is not None:
        w8, ws = dev(P[key + ".w8"]), dev(P[key + ".wscale"])
        osc = P.get(key + ".oscale"); osc = dev(osc) if osc is not None else None
        xk = torch.randn(1, 8, w8.shape[1], generator=g).to(backend.device)
        kk = backend.linear_fp8(xk, w8, ws, oscale=osc)
        rr = reference.linear_fp8(xk, w8, ws, oscale=osc) if hasattr(reference, "linear_fp8") else None
        print(f"{'linear_fp8':<28} {relerr(kk, rr):>12.4e}" if rr is not None else f"{'linear_fp8':<28} {'no ref':>12}")

    # silu_mul at intermediate width.
    I = cfg.intermediate_size
    a = torch.randn(1, 8, I, generator=g).to(backend.device)
    b = torch.randn(1, 8, I, generator=g).to(backend.device)
    print(f"{'silu_mul':<28} {relerr(backend.silu_mul(a, b), reference.silu_mul(a, b)):>12.4e}")

    # embedding lookup.
    ids = torch.arange(8, device=backend.device)
    emb = dev(P["embed_tokens"]); ke = backend.embedding(ids, emb)
    re_ = reference.embedding(ids, emb)
    print(f"{'embedding':<28} {relerr(ke, re_):>12.4e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
