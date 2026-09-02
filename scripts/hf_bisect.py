"""Per-layer cosine + relerr of tileRL's forward against /work/hf_ref.pt
(dumped by hf_reference.py) on the same token ids; the first layer with
cos << 1 is the bug site.
  python3 -u scripts/hf_bisect.py /data00/Qwen3.8-27B-NVFP4 --gpu 7
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--ref", default="/work/hf_ref.pt")
    ap.add_argument("--no-fuse", action="store_true", help="load unfused (isolate a fusion bug)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import torch

    from tilerl.config import qwen38_27b
    from tilerl.model import load_hf
    from tilerl_kernels.backend import get_backend
    from tilerl.train import _training_kv

    ref = torch.load(args.ref, map_location="cpu")
    ids = ref["ids"].numpy()  # [1, T]
    hf_h = ref["hidden"]  # [65, 5120] (embed + 64 layers), last token
    n = ids.shape[1]

    backend = get_backend()
    cfg = qwen38_27b()
    model = load_hf(cfg, args.source, fuse_projections=not args.no_fuse)
    cfg = model.cfg
    positions = torch.arange(n, dtype=torch.long, device=backend.device)

    kv = _training_kv(model, 1, n, device=backend.device)
    x = backend.embedding(
        torch.as_tensor(ids, dtype=torch.long, device=backend.device),
        model.params["embed_tokens"],
    )
    caps = [x.detach()[0, -1].float().cpu()]  # embed
    linear_idx = 0
    for i in range(cfg.num_layers):
        if cfg.is_full_attn(i):
            x = model._full_attn(i, x, positions, kv, backend)
        else:
            x = model._gdn(i, linear_idx, x, kv, backend)
            linear_idx += 1
        x = model._mlp(i, x, backend)
        caps.append(x.detach()[0, -1].float().cpu())

    print(f"\nprompt ids {ids.tolist()}  ({n} tokens)")
    print(f"{'layer':<8} {'cos(tilerl,hf)':>16} {'relerr':>12} {'|tl|':>10} {'|hf|':>10}")
    print("-" * 60)
    bad = None
    for i, (tl, hf) in enumerate(zip(caps, hf_h)):
        cos = float(torch.nn.functional.cosine_similarity(tl, hf, dim=0))
        rel = float((tl - hf).norm() / hf.norm().clamp_min(1e-30))
        name = "embed" if i == 0 else f"L{i-1}"
        print(f"{name:<8} {cos:>16.4f} {rel:>12.4e} {tl.norm():>10.3e} {hf.norm():>10.3e}")
        if bad is None and cos < 0.98:
            bad = (name, cos)
    if bad:
        print(f"\nFIRST DIVERGENCE: {bad[0]} (cos={bad[1]:.4f}) — the bug is in this layer's ops/wiring")
    else:
        print("\ntileRL matches HF at every layer (cos>=0.98) — bug is in final_norm/lm_head")
    return 0


if __name__ == "__main__":
    sys.exit(main())
