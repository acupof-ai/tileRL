"""Localize an input-independent output: per-sublayer cross-prompt cosine of
the last-token residual delta between two prompts. The first sublayer whose
delta cosine reaches ~1.0 is where the prompt stops mattering (a norm/finite
check cannot see two streams silently converging). Exit 0 if cos < 0.99 throughout.
  PYTHONPATH=src TILERL_TARGET=cuda python3 -u scripts/health_probe.py \
      /data00/Qwen3.8-27B-NVFP4 --gpu 7 [--tokens 256]   (no source = tiny smoke)
"""

from __future__ import annotations

import argparse
import os
import sys


def _pin_gpu(gpu: int | None) -> None:  # before importing torch
    if gpu is not None:
        if gpu not in (6, 7):
            print(f"FATAL: GPU {gpu} is not ours (only 6,7)")
            sys.exit(2)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ.setdefault("TILERL_TARGET", "cuda")
    else:
        os.environ.setdefault("TILERL_TARGET", "cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="checkpoint dir (omit for tiny smoke)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--tokens", type=int, default=256, help="prefill length to probe")
    args = ap.parse_args()
    _pin_gpu(args.gpu)

    import numpy as np
    import torch
    from tilerl_kernels.backend import get_backend

    from tilerl.config import qwen38_27b, tiny
    from tilerl.model import build_random, load_hf

    backend = get_backend()
    if args.source:
        cfg = qwen38_27b()
        model = load_hf(cfg, args.source, fuse_projections=True)
        cfg = model.cfg
    else:
        cfg = tiny()
        model = build_random(cfg, seed=0)
        args.tokens = min(args.tokens, 16)

    n = min(args.tokens, cfg.max_position_embeddings)
    rng = np.random.default_rng(0)
    ids_a = rng.integers(1, cfg.vocab_size, size=n, dtype=np.int64).reshape(1, n)
    ids_b = rng.integers(1, cfg.vocab_size, size=n, dtype=np.int64).reshape(1, n)
    positions = np.arange(n, dtype=np.int64)

    from tilerl.train import _training_kv

    def run(ids: np.ndarray):
        # mirrors Model.forward; captures each sublayer's delta, since the whole
        # residual is dominated by a shared component and cannot pin the op
        kv = _training_kv(model, 1, n, device=backend.device)
        pos = torch.as_tensor(positions, dtype=torch.long, device=backend.device)
        x = backend.embedding(
            torch.as_tensor(ids, dtype=torch.long, device=backend.device),
            model.params["embed_tokens"],
        )
        caps = [("embed", x.detach()[0, -1].float().clone())]
        linear_idx = 0
        for i in range(cfg.num_layers):
            x0 = x
            if cfg.is_full_attn(i):
                x = model._full_attn(i, x, pos, kv, backend)
                tag = f"attn{i}"
            else:
                x = model._gdn(i, linear_idx, x, kv, backend)
                linear_idx += 1
                tag = f"gdn{i}"
            caps.append((tag + "Δ", (x - x0).detach()[0, -1].float().clone()))
            x1 = x
            x = model._mlp(i, x, backend)
            caps.append((f"mlp{i}Δ", (x - x1).detach()[0, -1].float().clone()))
        x = backend.rmsnorm(x, model.params["final_norm"], cfg.rms_eps)
        head = cfg.head_key
        logits = model._linear(backend, x, head)
        caps.append(("logits", logits.detach()[0, -1].float().clone()))
        return caps

    try:
        ca, cb = run(ids_a), run(ids_b)
    except Exception as exc:
        print(f"\nFORWARD RAISED: {type(exc).__name__}: {exc}")
        return 1

    print(f"\n{'probe':<10} {'|a|':>11} {'|b|':>11} {'cos(a,b)':>10} {'finite':>7}")
    print("-" * 54)
    bad = None
    for (na, va), (_, vb) in zip(ca, cb):
        fin = bool(torch.isfinite(va).all() and torch.isfinite(vb).all())
        cos = float(torch.nn.functional.cosine_similarity(va, vb, dim=0)) if fin else float("nan")
        print(f"{na:<10} {va.norm():>11.3e} {vb.norm():>11.3e} {cos:>10.4f} {str(fin):>7}")
        if bad is None and na.endswith("Δ") and (not fin or cos > 0.99):
            bad = (na, cos)
    if bad:
        print(f"\nSIGNAL DIES AT: {bad[0]} (cos={bad[1]:.4f}) — inspect this layer's ops")
        return 1
    print("\nprompt signal preserved through the stack (cos stays < 0.99)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
