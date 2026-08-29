"""Localize check-2 (input-independent output) by tracking, layer by layer,
the last-token residual COSINE between two DIFFERENT prompts. check 2 collapses
every prompt to the same token, so the prompt signal dies somewhere in the
stack. The first layer whose cross-prompt cosine jumps to ~1.0 is where the
input stops mattering — inspect that op.

A norm/finite check can't see this: both streams stay finite and plausibly
normed while silently converging. Cosine catches it.

  PYTHONPATH=src TILERL_TARGET=cuda python3 -u scripts/health_probe.py \
      /data00/Qwen3.8-27B-NVFP4 --gpu 7 [--tokens 256]

Exit 0 if cosine stays < 0.99 through the stack, 1 otherwise. Dev tooling, no
bench entry. CPU/metal work too (drop --gpu) for a tiny-model smoke.
"""

from __future__ import annotations

import argparse
import os
import sys


def _pin_gpu(gpu: int | None) -> None:
    # Pin before importing torch, same discipline as verify_h20_fp4.py.
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

    from tilerl.config import qwen38_27b, tiny
    from tilerl.model import build_random, load_hf
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    if args.source:
        cfg = qwen38_27b()  # same config path as verify_h20_fp4.py
        model = load_hf(cfg, args.source, fuse_projections=True)
        cfg = model.cfg
    else:
        cfg = tiny()
        model = build_random(cfg, seed=0)
        args.tokens = min(args.tokens, 16)

    n = min(args.tokens, cfg.max_position_embeddings)
    # Two DISTINCT prompts. check 2 collapses all prompts to the same token, so
    # the output is ~input-independent: the prompt signal dies somewhere. Track
    # the last-token residual cosine BETWEEN the two prompts, layer by layer.
    # The first layer whose cosine jumps to ~1.0 is where the input stops
    # mattering — that op (or its predecessor) is the bug. A norm/finite check
    # alone can't see this: both streams stay finite and plausibly-normed.
    rng = np.random.default_rng(0)
    ids_a = rng.integers(1, cfg.vocab_size, size=n, dtype=np.int64).reshape(1, n)
    ids_b = rng.integers(1, cfg.vocab_size, size=n, dtype=np.int64).reshape(1, n)
    positions = np.arange(n, dtype=np.int64)

    from tilerl.train import _training_kv

    def run(ids: np.ndarray):
        # Manual per-layer forward mirroring Model.forward. For each SUBLAYER we
        # capture the added residual delta (out - x), last token — not just the
        # residual. A sublayer that washes out the input has an input-INDEPENDENT
        # delta: its cross-prompt cosine → 1 even while its input still varied.
        # That pins the culprit op, which the whole-residual cosine cannot (the
        # residual is dominated by a large shared component in any real model).
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
        head = "embed_tokens" if cfg.tie_word_embeddings else "lm_head"
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
        # A sublayer whose ADDED delta is ~input-independent (cos>0.99) is the
        # op washing out the prompt. embed/logits rows are residuals, skip them.
        if bad is None and na.endswith("Δ") and (not fin or cos > 0.99):
            bad = (na, cos)
    if bad:
        print(f"\nSIGNAL DIES AT: {bad[0]} (cos={bad[1]:.4f}) — inspect this layer's ops")
        return 1
    print("\nprompt signal preserved through the stack (cos stays < 0.99)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
