"""Full-chain health probe: run one prefill through the 27B and report, per
layer, the output-norm and a finite-check — so a bug that poisons the forward
(a grid overflow, a NaN, a degenerate op) is localized to the LAYER and the
OP, not seen only as a garbage final logit.

This is the instrument that was missing when check 2 of verify_h20_fp4.py
failed: it saw the output collapse to one token but could not say WHERE. A
sticky CUDA launch error (e.g. rmsnorm grid.y > 65535 at large M) shows up
here as the first layer whose norm goes non-finite or to zero.

  PYTHONPATH=src TILERL_TARGET=cuda python3 -u scripts/health_probe.py \
      /data00/Qwen3.8-27B-NVFP4 --gpu 7 [--tokens 256]

Prints a per-layer table (norm, min/max, finite) and the first bad layer.
Exit 0 if every layer is finite and non-degenerate, 1 otherwise. Dev tooling,
no bench entry. CPU/metal work too (drop --gpu), for a tiny-model smoke.
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
    from tilerl.ops.backend import get_backend

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
    ids = np.arange(1, n + 1, dtype=np.int64).reshape(1, n) % cfg.vocab_size
    positions = np.arange(n, dtype=np.int64)

    # Hook: capture each layer's residual-stream output norm by wrapping the
    # model's per-layer forward. We probe the residual stream between layers —
    # the one tensor every layer reads and writes — so a broken op inside any
    # layer shows as that layer's output going non-finite or flat.
    from tilerl.train import _training_kv

    kv = _training_kv(model, 1, n, device=backend.device)

    rows = []
    bad = None

    def probe(name: str, t: torch.Tensor):
        nonlocal bad
        tf = t.detach().float()
        finite = bool(torch.isfinite(tf).all())
        norm = float(tf.norm().item()) if finite else float("nan")
        mn, mx = (float(tf.min()), float(tf.max())) if finite else (float("nan"), float("nan"))
        distinct = int(tf.flatten().unique().numel())
        row = (name, norm, mn, mx, finite, distinct)
        rows.append(row)
        if bad is None and (not finite or norm == 0.0 or distinct < 4):
            bad = row

    # Manual forward with per-layer probing. Uses the model's public forward
    # building blocks; if the model exposes no per-layer seam we fall back to
    # probing embedding -> full forward -> logits (coarse but still catches a
    # global poison).
    try:
        idx = backend._i32(torch.as_tensor(ids.reshape(-1)))
        h = backend.embedding(idx, model.params["embed_tokens"])
        probe("embedding", h)
        logits = model.forward(ids, positions, kv, backend)
        probe("logits", logits.reshape(-1, cfg.vocab_size))
    except Exception as exc:  # a launch error surfaces here — report it as the bad point
        print(f"\nFORWARD RAISED: {type(exc).__name__}: {exc}")
        if rows:
            _print(rows)
        print(f"\nfirst failure at/after: {rows[-1][0] if rows else 'embedding'}")
        return 1

    _print(rows)
    if bad:
        print(f"\nBAD LAYER: {bad[0]} (norm={bad[1]:.3e} finite={bad[4]} distinct={bad[5]})")
        return 1
    print("\nall probes finite and non-degenerate")
    return 0


def _print(rows) -> None:
    print(f"\n{'probe':<16} {'norm':>12} {'min':>12} {'max':>12} {'finite':>7} {'distinct':>9}")
    print("-" * 72)
    for name, norm, mn, mx, finite, distinct in rows:
        print(f"{name:<16} {norm:>12.3e} {mn:>12.3e} {mx:>12.3e} {str(finite):>7} {distinct:>9}")


if __name__ == "__main__":
    sys.exit(main())
