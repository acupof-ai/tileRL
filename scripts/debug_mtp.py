"""Debug MTP head: test input-dependence with two different hidden states.

Loads the 27B + MTP head, runs the MTP head with two different hidden states
and token IDs, and compares the outputs. If the outputs are the same, the
MTP head is input-independent (broken). If different, the issue is with the
actual hidden states from the trunk.

Usage:
    PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=7 \
        python3 scripts/debug_mtp.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse

import torch

from tilerl.config import qwen38_27b
from tilerl.model import load_hf
from tilerl.mtp import load_mtp_head
from tilerl.ops.backend import get_backend


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    args = p.parse_args()

    backend = get_backend()
    cfg = qwen38_27b()

    import time

    t0 = time.perf_counter()
    model = load_hf(cfg, args.source, fuse_projections=True)
    mtp = load_mtp_head(args.source)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    # Migrate weights to GPU.
    mtp.params = {k: v.to(backend.device) for k, v in mtp.params.items()}
    model.params = {
        k: v.to(backend.device) if isinstance(v, torch.Tensor) else v
        for k, v in model.params.items()
    }

    # Test with TWO DIFFERENT hidden states and token IDs.
    torch.manual_seed(42)
    h1 = torch.randn(1, cfg.hidden_size, device=backend.device, dtype=torch.float32)
    t1 = torch.tensor([100], device=backend.device)
    torch.manual_seed(99)
    h2 = torch.randn(1, cfg.hidden_size, device=backend.device, dtype=torch.float32)
    t2 = torch.tensor([200], device=backend.device)

    # Run MTP head with both inputs.
    logits1 = mtp.forward(backend, h1, t1, model)
    logits2 = mtp.forward(backend, h2, t2, model)

    draft1 = logits1.argmax(dim=-1).item()
    draft2 = logits2.argmax(dim=-1).item()

    print(f"\nInput 1: token=100, hidden_seed=42", flush=True)
    print(f"  draft={draft1} logits_max={logits1.max().item():.4f} "
          f"logits_min={logits1.min().item():.4f}", flush=True)
    print(f"  top5={logits1.topk(5).indices[0].tolist()}", flush=True)

    print(f"\nInput 2: token=200, hidden_seed=99", flush=True)
    print(f"  draft={draft2} logits_max={logits2.max().item():.4f} "
          f"logits_min={logits2.min().item():.4f}", flush=True)
    print(f"  top5={logits2.topk(5).indices[0].tolist()}", flush=True)

    diff = (logits1 - logits2).abs()
    print(f"\nDiff: max={diff.max().item():.6f} mean={diff.mean().item():.6f}", flush=True)
    print(f"Same draft? {draft1 == draft2}", flush=True)

    # Also test with SAME hidden, DIFFERENT token (isolate embedding contribution).
    logits3 = mtp.forward(backend, h1, t2, model)
    draft3 = logits3.argmax(dim=-1).item()
    print(f"\nInput 3: token=200, hidden_seed=42 (same hidden as input 1)", flush=True)
    print(f"  draft={draft3} logits_max={logits3.max().item():.4f}", flush=True)
    diff13 = (logits1 - logits3).abs()
    print(f"  Diff vs input 1: max={diff13.max().item():.6f} mean={diff13.mean().item():.6f}", flush=True)
    print(f"  Same draft as input 1? {draft1 == draft3}", flush=True)

    # And SAME token, DIFFERENT hidden (isolate hidden contribution).
    logits4 = mtp.forward(backend, h2, t1, model)
    draft4 = logits4.argmax(dim=-1).item()
    print(f"\nInput 4: token=100, hidden_seed=99 (same token as input 1)", flush=True)
    print(f"  draft={draft4} logits_max={logits4.max().item():.4f}", flush=True)
    diff14 = (logits1 - logits4).abs()
    print(f"  Diff vs input 1: max={diff14.max().item():.6f} mean={diff14.mean().item():.6f}", flush=True)
    print(f"  Same draft as input 1? {draft1 == draft4}", flush=True)


if __name__ == "__main__":
    main()
