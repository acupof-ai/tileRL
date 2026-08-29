"""Where does a train step's memory actually sit? Phase-by-phase, plus the
largest live tensors at the backward peak.

Four LoRA peaks (47.0/50.5/57.6/71.8 GB at 1x64/128/256, 2x256) fit a line
with slope 0.055 GiB/token and a ~28.6 GiB intercept. The intercept does not
scale with T, so it is not activations, and it is bigger than anything the
full-FT ledger can save. This finds it.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_train_ledger.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import torch

from tilerl.autograd import AdamW, RecordingBackend, Tape
from tilerl.config import qwen38_27b
from tilerl.model import add_lora, load_hf
from tilerl.train import _training_kv
from tilerl_kernels.backend import get_backend

G = 2**30


def mark(label: str, prev: float) -> float:
    torch.cuda.synchronize()
    cur = torch.cuda.memory_allocated() / G
    peak = torch.cuda.max_memory_allocated() / G
    print(f"  {label:28s} live {cur:7.2f}  (+{cur - prev:6.2f})  peak {peak:7.2f}")
    torch.cuda.reset_peak_memory_stats()
    return cur


def live_tensors(top: int = 12, named: dict | None = None) -> None:
    """Largest live CUDA storages, labelled with the param key that owns them
    when one does — an unattributed vocab-shaped tensor is the interesting
    case, and shape alone cannot tell embed_tokens from lm_head."""
    owner = {v.untyped_storage().data_ptr(): k for k, v in (named or {}).items()
             if torch.is_tensor(v) and v.is_cuda}
    seen, rows = set(), []
    for o in gc.get_objects():
        try:
            if not torch.is_tensor(o) or not o.is_cuda:
                continue
        except Exception:
            continue
        st = o.untyped_storage()
        if st.data_ptr() in seen:
            continue
        seen.add(st.data_ptr())
        rows.append((st.nbytes() / G, tuple(o.shape), str(o.dtype).replace("torch.", ""),
                     owner.get(st.data_ptr(), "-")))
    rows.sort(reverse=True)
    print(f"  {'-- largest live storages':28s} total {sum(r[0] for r in rows):7.2f} GiB")
    for sz, shape, dt, who in rows[:top]:
        print(f"     {sz:7.3f} GiB  {dt:9s} {str(shape):20s} {who}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("-b", type=int, default=1)
    ap.add_argument("-t", type=int, default=64)
    args = ap.parse_args()

    b = get_backend()
    cfg = qwen38_27b()
    p = 0.0
    p = mark("start", p)
    model = load_hf(cfg, args.source, fuse_projections=False, num_layers=args.layers)
    p = mark("load_hf", p)
    model.params = b.materialize(model.params)
    p = mark("materialize", p)
    trainable = add_lora(model, rank=16)
    p = mark("add_lora", p)
    opt = AdamW(lr=1e-3)

    ids = np.arange(1, args.t + 1, dtype=np.int64).reshape(args.b, args.t) % cfg.vocab_size
    positions = np.arange(args.t, dtype=np.int64)
    kv = _training_kv(model, args.b, args.t, device=b.device)
    p = mark("training kv", p)
    live_tensors(named=model.params)

    tape = Tape()
    with torch.no_grad(), tape:
        logits = model.forward(ids, positions, kv, RecordingBackend(b))
    p = mark(f"forward (tape {len(tape._entries)})", p)
    live_tensors(named=model.params)

    grad_logits = torch.zeros_like(logits)
    grads = tape.backward(grad_logits)
    p = mark("backward", p)
    live_tensors(named=model.params)

    param_ids = {id(x) for x in trainable.values()}
    grads = {k: v for k, v in grads.items() if k in param_ids}
    p = mark("grads filtered", p)
    opt.step(trainable.values(), grads)
    mark("optimizer", p)


if __name__ == "__main__":
    main()
