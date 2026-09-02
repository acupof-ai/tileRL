"""How much of a train step's memory is intermediate gradients the tape holds past their last
use? `Tape.backward` keeps activation grads in the same dict as param grads until return."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--len", dest="length", type=int, default=256)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--full", action="store_true", help="full-parameter, not LoRA")
    args = ap.parse_args()

    import numpy as np
    import torch

    from tilerl.autograd import RecordingBackend, Tape
    from tilerl.train import _training_kv
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    if args.source:
        from tilerl.config import qwen38_27b
        from tilerl.model import add_lora, load_hf
        cfg = qwen38_27b()
        # keep_master builds the bf16 masters the STE path records onto; the fp4 base is frozen
        model = load_hf(cfg, args.source, fuse_projections=False, num_layers=args.layers,
                        keep_master=args.full)
        model.params = backend.materialize(model.params)
        trainable = None if args.full else add_lora(model, rank=args.lora_rank)
    else:
        from tilerl.cli import _build_model
        cfg, model = _build_model("tiny", seed=0, keep_master=True)
        trainable = None

    ids = np.arange(1, args.length + 1, dtype=np.int64).reshape(1, args.length) % cfg.vocab_size
    b, t = ids.shape
    kv = _training_kv(model, b, t, device=backend.device)
    tape = Tape()
    with torch.no_grad(), tape:
        logits = model.forward(ids, np.arange(t, dtype=np.int64), kv, RecordingBackend(backend))
    _, gl = backend.cross_entropy_loss_grad(logits, ids)

    entries = list(tape._entries)
    produced = {id(e.output) for e in entries}
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / 2**30
    grads = tape.backward(gl)
    if backend.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30

    kept = sum(g.numel() * g.element_size() for g in grads.values()) / 2**30
    print(f"{'full' if trainable is None else 'lora'}  layers={args.layers} T={t}")
    print(f"  tape entries              {len(entries)}")
    print(f"  produced (intermediates)  {len(produced)}")
    print(f"  returned param grads      {len(grads)}   {kept:.2f} GiB")
    if backend.device.type == "cuda":
        print(f"  backward peak             {peak:.2f} GiB (base {base:.2f})")
        print(f"  held beyond the result    {peak - base - kept:.2f} GiB")


if __name__ == "__main__":
    main()
