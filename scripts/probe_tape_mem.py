"""Peak allocated memory of one training step, one lever at a time.

Four arms on the same card in one process: the group in one backward or in
micro-batches, MLP activations stored or replayed, and the logit gradient
written into the logits or into three fresh vocab-sized buffers. `--layers`
truncates the checkpoint so the pre-fix arm completes instead of OOMing.
"""

from __future__ import annotations

import argparse


def legacy_ce(logits, input_ids):
    """The shape-for-shape CE this replaced: four [B,T,vocab] f32 tensors live."""
    import torch

    b, t, v = logits.shape
    flat = logits[:, :-1].float().reshape(-1, v)
    labels = torch.as_tensor(input_ids, dtype=torch.long, device=flat.device)[:, 1:].reshape(-1)
    loss = (torch.logsumexp(flat, -1) - flat.gather(-1, labels[:, None]).squeeze(-1)).mean()
    grad = torch.softmax(flat, -1)
    grad.scatter_add_(-1, labels[:, None], -torch.ones_like(labels[:, None], dtype=grad.dtype))
    grad /= flat.shape[0]
    out = torch.zeros(b, t, v, dtype=torch.float32, device=logits.device)
    out[:, :-1] = grad.reshape(b, t - 1, v)
    return float(loss), out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--len", dest="length", type=int, default=275)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--micro", type=int, default=0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--no-recompute", action="store_true", help="store MLP activations")
    ap.add_argument("--legacy-ce", action="store_true", help="the pre-fix vocab-sized CE")
    args = ap.parse_args()

    import numpy as np
    import torch

    from tilerl import autograd
    from tilerl.autograd import AdamW
    from tilerl.train import rl_step
    from tilerl_kernels.backend import get_backend

    if args.no_recompute:
        autograd.checkpoint = lambda fn, *a: fn(*a)
    backend = get_backend()
    if args.legacy_ce:
        backend.cross_entropy_loss_grad = legacy_ce
    if args.source:
        from tilerl.config import qwen38_27b
        from tilerl.model import add_lora, load_hf

        cfg = qwen38_27b()
        model = load_hf(cfg, args.source, fuse_projections=False, num_layers=args.layers)
        model.params = backend.materialize(model.params)
        trainable = add_lora(model, rank=args.lora_rank)
    else:
        from tilerl.cli import _build_model

        cfg, model = _build_model("tiny", seed=0, keep_master=True)
        trainable = None

    b, t = args.group, args.length
    rng = np.random.default_rng(0)
    ids = rng.integers(1, cfg.vocab_size, size=(b, t)).astype(np.int64)
    adv = rng.normal(size=b)
    plens = np.full(b, t // 4, dtype=np.int64)
    slens = np.full(b, t, dtype=np.int64)

    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        weights = torch.cuda.memory_allocated() / 2**30
    rl_step(model, ids, adv, plens, backend, AdamW(lr=1e-5), trainable=trainable,
            seq_lens=slens, micro=args.micro)
    print(f"layers={args.layers} T={t} group={b} micro={args.micro} "
          f"recompute={not args.no_recompute} ce={'legacy' if args.legacy_ce else 'in-place'}")
    if backend.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  weights {weights:.2f} GiB   step peak {peak:.2f} GiB   "
              f"activations {peak - weights:.2f} GiB")


if __name__ == "__main__":
    main()
