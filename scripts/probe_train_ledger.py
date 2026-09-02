"""Train-step memory phase by phase, plus the largest live tensors at the backward peak.
LoRA peaks fit 0.055 GiB/token with a ~28.6 GiB intercept that does not scale with T; this
finds it.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_train_ledger.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import torch

from tilerl.autograd import Adafactor, AdamW, RecordingBackend, Tape
from tilerl.config import qwen38_27b
from tilerl.model import add_lora, drop_quantized, load_hf
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
    # labelled by owning param key: shape alone cannot tell embed_tokens from lm_head
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


def backward_peaks(tape, backend, g, needs, top: int = 8) -> None:
    # the backward peaks 14.2 GiB above the forward's 2.7 GiB of activations at B=1 T=64;
    # this says which handler's transient allocation makes that gap
    from tilerl.autograd import _BWD

    entries = list(tape._entries)
    grads = {id(entries[-1].output): g}
    produced = {id(e.output) for e in entries}
    worst: dict[str, float] = {}
    for i in range(len(entries) - 1, -1, -1):
        e = entries[i]
        g_out = grads.get(id(e.output))
        if g_out is None:
            continue
        h = _BWD[e.op_name]
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        call = ((backend, g_out, e.args, e.kwargs, lambda t: id(t) in needs or id(t) in produced)
                if getattr(h, "wants", False) else (backend, g_out, e.args, e.kwargs))
        for slot, g_in in h(*call):
            target = e.args[slot] if isinstance(slot, int) else e.kwargs[slot[1]]
            grads[id(target)] = grads.get(id(target), 0) + g_in
        torch.cuda.synchronize()
        add = (torch.cuda.max_memory_allocated() - before) / G
        worst[e.op_name] = max(worst.get(e.op_name, 0.0), add)
    print(f"  {'-- worst backward peak, by op':28s}")
    for name, gb in sorted(worst.items(), key=lambda kv: -kv[1])[:top]:
        print(f"     {gb:7.3f} GiB  {name}")


def tape_bytes(tape, top: int = 8) -> None:
    # the frozen-linear rows are quantized weights held as entry args, not activations
    per: dict[str, list] = {}
    seen = set()
    for e in tape._entries:
        acc = per.setdefault(e.op_name, [0, 0.0])
        acc[0] += 1
        for t in list(e.args) + list(e.kwargs.values()) + [e.output]:
            if not torch.is_tensor(t) or not t.is_cuda:
                continue
            st = t.untyped_storage()
            if st.data_ptr() in seen:
                continue
            seen.add(st.data_ptr())
            acc[1] += st.nbytes() / G
    rows = sorted(per.items(), key=lambda kv: -kv[1][1])
    print(f"  {'-- tape holds':28s} total {sum(v[1] for _, v in rows):7.2f} GiB")
    for name, (n, gb) in rows[:top]:
        print(f"     {gb:7.3f} GiB  {n:5d} entries  {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("-b", type=int, default=1)
    ap.add_argument("-t", type=int, default=64)
    ap.add_argument("--peaks", action="store_true",
                    help="attribute the backward peak to the op that causes it")
    ap.add_argument("--full", action="store_true", help="full fine-tuning (bf16 masters)")
    args = ap.parse_args()

    b = get_backend()
    cfg = qwen38_27b()
    p = 0.0
    p = mark("start", p)
    model = load_hf(cfg, args.source, fuse_projections=False, num_layers=args.layers,
                    keep_master=args.full)
    if args.full:
        drop_quantized(model)
    p = mark("load_hf", p)
    model.params = b.materialize(model.params)
    p = mark("materialize", p)
    trainable = None if args.full else add_lora(model, rank=16)
    p = mark("add_lora", p)
    opt = Adafactor(lr=1e-2) if args.full else AdamW(lr=1e-3)

    ids = np.arange(1, args.b * args.t + 1, dtype=np.int64).reshape(args.b, args.t) \
        % cfg.vocab_size
    positions = np.arange(args.t, dtype=np.int64)
    kv = _training_kv(model, args.b, args.t, device=b.device)
    p = mark("training kv", p)
    live_tensors(named=model.params)

    tape = Tape()
    with torch.no_grad(), tape:
        logits = model.forward(ids, positions, kv, RecordingBackend(b))
    p = mark(f"forward (tape {len(tape._entries)})", p)
    tape_bytes(tape)
    live_tensors(named=model.params)

    grad_logits = torch.ones_like(logits) / logits.numel()
    params = model.params if trainable is None else trainable
    by_id = {id(x): x for x in params.values()}
    if args.full:
        # streamed update: each gradient is freed inside backward, so they never coexist
        opt.begin()
        tape.backward(grad_logits, needs=set(by_id),
                      on_grad=lambda t, g: (t in by_id and opt.step_one(by_id[t], g)) or True)
        p = mark("backward + streamed update", p)
        live_tensors(named=model.params)
        return
    if args.peaks:
        backward_peaks(tape, b, grad_logits, set(by_id))
        return
    grads = tape.backward(grad_logits, needs=set(by_id))
    p = mark("backward", p)
    live_tensors(named=model.params)

    grads = {k: v for k, v in grads.items() if k in by_id}
    p = mark("grads filtered", p)
    opt.step(params.values(), grads)
    mark("optimizer", p)


if __name__ == "__main__":
    main()
