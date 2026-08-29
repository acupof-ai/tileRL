"""Is a draft step's 2.06 ms compute, or eager dispatch?

A ONE-layer head measures 0.917 ms in full_attn alone, where the trunk's
SIXTEEN full-attn layers total 0.216 ms inside a captured graph. If that gap is
dispatch, capturing the draft step collapses it and depth-1 speculation turns
net-positive at B=1. Measure before building the engine seam.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_draft_graph.py /data00/Qwen3.8-27B-NVFP4 \
        --draft /data00/Qwen3.8-27B-NVFP4/model_mtp.safetensors
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

from tilerl.config import qwen38_27b
from tilerl.engine import BatchKv, SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl.spec import load_draft
from tilerl_kernels.backend import get_backend


def timed(fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--draft", required=True)
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--batches", default="1,8")
    args = p.parse_args()
    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    base = qwen38_27b()
    cfg = replace(base, num_layers=args.layers,
                  full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers))
    model = load_hf(cfg, args.source)
    engine = build_engine(cfg, model, backend, num_blocks=1024, num_slots=16,
                          decode_graph=True, draft=load_draft(model, args.draft), spec_depth=2)
    draft = engine._draft

    gen = torch.Generator().manual_seed(7)
    batches = [int(x) for x in args.batches.split(",")]
    for _ in range(max(batches)):
        engine.submit(torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist(),
                      SamplingParams(temperature=0.0, max_new_tokens=4096))
    for _ in range(8):
        engine.step()

    for B in batches:
        rows = list(engine._running)[:B]
        dev = backend.device
        h = torch.cat([r.hidden for r in rows], dim=0).contiguous()
        ones = torch.ones(B, dtype=torch.long, device=dev)
        kv = BatchKv(
            block_table=torch.arange(B, dtype=torch.long, device=dev).reshape(B, 1),
            seq_len=ones, state_slot=torch.zeros_like(ones), kv_pool=engine._draft_kv,
            state_pool=None, seq_q_lens=ones,
        )
        ids = torch.tensor([[r.output[-1]] for r in rows], dtype=torch.long, device=dev)
        pos = torch.tensor([[r.seq_len - 1] for r in rows], dtype=torch.long, device=dev)
        eager = timed(lambda: draft.forward(h, ids, pos, kv, backend, hidden_out=[]))

        # Capture: warm on a side stream first, since tilelang JIT is host work.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                draft.forward(h, ids, pos, kv, backend, hidden_out=[])
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                out = draft.forward(h, ids, pos, kv, backend, hidden_out=[])
        except Exception as exc:
            print(f"  B={B}: capture failed: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        graphed = timed(g.replay)
        ref = draft.forward(h, ids, pos, kv, backend, hidden_out=[])
        g.replay()
        rel = (out - ref).abs().max().item() / ref.abs().max().clamp_min(1e-6).item()
        print(f"  B={B}: eager {eager:.3f} ms  graph {graphed:.3f} ms  "
              f"{eager / graphed:.2f}x  rel {rel:.2e}")


if __name__ == "__main__":
    main()
