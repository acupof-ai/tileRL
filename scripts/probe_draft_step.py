"""Where do the ~11 ms of a one-layer draft step go? Times each stage of DraftHead.forward.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_draft_step.py /data00/Qwen3.8-27B-NVFP4 \
        --draft /data00/Qwen3.8-27B-NVFP4/model_mtp.safetensors --batches 1,8
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen38_27b
from tilerl.engine import BatchKv, SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl.spec import load_draft

REPS = 20


def timed(fn, reps=REPS):
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
    cfg = replace(
        base, num_layers=args.layers,
        full_attn_layers=tuple(i for i in base.full_attn_layers if i < args.layers),
    )
    model = load_hf(cfg, args.source)
    draft = load_draft(model, args.draft)
    engine = build_engine(cfg, model, backend, num_blocks=1024, num_slots=16,
                          decode_graph=True, draft=draft, spec_depth=2)
    draft = engine._draft  # build_engine re-serves the head's weights in place

    gen = torch.Generator().manual_seed(7)
    batches = [int(x) for x in args.batches.split(",")]
    for _ in range(max(batches)):
        engine.submit(torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist(),
                      SamplingParams(temperature=0.0, max_new_tokens=4096))
    for _ in range(8):
        engine.step()
    for B in batches:
        rows = list(engine._running)[:B]
        h = torch.cat([r.hidden for r in rows], dim=0)
        dev = backend.device
        ones = torch.ones(B, dtype=torch.long, device=dev)
        kv = BatchKv(
            block_table=torch.arange(B, dtype=torch.long, device=dev).reshape(B, 1),
            seq_len=ones, state_slot=torch.zeros_like(ones), kv_pool=engine._draft_kv,
            state_pool=None, seq_q_lens=ones,
        )
        ids = np.array([[r.output[-1]] for r in rows], dtype=np.int64)
        pos = np.array([[r.seq_len - 1] for r in rows], dtype=np.int64)
        eps = draft.cfg.rms_eps
        L = draft.layers

        # Stages, in the order DraftHead.forward runs them.
        t_ids = torch.as_tensor(ids, dtype=torch.long, device=dev)
        t_pos = torch.as_tensor(pos, dtype=torch.long, device=dev)
        e0 = backend.embedding(t_ids, model.params["embed_tokens"])
        e1 = backend.rmsnorm(e0, draft.params["pre_fc_norm_embedding"], eps)
        hn = backend.rmsnorm(h, draft.params["pre_fc_norm_hidden"], eps)
        cat = torch.cat([e1, hn], dim=-1)
        x0 = L._linear(backend, cat, "fc")
        x1 = L._full_attn(0, x0, t_pos, kv, backend)
        x2 = L._mlp(0, x1, kv, backend)
        x3 = backend.rmsnorm(x2, draft.params["norm"], eps)
        head = "embed_tokens" if cfg.tie_word_embeddings else "lm_head"
        lg = model._linear(backend, x3, head)

        stages = [
            ("h2d ids/pos", lambda: (torch.as_tensor(ids, dtype=torch.long, device=dev),
                                     torch.as_tensor(pos, dtype=torch.long, device=dev))),
            ("embedding", lambda: backend.embedding(t_ids, model.params["embed_tokens"])),
            ("norm embed", lambda: backend.rmsnorm(e0, draft.params["pre_fc_norm_embedding"], eps)),
            ("norm hidden", lambda: backend.rmsnorm(h, draft.params["pre_fc_norm_hidden"], eps)),
            ("cat", lambda: torch.cat([e1, hn], dim=-1)),
            ("fc", lambda: L._linear(backend, cat, "fc")),
            ("full_attn", lambda: L._full_attn(0, x0, t_pos, kv, backend)),
            ("mlp", lambda: L._mlp(0, x1, kv, backend)),
            ("final norm", lambda: backend.rmsnorm(x2, draft.params["norm"], eps)),
            ("lm_head", lambda: model._linear(backend, x3, head)),
            ("greedy", lambda: backend.greedy(lg)),
            ("greedy .tolist()", lambda: backend.greedy(lg)[0][:, -1].tolist()),
        ]
        print(f"\n=== draft step stages, B={B} ({args.layers} layers) ===")
        tot = 0.0
        for name, fn in stages:
            ms = timed(fn)
            if name != "greedy .tolist()":
                tot += ms
            print(f"  {name:>18} {ms:8.3f} ms")
        print(f"  {'sum of stages':>18} {tot:8.3f} ms")
        whole = timed(lambda: draft.forward(h, ids, pos, kv, backend, hidden_out=[]))
        print(f"  {'DraftHead.forward':>18} {whole:8.3f} ms")
        for d in (1, 2, 4):
            engine._spec_depth = d
            ch = timed(lambda: engine._draft_chains(rows, trim=False), reps=5)
            chains = engine._draft_chains(rows, trim=False)
            w = max(map(len, chains))
            for c in chains:
                c.extend([c[-1]] * (w - len(c)))
            gr = timed(lambda: engine._run_decode_graph(rows, chains), reps=5)
            print(f"  depth {d}: _draft_chains {ch:7.3f} ms  "
                  f"({ch / d:6.3f}/step)   verify replay w={w} {gr:7.3f} ms")
        engine._spec_depth = 2
        tick = timed(lambda: engine.step(), reps=5)
        print(f"  {'engine.step()':>18} {tick:8.3f} ms")


if __name__ == "__main__":
    main()
