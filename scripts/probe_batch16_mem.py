"""Peak device memory for a GRPO rollout at B=8 and B=16, against the card's 96 GB.

The bandwidth side says B=16 is nearly free -- only the per-batch terms double, +5.6% of
per-tick bytes for 2x the tokens (errors/2026-09-08-four-rulers-the-wrong-size-for-their-object.md).
Capacity is the term that table does not compute, and it is the only thing between the
dispatch finding (wgmma's M granularity is 16, so B=8 fills the tensor core 50% and B=16
fills it 100%) and a default flip. So: build the training engine at each B, submit a full
group, drive it to the generation cap, read the peak.

Rollout only, not a GRPO step: the tape, the gradients and the optimizer state are a
separate allocation this does not model, and `probe_train_mem.py` already measures them.
What this answers is whether the KV pool and the decode activations fit at B=16.

    scripts/pod_run.sh b16mem 2 -- python3 scripts/probe_batch16_mem.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import load_hf


def arm(cfg, model, backend, b: int, gen: int, prompt_tokens: int) -> dict:
    """One batch width, from a fresh engine. The pool is sized exactly as `_train_adapters`
    sizes it (cli.py: ctx*B/BLOCK_TOKENS + 8), because a hand-picked pool here would answer
    a question no training run asks."""
    gc.collect()  # the previous arm's engine is cyclic; its pool is not freed by refcount
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 2**30
    ctx = max(prompt_tokens + gen + 64, 1024)
    engine = build_engine(cfg, model, backend, num_slots=b, max_batch=b,
                          num_blocks=-(-ctx // BLOCK_TOKENS) * b + 8,
                          max_total_tokens=max(ctx, 8192),
                          # decode_graph off and no prefix store: the on-policy shape
                          # grpo_loop refuses an engine without (train.py _require_on_policy).
                          decode_graph=False, prefix_store=NoPrefixStore())
    after_build = torch.cuda.memory_allocated() / 2**30
    rng = np.random.default_rng(0)
    prompt = (rng.integers(1, cfg.vocab_size, size=prompt_tokens)).tolist()
    ids = [engine.submit(prompt, SamplingParams(max_new_tokens=gen, seed=i)) for i in range(b)]
    t0 = time.perf_counter()
    done: dict[int, list[int]] = {}
    ticks = 0
    # +64 ticks of headroom over the cap, so a stop token or a scheduler hold is visible as
    # a short arm rather than a hang. gen*2 would mask a request that never got admitted.
    peak_used = 0
    while ticks < gen + 64 and not all(i in done for i in ids):
        engine.step()
        done.update(engine.poll())
        ticks += 1
        # Sampled mid-drain, not after: a request releases its blocks on finish, so a
        # stats() read past the loop reports 0 and says nothing about what the pool held.
        # Every 64 ticks -- a per-tick stats() takes the step lock and would time the probe.
        if ticks % 64 == 0:
            peak_used = max(peak_used, engine.stats()["pool_used_blocks"])
    secs = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    st = engine.stats()
    lens = [len(done[i]) for i in ids if i in done]
    return {"B": b, "gen": gen, "ctx": ctx, "config": engine.config,
            "base_gib": round(base, 2), "after_build_gib": round(after_build, 2),
            "peak_gib": round(peak, 2), "reserved_gib": round(reserved, 2),
            "pool_blocks": st["blocks_total"], "pool_used": peak_used,
            "finished": len(lens), "min_len": min(lens, default=0),
            "max_len": max(lens, default=0), "ticks": ticks, "secs": round(secs, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=os.environ.get("TILERL_QWEN38_SOURCE", ""))
    ap.add_argument("--layers", type=int, default=0, help="0 = the whole model")
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--batches", default="8,16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    backend = get_backend()
    if backend.device.type != "cuda":
        raise SystemExit("this probe measures device memory; run it on a card")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if a.source:
        from tilerl.config import qwen38_27b
        cfg = qwen38_27b()
        model = load_hf(cfg, a.source, fuse_projections=False,
                        num_layers=a.layers or cfg.num_layers)
        model.params = backend.materialize(model.params)
    else:
        raise SystemExit("--source or TILERL_QWEN38_SOURCE: this is a 27B measurement")
    print(f"# card {torch.cuda.get_device_name(0)}, {total:.1f} GiB total")
    print(f"# {'B':>3} {'pool':>6} {'used':>6} {'peak GiB':>9} {'resv GiB':>9} "
          f"{'free GiB':>9} {'fin':>4} {'len':>5} {'secs':>7}")
    rows = []
    for b in (int(x) for x in a.batches.split(",")):
        try:
            r = arm(cfg, model, backend, b, a.gen, a.prompt_tokens)
        except torch.cuda.OutOfMemoryError as exc:
            # An OOM is a result, not a crash: it is the answer to "does B=16 fit".
            rows.append({"B": b, "oom": str(exc)[:200]})
            print(f"  {b:3d} OOM: {str(exc)[:120]}")
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(f"  {b:3d} {r['pool_blocks']:6d} {r['pool_used']:6d} {r['peak_gib']:9.2f} "
              f"{r['reserved_gib']:9.2f} {total - r['reserved_gib']:9.2f} "
              f"{r['finished']:4d} {r['min_len']:5d} {r['secs']:7.1f}")
    ok = [r for r in rows if "peak_gib" in r]
    for r in ok:
        # A short arm measures a rollout that stopped early, and its peak is then the peak of
        # a shorter sequence: reported, because a fitting-but-truncated arm is not a fit.
        if r["min_len"] < r["gen"]:
            print(f"# B={r['B']}: shortest completion {r['min_len']} of {r['gen']} asked -- "
                  f"the peak is for that length, not the cap")
        if r["finished"] != r["B"]:
            print(f"# B={r['B']}: {r['finished']} of {r['B']} finished in {r['ticks']} ticks; "
                  f"the pool held the rest and this peak is not the full batch's")
    if len(ok) == 2:
        lo, hi = ok
        print(f"# B={hi['B']}/B={lo['B']} peak {hi['peak_gib'] / lo['peak_gib']:.3f}x "
              f"({hi['peak_gib'] - lo['peak_gib']:+.2f} GiB), "
              f"reserved {hi['reserved_gib'] - lo['reserved_gib']:+.2f} GiB, "
              f"{total - hi['reserved_gib']:.1f} GiB left at B={hi['B']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"total_gib": round(total, 2), "rows": rows}, f, indent=2)
        print(f"# wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
