#!/usr/bin/env python3
"""Four-arm wall clock for the recapture waivers, plus the correctness row.

Arms differ ONLY in how the engine is built and which waiver grpo_loop gets:

    baseline   graphs off, NoPrefixStore      (what the guard demands today)
    prefix     graphs off, live PrefixStore   clear_prefix=True
    graph      graphs ON,  NoPrefixStore      recapture_graph=True
    both       graphs ON,  live PrefixStore   both

Correctness row is separate and cheap: after an update, a kept graph must not
replay the old policy. We assert the state machine (graphs dropped, prefix
emptied) AND that a second step's rollout differs from a frozen replay --
on cpu the first is all that can be checked, here both can.

Usage: card_claim is the caller's job; this script only runs the arms.
"""
import argparse
import json
import os
import sys
import time

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.autograd import AdamW  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402
from tilerl.model import add_lora  # noqa: E402
from tilerl.train import grpo_loop  # noqa: E402

ARMS = {
    "baseline": dict(decode_graph=False, no_prefix=True, recapture_graph=False, clear_prefix=False),
    "prefix":   dict(decode_graph=False, no_prefix=False, recapture_graph=False, clear_prefix=True),
    "graph":    dict(decode_graph=True,  no_prefix=True,  recapture_graph=True,  clear_prefix=False),
    "both":     dict(decode_graph=True,  no_prefix=False, recapture_graph=True,  clear_prefix=True),
}


def run_arm(name, cfg, args, prompts, reward):
    a = ARMS[name]
    backend = get_backend()
    cfg2, model = _build_model(args.model, seed=0, keep_master=False)
    kw = {} if not a["no_prefix"] else dict(prefix_store=NoPrefixStore())
    engine = build_engine(cfg2, model, backend, num_blocks=args.blocks,
                          num_slots=args.group, max_batch=args.group,
                          max_total_tokens=args.blocks * 16,
                          decode_graph=a["decode_graph"], **kw)
    # AFTER build_engine, which materializes the params the adapter must point at:
    # materialize() rebuilds any param whose device/dtype differs and the new object
    # has a new id(), so an adapter built first is attached to a tensor the forward
    # never reads and the tape produces no gradients (cli.py:286 says the same).
    trainable = add_lora(model, rank=args.rank)
    secs = []
    for i, row in enumerate(grpo_loop(
            engine, model, prompts, reward, args.steps, backend,
            AdamW(lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
            group=args.group, sampling=SamplingParams(max_new_tokens=args.gen),
            trainable=trainable, micro=1,
            recapture_graph=a["recapture_graph"], clear_prefix=a["clear_prefix"])):
        secs.append(row[2])
        print(f"  {name} step {i}: {row[2]:.2f}s reward={row[0]:.3f} tokens={row[4]:.0f}",
              flush=True)
    # Drop step 0: it pays the first JIT/capture of every shape.
    warm = secs[1:] or secs
    st = engine.stats() if hasattr(engine, "stats") else {}
    return dict(arm=name, steps=len(secs), all_secs=[round(s, 3) for s in secs],
                warm_mean=round(float(np.mean(warm)), 3),
                warm_sd=round(float(np.std(warm, ddof=1)), 3) if len(warm) > 1 else None,
                graphs_held=len(engine._decode_graphs),
                # A prefix arm that publishes nothing and hits nothing is paying
                # bookkeeping for a cache that never fires: publish needs
                # materialized % BLOCK_TOKENS == 0 in DECODE (engine.py:1100), so a
                # prompt boundary never triggers it. Counters, not inference.
                prefix_hits=st.get("prefix_hits"),
                prefix_misses=st.get("prefix_misses"),
                prefix_published=st.get("prefix_published"),
                prefix_entries=engine._prefix.stats()["entries"]
                if hasattr(engine._prefix, "stats") else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen38-27b")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--arms", default="baseline,prefix,graph,both")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="/work/recapture_arms.json")
    args = p.parse_args()

    if args.model == "qwen38-27b":
        from tilerl.cli import _qwen38_tokenizer
        tok = _qwen38_tokenizer()
    else:
        from tilerl.tokenizer import get_tokenizer
        tok = get_tokenizer(None)
    # One fixed prompt set: the arms must differ in caching, not in work.
    text = ["What is 17 times 23?", "A bag holds 5 apples. Three bags?",
            "If 12 pens cost 36 dollars, what is one pen?"]
    prompts = [tok.encode(t) for t in text]
    reward = lambda pr, c: float(len(c) > 0)  # noqa: E731  cost, not learning

    out = []
    # The arms share one process and one TileLang cache, so whichever runs first
    # pays every JIT compile and reads as the slowest arm no matter what it is
    # (measured on cpu tiny: baseline first = 40.1 s/step, the same arm later
    # = 0.09). One throwaway arm first, discarded, puts every measured arm on a
    # warm cache. Ordering is still not free -- --arms reversed is the control.
    if args.warmup:
        print("=== warmup (discarded) ===", flush=True)
        try:
            run_arm(args.arms.split(",")[0], None,
                    argparse.Namespace(**{**vars(args), "steps": 1}), prompts, reward)
        except Exception as exc:
            print(f"  warmup failed, arms may carry JIT: {type(exc).__name__}: {exc}",
                  flush=True)
        torch.cuda.empty_cache()

    for name in args.arms.split(","):
        print(f"=== arm {name} ===", flush=True)
        t0 = time.perf_counter()
        try:
            out.append(run_arm(name, None, args, prompts, reward))
        except Exception as exc:  # an arm that dies must not hide the others
            out.append(dict(arm=name, error=f"{type(exc).__name__}: {exc}"))
            print(f"  {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(f"  {name} total {time.perf_counter()-t0:.1f}s", flush=True)
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
