#!/usr/bin/env python3
"""Every _pad2d call on a real 27B step, bucketed by call site AND by M.

The 14.77% self time in the 09-03 profile was read as "the pad is recomputed on
every linear in every layer", which assumed the padded tensor is the WEIGHT.
Arithmetic over the 27B dims says otherwise -- every weight pad is a no-op
because K divides 512 and N divides 64 -- and predicts the real cost is the
M=8 -> Mp=16 activation row pad. This measures it instead of predicting it.

Two buckets matter and the entry conflated them:
  * real vs no-op: a _pad2d that returns its argument costs nothing.
  * M: rollout decode runs M=8, the backward runs the padded sequence batch.
    At M=256 every g2 row pad is already a no-op, so if the cost is in the
    backward the whole reading inverts.

Output: one row per (call site, M, real/no-op) with count and elements copied.
"""
import argparse
import collections
import json
import os
import sys
import traceback

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import tilerl_kernels.backend as B  # noqa: E402

HIST = collections.Counter()
ELEMS = collections.Counter()


def _site():
    """The tilerl frame that called _pad2d, as file:line in a named function."""
    for fr in traceback.extract_stack()[-4::-1]:
        if "tilerl" in fr.filename and "probe_pad" not in fr.filename:
            return f"{os.path.basename(fr.filename)}:{fr.name}"
    return "?"


def install():
    orig = B._pad2d

    def patched(t, rows, cols):
        real = not (rows == t.shape[0] and cols == t.shape[1])
        key = (_site(), int(t.shape[0]), "real" if real else "noop")
        HIST[key] += 1
        if real:
            ELEMS[key] += rows * cols
        return orig(t, rows, cols)

    B._pad2d = patched
    # backend.py imported _pad2d into its module namespace at def time; the
    # module-level rebind above is what its call sites resolve, since they
    # reference the global by name. Verified by the counts being non-zero.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen38-27b")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--out", default="/work/pad_histogram.json")
    args = p.parse_args()

    install()
    from tilerl_kernels.backend import get_backend

    from tilerl.autograd import AdamW
    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.model import add_lora
    from tilerl.train import grpo_loop

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=0, keep_master=False)
    engine = build_engine(cfg, model, backend, num_blocks=args.blocks,
                          num_slots=args.group, max_batch=args.group,
                          max_total_tokens=args.blocks * 16,
                          decode_graph=False, prefix_store=NoPrefixStore())
    # After build_engine: it materializes the params the adapter must point at.
    trainable = add_lora(model, rank=args.rank)
    if args.model == "qwen38-27b":
        from tilerl.cli import _qwen38_tokenizer
        tok = _qwen38_tokenizer()
    else:
        from tilerl.tokenizer import get_tokenizer
        tok = get_tokenizer(None)
    prompts = [tok.encode("What is 17 times 23?")]

    HIST.clear(); ELEMS.clear()
    for row in grpo_loop(engine, model, prompts, lambda p, c: float(len(c) > 0), 1,
                         backend, AdamW(lr=1e-4), group=args.group,
                         sampling=SamplingParams(max_new_tokens=args.gen),
                         trainable=trainable, micro=1):
        print(f"one step: {row[2]:.2f}s", flush=True)

    rows = [dict(site=s, M=m, kind=k, calls=n, elems=int(ELEMS[(s, m, k)]))
            for (s, m, k), n in sorted(HIST.items(), key=lambda kv: -kv[1])]
    tot_real = sum(r["calls"] for r in rows if r["kind"] == "real")
    tot_noop = sum(r["calls"] for r in rows if r["kind"] == "noop")
    summary = dict(real_calls=tot_real, noop_calls=tot_noop,
                   real_elems=sum(r["elems"] for r in rows),
                   by_M=dict(collections.Counter(
                       {f"M={r['M']}": r["calls"] for r in rows if r["kind"] == "real"})),
                   rows=rows)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"real {tot_real} calls, no-op {tot_noop} calls, "
          f"{summary['real_elems']/1e6:.1f} M elems copied", flush=True)
    for r in rows[:25]:
        print(f"  {r['kind']:4s} M={r['M']:<5d} {r['calls']:6d} calls "
              f"{r['elems']/1e6:9.2f} M elems  {r['site']}", flush=True)


if __name__ == "__main__":
    main()
