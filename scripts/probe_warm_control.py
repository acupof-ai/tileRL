#!/usr/bin/env python3
"""Controls for the warm-vs-cold card divergence (#564).

(a) cold-vs-cold: the same 8 followers run in TWO separate cold B=8 waves.
    Unequal tokens => a B=8 sparse spec wave is not reproducible on the card.
(b) warm-vs-cold at B=1: each follower runs alone, with first-generated-token
    logits captured per request; prints token equality and logits max_abs.

If (a) is equal and (b) is unequal, the warm restore differs on device.
If (b) is equal at B=1 but the B=8 gate diverges, the difference is the
sm90 B>1 sparse packed-prefill path, not warm adoption.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import load_hf

PROMPT_PAGES = 24
PREFIX_TOKENS = PROMPT_PAGES * BLOCK_TOKENS
TAIL_TOKENS = 20


def _ids(n: int, seed: int, vocab: int) -> list:
    return np.random.default_rng(seed).integers(100, min(vocab, 32000), n).tolist()


def _drain(engine, rid: int, n: int):
    out = []
    for _ in range(n * 6):
        d = engine.poll()
        out.extend(d.get(rid, ()))
        if len(out) >= n:
            break
        engine.step()
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sparse-k", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=512)
    p.add_argument("--out", default="/work/warm_control.json")
    args = p.parse_args()

    from tilerl.spec import load_draft

    cfg = qwen38_27b()
    model = load_hf(cfg, args.source)
    backend = get_backend()
    draft = load_draft(model, args.draft)

    def make_engine(prefix: bool):
        eng = build_engine(
            cfg, model, backend,
            num_blocks=args.num_blocks, num_slots=args.batch + 2,
            max_batch=args.batch + 2, max_total_tokens=8192,
            max_num_batched_tokens=512, sparse_k=args.sparse_k, scorer="bounds",
            kv_cold_bytes=1 << 30, draft=draft, spec_depth=1,
            prefix_store=NoPrefixStore() if not prefix else None)
        # capture the first logits vector committed per request
        first_logits: dict[int, object] = {}
        orig = eng._sample_commit

        def wrap(rows):
            for r, lg, _pos in rows:
                first_logits.setdefault(r.req_id, lg.detach().float().cpu())
            return orig(rows)

        eng._sample_commit = wrap
        eng._first_logits = first_logits
        return eng

    prompt = _ids(PREFIX_TOKENS, 7, cfg.vocab_size)
    followers = [prompt + _ids(TAIL_TOKENS, 100 + i, cfg.vocab_size)
                 for i in range(args.batch)]
    params = lambda: SamplingParams(temperature=0.0, max_new_tokens=args.steps, seed=0)

    warm = make_engine(prefix=True)
    pub = warm.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=256, seed=0))
    _drain(warm, pub, 256)
    entry = warm._sparse.prefix.lookup(prompt)
    assert entry is not None and len(entry["keys"]) == PROMPT_PAGES

    # ---- (a) cold A vs cold B, B=8 waves ----
    cold_a = make_engine(prefix=False)
    ra = [cold_a.submit(f, params()) for f in followers]
    ta = {r: [] for r in ra}
    for _ in range(args.steps * 6):
        d = cold_a.poll()
        for r in ra:
            ta[r].extend(d.get(r, ()))
        if all(len(ta[r]) >= args.steps for r in ra):
            break
        cold_a.step()

    cold_b = make_engine(prefix=False)
    rb = [cold_b.submit(f, params()) for f in followers]
    tb = {r: [] for r in rb}
    for _ in range(args.steps * 6):
        d = cold_b.poll()
        done = 0
        for r in rb:
            tb[r].extend(d.get(r, ()))
            if len(tb[r]) >= args.steps:
                done += 1
        if done == len(rb):
            break
        cold_b.step()

    cold_equal = []
    for a, b in zip(ra, rb):
        cold_equal.append(ta[a][:args.steps] == tb[b][:args.steps])

    # ---- (b) warm vs cold at B=1 (sequential followers) ----
    b1 = []
    for i, f in enumerate(followers):
        c1 = make_engine(prefix=False)
        crid = c1.submit(f, params())
        ct = _drain(c1, crid, args.steps)
        clg = c1._first_logits.get(crid)
        wrid = warm.submit(f, params())
        wt = _drain(warm, wrid, args.steps)
        wlg = warm._first_logits.get(wrid)
        max_abs = None
        if wlg is not None and clg is not None:
            n = min(wlg.numel(), clg.numel())
            max_abs = float((wlg[:n] - clg[:n]).abs().max())
        b1.append({"row": i, "tokens_equal": wt == ct,
                   "first_diff": (next((j for j in range(min(len(wt), len(ct)))
                                       if wt[j] != ct[j]), None)),
                   "first_logits_max_abs": max_abs})
        c1.shutdown()

    result = {
        "steps": args.steps, "batch": args.batch, "sparse_k": args.sparse_k,
        "cold_vs_cold_b8_n_equal": int(sum(cold_equal)),
        "cold_vs_cold_b8": [bool(x) for x in cold_equal],
        "b1_warm_vs_cold": b1,
        "b1_n_equal": sum(x["tokens_equal"] for x in b1),
    }
    print("WARM_CONTROL_RESULT " + json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    warm.shutdown()
    cold_a.shutdown()
    cold_b.shutdown()


if __name__ == "__main__":
    main()
