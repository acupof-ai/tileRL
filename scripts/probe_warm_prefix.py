#!/usr/bin/env python3
"""Card gate for warm spec prefix adoption (#564).

Builds ONE sparse+spec engine, publishes a long prefix from a publisher, then
runs TWO followers of that prefix in one B=8 wave:

  warm followers adopt the published draft prefix (skip its prefill);
  cold controls are built per-wave with an empty prefix store (full prefill).

Asserts warm and cold produce identical tokens over --steps generated tokens,
prints first-turn (prompt + first generated token) wall time warm vs cold and
the kv_prefix ledger row.

    TILERL_TARGET=cuda python3 scripts/card_warm_spec_gate.py \
        --source /work/Qwen3.8-27B-NVFP4 --draft /work/Qwen3.8-27B-DFlash2 \
        --steps 64 --batch 8
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine
from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf

PROMPT_PAGES = 24
PAGE = 16
PREFIX_TOKENS = PROMPT_PAGES * PAGE  # 384
TAIL_TOKENS = 20


def _ids(n: int, seed: int, vocab: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(100, min(vocab, 32000), n, dtype=np.int64)


def _params(steps: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_new_tokens=steps, seed=0)


def _drain(engine, rid: int, n: int):
    out = []
    for _ in range(n * 4):
        d = engine.poll()
        if rid in d:
            out.extend(d[rid])
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
    p.add_argument("--out", default="/work/warm_spec_gate.json")
    args = p.parse_args()

    from tilerl.spec import load_draft

    cfg = qwen38_27b()
    model = load_hf(cfg, args.source)
    backend = get_backend()
    draft = load_draft(model, args.draft)

    def make_engine(prefix: bool):
        return build_engine(
            cfg, model, backend,
            num_blocks=args.num_blocks, num_slots=args.batch + 2,
            max_batch=args.batch + 2, max_total_tokens=8192,
            max_num_batched_tokens=512, sparse_k=args.sparse_k, scorer="bounds",
            kv_cold_bytes=1 << 30, draft=draft, spec_depth=1,
            prefix_store=NoPrefixStore() if not prefix else None)

    prompt = _ids(PREFIX_TOKENS, 7, cfg.vocab_size).tolist()

    # publish: one request generates enough that the full prefix drops/publishes
    warm = make_engine(prefix=True)
    pub = warm.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=256, seed=0))
    _drain(warm, pub, 256)
    entry = warm._sparse.prefix.lookup(prompt)
    assert entry is not None and len(entry["keys"]) == PROMPT_PAGES, \
        "publisher did not publish the full prefix"
    assert warm.stats()["prefix_warm_adoptions"] == 0

    followers = [prompt + _ids(TAIL_TOKENS, 100 + i, cfg.vocab_size).tolist()
                 for i in range(args.batch)]

    # WARM wave (two+ real followers share ticks; here all B adopt the one prefix)
    t0 = time.perf_counter()
    wrids = [warm.submit(f, _params(args.steps)) for f in followers]
    warm.step()  # adoption + tail prefill tick
    matched = [next(x for x in warm._running if x.req_id == r).sparse_matched
               for r in wrids]
    warm_tokens = {r: [] for r in wrids}
    for _ in range(args.steps * 6):
        d = warm.poll()
        done = 0
        for r in wrids:
            warm_tokens[r].extend(d.get(r, ()))
            if len(warm_tokens[r]) >= args.steps:
                done += 1
        if done == len(wrids):
            break
        warm.step()
    warm_ms = (time.perf_counter() - t0) * 1000
    warm_stats = warm.stats()
    assert warm_stats["prefix_warm_adoptions"] == args.batch, warm_stats
    assert all(m == PREFIX_TOKENS for m in matched), matched

    # COLD controls: fresh engine, empty prefix store, same followers
    cold = make_engine(prefix=False)
    t0 = time.perf_counter()
    crids = [cold.submit(f, _params(args.steps)) for f in followers]
    cold_tokens = {r: [] for r in crids}
    for _ in range(args.steps * 6):
        d = cold.poll()
        done = 0
        for r in crids:
            cold_tokens[r].extend(d.get(r, ()))
            if len(cold_tokens[r]) >= args.steps:
                done += 1
        if done == len(crids):
            break
        cold.step()
    cold_ms = (time.perf_counter() - t0) * 1000

    # token equality, per follower
    eq, first_diff = [], {}
    for wr, cr in zip(wrids, crids):
        w = warm_tokens[wr][:args.steps]
        c = cold_tokens[cr][:args.steps]
        same = w == c
        eq.append(same)
        if not same:
            first_diff[wr] = next(i for i in range(min(len(w), len(c))) if w[i] != c[i])
    kv_prefix = [r for r in warm_stats["memory"] if r["owner"] == "kv_prefix"]

    result = {
        "steps": args.steps, "batch": args.batch, "sparse_k": args.sparse_k,
        "all_token_equal": all(eq),
        "n_followers_equal": sum(eq),
        "first_diff_position": first_diff,
        "warm_wave_ms": round(warm_ms, 1),
        "cold_wave_ms": round(cold_ms, 1),
        "speedup": round(cold_ms / warm_ms, 3),
        "prefix_warm_adoptions": warm_stats["prefix_warm_adoptions"],
        "kv_prefix_row": ({"measured": kv_prefix[0]["measured"],
                           "delta": kv_prefix[0]["delta"]} if kv_prefix else None),
        "warm_prefix_hits": warm_stats["prefix_hits"],
        "cold_prefix_hits": cold.stats()["prefix_hits"],
    }
    print("WARM_SPEC_GATE_RESULT " + json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    warm.shutdown(); cold.shutdown()
    if not result["all_token_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
