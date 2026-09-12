#!/usr/bin/env python3
"""Boot-from-store "只算一次" demo (unit C, #514).

Phase 1  build an engine backed by --kv-store DIR, submit one long prompt,
         wait for its prefill, save_boot() the block-aligned context, and
         record the in-process greedy continuation.
Stop     the engine is shut down: nothing survives in HBM or the prefix store.
Phase 2  build a FRESH engine on the same DIR, submit the same prompt: the
         block-aligned prefix bulk-loads from disk (boot_hits=1, only the
         <16-token tail forwards) and the continuation must equal phase 1's.

Prints prefill seconds vs boot seconds. CPU-tested on tiny with a small
context; cc launches the 256k 27B card run:

    python scripts/boot_from_store.py --model qwen38-27b \\
        --kv-store DIR --prompt-file ids.npy --prompt-tokens 262144

A length one token short of a page (T%16==15) is refused: save_boot would
store the first generated token and the later prefix would miss.

--sparse-k PAGES is parsed ahead of the sparse-attention landing: page
selection is not wired into serving attention yet (#497 holds the selector),
so a nonzero value refuses instead of silently booting a dense engine.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np


def _drain_to_decode(engine, rid: int, *, ticks: int = 1_000_000):
    for _ in range(ticks):
        for r in engine._running:
            if r.req_id == rid and r.phase == 2:
                return r
        engine.step()
    raise TimeoutError("request never reached decode")


def _drain_tokens(engine, rid: int, n: int, *, ticks: int = 1_000_000):
    for _ in range(ticks):
        done = engine.poll()
        if rid in done and len(done[rid]) >= n:
            return done[rid][:n]
        engine.step()
    raise TimeoutError("engine did not finish in the tick budget")


def _build_engine(cfg, model, backend, store_dir: str, max_ctx: int):
    from tilerl.engine import build_engine

    return build_engine(
        cfg, model, backend, num_slots=2, max_batch=1,
        max_total_tokens=max_ctx, kv_store=store_dir)


def run(model_name: str, store_dir: str, prompt_tokens: int, new_tokens: int,
        prompt_file: str, sparse_k: int, seed: int) -> dict:
    if sparse_k:
        raise NotImplementedError(
            f"--sparse-k {sparse_k}: page selection is not wired into serving "
            "attention yet (selector in #497); drop --sparse-k for the dense "
            "boot demo rather than booting dense silently.")
    from tilerl_kernels.backend import get_backend

    # save_boot at the first decode tick saves floor((T+1)/16) pages. A prompt
    # with T%16==15 puts the first SAMPLED token into the last saved page, so the
    # later prompt-only prefix lookup misses; the script refuses that length.
    # A page-aligned prompt is fine: the engine re-forwards its last loaded page
    # once to produce the first continuation logits (one prefill forward, same
    # cost as a <16-token tail). 256k (262144) is page-aligned and works directly.
    if prompt_tokens % 16 == 15:
        raise ValueError(
            f"--prompt-tokens {prompt_tokens} is one token short of a page "
            "(T%16==15): save_boot would store the first generated token and "
            "the prefix would miss. Use any other length, e.g. 262144.")

    # Serving builds with fused projections; reuse the exact serve path so the
    # card run exercises what `tilerl serve --kv-store` does.
    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams

    backend = get_backend()
    cfg, model = _build_model(model_name, seed=seed, fuse_projections=True,
                              backend=backend)
    if prompt_file:
        ids = np.asarray(_load_ids(prompt_file), dtype=np.int64)[:prompt_tokens]
    else:
        # Cycled ids exercise the boot mechanics without a tokenizer; a real 27B
        # demo passes --prompt-file with a 256k tokenised prompt.
        ids = (np.arange(7, 7 + prompt_tokens, dtype=np.int64) % cfg.vocab_size)
    params = SamplingParams(temperature=0.0, max_new_tokens=new_tokens, seed=seed)
    max_ctx = prompt_tokens + new_tokens + 64
    Path(store_dir).mkdir(parents=True, exist_ok=True)

    writer = _build_engine(cfg, model, backend, store_dir, max_ctx)
    t0 = time.perf_counter()
    rid = writer.submit(ids, params)
    req = _drain_to_decode(writer, rid)
    prefill_s = time.perf_counter() - t0
    written = writer.save_boot(req)
    expect = _drain_tokens(writer, rid, new_tokens)
    writer.shutdown()

    booter = _build_engine(cfg, model, backend, store_dir, max_ctx)
    t0 = time.perf_counter()
    rid2 = booter.submit(ids, params)
    _drain_to_decode(booter, rid2)
    boot_s = time.perf_counter() - t0
    got = _drain_tokens(booter, rid2, new_tokens)
    st = booter.stats()
    booter.shutdown()

    assert st["boot_hits"] == 1, st
    assert st["prefill_forwards"] == 1, st  # only the <16-token tail forwards
    assert got == expect, f"booted {got} != in-process {expect}"
    return {"prefill_s": prefill_s, "boot_s": boot_s, "bytes_written": written,
            "prompt_tokens": int(len(ids)), "continuation": got}


def _load_ids(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    return np.asarray(json.loads(Path(path).read_text()), dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tiny", choices=["tiny", "qwen38-27b"])
    ap.add_argument("--kv-store", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=83)
    ap.add_argument("--new-tokens", type=int, default=6)
    ap.add_argument("--prompt-file", default="",
                    help=".npy or .json int ids; default cycles ids (tiny mechanics)")
    ap.add_argument("--sparse-k", type=int, default=0)
    ap.add_argument("--fresh-store", action="store_true",
                    help="delete the store dir before phase 1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.fresh_store:
        shutil.rmtree(args.kv_store, ignore_errors=True)
    r = run(args.model, args.kv_store, args.prompt_tokens, args.new_tokens,
            args.prompt_file, args.sparse_k, args.seed)
    print(f"prompt {r['prompt_tokens']} tok, {r['bytes_written']} bytes stored")
    print(f"prefill {r['prefill_s']:.4f} s   boot {r['boot_s']:.4f} s   "
          f"speedup {r['prefill_s'] / max(r['boot_s'], 1e-12):.2f}x")
    print(f"continuation {r['continuation']}")


if __name__ == "__main__":
    main()
