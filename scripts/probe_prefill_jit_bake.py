"""Bake the sparse prefill/draft/verify TileLang JIT kernels into the on-disk
shared cache by driving REAL requests against an already-running server.

This is an out-of-band ops step, not engine startup warmup: first-use compile
is paid ONCE per kernel change and persisted in TILELANG_CACHE_DIR, so a later
fresh tree / restart serves with 0 compiles (measured: a new process reusing a
populated cache compiled nothing and took first-prompt prefill 103s -> 1.8s).

It covers, per concurrent batch size B, every prefill query bucket. The chunk
budget is 512 and query widths bucket to multiples of 64; an arbitrary prompt
length lands its block-trimmed tail in any of S=64..512, so prompts of
512*k + tail are sent for tails stepping across every bucket, with the draft
on and a few emitted tokens so the d1 verify widths (S=1/2) and the draft's
two-chunk back-fill span compile too.

Run ONCE on the serve host after deploy (dev box: the full cold bake is ~20
min; it only re-runs after a kernel change):

    python scripts/probe_prefill_jit_bake.py --base http://127.0.0.1:8000 \
        --batches 1,2,4 --emit 8 \
        --cache-dir ~/.tilelang_cache/0.1.13/cuda-binaries

--cache-dir (same host) prints the cubin count before/after; the bake is done
when a second pass over FRESH lengths adds zero cubins.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

# Tails (in CHARACTERS) chosen so the tokenized tail sweeps every 64-token
# prefill bucket after block-boundary trimming. Step 32 (well under 64) so no
# bucket is skipped regardless of the tokenizer's char/token ratio; offsets off
# the 16-token page grid so draft/trim variants are exercised too.
TAILS = list(range(0, 512 + 1, 32))
# Half-bucket offsets: pass 2 drives prompt lengths NOT seen in pass 1. They land
# in the SAME buckets, so a complete bake adds zero cubins; if any lit up, the
# sweep in pass 1 missed a bucket and the bake is incomplete.
VERIFY_TAILS = list(range(16, 512 + 1, 32))
FULL_CHUNKS = 10  # 5120 tokens of full chunks before the swept tail (~5.2k)


def _prompt(tail: int) -> str:
    # Digits tokenize close to 1 char/token on this model; long run of full
    # chunks plus the swept tail. A fixed prefix phrase keeps it a real turn.
    body = ("1234567890" * (FULL_CHUNKS * 51)) + ("7" * tail)
    return "请总结下面这串数字的长度，不要展开：" + body


def _post(base: str, prompt: str, emit: int) -> None:
    body = {
        "model": "qwen38-27b",
        "temperature": 0,
        "max_tokens": emit,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        json.load(r)


def _cubins(cache_dir: str | None) -> int | None:
    if not cache_dir:
        return None
    try:
        return len([f for f in os.listdir(cache_dir) if f.endswith(".cubin")])
    except FileNotFoundError:
        return 0


def _bake_ok(verify_added: int | None) -> bool:
    """Gate: without --cache-dir there is nothing to count (OK); with it, even one
    cubin added by the fresh-length verify sweep means a prefill bucket was
    missed and the disk cache is incomplete."""
    return verify_added in (None, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--batches", default="1,2,4", help="concurrent batch sizes to cover, comma list"
    )
    ap.add_argument("--emit", type=int, default=8, help="tokens per request (d1)")
    ap.add_argument("--cache-dir", default="", help="cubin dir to count (same host)")
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",") if x]

    before = _cubins(args.cache_dir)
    t0 = time.perf_counter()
    total = 0
    sweeps = (("bake", TAILS), ("verify-fresh-lengths", VERIFY_TAILS))
    verify_added = None
    for label, tails in sweeps:
        sweep0 = _cubins(args.cache_dir)
        for b in batches:
            for tail in tails:
                prompt = _prompt(tail)
                with cf.ThreadPoolExecutor(max_workers=b) as ex:
                    list(ex.map(lambda _: _post(args.base, prompt, args.emit), range(b)))
                total += b
        now = _cubins(args.cache_dir)
        added = None if now is None or sweep0 is None else now - sweep0
        total_added = None if now is None or before is None else now - before
        # The gate is THIS verify sweep's delta (fresh lengths vs the cache the
        # bake sweep left), not since-start: the bake sweep legitimately adds the
        # shapes it exists to bake; only a verify increment means a missed bucket.
        if label.startswith("verify"):
            verify_added = added
        print(
            f"{label}: {total} requests, cubins={now} "
            f"(+{added} this sweep, +{total_added} since start)",
            flush=True,
        )
    print(f"bake done: {total} requests in {time.perf_counter() - t0:.0f}s", flush=True)
    # Hard gate: with --cache-dir, the fresh-length verify sweep must compile
    # NOTHING. A nonzero count means the bake sweep missed a prefill bucket (the
    # half-bucket offsets land in every bucket), so the cache is incomplete.
    if args.cache_dir and not _bake_ok(verify_added):
        print(
            f"BAKE INCOMPLETE: verify over fresh prompt lengths added "
            f"{verify_added} cubin(s); the bake sweep did not cover every shape.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
