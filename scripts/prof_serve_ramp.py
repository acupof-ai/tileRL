"""What still ramps after precapture? Attribute it, don't guess.

With all 8 decode graphs captured at startup, served throughput still climbed
13.6 -> 21.2 -> 24.3 -> 25.5 tok/s over four requests. The serve log shows only
3 kernel compiles and all of them inside precapture, so per-request JIT is out.

Two candidates left, and this separates them by reading torch's own counters
rather than inferring from wall clock:

  * **First use of the NON-decode path.** precapture builds decode graphs only.
    A request's prefill runs kernels precapture never touched, and with a warm
    TILELANG_CACHE_DIR each still costs a cache load (~0.2 s) on first use in
    this process. Shows up as a big request-1 penalty that never returns.
  * **The caching allocator.** cudaMalloc is synchronous; the allocator only
    holds blocks after something has asked for them. Shows up as `reserved`
    growing and `num_alloc_retries` incrementing over the first requests.

Prints per request: ms/token, reserved MiB, and the cudaMalloc-ish counters.
Then runs one throwaway generation and repeats, so the fix is measured, not
assumed.
"""

from __future__ import annotations

import argparse
import os
import time


def _mem() -> dict:
    import torch

    s = torch.cuda.memory_stats()
    return {
        "reserved": s.get("reserved_bytes.all.current", 0) / 2**20,
        "retries": s.get("num_alloc_retries", 0),
        "allocs": s.get("segment.all.allocated", 0),
        "frees": s.get("num_device_free", 0),
    }


def decode_only(e, ids, n, mod):
    """ms/token over DECODE ticks only, with the prefill burned first."""
    import torch

    from tilerl.engine import SamplingParams

    rid = e.submit(ids, SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=n, seed=0))
    req = None
    while req is None or req.phase != mod._PHASE_DECODE:
        e.step()
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is None:
            raise SystemExit("finished during prefill — raise n or shorten the prompt")
    torch.cuda.synchronize()
    s0, t0 = e.stats(), time.perf_counter()
    out = None
    while out is None:
        e.step()
        out = e.take(rid)
    torch.cuda.synchronize()
    wall, s1 = time.perf_counter() - t0, e.stats()
    tok = s1["tokens_generated"] - s0["tokens_generated"]
    return wall * 1000 / max(tok, 1), tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--reqs", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    from tilerl_kernels.backend import get_backend

    from tilerl import cli
    from tilerl import engine as mod
    from tilerl.spec import load_draft

    cli._QWEN38_SOURCE = args.source
    backend = get_backend()
    cfg, model = cli._build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None
    e = cli._build_engine(cfg, model, backend, draft=draft, depth=args.depth,
                          slots=4, blocks=2048, max_ctx=4096)

    t0 = time.perf_counter()
    n_graphs = e.precapture()
    print(f"precapture: {n_graphs} graphs in {time.perf_counter() - t0:.0f}s")
    base = _mem()
    print(f"{'phase':<10} {'req':>4} {'ms/tok':>8} {'tok/s':>7} {'resvMiB':>9} "
          f"{'d resv':>8} {'retries':>8} {'segs':>6}")

    def sweep(tag: str, start: int):
        prev = _mem()["reserved"]
        for i in range(args.reqs):
            ids = list(range(1000 * (start + i + 1), 1000 * (start + i + 1) + args.ctx))
            ms, _ = decode_only(e, ids, args.tokens, mod)
            m = _mem()
            print(f"{tag:<10} {i:>4} {ms:>8.1f} {1000 / ms:>7.1f} {m['reserved']:>9.0f} "
                  f"{m['reserved'] - prev:>8.0f} {m['retries']:>8} {m['allocs']:>6}")
            prev = m["reserved"]

    sweep("cold", 0)
    print(f"\n(baseline reserved after precapture: {base['reserved']:.0f} MiB)")
    print("A ramp with flat reserved and 0 retries is first-use of the non-decode\n"
          "path, not the allocator. A ramp WITH reserved growth is the allocator.")


if __name__ == "__main__":
    main()
