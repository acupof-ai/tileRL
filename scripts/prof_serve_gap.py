"""Why is a served token slower than a benched one? Same engine, same weights.

Warm serve measured 127.7 ms/token where `bench_ctx_decode` reads 19.7 at the
same depth, and the served rate DEGRADED across four sequential requests
(7.83 -> 5.61 -> 2.10 tok/s) — an accumulation signature, not a fixed overhead.

Two things this measures that the first version of it got wrong:

  * **Decode only.** Timing from submit charges the prompt's prefill into
    ms/token (256 tokens x 7.89 ms = 2020 ms, i.e. 31.6 of a 64.9 ms "token").
    Burn the prefill first, like bench_ctx_decode.measure does.
  * **Distinct prompts.** Re-submitting one prompt hits the prefix cache, so it
    never publishes a GDN snapshot. Serve's requests were all different: each
    miss publishes 144 MiB every BLOCK_TOKENS decoded tokens.

Prints ms/token and free HBM per request, so a downward slope is visible.
"""

from __future__ import annotations

import argparse
import os
import time


def decode_only(e, ids, n, sp, mod):
    """ms/token and tok/forward over DECODE ticks only (prefill burned first)."""
    import torch

    rid = e.submit(ids, sp(n))
    req = None
    while req is None or req.phase != mod._PHASE_DECODE:
        e.step()
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is None:
            raise SystemExit("request finished during prefill — prompt too short for n")
    torch.cuda.synchronize()
    s0, t0 = e.stats(), time.perf_counter()
    out = None
    while out is None:
        e.step()
        out = e.take(rid)
    torch.cuda.synchronize()
    wall, s1 = time.perf_counter() - t0, e.stats()
    tok = s1["tokens_generated"] - s0["tokens_generated"]
    fwd = s1["decode_forwards"] - s0["decode_forwards"]
    free = torch.cuda.mem_get_info()[0] / 2**30
    return wall * 1000 / max(tok, 1), tok / max(fwd, 1), free, s1["prefix_published"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--reqs", type=int, default=6)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")

    from tilerl_kernels.backend import get_backend

    from tilerl import cli
    from tilerl import engine as mod
    from tilerl.engine import SamplingParams
    from tilerl.spec import load_draft

    cli._QWEN38_SOURCE = args.source
    backend = get_backend()
    cfg, model = cli._build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft) if args.draft else None

    def sp(n):
        return SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=n, seed=0)

    e = cli._build_engine(cfg, model, backend, draft=draft, depth=args.depth,
                          slots=4, blocks=2048, max_ctx=4096)
    # Distinct prompts: each is a prefix MISS, so each publishes snapshots the
    # way a real conversation does.
    prompts = [list(range(1000 * (i + 1), 1000 * (i + 1) + args.ctx)) for i in range(args.reqs)]
    decode_only(e, prompts[0], 8, sp, mod)  # warm: capture every (B, W)
    print(f"{'req':>4} {'ms/tok':>8} {'tok/s':>7} {'tok/fwd':>8} {'free GiB':>9} {'published':>10}")
    first = None
    for i, ids in enumerate(prompts):
        ms, per_fwd, free, pub = decode_only(e, ids, args.tokens, sp, mod)
        first = ms if first is None else first
        print(f"{i:>4} {ms:>8.1f} {1000 / ms:>7.1f} {per_fwd:>8.2f} {free:>9.2f} {pub:>10}")
    print(f"\nlast/first = {ms / first:.2f}x  (>1 means it degrades per request)")
    print("bench_ctx_decode at 512 ctx d3 reads 20.7 ms/token (48.4 tok/s).")
    e.shutdown()


if __name__ == "__main__":
    main()
