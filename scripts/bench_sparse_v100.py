"""V100 (sm70) dense vs Quest-sparse decode at 8k/32k/64k, 27B NVFP4.

Unit B card control (design #486): every page device-resident, so the ONLY
difference between arms is how many pages the unchanged sm70 split
paged_attention reads — the sparse arm hands it a per-4-layer-group selected
block table. Attention kernel, weights, pool and prefill are identical; both
arms run in one process with decode graphs OFF (the selection wrapper needs
eager), one request at a time.

Selection uses the registered backend cells (page_bounds f16 index, f32
page_bound_scores, reference select_pages) in DeepSeek-V4.1 form: the score is
computed once per 4-plane group on its source plane, and the last 8 pages
(128 tokens) are always unioned in. Bounds are the append-time index — prefill
fills every frozen page's bound on the 4 group source planes, decode refreshes
only the open last page, so the decode wall prices the real index cost
(4 last-page bound updates + 4 scorings per token).

The selected table stays in sequence order and ends on the true last page, so
the kernel's end-of-sequence truncation gives the exact subset softmax — no
attention kernel change.

  scripts/v100.sh run sparse27 'CKPT=/data00/home/chenkailun.c/models/Qwen3.8-27B-NVFP4;
    /usr/bin/python3 -u scripts/bench_sparse_v100.py --source $CKPT --k-pages 128'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402
from tilerl_kernels.reference import select_pages  # noqa: E402

from tilerl import cli  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import _PHASE_DECODE, SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import BLOCK_TOKENS  # noqa: E402

N_WINDOW = 8  # V4.1 local window: 8 pages x 16 tokens = 128 tokens


def _prompt(ctx: int, vocab: int) -> list[int]:
    # Same fixed distribution as bench_ctx_decode (randint seed 1000): length is
    # the only thing that differs between points.
    g = torch.Generator().manual_seed(1000)
    return torch.randint(0, vocab, (ctx,), generator=g).tolist()


def install_sparse(backend, pool, cfg, k_pages: int, timings: dict):
    """Wrap backend.paged_attention with Quest selection on B=1 ticks.

    Returns restore(). Bounds/selection state is per install, so a new one per
    context never sees a freed block id reused with stale bounds.
    """
    orig = backend.paged_attention
    hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    group = hq // hkv
    base = pool.k_pool.data_ptr()
    plane_bytes = pool.k_pool[0].numel() * pool.k_pool.element_size()
    dev = backend.device
    # One f16 bounds plane per GROUP SOURCE plane (4), indexed by PHYSICAL block
    # id (the free-list reuses ids across sequences, so position keying aliases).
    # 4 planes x 4096 blocks x 4 heads x 2 x 256 x 2B ~= 64 MB at the 64k pool.
    nblk = pool.num_blocks
    bounds: dict[int, torch.Tensor] = {}
    # Device-side frozen set per plane (block id -> bound current); a Python set
    # here would mean thousands of D2H syncs per decode tick.
    is_frozen: dict[int, torch.Tensor] = {}
    picked: dict[int, torch.Tensor] = {}
    last_n = 0

    def plane_state(p):
        b = bounds.get(p)
        if b is None:
            b = torch.empty(nblk, hkv, 2, d, dtype=torch.float16, device=dev)
            bounds[p] = b
            is_frozen[p] = torch.zeros(nblk, dtype=torch.bool, device=dev)
        return b, is_frozen[p]

    def refresh(p, k_cache, ids, n_pages):
        # Complete prefix pages are bounded once; the open last page is rebuilt
        # every tick — its zero-filled unwritten slots would put 0 in kmin, and
        # bounding it before it fills would freeze that bound.
        nonlocal last_n
        if n_pages < last_n:  # new request: physical ids are reused with new KV
            for m in is_frozen.values():
                m.zero_()
            picked.clear()
        last_n = n_pages
        bp, frz = plane_state(p)
        pref = ids[: n_pages - 1]
        need = ~frz[pref]
        if need.any():
            sel = pref[need]
            bp[sel] = backend.page_bounds(k_cache[sel])  # one launch for new prefix
            frz[sel] = True
        last = ids[n_pages - 1: n_pages]
        bp[last] = backend.page_bounds(k_cache[last])

    def wrapper(q, k_cache, v_cache, block_table, seq_lens, scale, gate=None,
                seq_q_lens=None, k_scale=None, v_scale=None):
        b, t = q.shape[0], q.shape[1]
        if b != 1:
            raise SystemExit(f"sparse harness is B=1 only, got B={b}")
        p = (k_cache.data_ptr() - base) // plane_bytes
        n_pages = (int(seq_lens[0]) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        ids = block_table[0, :n_pages].to(dev)
        if t > 1:  # prefill chunks always attend densely; source layers build the index
            if p % 4 == 0:
                refresh(p, k_cache, ids, n_pages)
            return orig(q, k_cache, v_cache, block_table, seq_lens, scale, gate=gate,
                        seq_q_lens=seq_q_lens, k_scale=k_scale, v_scale=v_scale)
        src = p - p % 4  # 4 full-attn planes per group; source is the first
        s0 = torch.cuda.Event(enable_timing=True)
        s0.record()
        if src == p:
            refresh(p, k_cache, ids, n_pages)  # open last page re-bounds every tick
            qi = q[0, 0].reshape(hkv, group, d).mean(1).unsqueeze(0)  # [1,Hkv,D]
            bnd = bounds[p][ids]  # gathered in sequence order, [P,Hkv,2,D]
            scores = backend.page_bound_scores(qi, bnd).sum(1)
            picked[src] = select_pages(
                block_table[:1, :n_pages], torch.tensor([n_pages], device=dev),
                scores.reshape(1, 1, n_pages), k_pages, N_WINDOW)[0, 0]
        width = picked[src].numel()
        last_tokens = (int(seq_lens[0]) - 1) % BLOCK_TOKENS + 1
        sub_len = torch.tensor([(width - 1) * BLOCK_TOKENS + last_tokens], dtype=torch.int32)
        sub_tbl = picked[src].to(torch.int32).reshape(1, width).contiguous()
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        e0.synchronize()
        timings["index_ms"] += s0.elapsed_time(e0)
        timings["ticks"] += 1
        return orig(q, k_cache, v_cache, sub_tbl, sub_len, scale, gate=gate,
                    seq_q_lens=None, k_scale=k_scale, v_scale=v_scale)

    backend.paged_attention = wrapper
    return lambda: setattr(backend, "paged_attention", orig)


def run_one(e, ctx: int, tokens: int, vocab: int) -> float:
    """tok/s over decode ticks only."""
    rid = e.submit(_prompt(ctx, vocab),
                   SamplingParams(temperature=0.0, max_new_tokens=tokens, seed=0))
    done = {}
    for _ in range(8192):
        done.update(e.poll())
        req = next((r for r in e._running if r.req_id == rid), None)
        if req is not None and req.phase == _PHASE_DECODE:
            break
        e.step()
    else:
        raise SystemExit(f"ctx={ctx}: never reached decode")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(16 * tokens + 64):
        e.step()
        done.update(e.poll())
        if rid in done:
            break
    else:
        raise SystemExit(f"ctx={ctx}: request did not finish")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    for _ in range(4096):
        if not e._running and not e._waiting:
            break
        e.step()
        e.poll()
    else:
        raise SystemExit(f"ctx={ctx}: drain stalled")
    return tokens / wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--k-pages", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--draws", type=int, default=2,
                    help="timed draws per arm (a prefill each; long contexts are prefill-bound "
                         "at ~31 ms/prompt token, so 1 draw after the untimed warmup bounds it)")
    ap.add_argument("--ctxs", default="8192,32768,65536")
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    ctxs = [int(c) for c in args.ctxs.split(",")]
    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    blocks = (-(-(max(ctxs) + args.tokens + 4) // BLOCK_TOKENS)) + 32
    print(f"pool: {blocks} blocks ({blocks * 2.125:.0f} MiB), f32 KV, "
          f"k_pages={args.k_pages} n_window={N_WINDOW}", flush=True)
    e = build_engine(cfg, model, backend, num_blocks=blocks, num_slots=3, max_batch=2,
                     max_total_tokens=max(ctxs) + args.tokens + 4, decode_graph=False)
    # The attention kernel keys on block-table width (dense: pool blocks; sparse:
    # k_pages+window), constant across ctxs; the scorer/bounds kernels key on P,
    # which changes per context. So the first sparse draw at EACH ctx is untimed
    # warmup absorbing that compile; the dense arm compiles once above.
    run_one(e, ctxs[0], args.tokens, cfg.vocab_size)
    print(f"{'ctx':>6} {'dense tok/s':>12} {'sparse tok/s':>13} {'speedup':>8} "
          f"{'index ms/tok':>13}", flush=True)
    for ctx in ctxs:
        dense = sum(run_one(e, ctx, args.tokens, cfg.vocab_size)
                    for _ in range(args.draws)) / args.draws
        timings = {"index_ms": 0.0, "ticks": 0}
        restore = install_sparse(backend, e._kv, cfg, args.k_pages, timings)
        run_one(e, ctx, args.tokens, cfg.vocab_size)  # P-specific scorer JIT, untimed
        sparse = sum(run_one(e, ctx, args.tokens, cfg.vocab_size)
                     for _ in range(args.draws)) / args.draws
        idx = timings["index_ms"] / max(timings["ticks"], 1)
        restore()
        print(f"{ctx:>6} {dense:>12.1f} {sparse:>13.1f} {sparse / dense:>7.2f}x "
              f"{idx:>13.2f}", flush=True)


if __name__ == "__main__":
    main()
