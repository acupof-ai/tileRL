"""V100 sm70 GPU-TIME cost of Quest sparse attention vs dense, 27B NVFP4.

The wall-clock arm (bench_sparse_v100.py) is host-dispatch-bound with graphs off:
dense stayed 8.4 / 8.0 tok/s from 8k to 32k while graph-on serving historically
scaled 38 -> 15 tok/s there, so wall time cannot see an attention saving. This
probe measures the thing selection actually changes — GPU ms in the attention
kernel and in the scorer — with CUDA events, host dispatch excluded.

One prefill to --ctx (default 65536, ~34 min on the V100), then on the first
decode tick the pool holds every page and the wrapper runs a sweep: for each
history L it calls the unchanged sm70 split paged_attention 1) over the first
L/16 pages (dense) and 2) over the k_pages+window pages the Quest scorer picks
among those, timing both with GPU events, plus the scorer itself. Median of
--reps calls. K/V reads are real pool pages; the decode q is reused across the
sweep (values do not move the timing). One prefill buys the whole curve.

  scripts/v100.sh run gputime 'CKPT=/data00/.../Qwen3.8-27B-NVFP4;
    /usr/bin/python3 -u scripts/bench_sparse_gpu_v100.py --source $CKPT'
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
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import BLOCK_TOKENS  # noqa: E402

N_WINDOW = 8
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def gpu_ms(fn, reps):
    for _ in range(3):  # warm
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def install(backend, pool, cfg, k_pages: int, reps: int, result: dict):
    orig = backend.paged_attention
    hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    group = hq // hkv
    base = pool.k_pool.data_ptr()
    plane_bytes = pool.k_pool[0].numel() * pool.k_pool.element_size()
    dev = backend.device
    nblk = pool.num_blocks
    bounds = {}
    is_frozen = {}
    swept = {"v": False}

    def plane_state(p):
        b = bounds.get(p)
        if b is None:
            b = torch.empty(nblk, hkv, 2, d, dtype=torch.float16, device=dev)
            bounds[p] = b
            is_frozen[p] = torch.zeros(nblk, dtype=torch.bool, device=dev)
        return b, is_frozen[p]

    def build_bounds(p, k_cache, ids):
        bp, frz = plane_state(p)
        need = ~frz[ids]
        if need.any():
            sel = ids[need]
            bp[sel] = backend.page_bounds(k_cache[sel])
            frz[sel] = True

    def sweep(q, k_cache, v_cache, scale, gate, k_scale, v_scale, table_full, n_total):
        """GPU-ms curves on the populated pool; one call per source plane is enough
        (the 4 planes of a group dispatch identically)."""
        p = (k_cache.data_ptr() - base) // plane_bytes
        if p != 0 or swept["v"]:
            return
        ids_all = table_full[0, :n_total].to(dev)
        build_bounds(0, k_cache, ids_all)
        qi = q[0, 0].reshape(hkv, group, d).mean(1).unsqueeze(0)
        rows = []
        for L in LENGTHS:
            np_ = L // BLOCK_TOKENS
            if np_ > n_total:
                continue
            ids = ids_all[:np_]
            tbl_d = ids.to(torch.int32).reshape(1, np_).contiguous()
            len_d = torch.tensor([L], dtype=torch.int32)

            def dense():
                orig(q, k_cache, v_cache, tbl_d, len_d, scale, gate=gate,
                     seq_q_lens=None, k_scale=k_scale, v_scale=v_scale)

            # Selection once per L (host-side; only the scorer KERNEL is timed below).
            bnd = bounds[0][ids]
            score_ms = gpu_ms(lambda: backend.page_bound_scores(qi, bnd), reps)
            scores = backend.page_bound_scores(qi, bnd).sum(1)
            picked = select_pages(
                table_full[:1, :np_], torch.tensor([np_], device=dev),
                scores.reshape(1, 1, np_), k_pages, N_WINDOW)[0, 0]
            w = picked.numel()
            tbl_s = picked.to(torch.int32).reshape(1, w).contiguous()
            len_s = torch.tensor([w * BLOCK_TOKENS], dtype=torch.int32)

            def sparse():
                orig(q, k_cache, v_cache, tbl_s, len_s, scale, gate=gate,
                     seq_q_lens=None, k_scale=k_scale, v_scale=v_scale)

            attn_d = gpu_ms(dense, reps)
            attn_s = gpu_ms(sparse, reps)
            rows.append((L, w, attn_d, attn_s, score_ms))
            print(f"L={L:>6} pages_dense={np_:>4} pages_sparse={w:>3} "
                  f"attn dense {attn_d:7.3f} ms  sparse {attn_s:7.3f} ms  "
                  f"({attn_d / attn_s:5.2f}x)  scorer {score_ms:6.3f} ms", flush=True)
        result["rows"] = rows
        swept["v"] = True

    def wrapper(q, k_cache, v_cache, block_table, seq_lens, scale, gate=None,
                seq_q_lens=None, k_scale=None, v_scale=None):
        t = q.shape[1]
        n_total = (int(seq_lens[0]) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        if t == 1 and n_total * BLOCK_TOKENS >= min(LENGTHS) and not swept["v"]:
            sweep(q, k_cache, v_cache, scale, gate, k_scale, v_scale,
                  block_table, n_total)
        return orig(q, k_cache, v_cache, block_table, seq_lens, scale, gate=gate,
                    seq_q_lens=seq_q_lens, k_scale=k_scale, v_scale=v_scale)

    backend.paged_attention = wrapper
    return lambda: setattr(backend, "paged_attention", orig)


def _prompt(ctx: int, vocab: int) -> list[int]:
    g = torch.Generator().manual_seed(1000)
    return torch.randint(0, vocab, (ctx,), generator=g).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--k-pages", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    os.environ.setdefault("TILERL_TARGET", "cuda")
    cli._QWEN38_SOURCE = args.source

    backend = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    blocks = (-(-(args.ctx + 8) // BLOCK_TOKENS)) + 32
    print(f"pool: {blocks} blocks ({blocks * 2.125:.0f} MiB); one prefill to {args.ctx}",
          flush=True)
    e = build_engine(cfg, model, backend, num_blocks=blocks, num_slots=3, max_batch=2,
                     max_total_tokens=args.ctx + 8, decode_graph=False)
    result = {}
    restore = install(backend, e._kv, cfg, args.k_pages, args.reps, result)
    rid = e.submit(_prompt(args.ctx, cfg.vocab_size),
                   SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
    done = {}
    t0 = time.perf_counter()
    for _ in range(60000):
        e.step()
        done.update(e.poll())
        if "rows" in result:
            break
        if rid in done:
            raise SystemExit("request finished before sweep ran")
    restore()
    print(f"SWEEP_DONE in {time.perf_counter() - t0:.0f}s, {len(result.get('rows', []))} points",
          flush=True)


if __name__ == "__main__":
    main()
