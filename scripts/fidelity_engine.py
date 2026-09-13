#!/usr/bin/env python3
"""Dense-vs-sparse fidelity through the PRODUCTION engine path (2026-09-12).

The first harness (fidelity_checks.py) subclassed SparseForward and replayed a
hand-built packed table; against the post-#525/#528 live engine its full-k arm
diverged on the V100 (KL 1.02) while the dense layout controls passed — the
replay, not the model. This rebuild runs the real thing:

  dense  = build_engine(sparse_k=0)
  sparse = build_engine(sparse_k=k, scorer="bounds")

Same model object, same token ids. Prefill logits at the 256 seeded positions
are captured by wrapping model.forward (it only slices logits; selection,
promotion, the packed table and the GDN state are the engine's own), then
compared as mean per-token KL(dense||sparse) and top-1 agreement. Greedy
agreement is 64 tokens of the engine's native greedy decode from one shared
prefix. There is no SparseForward subclass and no hand-built table.

--tiny: CPU self-gate. Sparse k = indexable pages selects every earlier page,
so the production sparse engine must reproduce dense PREFILL logits with
KL < 1e-5 / top1 1.0 (the engine's sampled-token gate already passes; this adds
per-position prefill-logit fidelity). No k above what the production hot pool
(n_groups*k+window+chunk per slot) fits is run on a card.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore


def head_at(model, backend, cfg, hidden, rows):
    """final rmsnorm + lm_head on selected prefill rows only."""
    h = hidden.index_select(0, rows)
    h = backend.rmsnorm(h, model.params["final_norm"], cfg.rms_eps, narrow=True)
    return model._linear(backend, h, cfg.head_key)


def sample_positions(t: int, j: int, nq: int = 256, minp: int = 2048):
    """Same deterministic seeded positions as output_fidelity.sample_positions."""
    pos = set()
    g = torch.Generator().manual_seed(j + 100003)
    while len(pos) < nq:
        p = int(torch.randint(minp, t, (1,), generator=g))
        pos.add(p)
    return torch.tensor(sorted(pos))


def attach_capture(model, cfg, backend, wanted: set[int]):
    """Snapshot pre-final-norm hidden at absolute wanted positions from the
    production prefill's hidden_out (the forward itself returns only the chunk's
    last logit row, so interior positions come from hidden, then head_at). The
    real forward is otherwise untouched. Returns (cap, restore)."""
    orig = model.forward
    cap: dict[int, torch.Tensor] = {}

    def wrapped(ids, positions, kv, be, **kw):
        if kw.get("hidden_out") is None:
            kw["hidden_out"] = []  # engine passes None with no draft
        out = orig(ids, positions, kv, be, **kw)
        hid = kw.get("hidden_out")
        if hid:
            pb = torch.as_tensor(positions)
            h = hid[-1]  # [B,T,H] pre-final-norm, this tick
            if h.dim() == 3:
                for b in range(h.shape[0]):
                    # one batched lm_head for every wanted row in THIS chunk
                    want_rows = [
                        jj
                        for jj in range(h.shape[1])
                        if int(pb[b, jj]) in wanted and int(pb[b, jj]) not in cap
                    ]
                    if want_rows:
                        rows = torch.tensor(want_rows, device=h.device)
                        lg = head_at(model, be, cfg, h[b], rows).detach().float().cpu()
                        for i, jj in enumerate(want_rows):
                            cap[int(pb[b, jj])] = lg[i]
        return out

    model.forward = wrapped
    return cap, lambda: setattr(model, "forward", orig)


def drain_prefill(engine, rid):
    """Step until the request leaves prefill (captures every prefill logit)."""
    for _ in range(1_000_000):
        for r in engine._running:
            if r.req_id == rid and r.phase == 2:
                return
        engine.step()
    raise TimeoutError("request never reached decode")


def drain_tokens(engine, rid, n):
    for _ in range(1_000_000):
        done = engine.poll()
        if rid in done and len(done[rid]) >= n:
            return done[rid][:n]
        engine.step()
    raise TimeoutError("engine did not finish")


def metrics(dense_lp, sparse_lp):
    kl = top = 0
    for a, b in zip(dense_lp, sparse_lp):
        kl += float((a.exp() * (a - b)).sum())
        top += int(a.argmax() == b.argmax())
    n = max(1, len(dense_lp))
    return {"kl": round(kl / n, 6), "top1": round(top / n, 4)}


def make_engine(model, backend, t, k, num_blocks, cold_format=""):
    common = dict(
        cfg=model.cfg,
        model=model,
        backend=backend,
        num_slots=1,
        max_batch=1,
        max_total_tokens=t + 4096,
        max_num_batched_tokens=512,
        prefix_store=NoPrefixStore(),
        # Both arms eager: sparse forces eager in build_engine, and a dense arm
        # that lazily captures on its first decode tick is not apples-to-apples.
        # On sm70 the capture also fails mid-kernel; eager fallback ran the tokens
        # but left the caching allocator's captures_underway set, so the
        # between-arm empty_cache asserted. decode_graph=False sidesteps both.
        decode_graph=False,
    )
    if k == 0:
        return build_engine(num_blocks=num_blocks, cold_format=cold_format, **common)
    return build_engine(
        sparse_k=k,
        scorer="bounds",
        kv_cold_bytes=1 << 34,
        num_blocks=num_blocks,
        cold_format=cold_format,
        **common,
    )


def run_arm(model, backend, ids, k, wanted, n_greedy, num_blocks, cold_format=""):
    """One full prefill: capture prefill logits at `wanted`, then take the
    engine's native greedy continuation from the span end. Production path only."""
    eng = make_engine(model, backend, len(ids), k, num_blocks, cold_format)
    cap, restore = attach_capture(model, model.cfg, backend, set(wanted.tolist()))
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=n_greedy, seed=0))
    drain_prefill(eng, rid)
    cont = drain_tokens(eng, rid, n_greedy)
    prefill = {p: cap[int(p)] for p in wanted.tolist()}
    restore()
    eng.shutdown()
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()  # release this arm's KV/state pools before the next build
    return prefill, cont


def first_divergence(a, b):
    """Index of the first mismatching greedy token, or None if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def compare(model, backend, ids, t, qpos, ks, n_greedy, num_blocks, out, cold_format=""):
    result = {}
    dc, dg = run_arm(model, backend, ids, 0, qpos, n_greedy, num_blocks, cold_format)
    dlp = [torch.log_softmax(dc[int(p)], -1) for p in qpos.tolist()]
    for k in ks:
        print(f"arm sparse_k={k}", flush=True)
        sc, sg = run_arm(model, backend, ids, k, qpos, n_greedy, num_blocks, cold_format)
        slp = [torch.log_softmax(sc[int(p)], -1) for p in qpos.tolist()]
        m = metrics(dlp, slp)
        m["greedy_agreement"] = round(sum(int(a == b) for a, b in zip(dg, sg)) / max(1, len(dg)), 4)
        m["diff_idx"] = first_divergence(dg, sg)
        m["dense_greedy"] = dg
        m["sparse_greedy"] = sg
        m["greedy_equal"] = dg == sg
        result[f"sparse_k={k}"] = m
        print("METRIC", f"k={k}", {kk: vv for kk, vv in m.items()}, flush=True)
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)
    print("RESULT", json.dumps(result), flush=True)
    return result


def tiny(out):
    from tilerl_kernels.backend import get_backend

    from tilerl import config as config_mod
    from tilerl.model import build_random

    cfg = config_mod.tiny(4096)
    backend = get_backend()
    model = build_random(cfg, seed=11)
    torch.manual_seed(0)
    # 48 pages. Full continuity needs k=PAGES (at decode the forced window is
    # the own span, not +8, so k=pages-8 would genuinely omit 8 earlier pages
    # and rightly differ in decode). n_groups=1 on tiny, so the pool fits.
    t = 48 * BLOCK_TOKENS
    ids = np.asarray(torch.randint(0, cfg.vocab_size, (t,)).numpy())
    qpos = torch.tensor(sorted({t - 2, t - 20, t // 2, t // 4, BLOCK_TOKENS + 1}))
    k_full = t // BLOCK_TOKENS
    res = compare(model, backend, ids, t, qpos, [k_full], 8, num_blocks=2048, out=out)
    m = res[f"sparse_k={k_full}"]
    assert m["kl"] < 1e-5 and m["top1"] == 1.0 and m["greedy_equal"], m
    print("TINY GATE PASS", m, flush=True)


def main27b(args):
    from pathlib import Path

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model

    backend = get_backend()
    _c, model = _build_model("qwen38-27b", seed=0, keep_master=False, backend=backend)
    with open(Path(args.corpus) / f"held_{args.ctx}.jsonl") as fh:
        r = json.loads(fh.readlines()[args.span])
    ids = np.asarray(r["ids"], dtype=np.int64)[: args.ctx]
    t = len(ids)
    qpos = sample_positions(t, args.span)
    ks = [int(x) for x in args.ks.split(",")]
    # num_blocks=0: dense fits the pool by _fit_blocks; sparse forces its own
    # n_groups*k+window+chunk pool regardless of this.
    compare(
        model,
        backend,
        ids,
        t,
        qpos,
        ks,
        64,
        num_blocks=0,
        out=args.out,
        cold_format=args.cold_format,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?")
    ap.add_argument("--span", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=32768)
    # V100 32GB: only k=128 fits the 4-group hot pool at slots=1 (1107 blocks);
    # k=1024 is ~8 GiB K+V over the ~4.9 GiB post-weights headroom. H20 takes more.
    ap.add_argument("--ks", default="128")
    # ""=engine default (f16 cold on an f32 sm70 pool), "native"=pool dtype.
    ap.add_argument("--cold-format", default="", choices=["", "f16", "native"])
    ap.add_argument("--out", default="/tmp/fidelity-engine.json")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    tiny(a.out) if a.tiny else main27b(a)
