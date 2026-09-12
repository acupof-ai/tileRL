#!/usr/bin/env python3
"""Output fidelity of the packed-sparse forward (#518 bounds seam) vs the dense
forward on held 32k/16k spans.

Per span the dense arm replays 512-token prefill chunks once (paged dense, the
capture path: exact dense-causal attention). Each sparse arm replays the SAME
chunks with its own KV pool, GDN state, SparseTracker and SparseForward — the
engine's objects, selection and packed [selected; own] table unchanged; the
only engine bit omitted is host demotion (all pages device-resident: moving
bytes cannot move logits).

Metrics per k in {128,256,512,1024} at the same 256 seeded positions the recall
runs use (uniform over positions >= 2048), FULL-VOCAB logits:
  - mean per-token KL(dense || sparse);
  - top-1 greedy agreement;
  - 64-token independent greedy continuation from one fixed position per span
    (32k: 20480, 16k: 10240; 512-aligned). The GDN state is snapshotted at the
    boundary after prefill reaches it and restored before each continuation,
    which appends to fresh KV pages, so the rollout is independent of the
    prefill that ran past the boundary.
Oracle k=1024: selection forced by true dense attention mass streamed from the
arm's own pool (max over the chunk's sampled query rows; chunks without a
sampled row use 8 evenly spaced rows — ponytail: full-chunk max is
O(chunk*prefix)). No [T,T] matrix anywhere.

Read-only eval. Usage:
  output_fidelity.py CORPUS OUT [--ctxs 32768:8 16384:4] [--gpu 7]
  output_fidelity.py --selftest
"""
import argparse
import json
from types import SimpleNamespace

import torch

from tilerl.cli import _build_model
from tilerl.kv_cache import BLOCK_TOKENS, LinearStatePool, PagedKvPool
from tilerl.sparse_engine import SparseForward, SparseTracker
from tilerl.sparse_index import WINDOW_PAGES
from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import select_pages

KS = (128, 256, 512, 1024)
ORACLE_K = 1024
CHUNK, NQ, MINP, SEED, GREEDY_N = 512, 256, 2048, 0, 64
GROUP = {8192: 2, 16384: 0, 32768: 1}
GREEDY_P = {16384: 10240, 32768: 20480}
FRESH_PAD = 8                          # extra pool pages for the 64-token tail


def sample_positions(t, j):
    gen = torch.Generator().manual_seed(SEED + 100003 * GROUP.get(t, 0) + j)
    perm = torch.randperm(t - MINP, generator=gen)[: min(NQ, t - MINP)]
    return (perm + MINP).sort().values


def new_state(cfg, device):
    from tilerl import precision

    return LinearStatePool(1, cfg.num_layers - len(cfg.full_attn_layers),
                           cfg.linear_num_value_heads, cfg.linear_value_head_dim,
                           device=device,
                           dtype=precision.dtype("recurrent_state", device),
                           conv_window=cfg.linear_conv_kernel_dim - 1,
                           conv_dim=cfg.linear_qkv_dim)


def snapshot_state(state):
    return (state.states[0].clone(),
            None if state.conv_windows is None else state.conv_windows[0].clone(),
            int(state.win_parity[0]))


def restore_state(state, snap):
    state.states[0].copy_(snap[0])
    if snap[1] is not None:
        state.conv_windows[0].copy_(snap[1])
    state.win_parity[0] = snap[2]


def kv_desc(pool, state, table, seq_len, tq, sf=None):
    return SimpleNamespace(
        dense=False, kv_pool=pool, block_table=table,
        seq_len=torch.tensor([seq_len], device=pool.k_pool.device),
        seq_q_lens=torch.tensor([tq], device=pool.k_pool.device),
        state_pool=state,
        state_slot=torch.zeros(1, dtype=torch.long, device=pool.k_pool.device),
        sparse=sf, page_base=None if sf is None else sf.page_base)


def finalize_bounds(tracker, pool, resolve, complete_hi):
    """Engine _sparse_finalize's bounds write (demotion omitted)."""
    for p in range(len(tracker.bounds[0]), complete_hi // BLOCK_TOKENS):
        phys = resolve(p)
        b = torch.stack([
            torch.stack((pool.k_pool[pl, phys].amin(dim=1),
                         pool.k_pool[pl, phys].amax(dim=1)), dim=1)
            for pl in range(pool.num_layers)]).to(torch.float16)
        tracker.set_bounds(0, p, b)


def head_at(model, backend, cfg, hidden, rows):
    """final rmsnorm + lm_head on selected prefill rows only; the full-chunk
    head is ~0.5 GiB and rows here are the <=15 sampled positions."""
    h = hidden.index_select(0, rows)
    h = backend.rmsnorm(h, model.params["final_norm"], cfg.rms_eps, narrow=True)
    return model._linear(backend, h, cfg.head_key)


def stream_mass(q, qpos, page_k):
    """Per-query causal attention mass on key pages, streamed a page at a time.
    q [n,hq,d] post-rope, qpos [n]; page_k(p) -> [16,hkv,d]. -> [n,P] mass."""
    n, hq, d = q.shape
    rep = None

    def expand(kp):
        nonlocal rep
        kp = kp.transpose(0, 1).float()          # pool stores [hkv,blk,d]
        if rep is None:
            rep = hq // kp.shape[1]
        return kp.repeat_interleave(rep, dim=1)

    fmin = torch.finfo(torch.float32).min
    run_max = torch.full((n, hq), fmin, device=q.device)
    run_sum = torch.zeros(n, hq, device=q.device)
    seen = []
    p = 0
    while True:
        try:
            k = expand(page_k(p))
        except IndexError:
            break
        blk = k.shape[0]
        s = torch.einsum("nhd,khd->nhk", q.float(), k) * (d ** -0.5)
        valid = qpos[:, None] >= (p * blk + torch.arange(blk, device=q.device))[None]
        sv = s.masked_fill(~valid[:, None], float("-inf"))
        bmax = sv.amax(-1)
        bsum = torch.where(valid[:, None].expand_as(sv),
                           torch.exp(sv - bmax[..., None]), 0.0).sum(-1)
        new_max = torch.maximum(run_max, bmax)
        run_sum = run_sum * torch.exp(run_max - new_max) + bsum * torch.exp(bmax - new_max)
        run_max = new_max
        seen.append((sv, valid))
        p += 1
    lse = run_max + run_sum.clamp_min(1e-30).log()
    mass = torch.zeros(n, p, device=q.device)
    for i, (sv, valid) in enumerate(seen):
        mass[:, i] = torch.where(valid[:, None].expand_as(sv),
                                 torch.exp(sv - lse[..., None]), 0.0).sum(-1).mean(1)
    return mass


class OracleForward(SparseForward):
    """SparseForward scoring candidates with true dense mass from this arm's own
    pool instead of Quest bounds, then the same select_pages top-k + window."""

    def __init__(self, tracker, rows, device, pool, resolve, score_rows):
        super().__init__(tracker, rows, device)
        self._pool, self._resolve, self._score_rows = pool, resolve, score_rows

    def _select(self, bi, plane, q):
        g = self.tracker.group_of[plane]
        key = (bi, g)
        if key in self._phys:
            return self._phys[key]
        r = self.rows[bi]
        chosen = []
        cand = r["cand"]
        if cand:
            rows = self._score_rows.get((bi, r["q_start"], r["q_hi"]))
            if rows is None:
                rows = torch.arange(q.shape[0], device=q.device)
            pages = cand + r["own"]
            mass = stream_mass(
                q.index_select(0, rows),
                (r["q_start"] + rows).to(q.device),
                lambda p, plane=plane, pages=pages:
                    self._pool.k_pool[plane, self._resolve(pages[p])]
                    if p < len(pages) else _stop_page())
            score = mass[:, : len(cand)].amax(0)
            table = (torch.tensor(cand, device=self.device) + 1).reshape(1, -1)
            sel = select_pages(
                table, torch.tensor([len(cand)], device=self.device),
                score.reshape(1, 1, len(cand)), self.tracker.k_pages,
                n_window=r["force_window"])[0, 0]
            chosen = [int(x) - 1 for x in sel.tolist() if int(x) != 0]
        phys_t = torch.tensor([r["resolve"](p) for p in chosen],
                              dtype=torch.long, device=self.device)
        self._chosen[key] = chosen
        self._phys[key] = phys_t
        return phys_t


def _stop_page(*_):
    raise IndexError


def sparse_row(lo, hi, decoding, bound_pages=None):
    own_first = (max(0, (lo // BLOCK_TOKENS) - (WINDOW_PAGES - 1)) if decoding
                 else lo // BLOCK_TOKENS)
    own = list(range(own_first, (hi - 1) // BLOCK_TOKENS + 1))
    cand = [p for p in range(own_first)
            if bound_pages is None or p in bound_pages]
    return own_first, dict(
        req_id=0,
        own=own, own_len=hi - own_first * BLOCK_TOKENS, q_start=lo, q_hi=hi,
        decoding=decoding, tq=hi - lo, cand=cand,
        force_window=0 if decoding else WINDOW_PAGES)


def evaluate_span(model, backend, cfg, ids, t, qpos, greedy_p, n_greedy=GREEDY_N):
    device = backend.device
    chunk = CHUNK
    bnd_page = greedy_p // BLOCK_TOKENS
    n_prefix_pages = t // BLOCK_TOKENS

    def make_pool():
        # prefix pages + FRESH_PAD untouched pages for the tail rollout.
        npp = n_prefix_pages + FRESH_PAD
        pool = PagedKvPool(npp, cfg.num_kv_heads, cfg.head_dim,
                           device=device, layer_map=cfg.full_attn_layers,
                           dtype=getattr(backend, "io", torch.bfloat16))
        phys = [pool.alloc_block() for _ in range(npp)]
        return pool, phys, new_state(cfg, device)

    # Sampled local rows that drive the oracle max-pool, per chunk.
    score_rows = {}
    for lo in range(0, t, chunk):
        hi = min(lo + chunk, t)
        rows = (qpos[(qpos >= lo) & (qpos < hi)] - lo).tolist()
        if not rows:
            rows = torch.linspace(0, hi - lo - 1, 8).long().tolist()
        score_rows[(0, lo, hi)] = torch.tensor(rows, device=device)

    def prefill_forward(pool, phys, state, lo, hi, tracker, mode):
        """Returns logits [tq,V] (head only on the rows this chunk needs) and
        the full pre-final-norm hidden (for the boundary row)."""
        _, row = sparse_row(lo, hi, False)
        if mode == "dense":
            kv = kv_desc(pool, state,
                         torch.tensor([phys[: hi // BLOCK_TOKENS]], device=device),
                         hi, hi - lo)
            sf = None
        elif mode == "oracle":
            row["resolve"] = lambda p, phys=phys: phys[p]
            sf = OracleForward(tracker, [row], device, pool,
                               lambda p, phys=phys: phys[p], score_rows)
            kv = kv_desc(pool, state, sf.own_table, hi, hi - lo, sf)
        else:
            row["resolve"] = lambda p, phys=phys: phys[p]
            sf = SparseForward(tracker, [row], device)
            kv = kv_desc(pool, state, sf.own_table, hi, hi - lo, sf)
        hidden = []
        model.forward(ids[:, lo:hi], torch.arange(lo, hi, device=device), kv,
                      backend, hidden_out=hidden, last_only=True)
        if mode != "dense":
            finalize_bounds(tracker, pool, lambda p, phys=phys: phys[p], hi)
        return hidden[0][0]                        # [tq,H] pre-final-norm

    def sampled_local(lo, hi):
        return (qpos[(qpos >= lo) & (qpos < hi)] - lo).to(device)

    def rollout(pool, phys, state, tracker, mode, boundary_hidden):
        """64 independent greedy tokens from greedy_p. The prefix's GDN state is
        restored by the caller; tail logical pages (>= bnd_page) map to FRESH
        physical blocks, so this never reads or overwrites the later prefill."""
        tail = {}

        def resolve_tail(p):
            if p < bnd_page:
                return phys[p]
            if p not in tail:
                tail[p] = phys[n_prefix_pages + len(tail)]
            return tail[p]

        toks, h, s = [], boundary_hidden[0], greedy_p
        for _ in range(n_greedy):
            hh = backend.rmsnorm(h, model.params["final_norm"], cfg.rms_eps,
                                 narrow=True)
            nxt = int(model._linear(backend, hh[None], cfg.head_key)[0].argmax())
            toks.append(nxt)
            _, row = sparse_row(s, s + 1, True,
                                None if mode == "dense" else tracker.bounds[0])
            row["resolve"] = resolve_tail
            hidden = []
            if mode == "dense":
                resolve_tail(s // BLOCK_TOKENS)      # own page must exist pre-write
                pages = list(range(bnd_page)) + sorted(tail)
                kv = kv_desc(pool, state,
                             torch.tensor([[resolve_tail(p) for p in pages]],
                                          device=device),
                             s + 1, 1)
            else:
                sf = (OracleForward(tracker, [row], device, pool, resolve_tail,
                                    score_rows) if mode == "oracle"
                      else SparseForward(tracker, [row], device))
                kv = kv_desc(pool, state, sf.own_table, s + 1, 1, sf)
            model.forward(torch.tensor([[nxt]], device=device),
                          torch.tensor([s], device=device), kv, backend,
                          hidden_out=hidden, last_only=False)
            if mode != "dense":
                finalize_bounds(tracker, pool, resolve_tail, s + 1)
            h = hidden[0][0, -1]
            s += 1
        return toks

    # ---- dense arm ----
    pool, phys, state = make_pool()
    teacher_lp = [None] * len(qpos)
    teacher_arg = [None] * len(qpos)
    qi = 0
    boundary_hidden = snap = None
    for lo in range(0, t, chunk):
        hi = min(lo + chunk, t)
        hidden = prefill_forward(pool, phys, state, lo, hi, None, "dense")
        locs = sampled_local(lo, hi)
        if len(locs):
            lg = head_at(model, backend, cfg, hidden, locs).float()
            lp = torch.log_softmax(lg, -1)
            for r, local in enumerate(locs.tolist()):
                teacher_lp[qi + r] = lp[r].cpu()
                teacher_arg[qi + r] = int(lp[r].argmax())
            qi += len(locs)
        if hi == greedy_p:
            boundary_hidden = hidden[-1:].clone()
            snap = snapshot_state(state)
    assert qi == len(qpos) and boundary_hidden is not None and snap is not None
    restore_state(state, snap)
    teacher_greedy = rollout(pool, phys, state, None, "dense", boundary_hidden)
    del pool, state
    device.type == "cuda" and torch.cuda.empty_cache()

    def run_arm(mode, k):
        pool, phys, state = make_pool()
        tracker = SparseTracker(cfg, k, "bounds")
        tracker.attach(0)
        qi = kl_sum = top1 = npos = 0
        boundary_hidden = snap = None
        for lo in range(0, t, chunk):
            hi = min(lo + chunk, t)
            hidden = prefill_forward(pool, phys, state, lo, hi, tracker, mode)
            locs = sampled_local(lo, hi)
            if hi == greedy_p:
                boundary_hidden = hidden[-1:].clone()
                snap = snapshot_state(state)
            if len(locs):
                lg = head_at(model, backend, cfg, hidden, locs).float()
                lp = torch.log_softmax(lg, -1)
                for r in range(len(locs)):
                    tp = teacher_lp[qi + r].float().to(lp.device)
                    kl_sum += float((tp.exp() * (tp - lp[r])).sum())
                    top1 += int(lp[r].argmax() == teacher_arg[qi + r])
                    npos += 1
                qi += len(locs)
        assert qi == len(qpos) and boundary_hidden is not None and snap is not None
        # Rollout sees only prefix pages: drop bounds/KV the prefill wrote past
        # greedy_p (tail logical pages resolve to fresh physical blocks).
        for p in [p for p in tracker.bounds[0] if p >= bnd_page]:
            del tracker.bounds[0][p]
        restore_state(state, snap)
        own_greedy = rollout(pool, phys, state, tracker, mode, boundary_hidden)
        g = sum(a == b for a, b in zip(own_greedy, teacher_greedy)) / n_greedy
        del pool, state
        device.type == "cuda" and torch.cuda.empty_cache()
        return dict(kl=kl_sum / npos, top1=top1 / npos, greedy=g)

    return {**{f"bounds{k}": run_arm("bounds", k) for k in KS},
            f"oracle{ORACLE_K}": run_arm("oracle", ORACLE_K)}


def selftest():
    """Tiny CPU model: when k >= page count the sparse arm equals dense."""
    from tilerl import config as config_mod
    from tilerl.model import build_random

    cfg = config_mod.tiny(4096)
    model = build_random(cfg, 0)
    backend = get_backend()
    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (1, 3072))
    qpos = sample_positions(3072, 0)
    res = evaluate_span(model, backend, cfg, ids, 3072, qpos, 1536, n_greedy=8)
    print(json.dumps(res, indent=1))
    assert res["bounds1024"]["top1"] == 1.0
    assert res["bounds1024"]["kl"] < 1e-3
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--ctxs", nargs="*", default=["32768:8", "16384:4"])
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    from pathlib import Path

    corpus = Path(args.corpus)
    backend = get_backend()
    _cfg, model = _build_model("qwen38-27b", seed=SEED, keep_master=False,
                               backend=backend)
    cfg = model.cfg
    result = {"seed": SEED, "nq": NQ, "chunk": CHUNK, "ks": list(KS),
              "oracle_k": ORACLE_K, "greedy_n": GREEDY_N, "ctxs": {}}
    for spec in args.ctxs:
        ctx, nspan = (int(x) for x in spec.split(":"))
        rows = [json.loads(l) for l in open(corpus / f"held_{ctx}.jsonl")][:nspan]
        gp = GREEDY_P.get(ctx, min(20480, (ctx // 512) * 256))
        per_span = []
        result["ctxs"][str(ctx)] = {"n_spans": nspan, "spans": per_span}
        for j, row in enumerate(rows):
            ids = torch.tensor([row["ids"]], device=backend.device)
            qpos = sample_positions(ctx, j).to(backend.device)
            r = evaluate_span(model, backend, cfg, ids, ctx, qpos, gp)
            per_span.append(r)
            print("span", ctx, j, json.dumps(r), flush=True)
            with open(args.out, "w") as fh:
                json.dump(result, fh)
        arms = [*(f"bounds{k}" for k in KS), f"oracle{ORACLE_K}"]
        agg = {a: {m: round(sum(s[a][m] for s in per_span) / len(per_span), 5)
                   for m in ("kl", "top1", "greedy")} for a in arms}
        result["ctxs"][str(ctx)]["arms"] = agg
        print("CTX", ctx, json.dumps(agg), flush=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh)
    print("RESULT", json.dumps(result["ctxs"]))


if __name__ == "__main__":
    main()
