#!/usr/bin/env python3
"""Fidelity-harness validity checks (a3 2026-09-12), prefill KL/top1 on seeded
positions. Two modes:

--tiny: CPU tiny model, self-contained (no corpus/checkpoint). Validates the
  HARNESS, not the 27B science:
  A continuity  k>=pages must be dense (KL ~0, top1 ~1)
  C controls    random128 and window-only must each differ from bounds and
                from each other; bounds beats both
  B set dump    bounds vs oracle selected sets at 3 chunks x all groups
  D table content  dense identity vs dense SCRAMBLED physical layout agree
(default 27B): same checks on one held span on a card.

Every arm's number prints and is flushed to JSON as it lands (never only at
end of job).
"""
import argparse
import json
from types import SimpleNamespace

import output_fidelity as F
import torch
from tilerl_kernels.reference import select_pages

from tilerl.kv_cache import BLOCK_TOKENS, PagedKvPool
from tilerl.sparse_engine import SparseForward, SparseTracker, quest_scores
from tilerl.sparse_index import WINDOW_PAGES


def make_pool(cfg, backend, device, n_pages, seed=None, dtype=None):
    npp = n_pages + F.FRESH_PAD
    pool = PagedKvPool(npp, cfg.num_kv_heads, cfg.head_dim, device=device,
                       layer_map=cfg.full_attn_layers,
                       dtype=dtype or getattr(backend, "io", torch.bfloat16))
    ph = [pool.alloc_block() for _ in range(npp)]
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(ph), generator=g).tolist()
        ph = [ph[i] for i in order]
    return pool, ph, F.new_state(cfg, device)


def kvdesc(pool, state, table, hi, tq, sf=None):
    return SimpleNamespace(
        dense=False, kv_pool=pool, block_table=table,
        seq_len=torch.tensor([hi], device=pool.k_pool.device),
        seq_q_lens=torch.tensor([tq], device=pool.k_pool.device),
        state_pool=state,
        state_slot=torch.zeros(1, dtype=torch.long, device=pool.k_pool.device),
        sparse=sf, page_base=None if sf is None else sf.page_base)


def row_dict(lo, hi, phys, tracker=None):
    own_first = lo // BLOCK_TOKENS
    own = list(range(own_first, (hi - 1) // BLOCK_TOKENS + 1))
    cand = list(range(own_first))
    if tracker is not None:
        cand = [p for p in cand if p in tracker.bounds[0]]
    return dict(req_id=0, own=own, own_len=hi - own_first * BLOCK_TOKENS,
                q_start=lo, q_hi=hi, decoding=False, tq=hi - lo, cand=cand,
                force_window=WINDOW_PAGES, resolve=lambda p, phys=phys: phys[p])


class CaptureForward(SparseForward):
    """Records (chunk, group, set); custom scorer for random/window/oracle."""

    def __init__(self, tracker, rows, device, pool, phys, mode, dump, chunk_idx,
                 score_rows, check_chunks):
        super().__init__(tracker, rows, device)
        self._cpool, self._cphys = pool, phys
        self._mode, self._dump, self._ci = mode, dump, chunk_idx
        self._score_rows = score_rows
        self._check_chunks = check_chunks

    def _select(self, bi, plane, q, h=None):
        g = self.tracker.group_of[plane]
        key = (bi, g)
        if key in self._phys:
            return self._phys[key]
        r = self.rows[bi]
        cand = r["cand"]
        chosen: list[int] = []
        if cand:
            if self._mode == "bounds":
                b = torch.stack([self.tracker.bounds[0][p][plane] for p in cand])
                score = quest_scores(q, b)
            elif self._mode == "random":
                gen = torch.Generator(device=q.device).manual_seed(1000 + self._ci)
                score = torch.rand(len(cand), generator=gen, device=q.device)
            elif self._mode == "window":
                score = torch.full((len(cand),), -1e9, device=q.device)
            elif self._mode == "oracle":
                rows = self._score_rows.get((0, r["q_start"], r["q_hi"]))
                if rows is None:
                    rows = torch.arange(q.shape[0], device=q.device)
                pages = cand + r["own"]
                mass = F.stream_mass(
                    q.index_select(0, rows), (r["q_start"] + rows).to(q.device),
                    lambda p, plane=plane, pages=pages:
                        self._cpool.k_pool[plane, self._cphys[pages[p]]]
                        if p < len(pages) else F._stop_page())
                score = mass[:, : len(cand)].amax(0)
            else:
                raise ValueError(self._mode)
            table = (torch.tensor(cand, device=self.device) + 1).reshape(1, -1)
            sel = select_pages(
                table, torch.tensor([len(cand)], device=self.device),
                score.reshape(1, 1, len(cand)), self.tracker.k_pages,
                n_window=r["force_window"])[0, 0]
            chosen = [int(x) - 1 for x in sel.tolist() if int(x) != 0]
            if self._ci in self._check_chunks:
                self._dump.setdefault(self._mode, {}).setdefault(self._ci, {})[g] = sorted(chosen)
        ph = torch.tensor([r["resolve"](p) for p in chosen], device=self.device)
        self._chosen[key] = chosen
        self._phys[key] = ph
        return ph


def metrics(teacher, student):
    kl = top = 0
    for a, b in zip(teacher, student):
        kl += float((a.exp() * (a - b)).sum())
        top += int(a.argmax() == b.argmax())
    n = max(1, len(teacher))
    return {"kl": round(kl / n, 5), "top1": round(top / n, 4)}


def replay(model, backend, cfg, ids, t, qpos, mode, k, seed_phys, dump,
           check_chunks, chunk):
    pool, phys, state = make_pool(cfg, backend, backend.device, t // BLOCK_TOKENS,
                                  seed_phys)
    tracker = None
    if mode != "dense":
        tracker = SparseTracker(cfg, k, "bounds")
        tracker.attach(0)
    score_rows = {}
    lps = []
    for ci, lo in enumerate(range(0, t, chunk)):
        hi = min(lo + chunk, t)
        locs = (qpos[(qpos >= lo) & (qpos < hi)] - lo).tolist()
        if not locs:
            locs = torch.linspace(0, hi - lo - 1, 4).long().tolist()
        score_rows[(0, lo, hi)] = torch.tensor(locs, device=backend.device)
    for ci, lo in enumerate(range(0, t, chunk)):
        hi = min(lo + chunk, t)
        if mode == "dense":
            kv = kvdesc(pool, state,
                        torch.tensor([phys[: hi // BLOCK_TOKENS]],
                                     device=backend.device), hi, hi - lo)
            sf = None
        else:
            r = row_dict(lo, hi, phys, tracker)
            sf = CaptureForward(tracker, [r], backend.device, pool, phys, mode,
                                dump, ci, score_rows, check_chunks)
            kv = kvdesc(pool, state, sf.own_table, hi, hi - lo, sf)
        hidden = []
        model.forward(ids[:, lo:hi], torch.arange(lo, hi, device=backend.device),
                      kv, backend, hidden_out=hidden, last_only=True)
        if tracker is not None:
            F.finalize_bounds(tracker, pool, lambda p, phys=phys: phys[p], hi)
        locs = (qpos[(qpos >= lo) & (qpos < hi)] - lo)
        if len(locs):
            h = hidden[0][0].index_select(0, locs)
            lg = F.head_at(model, backend, cfg, h,
                           torch.arange(h.shape[0], device=backend.device)).float()
            lps.extend(torch.log_softmax(lg, -1).cpu())
    return lps


def run_checks(model, cfg, backend, ids, t, qpos, arms, out, check_chunks,
               chunk):
    print("arm dense identity", flush=True)
    teacher = replay(model, backend, cfg, ids, t, qpos, "dense", 0, None, {},
                     check_chunks, chunk)
    result = {"checks": {}, "set_equality": {}}
    print("arm dense scrambled", flush=True)
    ds = replay(model, backend, cfg, ids, t, qpos, "dense", 0, 7, {},
                check_chunks, chunk)
    result["checks"]["dense_scrambled_vs_identity"] = metrics(teacher, ds)
    dump = {}
    for name, mode, k in arms:
        print("arm", name, flush=True)
        lp = replay(model, backend, cfg, ids, t, qpos, mode, k, None, dump,
                    check_chunks, chunk)
        result["checks"][name] = metrics(teacher, lp)
        print("METRIC", name, result["checks"][name], flush=True)
        with open(out, "w") as fh:
            json.dump(result, fh)
    for ci in check_chunks:
        for g in sorted(dump.get("bounds", {}).get(ci, {})):
            b = dump["bounds"][ci][g]
            o = dump.get("oracle", {}).get(ci, {}).get(g, [])
            result["set_equality"][f"chunk{ci}_g{g}"] = {
                "bounds_n": len(b), "oracle_n": len(o), "identical": b == o,
                "overlap": len(set(b) & set(o))}
    with open(out, "w") as fh:
            json.dump(result, fh)
    print("RESULT", json.dumps(result["checks"]), flush=True)
    print("SETEQ", json.dumps(result["set_equality"]), flush=True)
    return result


def tiny():
    from tilerl_kernels.backend import get_backend

    from tilerl import config as config_mod
    from tilerl.model import build_random

    cfg = config_mod.tiny(4096)
    backend = get_backend()
    model = build_random(cfg, seed=11)
    torch.manual_seed(0)
    # chunk MUST be >= WINDOW_PAGES pages: the prefill path forces the 8
    # pre-chunk pages, and a smaller chunk overlaps that window with the own
    # span (duplicate/missing frames). Use 8 pages = the engine minimum.
    chunk = WINDOW_PAGES * BLOCK_TOKENS
    t = 48 * BLOCK_TOKENS
    ids = torch.randint(0, cfg.vocab_size, (1, t))
    qpos = torch.tensor(sorted({t - 2, t - 20, t // 2, t // 4, BLOCK_TOKENS + 1}))
    n_pages = t // BLOCK_TOKENS
    half = n_pages // 2 - WINDOW_PAGES
    # continuity at full coverage, slope at a genuinely partial k, and controls
    arms = [("fullk_bounds", "bounds", n_pages - WINDOW_PAGES),
            ("bounds_half", "bounds", half),
            ("random_half", "random", half),
            ("window_only", "window", half),
            ("oracle_half", "oracle", half)]
    return run_checks(model, cfg, backend, ids, t, qpos, arms,
                      "/tmp/fidelity-checks-tiny.json", (0, 2, 5), chunk)


def main27b(args):
    from pathlib import Path

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model

    backend = get_backend()
    _c, model = _build_model("qwen38-27b", seed=0, keep_master=False, backend=backend)
    cfg = model.cfg
    span_path = Path(args.corpus) / f"held_{args.ctx}.jsonl"
    with open(span_path) as fh:
        r = json.loads(fh.readlines()[args.span])
    ids = torch.tensor([r["ids"]], device=backend.device)
    t = args.ctx
    qpos = F.sample_positions(t, args.span).to(backend.device)
    n_pages = t // BLOCK_TOKENS
    arms = [("fullk_bounds", "bounds", n_pages - WINDOW_PAGES),
            ("bounds1024", "bounds", 1024),
            ("bounds128", "bounds", 128),
            ("random128", "random", 128),
            ("window_only", "window", 128),
            ("oracle1024", "oracle", 1024)]
    run_checks(model, cfg, backend, ids, t, qpos, arms, args.out,
               (0, 8, 63), F.CHUNK)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?")
    ap.add_argument("--span", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--out", default="/work/fidelity-checks.json")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    tiny() if a.tiny else main27b(a)
