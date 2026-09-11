"""Sparse-KV selection wired into the engine (Unit F, docs/design-sparse-kv.md).

Bounds scorer, the training-free day-1 path:

- every COMPLETE 16-token page carries Quest bounds (per full-attn plane, per KV
  head, a kmin/kmax pair in fp16), written once from the K the pool holds and
  kept by logical page index across demotions — scoring a cold page never reads
  its K back;
- each tick the own span is decoded/prefilled dense: decode's own is the trailing
  8-page window, a prefill chunk's own is the pages it writes;
- every full-attn SOURCE layer scores the row's own queries against earlier
  complete pages (max pooled over a chunk's queries, top-k union the forced 8
  pre-chunk pages), and the group's other layers reuse that one selection;
- selected cold pages promote through PagedKvPool before attention, dropped live
  pages demote after the forward (#500 seam);
- paged_attention is handed ONE packed ``[selected earlier ; own]`` block table
  with ``seq_len = n_sel*16 + own_len``; the slot-causal kernel needs no change
  because every selected page is a complete earlier page and
  ``own_len - sq == q_start - own_offsets``. RefBackend is the CPU
  parity oracle; cc's sm70 cell matches it without a new kernel (sm90 pending).

``scorer="index"`` is the later learned-indexer PR and refuses here.
"""

from __future__ import annotations

import torch
from tilerl_kernels.reference import select_pages
from torch import Tensor

from .kv_cache import BLOCK_TOKENS
from .sparse_index import WINDOW_PAGES, index_source_groups

#: block-table ids are logical+1 so selected logical page 0 is not the 0 pad.
_SENTINEL = 1


def group_map(cfg) -> tuple[list[int], dict[int, int]]:
    """Full-attn PLANE indices -> (source planes, plane -> group). A group's
    layers reuse the source layer's selection. Tiny (<4 full-attn): each layer
    is its own source/group."""
    n = len(cfg.full_attn_layers)
    groups = index_source_groups(n)[1] if n >= 4 else [[j] for j in range(n)]
    src, of = [], {}
    for g, idxs in enumerate(groups):
        src.append(idxs[0])
        for j in idxs:
            of[j] = g
    return src, of


def page_bounds_one(k_page: Tensor) -> Tensor:
    """Quest bounds of one page from its K plane: ``[Hkv,BLOCK,D]`` ->
    ``[Hkv,2,D]`` (kmin,kmax), fp16 — one plane's slice of the stored bounds."""
    return torch.stack((k_page.amin(dim=1), k_page.amax(dim=1)), dim=1).to(torch.float16)


def quest_scores(q: Tensor, bounds: Tensor) -> Tensor:
    """Per-page Quest upper-bound scores, MAX-pooled over the span's queries.

    ``q`` [Tq,hq,D] one row's post-rope queries, ``bounds`` [Cp,Hkv,2,D] for
    that plane. Returns [Cp]: ``sum_h max_t sum_d max(q*kmin, q*kmax)`` — a page
    hot for ANY query in the chunk is selectable (the same query max-pool
    ``page_scores_for_selector`` uses). f32 on the CPU oracle."""
    t, hq, d = q.shape
    hkv = bounds.shape[1]
    qi = q.float().reshape(t, hkv, hq // hkv, d).mean(2)        # [Tq,Hkv,D]
    kmin, kmax = bounds.unbind(dim=2)                            # each [Cp,Hkv,D]
    per = torch.maximum(
        qi[:, None, :, :] * kmin[None, :, :, :],
        qi[:, None, :, :] * kmax[None, :, :, :],
    ).sum(dim=-1)                                                # [Tq,Cp,Hkv]
    return per.amax(dim=0).sum(dim=-1)                           # [Cp], heads merged


class SparseTracker:
    """Engine-scoped bounds store: ``bounds[req_id][logical page]`` is an fp16
    ``[n_full, Hkv, 2, D]`` tensor living independently of the KV pool, so it
    survives a page's demotion to the host."""

    def __init__(self, cfg, k_pages: int, scorer: str):
        if scorer != "bounds":
            raise NotImplementedError(
                f'sparse engine scorer {scorer!r}: Unit F wires only "bounds"; '
                "the learned indexer is the later PR")
        self.cfg = cfg
        self.k_pages = k_pages
        self.scorer = scorer
        self.src_planes, self.group_of = group_map(cfg)
        self.bounds: dict[int, dict[int, Tensor]] = {}
        #: resident private pages: req_id -> {logical page: physical block}
        self.resident: dict[int, dict[int, int]] = {}
        self.bytes_per_page = 0

    def attach(self, req_id: int) -> None:
        self.bounds.setdefault(req_id, {})
        self.resident.setdefault(req_id, {})

    def drop(self, req_id: int) -> None:
        self.bounds.pop(req_id, None)
        self.resident.pop(req_id, None)

    def set_bounds(self, req_id: int, page: int, b: Tensor) -> None:
        b = b.contiguous().to(torch.float16)
        if not self.bytes_per_page:
            self.bytes_per_page = b.numel() * b.element_size()
        self.bounds[req_id][page] = b

    def bounds_bytes(self) -> int:
        return sum(len(pages) * self.bytes_per_page for pages in self.bounds.values())


class SparseForward:
    """One tick's sparse geometry, attached to BatchKv. The engine builds it per
    row with a ``resolve`` closure that maps a candidate logical page to a
    resident physical block (promoting through the host tier on demand); the
    model calls :meth:`attention_args` at each full-attn plane.

    Attention gets ONE packed block table ``[selected earlier ; own pages]`` and
    ``seq_len = n_sel*16 + own_len``. The existing SLOT-CAUSAL kernel needs no
    sparse kwarg: every selected page is a complete earlier page whose 16 keys
    precede every query, and the own span's own causal mask folds into the slot
    test because ``own_len - sq == q_start - own_offsets`` (decode: the trailing
    8-page window; prefill: own starts at floor(prefill_from/16)). RefBackend is
    the CPU parity oracle; cc matches it token-for-token on sm70 (V100 3.2e-4).
    The sm90 parity run is pending a free card.

    Per row: ``cand`` earlier complete candidate pages; ``force_window`` forces
    the last N candidates (8 for a prefill chunk, 0 on decode where the window
    IS the own span); ``own`` the own span's logical pages.

    The packed table is ONE fixed-width tensor, ``k_pages + force_window + own``
    per row at most, allocated at construction and refilled each tick: a CUDA
    graph bakes the shape in, so the width is a tick constant (unselected slots
    stay 0; paged_attention derives the live count from per-row seq_len)."""

    def __init__(self, tracker: SparseTracker, rows: list[dict], device):
        self.tracker = tracker
        self.device = device
        self.rows = rows
        self.b = len(rows)
        self.n_groups = len(tracker.src_planes)
        self._chosen: dict[tuple[int, int], list[int]] = {}
        self._phys: dict[tuple[int, int], Tensor] = {}
        own_w = max(len(r["own"]) for r in rows)
        self.page_base = torch.zeros(self.b, dtype=torch.long, device=device)
        self.own_table = torch.zeros(self.b, own_w, dtype=torch.long, device=device)
        # The graph-captured width is constant PER VERIFY WIDTH, never per context:
        # a plain decode row (tq=1) is k_pages + 8-window; a verify row's chain can
        # cross one page boundary (W-1 <= 15), so pad its own region to 9. Unused
        # own slots sit after sel+own; attention derives the page count from
        # seq_len, so the trailing pad ids are never read. Prefill rows run eager
        # and take their actual own width.
        def own_bound(r) -> int:
            if not r["decoding"]:
                return len(r["own"])
            return WINDOW_PAGES + (1 if r["tq"] > 1 else 0)

        sel_w = max(
            tracker.k_pages + r["force_window"] + own_bound(r) for r in rows
        )
        self.table = torch.zeros(self.b, sel_w, dtype=torch.long, device=device)
        self.seq_len = torch.zeros(self.b, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            self.page_base[i] = r["own"][0]
            self.own_table[i, : len(r["own"])] = torch.tensor(
                [r["resolve"](p) for p in r["own"]])

    def _select(self, bi: int, plane: int, q: Tensor) -> Tensor:
        """Physical selected-block table [n] for row bi at this plane; a source
        plane scores and promotes, group-mates reuse the cached decision."""
        g = self.tracker.group_of[plane]
        key = (bi, g)
        if key in self._phys:
            return self._phys[key]
        r = self.rows[bi]
        cand = r["cand"]
        chosen: list[int] = []
        if cand:
            bounds = torch.stack(
                [self.tracker.bounds[r["req_id"]][p][plane] for p in cand])
            scores = quest_scores(q, bounds).reshape(1, 1, len(cand))
            # logical+1 ids keep real page 0 distinct from the right-pad 0.
            table = (torch.tensor(cand, device=self.device) + _SENTINEL
                     ).reshape(1, -1)
            sel = select_pages(
                table, torch.tensor([len(cand)], device=self.device),
                scores, self.tracker.k_pages, n_window=r["force_window"])[0, 0]
            chosen = [int(x) - _SENTINEL for x in sel.tolist() if int(x) != 0]
        phys = torch.tensor(
            [r["resolve"](p) for p in chosen], dtype=torch.long, device=self.device)
        self._chosen[key] = chosen
        self._phys[key] = phys
        return phys

    def attention_args(self, plane: int, q: Tensor) -> tuple[Tensor, Tensor]:
        """Packed ``[selected ; own]`` table ``[B,W]`` and per-row packed
        ``seq_len = n_sel*16 + own_len`` for this plane's paged_attention. The
        table is the construction-width buffer refilled in place (graph-safe)."""
        for bi, r in enumerate(self.rows):
            sel = self._select(bi, plane, q[bi, : r["tq"]])
            own_n = len(r["own"])
            self.table[bi].zero_()
            self.table[bi, : sel.shape[0]] = sel
            self.table[bi, sel.shape[0] : sel.shape[0] + own_n] = self.own_table[bi, :own_n]
            self.seq_len[bi] = sel.shape[0] * BLOCK_TOKENS + int(r["own_len"])
        return self.table, self.seq_len
