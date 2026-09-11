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
from torch import Tensor

from .kv_cache import BLOCK_TOKENS
from .sparse_index import index_source_groups

#: block-table ids are logical+1 so selected logical page 0 is not the 0 pad.
_SENTINEL = 1
#: candidate pages scored per chunk so the f32 scoring intermediates stay bounded
_SCORE_PAGE_CHUNK = 64


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
    ``page_scores_for_selector`` uses). f32 on the CPU oracle.

    Candidate pages are scored in chunks: the unchunked f32 intermediates are
    Tq*Cp*Hkv*D each for kmin and kmax — 4.2 GiB at Tq=512, Cp=2048 on a V100
    whose whole headroom is ~4 GiB, and it grows with context. max-over-query
    and sum-over-head/dim commute with a split over pages, so chunking is exact;
    64 pages keeps the two operands under ~270 MiB regardless of context."""
    t, hq, d = q.shape
    hkv = bounds.shape[1]
    qi = q.float().reshape(t, hkv, hq // hkv, d).mean(2)        # [Tq,Hkv,D]
    kmin, kmax = bounds.unbind(dim=2)                            # each [Cp,Hkv,D]
    cp = bounds.shape[0]
    out = qi.new_empty(cp)
    for c0 in range(0, cp, _SCORE_PAGE_CHUNK):
        sl = slice(c0, c0 + _SCORE_PAGE_CHUNK)
        qb = qi[:, None, :, :]                                   # [Tq,1,Hkv,D]
        per = torch.maximum(qb * kmin[None, sl], qb * kmax[None, sl]).sum(dim=-1)
        out[sl] = per.amax(dim=0).sum(dim=-1)                    # [b], heads merged
    return out


def quest_scores_batched(q: Tensor, bounds: Tensor) -> Tensor:
    """Batched form of :func:`quest_scores` for the captured decode tick:
    ``q`` [B,Tq,hq,D], ``bounds`` [B,Cp,Hkv,2,D] -> scores [B,Cp], chunked over
    pages exactly like the single-row form (same split, same commute)."""
    b, t, hq, d = q.shape
    hkv = bounds.shape[2]
    qi = q.float().reshape(b, t, hkv, hq // hkv, d).mean(3)      # [B,Tq,Hkv,D]
    kmin, kmax = bounds.unbind(dim=3)                            # each [B,Cp,Hkv,D]
    cp = bounds.shape[1]
    out = q.new_empty(b, cp)
    for c0 in range(0, cp, _SCORE_PAGE_CHUNK):
        sl = slice(c0, c0 + _SCORE_PAGE_CHUNK)
        qb = qi[:, :, None, :, :]                                # [B,Tq,1,Hkv,D]
        per = torch.maximum(qb * kmin[:, None, sl], qb * kmax[:, None, sl]).sum(-1)
        out[:, sl] = per.amax(dim=1).sum(dim=-1)                 # [B,b]
    return out


def select_members(scores: Tensor, n_cand: Tensor, k_pages: int,
                   n_window: Tensor) -> Tensor:
    """Top-k UNION forced-window membership, pure device ops (no host sync —
    this runs inside the captured decode tick).

    ``scores`` [B,Cp] with padding positions already -inf-able, ``n_cand`` [B]
    valid candidate count, ``n_window`` [B] forced trailing pages per row.
    Returns bool member [B,Cp] (sequence positions kept, in candidate order)."""
    b, cp = scores.shape
    pos = torch.arange(cp, device=scores.device)
    valid = pos[None, :] < n_cand[:, None]
    k = min(k_pages, cp)
    member = torch.zeros_like(scores, dtype=torch.bool)
    member.scatter_(
        1, torch.topk(scores.masked_fill(~valid, float("-inf")), k, dim=1).indices,
        True)
    member &= valid  # a row with <k valid candidates must not mark padding
    # forced trailing window; decode rows carry n_window=0 so their mask is empty
    # (unconditional tensor OR — no host branch, this runs under capture).
    window = pos[None, :] >= (n_cand[:, None] - n_window[:, None]).clamp_min(0)
    member |= window & valid & (n_window[:, None] > 0)
    return member


def order_members(member: Tensor, k_pages: int) -> tuple[Tensor, Tensor]:
    """Compact member POSITIONS (into the candidate axis) to a FIXED width k in
    sequence order, with no host read: returns ``(positions [B,k] padded to cp,
    n_sel [B])``. cumsum gives each member's output slot (the same compaction
    ``reference.select_pages`` uses). ``select_members`` never marks more than k
    members, so every member rank is a valid slot; padded slots hold ``cp`` and
    the gather caller masks them."""
    b, cp = member.shape
    k = min(k_pages, cp)
    n_sel = member.sum(dim=1)
    # Fixed-shape stable sort: members (key 0) before padding (key 1), ties keep
    # candidate order, so the first k columns are the chosen positions in sequence
    # order. argsort has a static [B,cp] output shape (unlike nonzero), so this is
    # graph-capture-safe; padded slots hold cp and the caller masks them.
    order = torch.sort(member.long(), dim=1, stable=True, descending=True).indices
    idx = torch.arange(cp, device=member.device).expand(b, cp)
    positions = idx.gather(1, order[:, :k])
    return positions, n_sel


class SparseTracker:
    """Engine-scoped bounds store, living independently of the KV pool so a
    page's bounds survive its demotion to the host: one preallocated fp16
    tensor per request, ``bounds_t[rid] = [cap, n_full, Hkv, 2, D]`` grown by
    doubling, not one small tensor per page in a dict — at 128k a per-page
    dict forced :meth:`SparseForward._select` to ``torch.stack`` ~8192 tensors
    x4 planes every tick (host-bound). A logical page addresses its row
    directly; ``bounds_valid`` marks written rows (a promoted candidate skips
    its write on the tick it is promoted)."""

    #: initial page-row capacity of a request's bounds tensor
    INIT_CAP = 64

    def __init__(self, cfg, k_pages: int, scorer: str):
        if scorer != "bounds":
            raise NotImplementedError(
                f'sparse engine scorer {scorer!r}: Unit F wires only "bounds"; '
                "the learned indexer is the later PR")
        self.cfg = cfg
        self.k_pages = k_pages
        self.scorer = scorer
        self.src_planes, self.group_of = group_map(cfg)
        self.n_full = len(self.src_planes)
        self.hkv = cfg.num_kv_heads
        self.dim = cfg.head_dim
        #: [cap, n_full, Hkv, 2, D] fp16 per request + per-row-valid mask/count
        self.bounds_t: dict[int, Tensor] = {}
        self.bounds_valid: dict[int, Tensor] = {}
        self.bounds_count: dict[int, int] = {}
        #: device twin of ``resident``: logical page -> physical block, -1 if not
        #: resident. The captured decode tick gathers selected blocks with one
        #: index_select instead of a host resolve loop; the dict stays the owner
        #: of residency and this mirrors it one scalar write per resolve/demote.
        self.l2p_t: dict[int, Tensor] = {}
        #: resident private pages: req_id -> {logical page: physical block}
        self.resident: dict[int, dict[int, int]] = {}
        self.bytes_per_page = 0

    @property
    def _device(self):
        return (next(iter(self.bounds_t.values())).device
                if self.bounds_t else torch.device("cpu"))

    def attach(self, req_id: int) -> None:
        dev = self._device
        self.bounds_t[req_id] = torch.empty(
            (self.INIT_CAP, self.n_full, self.hkv, 2, self.dim),
            dtype=torch.float16, device=dev)
        self.bounds_valid[req_id] = torch.zeros(self.INIT_CAP, dtype=torch.bool, device=dev)
        self.l2p_t[req_id] = torch.full(
            (self.INIT_CAP,), -1, dtype=torch.long, device=dev)
        self.bounds_count[req_id] = 0
        self.resident.setdefault(req_id, {})

    def drop(self, req_id: int) -> None:
        self.bounds_t.pop(req_id, None)
        self.bounds_valid.pop(req_id, None)
        self.l2p_t.pop(req_id, None)
        self.bounds_count.pop(req_id, None)
        self.resident.pop(req_id, None)

    def _grow(self, rid: int, need: int) -> None:
        t = self.bounds_t[rid]
        cap = t.shape[0]
        new_cap = max(need, cap * 2)
        nt = torch.empty((new_cap, *t.shape[1:]), dtype=t.dtype, device=t.device)
        nt[:cap] = t
        nv = torch.zeros(new_cap, dtype=torch.bool, device=t.device)
        nv[:cap] = self.bounds_valid[rid]
        nl = torch.full((new_cap,), -1, dtype=torch.long, device=t.device)
        nl[:cap] = self.l2p_t[rid]
        self.bounds_t[rid] = nt
        self.bounds_valid[rid] = nv
        self.l2p_t[rid] = nl

    def map_resident(self, rid: int, page: int, phys: int) -> None:
        """Mirror a resolve into the device l2p (grows if a new own page passed
        the bounds tensor's capacity)."""
        if page >= self.l2p_t[rid].shape[0]:
            self._grow(rid, page + 1)
        self.l2p_t[rid][page] = phys

    def map_evict(self, rid: int, page: int) -> None:
        self.l2p_t[rid][page] = -1

    def set_bounds(self, req_id: int, page: int, b: Tensor) -> None:
        b = b.contiguous().to(torch.float16)
        if page >= self.bounds_t[req_id].shape[0]:
            self._grow(req_id, page + 1)
        self.bounds_t[req_id][page] = b
        self.bounds_valid[req_id][page] = True
        if page >= self.bounds_count[req_id]:
            self.bounds_count[req_id] = page + 1
        if not self.bytes_per_page:
            # one plane row [Hkv,2,D], x n_full planes
            self.bytes_per_page = b[0].numel() * b.element_size() * self.n_full

    def has_bounds(self, req_id: int, page: int) -> bool:
        v = self.bounds_valid.get(req_id)
        return v is not None and page < v.shape[0] and bool(v[page])

    def bounds_rows(self, req_id: int, pages: list[int]) -> Tensor:
        """Bounds of ``pages`` across every source plane in ``pages`` order:
        ``[n_pages, n_full, Hkv, 2, D]`` via ONE index_select (no Python stack)."""
        idx = torch.as_tensor(pages, dtype=torch.long, device=self.bounds_t[req_id].device)
        return self.bounds_t[req_id].index_select(0, idx)

    def bounds_bytes(self) -> int:
        return sum(int(v.sum()) * self.bytes_per_page for v in self.bounds_valid.values())


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
    IS the own span); ``own`` the own span's logical pages."""

    def __init__(self, tracker: SparseTracker, rows: list[dict], device,
                 device_select: bool = False):
        self.tracker = tracker
        self.device = device
        self.rows = rows
        self.b = len(rows)
        self.n_groups = len(tracker.src_planes)
        self.device_select = device_select
        self._chosen: dict[tuple[int, int], list[int]] = {}
        self._phys: dict[tuple[int, int], Tensor] = {}
        own_w = max(len(r["own"]) for r in rows)
        self.own_w = own_w
        self.page_base = torch.zeros(self.b, dtype=torch.long, device=device)
        self.own_table = torch.zeros(self.b, own_w, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            self.page_base[i] = r["own"][0]
            self.own_table[i, : len(r["own"])] = torch.tensor(
                [r["resolve"](p) for p in r["own"]])
        if device_select:
            self._init_device_tables()

    def _init_device_tables(self) -> None:
        """Fixed-width, capture-ready per-tick buffers (built once pre-forward;
        replay copies into them). Candidate/own logical indices and the packed
        own_len are static for the tick; only KV contents and q move."""
        cmax = max((len(r["cand"]) for r in self.rows), default=0)
        self.cmax = cmax
        self.cand_idx = torch.zeros(self.b, cmax, dtype=torch.long, device=self.device)
        self.own_log = torch.zeros(self.b, self.own_w, dtype=torch.long, device=self.device)
        self.own_valid = torch.zeros(self.b, self.own_w, dtype=torch.bool, device=self.device)
        self.n_cand = torch.zeros(self.b, dtype=torch.long, device=self.device)
        self.win = torch.zeros(self.b, dtype=torch.long, device=self.device)
        self.own_len_t = torch.zeros(self.b, dtype=torch.long, device=self.device)
        for bi, r in enumerate(self.rows):
            nc = len(r["cand"])
            if nc:
                self.cand_idx[bi, :nc] = torch.tensor(r["cand"], device=self.device)
            no = len(r["own"])
            self.own_log[bi, :no] = torch.tensor(r["own"], device=self.device)
            self.own_valid[bi, :no] = True
            self.n_cand[bi] = nc
            self.win[bi] = r["force_window"]
            self.own_len_t[bi] = r["own_len"]
        #: cached per-group device outputs: phys [B,k] (0 pad), chosen logical [B,k]
        self._dphys: dict[int, Tensor] = {}
        self._dchosen: dict[int, Tensor] = {}
        self._dnsel: dict[int, Tensor] = {}

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
            # one index_select gathers all candidate rows, then the plane slice;
            # no torch.stack over up-to-8192 per-page tensors per tick.
            bounds = self.tracker.bounds_rows(r["req_id"], cand)[:, plane]
            scores = quest_scores(q, bounds).reshape(1, 1, len(cand))
            # logical+1 ids keep real page 0 distinct from the right-pad 0.
            table = (torch.tensor(cand, device=self.device) + _SENTINEL
                     ).reshape(1, -1)
            from tilerl_kernels.reference import select_pages

            sel = select_pages(
                table, torch.tensor([len(cand)], device=self.device),
                scores, self.tracker.k_pages, n_window=r["force_window"])[0, 0]
            chosen = [int(x) - _SENTINEL for x in sel.tolist() if int(x) != 0]
        r["reserved"].update(chosen)  # protect this tick's picks from pin eviction
        phys = torch.tensor(
            [r["resolve"](p) for p in chosen], dtype=torch.long, device=self.device)
        self._chosen[key] = chosen
        self._phys[key] = phys
        return phys

    def _select_device(self, plane: int, q: Tensor):
        """Whole-batch selection for one source GROUP as pure device ops — no
        ``.tolist()``/``.item()``, no per-row ``torch.tensor``, no host sync (the
        captured decode tick). Gathers bounds once, scores all B rows, top-k UNION
        window membership, compacts to a FIXED k width in sequence order, and maps
        chosen logical pages to physical blocks via the device l2p. A gathered
        -1 (a cold, non-resident pick) is masked to pad 0: the engine must only
        reach here with the selection RESIDENT (the pin steady state); a tick that
        would promote runs the eager path instead. Group-mates reuse the cache."""
        g = self.tracker.group_of[plane]
        if g in self._dphys:
            return self._dphys[g], self._dnsel[g]
        k = min(self.tracker.k_pages, self.cmax)
        if self.cmax == 0:
            phys = torch.zeros(self.b, 0, dtype=torch.long, device=self.device)
            nsel = torch.zeros(self.b, dtype=torch.long, device=self.device)
            chosen = torch.zeros(self.b, 0, dtype=torch.long, device=self.device)
            self._dphys[g] = phys
            self._dnsel[g] = nsel
            self._dchosen[g] = chosen
            return phys, nsel
        # gather candidate bounds per row (B fixed index_selects — constant in the
        # context length), then this plane's slice: [B,Cmax,Hkv,2,D].
        br = [
            self.tracker.bounds_rows(self.rows[bi]["req_id"], self.rows[bi]["cand"])
            [:, plane]
            for bi in range(self.b)]
        # rows have equal Cmax candidate slots; pad short rows' bounds along axis 1.
        bounds = torch.stack([
            torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, self.cmax - x.shape[0]))
            for x in br])
        scores = quest_scores_batched(q, bounds)            # [B,Cmax]
        member = select_members(scores, self.n_cand, self.tracker.k_pages, self.win)
        positions, nsel = order_members(member, self.tracker.k_pages)  # [B,k]
        valid = torch.arange(k, device=self.device)[None, :] < nsel[:, None]
        safe_pos = positions.clamp_max(self.cmax - 1)
        chosen = self.cand_idx.gather(1, safe_pos)          # logical pages [B,k]
        chosen = torch.where(valid, chosen, torch.zeros_like(chosen))
        phys = torch.stack([
            self.tracker.l2p_t[self.rows[bi]["req_id"]].index_select(0, chosen[bi])
            for bi in range(self.b)])                       # [B,k], -1 if cold
        # A gathered -1 (cold pick) maps to pad 0 here. The captured gather cannot
        # promote, so the ENGINE routes a tick here only in the pin steady state
        # with every pick resident; that residency check runs on the host before
        # capture, never in this tensor-only path.
        phys = torch.where(valid, phys.clamp_min(0), torch.zeros_like(phys))
        self._dphys[g] = phys
        self._dnsel[g] = nsel
        self._dchosen[g] = chosen
        return phys, nsel

    def selected_pages(self, bi: int) -> set[int]:
        """Union of this row's logical pages chosen across ALL source groups this
        tick, plus the own span. Every group's choice co-resides until finalize, so
        the cross-tick pin keeps this exact set and demotes only what left it."""
        pages = set(self.rows[bi]["own"])
        if self.device_select:
            for g in range(self.n_groups):
                ch, ns = self._dchosen.get(g), self._dnsel.get(g)
                if ch is not None:
                    pages.update(int(x) for x in ch[bi, : int(ns[bi])].tolist())
        else:
            for g in range(self.n_groups):
                pages.update(self._chosen.get((bi, g), ()))
        return pages

    def attention_args(self, plane: int, q: Tensor) -> tuple[Tensor, Tensor]:
        """Packed ``[selected ; own]`` table ``[B,W]`` and per-row packed
        ``seq_len = n_sel*16 + own_len`` for this plane's paged_attention."""
        if self.device_select:
            return self._attention_args_device(plane, q)
        packed, sl = [], []
        for bi, r in enumerate(self.rows):
            sel = self._select(bi, plane, q[bi, : r["tq"]])
            own_n = len(r["own"])
            packed.append(torch.cat((sel, self.own_table[bi, :own_n])))
            sl.append(sel.shape[0] * BLOCK_TOKENS + int(r["own_len"]))
        width = max(t.shape[0] for t in packed)
        table = torch.zeros(self.b, width, dtype=torch.long, device=self.device)
        for i, t in enumerate(packed):
            table[i, : t.shape[0]] = t
        return table, torch.tensor(sl, dtype=torch.long, device=self.device)

    def _attention_args_device(self, plane: int, q: Tensor) -> tuple[Tensor, Tensor]:
        """Fixed-width ``[selected k ; own window]`` table ``[B,k+own_w]`` and
        packed seq_len, both device tensors, one shape for the tick's life (the
        graph captures one bucket). Selected pages occupy the compact leading
        columns (padded 0 at the tail); each row's OWN follows at its own
        ``nsel`` offset, so the packed physical order is causal. Zero host syncs;
        seq_len already excludes the padding columns."""
        sel, nsel = self._select_device(plane, q)
        k = sel.shape[1]
        # own physical blocks via the device l2p (own was resolved pre-forward).
        own_phys = torch.stack([
            self.tracker.l2p_t[self.rows[bi]["req_id"]].index_select(0, self.own_log[bi])
            for bi in range(self.b)])
        table = torch.zeros(self.b, k + self.own_w, dtype=torch.long, device=self.device)
        table[:, :k] = sel
        col = nsel[:, None] + torch.arange(self.own_w, device=self.device)[None, :]
        # own_valid zeroes the physical pad beyond each row's own pages before scatter.
        own_phys = own_phys.masked_fill(~self.own_valid, 0)
        table.scatter_(1, col, own_phys)
        sl = nsel * BLOCK_TOKENS + self.own_len_t
        return table, sl

