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

``scorer="index"`` (this PR): the learned indexer.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from tilerl_kernels.reference import select_pages
from torch import Tensor

from .kv_cache import BLOCK_TOKENS
from .sparse_index import WINDOW_PAGES, index_source_groups

#: block-table ids are logical+1 so selected logical page 0 is not the 0 pad.
_SENTINEL = 1
#: candidate pages scored per chunk so the f32 scoring intermediates stay bounded
_SCORE_PAGE_CHUNK = 64

#: Decode ticks between eager full-candidate re-selection (refresh) when device
#: selection is on. Device ticks score only among RESIDENT candidates (zero
#: syncs, graph-capturable), so a page cold/on-SSD that the Quest top-k now
#: wants is brought into the hot set at the next refresh: every R decode ticks
#: the engine runs the eager path, which scores ALL candidates and promotes what
#: it names through the pin. R=1 makes every decode tick a refresh (token-equal
#: to eager sparse, no staleness); larger R trades <=R-tick selection staleness
#: for more captured ticks. Quest decode selections drift slowly, so 8 is the
#: default; measure output fidelity at R=1 vs R with the k-fidelity table.
SPARSE_REFRESH_TICKS = 8


def group_map(cfg) -> tuple[list[int], dict[int, int]]:
    """Full-attn PLANE indices -> (source planes, plane -> group). A group's
    layers reuse the source layer's selection. Tiny (<4 full-attn): each layer
    is its own source/group."""
    n = len(cfg.full_attn_layers)
    if n >= 4:
        _, groups = index_source_groups(n)
    else:
        groups = [[j] for j in range(n)]
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
                   n_window: Tensor, eligible: Tensor | None = None) -> Tensor:
    """Top-k UNION forced-window membership, pure device ops (no host sync —
    this runs inside the captured decode tick).

    ``scores`` [B,Cp] with padding positions already -inf-able, ``n_cand`` [B]
    valid candidate count, ``n_window`` [B] forced trailing pages per row,
    ``eligible`` optional bool [B,Cp] restricting which valid candidates may be
    chosen (the device path passes the residency mask l2p>=0). Returns bool
    member [B,Cp] (sequence positions kept, in candidate order)."""
    b, cp = scores.shape
    pos = torch.arange(cp, device=scores.device)
    valid = pos[None, :] < n_cand[:, None]
    if eligible is not None:
        valid = valid & eligible
    k = min(k_pages, cp)
    member = torch.zeros_like(scores, dtype=torch.bool)
    member.scatter_(
        1, torch.topk(scores.masked_fill(~valid, float("-inf")), k, dim=1).indices,
        True)
    member &= valid  # a row with <k eligible candidates must not mark padding
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


class _BoundsView:
    """Read-only ``page -> bound`` face over one request's contiguous
    ``bounds_t``, so prefix publish can keep its mapping-shaped contract
    without a second per-page dict. Each lookup clones — the published entry
    owns its tensor."""

    __slots__ = ("tr", "rid")

    def __init__(self, tr, rid: int):
        self.tr, self.rid = tr, rid

    def __contains__(self, page) -> bool:
        return self.tr.has_bounds(self.rid, int(page))

    def __getitem__(self, page) -> Tensor:
        page = int(page)
        if not self.tr.has_bounds(self.rid, page):
            raise KeyError(page)
        return self.tr.bounds_one(self.rid, page)


class SparseTracker:
    """Engine-scoped scorer store, independent of the KV pool so a page's state
    survives its demotion to host:

    - ``bounds`` scorer: one preallocated fp16 tensor per request,
      ``bounds_t[rid] = [n_full, cap, Hkv, 2, D]`` (PLANE first) grown by
      doubling, not one small tensor per page in a dict (at 128k a per-page dict
      forced _select to torch.stack ~8192 tensors x4 planes every tick). Plane-first
      lets scoring slice the plane before gathering the cap axis, so a call moves
      one plane (64 MiB at 256k), not all 16 (1 GiB). A logical page addresses
      its cap row directly; bounds_valid marks written rows.
    - ``keys`` (scorer="index"): req -> page -> fp8 indexer keys
      ``[n_src, Hkv, di]`` plus an f32 scale per key ``[n_src, Hkv]`` (one scale
      over di=128), projected from the page's mean K at append time by the
      learned ik_weight. Indexer-Q is projected live from each source layer's H.

    The untrained day-0 path inits small iq/ik weights deterministically; the
    learned weights are a later load. At full k both scorers select every
    candidate page and are token-identical to dense."""

    #: initial page-row capacity of a request's bounds tensor
    INIT_CAP = 64

    def __init__(self, cfg, k_pages: int, scorer: str, weights: dict | None = None,
                 device=None):
        if scorer not in ("bounds", "index"):
            raise NotImplementedError(
                f'sparse engine scorer {scorer!r}: want "bounds" or "index"')
        self.cfg = cfg
        self.k_pages = k_pages
        self.scorer = scorer
        #: Device the preallocated bounds/l2p tensors live on. Must be the
        #: BACKEND device: attach runs before any bound exists, so inferring it
        #: from existing tensors put the first request's bounds_t/l2p_t on CPU on
        #: a GPU build, and the next GPU q scored against CPU bounds raised a
        #: cross-device error (289b28ca GPU regression; CPU CI cannot see it).
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.src_planes, self.group_of = group_map(cfg)
        self.src_index = {plane: j for j, plane in enumerate(self.src_planes)}
        #: number of full-attn PLANES (= PagedKvPool.num_layers): bounds are
        #: stored and scored per plane; only the SELECTION is shared per group.
        #: Not len(src_planes) (the source count, /4) — that dim fit tiny (1
        #: full-attn plane) but the 27B's 16-plane finalize write crashes it.
        self.n_full = len(cfg.full_attn_layers)
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
        #: logical pages adopted from a shared prefix entry: req_id -> {page: content key}
        self.shared: dict[int, dict[int, int]] = {}
        #: host-blob-backed prefix index (None = the NoPrefixStore stopgap)
        self.prefix: SparsePrefixCache | None = None
        #: False when the caller explicitly chose NoPrefixStore (sharing disabled)
        self.sharing_enabled = True
        #: last tick's served selection per req/group (set after each forward)
        self.last_selected: dict[int, dict[int, list[int]]] = {}
        self.bytes_per_page = 0
        if scorer == "bounds":
            self.keys = None
            self.iq = self.ik = None
        else:
            from .sparse_index import INDEX_HEADS

            # The shipped model has num_kv_heads==4==ih; the tiny cell has 2 KV
            # heads, so it runs ih=2 (head grouping still divides), matching the
            # indexer-weight init used by the warm-up.
            ih = min(INDEX_HEADS, cfg.num_kv_heads)
            self.ih = ih
            self.keys: dict[int, dict[int, tuple[Tensor, Tensor]]] = {}
            di = min(16, cfg.head_dim)          # tiny cell di=16; served di=128
            gen = torch.Generator().manual_seed(0)
            scale = 0.1

            def _w(shape):
                return scale * torch.randn(*shape, generator=gen)

            if weights is None:
                iq = _w((ih, cfg.hidden_size, di))
                ik = _w((ih, cfg.head_dim, di))
            else:
                iq, ik = weights["iq"], weights["ik"]
            self.iq, self.ik = iq, ik
            self.di = int(ik.shape[-1])
            self._fp8 = torch.float8_e4m3fn
            # keys pages*n_src*ih*di fp8 + one f32 scale per (page,src,head).
            self.bytes_per_page = (
                len(self.src_planes) * ih * di + len(self.src_planes) * ih * 4)

    def attach(self, req_id: int) -> None:
        dev = self.device
        self.bounds_t[req_id] = torch.empty(
            (self.n_full, self.INIT_CAP, self.hkv, 2, self.dim),
            dtype=torch.float16, device=dev)
        self.bounds_valid[req_id] = torch.zeros(self.INIT_CAP, dtype=torch.bool, device=dev)
        self.l2p_t[req_id] = torch.full(
            (self.INIT_CAP,), -1, dtype=torch.long, device=dev)
        self.bounds_count[req_id] = 0
        if self.scorer == "index":
            self.keys.setdefault(req_id, {})
        self.resident.setdefault(req_id, {})
        self.shared.setdefault(req_id, {})

    def drop(self, req_id: int) -> None:
        self.bounds_t.pop(req_id, None)
        self.bounds_valid.pop(req_id, None)
        self.l2p_t.pop(req_id, None)
        self.bounds_count.pop(req_id, None)
        if self.scorer == "index":
            self.keys.pop(req_id, None)
        self.last_selected.pop(req_id, None)
        self.resident.pop(req_id, None)
        self.shared.pop(req_id, None)
        if self.prefix is not None:
            self.prefix.drop_request(req_id)

    def _grow(self, rid: int, need: int) -> None:
        t = self.bounds_t[rid]
        cap = t.shape[1]
        new_cap = max(need, cap * 2)
        nt = torch.empty((t.shape[0], new_cap, *t.shape[2:]),
                         dtype=t.dtype, device=t.device)
        nt[:, :cap] = t
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
        if page >= self.bounds_t[req_id].shape[1]:
            self._grow(req_id, page + 1)
        self.bounds_t[req_id][:, page] = b
        self.bounds_valid[req_id][page] = True
        if page >= self.bounds_count[req_id]:
            self.bounds_count[req_id] = page + 1
        if not self.bytes_per_page:
            # one plane row [Hkv,2,D], x n_full planes
            self.bytes_per_page = b[0].numel() * b.element_size() * self.n_full

    def bounds_view(self, req_id: int) -> _BoundsView:
        return _BoundsView(self, req_id)

    def has_bounds(self, req_id: int, page: int) -> bool:
        v = self.bounds_valid.get(req_id)
        return v is not None and page < v.shape[0] and bool(v[page])

    def bounds_rows(self, req_id: int, plane: int, pages: list[int]) -> Tensor:
        """One plane's bounds over ``pages`` in pages order:
        ``[n_pages, Hkv, 2, D]`` via ONE index_select on the CAP axis. The plane
        is sliced BEFORE the gather so a 256k scoring call moves one plane
        (64 MiB), not all 16 (1 GiB) to read 1/16 of it."""
        t = self.bounds_t[req_id]
        idx = torch.as_tensor(pages, dtype=torch.long, device=t.device)
        return t[plane].index_select(0, idx)

    def bounds_one(self, req_id: int, page: int) -> Tensor:
        """One page's bounds ``[n_full,Hkv,2,D]`` (prefix-publish clone face)."""
        return self.bounds_t[req_id][:, page].clone()

    def set_index_keys(self, req_id: int, page: int, keys: Tensor,
                       scales: Tensor) -> None:
        """Store one page's fp8 indexer keys ``[n_src,Hkv,di]`` and per-key f32
        scales ``[n_src,Hkv]``. Bytes counted from the stored face so stats'
        measured reconciles with memory.index_keys_bytes."""
        keys = keys.contiguous()
        scales = scales.contiguous()
        self.keys[req_id][page] = (keys, scales)

    def bounds_bytes(self) -> int:
        if self.scorer == "bounds":
            return sum(int(v.sum()) * self.bytes_per_page
                       for v in self.bounds_valid.values())
        return self.index_keys_bytes()

    def index_keys_bytes(self) -> int:
        return sum(len(keys) * self.bytes_per_page
                   for keys in self.keys.values()) if self.keys else 0



def project_index_page_keys(k_page_means: Tensor, ik: Tensor) -> tuple[Tensor, Tensor]:
    """Project one complete page's mean K per source plane into fp8 indexer keys.

    ``k_page_means`` [n_src,Hkv,D], ``ik`` [ih,D,di] -> fp8 [n_src,Hkv,di] plus
    one f32 scale per (source,head). The attention-head group is a single KV head
    here (ih==hkv on both tiny and the 27B), so project_page_keys' head mean is
    the identity; call it for the boundary casts. fp8 = the ledger's storage face
    (di=128 -> 128 payload B + 4 B scale per key)."""
    from .sparse_index import project_page_keys

    proj = project_page_keys(k_page_means[None], ik)[0]   # [n_src,Hkv,di], f32
    scale = proj.float().abs().amax(dim=-1).clamp_min(1e-12) / torch.finfo(
        torch.float8_e4m3fn).max                            # [n_src,Hkv]
    keys = (proj.float() / scale[..., None]).to(torch.float8_e4m3fn)
    return keys, scale.float()


def index_scores(q_h: Tensor, tracker: SparseTracker, req_id: int,
                 cand: list[int], g: int) -> Tensor:
    """V4.1 learned-indexer page scores for one row/source group over this
    tick's queries: ReLU(q.k)/sqrt(di) max-pooled over queries, heads summed.
    ``q_h`` is the layer INPUT H at the source plane ``[Tq,hidden]``; stored
    candidate keys are dequantized fp8. Returns [len(cand)] f32."""
    src = tracker.src_planes[g]
    iq = torch.einsum("qd,hde->qhe", q_h.float().to(tracker.iq.device),
                      tracker.iq)                       # [Tq,ih,di]
    keys, scales = zip(*(tracker.keys[req_id][p] for p in cand))
    ik8 = torch.stack(keys)[:, tracker.src_index[src]]        # [Cp,ih,di]
    sc = torch.stack(scales)[:, tracker.src_index[src]]       # [Cp,ih]
    ikd = ik8.float() * sc[..., None]                         # dequant
    dots = torch.einsum("qhe,phe->qph", iq, ikd) * (tracker.di ** -0.5)
    return torch.relu(dots).amax(dim=0).sum(dim=-1)          # [Cp]


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

    def __init__(self, tracker: SparseTracker, rows: list[dict], device,
                 device_select: bool = False):
        self.tracker = tracker
        self.device = device
        self.rows = rows
        self.b = len(rows)
        self.n_groups = len(tracker.src_planes)
        #: bounds scores post-rope q only; index scoring also needs the
        #: full-precision post-input-norm hidden (model.py runs the extra norm
        #: only when this is true, so bounds pays no second rmsnorm).
        self.index_scorer = tracker.scorer == "index"
        self.device_select = device_select
        self._chosen: dict[tuple[int, int], list[int]] = {}
        self._phys: dict[tuple[int, int], Tensor] = {}
        own_w = max(len(r["own"]) for r in rows)
        self.own_w = own_w
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

    def _select(self, bi: int, plane: int, q: Tensor, h: Tensor | None) -> Tensor:
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
            if self.tracker.scorer == "bounds":
                # one index_select gathers this plane's candidate rows (the plane
                # is sliced inside bounds_rows, so only 1/16 of the bounds moves).
                bounds = self.tracker.bounds_rows(r["req_id"], plane, cand)
                scores = quest_scores(q, bounds).reshape(1, 1, len(cand))
            else:
                # learned indexer: score from the source plane's layer input h
                if plane != self.tracker.src_planes[g]:
                    raise KeyError("index scoring requested off a source plane")
                scores = index_scores(
                    h, self.tracker, r["req_id"], cand, g).reshape(1, 1, len(cand))
            # logical+1 ids keep real page 0 distinct from the right-pad 0.
            table = (torch.tensor(cand, device=self.device) + _SENTINEL
                     ).reshape(1, -1)
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
        # gather this plane's candidate bounds per row (B fixed index_selects,
        # constant in context length): [B,Cmax,Hkv,2,D]. The plane slice happens
        # inside bounds_rows, so each gather is one plane, not all n_full.
        br = [
            self.tracker.bounds_rows(self.rows[bi]["req_id"], plane,
                                     self.rows[bi]["cand"])
            for bi in range(self.b)]
        # rows have equal Cmax candidate slots; pad short rows' bounds along axis 1.
        bounds = torch.stack([
            torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, self.cmax - x.shape[0]))
            for x in br])
        scores = quest_scores_batched(q, bounds)            # [B,Cmax]
        # Score only RESIDENT candidates: l2p[page] >= 0. The captured path never
        # promotes, so a cold/SSD candidate is excluded from selection here rather
        # than chosen-and-dropped; the eager path re-selects over ALL candidates
        # (promoting what it names) every SPARSE_REFRESH_TICKS and on prefill,
        # refreshing the resident hot set. Every chosen page is resident BY
        # CONSTRUCTION, so the phys gather below cannot return -1 — no host sync.
        resident = torch.stack([
            self.tracker.l2p_t[self.rows[bi]["req_id"]]
            .index_select(0, self.cand_idx[bi]) >= 0
            for bi in range(self.b)])                       # [B,Cmax]
        member = select_members(scores, self.n_cand, self.tracker.k_pages,
                                self.win, resident)
        positions, nsel = order_members(member, self.tracker.k_pages)  # [B,k]
        valid = torch.arange(k, device=self.device)[None, :] < nsel[:, None]
        safe_pos = positions.clamp_max(self.cmax - 1)
        chosen = self.cand_idx.gather(1, safe_pos)          # logical pages [B,k]
        chosen = torch.where(valid, chosen, torch.zeros_like(chosen))
        phys = torch.stack([
            self.tracker.l2p_t[self.rows[bi]["req_id"]].index_select(0, chosen[bi])
            for bi in range(self.b)])                       # [B,k], always >= 0
        phys = torch.where(valid, phys, torch.zeros_like(phys))
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

    def attention_args(self, plane: int, q: Tensor,
                       h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Packed ``[selected ; own]`` table ``[B,W]`` and per-row packed
        ``seq_len = n_sel*16 + own_len`` for this plane's paged_attention."""
        if self.device_select:
            return self._attention_args_device(plane, q)
        packed, sl = [], []
        for bi, r in enumerate(self.rows):
            sel = self._select(bi, plane, q[bi, : r["tq"]],
                               None if h is None else h[bi, : r["tq"]])
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

    def selected(self, bi: int, g: int) -> list[int]:
        """Chosen LOGICAL candidate pages for one row/group this tick (set after
        the group's _select). Reads the device cache on a captured tick."""
        if self.device_select:
            ch, ns = self._dchosen.get(g), self._dnsel.get(g)
            if ch is None:
                return []
            return [int(x) for x in ch[bi, : int(ns[bi])].tolist()]
        return list(self._chosen.get((bi, g), ()))


# --- sparse prefix cache (host-blob backed; replaces the NoPrefixStore stopgap) ---
_MASK64 = (1 << 64) - 1


def _page_hash(prev: int, token: int) -> int:
    """Same rolling hash as kv_cache._rolling_hash: a page is keyed by the hash of
    its token span's prefix, so page p's key = hash(tokens[: (p+1)*16])."""
    return ((prev * 1_000_003) ^ (token + 1)) & _MASK64


def page_key(tokens, page: int) -> int:
    """Content key of whole page ``page`` (tokens [page*16, (page+1)*16))."""
    h = 0
    for t in tokens[: (page + 1) * BLOCK_TOKENS]:
        h = _page_hash(h, int(t))
    return h


class SparsePrefixCache:
    """Prefix sharing for the sparse engine, backed by SHARED host blobs.

    The dense PrefixStore retains live device blocks; sparse frees the device
    frame on demote, so this index instead retains each published page's HOST
    blob through :meth:`HostKvPages.share_hold` (keyed by the page content hash)
    plus the page's Quest bounds and, per whole-prefix entry, the GDN snapshot.

    An entry covers a CONTIGUOUS block-aligned prefix 0..n_pages-1: every page
    has a held blob, a content key and bounds, plus one GDN ``state`` snapshot
    at the n-page boundary. Pages leave the resident union one at a time and not
    in order (the cross-tick pin keeps a selected page hot for arbitrary ticks),
    so dropped pages are buffered; once 0..m-1 have ALL dropped AND a state
    snapshot at m exists, the publisher's entry covers m pages. Only that one
    longest-frontier entry exists per publisher — intermediate lengths are never
    published (same trade the dense store makes: it snapshots state only at
    aligned chunk boundaries, not per page). A follower adopts exactly the pages
    the entry lists — no hole can shift the page->key mapping. On a hit the
    engine adopts the bounds (zero recompute), restores the state, and promotes
    a page's K/V into a private fresh block only when the selector names it.
    """

    def __init__(self, cold, states, capacity: int = 4096):
        self._cold = cold
        self._states = states
        self._entries: dict[int, list[dict]] = {}  # hash -> verified entries
        #: LRU over entries by id, like the dense PrefixStore; a hit moves to end.
        self._by_id: OrderedDict[int, dict] = OrderedDict()
        self._next_id = 0
        self.capacity = capacity
        # Per-publisher incremental state, cleared on drop_request:
        # _snap[rid][m] = host GDN snapshot at boundary m, captured only on a
        # tick that lands exactly on it (aligned prefill chunk / decode page
        # crossing — the only ticks the recurrent state equals the boundary);
        # _pending[rid][p] host blob from when page p LEFT the resident union;
        # _grow[rid] the one live entry advancing as pages drop; _prompt[rid]
        # the request's own prompt length in pages. Frozen, LRU-managed copies of
        # the grow entry are retained at TWO boundaries, mirroring the dense
        # PrefixStore (first frontier closure + prompt end): one grow entry is
        # not enough because once decode advances it ends in generated-token
        # territory and a same-prompt follower with a different continuation
        # cannot match it.
        self._snap: dict[int, dict[int, tuple]] = {}
        self._pending: dict[int, dict[int, dict]] = {}
        self._grow: dict[int, dict] = {}
        self._prompt_pages: dict[int, int] = {}
        self._frozen: dict[int, set[int]] = {}  # req -> boundaries frozen
        #: (rid, frozen-entry eid) waiting for the engine to add their share refs
        self._freeze_pending: list[tuple[int, int]] = []
        self.published = 0
        self.hits = 0
        self.evictions = 0

    def set_request(self, req_id: int, prompt_pages: int) -> None:
        """Record the request's prompt length in pages so the grow entry is
        frozen at that boundary (the sparse counterpart of the dense store's
        prompt-end publish)."""
        self._prompt_pages[req_id] = prompt_pages

    def note_boundary(self, req_id: int, complete: int, state) -> None:
        """Capture the GDN snapshot at the current whole-page boundary, one per
        finalize that lands EXACTLY on one (a decode page crossing or an aligned
        prefill chunk). The recurrent state then advances past the boundary and
        is unrecoverable, so an unaligned chunk leaves no snapshot and its length
        is never published. Snapshots live on the host: the device copy is
        ~144 MiB at 27B and the lag between a boundary and the contiguous drop of
        its lowest still-pinned page can span many ticks.
        # ponytail: retained per request without a byte budget (bounded by the pin
        # lag); add an LRU eviction freezing the prefix at that boundary if a
        # long-pinned frontier page ever makes this large."""
        if complete <= 0:
            return
        # Only boundaries up to the prompt end can freeze an entry; decode-cross
        # boundaries beyond it are generated-token state nobody adopts.
        if req_id in self._prompt_pages and complete > self._prompt_pages[req_id]:
            return
        states, window = state
        window = None if window is None else window.cpu()
        self._snap.setdefault(req_id, {})[complete] = (states.cpu(), window)

    def publish_dropped(self, req_id: int, tokens, bounds, page: int,
                        offered) -> dict[int, int]:
        """Offer one page the moment it LEFT the resident union (finalize demote).
        ``offered`` is either a host blob dict (held directly here) or the page's
        opaque PRIVATE tier key - in which case NOTHING is copied: the engine
        rehomes the blob to the content key on return (transfer, not clone). A
        page need only be offered once. Once pages 0..m-1 are all offered and the
        boundary-m state snapshot exists, the publisher's single entry (re)attaches
        at length m and this returns {page: content key}; dict offers are
        share_held inline, key offers are transferred by the caller. Out-of-order
        or early pages wait; the longest length with a snapshot is the ceiling.
        A duplicate prefix at m stays off the lookup chains (its blobs still
        rehome for the publisher's own fallback)."""
        self._pending.setdefault(req_id, {})[page] = offered
        tokens = tuple(int(t) for t in tokens)
        pend = self._pending[req_id]
        snaps = self._snap.get(req_id, {})
        e = self._grow.get(req_id)
        old_len = 0 if e is None else len(e["keys"])
        m = old_len
        while m in pend and m in bounds:
            m += 1
        while m > old_len and m not in snaps:
            m -= 1
        if m == old_len:
            return {}
        out = {p: page_key(tokens, p) for p in range(old_len, m)}
        for p in range(old_len, m):
            item = pend.pop(p)
            if isinstance(item, dict):
                # direct blob offer: attach this page's bound to the held blob
                # (the engine path attaches it during the private->shared transfer)
                b = bounds[p]
                if b is not None:
                    item["bounds"] = b
                self._cold.share_hold(out[p], item,
                                      sum(t.numel() * t.element_size()
                                          for t in item.values()
                                          if torch.is_tensor(t)))
        ptokens = tokens[: m * BLOCK_TOKENS]
        if e is None:
            e = {"eid": self._next_id, "tokens": (), "keys": [],
                 "state": None}
            self._next_id += 1
            self._grow[req_id] = e
            self._by_id[e["eid"]] = e
        self._detach(e)
        e["tokens"] = ptokens
        e["keys"].extend(out[p] for p in range(old_len, m))
        e["state"] = snaps[m]
        snaps.pop(m, None)  # consumed: one snapshot, not a retained second copy
        dup = any(
            x is not e and x["tokens"] == ptokens
            for x in self._entries.get(page_key(ptokens, m - 1), ()))
        if not dup:
            self._entries.setdefault(page_key(ptokens, m - 1), []).append(e)
        at_first = old_len == 0
        at_prompt_end = m == self._prompt_pages.get(req_id)
        if at_first or at_prompt_end:
            self._freeze(req_id, m, e)
        self.published += 1
        while len(self._by_id) > self.capacity and self._evict_one():
            pass
        # Blob offers were share_held inline, so a frozen copy's extra refs can
        # be bumped now; private-key offers leave the bump to the engine after it
        # transfers the blobs (take_freeze_refs).
        for _rid, eid in list(self._freeze_pending):
            entry = self._by_id[eid]
            if all(k in self._cold.share_keys() for k in entry["keys"]):
                self._freeze_pending.remove((_rid, eid))
                for k in entry["keys"]:
                    self._cold.share_ref(k)
        return out

    def bound_of_key(self, content_key: int):
        """A page's stored Quest bound from its (possibly spilled) shared blob,
        or None. Adopt reads bounds by field so the bounds plane is not pinned
        inside the entry dict."""
        return self._cold.share_take_field(content_key, "bounds")


    def _freeze(self, req_id: int, m: int, e: dict) -> None:
        """Retain an immutable copy of the grow entry at length m on the lookup
        chains, with its own share refs so it ages independently of the growing
        entry. The first closure and the prompt end may be the same boundary."""
        done = self._frozen.setdefault(req_id, set())
        if m in done:
            return
        snap = {"eid": self._next_id, "tokens": e["tokens"],
                "keys": list(e["keys"]), "state": e["state"]}
        self._next_id += 1
        # Refs are NOT bumped here: the engine transfers the blobs to these keys
        # after publish_dropped returns, then calls add_freeze_refs.
        self._freeze_pending.append((req_id, snap["eid"]))
        self._entries.setdefault(page_key(snap["tokens"], m - 1), []).append(snap)
        self._by_id[snap["eid"]] = snap
        done.add(m)

    def take_freeze_refs(self) -> list[int]:
        """Drain content keys whose frozen entry needs one extra share ref. The
        engine calls this after transferring this publish's blobs."""
        keys: list[int] = []
        pending, self._freeze_pending = self._freeze_pending, []
        for _rid, eid in pending:
            keys.extend(self._by_id[eid]["keys"])
        return keys

    def _detach(self, entry: dict) -> None:
        if entry["tokens"]:
            chain = self._entries.get(self._entry_hash(entry["tokens"]))
            if chain is not None and entry in chain:
                chain.remove(entry)

    def _evict_one(self) -> bool:
        """Evict the oldest entry not still growing with a live publisher. False
        when every entry is a live grow entry (fewer long-lived publishers than
        capacity in practice): the limit is soft there, dropping a grow entry
        would strand its publisher's next extension."""
        growing = {id(e) for e in self._grow.values()}
        for eid, entry in self._by_id.items():
            if id(entry) not in growing:
                self._by_id.pop(eid)
                self._drop(entry)
                self.evictions += 1
                return True
        return False

    def _drop(self, entry: dict) -> None:
        """Remove one entry and release every shared page blob it references. A
        page shared with a surviving entry keeps its blob (share refcount)."""
        self._by_id.pop(entry["eid"], None)
        chain = self._entries.get(self._entry_hash(entry["tokens"]))
        if chain is not None:
            chain.remove(entry)
        for key in entry["keys"]:
            self._cold.share_release(key)

    @staticmethod
    def _entry_hash(tokens) -> int:
        return page_key(tokens, len(tokens) // BLOCK_TOKENS - 1)

    def lookup(self, tokens):
        """Longest block-aligned published prefix of ``tokens`` -> entry dict or None."""
        tokens = tuple(int(t) for t in tokens)
        h = 0
        hashes = []
        for t in tokens:
            h = _page_hash(h, int(t))
            hashes.append(h)
        for i in range(len(tokens) // BLOCK_TOKENS, 0, -1):
            for e in self._entries.get(hashes[i * BLOCK_TOKENS - 1], ()):
                if e["tokens"] == tokens[: i * BLOCK_TOKENS]:
                    self.hits += 1
                    self._by_id.move_to_end(e["eid"])  # LRU
                    return e
        return None

    def drop_request(self, req_id: int) -> None:
        """A finished publisher stops growing; its frozen prompt entries stay on
        the lookup chains and age out under the normal LRU. Gapped buffers,
        unconsumed snapshots and the grow link are dropped."""
        self._snap.pop(req_id, None)
        self._pending.pop(req_id, None)
        self._grow.pop(req_id, None)
        self._prompt_pages.pop(req_id, None)
        self._frozen.pop(req_id, None)

    def clear(self) -> None:
        for entry in list(self._by_id.values()):
            self._drop(entry)
        self._snap.clear()
        self._pending.clear()
        self._grow.clear()
        self._prompt_pages.clear()
        self._frozen.clear()
        self.evictions = 0


