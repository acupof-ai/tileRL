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


class SparseTracker:
    """Engine-scoped scorer store, independent of the KV pool so it survives a
    page's demotion to host:

    - ``bounds`` (scorer="bounds"): req -> page -> fp16 ``[n_full, Hkv, 2, D]``
      Quest kmin/kmax, one per full-attn plane;
    - ``keys`` (scorer="index"): req -> page -> fp8 indexer keys
      ``[n_src, Hkv, di]`` plus an f32 scale per key ``[n_src, Hkv]`` (one scale
      over di=128), projected from the page's mean K at append time by the
      learned ik_weight. Indexer-Q is projected live from each source layer's H.

    The untrained day-0 path inits small iq/ik weights deterministically; the
    learned weights are a later load. At full k both scorers select every
    candidate page and are token-identical to dense."""

    def __init__(self, cfg, k_pages: int, scorer: str, weights: dict | None = None):
        if scorer not in ("bounds", "index"):
            raise NotImplementedError(
                f'sparse engine scorer {scorer!r}: want "bounds" or "index"')
        self.cfg = cfg
        self.k_pages = k_pages
        self.scorer = scorer
        self.src_planes, self.group_of = group_map(cfg)
        self.src_index = {plane: j for j, plane in enumerate(self.src_planes)}
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
            self.bounds: dict[int, dict[int, Tensor]] = {}
            self.keys = None
            self.iq = self.ik = None
        else:
            from .sparse_index import INDEX_HEADS

            # The shipped model has num_kv_heads==4==ih; the tiny cell has 2 KV
            # heads, so it runs ih=2 (head grouping still divides), matching the
            # indexer-weight init used by the warm-up.
            ih = min(INDEX_HEADS, cfg.num_kv_heads)
            self.ih = ih
            self.bounds = None
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
        if self.scorer == "bounds":
            self.bounds.setdefault(req_id, {})
        else:
            self.keys.setdefault(req_id, {})
        self.resident.setdefault(req_id, {})
        self.shared.setdefault(req_id, {})

    def drop(self, req_id: int) -> None:
        if self.scorer == "bounds":
            self.bounds.pop(req_id, None)
        else:
            self.keys.pop(req_id, None)
        self.last_selected.pop(req_id, None)
        self.resident.pop(req_id, None)
        self.shared.pop(req_id, None)

    def set_bounds(self, req_id: int, page: int, b: Tensor) -> None:
        b = b.contiguous().to(torch.float16)
        if not self.bytes_per_page:
            self.bytes_per_page = b.numel() * b.element_size()
        self.bounds[req_id][page] = b

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
            return sum(len(pages) * self.bytes_per_page for pages in self.bounds.values())
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
    IS the own span); ``own`` the own span's logical pages."""

    def __init__(self, tracker: SparseTracker, rows: list[dict], device):
        self.tracker = tracker
        self.device = device
        self.rows = rows
        self.b = len(rows)
        self.n_groups = len(tracker.src_planes)
        #: bounds scores post-rope q only; index scoring also needs the
        #: full-precision post-input-norm hidden (model.py runs the extra norm
        #: only when this is true, so bounds pays no second rmsnorm).
        self.index_scorer = tracker.scorer == "index"
        self._chosen: dict[tuple[int, int], list[int]] = {}
        self._phys: dict[tuple[int, int], Tensor] = {}
        own_w = max(len(r["own"]) for r in rows)
        self.page_base = torch.zeros(self.b, dtype=torch.long, device=device)
        self.own_table = torch.zeros(self.b, own_w, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            self.page_base[i] = r["own"][0]
            self.own_table[i, : len(r["own"])] = torch.tensor(
                [r["resolve"](p) for p in r["own"]])

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
                bounds = torch.stack(
                    [self.tracker.bounds[r["req_id"]][p][plane] for p in cand])
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

    def selected_pages(self, bi: int) -> set[int]:
        """Union of this row's logical pages chosen across ALL source groups this
        tick, plus the own span. Every group's choice co-resides until finalize, so
        the cross-tick pin keeps this exact set and demotes only what left it."""
        pages = set(self.rows[bi]["own"])
        for g in range(self.n_groups):
            pages.update(self._chosen.get((bi, g), ()))
        return pages

    def attention_args(self, plane: int, q: Tensor,
                       h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Packed ``[selected ; own]`` table ``[B,W]`` and per-row packed
        ``seq_len = n_sel*16 + own_len`` for this plane's paged_attention."""
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

    def chosen(self, bi: int) -> set[int]:
        """Union of the row's group selections (kept for diagnostics)."""
        out: set[int] = set()
        for g in range(self.n_groups):
            out.update(self._chosen.get((bi, g), ()))
        return out

    def selected(self, bi: int, g: int) -> list[int]:
        """Chosen LOGICAL candidate pages for one row/group (set after _select)."""
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

    An entry covers a block-aligned prefix: ``length`` tokens, ``keys[p]`` the
    content hash of page p, ``bounds[p]`` its fp16 bounds tensor, and one GDN
    ``state`` snapshot taken at the boundary. On a hit the engine adopts the
    bounds (zero recompute), restores the state, and promotes a page's K/V into
    a private fresh block only when the selector names it (lazy).
    """

    def __init__(self, cold, states, capacity: int = 4096):
        self._cold = cold
        self._states = states
        self._entries: dict[int, list[dict]] = {}  # hash -> verified entries
        #: LRU over entries by id, like the dense PrefixStore; a hit moves to end.
        self._by_id: OrderedDict[int, dict] = OrderedDict()
        self._next_id = 0
        self.capacity = capacity
        self.published = 0
        self.hits = 0
        self.evictions = 0

    def publish(self, tokens, page_blobs, bounds, state) -> bool:
        """Publish one block-aligned prefix.

        ``page_blobs``: {logical page: host blob} for every whole page in the
        prefix (the sparse finalize already demoted them to host blobs).
        ``bounds``: {page: bounds tensor}. ``state``: (states, windows) snapshot.
        Each blob is share_held under its content key; duplicate prefixes no-op.
        """
        tokens = tuple(int(t) for t in tokens)
        n_pages = len(tokens) // BLOCK_TOKENS
        if n_pages == 0:
            return False
        h = page_key(tokens, n_pages - 1)
        for e in self._entries.get(h, ()):
            if e["tokens"] == tokens:
                return False
        keys, key_by_page, kept_bounds = [], {}, {}
        for p in range(n_pages):
            if p not in page_blobs:
                continue  # a selected-candidate page with no own blob: not publishable
            blob = dict(page_blobs[p])
            if p in bounds:
                blob["bounds"] = bounds[p]
            key = page_key(tokens, p)
            self._cold.share_hold(key, blob, _blob_nbytes(blob))
            keys.append(key)
            key_by_page[p] = key
            kept_bounds[p] = bounds[p]
        eid = self._next_id
        self._next_id += 1
        entry = {"eid": eid, "tokens": tokens, "keys": keys,
                 "bounds": kept_bounds, "state": state}
        self._entries.setdefault(h, []).append(entry)
        self._by_id[eid] = entry
        self.published += 1
        while len(self._by_id) > self.capacity:
            self._evict_one()
        return key_by_page  # {page: content key} for the publisher's own resolve

    def _evict_one(self) -> None:
        _, entry = next(iter(self._by_id.items()))
        self._drop(entry)
        self.evictions += 1

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

    def drop_request(self, *_):
        """Shared blobs are owned by PREFIX entries, not requests; nothing per-request."""

    def clear(self) -> None:
        for entry in list(self._by_id.values()):
            self._drop(entry)
        self.evictions = 0


def _blob_nbytes(blob: dict) -> int:
    return sum(t.numel() * t.element_size() for t in blob.values()
               if torch.is_tensor(t))
