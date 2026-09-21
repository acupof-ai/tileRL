"""Sparse-tick runtime: residency, promotion/demotion, prefix offers and the
captured sparse decode graph, split out of engine.py
(docs/design-architecture.md row 11).

The Engine owns scheduling (which rows tick, dense vs sparse predicates) and
the queues; SparseRuntime owns everything a sparse tick does to the tracker,
the KV cold tier and its own captured graphs. It never imports the Engine:
pools, model, backend and the engine's verify/sample/draft-step callbacks
arrive on an immutable SparseCtx bound once at construction. The tracker
attributes the rest of the tree reads are proxied through, so ``engine._sparse``
keeps its old surface.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .decode_graph import graph_bucket, make_sparse_graph
from .kv_cache import BLOCK_TOKENS


@dataclass(frozen=True)
class SparseCtx:
    """Read-only Engine surface a sparse tick needs: pools/model/backend, the
    draft geometry, and the named engine callbacks that mutate engine state
    (verify/sample commit/draft step/decode-forward counter). No Engine object
    crosses the seam."""

    model: Any
    backend: Any
    kv: Any
    states: Any
    draft: Any
    width: int
    aux_layers: tuple
    max_batch: int
    graph_capture: Any
    verify: Callable
    sample_commit: Callable
    draft_step: Callable
    bump_decode_forwards: Callable
    #: The Engine's env-gated step timer (None unless TILERL_STEP_TIMING), so a
    #: publish can attribute its own device costs without importing the Engine.
    step_timing: Any = None


def retier(pool, keep, running, waiting) -> tuple[int, int]:
    """Apply one selector decision to the live pages. ``keep`` names the
    physical block ids this tick selects; the caller passes the UNION of the
    chunk's selected sets (a V4.1 index source shares one selection across its
    group of 4 full-attn layers, and a block id is one physical page moved
    across all its planes in a single host fetch) and never names the last 8
    pages (the 128-token window).

    A private page not selected demotes to the pinned host tier (prefix-shared
    pages are refused by the pool); a selected host page promotes into a FRESH
    device block. Either way the page keeps its immutable logical index, so the
    rebuilt block list stays in sequence order — paged_attention derives
    causal positions from that order. Returns ``(demoted, promoted)``.
    """
    cold = pool.cold
    if cold is None:
        raise RuntimeError("sparse_retier: engine built without a cold page tier")
    d0, p0 = cold.demotions, cold.promotions
    remap: dict[int, int] = {}
    reqs = list(running) + list(waiting)
    # Batch this retier's D2Hs: demoted frames stay live until the single
    # end sync, so an interleaved promote's alloc_block cannot recycle a
    # frame a non-blocking D2H is still reading.
    with pool.demotions():
        for r in reqs:
            cold_map = dict(r.cold_pages)  # logical index -> host block id
            live = iter(r.blocks)
            n = len(r.blocks) + len(cold_map)
            ordered: list[tuple[int, int, bool]] = []
            for idx in range(n):
                ordered.append(
                    (idx, cold_map[idx], True) if idx in cold_map else (idx, next(live), False)
                )
            new_live: list[int] = []
            new_cold: list[tuple[int, int]] = []
            for idx, b, was_cold in ordered:
                if b in keep:
                    if was_cold:  # selected host page -> one fetch, a fresh block
                        if b not in remap:
                            remap[b] = pool.promote_page(b)
                        new_live.append(remap[b])
                    else:
                        new_live.append(b)  # selected device page, untouched
                elif was_cold:
                    new_cold.append((idx, b))  # still unselected on the host
                elif pool.is_shared(b):
                    new_live.append(b)  # prefix-shared: read-only, never demoted
                else:
                    pool.demote_page(b)
                    new_cold.append((idx, b))
            # Both lists were appended in ascending logical idx, so the live block
            # table is in sequence order and cold pages keep their absolute index.
            r.blocks = new_live
            r.cold_pages = new_cold
    return cold.demotions - d0, cold.promotions - p0


class SparseRuntime:
    """The sparse half of one Engine. State the Engine used to hold lives here:
    refresh counter, per-bucket captured sparse graphs, the graph-on flag, the
    warm-adoption counter; the SparseTracker remains the scorer/residency store
    and is exposed read-only through the proxy properties below."""

    def __init__(self, tracker, device_select: bool, graph_on: bool):
        self.tracker = tracker
        self.device_select = device_select
        self.graph_on = graph_on
        self.ticks_since_refresh = 0
        self.graphs: dict = {}
        self.warm_adoptions = 0
        self.ctx: SparseCtx | None = None

    # ------------------------------------------------ tracker read proxies
    @property
    def prefix(self):
        return self.tracker.prefix

    @property
    def resident(self):
        return self.tracker.resident

    @property
    def shared(self):
        return self.tracker.shared

    @property
    def scorer(self):
        return self.tracker.scorer

    @property
    def k_pages(self):
        return self.tracker.k_pages

    @property
    def di(self):
        return self.tracker.di

    @property
    def last_selected(self):
        return self.tracker.last_selected

    @last_selected.setter
    def last_selected(self, value) -> None:
        self.tracker.last_selected = value

    @property
    def bounds_valid(self):
        return self.tracker.bounds_valid

    @property
    def keys(self):
        return self.tracker.keys

    @property
    def ik(self):
        return self.tracker.ik

    @property
    def src_planes(self):
        return self.tracker.src_planes

    @property
    def bounds_count(self):
        return self.tracker.bounds_count

    def attach(self, req_id: int) -> None:
        self.tracker.attach(req_id)

    def drop(self, req_id: int) -> None:
        self.tracker.drop(req_id)

    def set_index_keys(self, *args, **kwargs):
        return self.tracker.set_index_keys(*args, **kwargs)

    def set_bounds(self, *args, **kwargs):
        return self.tracker.set_bounds(*args, **kwargs)

    def map_evict(self, *args, **kwargs):
        return self.tracker.map_evict(*args, **kwargs)

    def map_resident(self, *args, **kwargs):
        return self.tracker.map_resident(*args, **kwargs)

    def has_bounds(self, *args, **kwargs):
        return self.tracker.has_bounds(*args, **kwargs)

    def bounds_view(self, *args, **kwargs):
        return self.tracker.bounds_view(*args, **kwargs)

    def bounds_bytes(self, *args, **kwargs):
        return self.tracker.bounds_bytes(*args, **kwargs)

    def __getattr__(self, name):
        # Facade fallback for tracker internals tests and probes read ad hoc
        # (ih, bounds_t, l2p_t, ...); explicit properties above stay the gate.
        return getattr(self.tracker, name)

    def recall(self, req_id: int, target_mass: torch.Tensor) -> dict[int, float]:
        """Recall of the LAST tick's served selection against an offline dense
        page-mass target ``[1, n_groups, nq, pages]`` (window excluded), per
        source group. Lets the card run score what the engine actually selected
        rather than only the offline teacher. Full k -> 1.0."""
        sel = self.tracker.last_selected.get(req_id)
        if sel is None:
            return {}
        out: dict[int, float] = {}
        for g, (cand, chosen) in sel.items():
            k = min(self.tracker.k_pages, len(cand))
            if k == 0:
                out[g] = 0.0
                continue
            ci = torch.tensor(cand, device=target_mass.device)
            mass = target_mass[0, g].sum(0).index_select(0, ci)
            dense = {cand[int(i)] for i in mass.topk(k).indices.tolist()}
            out[g] = len(dense & set(chosen)) / k
        return out

    def build_rows(self, rows, seq_q, decodes):
        """Build this tick's SparseForward: per-row own span (allocated/promoted),
        earlier complete candidate pages, and a resolve closure that promotes a cold
        selection. The model scores and installs page_sel mid-forward."""
        ctx = self.ctx
        from .sparse_engine import SparseForward
        from .sparse_index import WINDOW_PAGES as _WP

        tr = self.tracker
        srows = []
        for r, tq in zip(rows, seq_q):
            decoding = r in decodes
            reserved: set[int] = set()
            q_hi = int(r.seq_len) if decoding else int(r.prefill_from + tq)
            q_lo = q_hi - tq
            if decoding:
                # Chain query positions are [seq_len-1 .. seq_len-1+tq): the
                # verify tick includes the W-1 draft queries, so a chain crossing
                # a page boundary must allocate that next own page.
                q_lo, q_hi = r.seq_len - 1, r.seq_len - 1 + tq
                # own span = the trailing 8-page window the new token writes into
                own_first = max(0, (q_lo // BLOCK_TOKENS) - (_WP - 1))
                own_last = (q_hi - 1) // BLOCK_TOKENS
                force_window = 0  # the window IS the own span
            else:
                q_hi = int(r.prefill_from + tq)
                q_lo = q_hi - tq
                own_first = q_lo // BLOCK_TOKENS
                own_last = (q_hi - 1) // BLOCK_TOKENS
                force_window = _WP  # force the 8 pre-chunk pages
            own = list(range(own_first, own_last + 1))
            own_len = q_hi - own_first * BLOCK_TOKENS
            if tr.scorer == "bounds":
                cand = [p for p in range(0, own_first) if tr.has_bounds(r.req_id, p)]
            else:
                cand = [p for p in range(0, own_first) if p in tr.keys[r.req_id]]

            def resolve(p, r=r, reserved=reserved):
                return self.resolve(r, p, reserved)

            for p in own:
                reserved.add(p)
                resolve(p)
            srows.append(
                dict(
                    req_id=r.req_id,
                    own=own,
                    own_len=own_len,
                    q_hi=q_hi,
                    tq=tq,
                    decoding=decoding,
                    cand=cand,
                    force_window=force_window,
                    resolve=resolve,
                    reserved=reserved,
                )
            )
        # Pure-decode ticks run the device (resident-only) path except every
        # SPARSE_REFRESH_TICKS-th, which goes eager to re-score ALL candidates and
        # promote the ones the resident hot set is missing. Prefill/mixed ticks are
        # always eager (chunks force the window and grow the candidate set).
        from .sparse_engine import SPARSE_REFRESH_TICKS

        pure_decode = bool(decodes) and len(decodes) == len(rows)
        if self.device_select and pure_decode:
            self.ticks_since_refresh += 1
            do_refresh = self.ticks_since_refresh >= SPARSE_REFRESH_TICKS
        else:
            do_refresh = False
        device_select = self.device_select and pure_decode and not do_refresh
        if do_refresh:
            self.ticks_since_refresh = 0
        return SparseForward(
            tr, srows, ctx.backend.device, ctx.backend, device_select=device_select
        )

    def evict_victim(self, r, reserved: set[int]) -> None:
        """Free one frame this tick does NOT need, so a promotion can allocate.

        Cross-tick pin leaves last tick's selected pages resident; when this tick's
        selection differs, a newly named page needs that frame. Demote any resident
        page of this row outside the tick's reserved set (own span + this tick's
        picks across groups). Raises if every resident page is reserved — that would
        mean the pool was undersized below the pin ceiling, a build_engine bug."""
        ctx = self.ctx
        live = self.tracker.resident[r.req_id]
        for p, phys in live.items():
            if p in reserved:
                continue
            ctx.kv.demote_page(phys, key=(r.req_id, p))
            r.cold_pages.append(p)
            r.blocks.remove(phys)
            self.tracker.map_evict(r.req_id, p)
            del live[p]
            return
        raise RuntimeError(
            "sparse: no unreserved resident page to evict for a "
            "promotion; hot pool undersized below the pin ceiling"
        )

    def resolve(self, r, page: int, reserved: set[int] | None = None) -> int:
        """Physical block for a resident, private-cold, or shared-prefix logical
        page, allocating a fresh block for a never-written own page or promoting
        the host blob. A shared prefix page promotes its read-only host blob into
        a fresh PRIVATE block (the store entry keeps the blob). The promote makes
        the block private, but the page's shared content-key label is KEPT: when
        that block later leaves the hot union again the bytes already exist under
        the same key, so the redemote skips device bytes (#783). Under the
        cross-tick pin the pool is full of last tick's pages, so evict one
        unreserved frame first when no block is free."""
        ctx = self.ctx
        tr = self.tracker
        live = tr.resident[r.req_id]
        if page in live:
            return live[page]
        if ctx.kv.free_blocks == 0 and reserved is not None:
            self.evict_victim(r, reserved)
        # Automatic path: cold_pages are bare logical ints, blob keyed (req, page).
        # A page the publisher itself dropped also has a shared key: prefer its
        # private blob, fall back to the shared clone on byte-LRU eviction.
        shared_keys = tr.shared.get(r.req_id, {})
        if page in r.cold_pages and (r.req_id, page) in ctx.kv.cold:
            new = ctx.kv.promote_keyed((r.req_id, page))
            r.cold_pages.remove(page)
        elif page in shared_keys:
            blob = ctx.kv.cold.share_take(shared_keys[page])
            if blob is None:
                raise RuntimeError(f"sparse prefix page {page} missing its shared blob")
            new = ctx.kv.shared_promote(blob)
            if page in r.cold_pages:
                r.cold_pages.remove(page)  # transfer moved the blob to the content key
        else:
            new = ctx.kv.alloc_block()
        live[page] = new
        r.blocks.append(new)
        tr.map_resident(r.req_id, page, new)
        return new

    def warm_draft(self, r, entry: dict, matched: int) -> None:
        """Restore a WARM prefix into a spec follower's dense draft pool: copy the
        publisher's per-page draft K/V into this row's reserved draft blocks,
        zero the boundary slot, and prime draft state so the first tail draft
        conditions on the saved boundary trunk hidden. Bit-equal to a cold
        follower: cold zeroed position 0 in block 0; warm zeroes position
        ``matched`` in block M, and every earlier draft slot is the publisher's
        own value (already zero at its position 0)."""
        ctx = self.ctx
        dpool = ctx.draft.kv
        dev = dpool.k_pool.device
        keys = entry["keys"]
        for p, key in enumerate(keys):
            # field-only read: restore the draft planes without pinning the page's
            # trunk K/V blob into host RAM
            dk = ctx.kv.cold.share_take_field(key, "dk")
            dv = ctx.kv.cold.share_take_field(key, "dv")
            if dk is None or dv is None:
                raise RuntimeError(f"warm prefix entry lost draft KV for page {p} (key {key})")
            blk = r.draft_blocks[p]
            dpool.k_pool[:, blk].copy_(dk.to(dev))
            dpool.v_pool[:, blk].copy_(dv.to(dev))
        # boundary slot: no draft attends past the matched prefix there
        boundary_blk = r.draft_blocks[matched // BLOCK_TOKENS]
        dpool.k_pool[:, boundary_blk, :, 0, :].zero_()
        dpool.v_pool[:, boundary_blk, :, 0, :].zero_()
        # condition the first tail draft (position matched) on hidden matched-1
        r.hidden = entry["hidden"].to(dev).reshape(1, 1, -1)
        r.hidden_prev = None
        r.hidden_from = matched - 1
        r.draft_pos = matched - 1
        self.warm_adoptions += 1

    def offer_drop(self, r, page: int, draft_pages: dict | None = None) -> None:
        """Page ``page`` just LEFT the resident union: offer it to the prefix index.
        Drop-only — a stable pin never reaches here. No blob is copied or moved
        yet: the index buffers out-of-order drops behind the contiguous frontier
        and skips a page with no bound, so an entry never names a page it cannot
        serve. When the frontier closes, :meth:`SparsePrefixCache.publish_dropped`
        hands back the content keys and the engine rehomes each private blob to
        its content key (one copy, not two). A
        newly published page's draft K/V is copied from the request's draft pool
        (warm spec adoption)."""
        ctx = self.ctx
        tr = self.tracker
        if tr.prefix is None or not tr.has_bounds(r.req_id, page):
            return
        keys = tr.prefix.publish_dropped(
            r.req_id, r.tokens, tr.bounds_view(r.req_id), page, (r.req_id, page)
        )
        # The frontier can close over MANY pages though only ``page`` dropped
        # this tick, so resolve each new page's draft block from the reserved
        # draft span, not from the one dropped page.
        written_page = (
            (r.draft_pos + 1) // BLOCK_TOKENS if ctx.draft is not None and r.draft_blocks else -1
        )
        for p, content_key in keys.items():
            draft_block = r.draft_blocks[p] if p <= written_page else None
            self.transfer_to_shared(r, p, content_key, draft_block)
        for content_key in tr.prefix.take_freeze_refs():
            ctx.kv.cold.share_ref(content_key)

    def transfer_to_shared(
        self, r, page: int, content_key: int, draft_block: int | None = None
    ) -> None:
        """Publish one page under its content key: attach bounds (+ draft K/V for
        a warm spec entry) to the page's trunk K/V.

        Four states the page can be in when its frontier closes:
        - its content key already exists in the shared tier (an adopted page
          re-leaving the union): zero device bytes, just one more share ref;
        - held on the host as the PRIVATE (rid,page) blob: transfer it (no copy);
        - on the private SSD: share_hold_kv lifts it into the prefix spill file;
        - still DEVICE-resident (in the own window, never dropped): build the host
          blob from its live physical frame here.
        Returning without a blob would leave a lookup entry naming a dead key.

        Charged in five segments because they have different fixes: bounds D2H,
        draft K/V clone, the cold tier's host-RAM transfer, the live-frame
        snapshot, and the shared-namespace hold. Disk IO is NOT here — the spill
        file measures itself and the engine reports it as ``ssd_mmap``, so a
        profile attributing disk time to any of these five is misreading.

        Every page commits inline here (bounds/draft/frame D2H are synchronous);
        pages whose source is already host/SSD move no device bytes."""
        ctx = self.ctx
        tr = self.tracker
        tm = ctx.step_timing
        tr.shared.setdefault(r.req_id, {})[page] = content_key
        # Dup content key (typically an adopted prefix whose frame left the union
        # again): the identical trunk/bounds/draft bytes are already shared, so no
        # D2H and no private transfer — one ref keeps the grow/frozen entry whole.
        if content_key in ctx.kv.cold.share_keys():
            ctx.kv.cold.share_ref(content_key)
            return
        t = time.perf_counter() if tm is not None else 0.0
        # bounds: one blocking page D2H.
        bv = tr.bounds_view(r.req_id)[page]
        bounds_host = bv.cpu()
        if tm is not None:
            tm.mark("pub_bounds_d2h", t)
            t = time.perf_counter()
        extra_host = {"bounds": bounds_host}
        if draft_block is not None and ctx.draft is not None:
            dpool = ctx.draft.kv
            # clone: .cpu() is a no-op on the CPU cell, so without it the blob
            # aliases a draft block that gets recycled and overwritten.
            extra_host["dk"] = dpool.k_pool[:, draft_block].detach().cpu().clone()
            extra_host["dv"] = dpool.v_pool[:, draft_block].detach().cpu().clone()
        if tm is not None:
            tm.mark("pub_draft_clone", t)
            t = time.perf_counter()
        if (r.req_id, page) in ctx.kv.cold:
            # Host-resident or already on the private SSD: no device bytes.
            n = ctx.kv.cold.share_hold_kv((r.req_id, page), content_key, extra=extra_host)
            if tm is not None:
                tm.mark("pub_cold_transfer", t)
            if n:
                return
            r.cold_pages = [
                content_key if (isinstance(p, tuple) and p == (r.req_id, page)) else p
                for p in r.cold_pages
            ]
            return
        phys = tr.resident.get(r.req_id, {}).get(page)
        if phys is None:
            raise RuntimeError(
                f"publish page {page}: neither a private host blob nor a resident "
                f"frame exists (req {r.req_id}, content key {content_key})"
            )
        # Device-resident frame snapshot, committed inline.
        blob, n = ctx.kv._page_blob(phys)
        if tm is not None:
            tm.mark("pub_frame_d2h", t)
            t = time.perf_counter()
        self._commit_frame(blob, n, extra_host, content_key, tm, t)

    def _commit_frame(self, blob, n, extra_host, content_key, tm, t):
        """Commit one already-valid host frame blob to the shared cold tier."""
        blob.update(extra_host)
        n += sum(x.numel() * x.element_size() for x in extra_host.values() if torch.is_tensor(x))
        self.ctx.kv.cold.share_hold(content_key, blob, n)
        if tm is not None:
            tm.mark("pub_share_hold", t)

    def finalize(self, sf, rows, hidden=None) -> list:
        """After the forward: store Quest bounds of every now-complete page, then
        keep resident the pages selected THIS tick (the union of the source groups
        plus the own span) and demote only resident pages that LEFT that set.

        Returns (row, dropped pages) offers the caller processes LATER — after
        ``draft.step`` — so a published blob can also carry the page's draft K/V
        (this chunk's draft forward runs after the trunk forward). ``hidden`` is
        the forward's per-row trunk hidden [1,q,H], used to save the boundary
        vector a spec follower's first tail draft conditions on.

        Cross-tick pin: a stable selection keeps the same physical frames pinned,
        so the next tick promotes nothing; a changed selection demotes only the
        dropped pages and promotes the newly named ones. The device pool is sized
        n_groups*k + window + chunk per slot, i.e. exactly this pin ceiling.
        Bounds stay device-resident regardless, so scoring a cold page needs no K.

        Prefix publishing is DROP-ONLY (bounds scorer): a page is offered only
        when it leaves the union, and AFTER the demotions() scope exits so its
        private blob is already held (a batched implementation holds it at the
        batch sync; peeking inside would clone nothing). The GDN boundary
        snapshot is captured only on a finalize landing on a whole-page boundary.
        """
        ctx = self.ctx
        tr = self.tracker
        pool = ctx.kv
        from .sparse_engine import page_bounds_one, project_index_page_keys

        # One batched D2H for the tick's departing pages across every row: each
        # demote launches non-blocking into pinned staging while its frame stays
        # live; the context syncs once before the frames return to the pool.
        # (row, dropped pages) collected in the scope, offered to the prefix index
        # after it exits — the one point every demoted blob is guaranteed held.
        dropped_offers: list[tuple] = []
        with pool.demotions():
            for bi, r in enumerate(rows):
                rid = r.req_id
                if rid not in tr.resident:
                    continue  # request finished and dropped its tracker state this tick
                live = tr.resident[rid]
                q_hi = sf.rows[bi]["q_hi"]
                complete = q_hi // BLOCK_TOKENS
                n_stored = tr.bounds_count[rid] if tr.scorer == "bounds" else len(tr.keys[rid])
                for p in range(n_stored, complete):
                    if p not in live:
                        # selected candidate promoted with its scorer state already
                        continue
                    phys = live[p]
                    if pool.kv_fp8 is not None:
                        raise NotImplementedError(
                            f"sparse {tr.scorer} state over an fp8 pool: card PR"
                        )
                    if tr.scorer == "bounds":
                        b = torch.stack(
                            [
                                page_bounds_one(pool.k_pool[plane, phys])
                                for plane in range(pool.num_layers)
                            ]
                        )
                        tr.set_bounds(rid, p, b)
                    else:
                        # mean K per source plane -> learned fp8 indexer keys
                        kmean = torch.stack(
                            [
                                pool.k_pool[plane, phys].to(tr.ik.dtype).mean(dim=1)
                                for plane in tr.src_planes
                            ]
                        )
                        keys, scales = project_index_page_keys(kmean[None], tr.ik)
                        keys, scales = keys[0], scales[0]
                        tr.set_index_keys(
                            rid, p, keys.to(pool.k_pool.device), scales.to(pool.k_pool.device)
                        )
                if tr.scorer == "bounds" and tr.prefix is not None and q_hi % BLOCK_TOKENS == 0:
                    sp = ctx.states
                    # vector at position q_hi-1; note_boundary moves it to host
                    boundary_h = (
                        None
                        if hidden is None or ctx.draft is None
                        else hidden[bi, sf.rows[bi]["tq"] - 1]
                    )
                    tr.prefix.note_boundary(
                        rid,
                        complete,
                        (sp.states[r.state_slot].clone(), sp.window_snapshot(r.state_slot)),
                        boundary_h,
                    )
                if getattr(sf, "device_select", False) and sf.device.type == "cuda":
                    # Captured tick: skip the device→host pin readback here;
                    # evict_victim prunes on promotion and the eager refresh tick
                    # reconciles the pin set and publishes drops.
                    continue
                kept = sf.selected_pages(bi)
                dropped = [p for p in live if p not in kept]
                labels = tr.shared.get(rid, {})
                for p in dropped:
                    phys = live[p]
                    # An adopted (or self-republished) page whose identical bytes
                    # are already held under a shared content key needs no D2H:
                    # return the frame straight to the pool. The label must be
                    # checked against the live set — the key can be gone since
                    # resolve (shared LRU eviction), in which case the page is
                    # demoted normally and republished (#783). The page is still
                    # offered below so publish_dropped keeps a hole-free frontier.
                    label = labels.get(p)
                    if label is not None and label in ctx.kv.cold.share_keys():
                        ctx.kv.free_block(phys)
                    else:
                        pool.demote_page(phys, key=(rid, p))
                        r.cold_pages.append(p)
                    tr.map_evict(rid, p)
                    r.blocks.remove(phys)
                kept_live = {p: live[p] for p in kept if p in live}
                # r.blocks mirrors the live frames in LOGICAL page order (paged_attention
                # derives causal positions from the order), so sort the pinned set.
                r.blocks = [kept_live[p] for p in sorted(kept_live)]
                live.clear()
                live.update(kept_live)
                if dropped:
                    dropped_offers.append((r, dropped))

        return dropped_offers

    def process_offers(self, dropped_offers) -> None:
        """Publish dropped pages AFTER ``draft.step``: this chunk's draft K/V now
        exist in the request's draft pool for the transfer to copy."""
        for r, pages in dropped_offers:
            for p in pages:
                self.offer_drop(r, p)

    def decode_rows(self, decodes, q_dec: list[int]) -> list[dict]:
        """Decode-only geometry for a captured sparse tick — the decode branch of
        ``_sparse_rows`` without its refresh bookkeeping (the runner owns that).
        Own pages resolve via the same closure; candidates are the complete
        earlier pages with stored bounds."""
        from .sparse_index import WINDOW_PAGES as _WP

        tr = self.tracker
        srows = []
        for r, tq in zip(decodes, q_dec):
            reserved: set[int] = set()
            q_lo, q_hi = r.seq_len - 1, r.seq_len - 1 + tq
            own_first = max(0, (q_lo // BLOCK_TOKENS) - (_WP - 1))
            own_last = (q_hi - 1) // BLOCK_TOKENS
            own = list(range(own_first, own_last + 1))
            cand = (
                [p for p in range(0, own_first) if tr.has_bounds(r.req_id, p)]
                if tr.scorer == "bounds"
                else [p for p in range(0, own_first) if p in tr.keys[r.req_id]]
            )

            def resolve(p, r=r, reserved=reserved):
                return self.resolve(r, p, reserved)

            for p in own:
                reserved.add(p)
            srows.append(
                dict(
                    req=r,
                    req_id=r.req_id,
                    own=own,
                    own_len=q_hi - own_first * BLOCK_TOKENS,
                    q_hi=q_hi,
                    tq=tq,
                    decoding=True,
                    cand=cand,
                    force_window=0,
                    resolve=resolve,
                    reserved=reserved,
                )
            )
        return srows

    def run_decode_graph(self, reqs, chains) -> bool:
        """Capture/replay the sparse steady-state decode tick. Returns False (caller
        runs eager) on a refresh tick (needed promotions), a bounds-scorer-only
        configuration, or a failed capture. The refresh counter is advanced ONLY
        on a captured tick; eager goes through ``_sparse_rows`` which owns it."""
        ctx = self.ctx
        from .sparse_engine import SPARSE_REFRESH_TICKS, SparseForward, cmax_bucket
        from .sparse_index import WINDOW_PAGES as _WP

        tr = self.tracker
        if tr.scorer != "bounds":
            return False
        q_dec = [len(c) for c in chains] if chains else [1] * len(reqs)
        # Read-only peek: let _sparse_rows do the reset when this tick is a refresh.
        if self.ticks_since_refresh + 1 >= SPARSE_REFRESH_TICKS:
            return False
        rows = self.decode_rows(reqs, q_dec)
        n = len(reqs)
        B = graph_bucket(n, ctx.max_batch)
        W = max(q_dec)
        own_w = _WP + (1 if W > 1 else 0)
        cmax = max(len(r["cand"]) for r in rows)
        key = (B, W, cmax_bucket(cmax), own_w)
        g = self.graphs.get(key)
        if g is None:
            if n < B and not ctx.graph_capture.ensure_pad():
                return False
            sf = SparseForward(
                tr,
                None,
                ctx.backend.device,
                ctx.backend,
                device_select=True,
                reuse=True,
                b=B,
                cmax_cap=key[2],
                own_w_cap=own_w,
            )
            g, ctx.graph_capture.pool, err = make_sparse_graph(
                ctx.model,
                ctx.backend,
                ctx.kv,
                ctx.states,
                self.tracker,
                sf,
                B,
                W,
                ctx.aux_layers,
                ctx.graph_capture.pool,
            )
            if g is None:
                warnings.warn(
                    f"sparse decode graph capture failed for {key}: {err}; eager fallback"
                )
                self.graph_on = False
                return False
            self.graphs[key] = g
        logits = g.run(
            rows,
            chains or [(r.output[-1],) for r in reqs],
            pad=ctx.graph_capture.pad,
        )
        self.ticks_since_refresh += 1
        ctx.bump_decode_forwards()
        # Finalize residency immediately after the forward (same order as the
        # eager path (finalize before sample/verify): the pin reads
        # this tick's selection out of the captured sf and demotes the rest.
        self.finalize(g.sf, reqs)
        if chains:
            ctx.verify(reqs, chains, logits, g.hidden)
        else:
            if ctx.draft is not None and g.hidden is not None:
                for i, r in enumerate(reqs):
                    r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
                    r.hidden, r.hidden_from = g.hidden[i : i + 1], r.seq_len - 1
            ctx.sample_commit([(r, logits[i, -1], len(r.output)) for i, r in enumerate(reqs)])
        if ctx.draft is not None:
            end = ctx.width - 1
            for r in reqs:
                assert len(r.draft_blocks) * BLOCK_TOKENS > r.seq_len - 1 + end
            ctx.draft_step(reqs)
        return True
