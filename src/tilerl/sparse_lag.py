"""PROBE-ONLY (#805, branch probe/805-v2; never merge to main as-is).

v2 real 1-tick-delay sparse refresh — controller + promote worker.

Cadence (SPARSE_REFRESH_TICKS=8): seven plain captured graph ticks, then one
CARRY tick. At the END of plain tick 7 the controller snapshots the just-played
staging and previous q and runs ONE job:

    select_refresh (quest over ALL candidates, no residency mask)
      -> classify every union pick: resident (reuse phys) or cold private
         (HostKvPages.take mmap + H2D into a RESERVED block)
      -> per-row, per-group (phys, chosen, nsel)

On the carry tick the controller commits BEFORE fill(): joins the worker and
(the main stream waits the H2D end event — a background that overruns one
interval makes this tick wait, it never falls back), takes every cold blob and
maps residency on the step thread, and adds the picks to the carry tick's
reserved set so fill()'s own-page resolve cannot evict them. pre_replay then
arms the physical blocks into the captured sf; the carry tick is a graph
replay, never an eager trunk.

Two modes, ONE job (select_refresh + reserve + H2D):
  inline: job runs immediately on the step thread over LIVE staging.
  async:  over post-replay SNAPSHOTS; CUDA -> worker thread, quest+H2D on a
          side stream (the real overlap); CPU -> immediate but through the SAME
          snapshot/job/commit code, so the CPU installation gate exercises the
          device path. A cycle the reservation cannot cover, or that names an
          unsupported (shared-prefix/missing) page, falls back to the normal
          eager refresh and is counted in fallback_cycles.

Env: TILERL_SPARSE_V2=inline|async|off; TILERL_SPARSE_V2_RESERVE reserve pages
(default 200; measured max promotions/refresh 156 + margin).
"""

from __future__ import annotations

import os
import threading

import torch


def lag_mode() -> str | None:
    m = os.environ.get("TILERL_SPARSE_V2", "off")
    if m in ("", "off", "0"):
        return None
    if m not in ("inline", "async"):
        raise ValueError(f"TILERL_SPARSE_V2={m!r}; want off|inline|async")
    return m


class LagController:
    def __init__(self, runtime, mode: str):
        self.rt = runtime
        self.mode = mode
        self.reserve_cap = int(os.environ.get("TILERL_SPARSE_V2_RESERVE", "200"))
        # Blocks carved from the pool free list, owned exclusively by a job
        # until commit maps them. Pool._free is never touched mid-job.
        self.reserve: list[int] = []
        self.stream = None
        self._thread = None
        self._result = None
        self._error = None
        self._end_event = None
        self._pending = None
        self._use_rows = None
        # H2D host sources must outlive the async copies. They are retained
        # across the carry and released at the NEXT prepare(), after the prior
        # side-stream end event is synchronized (then the copies have truly
        # finished; a stream wait_event orders GPU work but does not keep a
        # pinned host buffer alive).
        self._held_blobs: list = []
        self._held_event = None
        self.carry_cycles = 0
        self.fallback_cycles = 0

    # ------------------------------------------------------------ cadence
    def is_carry(self) -> bool:
        from .sparse_engine import SPARSE_REFRESH_TICKS

        return self.rt.ticks_since_refresh + 1 >= SPARSE_REFRESH_TICKS

    def is_ready(self) -> bool:
        return self._thread is not None or self._result is not None

    def _kv(self):
        return self.rt.ctx.kv

    def _topup_reserve(self) -> None:
        """Refill the exclusive block list from the pool free list, capped at
        reserve_cap and at half the current free pool so the live tick is never
        starved (mirrors the shadow-v1 carve rule)."""
        pool = self._kv()
        n = min(self.reserve_cap - len(self.reserve), pool.free_blocks // 2)
        for _ in range(n):
            self.reserve.append(pool._free.pop())

    def _release_prior_held(self, sf) -> None:
        """Release the previous carry's host blob sources after its side-stream
        H2D is known (host-side) complete. Called at the NEXT prepare(), on a
        plain tick, so this sync is off the carry's critical path."""
        ev = self._held_event
        if ev is not None:
            ev.synchronize()
            self._held_event = None
        self._held_blobs = []

    # ----------------------------------------------------- plain tick 7
    def prepare(self, sf, srows) -> None:
        """Snapshot + launch the carry-selection job at end of plain tick 7."""
        self._release_prior_held(sf)
        self._reset()
        q_prev = sf._q_prev
        if not q_prev:
            return  # no captured q yet: the first carry falls back
        self._topup_reserve()
        snap = (sf.cand_idx.clone(), sf.n_cand.clone(), sf.win.clone(),
                ([b.clone() for b in sf.s_bounds] if sf.s_bounds else None),
                sf.s_l2p.clone())
        q = {g: t.clone() for g, t in q_prev.items()}
        cold_pages = [set(rw["req"].cold_pages) for rw in srows]
        rids = [rw["req_id"] for rw in srows]
        job = (sf, q, snap, rids, cold_pages)
        if self.mode == "inline":
            self._result = self._run_job(job)
        elif sf.device.type == "cuda":
            self._launch_cuda(job)
        else:
            # CPU: no device to overlap, but the identical snapshot/job/commit
            # path runs, so the installation gate covers the worker code.
            self._result = self._run_job(job)

    def _launch_cuda(self, job) -> None:
        sf = job[0]
        if self.stream is None:
            self.stream = torch.cuda.Stream(sf.device)
        start = torch.cuda.Event()
        start.record(torch.cuda.current_stream(sf.device))
        self._end_event = None

        def target():
            try:
                self.stream.wait_event(start)
                with torch.cuda.stream(self.stream):
                    self._result = self._run_job(job)
                end = torch.cuda.Event()
                end.record(self.stream)
                self._end_event = end
            except Exception as ex:
                self._error = ex

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- the job
    def _run_job(self, job):
        """Select over snapshots; classify union picks; validate all cold
        sources BEFORE mutating anything; then take + reserve + H2D. Returns
        {"plan", "page_phys", "blobs"} or {"fallback": reason}."""
        sf, q, snap, rids, cold_pages = job
        cand_idx, n_cand, win_t, s_bounds, s_l2p = snap
        sel = sf.select_refresh(q, (cand_idx, n_cand, win_t, s_bounds))
        B, dev = sf.b, sf.device
        ksel = sf.tracker.k_pages
        pool = self._kv()

        # logical candidate column -> resident phys (-1 cold) snapshot, per row
        resident_of = []
        for bi in range(len(rids)):
            cols = cand_idx[bi].tolist()
            phys = s_l2p[bi].tolist()
            resident_of.append({p: phys[j] for j, p in enumerate(cols)})

        # Phase 1: union picks, classify, validate (no mutation).
        picks: dict[int, list[int]] = {bi: [] for bi in range(len(rids))}
        cold_need: dict[tuple, int] = {}   # (rid,page) -> None, ordered
        for _g, (chosen_t, nsel_t) in sel.items():
            cc = chosen_t.tolist()
            nn = [int(x) for x in nsel_t.tolist()]
            for bi in range(len(rids)):
                pages = [p for p in cc[bi][: nn[bi]] if p != 0]
                for p in pages:
                    if p not in picks[bi]:
                        picks[bi].append(p)
                    rp = resident_of[bi].get(p, -1)
                    key = (rids[bi], p)
                    if rp is not None and rp >= 0:
                        continue
                    # Cold private page only: shared-prefix adoption is not
                    # supported by the worker -> eager fallback this cycle.
                    if (p not in cold_pages[bi] or pool.cold is None
                            or key not in pool.cold):
                        return {"fallback": f"unsupported/missing page {p} rid {rids[bi]}"}
                    cold_need[key] = None
        if len(cold_need) > len(self.reserve):
            return {"fallback":
                    f"reserve {len(self.reserve)} < cold picks {len(cold_need)}"}

        # Phase 2: take + alloc-reserve + H2D (all sources validated above).
        page_phys: dict[tuple, int] = {}
        blobs = []
        promoted: list[tuple] = []  # (key, blk, blob) already taken/H2D'd
        try:
            for key in cold_need:
                blob = pool.cold.take(key)
                if blob is None:  # unreachable after phase-1; defensive
                    raise RuntimeError(f"take vanished {key}")
                blk = self.reserve.pop()
                pool.refcount[blk] = 1  # own the frame; off the free list
                self._h2d(pool, blk, blob)
                promoted.append((key, blk, blob))
                blobs.append(blob)
                page_phys[key] = blk
        except Exception as ex:
            self._rollback_promotes(pool, promoted)
            return {"fallback": f"promote failed: {ex}"}
        for bi in range(len(rids)):
            for p in picks[bi]:
                key = (rids[bi], p)
                if key not in page_phys:
                    page_phys[key] = resident_of[bi][p]

        # Phase 3: per-group fixed [B,k] phys staging from the union mapping.
        groups = {}
        for g, (chosen_t, nsel_t) in sel.items():
            phys_t = torch.zeros(B, ksel, dtype=torch.long, device=dev)
            cc = chosen_t.tolist()
            nn = [int(x) for x in nsel_t.tolist()]
            for bi in range(len(rids)):
                for j, p in enumerate(cc[bi][: nn[bi]]):
                    if p != 0:
                        phys_t[bi, j] = page_phys[(rids[bi], p)]
            groups[g] = (phys_t, chosen_t.clone(), nsel_t.clone())
        return {"picks": picks, "groups": groups,
                "page_phys": page_phys, "blobs": blobs}

    @staticmethod
    def _blob_nbytes(blob) -> int:
        return sum(t.numel() * t.element_size()
                   for name in ("k", "v", "ks", "vs")
                   if (t := blob.get(name)) is not None)

    def _rollback_promotes(self, pool, promoted) -> None:
        """Undo phase-2 promotes on a fallback: re-home each taken blob on the
        cold store so the eager refresh that follows can still resolve the
        page, and return its frame to the worker reserve (LIFO)."""
        for key, blk, blob in reversed(promoted):
            pool.cold.hold(key, blob, self._blob_nbytes(blob))
            if pool.refcount[blk] > 0:
                pool.refcount[blk] = 0
            self.reserve.append(blk)

    def _h2d(self, pool, blk, blob) -> None:
        on_side = self.stream is not None
        pool.k_pool[:, blk].copy_(blob["k"], non_blocking=on_side)
        pool.v_pool[:, blk].copy_(blob["v"], non_blocking=on_side)
        if pool.k_scale is not None:
            pool.k_scale[:, blk].copy_(blob["ks"], non_blocking=on_side)
            pool.v_scale[:, blk].copy_(blob["vs"], non_blocking=on_side)

    # ------------------------------------------------------------- carry
    def commit(self, sf, srows) -> bool:
        """Join the job and install residency on the STEP thread BEFORE
        sf.fill(). False -> the caller runs the normal eager refresh."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._end_event is not None:
            # A background overrunning the interval WAITS here (carry may wait;
            # it never falls back for time). This fences the H2D before the
            # carry replay reads the promoted frames.
            torch.cuda.current_stream(sf.device).wait_event(self._end_event)
        res = self._result
        if self._error is not None or not isinstance(res, dict):
            self.fallback_cycles += 1
            self._reset()
            return False
        if "groups" not in res:  # {"fallback": reason}
            self.fallback_cycles += 1
            self._reset()
            return False
        rt = self.rt
        B = sf.b
        dev = sf.device
        use = torch.zeros(B, dtype=torch.bool, device=dev)
        for bi, rw in enumerate(srows):
            if bi not in res["picks"]:
                continue
            r, rid = rw["req"], rw["req_id"]
            pages = res["picks"][bi]
            use[bi] = True
            # Protect picks from a full-pool own-page eviction during fill(),
            # then commit residency/l2p (the lock-free structures are written
            # only here on the step thread).
            rw["reserved"].update(pages)
            live = rt.tracker.resident.setdefault(rid, {})
            for p in pages:
                blk = res["page_phys"][(rid, p)]
                if p not in live:
                    live[p] = blk
                    if blk not in r.blocks:
                        r.blocks.append(blk)
                    if p in r.cold_pages:
                        r.cold_pages.remove(p)
                    rt.map_resident(rid, p, blk)
        self._pending = res["groups"]
        self._use_rows = use
        return True

    def arm_callback(self, sf):
        """pre_replay (after fill, before replay): install committed staging."""
        if self._pending is None:
            self.fallback_cycles += 1
            return
        # Save the committed blobs + side end event; the main stream already
        # waited the event at commit (so the carry replay is fenced), but host
        # release waits for true completion at the next prepare().
        res = self._result
        if isinstance(res, dict) and "blobs" in res:
            self._held_blobs = list(res["blobs"])
            self._held_event = self._end_event
        sf.arm_override(self._pending, self._use_rows)
        self._pending = None
        self._use_rows = None
        self.carry_cycles += 1
        self._reset()

    def _reset(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # H2D already fenced by the carry wait; host blobs can be released.
        self._result = None
        self._error = None
        self._end_event = None
        self._pending = None
        self._use_rows = None

    def close(self) -> None:
        self._reset()
        if self._held_event is not None:
            self._held_event.synchronize()
        self._held_blobs = []
        pool = self.rt.ctx.kv if self.rt.ctx is not None else None
        if pool is not None:
            for blk in self.reserve:
                if pool.refcount[blk] > 0:
                    pool.refcount[blk] = 0
                pool._free.append(blk)
        self.reserve = []
