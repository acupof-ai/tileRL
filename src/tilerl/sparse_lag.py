"""PROBE-ONLY (#805, branch probe/805-v2; never merge to main as-is).

v2 real 1-tick-delay sparse refresh — controller + promote worker.

Cadence (SPARSE_REFRESH_TICKS=8): seven plain captured graph ticks, then one
CARRY tick. At the END of plain tick 7 the controller snapshots the just-played
staging and previous q and runs ONE job:

    select_refresh (quest over ALL candidates, no residency mask)
      -> every union pick: resident (reuse phys) or cold private
         (HostKvPages.take mmap + H2D into a RESERVED block)
      -> per-row, per-group (phys, chosen, nsel)

On the carry tick the controller commits BEFORE fill(): joins the worker and
(the main stream waits the side-stream end event — a background that overruns
one interval makes this tick wait, it never falls back for TIME), takes every
cold blob and maps residency on the step thread, adds the picks to the carry
tick's reserved set so fill()'s own-page resolve cannot evict them. pre_replay
then arms the physical blocks into the captured sf. A job that is absent,
fails, names an unsupported (shared-prefix/missing) page, or belongs to a
request set that changed returns False WITHOUT touching the cadence counter,
so the caller falls through to the normal all-candidate EAGER refresh.

Logical page 0 is real (early context). Chosen ids are 0-padded only by
POSITION (tail slots >= nsel); the worker indexes [:nsel] and never filters on
the value 0, so a valid pick of page 0 is not mistaken for padding.

Two modes, ONE job path:
  inline: job runs immediately on the step thread over LIVE staging.
  async:  over post-replay SNAPSHOTS; CUDA -> worker thread + side stream
          (real overlap); CPU -> immediate but through the same snapshot/job/
          commit code. The side-stream end event is recorded in a finally, so a
          job exception can never suppress the commit fence; failed promotes
          roll back (blobs re-homed, frames returned to reserve) AFTER the side
          stream is synchronized.

Env: TILERL_SPARSE_V2=inline|async|off; TILERL_SPARSE_V2_RESERVE reserve pages.
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
        # Frames carved from pool._free, owned exclusively by a job until commit
        # maps them. _free is never touched mid-job.
        self.reserve: list[int] = []
        self.stream = None
        self._thread = None
        self._result = None
        self._error = None
        self._end_event = None
        self._pending = None
        self._use_rows = None
        # Request ids the in-flight job was built for; commit refuses a changed
        # request set (a request ending between prepare and carry).
        self._job_rids: tuple | None = None
        self._held_blobs: list = []
        self._held_event = None
        self.carry_cycles = 0
        self.fallback_cycles = 0
        # B>1 carries are refused: the captured width-2 sparse-graph verify
        # path has an unresolved CUDA illegal-access. Flip only from the CPU
        # negative-control gate to prove this guard is what blocks arming.
        self.enforce_b1 = True
        self.b1_guard_fallbacks = 0
        # PROBE-ONLY diagnostics: why each eager fallback happened. Counts the
        # classified reasons and keeps the last message; a 17/17 fallback on a
        # new geometry is otherwise invisible (the string was discarded).
        self.fallback_reasons: dict[str, int] = {}
        self.last_fallback_reason = None
        self.last_fallback_diag = None

    # ------------------------------------------------------------ cadence
    def is_carry(self) -> bool:
        from .sparse_engine import SPARSE_REFRESH_TICKS

        return self.rt.ticks_since_refresh + 1 >= SPARSE_REFRESH_TICKS

    def is_ready(self) -> bool:
        return self._thread is not None or self._result is not None

    def _kv(self):
        return self.rt.ctx.kv

    def _topup_reserve(self) -> None:
        pool = self._kv()
        n = min(self.reserve_cap - len(self.reserve), pool.free_blocks // 2)
        for _ in range(n):
            self.reserve.append(pool._free.pop())

    def _release_prior_held(self) -> None:
        """Release the previous carry's host blob sources after their side
        H2D is host-side complete. Runs on a plain tick, off the carry path."""
        if self._held_event is not None:
            self._held_event.synchronize()
            self._held_event = None
        self._held_blobs = []

    # ----------------------------------------------------- plain tick 7
    def prepare(self, sf, srows) -> None:
        self._release_prior_held()
        self._reset()
        q_prev = getattr(sf, "_q_prev", None)
        if not q_prev:
            return  # no captured q yet: first carry falls back
        if self.enforce_b1 and len(srows) != 1:
            # Common B>1 case: never snapshot, top up, take, or start a thread.
            # The sentinel makes is_ready() true so the runtime still reaches
            # commit(), which counts the guard refusal and runs eager refresh.
            self._result = {"b1_guard": True}
            return
        self._topup_reserve()
        snap = (sf.cand_idx.clone(), sf.n_cand.clone(), sf.win.clone(),
                ([b.clone() for b in sf.s_bounds] if sf.s_bounds else None),
                sf.s_l2p.clone())
        q = {g: t.clone() for g, t in q_prev.items()}
        cold_pages = [set(rw["req"].cold_pages) for rw in srows]
        # Shared-prefix pages (#796) resolve through tracker.shared ->
        # share_take/shared_promote, NOT the private cold tier. Snapshot the
        # page->content-key map so the side stream classifies a picked shared
        # page correctly; without this a prefix-adopting row's early pages
        # (e.g. logical page 0) read as "unsupported" and every carry fell
        # back eager.
        shared_maps = [dict(self.rt.tracker.shared.get(rw["req_id"], {}))
                       for rw in srows]
        rids = tuple(rw["req_id"] for rw in srows)
        self._job_rids = rids
        job = (sf, q, snap, rids, cold_pages, shared_maps)
        if self.mode == "inline" or sf.device.type != "cuda":
            self._result = self._run_job(job)
        else:
            self._launch_cuda(job)

    def _launch_cuda(self, job) -> None:
        sf = job[0]
        if self.stream is None:
            self.stream = torch.cuda.Stream(sf.device)
        start = torch.cuda.Event()
        start.record(torch.cuda.current_stream(sf.device))
        self._end_event = None

        def target():
            end = None
            try:
                self.stream.wait_event(start)
                with torch.cuda.stream(self.stream):
                    self._result = self._run_job(job)
                end = torch.cuda.Event()
                end.record(self.stream)
            except Exception as ex:
                self._error = ex
            finally:
                # Recorded even on failure so commit() always fences the side
                # stream before reusing frames or running the eager fallback.
                if end is None:
                    end = torch.cuda.Event()
                    end.record(self.stream)
                self._end_event = end

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- the job
    def _run_job(self, job):
        sf, q, snap, rids, cold_pages, shared_maps = job
        cand_idx, n_cand, win_t, s_bounds, s_l2p = snap
        # sel: {group: (chosen_logical positional-padded, nsel)}.
        sel = sf.select_refresh(q, (cand_idx, n_cand, win_t, s_bounds))
        B, dev = sf.b, sf.device
        pool = self._kv()

        resident_of = []
        for bi in range(len(rids)):
            cols = cand_idx[bi].tolist()
            phys = s_l2p[bi].tolist()
            resident_of.append({p: phys[j] for j, p in enumerate(cols)})

        # Valid chosen pages are positions [:nsel]; padding is positional, so a
        # real logical page 0 in a valid slot is preserved (never filter ==0).
        picks: dict[int, list[int]] = {bi: [] for bi in range(len(rids))}
        # Each non-resident pick is one of: a PRIVATE cold page (blob keyed
        # (rid,page), take()) or a SHARED prefix page (content key in
        # shared_maps, share_take() read-only). Both H2D into a reserve frame;
        # they differ only in blob acquisition and post-commit bookkeeping.
        cold_need: list[tuple] = []     # (rid, page) -> private cold
        shared_need: list[tuple] = []   # (rid, page, content_key)
        for _g, (chosen_t, nsel_t) in sel.items():
            cc = chosen_t.tolist()
            nn = [int(x) for x in nsel_t.tolist()]
            for bi in range(len(rids)):
                for p in cc[bi][: nn[bi]]:
                    if p not in picks[bi]:
                        picks[bi].append(p)
                    rp = resident_of[bi].get(p, -1)
                    if rp is not None and rp >= 0:
                        continue
                    key = (rids[bi], p)
                    if p in cold_pages[bi] and pool.cold is not None \
                            and key in pool.cold:
                        cold_need.append(key)
                    elif p in shared_maps[bi] and pool.cold is not None:
                        shared_need.append((rids[bi], p, shared_maps[bi][p]))
                    else:
                        # PROBE DIAGNOSTIC: classify why a selected early page
                        # cannot be promoted. One-time dump into the instance.
                        diag = {
                            "page": p, "rid": rids[bi],
                            "in_cold_pages": p in cold_pages[bi],
                            "in_shared": p in shared_maps[bi],
                            "in_resident_snap": p in resident_of[bi],
                            "private_key_in_cold": key in pool.cold,
                            "cold_pages_n": len(cold_pages[bi]),
                            "shared_n": len(shared_maps[bi]),
                            "cold_tier_keys_sample":
                                (p in cold_pages[bi]) and (key in pool.cold),
                            "cold_minmax":
                                (min(cold_pages[bi]), max(cold_pages[bi]))
                                if cold_pages[bi] else None,
                            "shared_minmax":
                                (min(shared_maps[bi]), max(shared_maps[bi]))
                                if shared_maps[bi] else None,
                        }
                        return {"fallback":
                                f"unsupported/missing page {p} rid {rids[bi]}",
                                "diag": diag}
        # de-dup preserving order
        seen = set()
        cold_need = [k for k in cold_need if not (k in seen or seen.add(k))]
        seen = set()
        shared_need = [k for k in shared_need
                       if not (k[:2] in seen or seen.add(k[:2]))]
        n_promote = len(cold_need) + len(shared_need)
        if n_promote > len(self.reserve):
            return {"fallback":
                    f"reserve {len(self.reserve)} < picks {n_promote} "
                    f"(cold {len(cold_need)} shared {len(shared_need)})"}

        page_phys: dict[tuple, int] = {}
        blobs = []
        # promoted entries: (key, blk, blob, is_shared, content_key|None).
        promoted: list[tuple] = []
        # Content keys pinned for the async H2D window AND kept after commit so
        # a capacity LRU cannot delete a blob this row may re-resolve. The same
        # pin the eager resolve path takes; unpinned in drop()/rollback.
        pinned_keys: list[int] = []
        try:
            # Phase 2: take/share + reserve alloc + side-stream H2D.
            for key in cold_need:
                blob = pool.cold.take(key)
                if blob is None:
                    raise RuntimeError(f"take vanished {key}")
                blk = self.reserve.pop()
                pool.refcount[blk] = 1
                self._h2d(pool, blk, blob)
                promoted.append((key, blk, blob, False, None))
                blobs.append(blob)
                page_phys[key] = blk
            for rid, p, content_key in shared_need:
                blob = pool.cold.share_take(content_key) if pool.cold else None
                if blob is None:
                    # The shared label outlived its index entry: resolve() turns
                    # this into a fresh never-written block, which a lag carry
                    # cannot populate (it only has the shared blob). Fall back to
                    # the eager refresh, which allocates the fresh block.
                    return {"fallback":
                            f"shared label aged out page {p} rid {rid}"}
                if content_key not in pinned_keys \
                        and pool.cold.pin_if_present(content_key):
                    pinned_keys.append(content_key)
                blk = self.reserve.pop()
                pool.refcount[blk] = 1
                self._h2d(pool, blk, blob)
                key = (rid, p)
                promoted.append((key, blk, blob, True, content_key))
                blobs.append(blob)
                page_phys[key] = blk
            for bi in range(len(rids)):
                for p in picks[bi]:
                    key = (rids[bi], p)
                    if key not in page_phys:
                        page_phys[key] = resident_of[bi][p]
            # Phase 3: per-group fixed [B,k] phys staging. Width matches the
            # captured replay's k = min(k_pages, cmax), using sel's own tensor
            # width (never a hard-coded k_pages), so small-cmax buckets cannot
            # broadcast-mismatch.
            groups = {}
            for g, (chosen_t, nsel_t) in sel.items():
                kw = chosen_t.shape[1]
                phys_t = torch.zeros(B, kw, dtype=torch.long, device=dev)
                cc = chosen_t.tolist()
                nn = [int(x) for x in nsel_t.tolist()]
                for bi in range(len(rids)):
                    for j, p in enumerate(cc[bi][: nn[bi]]):
                        phys_t[bi, j] = page_phys[(rids[bi], p)]
                groups[g] = (phys_t, chosen_t.clone(), nsel_t.clone())
        except Exception as ex:
            # Side work (incl. any queued H2D) must finish before re-homing
            # blobs / returning frames, so the eager fallback cannot read a
            # half-DMA frame or free a DMA source mid-copy.
            if self.stream is not None:
                self.stream.synchronize()
            self._rollback_promotes(pool, promoted)
            for ck in pinned_keys:
                pool.cold.unpin(ck)
            return {"fallback": f"promote failed: {ex}"}
        return {"picks": picks, "groups": groups,
                "page_phys": page_phys, "blobs": blobs,
                "promoted": promoted, "pinned_keys": pinned_keys,
                "shared_pages": {rid: [p for (_r, p, c) in shared_need
                                        if _r == rid] for rid in rids}}

    def _h2d(self, pool, blk, blob) -> None:
        on_side = self.stream is not None
        pool.k_pool[:, blk].copy_(blob["k"], non_blocking=on_side)
        pool.v_pool[:, blk].copy_(blob["v"], non_blocking=on_side)
        if pool.k_scale is not None:
            pool.k_scale[:, blk].copy_(blob["ks"], non_blocking=on_side)
            pool.v_scale[:, blk].copy_(blob["vs"], non_blocking=on_side)

    @staticmethod
    def _blob_nbytes(blob) -> int:
        return sum(t.numel() * t.element_size()
                   for name in ("k", "v", "ks", "vs")
                   if (t := blob.get(name)) is not None)

    def _rollback_promotes(self, pool, promoted) -> None:
        for entry in reversed(promoted):
            key, blk, blob, is_shared, _content_key = entry
            if not is_shared:
                # Private blob re-homed into its (rid,page) host slot. A shared
                # blob is a read-only reference owned by the prefix index; it
                # just drops here (its pin, if any, is released separately).
                pool.cold.hold(key, blob, self._blob_nbytes(blob))
            if pool.refcount[blk] > 0:
                pool.refcount[blk] = 0
            self.reserve.append(blk)

    # ------------------------------------------------------------- carry
    def commit(self, sf, srows) -> bool:
        """Join the job and install residency on the STEP thread BEFORE
        sf.fill(). False (counter left untouched) -> caller runs the normal
        eager all-candidate refresh."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._end_event is not None:
            # Overrun WAITS; also the mandatory fence before reusing frames.
            torch.cuda.current_stream(sf.device).wait_event(self._end_event)
        res = self._result
        rids_now = tuple(rw["req_id"] for rw in srows)
        # B>1 is unverified for the captured width-2 sparse-graph verify path
        # (a production CUDA illegal-access is under investigation there). v2
        # rides that exact path, so never ARM a graph carry with >1 active row:
        # fall through to the normal eager refresh and count it. This makes any
        # v2 GO statement B=1-scoped until B>1 is settled.
        bad = (self._error is not None or not isinstance(res, dict)
               or "groups" not in res
               or self._job_rids != rids_now)
        if bad:
            if isinstance(res, dict) and res.get("b1_guard"):
                # prepare() refused a B>1 carry before taking anything; no
                # frames can be held, but count it distinctly.
                self.b1_guard_fallbacks += 1
                reason = "b1_guard"
            elif self._error is not None:
                reason = f"exception: {type(self._error).__name__}: {self._error}"
            elif not isinstance(res, dict) or "groups" not in res:
                reason = (res.get("fallback", "job-without-groups")
                          if isinstance(res, dict) else f"non-dict {type(res).__name__}")
            elif self._job_rids != rids_now:
                reason = "stale_rids"
            else:
                reason = "unknown"
            key = reason.split(":")[0].split()[0]
            self.fallback_reasons[key] = self.fallback_reasons.get(key, 0) + 1
            self.last_fallback_reason = reason
            if isinstance(res, dict) and res.get("diag"):
                self.last_fallback_diag = res["diag"]
            # Stale request set, failed job, or B>1 guard: eager fallback. Any
            # promoted frames in a fallback dict were rolled back in the job.
            self.fallback_cycles += 1
            # A completed job refused for a changed request set still owns its
            # taken blobs and reserve frames; re-home them before eager refresh.
            promoted = res.get("promoted") if isinstance(res, dict) else None
            if promoted:
                if self.stream is not None:
                    self.stream.synchronize()
                self._rollback_promotes(self._kv(), promoted)
                for ck in (res.get("pinned_keys") or []):
                    self._kv().cold.unpin(ck)
            self._reset()
            return False
        rt = self.rt
        dev = sf.device
        # Adopt the shared-content pins the job took: register them on the
        # tracker's per-rid pin set so drop() releases them once, the same
        # ownership the eager resolve path establishes. B==1 (enforce_b1).
        if res.get("pinned_keys"):
            pins_by_rid = rt.tracker.request_pins
            for rid in rids_now:
                if res.get("shared_pages", {}).get(rid):
                    pins_by_rid.setdefault(rid, set()).update(res["pinned_keys"])
        use = torch.zeros(sf.b, dtype=torch.bool, device=dev)
        for bi, rw in enumerate(srows):
            if bi not in res["picks"]:
                continue
            r, rid = rw["req"], rw["req_id"]
            pages = res["picks"][bi]
            use[bi] = True
            rw["reserved"].update(pages)  # protect from own-page eviction
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
        res = self._result
        if isinstance(res, dict) and "blobs" in res:
            # Released at the next prepare() after the side event host-syncs.
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
        self._result = None
        self._error = None
        self._end_event = None
        self._pending = None
        self._use_rows = None
        self._job_rids = None

    def close(self) -> None:
        """Cancel/join any worker, release held blobs, return reserve frames.
        Called by engine.shutdown when the v2 controller exists."""
        self._reset()
        if self._held_event is not None:
            self._held_event.synchronize()
            self._held_event = None
        self._held_blobs = []
        pool = self.rt.ctx.kv if self.rt.ctx is not None else None
        if pool is not None:
            for blk in self.reserve:
                if pool.refcount[blk] > 0:
                    pool.refcount[blk] = 0
                pool._free.append(blk)
        self.reserve = []
