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
        rids = tuple(rw["req_id"] for rw in srows)
        self._job_rids = rids
        job = (sf, q, snap, rids, cold_pages)
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
        sf, q, snap, rids, cold_pages = job
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
        cold_need: list[tuple] = []
        for _g, (chosen_t, nsel_t) in sel.items():
            cc = chosen_t.tolist()
            nn = [int(x) for x in nsel_t.tolist()]
            for bi in range(len(rids)):
                for p in cc[bi][: nn[bi]]:
                    if p not in picks[bi]:
                        picks[bi].append(p)
                    key = (rids[bi], p)
                    rp = resident_of[bi].get(p, -1)
                    if rp is not None and rp >= 0:
                        continue
                    if (p not in cold_pages[bi] or pool.cold is None
                            or key not in pool.cold):
                        return {"fallback": f"unsupported/missing page {p} rid {rids[bi]}"}
                    cold_need.append(key)
        # de-dup cold_need preserving order
        seen = set()
        cold_need = [k for k in cold_need if not (k in seen or seen.add(k))]
        if len(cold_need) > len(self.reserve):
            return {"fallback":
                    f"reserve {len(self.reserve)} < cold picks {len(cold_need)}"}

        page_phys: dict[tuple, int] = {}
        blobs = []
        promoted: list[tuple] = []
        try:
            # Phase 2: take + reserve alloc + side-stream H2D.
            for key in cold_need:
                blob = pool.cold.take(key)
                if blob is None:
                    raise RuntimeError(f"take vanished {key}")
                blk = self.reserve.pop()
                pool.refcount[blk] = 1
                self._h2d(pool, blk, blob)
                promoted.append((key, blk, blob))
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
            return {"fallback": f"promote failed: {ex}"}
        return {"picks": picks, "groups": groups,
                "page_phys": page_phys, "blobs": blobs,
                "promoted": promoted}

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
        for key, blk, blob in reversed(promoted):
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
            self._reset()
            return False
        rt = self.rt
        dev = sf.device
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
