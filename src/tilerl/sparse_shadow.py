"""PROBE-ONLY (#805, branch probe/805-serve-sm70; never merge to main as-is).

Shadow v1: a go/no-go timing probe for the 1-tick-delay async refresh. After
each captured sparse graph tick it launches, on a side stream, the work the
real delay would have to fit into one graph-tick interval:

  - QUEST: full bounds/quest/topk compute for every full-attn source plane
    with a SYNTHETIC post-rope query (same shape/dtype as the real one). The
    compute/SM cost is identical to the real q; the result is unused.
  - H2D: worst-case single-refresh churn copies — 4 groups x k(128) = 512
    cold pages of f32 K/V — into SCRATCH blocks carved out of num_blocks.

Nothing is mapped into l2p and no live tick reads the scratch, so residency
and tokens are unchanged (the CPU gate asserts shadow on/off token equality).
The real (lagged) query and the actual selection mapping are v2.

Three modes (TILERL_SPARSE_SHADOW): "quest" | "h2d" | "both" | "off".
The driver alternates graph ticks in on/off segments as a placement control
and reports per-mode graph-tick p50/p90 and background p50/p90.
"""

from __future__ import annotations

import os

import torch

from .sparse_engine import order_members, quest_scores_batched, select_members

# Per-TICK amortised promote rate the real 1-tick delay must sustain, from the
# churn window: median 100 pages replaced per group per REFRESH, a refresh runs
# every 8 ticks, over 4 source groups -> 100/8*4 = 50 pages/tick (measured
# mean 49.8). The 512 = 4*k is a whole single refresh, i.e. 8 ticks of work;
# it is reported as a worst-case upper bound, never the per-tick gate size.
SHADOW_PAGES = 50
SHADOW_PAGES_REFRESH_BURST = 4 * 128


class SparseShadow:
    def __init__(self, ctx, tracker, num_scratch: int | None = None):
        if num_scratch is None:
            num_scratch = int(os.environ.get("TILERL_SPARSE_SHADOW_PAGES",
                                             SHADOW_PAGES))
        self.ctx = ctx
        self.tracker = tracker
        self.device = ctx.backend.device
        self.cuda = self.device.type == "cuda"
        self.mode = os.environ.get("TILERL_SPARSE_SHADOW", "off")
        self.enabled = self.mode in ("quest", "h2d", "both")
        self.do_quest = self.mode in ("quest", "both")
        self.do_h2d = self.mode in ("h2d", "both")
        if self.enabled and self.mode not in ("quest", "h2d", "both"):
            raise ValueError(
                f"TILERL_SPARSE_SHADOW={self.mode!r}; want off|quest|h2d|both")
        self.stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.kv = ctx.kv
        self.cfg = tracker.cfg
        self.k_pages = tracker.k_pages
        # Carve scratch OUT of the free pool while the shadow runs; restored in
        # close(). Honest capacity reduction from num_blocks (no +k).
        self.carved = 0
        self.scratch: list[int] = []
        self._host_k = None
        self._host_v = None
        if self.enabled and self.do_h2d:
            # Keep a working margin: the live graph tick still needs its
            # resident blocks and pad. Carve at most half the current free
            # pool so the probe never starves the very ticks it measures
            # (on a tiny CPU pool this correctly carves a small number).
            want = min(num_scratch, self.kv.free_blocks // 2)
            for _ in range(want):
                self.scratch.append(self.kv._free.pop())
            self.carved = len(self.scratch)
        # timing: lists of ms
        self.bg_ms: list[float] = []
        self.ready_events = []
        # Segment switch for the on/off placement control.
        self.active_seg = True
        # synthetic q cache per plane shape, lazily allocated
        self._q: dict = {}

    # -- introspection for the verdict -----------------------------------
    def info(self) -> dict:
        per_page = (
            self.kv.k_pool.shape[0] * self.kv.k_pool.shape[2]
            * self.kv.k_pool.shape[3] * self.kv.k_pool.shape[4]
            * self.kv.k_pool.element_size()
            + self.kv.v_pool.shape[0] * self.kv.v_pool.shape[2]
            * self.kv.v_pool.shape[3] * self.kv.v_pool.shape[4]
            * self.kv.v_pool.element_size()
        )
        return {"mode": self.mode, "enabled": self.enabled,
                "carved_scratch_pages": self.carved,
                # PagedKvPool.num_blocks is fixed at construction; carving pops
                # blocks off _free, so capacity leaves via the free list, not
                # num_blocks. Report the actual free-pool before/after.
                "num_blocks_fixed": self.kv.num_blocks,
                "free_blocks_before_carve":
                    self.kv.free_blocks + self.carved,
                "free_blocks_after_carve": self.kv.free_blocks,
                "blocks_in_circulation": self.kv.num_blocks,
                "scratch_page_bytes": per_page,
                # The gate copies `carved` pages: the measured cold-promotion
                # count of ONE refresh (must finish in one tick). 512 is the
                # whole-refresh upper bound, reported for reference only.
                "h2d_bytes_gate_window": self.carved * per_page,
                "h2d_pages_whole_refresh_burst": SHADOW_PAGES_REFRESH_BURST,
                "h2d_bytes_burst_512_upper_bound":
                    SHADOW_PAGES_REFRESH_BURST * per_page}

    def _synthetic_q(self, plane: int, bounds) -> torch.Tensor:
        """[B=1,Tq=1,hq,D] post-rope-shaped query for one plane's bounds."""
        # bounds here: [Cp,Hkv,2,D] for one row; quest_batched expects
        # [B,Cp,Hkv,2,D] and q [B,Tq,hq,D].
        b = bounds
        d = b.shape[4]
        # Query head count (GQA): quest_batched reshapes q [B,T,hq,D] into
        # [B,T,hkv,hq/hkv,D], so hq must be the model's real attention-head
        # count, not derived from the KV heads.
        key = (b.shape, b.dtype)
        if key not in self._q:
            hq = self.cfg.num_attention_heads
            self._q[key] = torch.randn(1, 1, hq, d, device=self.device,
                                       dtype=torch.float32)
        return self._q[key]

    def set_active(self, on: bool) -> None:
        """Segment control: keep the constructed mode but stop/start launching
        background work, so the same process alternates off/on as a placement
        control. Resetting clears prior background timings for a clean segment."""
        self.active_seg = on

    def reset_timing(self) -> None:
        if self.stream is not None:
            self.wait_pending()
        self.bg_ms = []

    def after_graph_tick(self, rows):
        """Launch the background work after one captured graph tick. Returns a
        handle the caller queries at the NEXT graph tick's start to see whether
        the background finished within one interval: a CUDA end event (queryable
        non-blocking), or None on CPU where work runs inline (always done)."""
        if not self.enabled or not self.active_seg:
            return None
        t0 = torch.cuda.Event(enable_timing=True) if self.cuda else None
        t1 = torch.cuda.Event(enable_timing=True) if self.cuda else None
        import time

        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            cm = torch.cuda.stream(self.stream)
        else:
            cm = _NullCtx()
        if t0 is not None:
            t0.record(self.stream)
        wall0 = time.perf_counter()
        with cm:
            if self.do_quest:
                self._quest_work(rows)
            if self.do_h2d:
                self._h2d_work()
        if t1 is not None:
            t1.record(self.stream)
            self.ready_events.append(t1)
            self.bg_ms.append((t0, t1))
            return t1
        self.bg_ms.append((time.perf_counter() - wall0) * 1000.0)
        return "inline-done"

    def _quest_work(self, rows) -> None:
        # Score every source plane over that row's full candidate bounds with a
        # synthetic q. Mirrors SparseForward._select_device's device math minus
        # the residency mask / l2p gather (those are cheap and residency-bound;
        # the SM contention under test is quest+topk).
        if not rows:
            return
        r = rows[0]
        rid = r["req_id"]
        cand = r["cand"]
        if not cand:
            return
        nc = torch.tensor([len(cand)], device=self.device)
        win = torch.tensor([int(r["force_window"])], device=self.device)
        eligible = torch.ones(1, len(cand), dtype=torch.bool, device=self.device)
        for plane in self.tracker.src_planes:
            bounds = self.tracker.bounds_rows(rid, plane, cand).unsqueeze(0)
            q = self._synthetic_q(plane, bounds)
            scores = quest_scores_batched(q, bounds)
            member = select_members(scores, nc, self.k_pages, win, eligible)
            order_members(member, self.k_pages)

    def _h2d_work(self) -> None:
        # Worst-case cold H2D into the carved scratch planes: one pinned host
        # buffer per plane shape, non-blocking copies for all scratch pages.
        # The pinned sources are retained on self until the next main-stream
        # wait, so an in-flight copy never reads a reused/freed host buffer.
        if not self.scratch:
            return
        # k_pool is [plane, block, h, t, d]; k_pool[:, blk] is one block's
        # planes [plane,h,t,d]. The host staging buffer matches that exactly.
        kshape = (self.kv.k_pool.shape[0], *self.kv.k_pool.shape[2:])
        vshape = (self.kv.v_pool.shape[0], *self.kv.v_pool.shape[2:])
        self._host_k = torch.empty(kshape, dtype=self.kv.k_pool.dtype,
                                   device="cpu", pin_memory=self.cuda)
        self._host_v = torch.empty(vshape, dtype=self.kv.v_pool.dtype,
                                   device="cpu", pin_memory=self.cuda)
        for blk in self.scratch:
            self.kv.k_pool[:, blk].copy_(self._host_k, non_blocking=self.cuda)
            self.kv.v_pool[:, blk].copy_(self._host_v, non_blocking=self.cuda)

    def wait_pending(self) -> None:
        """Main-stream wait before the next measured graph tick (so timing is
        bounded correctly); called by the harness when it wants a conservative
        interval measurement."""
        if self.stream is not None and self.ready_events:
            cur = torch.cuda.current_stream(self.device)
            for ev in self.ready_events:
                cur.wait_event(ev)
            self.ready_events = []

    def background_ms(self) -> list[float]:
        if self.cuda:
            torch.cuda.synchronize(self.device)
            return [a.elapsed_time(b) for a, b in self.bg_ms]
        return list(self.bg_ms)

    def close(self) -> None:
        """Wait for the side stream and return scratch blocks to the pool."""
        if not self.enabled:
            return
        if self.stream is not None:
            self.wait_pending()
            torch.cuda.current_stream(self.device).synchronize()
        for blk in self.scratch:
            if blk not in self.kv._free:
                self.kv._free.append(blk)
        self.scratch = []
        self.carved = 0
        self._host_k = self._host_v = None


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
