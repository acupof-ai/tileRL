"""Captured decode graphs: one replay per (batch, width) bucket.

Split out of engine.py (docs/design-architecture.md step 10a). This module
owns CAPTURE and REPLAY keyed by (B, W) for dense and sparse steady-state
decode; the Engine owns when a tick uses a graph, the per-bucket cache and
the capture-failure fallback. No symbol here takes or imports an Engine -
pools, model and backend arrive as arguments.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from .kv_cache import BatchKv

#: Decode-graph size ladder: a tick pads up to the first bucket >= its row count.
GRAPH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

#: Forward phase a row is in while chunk-prefilling; graph staging fills its
#: chunk tokens rather than a decode chain. Shared with engine without an import
#: cycle via the same int literal the engine defines.
PHASE_PREFILL = 1


def graph_bucket(rows: int, max_batch: int) -> int:
    """The batch dimension a tick of ``rows`` decodes keys its graph on: the
    next bucket up, or the exact size above the ladder. ``precapture`` walks
    this over every admissible row count, so the two cannot disagree about
    which graphs exist."""
    b = next((c for c in GRAPH_BUCKETS if c >= rows), None)
    return rows if b is None or max_batch < b else b


class DecodeGraph:
    """Captured ``model.forward`` for one (batch, width) bucket: per tick, small
    H2D copies of the inputs plus one replay. Replay mutates the engine pools
    pools like the eager path; warmup writes to block 0 / slot 0 are overwritten
    before any real request reads them.
    """

    def __init__(
        self,
        model,
        backend,
        kv_pool,
        state_pool,
        batch_size,
        width=1,
        pool=None,
        last_only=False,
        keep=0,
        aux_layers=(),
    ):
        device = backend.device
        B, W = batch_size, width
        # int32 end to end: a long buffer costs a cast launch per use inside the graph.
        self._b = B
        self._w = W
        self._ids = torch.empty(B, W, dtype=torch.int32, device=device)
        self._pos = torch.empty(B, W, dtype=torch.int32, device=device)
        self._bt = torch.zeros(B, kv_pool.num_blocks, dtype=torch.int32, device=device)
        self._sl = torch.empty(B, dtype=torch.int32, device=device)
        self._ss = torch.empty(B, dtype=torch.int32, device=device)
        # Uniform W per row, as a static device buffer: a CPU->GPU fallback breaks capture.
        self._sql = torch.full((B,), W, dtype=torch.int32, device=device)
        # Pinned staging: an unpinned H2D copy_ is synchronous, ms per tick under contention.
        self._ids_h = torch.empty(B, W, dtype=torch.int32, pin_memory=True)
        self._pos_h = torch.empty(B, W, dtype=torch.int32, pin_memory=True)
        self._bt_h = torch.zeros(B, kv_pool.num_blocks, dtype=torch.int32, pin_memory=True)
        self._sl_h = torch.empty(B, dtype=torch.int32, pin_memory=True)
        self._ss_h = torch.empty(B, dtype=torch.int32, pin_memory=True)
        self._kv = BatchKv(
            block_table=self._bt,
            seq_len=self._sl,
            state_slot=self._ss,
            kv_pool=kv_pool,
            state_pool=state_pool,
            seq_q_lens=self._sql,
            keep_steps=keep,  # verify ticks only; a W>1 prefill chunk has no step buffers
        )
        # Warmup on a side stream: tilelang JIT (host work) must finish before capture.
        self._ids.fill_(0)
        self._pos.fill_(0)
        self._sl.fill_(W)
        self._ss.fill_(0)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model.forward(self._ids, self._pos, self._kv, backend, last_only=last_only)
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        hid: list = []
        # One memory pool across buckets: a private pool per graph is never returned.
        with torch.cuda.graph(self._graph, pool=pool):
            self._logits = model.forward(
                self._ids,
                self._pos,
                self._kv,
                backend,
                hidden_out=hid,
                last_only=last_only,
                aux_layers=aux_layers,
            )
            # inside the capture, so replay rewrites it like every other static buffer
            aux = torch.cat(hid[: len(aux_layers)], -1) if aux_layers else None
        self.hidden = hid[-1] if hid else None  # rewritten in place by every replay
        self.aux = aux

    def run(self, reqs, chains=None, pad=None):
        """Copy per-tick inputs into the static buffers and replay; returns the
        static logits [B,W,V], valid until the next replay. ``chains[i]`` is row
        i's ``[last committed token, drafts...]``. ``pad`` is ``(state_slot,
        block)`` for rows beyond ``len(reqs)``: padding rows still write to
        the pools, so they must not land on a slot a live request owns."""
        for i, r in enumerate(reqs):
            if r.phase == PHASE_PREFILL:
                start = r.prefill_from
                for j, tok in enumerate(r.tokens[start : start + self._w]):
                    self._ids_h[i, j] = tok
                    self._pos_h[i, j] = start + j
                self._sl_h[i] = start + self._w
            else:
                chain = chains[i] if chains else (r.output[-1],)
                for j, tok in enumerate(chain):
                    self._ids_h[i, j] = tok
                    self._pos_h[i, j] = r.seq_len - 1 + j
                self._sl_h[i] = r.seq_len - 1 + self._w
            self._ss_h[i] = r.state_slot
            n = len(r.blocks)
            self._bt_h[i, :n] = torch.tensor(r.blocks, dtype=torch.int32)
        if pad is not None and len(reqs) < self._b:
            pad_slot, pad_block = pad
            for i in range(len(reqs), self._b):
                self._ids_h[i, :] = 0
                self._pos_h[i, :] = 0
                self._sl_h[i] = self._w
                self._ss_h[i] = pad_slot
                self._bt_h[i, :] = pad_block
        self._ids.copy_(self._ids_h, non_blocking=True)
        self._pos.copy_(self._pos_h, non_blocking=True)
        self._sl.copy_(self._sl_h, non_blocking=True)
        self._ss.copy_(self._ss_h, non_blocking=True)
        self._bt.copy_(self._bt_h, non_blocking=True)
        self._graph.replay()
        return self._logits


class SparseDecodeGraph:
    """Captured sparse decode/verify tick for one (B, W, cmax) bucket.

    Dense ``_DecodeGraph`` cannot serve sparse rows: it replays the FULL dense
    block table, but a sparse row's ``r.blocks`` holds only the hot set and
    attention reads the packed [selected ; own] table the SparseForward builds.
    This wrapper captures a forward bound to a PERSISTENT, refillable
    SparseForward (reuse=True):

    - ``fill()`` runs OUTSIDE capture: it resolves the own window, and gathers
      each row's candidate l2p/bounds into fixed staging tensors.
    - the replay only reads those staging tensors, so selection, the packed
      table and write offsets have one fixed shape per bucket and zero host
      syncs. No promotion happens inside capture; a tick that needs one (or a
      refresh, prefill, or a larger cmax bucket) runs eager instead.
    """

    def __init__(
        self,
        model,
        backend,
        kv_pool,
        state_pool,
        tracker,
        sf,
        batch_size,
        width,
        pool=None,
        aux_layers=(),
    ):
        device = backend.device
        B, W = batch_size, width
        self._b, self._w = B, W
        self.sf = sf
        self.tm = None  # set by SparseRuntime.run_decode_graph when step timing is on
        self._ids = torch.zeros(B, W, dtype=torch.long, device=device)
        self._pos = torch.zeros(B, W, dtype=torch.long, device=device)
        self._slots = torch.zeros(B, dtype=torch.long, device=device)
        # full per-row logical length for write_tokens offset arithmetic
        self._sl = torch.zeros(B, dtype=torch.long, device=device)
        # valid query count per row (chain width for live rows)
        self._sql = torch.full((B,), W, dtype=torch.long, device=device)
        self._kv = BatchKv(
            block_table=sf.own_table,
            seq_len=self._sl,
            state_slot=self._slots,
            kv_pool=kv_pool,
            state_pool=state_pool,
            seq_q_lens=self._sql,
            keep_steps=W if W > 1 else 0,  # match eager: a verify tick keeps every step
            page_base=sf.page_base,
            sparse=sf,
        )
        # Pinned host staging for the per-tick ids/pos/scalars. Filling a pinned
        # tensor and one non_blocking copy_ replaces a torch.tensor(list,
        # device=) per field per row, which allocates and forces a synchronous
        # H2D each tick (part of the residual post-#678 out-of-graph stalls).
        # Long here, not int32: sparse packed tables still index long; int32 is
        # a later lever gated on a sparse-kernel dtype pass. CUDA-only class.
        pin = str(device) == "cuda"
        self._ids_h = torch.zeros(B, W, dtype=torch.long, device="cpu", pin_memory=pin)
        self._pos_h = torch.zeros(B, W, dtype=torch.long, device="cpu", pin_memory=pin)
        self._sl_h = torch.zeros(B, dtype=torch.long, device="cpu", pin_memory=pin)
        self._slots_h = torch.zeros(B, dtype=torch.long, device="cpu", pin_memory=pin)
        self._sql_h = torch.full((B,), W, dtype=torch.long, device="cpu", pin_memory=pin)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model.forward(
                    self._ids, self._pos, self._kv, backend, last_only=False, aux_layers=aux_layers
                )
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        hid: list = []
        with torch.cuda.graph(self._graph, pool=pool):
            self._logits = model.forward(
                self._ids,
                self._pos,
                self._kv,
                backend,
                hidden_out=hid,
                last_only=False,
                aux_layers=aux_layers,
            )
            aux = torch.cat(hid[: len(aux_layers)], -1) if aux_layers else None
        self.hidden = hid[-1] if hid else None
        self.aux = aux

    def run(self, srows, chains, pad=None):
        """Fill every static input and the sparse staging buffers, then replay.
        ``srows`` are the decode geometry dicts (carry ``req``); pad slots past
        n use ``(slot, block)``."""
        W = self._w
        sf = self.sf
        tm = self.tm
        t = time.perf_counter()
        sf.fill(srows)
        if tm is not None:
            tm.mark("p0_fill", t)
            t = time.perf_counter()
        self.fill_staging(
            srows,
            chains,
            W,
            self._ids_h,
            self._pos_h,
            self._sl_h,
            self._slots_h,
            self._sql_h,
            sf,
            pad,
        )
        self._ids.copy_(self._ids_h, non_blocking=True)
        self._pos.copy_(self._pos_h, non_blocking=True)
        self._sl.copy_(self._sl_h, non_blocking=True)
        self._slots.copy_(self._slots_h, non_blocking=True)
        self._sql.copy_(self._sql_h, non_blocking=True)
        if tm is not None:
            tm.mark("p1_h2d", t)
            t = time.perf_counter()
        self._graph.replay()
        if tm is not None:
            tm.mark("p2_replay", t)
        return self._logits

    @staticmethod
    def fill_staging(srows, chains, W, ids_h, pos_h, sl_h, slots_h, sql_h, sf, pad=None):
        """Write one decode tick's inputs into the pinned host staging tensors.
        Pure host indexing (no device), so it is unit-tested on CPU: ids are the
        per-row chains, pos the [seq_len-1 ..] range, scalars the slot/logical
        length/chain width; rows past len(srows) take the pad frame. The device
        buffers then receive one non_blocking copy_. Kept free of CUDA so the
        fill values are byte-for-byte checkable without a card."""
        B = ids_h.shape[0]
        ids_h.zero_()
        pos_h.zero_()
        for i, rw in enumerate(srows):
            r = rw["req"]
            chain = chains[i] if chains else (r.output[-1],)
            for j, tok in enumerate(chain):
                ids_h[i, j] = tok
                pos_h[i, j] = r.seq_len - 1 + j
            sl_h[i] = r.seq_len - 1 + W
            slots_h[i] = r.state_slot
            sql_h[i] = len(chain)
        n = len(srows)
        if pad is not None and n < B:
            pad_slot, pad_block = pad
            for i in range(n, B):
                sl_h[i] = W
                slots_h[i] = pad_slot
                sql_h[i] = W
                if sf is not None:
                    sf.own_table[i, 0] = pad_block


class CpuSparseGraph:
    """CPU test seam for _SparseDecodeGraph: fills the same persistent
    SparseForward staging buffers and runs the model eagerly, so the
    selection/packed-table math the captured graph replays is exercised without a
    CUDA card. The tensors the selection reads are the captured-sf staging ones;
    a test asserts they are reused (stable data_ptr) and no host sync fires."""

    def __init__(self, model, backend, kv_pool, state_pool, sf, B, W):
        self.sf = sf
        self.tm = None  # set by SparseRuntime.run_decode_graph when step timing is on
        self._model, self._backend, self._kv, self._states = (model, backend, kv_pool, state_pool)
        self._b, self._w = B, W
        self.hidden = None

    def run(self, srows, chains, pad=None):
        sf = self.sf
        tm = self.tm
        t = time.perf_counter()
        sf.fill(srows)
        if tm is not None:
            tm.mark("p0_fill", t)
            t = time.perf_counter()
        B, W = self._b, self._w
        ids = torch.zeros(B, W, dtype=torch.long)
        pos = torch.zeros(B, W, dtype=torch.long)
        sl = torch.zeros(B, dtype=torch.long)
        ss = torch.zeros(B, dtype=torch.long)
        sql = torch.full((B,), W, dtype=torch.long)
        for i, rw in enumerate(srows):
            r = rw["req"]
            chain = chains[i]
            ids[i, : len(chain)] = torch.tensor(chain)
            pos[i, : len(chain)] = torch.arange(r.seq_len - 1, r.seq_len - 1 + len(chain))
            sl[i] = r.seq_len - 1 + W
            ss[i] = r.state_slot
            sql[i] = len(chain)
        for i in range(len(srows), B):
            ss[i] = pad[0] if pad else 0
            if pad:
                sf.own_table[i, 0] = pad[1]
        kv = BatchKv(
            block_table=sf.own_table,
            seq_len=sl,
            state_slot=ss,
            kv_pool=self._kv,
            state_pool=self._states,
            seq_q_lens=sql,
            keep_steps=W if W > 1 else 0,  # match eager: a verify tick keeps every step
            page_base=sf.page_base,
            sparse=sf,
        )
        if tm is not None:
            # CPU has no device H2D; this is the host staging-build analogue.
            tm.mark("p1_h2d", t)
            t = time.perf_counter()
        hid: list = []
        logits = self._model.forward(ids, pos, kv, self._backend, hidden_out=hid, last_only=False)
        self.hidden = hid[-1] if hid else None
        if tm is not None:
            tm.mark("p2_replay", t)
        return logits


class GraphCapture:
    """Shared capture infrastructure for dense and sparse decode graphs:

    - one CUDA graph memory ``pool`` across every bucket (a private pool per
      graph is never returned to the allocator);
    - the padding row's ``(slot, block)``: a replay's pad rows write to both
      pools, so they must never land on a slot a live request owns.

    The pad row is reserved up front when ``reserve`` is set (dense capture is
    on and build_engine sized the pools one larger), or lazily on the first
    bucket that pads. ``ensure_pad`` returns False only when the pools were
    sized without the spare row, so the caller runs an exact-size eager tick.
    """

    def __init__(self, alloc_slot, alloc_block, reserve: bool = False):
        self._alloc_slot = alloc_slot
        self._alloc_block = alloc_block
        self.pool: Any = None
        self.pad_slot: int | None = None
        self.pad_block: int | None = None
        if reserve:
            self.ensure_pad()

    def ensure_pad(self) -> bool:
        if self.pad_slot is None:
            try:
                self.pad_slot = self._alloc_slot()
                self.pad_block = self._alloc_block()
            except RuntimeError:
                return False
        return True

    @property
    def pad(self) -> tuple[int, int] | None:
        return None if self.pad_slot is None else (self.pad_slot, self.pad_block)


def make_decode_graph(
    model, backend, kv_pool, state_pool, B, W, keep, aux_layers, pool
) -> tuple[DecodeGraph | None, Any, str | None]:
    """Capture one dense (B, W) decode graph. Returns (graph, pool, error).
    On success error is None and pool is the (possibly freshly created) shared
    CUDA graph pool; on capture failure graph is None and error is the message
    for the caller to warn with - the Engine owns switching itself to eager."""
    try:
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
        g = DecodeGraph(
            model,
            backend,
            kv_pool,
            state_pool,
            B,
            width=W,
            pool=pool,
            keep=W if keep else 0,
            aux_layers=aux_layers,
        )
    except Exception as exc:  # capture is best-effort: eager fallback always works
        return None, pool, f"decode graph capture failed for B={B} W={W} ({exc}); eager fallback"
    return g, pool, None


def make_sparse_graph(
    model, backend, kv_pool, state_pool, tracker, sf, B, W, aux_layers, pool
) -> tuple[Any, Any, str | None]:
    """Capture/replay one sparse decode graph. A real CUDAGraph on cuda; on CPU
    a plain recorded forward (CpuSparseGraph) so the staging math is testable
    without a card. Returns (graph, pool, error)."""
    try:
        if backend.device.type == "cuda":
            if pool is None:
                pool = torch.cuda.graph_pool_handle()
            g = SparseDecodeGraph(
                model,
                backend,
                kv_pool,
                state_pool,
                tracker,
                sf,
                B,
                W,
                pool=pool,
                aux_layers=aux_layers,
            )
        else:
            g = CpuSparseGraph(model, backend, kv_pool, state_pool, sf, B, W)
    except Exception as exc:
        return (
            None,
            pool,
            (f"sparse decode graph capture failed for B={B} W={W} ({exc}); eager fallback"),
        )
    return g, pool, None
