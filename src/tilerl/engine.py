"""Serving engine: submit/poll loop with one model forward per tick.

Mirrors agent-infer's infer-core ``Engine`` (submit/step/poll) and infer-seam
``StepLimits``, stripped to the day-1 contract.

Tick semantics: ONE forward per tick over a planned row set. A row is either
a decode row (a running request, T=1) or ONE prefill chunk (the next slice of
a waiting or chunking request's prompt, up to ``max_num_batched_tokens``
minus the decode rows). A tick is therefore mixed (decode rows + the prefill
chunk share one forward), decode-only, or prefill-only — vLLM/sglang
continuous batching with chunked prefill, mirrored through agent-infer's
``build_forward_plan`` (waiting/running queues, per-tick token budget, decode
rows first; no preemption/swap day-1 — admission is capped at ``max_batch``).
The decode tick is a captured kernel sequence, not an interpreted one
(design-engine.md): on CUDA, a pure-decode tick replays a per-batch-size
``_DecodeGraph`` (static input buffers + static pools, captured lazily per
bucket) instead of dispatching ~900 kernels per token; the eager path stays
the default on other targets and the fallback on capture failure. Mixed
ticks (decode rows + a prefill chunk) run eager.
# ponytail: chunked prefill is bounded by ``max_num_batched_tokens``; a prompt
# tail longer than ``max_total_tokens`` is still rejected at ``submit``.

Prefix reuse: :class:`~tilerl.kv_cache.PrefixStore` may return hits of any
length; the engine adopts only block-aligned prefixes (``BLOCK_TOKENS`` = 16):
retain the matched blocks and restore a snapshot of the gated-delta state at
the match boundary — both the recurrent state and the conv1d carry window — so
the prefill forward computes only the tail. The engine is the sole publisher:
after a forward it inserts the block-aligned prefix into the store (which
retains the blocks) and snapshots the request's state slot, keyed by the
matched token tuple (collision-safe). Only published boundaries are inserted —
an entry without a snapshot can never be adopted. Full-length hits are misses
day-1 (a read-only last-token forward is day-2).

Determinism: sampling is per request with a deterministic per-step seed from
``(params.seed, generated_count)``; forwards run under the engine lock. Same
seed + input => same output on the CPU target. The engine is tokenizer-free —
it speaks token ids only.
"""

from __future__ import annotations

import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .kv_cache import BLOCK_TOKENS, LinearStatePool, PagedKvPool, PrefixStore

__all__ = ["Engine", "SamplingParams", "StepLimits", "BatchKv", "build_engine"]

_PHASE_PREFILL = 1
_PHASE_DECODE = 2
_PHASE_DONE = 3

_HASH_MASK = 0x7FFFFFFF


def _step_seed(seed: int, generated: int) -> int:
    """Deterministic per-(request, position) sampling seed."""
    return ((int(seed) << 20) ^ (generated * 2_654_435_761)) & _HASH_MASK


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 16
    seed: int = 0
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class StepLimits:
    max_batch: int = 8
    max_total_tokens: int = 512
    max_num_batched_tokens: int = 512


@dataclass
class _Req:
    req_id: int
    params: SamplingParams
    tokens: list[int]  # prompt + generated, in order
    blocks: list[int]  # physical KV block ids, oldest first
    state_slot: int
    seq_len: int  # == len(tokens); the logical materialized length
    phase: int  # _PHASE_PREFILL | _PHASE_DECODE | _PHASE_DONE
    prefill_from: int  # prefix-reuse offset for the prefill forward
    own_blocks: int  # blocks the engine allocated (vs adopted from a hit)
    output: list[int] = field(default_factory=list)


@dataclass
class BatchKv:
    """Batch-level KV descriptor for one model forward.

    The engine's wire object; the model reads these attributes duck-typed.
    ``seq_len`` is the logical length AFTER this forward per row (prefill row:
    materialized length after its chunk; decode row: current length, the
    sampled token lands at ``seq_len - 1``). ``seq_q_lens`` is the per-row
    valid query count (decode row: 1, prefill row: the chunk length) — rows
    are left-aligned and padded to a shared T; None means every row is valid
    for the full T (training, captured decode).
    """

    block_table: torch.Tensor  # [B, num_blocks] long, padded with 0
    seq_len: torch.Tensor  # [B] long
    state_slot: torch.Tensor  # [B] long
    kv_pool: Any
    state_pool: Any
    seq_q_lens: torch.Tensor | None = None  # [B] valid query tokens per row


class _DecodeGraph:
    """Captured decode forward for one batch-size bucket.

    The decode tick is a static kernel sequence — same ops, same shapes,
    every token (design-engine.md). Capture ``model.forward`` once and replay
    per token: per-tick cost is one graph replay plus small H2D copies of the
    per-tick inputs (token id, position, block table, seq_len, state slot).

    Inside the graph: embedding, every layer op (norms, linears, RoPE, KV
    scatter, paged attention / fused GDN, state gather-scatter, MLP), final
    norm, lm_head. Outside (host work or syncing): block allocation, the
    input copies, sampling, prefix publish.

    The static KV/state pools are the engine's own — replay mutates them,
    exactly like the eager path. The warmup forwards write dummy data into
    block 0 / slot 0; real requests overwrite every pool position before it
    is read (prefill writes [0, prompt_len), decodes append), and slots are
    zeroed on alloc, so the dummy data is never observable.

    # ponytail: capture at engine build time day-2 — lazy on the first
    # decode tick today, so first token pays JIT + capture.
    # ponytail: recapture after training day-2 — the graph bakes weight
    # pointers (fine, optimizer updates in place) but also the f32 embed
    # cast, which the optimizer's copy_ invalidates in eager only.
    """

    def __init__(self, model, backend, kv_pool, state_pool, batch_size):
        device = backend.device
        B = batch_size
        self._ids = torch.empty(B, 1, dtype=torch.long, device=device)
        self._pos = torch.empty(B, 1, dtype=torch.long, device=device)
        self._bt = torch.zeros(B, kv_pool.num_blocks, dtype=torch.int32, device=device)
        self._sl = torch.empty(B, dtype=torch.int32, device=device)
        self._ss = torch.empty(B, dtype=torch.long, device=device)
        # Decode rows always have exactly 1 query token; a static GPU buffer
        # (not a per-tick CPU copy) keeps seq_q_lens out of the captured
        # region — the kernels' CPU->GPU fallback breaks CUDA graph capture.
        self._sql = torch.ones(B, dtype=torch.int32, device=device)
        # Pinned staging buffers: a plain copy_ from an unpinned CPU tensor is
        # synchronous (it blocks until the copy engine drains), which under
        # GPU contention costs ms per tick. Pinned + non_blocking makes the
        # H2D copies async — stream ordering keeps replay after them.
        self._ids_h = torch.empty(B, 1, dtype=torch.long, pin_memory=True)
        self._pos_h = torch.empty(B, 1, dtype=torch.long, pin_memory=True)
        self._bt_h = torch.zeros(B, kv_pool.num_blocks, dtype=torch.int32, pin_memory=True)
        self._sl_h = torch.empty(B, dtype=torch.int32, pin_memory=True)
        self._ss_h = torch.empty(B, dtype=torch.long, pin_memory=True)
        self._kv = BatchKv(
            block_table=self._bt,
            seq_len=self._sl,
            state_slot=self._ss,
            kv_pool=kv_pool,
            state_pool=state_pool,
            seq_q_lens=self._sql,
        )
        # Warmup on a side stream: tilelang JIT-compiles per (shape, dtype),
        # and JIT is host work — it must finish before capture starts.
        self._ids.fill_(0)
        self._pos.fill_(0)
        self._sl.fill_(1)
        self._ss.fill_(0)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model.forward(self._ids, self._pos, self._kv, backend)
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._logits = model.forward(self._ids, self._pos, self._kv, backend)

    def run(self, reqs):
        """Copy per-tick inputs into the static buffers and replay.

        Returns the static logits [B,1,V]; valid until the next replay.
        """
        for i, r in enumerate(reqs):
            self._ids_h[i, 0] = r.output[-1]
            self._pos_h[i, 0] = r.seq_len - 1
            self._sl_h[i] = r.seq_len
            self._ss_h[i] = r.state_slot
            n = len(r.blocks)
            self._bt_h[i, :n] = torch.tensor(r.blocks, dtype=torch.int32)
        self._ids.copy_(self._ids_h, non_blocking=True)
        self._pos.copy_(self._pos_h, non_blocking=True)
        self._sl.copy_(self._sl_h, non_blocking=True)
        self._ss.copy_(self._ss_h, non_blocking=True)
        self._bt.copy_(self._bt_h, non_blocking=True)
        self._graph.replay()
        return self._logits


class Engine:
    """submit/poll serving loop over one model forward per tick.

    All public methods are thread-safe (an internal lock serializes against
    the daemon thread started by :meth:`run`).
    """

    def __init__(
        self,
        model: Any,
        backend: Any,
        kv_pool: Any,
        state_pool: Any,
        prefix_store: Any,
        limits: StepLimits,
        decode_graph: bool | None = None,
    ) -> None:
        self._model = model
        self._backend = backend
        self._kv = kv_pool
        self._states = state_pool
        self._prefix = prefix_store
        self.limits = limits

        # Captured decode (design-engine.md): the decode tick is replayed,
        # not interpreted. Auto-on for CUDA; the eager path stays the
        # default/fallback everywhere else and on capture failure.
        if decode_graph is None:
            decode_graph = backend.device.type == "cuda"
        self._decode_graph_on = decode_graph
        self._decode_graphs: dict[int, _DecodeGraph] = {}

        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

        self._next_id = 1
        self._waiting: deque[_Req] = deque()
        self._running: list[_Req] = []
        self._finished: dict[int, list[int]] = {}
        self._failed: dict[int, str] = {}
        self._finished_count = 0

        # Prefix-boundary state snapshots, keyed by the matched token tuple
        # (collision-safe: a hash-only key could restore the wrong GDN state on
        # a hash collision). One snapshot is 74.81 MiB at 27B, so it lives and
        # dies with its store entry: the store drops ours on eviction, and a
        # key present here is exactly a key the store still holds.
        self._prefix_state: dict[tuple[int, ...], tuple[torch.Tensor, "torch.Tensor | None"]] = {}
        prefix_store.on_evict = lambda tokens: self._prefix_state.pop(tokens, None)

        self._blocks_used = 0  # engine allocations outstanding (retains excluded)
        self._slots_used = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._prefix_published = 0
        self._prefill_forwards = 0
        self._decode_forwards = 0
        self._mixed_forwards = 0
        self._tokens_generated = 0

    # ------------------------------------------------------------------ API

    def submit(self, input_ids: Any, params: SamplingParams | None = None) -> int:
        """Queue a request; returns its opaque id.

        Performs prefix lookup immediately: a block-aligned hit retains the
        matched blocks and restores the boundary state, the tail gets fresh
        blocks, and a linear state slot is allocated.
        """
        if params is None:
            params = SamplingParams()
        tokens = [int(t) for t in input_ids]
        if not tokens:
            raise ValueError("prompt must be non-empty")
        if params.max_new_tokens > 0:
            total = len(tokens) + params.max_new_tokens
            if total > self.limits.max_total_tokens:
                raise ValueError(
                    f"request ({total} tokens) exceeds max_total_tokens "
                    f"({self.limits.max_total_tokens})"
                )
            if self._kv.blocks_for_tokens(total) > self._kv.num_blocks:
                raise ValueError(f"request ({total} tokens) exceeds KV pool capacity")
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            if params.max_new_tokens <= 0:
                self._finished[rid] = []
                self._finished_count += 1
                return rid

            matched, hit_blocks = self._match_prefix(tokens)
            # Read the snapshot now: evict_until_free below can drop the very
            # store entry we matched, and on_evict takes the snapshot with it.
            snap = self._prefix_state[self._snapshot_key(tokens[:matched])] if matched else None
            if matched:
                self._prefix_hits += 1
            else:
                self._prefix_misses += 1

            total_blocks = (len(tokens) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            blocks = list(hit_blocks)
            slot = None
            try:
                slot = self._states.alloc_slot()
                for b in blocks:
                    self._kv.retain(b)  # adopt the store's blocks
                needed = total_blocks - len(blocks)
                evict = getattr(self._prefix, "evict_until_free", None)
                if evict is not None:
                    evict(needed)
                if self._kv.free_blocks < needed:
                    raise RuntimeError("insufficient KV blocks for request")
                while len(blocks) < total_blocks:
                    blocks.append(self._kv.alloc_block())
            except Exception:
                for b in blocks:
                    self._kv.free_block(b)
                if slot is not None:
                    self._states.free_slot(slot)
                raise
            own_blocks = total_blocks - matched // BLOCK_TOKENS
            self._blocks_used += own_blocks
            self._slots_used += 1
            if matched:
                snap_states, snap_windows = snap
                self._states.states[slot].copy_(snap_states)
                if snap_windows is not None:
                    self._states.conv_windows[slot].copy_(snap_windows)

            req = _Req(
                req_id=rid,
                params=params,
                tokens=tokens,
                blocks=blocks,
                state_slot=slot,
                seq_len=matched,  # materialized length (adopted prefix; 0 on a miss)
                phase=_PHASE_PREFILL,
                prefill_from=matched,
                own_blocks=own_blocks,
            )
            self._waiting.append(req)
            return rid

    def poll(self) -> dict[int, list[int]]:
        """Return and clear all requests finished since the last poll."""
        with self._lock:
            if self._failed:
                rid, message = self._failed.popitem()
                raise RuntimeError(f"request {rid} failed: {message}")
            out = dict(self._finished)
            self._finished.clear()
            return out

    def take(self, request_id: int) -> list[int] | None:
        """Pop one finished request's output, or None if not finished yet.

        Unlike :meth:`poll`, this is safe for multiple concurrent consumers:
        each caller only observes its own request.
        """
        with self._lock:
            message = self._failed.pop(request_id, None)
            if message is not None:
                raise RuntimeError(f"request {request_id} failed: {message}")
            return self._finished.pop(request_id, None)

    def step(self) -> None:
        """Run one tick: one forward over the planned rows (mixed
        prefill+decode, decode-only, or prefill-only)."""
        with self._lock:
            decodes, prefill, chunk = self._build_plan()
            if not decodes and prefill is None:
                return
            try:
                self._run_forward(decodes, prefill, chunk)
            except Exception as exc:
                for req in list(self._running):
                    self._finish(req, error=str(exc))
                raise

    def _build_plan(self) -> tuple[list[_Req], _Req | None, int]:
        """Plan one tick, mirroring agent-infer's ``build_forward_plan``.

        Admit one waiting request into running under ``max_batch``, then take
        all running decodes as decode rows and at most one prefill row — the
        next chunk of a prefilling request, sized by the per-tick token
        budget (``max_num_batched_tokens`` minus the decode rows). A prompt
        longer than the budget stays in PREFILL and is chunked across ticks.
        """
        if self._waiting and len(self._running) < self.limits.max_batch:
            self._running.append(self._waiting.popleft())
        decodes = [r for r in self._running if r.phase == _PHASE_DECODE]
        prefill = next((r for r in self._running if r.phase == _PHASE_PREFILL), None)
        chunk = 0
        if prefill is not None:
            remaining = len(prefill.tokens) - prefill.prefill_from
            chunk = min(remaining, self.limits.max_num_batched_tokens - len(decodes))
            if chunk <= 0:
                prefill = None  # no token budget this tick; decode-only
        return decodes, prefill, chunk

    def run(self) -> None:
        """Start the daemon loop (raises if already running)."""
        if self._thread is not None:
            raise RuntimeError("engine already running")
        self._wake.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the daemon loop and join it."""
        self._wake.set()
        t = self._thread
        if t is not None:
            t.join(timeout)
        self._thread = None

    def is_idle(self) -> bool:
        with self._lock:
            return not self._waiting and not self._running

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_blocks = getattr(self._kv, "num_blocks", None)
            total_slots = getattr(self._states, "num_slots", None)
            if total_slots is None and hasattr(self._states, "states"):
                total_slots = int(self._states.states.shape[0])
            return {
                "waiting": len(self._waiting),
                "running": len(self._running),
                "finished": self._finished_count,
                "blocks_used": self._blocks_used,
                "blocks_total": total_blocks,
                "pool_used_blocks": getattr(self._kv, "used_blocks", None),
                "slots_used": self._slots_used,
                "slots_total": total_slots,
                "prefix_hits": self._prefix_hits,
                "prefix_misses": self._prefix_misses,
                "prefix_published": self._prefix_published,
                "prefill_forwards": self._prefill_forwards,
                "decode_forwards": self._decode_forwards,
                "mixed_forwards": self._mixed_forwards,
                "tokens_generated": self._tokens_generated,
            }

    # -------------------------------------------------------------- internals

    def _match_prefix(self, tokens: list[int]) -> tuple[int, list[int]]:
        """Longest block-aligned prefix hit, or (0, []).

        Full-length hits are treated as misses day-1 (no read-only last-token
        forward yet). A hit whose boundary snapshot is missing also degrades
        to a miss — the engine is the sole publisher, so this only guards
        against bookkeeping drift.
        """
        hit = self._prefix.lookup(tokens)
        if hit is None:
            return 0, []
        matched = (hit.length // BLOCK_TOKENS) * BLOCK_TOKENS
        if matched == 0 or matched >= len(tokens):
            return 0, []
        if self._snapshot_key(tokens[:matched]) not in self._prefix_state:
            return 0, []
        return matched, list(hit.blocks[: matched // BLOCK_TOKENS])

    @staticmethod
    def _snapshot_key(tokens: list[int]) -> tuple[int, ...]:
        """Collision-safe key for a prefix's boundary-state snapshot."""
        return tuple(tokens)

    def _make_kv(self, reqs: list[_Req], seq_q: list[int]) -> BatchKv:
        # Fixed width = pool size: the kernels bake the table width into the
        # compiled kernel (Mb is a compile const), so a per-tick max-blocks
        # width recompiles on every block growth. Rows zero-pad; the kernels
        # index by position bounded by seq_lens, so padding is never read.
        bt = torch.zeros(len(reqs), self._kv.num_blocks, dtype=torch.long)
        sl = torch.empty(len(reqs), dtype=torch.long)
        ss = torch.empty(len(reqs), dtype=torch.long)
        sql = torch.empty(len(reqs), dtype=torch.long)
        for i, r in enumerate(reqs):
            bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
            # Materialized length AFTER this forward: a prefill row completes
            # its chunk (prefill_from is the chunk start); a decode row writes
            # the sampled token at [seq_len-1, seq_len), so seq_len itself is
            # the post-write length.
            sl[i] = r.prefill_from + seq_q[i] if r.phase == _PHASE_PREFILL else r.seq_len
            ss[i] = r.state_slot
            sql[i] = seq_q[i]
        return BatchKv(
            block_table=bt,
            seq_len=sl,
            state_slot=ss,
            kv_pool=self._kv,
            state_pool=self._states,
            seq_q_lens=sql,
        )

    def _sample(self, logits_row: torch.Tensor, req: _Req, generated_idx: int) -> int:
        seed = _step_seed(req.params.seed, generated_idx)
        tok = self._backend.sample(
            logits_row.reshape(1, -1), req.params.temperature, req.params.top_p, seed
        )
        return int(torch.as_tensor(tok).flatten()[0])

    def _run_forward(self, decodes: list[_Req], prefill: _Req | None, chunk: int) -> None:
        """Run one mixed/decode/prefill forward over the planned rows.

        Rows are left-aligned valid tokens padded to a shared T (decode rows:
        1 token at t=0; the prefill row: its chunk), with per-row
        ``seq_q_lens`` so the kernels touch only valid positions.
        """
        growth = sum(
            max(0, (r.seq_len + BLOCK_TOKENS) // BLOCK_TOKENS - len(r.blocks)) for r in decodes
        )
        if growth:
            evict = getattr(self._prefix, "evict_until_free", None)
            if evict is not None:
                evict(growth)
        for r in decodes:
            # Blocks must cover the new token at position r.seq_len.
            # Exhaustion raises -> step() finishes the running requests.
            while len(r.blocks) * BLOCK_TOKENS <= r.seq_len:
                r.blocks.append(self._kv.alloc_block())
                r.own_blocks += 1
                self._blocks_used += 1
        if (
            prefill is None
            and decodes
            and self._decode_graph_on
            and self._run_decode_graph(decodes)
        ):
            return
        rows = decodes + ([prefill] if prefill is not None else [])
        seq_q = [1] * len(decodes) + ([chunk] if prefill is not None else [])
        width = max(seq_q)
        input_ids = np.zeros((len(rows), width), dtype=np.int64)
        positions = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(decodes):
            input_ids[i, 0] = r.output[-1]
            positions[i, 0] = r.seq_len - 1
        if prefill is not None:
            j = len(decodes)
            start = prefill.prefill_from
            input_ids[j, :chunk] = prefill.tokens[start : start + chunk]
            positions[j, :chunk] = np.arange(start, start + chunk)
        logits = self._model.forward(
            input_ids, positions, self._make_kv(rows, seq_q), self._backend
        )
        for i, r in enumerate(decodes):
            self._after_forward(r, logits[i, 0], len(r.output))
        if prefill is not None:
            self._prefill_forwards += 1
            j = len(decodes)
            prefill.prefill_from += chunk
            prefill.seq_len = prefill.prefill_from
            if prefill.prefill_from >= len(prefill.tokens):
                self._after_forward(prefill, logits[j, chunk - 1], generated_idx=0)
                # Publish the prompt prefix at a block boundary: the state
                # slot still covers exactly the prompt tokens, so the
                # snapshot is exact.
                prompt_len = len(prefill.tokens) - len(prefill.output)
                if prefill.phase != _PHASE_DONE and prompt_len % BLOCK_TOKENS == 0:
                    self._publish_prefix(prefill, prompt_len)
                if prefill.phase != _PHASE_DONE:
                    if len(prefill.output) >= prefill.params.max_new_tokens:
                        self._finish(prefill)
                    else:
                        prefill.phase = _PHASE_DECODE
        if decodes:
            self._decode_forwards += 1
        if decodes and prefill is not None:
            self._mixed_forwards += 1

    def _run_decode_graph(self, reqs: list[_Req]) -> bool:
        """Captured decode for a pure-decode tick (one graph per batch-size
        bucket, captured lazily on the first tick of that size). Returns
        False (and flips the flag off) when capture failed, so the caller
        runs eager."""
        B = len(reqs)
        g = self._decode_graphs.get(B)
        if g is None:
            try:
                g = _DecodeGraph(self._model, self._backend, self._kv, self._states, B)
            except Exception as exc:
                warnings.warn(f"decode graph capture failed for B={B} ({exc}); eager fallback")
                self._decode_graph_on = False
                return False
            self._decode_graphs[B] = g
        logits = g.run(reqs)
        self._decode_forwards += 1
        for i, r in enumerate(reqs):
            self._after_forward(r, logits[i, -1], len(r.output))
        return True

    def _after_forward(self, req: _Req, last_logits: torch.Tensor, generated_idx: int) -> None:
        tok = self._sample(last_logits, req, generated_idx)
        if tok in req.params.stop_token_ids:
            self._finish(req)
            return
        req.output.append(tok)
        req.tokens.append(tok)
        req.seq_len += 1
        self._tokens_generated += 1
        materialized = req.seq_len - 1
        if req.phase == _PHASE_DECODE and materialized % BLOCK_TOKENS == 0:
            self._publish_prefix(req, materialized)
        if len(req.output) >= req.params.max_new_tokens:
            self._finish(req)

    def _publish_prefix(self, req: _Req, length: int) -> None:
        """Insert tokens[:length] into the store (it retains the blocks) and
        snapshot the request's linear state at that boundary.

        Only the full-length entry is inserted: it is the only boundary whose
        state is snapshotted here, so shorter intermediate spans could never
        be adopted by :meth:`_match_prefix` (a hit without a boundary snapshot
        degrades to a miss). A later query that shares this prefix up to a
        smaller block boundary hits an entry published at THAT length by some
        other request.
        """
        tokens = req.tokens[:length]
        self._prefix.insert(tokens, req.blocks[: length // BLOCK_TOKENS])
        key = self._snapshot_key(tokens)
        if key not in self._prefix_state:
            windows = self._states.conv_windows
            self._prefix_state[key] = (
                self._states.states[req.state_slot].clone(),
                windows[req.state_slot].clone() if windows is not None else None,
            )
            self._prefix_published += 1  # snapshots written; an evicted prefix republishes

    def _finish(self, req: _Req, error: str | None = None) -> None:
        # Resources are freed at finish (not at poll) so pool capacity returns
        # immediately; poll only retrieves the output tokens.
        req.phase = _PHASE_DONE
        for b in req.blocks:
            self._kv.free_block(b)
        self._blocks_used -= req.own_blocks
        self._states.free_slot(req.state_slot)
        self._slots_used -= 1
        if error is None:
            self._finished[req.req_id] = req.output
        else:
            self._failed[req.req_id] = error
        self._finished_count += 1
        self._running.remove(req)

    def _loop(self) -> None:
        while not self._wake.is_set():
            with self._lock:
                busy = bool(self._running or self._waiting)
            if busy:
                try:
                    self.step()
                except Exception:
                    # ponytail: log-and-continue; a crashed daemon thread
                    # hangs the server silently. Proper backpressure (retry
                    # after block free) is the day-2 upgrade path.
                    import traceback

                    traceback.print_exc()
            else:
                self._wake.wait(0.005)


# --- Engine wiring: one factory for every caller (CLI, tests) ---------------


def build_engine(
    cfg,
    model: Any,
    backend: Any,
    *,
    num_blocks: int = 64,
    num_slots: int = 8,
    max_batch: int = 8,
    max_total_tokens: int = 8192,
    max_num_batched_tokens: int = 512,
    prefix_store: Any = None,
    decode_graph: bool | None = None,
) -> "Engine":
    """Wire a model + backend into a running Engine (pools + prefix store).

    ``cfg`` is a :class:`~tilerl.config.ModelConfig`; the factory derives the
    pool shapes from it. Pass ``prefix_store`` to inject a test double (e.g. a
    never-match store for the miss path). ``decode_graph`` auto-enables the
    captured decode tick on CUDA (design-engine.md); pass False to force the
    eager path.
    """
    n_linear = cfg.num_layers - len(cfg.full_attn_layers)
    model.params = backend.materialize(model.params)
    kv_pool = PagedKvPool(
        num_blocks,
        cfg.num_kv_heads,
        cfg.head_dim,
        num_layers=cfg.num_layers,
        device=backend.device,
    )
    state_pool = LinearStatePool(
        num_slots,
        n_linear,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        device=backend.device,
        conv_window=cfg.linear_conv_kernel_dim - 1,
        conv_dim=cfg.linear_qkv_dim,
    )
    store = PrefixStore(kv_pool) if prefix_store is None else prefix_store
    return Engine(
        model,
        backend,
        kv_pool,
        state_pool,
        store,
        StepLimits(
            max_batch=max_batch,
            max_total_tokens=max_total_tokens,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        decode_graph=decode_graph,
    )
