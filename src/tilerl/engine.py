"""Serving engine: submit/poll loop with one model forward per tick.

Mirrors agent-infer's infer-core ``Engine`` (submit/step/poll) and infer-seam
``StepLimits``, stripped to the day-1 contract.

Tick semantics: ONE forward per tick, EITHER a decode batch (running requests,
T=1 each, up to ``max_batch`` rows) OR a single prefill (one waiting request,
its whole remaining prompt tail). Decode-first: a prefill is admitted only
when no decode is pending, so requests are served serially day-1.
The decode tick is a captured kernel sequence, not an interpreted one
(design-engine.md): on CUDA, ``_step_decode`` replays a ``_DecodeGraph``
(static input buffers + static pools, captured once per batch-size bucket)
instead of dispatching ~900 kernels per token; the eager path stays the
default on other targets and the fallback on capture failure.
# ponytail: mixed prefill+decode batches day-2 — needs a ragged-row model
# contract (per-row query lengths); the pinned model.forward is dense [B,T].
# ponytail: chunked prefill day-2 — a prompt tail longer than
# ``max_total_tokens`` is rejected at ``submit`` today.

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


@dataclass(frozen=True)
class StepLimits:
    max_batch: int = 8
    max_total_tokens: int = 512


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
    full prompt length; decode row: current length, the sampled token lands at
    ``seq_len - 1``).
    """

    block_table: torch.Tensor  # [B, max_blocks] long, padded with 0
    seq_len: torch.Tensor  # [B] long
    state_slot: torch.Tensor  # [B] long
    kv_pool: Any
    state_pool: Any


class _DecodeGraph:
    """Captured decode forward for one batch-size bucket (day-1: B=1).

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

    # ponytail: batch-size buckets for continuous batching day-2 — M=1 is
    # the day-1 scope (serving is serial, decode-first); B>1 runs eager.
    # ponytail: capture at engine build time day-2 — lazy on the first
    # decode tick today, so first token pays JIT + capture.
    """

    def __init__(self, model, backend, kv_pool, state_pool, batch_size):
        device = backend.device
        B = batch_size
        self._ids = torch.empty(B, 1, dtype=torch.long, device=device)
        self._pos = torch.empty(B, 1, dtype=torch.long, device=device)
        self._bt = torch.zeros(B, kv_pool.num_blocks, dtype=torch.int32, device=device)
        self._sl = torch.empty(B, dtype=torch.int32, device=device)
        self._ss = torch.empty(B, dtype=torch.long, device=device)
        self._kv = BatchKv(
            block_table=self._bt,
            seq_len=self._sl,
            state_slot=self._ss,
            kv_pool=kv_pool,
            state_pool=state_pool,
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
        self._ids.copy_(torch.tensor([[r.output[-1]] for r in reqs], dtype=torch.long))
        self._pos.copy_(torch.tensor([[r.seq_len - 1] for r in reqs], dtype=torch.long))
        self._sl.copy_(torch.tensor([r.seq_len for r in reqs], dtype=torch.int32))
        self._ss.copy_(torch.tensor([r.state_slot for r in reqs], dtype=torch.long))
        for i, r in enumerate(reqs):
            n = len(r.blocks)
            if n:
                self._bt[i, :n].copy_(torch.tensor(r.blocks, dtype=torch.int32))
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
        self._finished_count = 0

        # Prefix-boundary state snapshots, keyed by the matched token tuple
        # (collision-safe: a hash-only key could restore the wrong GDN state on
        # a hash collision). The engine is the sole publisher, so a hit always
        # finds its entry. Entries may outlive store eviction; they are small
        # and the snapshot is deterministic per (model, tokens), so a stale
        # entry stays correct. # ponytail: prune on store eviction day-2.
        self._prefix_state: dict[tuple[int, ...], tuple[torch.Tensor, "torch.Tensor | None"]] = {}
        self._published: set[tuple[int, ...]] = set()

        self._blocks_used = 0  # engine allocations outstanding (retains excluded)
        self._slots_used = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._prefix_published = 0
        self._prefill_forwards = 0
        self._decode_forwards = 0
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
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            if params.max_new_tokens <= 0:
                self._finished[rid] = []
                self._finished_count += 1
                return rid

            matched, hit_blocks = self._match_prefix(tokens)
            if matched:
                self._prefix_hits += 1
            else:
                self._prefix_misses += 1

            tail = len(tokens) - matched
            if tail > self.limits.max_total_tokens:
                raise ValueError(
                    f"prompt tail ({tail} tokens) exceeds max_total_tokens "
                    f"({self.limits.max_total_tokens}); chunked prefill is day-2"
                )

            blocks = hit_blocks
            for b in blocks:
                self._kv.retain(b)  # adopt the store's blocks
            total_blocks = (len(tokens) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            while len(blocks) < total_blocks:
                blocks.append(self._kv.alloc_block())
                self._blocks_used += 1
            slot = self._states.alloc_slot()
            self._slots_used += 1
            if matched:
                snap_states, snap_windows = self._prefix_state[self._snapshot_key(tokens[:matched])]
                self._states.states[slot].copy_(snap_states)
                if snap_windows is not None:
                    self._states.conv_windows[slot].copy_(snap_windows)

            req = _Req(
                req_id=rid,
                params=params,
                tokens=tokens,
                blocks=blocks,
                state_slot=slot,
                seq_len=len(tokens),
                phase=_PHASE_PREFILL,
                prefill_from=matched,
                own_blocks=total_blocks - matched // BLOCK_TOKENS,
            )
            self._waiting.append(req)
            return rid

    def poll(self) -> dict[int, list[int]]:
        """Return and clear all requests finished since the last poll."""
        with self._lock:
            out = dict(self._finished)
            self._finished.clear()
            return out

    def take(self, request_id: int) -> list[int] | None:
        """Pop one finished request's output, or None if not finished yet.

        Unlike :meth:`poll`, this is safe for multiple concurrent consumers:
        each caller only observes its own request.
        """
        with self._lock:
            return self._finished.pop(request_id, None)

    def step(self) -> None:
        """Run one tick: one decode batch OR one prefill (decode-first)."""
        with self._lock:
            decodes = [r for r in self._running if r.phase == _PHASE_DECODE]
            if decodes:
                self._step_decode(decodes[: self.limits.max_batch])
                return
            if self._waiting:
                req = self._waiting.popleft()
                self._running.append(req)
                self._step_prefill(req)

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

    def _make_kv(self, reqs: list[_Req]) -> BatchKv:
        max_blocks = max(len(r.blocks) for r in reqs)
        bt = torch.zeros(len(reqs), max_blocks, dtype=torch.long)
        sl = torch.empty(len(reqs), dtype=torch.long)
        ss = torch.empty(len(reqs), dtype=torch.long)
        for i, r in enumerate(reqs):
            bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
            # Materialized length AFTER this forward: a prefill row completes
            # the prompt; a decode row writes the sampled token at
            # [seq_len-1, seq_len), so seq_len itself is the post-write length.
            sl[i] = len(r.tokens) if r.phase == _PHASE_PREFILL else r.seq_len
            ss[i] = r.state_slot
        return BatchKv(
            block_table=bt, seq_len=sl, state_slot=ss, kv_pool=self._kv, state_pool=self._states
        )

    def _sample(self, logits_row: torch.Tensor, req: _Req, generated_idx: int) -> int:
        seed = _step_seed(req.params.seed, generated_idx)
        tok = self._backend.sample(
            logits_row.reshape(1, -1), req.params.temperature, req.params.top_p, seed
        )
        return int(torch.as_tensor(tok).flatten()[0])

    def _step_prefill(self, req: _Req) -> None:
        start = req.prefill_from
        input_ids = np.asarray([req.tokens[start:]], dtype=np.int64)
        positions = np.asarray([list(range(start, len(req.tokens)))], dtype=np.int64)
        logits = self._model.forward(input_ids, positions, self._make_kv([req]), self._backend)
        self._prefill_forwards += 1
        self._after_forward(req, logits[0, -1], generated_idx=0)
        # Publish the prompt prefix at a block boundary: the state slot still
        # covers exactly the prompt tokens, so the snapshot is exact.
        prompt_len = len(req.tokens) - len(req.output)
        if req.phase != _PHASE_DONE and prompt_len % BLOCK_TOKENS == 0:
            self._publish_prefix(req, prompt_len)
        if req.phase != _PHASE_DONE:
            if len(req.output) >= req.params.max_new_tokens:
                self._finish(req)
            else:
                req.phase = _PHASE_DECODE

    def _step_decode(self, reqs: list[_Req]) -> None:
        for r in reqs:
            # Blocks must cover the new token at position r.seq_len.
            while len(r.blocks) * BLOCK_TOKENS <= r.seq_len:
                r.blocks.append(self._kv.alloc_block())
                r.own_blocks += 1
                self._blocks_used += 1
        if self._decode_graph_on and len(reqs) == 1:
            # Captured decode (day-1 bucket: M=1). Capture is lazy on the
            # first decode tick; a capture failure degrades to eager loudly.
            # ponytail: batch-size buckets day-2 — B>1 runs eager.
            g = self._decode_graphs.get(1)
            if g is None:
                try:
                    g = _DecodeGraph(self._model, self._backend, self._kv, self._states, 1)
                except Exception as exc:
                    warnings.warn(f"decode graph capture failed ({exc}); eager fallback")
                    self._decode_graph_on = False
                else:
                    self._decode_graphs[1] = g
            if g is not None:
                logits = g.run(reqs)
                self._decode_forwards += 1
                for i, r in enumerate(reqs):
                    self._after_forward(r, logits[i, -1], generated_idx=len(r.output))
                return
        input_ids = np.asarray([[r.output[-1]] for r in reqs], dtype=np.int64)
        # The sampled token was appended to r.tokens at the previous forward;
        # its position is seq_len-1 (0-indexed, matching prefill positions).
        positions = np.asarray([[r.seq_len - 1] for r in reqs], dtype=np.int64)
        logits = self._model.forward(input_ids, positions, self._make_kv(reqs), self._backend)
        self._decode_forwards += 1
        for i, r in enumerate(reqs):
            self._after_forward(r, logits[i, -1], generated_idx=len(r.output))

    def _after_forward(self, req: _Req, last_logits: torch.Tensor, generated_idx: int) -> None:
        tok = self._sample(last_logits, req, generated_idx)
        req.output.append(tok)
        req.tokens.append(tok)
        req.seq_len += 1
        self._tokens_generated += 1
        if req.phase == _PHASE_DECODE and req.seq_len % BLOCK_TOKENS == 0:
            self._publish_prefix(req, req.seq_len)
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
        if key not in self._published:
            self._published.add(key)
            windows = self._states.conv_windows
            self._prefix_state[key] = (
                self._states.states[req.state_slot].clone(),
                windows[req.state_slot].clone() if windows is not None else None,
            )
            self._prefix_published += 1

    def _finish(self, req: _Req) -> None:
        # Resources are freed at finish (not at poll) so pool capacity returns
        # immediately; poll only retrieves the output tokens.
        req.phase = _PHASE_DONE
        for b in req.blocks:
            self._kv.free_block(b)
        self._blocks_used -= req.own_blocks
        self._states.free_slot(req.state_slot)
        self._slots_used -= 1
        self._finished[req.req_id] = req.output
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
    if backend.device.type != "cpu":
        # Non-CPU targets (metal): params are built on CPU; migrate them to
        # the backend device once, at wiring time, so kernels and the
        # optimizer see consistent devices.
        model.params = {k: v.to(backend.device) for k, v in model.params.items()}
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
        StepLimits(max_batch=max_batch, max_total_tokens=max_total_tokens),
        decode_graph=decode_graph,
    )
