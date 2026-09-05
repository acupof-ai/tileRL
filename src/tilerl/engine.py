"""Serving engine: submit/poll loop with one model forward per tick.

A tick runs ONE forward over the planned rows: every running decode row (T=1,
or a 1+depth draft chain on a verify tick) plus prefill chunks up to
``max_num_batched_tokens`` — vLLM/sglang continuous batching with chunked
prefill, after agent-infer's ``build_forward_plan``. ponytail: chunked prefill
is bounded by ``max_num_batched_tokens``; a prompt over ``max_total_tokens`` is
still rejected at ``submit``. On CUDA a pure-decode tick
replays a captured ``_DecodeGraph`` per batch-size bucket; mixed ticks and every
other target run eager.

Speculation (``draft=``): a decode row drafts up to ``spec_depth`` tokens and the
same forward verifies them. Paged KV needs no rollback (a rejected draft's slot
is overwritten next tick); the gated-delta state does, so the verify forward
keeps the state after every chain step (``BatchKv.keep_steps``). A spec tick is
captured too, one graph per (batch bucket, chain width) — a width first seen
inside a timed window puts its capture in the number.

Prefix reuse adopts only block-aligned hits: retain the matched blocks and
restore the gated-delta snapshot at the boundary (state + conv1d window), keyed
by the matched token tuple. The engine is the sole publisher, so an entry
without a snapshot can never be adopted. Full-length hits are misses.

Sampling is seeded per (request, position), so same seed + input => same
output. The engine is tokenizer-free.
# ponytail: no preemption/swap — admission is capped at ``max_batch``.
"""

from __future__ import annotations

import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import precision
from .kv_cache import (
    BLOCK_TOKENS,
    BatchKv,
    DramSnapshots,
    KvTier,
    LinearStatePool,
    NoPrefixStore,
    PagedKvPool,
    PrefixStore,
)
from .spec import _PREFILL_BUCKET, LADDER_WIDTHS


def _graph_on(backend, decode_graph: bool | None) -> bool:
    """The captured decode tick is on by default on CUDA only. One definition:
    ``build_engine`` sizes the pools for the pad row from the same answer the
    engine reserves it on."""
    return backend.device.type == "cuda" if decode_graph is None else decode_graph



#: Decode-graph size ladder: a tick pads up to the first bucket >= its row count.
_GRAPH_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

_PHASE_PREFILL = 1
_PHASE_DECODE = 2
_PHASE_DONE = 3

_HASH_MASK = 0x7FFFFFFF


def _quantize_draft(params: dict[str, torch.Tensor], skip: tuple[str, ...] = (),
                    fp4: bool = False) -> dict[str, torch.Tensor]:
    """Re-serve a draft head's [N,K] projections block-quantized: fp8 by default,
    fp4 where that is the arch's only fused GEMV (sm70 has no ``linear_fp8``).

    ``skip`` names tensors the head GATHERS rows from: shape cannot tell a
    [248320,256] codebook from a projection, and packing one leaves a .w8 the
    walk cannot index.

    Idempotent: `build_engine` writes the result back into `draft.params` in
    place, so a second engine over the same draft would otherwise re-pack the
    already-packed `fc.wq` into `fc.wq.wq` and the plain `fc` lookup would raise
    `KeyError: 'fc'`. One engine per process is the shipped path, but a train loop
    or a profiler comparing configurations builds several.
    """
    from tilerl_kernels import reference

    if any(k.endswith((".wq", ".w8")) for k in params):
        return dict(params)  # already served
    out: dict[str, torch.Tensor] = {}
    for k, v in params.items():
        if k not in skip and v.ndim == 2 and v.shape[0] >= 128 and v.shape[1] >= 128:
            if fp4:
                wq, scale = reference.pack_fp4(v)
                scale, oscale = reference.renorm_fp4_scale(scale)
                out[f"{k}.wq"], out[f"{k}.scale"], out[f"{k}.oscale"] = wq, scale, oscale
            else:
                out[f"{k}.w8"], out[f"{k}.wscale"] = reference.quant_fp8(v)
        else:
            out[k] = v
    return out


def _serve_draft(draft: Any, backend: Any) -> None:
    """Quantize and materialize a draft head's weights into its own params dict.

    Called from `build_engine` (before anything reads free memory) and again from
    `Engine.__init__` for a direct caller. One function rather than two copies
    because `fp4=not has_kernel("linear_fp8")` is the arch policy: a second copy
    is a second thing to keep in step, and `_quantize_draft` is idempotent so the
    common path just pays a dict copy.
    """
    served = backend.materialize(
        _quantize_draft(draft.params, skip=draft.no_quant,
                        fp4=not backend.has_kernel("linear_fp8"))
    )
    draft.params.clear()  # in place: the head's Model holds THIS dict
    draft.params.update(served)


def _step_seed(seed: int, generated: int) -> int:
    # Full-width hashes: a shift-then-mask collapsed seeds 1/2049/16385 to one stream.
    return ((int(seed) * 2_654_435_761) ^ (generated * 2_246_822_519)) & _HASH_MASK


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = off; Qwen's generation_config ships top_k=20
    max_new_tokens: int = 16
    seed: int = 0
    stop_token_ids: tuple[int, ...] = ()
    allowed_ids: tuple[int, ...] | None = None  # restrict sampling to these ids
    #: cap on <think>: ``end_think_ids`` are forced after this many tokens; None = unbounded
    max_think_tokens: int | None = None
    end_think_ids: tuple[int, ...] = ()
    logprobs: bool = False  # log p of each token under the distribution it was drawn from


def _restrict(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    if params.allowed_ids is not None:
        keep = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(params.allowed_ids, device=logits.device)
        keep[..., idx] = logits[..., idx]
        logits = keep
    if 0 < params.top_k < logits.shape[-1]:  # on-device, no sync: a threshold and a mask
        # ponytail: masks strictly below the kth value, so tied logits at the boundary
        # leave the support wider than top_k (vLLM #49577 review); exact k needs a sort.
        kth = torch.topk(logits, params.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return logits


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
    logprobs: list[float] = field(default_factory=list)
    thought_closed: bool = False  # the reasoning block ended (model's or forced)
    #: trunk hidden [1,w,H] at positions [hidden_from, hidden_from+w): the draft's fc input
    hidden: torch.Tensor | None = None
    hidden_prev: torch.Tensor | None = None  # [1,1,H] at hidden_from-1
    hidden_from: int = 0
    draft_pos: int = 0  # highest position whose draft KV belongs to a committed token
    drafts: list[int] = field(default_factory=list)  # next tick's chain, minus its first token
    #: block drafter: the trunk's aux-layer taps over the same positions as ``hidden``,
    #: [1,w,len(target_layers)*H]. Tick-scoped — ``_draft_block`` consumes it and it dies.
    aux: torch.Tensor | None = None
    #: block drafter: per-layer (k, v) for the context, [1,T,heads,dim]. ``context_kv`` is
    #: per-position pure, so a position is projected the tick it commits and never again.
    ctx: list | None = None
    ctx_len: int = 0  # context positions already projected into ``ctx``

    @property
    def prefilling(self) -> bool:
        return self.phase == _PHASE_PREFILL

    @property
    def decoding(self) -> bool:
        return self.phase == _PHASE_DECODE

    @property
    def done(self) -> bool:
        return self.phase == _PHASE_DONE


class _DecodeGraph:
    """Captured ``model.forward`` for one (batch, width) bucket: per tick, small
    H2D copies of the inputs plus one replay. Replay mutates the engine's own
    pools like the eager path; warmup writes to block 0 / slot 0 are overwritten
    before any real request reads them.
    # ponytail: no recapture after training — the graph bakes the f32 embed cast.
    """

    def __init__(self, model, backend, kv_pool, state_pool, batch_size, width=1, pool=None,
                 last_only=False, keep=0, aux_layers=()):
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
            self._logits = model.forward(self._ids, self._pos, self._kv, backend,
                                         hidden_out=hid, last_only=last_only,
                                         aux_layers=aux_layers)
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
            if r.phase == _PHASE_PREFILL:
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
        draft: Any = None,
        spec_depth: int | None = None,
    ) -> None:
        self._model = model
        self._backend = backend
        self._kv = kv_pool
        self._states = state_pool
        self._prefix = prefix_store
        self.limits = limits

        self._decode_graph_on = _graph_on(backend, decode_graph)
        self._decode_graphs: dict = {}
        # A replay's padding rows write to both pools, so they need a slot and a
        # block of their own. Reserved here, not on the first tick that pads:
        # ``build_engine`` sized the pools for this row, and taking it up front
        # keeps the capacity the caller asked for whole instead of removing one
        # request's worth of it partway through a run.
        self._pad_slot: int | None = None
        self._pad_block: int | None = None
        if self._decode_graph_on:
            try:
                self._pad_slot = state_pool.alloc_slot()
                self._pad_block = kv_pool.alloc_block()
            except RuntimeError:
                pass  # pools sized without the spare: fall back to exact-size graphs
        self._graph_pool = None
        # A slot is held from submit() to finish, so usable_slots -- not max_batch --
        # is the real concurrency ceiling: below it, `submit` raises before a row can
        # ever be admitted, and _build_plan's max_batch is unreachable. Warn rather
        # than clamp, because a test that submits two rows into a 2-slot pool with the
        # default max_batch=8 is a legitimate config, not a mistake.
        if self.usable_slots < limits.max_batch:
            warnings.warn(
                f"{self.usable_slots} usable state slots against max_batch="
                f"{limits.max_batch}: a slot is held from submit to finish, so "
                f"concurrency is capped at {self.usable_slots} and submit raises "
                f"beyond it. Pass num_slots >= max_batch"
                + (" + 1 for the decode graph's pad row" if self._pad_slot is not None
                   else ""),
                stacklevel=2,
            )

        self._draft = draft
        self._aux_layers = draft.aux_layers if draft is not None else ()
        self._width = 1  # verify tick width: 1 committed token + width-1 drafts
        if draft is not None:
            from tilerl_kernels.backend import _MAX_VERIFY_W

            if not hasattr(draft, "step"):
                raise TypeError(
                    f"draft head {type(draft).__name__} is not a drafter: it has no "
                    f"step(rows). See the contract in spec.py."
                )
            draft.set_depth(spec_depth)
            self._width = draft.width
            if not 1 < self._width <= BLOCK_TOKENS:
                raise ValueError(
                    f"verify width must be in (1, {BLOCK_TOKENS}], got {self._width}"
                )
            if self._width > _MAX_VERIFY_W:
                raise ValueError(
                    f"verify width {self._width} exceeds the verify tile's {_MAX_VERIFY_W}: "
                    f"paged_attention would route every verify tick off the decode path onto "
                    f"the M-tiled prefill kernel, which costs more than the drafts save"
                )
            if draft.aux_layers and not isinstance(prefix_store, NoPrefixStore):
                raise ValueError(
                    "a drafter that taps the trunk's aux layers cannot serve behind a prefix "
                    "cache. Its context is built only from positions this process forwarded; "
                    "an adopted prefix skips them, so the draft would attend over whatever the "
                    "recycled blocks hold and the failure would look like a weak drafter, not "
                    "a bug. Pass prefix_store=NoPrefixStore()."
                    # ponytail: rebuilding ctx from an adopted prefix is the upgrade.
                )
            # The draft's weights are served in `build_engine`, BEFORE the KV fit reads
            # free memory -- see the comment there. A direct `Engine(...)` caller that
            # passes an unquantized draft still gets one here.
            _serve_draft(draft, backend)
            if backend.arch == "sm70":
                self._warn_sm70_ladder(limits.max_batch, self._width)
            draft.attach(backend, kv_pool.num_blocks, dtype=kv_pool.k_pool.dtype)

        self._pin = backend.device.type == "cuda"
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

        self._next_id = 1
        self._waiting: deque[_Req] = deque()
        self._running: list[_Req] = []
        self._finished: dict[int, list[int]] = {}
        self._failed: dict[int, str] = {}
        self._finished_count = 0

        self._blocks_used = 0  # engine allocations outstanding (retains excluded)
        self._slots_used = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._prefix_published = 0
        self._prefill_forwards = 0
        self._decode_forwards = 0
        self._mixed_forwards = 0
        self._tokens_generated = 0
        self._spec_drafted = 0
        self._spec_accepted = 0
        # Diagnostic only: set True to keep the last tick's trunk logits and the
        # chains they scored, so a probe can rank the trunk's pick inside the
        # draft's ordering. A [rows, vocab] copy per tick, so never on in serving.
        self._keep_draft_logits = False
        self._trunk_logits = None
        self._verify_chains = None
        #: Set to a list to time each draft forward directly, as (forwards, ms). A
        #: per-tick sync, so never on in serving; None keeps the path unchanged.
        self._draft_ms: list[tuple[int, float]] | None = None
        self._finished_logprobs: dict[int, list[float]] = {}
        self._taken_logprobs: set[int] = set()
        self._last_logprobs: list[float] | None = None

    # ------------------------------------------------------------------ API

    @property
    def usable_blocks(self) -> int:
        """KV blocks a request may have. The pools are sized one larger than the
        caller asked for when the captured tick is on, and that row is the
        engine's — every capacity answer is net of it, or a request sized to the
        whole pool passes the guard and fails on the allocation behind it."""
        return self._kv.num_blocks - (self._pad_block is not None)

    @property
    def usable_slots(self) -> int:
        return self._states.num_slots - (self._pad_slot is not None)

    def submit(self, input_ids: Any, params: SamplingParams | None = None) -> int:
        """Queue a request; returns its opaque id. Prefix lookup, block and
        state-slot allocation happen here, not at admission."""
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
            # +depth: a verify tick materializes the drafts past the last token
            if self._kv.blocks_for_tokens(total + self._width - 1) > self.usable_blocks:
                raise ValueError(f"request ({total} tokens) exceeds KV pool capacity")
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            if params.max_new_tokens <= 0:
                self._finished[rid] = []
                self._finished_count += 1
                return rid

            # The hit carries its snapshot: evict_until_free below can drop the entry.
            matched, hit_blocks, snap = self._match_prefix(tokens)
            if matched:
                self._prefix_hits += 1
            else:
                self._prefix_misses += 1

            total_blocks = (len(tokens) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            # The slot first, and `blocks` empty: the unwind frees what this request
            # incremented, and nothing is incremented until retain() runs. Seeding it
            # with hit_blocks made an alloc_slot() failure decrement refcounts the
            # PrefixStore still holds -- free_block cannot tell that from a release.
            slot = self._states.alloc_slot()
            blocks: list[int] = []
            try:
                # Releasing all of `blocks` on the way out is safe because retain
                # cannot fail here: it raises only at refcount 0, and hit_blocks
                # came from _match_prefix under the same lock. Everything after
                # this loop CAN raise, and by then every entry is retained.
                for b in hit_blocks:
                    self._kv.retain(b)  # adopt the store's blocks
                    blocks.append(b)
                needed = total_blocks - len(blocks)
                self._prefix.evict_until_free(needed)
                if self._kv.free_blocks < needed:
                    raise RuntimeError("insufficient KV blocks for request")
                while len(blocks) < total_blocks:
                    blocks.append(self._kv.alloc_block())
            except Exception:
                for b in blocks:
                    self._kv.free_block(b)
                self._states.free_slot(slot)
                raise
            own_blocks = total_blocks - matched // BLOCK_TOKENS
            self._blocks_used += own_blocks
            self._slots_used += 1
            if matched:
                snap_states, snap_windows = snap
                self._states.states[slot].copy_(snap_states)
                if snap_windows is not None:
                    self._states.window_restore(slot, snap_windows)

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

    def logprobs(self, request_id: int) -> list[float] | None:
        """log q of each returned token under the truncated, tempered distribution
        it was drawn from -- not the full softmax. None unless the request asked.
        Pops; a second read of the same id raises, so "never asked" and "already
        taken" stay distinguishable at the RL call site.
        # ponytail: scores nobody reads live until the engine is dropped; a TTL sweep is the upgrade.
        """
        with self._lock:
            if request_id in self._finished_logprobs:
                self._taken_logprobs.add(request_id)
                return self._finished_logprobs.pop(request_id)
            if request_id in self._taken_logprobs:
                raise KeyError(
                    f"logprobs for request {request_id} were already taken -- they pop, so "
                    "exactly one reader may have them. Record them once at that reader."
                )
            return None

    def peek(self, request_id: int) -> list[int] | None:
        """Tokens emitted so far, or None once the request has left the queues.

        Deliberately lock-free: ``step()`` holds ``_lock`` across the whole forward, so any
        reader that took the lock would block for the entire generation (measured: one
        blocked call covered 325 ms of a 335 ms run). Under the GIL both the writer's
        ``output.append`` and this ``list()`` are single bytecodes, so the copy is a
        consistent prefix -- never a torn read; a stale one is fine.

        None means "no longer waiting or running", so ``_finish`` has filed it under
        ``_finished`` or ``_failed`` and ``take()`` will answer. That is what lets a caller
        poll here without ever touching the lock until the run is over.
        """
        for req in (*self._waiting, *self._running):
            if req.req_id == request_id:
                return list(req.output)
        return None

    def take(self, request_id: int) -> list[int] | None:
        """Pop one finished request's output, or None if not finished yet."""
        with self._lock:
            message = self._failed.pop(request_id, None)
            if message is not None:
                raise RuntimeError(f"request {request_id} failed: {message}")
            return self._finished.pop(request_id, None)

    def step(self) -> None:
        """Run one tick: one forward over the planned rows."""
        with self._lock:
            decodes, prefills, chunks = self._build_plan()
            if not decodes and not prefills:
                return
            try:
                self._run_forward(decodes, prefills, chunks)
            except Exception as exc:
                for req in list(self._running):
                    self._finish(req, error=str(exc))
                raise

    def _build_plan(self) -> tuple[list[_Req], list[_Req], list[int]]:
        """Admit the whole waiting queue up to ``max_batch``, then all running
        decodes plus as many prefill rows as the token budget and one width
        bucket allow; a longer prompt stays in PREFILL and chunks across ticks."""
        while self._waiting and len(self._running) < self.limits.max_batch:
            self._running.append(self._waiting.popleft())
        decodes = [r for r in self._running if r.phase == _PHASE_DECODE]
        prefills: list[_Req] = []
        chunks: list[int] = []
        budget = self.limits.max_num_batched_tokens - len(decodes)
        bucket = 0
        for r in self._running:
            if r.phase != _PHASE_PREFILL:
                continue
            if len(decodes) + len(prefills) >= self.limits.max_batch:
                break
            chunk = min(len(r.tokens) - r.prefill_from, budget)
            if chunk <= 0:
                break
            # Cut a ragged tail off the FIRST chunk so at least one publish point exists.
            # Two separate conditions, and only the first gates publishing: a publish needs
            # `% BLOCK_TOKENS == 0` (the entry slices whole blocks) and a state that is
            # exact, which holds at ANY chunk end. 64 is not required -- measured, a chunk
            # ending at 48 publishes and its restored state is allclose to a NoPrefixStore
            # engine's, max|delta| 0.000e+00. What 64 buys is reachability: a prompt shorter
            # than the token budget is ONE chunk, and `_finish_prefills` then had nowhere
            # aligned to publish, since it required the WHOLE prompt length to be aligned
            # and 15 of every 16 lengths are not. Measured on the live V100 before this:
            # prefix_published 4, all four from decode, prefix_hits 0 over a 6-turn chat.
            # The tail is one extra forward -- a launch, not extra tokens.
            # ponytail: the first chunk only. A later chunk is ragged whenever a decode row
            # shares the tick (budget = max_num_batched_tokens - len(decodes)), and aligning
            # those would round that budget down to 64 -- shrinking the token budget for the
            # DECODE rows sharing the tick, a throughput cost on every batched tick to help
            # prompts that already published at their first boundary.
            aligned = (chunk // _PREFILL_BUCKET) * _PREFILL_BUCKET
            if r.prefill_from == 0 and aligned and aligned != chunk:
                chunk = aligned
            # Rows pad to a shared width: pack only within one bucket.
            b = -(-chunk // _PREFILL_BUCKET) * _PREFILL_BUCKET
            if prefills and b != bucket:
                break
            bucket = b
            prefills.append(r)
            chunks.append(chunk)
            budget -= chunk
        return decodes, prefills, chunks

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
            store = self._prefix.stats()
            return {
                "waiting": len(self._waiting),
                "running": len(self._running),
                "finished": self._finished_count,
                "blocks_used": self._blocks_used,
                "blocks_total": self.usable_blocks,
                "pool_used_blocks": self._kv.used_blocks,
                "slots_used": self._slots_used,
                "slots_total": self.usable_slots,
                "prefix_hits": self._prefix_hits,
                "prefix_misses": self._prefix_misses,
                "prefix_published": self._prefix_published,
                # Whether the store is under pressure at all: a DRAM/SSD tier below it can
                # only recover entries that were actually evicted, and at 144 MiB a 27B
                # snapshot the sm70 budget (free/4 = 1417 MiB) holds 9 of them.
                "prefix_evictions": store["evictions"],
                "prefix_state_bytes": store["state_bytes"],
                # Present only with a host tier; a demotion is a prefix the card could not
                # keep but did not have to lose.
                **{k: v for k, v in store.items() if k.startswith("dram_")},
                "prefix_demoted": store.get("demoted", 0),
                "prefill_forwards": self._prefill_forwards,
                "decode_forwards": self._decode_forwards,
                "mixed_forwards": self._mixed_forwards,
                "tokens_generated": self._tokens_generated,
                "spec_drafted": self._spec_drafted,
                "spec_accepted": self._spec_accepted,
            }

    # -------------------------------------------------------------- internals

    def _match_prefix(self, tokens: list[int]) -> tuple[int, list[int], Any]:
        """Longest block-aligned prefix hit as (length, blocks, snapshot), or (0, [], None);
        a full-length hit is a miss."""
        hit = self._prefix.lookup(tokens)
        if hit is None:
            return 0, [], None
        matched = (hit.length // BLOCK_TOKENS) * BLOCK_TOKENS
        if matched == 0 or matched >= len(tokens):
            return 0, [], None
        return matched, list(hit.blocks[: matched // BLOCK_TOKENS]), hit.state

    def _make_kv(self, reqs: list[_Req], seq_q: list[int], keep_steps: int = 0) -> BatchKv:
        # Table width = pool size: the kernels compile it in, so a per-tick width recompiles.
        bt = torch.zeros(len(reqs), self._kv.num_blocks, dtype=torch.long, pin_memory=self._pin)
        sl = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        ss = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        sql = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        for i, r in enumerate(reqs):
            bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
            # Length after this forward; a decode row's chain starts at seq_len-1.
            sl[i] = (
                r.prefill_from + seq_q[i]
                if r.phase == _PHASE_PREFILL
                else r.seq_len - 1 + seq_q[i]
            )
            ss[i] = r.state_slot
            sql[i] = seq_q[i]
        if self._pin:
            # Move once here, not per layer inside every kernel (971 pageable copies a prefill).
            dev = self._backend.device
            bt = bt.to(dev, non_blocking=True)
            sl = sl.to(dev, non_blocking=True)
            ss = ss.to(dev, non_blocking=True)
            sql = sql.to(dev, non_blocking=True)
        return BatchKv(
            block_table=bt,
            seq_len=sl,
            state_slot=ss,
            kv_pool=self._kv,
            state_pool=self._states,
            seq_q_lens=sql,
            keep_steps=keep_steps,
        )

    def _run_forward(self, decodes: list[_Req], prefills: list[_Req], chunks: list[int]) -> None:
        # Speculate on pure-decode ticks only: the step-state buffers cannot
        # cover a bucketed prefill width.
        chains = (
            [[r.output[-1], *r.drafts] for r in decodes]
            if self._draft is not None and decodes and not prefills
            else None
        )
        if chains is not None and max(map(len, chains)) == 1:
            chains = None  # the policy kept nothing: a plain decode tick
        elif chains is not None:
            # Pad to the widest chain: one graph per (B, width), and the fused
            # decode kernels take one width for the whole tick. A repeated pad
            # token is just a draft that gets rejected.
            w = max(map(len, chains))
            for c in chains:
                c.extend([c[-1]] * (w - len(c)))
        q_dec = [len(c) for c in chains] if chains else [1] * len(decodes)
        growth = sum(
            max(0, (r.seq_len + q - 1 + BLOCK_TOKENS) // BLOCK_TOKENS - len(r.blocks))
            for r, q in zip(decodes, q_dec)
        )
        if growth:
            self._prefix.evict_until_free(growth)
        for r, q in zip(decodes, q_dec):
            # Cover the chain's last position; exhaustion raises and step() fails the batch.
            while len(r.blocks) * BLOCK_TOKENS <= r.seq_len - 1 + q:
                r.blocks.append(self._kv.alloc_block())
                r.own_blocks += 1
                self._blocks_used += 1
        if (
            not prefills
            and decodes
            and self._decode_graph_on
            and self._run_decode_graph(decodes, chains)
        ):
            return
        rows = decodes + prefills
        seq_q = q_dec + chunks
        # Bucket a prefill width: kernels specialize per shape (MMLU compiled
        # 662 variants). A verify width is exact, at most 1+depth.
        chunk = max(chunks, default=0)
        width = -(-max(seq_q) // _PREFILL_BUCKET) * _PREFILL_BUCKET if chunk > 1 else max(seq_q)
        input_ids = np.zeros((len(rows), width), dtype=np.int64)
        positions = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(decodes):
            chain = chains[i] if chains else [r.output[-1]]
            input_ids[i, : len(chain)] = chain
            positions[i, : len(chain)] = np.arange(r.seq_len - 1, r.seq_len - 1 + len(chain))
        for k, (pf, c) in enumerate(zip(prefills, chunks)):
            j = len(decodes) + k
            start = pf.prefill_from
            input_ids[j, :c] = pf.tokens[start : start + c]
            positions[j, :c] = np.arange(start, start + c)
        hid: list | None = [] if self._draft else None
        logits = self._model.forward(
            input_ids, positions, self._make_kv(rows, seq_q, width if chains else 0),
            self._backend, hidden_out=hid, aux_layers=self._aux_layers,
            last_only=False if chains else seq_q,  # a verify tick needs every chain position
        )
        if hid is not None:
            n_aux = len(self._aux_layers)
            for i, r in enumerate(rows):  # hidden_out is full width, appended before last_only
                r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
                r.hidden = hid[-1][i : i + 1, : seq_q[i]]
                r.hidden_from = int(positions[i, 0])
                if n_aux:
                    r.aux = torch.cat([h[i : i + 1, : seq_q[i]] for h in hid[:n_aux]], -1)
        if chains:
            self._verify(decodes, chains, logits, hid[-1])
        else:
            self._sample_commit([(r, logits[i, 0], len(r.output)) for i, r in enumerate(decodes)])
        if prefills:
            self._prefill_forwards += 1
            self._finish_prefills(prefills, chunks, logits, len(decodes))
        if decodes:
            self._decode_forwards += 1
        if decodes and prefills:
            self._mixed_forwards += 1
        if self._draft is not None:
            # The draft writes position seq_len-1 on EVERY row it sees, including a
            # row that just left prefill this tick -- and the growth loop above only
            # covers `decodes`. A 15-token prompt therefore reached the draft owning
            # one block while position 15 needs the second, which raised
            # `IndexError: index 1 is out of bounds` from kv_cache.py:149, three
            # frames away inside the trunk's own writer. Clamping the draft's span
            # instead leaves a hole in its KV and the next position attends over it:
            # measured, the engine then drafted token 79 where full context drafts 61.
            for r in rows:
                while r.blocks and len(r.blocks) * BLOCK_TOKENS <= r.seq_len - 1:
                    r.blocks.append(self._kv.alloc_block())
                    r.own_blocks += 1
                    self._blocks_used += 1
            if self._draft_ms is None:
                self._draft.step(rows)  # every tick, or a chunked prefill leaves the draft KV empty
            else:
                self._draft_step_timed(rows)

    def _finish_prefills(self, prefills: list[_Req], chunks: list[int], logits, base: int) -> None:
        done = []
        for k, (pf, c) in enumerate(zip(prefills, chunks)):
            pf.prefill_from += c
            pf.seq_len = pf.prefill_from
            if pf.prefill_from >= len(pf.tokens):
                done.append((pf, logits[base + k, min(c, logits.shape[1]) - 1], 0))
            elif pf.prefill_from % BLOCK_TOKENS == 0:
                # A chunk end IS a state-pool boundary: nothing has been sampled yet, so
                # the slot holds exactly tokens[:prefill_from]. This is the publish that
                # makes a ragged prompt shareable -- `_pick` cut the chunk short for it.
                self._publish_prefix(pf, pf.prefill_from)
        if not done:
            return
        self._sample_commit(done)
        for pf, _, _ in done:
            # The state slot still covers exactly the prompt, so the snapshot is exact.
            prompt_len = len(pf.tokens) - len(pf.output)
            if pf.phase != _PHASE_DONE and prompt_len % BLOCK_TOKENS == 0:
                self._publish_prefix(pf, prompt_len)
            if pf.phase != _PHASE_DONE:
                if len(pf.output) >= pf.params.max_new_tokens:
                    self._finish(pf)
                else:
                    pf.phase = _PHASE_DECODE

    def _graph_bucket(self, rows: int) -> int:
        """The batch dimension a tick of ``rows`` decodes keys its graph on: the
        next bucket up, or the exact size above the ladder. `precapture` walks
        this over every admissible row count, so the two cannot disagree about
        which graphs exist."""
        b = next((c for c in _GRAPH_BUCKETS if c >= rows), None)
        return rows if b is None or self.limits.max_batch < b else b

    def _graph_for(self, B: int, W: int, keep: bool) -> _DecodeGraph | None:
        """The (B, W) graph, capturing it on first use. None (and graphs off) if
        capture fails, so the caller runs eager."""
        g = self._decode_graphs.get((B, W))
        if g is not None:
            return g
        try:
            if self._graph_pool is None:
                self._graph_pool = torch.cuda.graph_pool_handle()
            g = _DecodeGraph(self._model, self._backend, self._kv, self._states, B,
                             width=W, pool=self._graph_pool, keep=W if keep else 0,
                             aux_layers=self._aux_layers)
        except Exception as exc:
            warnings.warn(f"decode graph capture failed for B={B} W={W} ({exc}); eager fallback")
            self._decode_graph_on = False
            return None
        self._decode_graphs[(B, W)] = g
        return g

    @staticmethod
    def _warn_sm70_ladder(max_batch: int, w: int) -> None:
        """The sm70 GEMV serves 1/2/4/8/32 rows and rounds up, so a verify tick's
        B*W rows can pay for a rung it does not fill. Warn rather than clamp -- the
        ladder is one arch's shape, not a property of speculation."""
        if w not in LADDER_WIDTHS:
            # depth 4 (W=5) buys an 8-row launch: 31.5 tok/s on coding against 43.8
            # at depth 3 and 32.6 with no speculation at all. A verify tick costs
            # 0.67 + 0.53*W dense ticks, so rounding W up is a real cost.
            warnings.warn(
                f"verify width {w} is not an sm70 rung; it rounds up to "
                f"{next(x for x in LADDER_WIDTHS if x >= w)} rows. Use depth "
                f"{max(x for x in LADDER_WIDTHS if x <= w) - 1} or "
                f"{next(x for x in LADDER_WIDTHS if x > w) - 1}",
                stacklevel=3,
            )
        rows = max_batch * w
        if rows > max(LADDER_WIDTHS):
            # Past the top rung the dispatch chunks at 32, so a wide batch costs
            # extra launches rather than extra per-row time.
            warnings.warn(
                f"max_batch={max_batch} x verify width {w} = {rows} rows exceeds the sm70 "
                f"ladder's top rung ({max(LADDER_WIDTHS)}); a full batch verifies in "
                f"{-(-rows // 32)} launches per layer",
                stacklevel=3,
            )
        elif rows not in LADDER_WIDTHS:
            # Between rungs is worse than past the top: the launch pays for the whole
            # rung, and a padding row costs what a useful one costs. Measured on the
            # same rung 8: 82.15 ms with 3 of 8 rows idle against 83.40 ms fully
            # packed -- 60% more useful rows for 1.5% more time
            # (wins/2026-09-04-rung-cost-not-useful-rows.md). So B=4 depth 3 -- 16
            # rows on the 32 rung -- measures 42.7 tok/s where B=8's full rung gets 75.0.
            rung = next(x for x in LADDER_WIDTHS if x > rows)
            # Only advise a batch when the width divides the rung: at W=3 NO batch
            # lands on a rung, and rung // w would name one that also pads.
            fix = f"; use max_batch={rung // w} to fill it" if rung % w == 0 else ""
            warnings.warn(
                f"max_batch={max_batch} x verify width {w} = {rows} rows launches the "
                f"{rung}-row rung, so {rung - rows} of every {rung} rows are padding{fix}",
                stacklevel=3,
            )

    def graph_keys(self) -> set[tuple[int, int]]:
        """Every (bucket, width) a decode tick can key on under these limits."""
        # self._width, not spec_depth+1: it is the width the drafter SETTLED on
        # (set_depth may clamp) and the one every tick keys on, and it is already
        # range-checked in __init__. A second copy of the arithmetic here is how
        # precapture came to reference a _spec_depth attribute that does not exist.
        widths = range(1, 1 + self._width) if self._draft is not None else (1,)
        return {(self._graph_bucket(rows), w)
                for rows in range(1, self.limits.max_batch + 1) for w in widths}

    def precapture(self) -> int:
        """Capture every graph a decode tick can ask for; return how many exist.

        Capture costs ~14 s each and, until a graph exists, that tick IS the
        capture rather than a replay — 1088 ms/token on a cold server against 26
        warm. Waiting for real traffic to produce each width is a lottery: chain
        width varies per tick because the draft's confidence truncates it, so a
        warmup that merely generated tokens left two widths uncaptured and the
        first two requests paid 14 s and 12 s. `graph_keys` enumerates instead.
        """
        if not self._decode_graph_on:
            return 0
        for B, W in sorted(self.graph_keys()):
            # keep matches the tick that will use this graph: W>1 is a verify
            # (chains present, keep=W), W==1 is a plain decode (chains None).
            if self._graph_for(B, W, keep=W > 1) is None:
                break  # capture failed: graphs are off now
        return len(self._decode_graphs)

    def invalidate_weights(self) -> int:
        """Drop everything computed under the previous weights; return graphs dropped.

        An optimizer step makes both caches lie: a captured graph replays the
        forward as it was traced, and a cached prefix serves KV from the old
        policy. Both are silent -- nothing raises, the rollout is just off-policy
        -- which is why ``_require_on_policy`` refuses an engine carrying either.
        Calling this after each update is what lets a training engine keep them.

        The graphs are dropped rather than re-traced here: the next tick that
        needs one captures it, so a bucket the run never reaches costs nothing.
        """
        n = len(self._decode_graphs)
        self._decode_graphs.clear()
        # The pool owns the captured memory; a new pool per invalidation would
        # leak one arena per step.
        self._prefix.clear()
        return n

    def _run_decode_graph(self, reqs: list[_Req], chains=None) -> bool:
        """Captured decode for a pure-decode tick, one graph per size bucket (a
        graph per exact size OOMed B=64 on the drain). Returns False and flips
        the flag off when capture failed, so the caller runs eager."""
        n, W = len(reqs), len(chains[0]) if chains else 1
        B = self._graph_bucket(n)
        if n < B and self._pad_slot is None:
            try:
                self._pad_slot = self._states.alloc_slot()
                self._pad_block = self._kv.alloc_block()
            except RuntimeError:
                B = n  # no spare capacity to park padding rows on: exact size
        g = self._graph_for(B, W, keep=bool(chains))
        if g is None:
            return False
        pad = None if self._pad_slot is None else (self._pad_slot, self._pad_block)
        logits = g.run(reqs, chains, pad=pad)
        self._decode_forwards += 1
        if g.aux is not None:  # _verify sets hidden_from, which is aux's base position too
            for i, r in enumerate(reqs):
                r.aux = g.aux[i : i + 1]
        if chains:
            self._verify(reqs, chains, logits, g.hidden)
        else:
            if self._draft is not None and g.hidden is not None:
                for i, r in enumerate(reqs):  # keep the draft's fc input current
                    r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
                    r.hidden, r.hidden_from = g.hidden[i : i + 1], r.seq_len - 1
            self._sample_commit([(r, logits[i, -1], len(r.output)) for i, r in enumerate(reqs)])
        if self._draft is not None:
            # No block-growth loop needed here: this path runs only with `not prefills`, so
            # the pre-fork `seq_len - 1 + q` loop already covers the draft's furthest write.
            if self._draft_ms is None:
                self._draft.step(reqs)
            else:
                self._draft_step_timed(reqs)
        return True

    def _draft_step_timed(self, rows: list[_Req]) -> None:
        """``_draft.step`` with CUDA events around it, recording (forwards, ms).

        One helper because there are TWO draft call sites -- this graph path and the
        eager one in ``_run_forward`` -- and instrumenting only the eager one produced
        a number 31x too large: the graph path takes 212 of 218 ticks, so the timer saw
        only the 6 warm/mixed ticks, which carry prefill work. It read 165.97 ms/forward
        against a subtracted 4.80-5.30, and the tick it sat in read 155.74 against a
        known 35.04. Both are >2x off a known number, which is the tell; the count in
        its own output (6 of 218) is what named the cause.

        Timed rather than subtracted because the subtraction of two rung-sharing tick
        means amplifies their noise by operand/difference, measured 12.9x
        (wins/2026-09-04-a-difference-amplifies-its-operands-noise.md). Events bracket
        the launches, so nothing cancels -- at the price of a sync per tick.
        """
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        f0 = self._draft.forwards
        a.record()
        self._draft.step(rows)
        b.record()
        b.synchronize()
        self._draft_ms.append((self._draft.forwards - f0, a.elapsed_time(b)))

    def _verify(self, rows, chains, logits, hidden) -> None:
        """Accept the leading run of drafts the trunk agrees with, adopt the
        recurrent state at that length, and commit the prefix plus the trunk's
        bonus token. Every committed token is this tick's own draw from the
        trunk at that chain position, under the per-generated-index seed the
        unspeculated arm uses. That is the guarantee; the token is NOT
        bit-identical to the unspeculated one, because a W>1 tile and a W=1
        tile do not agree bit-for-bit off the CPU reference."""
        if self._keep_draft_logits:  # rank of the trunk's pick in the draft's order
            self._trunk_logits = logits.detach().clone()
            self._verify_chains = [list(c) for c in chains]
        flat = [
            (r, logits[i, j], len(r.output) + j)
            for i, r in enumerate(rows)
            for j in range(len(chains[i]))
        ]
        toks, at = self._sample_batch(flat), 0
        lps = self._last_logprobs
        for i, r in enumerate(rows):
            got = toks[at : at + len(chains[i])]
            at += len(chains[i])
            n_ok = 0
            while n_ok < len(got) - 1 and got[n_ok] == chains[i][n_ok + 1]:
                n_ok += 1
            self._spec_accepted += n_ok
            self._spec_drafted += len(chains[i]) - 1
            self._states.select_step(r.state_slot, n_ok)
            r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
            r.hidden, r.hidden_from = hidden[i : i + 1], r.seq_len - 1
            self._commit(r, got[: n_ok + 1],
                         None if lps is None else lps[at - len(chains[i]) : at][: n_ok + 1])

    def _sample_batch(self, rows: list[tuple]) -> list[int]:
        """One batched sample over all rows (B per-row sorts were 8.2% of a B=8
        tick); per-row seeds keep the draws identical. The caller commits."""
        if not rows:
            return []
        params = [r.params for r, _, _ in rows]
        logits = torch.stack([l for _, l, _ in rows])
        cut = params[0]
        if all((p.allowed_ids, p.top_k) == (cut.allowed_ids, cut.top_k) for p in params):
            logits = _restrict(logits, cut)  # one topk and one id upload, not N
        else:
            logits = torch.stack([_restrict(logits[i], p) for i, p in enumerate(params)])
        want_lp = any(p.logprobs for p in params)  # a greedy score is a second full softmax
        toks, lps = self._backend.sample_batch(
            logits, [p.temperature for p in params], [p.top_p for p in params],
            [_step_seed(r.params.seed, g) for r, _, g in rows], logprobs=want_lp,
        )
        self._last_logprobs = lps.tolist() if want_lp else None
        return toks.tolist()

    def _sample_commit(self, rows: list[tuple]) -> None:
        toks = self._sample_batch(rows)
        lps = self._last_logprobs
        for i, ((r, _, _), tok) in enumerate(zip(rows, toks)):
            self._commit(r, [tok], None if lps is None else [lps[i]])

    def _commit(self, req: _Req, toks: list[int], lps: list[float] | None = None) -> None:
        """Append sampled tokens in order, stopping at the first one the request
        did not take verbatim. Only the chain's last token may publish a prefix:
        the snapshot holds the state at the END of the commit."""
        p = req.params
        n, last = len(p.end_think_ids), len(toks) - 1
        for i, raw in enumerate(toks):
            tok = raw
            if (
                p.max_think_tokens is not None
                and n
                and not req.thought_closed
                and len(req.output) >= p.max_think_tokens
            ):  # budget spent: close the reasoning block instead of sampling
                tok = p.end_think_ids[len(req.output) - p.max_think_tokens]
            elif tok in p.stop_token_ids:
                self._finish(req)
                return
            req.output.append(tok)
            if lps is not None and i < len(lps):
                # a forced end-think token was not drawn, so it has no logprob
                req.logprobs.append(float("nan") if tok != raw else lps[i])
            if n and not req.thought_closed and tuple(req.output[-n:]) == p.end_think_ids:
                req.thought_closed = True
            req.tokens.append(tok)
            req.seq_len += 1
            self._tokens_generated += 1
            materialized = req.seq_len - 1
            if i == last and req.phase == _PHASE_DECODE and materialized % BLOCK_TOKENS == 0:
                self._publish_prefix(req, materialized)
            if len(req.output) >= p.max_new_tokens:
                self._finish(req)
                return
            if tok != raw:  # a forced end-think token: the rest of the chain is stale
                return

    def _publish_prefix(self, req: _Req, length: int) -> None:
        """Hand tokens[:length], its blocks and the linear-state snapshot at that
        boundary to the store; the store owns and evicts all three together."""
        snap = (
            self._states.states[req.state_slot].clone(),
            self._states.window_snapshot(req.state_slot),
        )
        self._prefix_published += self._prefix.insert(
            req.tokens[:length], req.blocks[: length // BLOCK_TOKENS], snap
        )

    def _finish(self, req: _Req, error: str | None = None) -> None:
        # Freed here, not at poll, so pool capacity returns immediately.
        req.phase = _PHASE_DONE
        for b in req.blocks:
            self._kv.free_block(b)
        self._blocks_used -= req.own_blocks
        self._states.free_slot(req.state_slot)
        self._slots_used -= 1
        if error is None:
            self._finished[req.req_id] = req.output
            if req.params.logprobs:
                self._finished_logprobs[req.req_id] = req.logprobs
        else:
            self._failed[req.req_id] = error
        self._finished_count += 1
        self._running.remove(req)

    def _loop(self) -> None:
        while not self._wake.is_set():
            with self._lock:
                has_running = bool(self._running)
                has_waiting = bool(self._waiting)
            if has_running or has_waiting:
                # Batch concurrent submissions: a burst of HTTP requests
                # arrives over ~10ms. Without this window the first one
                # starts a prefill alone and the rest land in eager mixed
                # ticks (decode graph off, ~10x slower per tick).
                if not has_running and has_waiting:
                    self._wake.wait(0.01)
                try:
                    self.step()
                except Exception:
                    # ponytail: log-and-continue (a crashed daemon hangs the server); backpressure is the upgrade.
                    import traceback

                    traceback.print_exc()
            else:
                self._wake.wait(0.005)


def _fit_blocks(cfg, backend, io, cap: int, draft_layers: int = 0) -> int:
    """KV blocks that fit the free memory left after the weights and the GDN pools.

    Called with the state pool already allocated, so free memory is measured, not
    estimated -- and only after ``empty_cache``, without which the allocator's load
    reserve hides 13 GiB and this returns 64 blocks. The default it replaces was
    ``ctx * max_batch``, which on the 27B's own 262144-token limit is 275 GB of f32
    KV; every bench passed num_blocks explicitly, so serve was the one caller that
    ever saw it.

    ``draft_layers`` is the draft's own layer count, 0 for a dense engine. The draft
    mirrors ``num_blocks`` into a pool of its own (``DraftHead.attach``), and that
    pool's planes are not free: at the 27B's 16 full-attn planes one draft layer is
    1/16 of the trunk's bytes, so charging for it when there is no draft under-fits
    every training engine, and not charging when there is one over-fits serve.

    Holds back a third of what is free: ``PrefixStore`` takes a quarter of what
    remains after this, and the attention partials are transient and scale with B*S.
    ``cap`` wins over the 64-block floor -- a caller asking for fewer means it.
    """
    if backend.device.type != "cuda":
        return cap or 256
    planes = 2 * len(cfg.full_attn_layers)
    per_block = planes * cfg.num_kv_heads * BLOCK_TOKENS * cfg.head_dim
    per_block *= torch.tensor([], dtype=io).element_size()
    # per_block already carries the K+V factor, so one draft layer is per_block over
    # the PLANE-PAIR count, not over `planes`. Dividing by `planes` charged half a
    # layer and over-asked by 3.03% on the 27B.
    per_block += draft_layers * per_block // len(cfg.full_attn_layers)
    fit = max(64, int(torch.cuda.mem_get_info()[0] * 2 / 3) // per_block)
    return min(fit, cap) if cap else fit


def _weight_fingerprint(cfg) -> str:
    """What the spilled KV was computed under, as far as the config knows.

    EVERY config field, not a hand-picked list of the ones that look load-bearing: a
    mismatch is the only thing standing between a restart and serving KV computed under
    other weights, and a field left out of the list is exactly how that happens. The
    first draft of this named `cfg.num_heads`, which does not exist -- the real field is
    `num_attention_heads` -- so the list was already wrong when it was written.

    It does NOT distinguish two checkpoints of the same architecture. Pass
    `ssd_fingerprint` explicitly when one spill directory serves both.
    """
    import dataclasses

    fields = "-".join(f"{f.name}={getattr(cfg, f.name)!r}" for f in dataclasses.fields(cfg))
    return f"{fields}-block{BLOCK_TOKENS}"


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
    max_blocks: int = 0,
    prefix_store: Any = None,
    #: host-tier budget for demoted GDN snapshots; 0 is off, which is the default until a
    #: workload is measured where it wins. A single conversation is not one: it re-reads
    #: only its newest entry, so the LRU snapshot a demotion picks is never asked for
    #: again -- measured, 43 demotions and 0 promotions, with the wall clock 1.51x worse.
    #: **No CLI flag, deliberately: settable only from here.** The condition is
    #: `concurrent sessions > HBM snapshot budget` (measured: 2 -> 0 promotions, 9 -> 17,
    #: 12 -> 24), and a serving operator cannot read either operand off the command line,
    #: so a flag would mostly be turned on below the threshold where it is 1.51x worse.
    #: `/health`'s `dram_promotions` is what says the workload crossed it.
    dram_bytes: int = 0,
    #: directory for the SSD prefix tier; "" is off. Unlike the DRAM tier this one does
    #: not need concurrent sessions to pay: after a restart HBM is empty, so the first
    #: lookup of every returning conversation reaches back and the disk is what answers.
    #: ``ssd_fingerprint`` must change whenever the weights do -- a tier serving KV
    #: computed under other weights is silently wrong, and the fingerprint is the only
    #: thing that stops it. Defaults to the model's shape, which does NOT cover a
    #: different checkpoint at the same shape.
    #: # ponytail: shape-derived fingerprint; hash the weights when two checkpoints of
    #: #   one architecture are served from one spill dir
    ssd_path: str = "",
    ssd_fingerprint: str = "",
    decode_graph: bool | None = None,
    draft: Any = None,
    spec_depth: int | None = None,
) -> Engine:
    """Wire a model + backend into an Engine; pool shapes come from ``cfg``.
    ``decode_graph`` None auto-enables the captured decode tick on CUDA.
    ``num_blocks`` 0 fits the KV pool to free memory, capped at ``max_blocks``."""
    n_linear = cfg.num_layers - len(cfg.full_attn_layers)
    if draft is not None:
        draft.set_depth(spec_depth)  # the state pool is sized by the width it settles on
    model.params = backend.materialize(model.params)
    # Serve the draft's own weights HERE, before anything reads free memory. They used
    # to be quantized inside Engine.__init__, i.e. after the KV fit had already spent
    # 2/3 of what was free and PrefixStore a quarter of the rest -- so the draft's fp4
    # weights were charged to nothing and `serve --blocks 0 --draft` died in
    # `materialize`'s twiddle with 104 MiB free, never reaching the fit's own print
    # (measured at serve's default --slots 16 on a 32 GB V100). Same ordering fix the
    # state pool below already uses: allocate, then MEASURE what is left.
    if draft is not None:
        _serve_draft(draft, backend)
    # Reclaim the allocator's load-time reserve before anything reads free memory.
    # Loading and quantizing 27B leaves 29.02 GiB reserved against 15.96 allocated on
    # a 31.74 GiB card, and `mem_get_info` counts all 13.06 GiB of that as USED -- so
    # the store's budget below, and the KV fit, see 2.36 GiB free instead of 15.17.
    # Measured on sm70: this one call is the difference between a 1024-token and a
    # 62832-token context at B=1.
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()
    pad = _graph_on(backend, decode_graph)  # the replay's padding row owns a slot and a block
    # The GDN pools first, so fitting the KV pool below can MEASURE what is left
    # instead of estimating it: at slots=3 depth=3 they are 2.94 GiB, 79% of it the
    # per-step verify states, which scale with slots*width and not with max_batch.
    state_pool = LinearStatePool(
        num_slots + pad,
        n_linear,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        device=backend.device,
        dtype=precision.dtype("recurrent_state", backend.device),
        conv_window=cfg.linear_conv_kernel_dim - 1,
        conv_dim=cfg.linear_qkv_dim,
        spec_steps=draft.width if draft is not None else 0,
    )
    # The KV pool's dtype IS the attention kernel's ABI, but only a cuda kernel has one
    # here: main passed NO dtype and every target took PagedKvPool's bf16 default, so
    # routing backend.io in unconditionally moved cpu and metal from bf16 to f32 --
    # K and V stopped being rounded on store on the cell that certifies every kernel in
    # this repo, at 2x the bytes, with the whole suite green (a parity check moves the
    # TileLang and torch sides together, so nothing could see it). io itself is right on
    # cpu; PagedKvPool's DEFAULT is what disagrees with it, so narrow the call site.
    # getattr on BOTH: RefBackend and the other test doubles declare neither, and
    # asking for .arch directly raised AttributeError in 7 tests.
    kv_io = (getattr(backend, "io", torch.bfloat16)
             if getattr(backend, "arch", "").startswith("sm") else torch.bfloat16)
    if not num_blocks:
        num_blocks = _fit_blocks(cfg, backend, kv_io, max_blocks,
                                 draft_layers=0 if draft is None else draft.cfg.num_layers)
    kv_pool = PagedKvPool(
        num_blocks + pad,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=backend.device,
        layer_map=cfg.full_attn_layers,
        # Match the attention kernel's IO dtype. sm70's is f32, and a bf16 pool
        # made every attention call cast the WHOLE plane (all num_blocks, not
        # the live ones): 4.71 ms/token, 14% of a 4096-ctx token, independent of
        # context. Same trade the state pool makes below. getattr: test doubles
        # stand in for Backend without declaring an io dtype.
        dtype=kv_io,
    )
    # A resident store entry owns a GDN state snapshot in HBM (144 MiB at 27B f32)
    # and a decode publishes one every BLOCK_TOKENS, so the store's byte budget must
    # fit the card: the 8 GiB default is most of a 32 GB V100's post-weights headroom.
    # Spend a quarter of what is still free after weights and pools.
    kw = {}
    if backend.device.type == "cuda":
        kw["state_bytes"] = int(torch.cuda.mem_get_info()[0] // 4)
        # Host tier for snapshots the card cannot keep resident. Measured on the live
        # V100: 43 of 43 evictions happened with 64% of the block pool free, so every one
        # was state bytes -- a prefix thrown away for a byte the host could hold. 4 GiB
        # rather than the ~25 GiB free: pinned pages cannot be swapped and this pod has
        # 31 GiB of RAM against a 32 GiB card, so pinning most of it destabilises the
        # host, not the process. 4 GiB is 28 snapshots against HBM's 9.
        if dram_bytes:
            kw["dram"] = DramSnapshots(budget_bytes=dram_bytes)
    if ssd_path:
        # Not gated on cuda: the tier is target-independent, and the CPU target is where
        # its parity is checked.
        kw["ssd"] = KvTier(ssd_path, ssd_fingerprint or _weight_fingerprint(cfg))
    store = PrefixStore(kv_pool, **kw) if prefix_store is None else prefix_store
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
        draft=draft,
        spec_depth=spec_depth,
    )
