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
Speculation (optional, ``draft=``): a decode row drafts up to ``spec_depth``
tokens off the trunk's last hidden, and the SAME forward verifies them as a
seq_q = 1+depth row — no second code path. The trunk's paged KV needs no
rollback (a rejected draft's slot is overwritten next tick), but the gated-delta
recurrent state does: the verify forward keeps the state after every chain step
(``BatchKv.keep_steps``) and the engine adopts the one at the accepted length.
Spec ticks run eager; pure-decode ticks without a draft still replay the graph.

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
from .spec import survival, verify_lens

_PREFILL_BUCKET = 64  # prefill widths are padded to this: bounded kernel shapes

#: Decode-graph size ladder. A tick rounds UP to the first entry >= its row
#: count and pads; without it every distinct batch size captured its own graph
#: and its own memory pool.
_GRAPH_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

__all__ = ["Engine", "SamplingParams", "StepLimits", "BatchKv", "build_engine"]

_PHASE_PREFILL = 1
_PHASE_DECODE = 2
_PHASE_DONE = 3

_HASH_MASK = 0x7FFFFFFF


def _quantize_draft(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Re-serve a draft head's dense weights as block-quantized fp8.

    Norms, embeddings and anything 1-D stay as they are; only the [N,K]
    projections move, which is where all of the head's bulk and all of its time
    is."""
    from tilerl_kernels import reference

    out: dict[str, torch.Tensor] = {}
    for k, v in params.items():
        if v.ndim == 2 and v.shape[0] >= 128 and v.shape[1] >= 128:
            out[f"{k}.w8"], out[f"{k}.wscale"] = reference.quant_fp8(v)
        else:
            out[k] = v
    return out


def _step_seed(seed: int, generated: int) -> int:
    """Deterministic per-(request, position) sampling seed.

    Both terms are full-width multiplicative hashes: a shift-then-mask would
    keep only the seed's low bits (the old ``seed << 20`` form collapsed seeds
    1/2049/16385 to one stream — OPD replays byte-identical rollouts past
    step 2048)."""
    return ((int(seed) * 2_654_435_761) ^ (generated * 2_246_822_519)) & _HASH_MASK


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 16
    seed: int = 0
    stop_token_ids: tuple[int, ...] = ()
    #: restrict sampling to these ids (multiple-choice eval, constrained RL
    #: actions); None = full vocabulary
    allowed_ids: tuple[int, ...] | None = None
    #: thinking effort: force ``end_think_ids`` after this many generated
    #: tokens if the model has not closed its reasoning block itself. 0 = do
    #: not think at all; None = unbounded. The ids are the caller's, so the
    #: engine stays tokenizer-free.
    thinking_budget: int | None = None
    end_think_ids: tuple[int, ...] = ()
    #: also return log p of each sampled token under the temperature-scaled
    #: distribution it was drawn from — what a policy gradient needs, and what
    #: an eval needs to score a completion without a second forward.
    logprobs: bool = False


def _restrict(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    if params.allowed_ids is None:
        return logits
    keep = torch.full_like(logits, float("-inf"))
    idx = torch.tensor(params.allowed_ids, device=logits.device)
    keep[..., idx] = logits[..., idx]
    return keep


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
    #: trunk hidden [1,w,H] for absolute positions
    #: ``[hidden_from .. hidden_from+w-1]`` — the draft head's fc input. The
    #: WHOLE forward, not its last position: the draft runs over every position
    #: it has not covered yet, and one position of overlap is what lets a
    #: chunked prefill's runs meet without a gap (``hidden_prev``).
    hidden: torch.Tensor | None = None
    hidden_prev: torch.Tensor | None = None  # [1,1,H] at hidden_from-1
    hidden_from: int = 0
    #: highest absolute position whose draft KV belongs to a COMMITTED token.
    #: A rejected chain writes past it; the next run overwrites those.
    draft_pos: int = 0
    drafts: list[int] = field(default_factory=list)  # next tick's chain, minus its first token


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
    #: speculative verify: keep the recurrent state after each of the first
    #: ``keep_steps`` chain tokens, so the accepted length can select one.
    keep_steps: int = 0


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

    def __init__(self, model, backend, kv_pool, state_pool, batch_size, width=1, pool=None,
                 last_only=False, keep=0):
        device = backend.device
        B, W = batch_size, width
        # int32 end to end: every consumer is a kernel taking int32, and a
        # long buffer costs an int64->int32 cast launch per use inside the graph.
        self._b = B
        self._w = W
        self._ids = torch.empty(B, W, dtype=torch.int32, device=device)
        self._pos = torch.empty(B, W, dtype=torch.int32, device=device)
        self._bt = torch.zeros(B, kv_pool.num_blocks, dtype=torch.int32, device=device)
        self._sl = torch.empty(B, dtype=torch.int32, device=device)
        self._ss = torch.empty(B, dtype=torch.int32, device=device)
        # Every row carries the same W query tokens — a plain decode tick, or a
        # speculative verify of one fixed-length chain. A static GPU buffer (not
        # a per-tick CPU copy) keeps seq_q_lens out of the captured region: the
        # kernels' CPU->GPU fallback breaks CUDA graph capture. Uniform W is why
        # a captured spec tick cannot use verify_lens' per-row trim.
        self._sql = torch.full((B,), W, dtype=torch.int32, device=device)
        # Pinned staging buffers: a plain copy_ from an unpinned CPU tensor is
        # synchronous (it blocks until the copy engine drains), which under
        # GPU contention costs ms per tick. Pinned + non_blocking makes the
        # H2D copies async — stream ordering keeps replay after them.
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
            # Only a verify tick keeps the recurrent state after every chain
            # step; the step buffers are static, so the write captures like any
            # other kernel. A prefill chunk is also W>1 and must NOT ask for
            # them — its pool has none, and the request dies mid-capture.
            keep_steps=keep,
        )
        # Warmup on a side stream: tilelang JIT-compiles per (shape, dtype),
        # and JIT is host work — it must finish before capture starts.
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
        # One memory pool shared by every bucket's graph. Without it each
        # capture takes a private pool that is never returned, so the ladder's
        # seven buckets cost seven copies of a decode tick's whole working
        # set. Only one graph replays at a time and the logits contract is
        # already "valid until the next replay", so sharing is safe - it is
        # what vLLM and sglang do.
        with torch.cuda.graph(self._graph, pool=pool):
            self._logits = model.forward(self._ids, self._pos, self._kv, backend,
                                         hidden_out=hid, last_only=last_only)
        # Captured, so this tensor is rewritten in place by every replay — the
        # draft head reads the previous tick's hidden from it directly.
        self.hidden = hid[-1] if hid else None

    def run(self, reqs, chains=None, pad=None):
        """Copy per-tick inputs into the static buffers and replay.

        Returns the static logits [B,W,V]; valid until the next replay.
        ``chains[i]`` is row i's ``[last committed token, drafts...]``, all of
        length W. ``pad`` is ``(state_slot, block)`` for the rows beyond
        ``len(reqs)``: a captured graph runs its full width every replay, so
        those rows still WRITE to the KV and recurrent pools. Leaving them on
        a finished request's slot would let them overwrite whatever request
        was given that slot next.
        """
        for i, r in enumerate(reqs):
            if r.phase == _PHASE_PREFILL:
                # A prefill chunk is the same static shape as a decode chain —
                # the engine buckets its width — so it captures the same way.
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
        spec_depth: int = 4,
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
        self._decode_graphs: dict = {}
        # Reserved LAZILY, on the first tick that actually pads: a replay runs
        # its full captured width and the padding rows still write to both
        # pools, so they must not land on a slot a live request owns. Taking
        # them at construction would shrink every engine's usable pool by one
        # slot even when nothing ever pads.
        self._pad_slot: int | None = None
        self._pad_block: int | None = None
        self._graph_pool = None

        # Speculation: the draft is one full-attn stack with its OWN kv plane
        # and no recurrent state — it must never reach the trunk's GDN slots.
        # The plane spans the trunk's whole block space and is indexed by the
        # request's own block list, so the draft attends over the same prefix
        # the trunk does. A chain-local block would leave its attention reading
        # ONE token (softmax over one position is the identity on v), which is
        # what made the loop's acceptance 55.8% against the probe's 84.4%.
        self._draft = draft
        self._spec_depth = spec_depth if draft is not None else 0
        if draft is not None:
            # The head ships dense bf16, which Backend.linear serves on its
            # generic path at ~30 GB/s: 9.7 ms per projection against 0.13 ms
            # for the same shape on the trunk's fp8 kernel, so one draft step
            # cost more than the whole 64-layer trunk forward. Serve it the way
            # the trunk is served — Model._linear picks .w8/.wscale up itself.
            served = backend.materialize(_quantize_draft(draft.params))
            # In place, never rebound: DraftHead.layers is a Model holding THIS
            # dict, and a fresh one leaves it reading the original bf16 weights.
            draft.params.clear()
            draft.params.update(served)
            if not 0 < spec_depth < BLOCK_TOKENS:
                raise ValueError(f"spec_depth must be in [1, {BLOCK_TOKENS}), got {spec_depth}")
            self._draft_kv = PagedKvPool(
                kv_pool.num_blocks, draft.cfg.num_kv_heads, draft.cfg.head_dim,
                num_layers=draft.cfg.num_layers, device=backend.device,
                layer_map=tuple(range(draft.cfg.num_layers)),
            )

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
        self._spec_drafted = 0
        self._spec_accepted = 0
        self._finished_logprobs: dict[int, list[float]] = {}
        self._last_logprobs: list[float] | None = None

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
            # +depth: a verify tick materializes the drafts past the last token
            if self._kv.blocks_for_tokens(total + self._spec_depth) > self._kv.num_blocks:
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
        """log p of each returned token, or None unless the request asked.

        Pops: take() drains the tokens and this drains the scores, so a served
        request leaves nothing behind.
        # ponytail: a caller that asks for scores and never reads them holds
        # them until the engine is dropped; a TTL sweep is the upgrade path.
        """
        with self._lock:
            return self._finished_logprobs.pop(request_id, None)

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
        """Plan one tick, mirroring agent-infer's ``build_forward_plan``.

        Admit as many waiting requests as ``max_batch`` allows, then take
        all running decodes as decode rows and as many prefill rows as the
        per-tick token budget (``max_num_batched_tokens`` minus the decode
        rows) and one width bucket allow. A prompt longer than the budget
        stays in PREFILL and is chunked across ticks.

        One prefill row per tick left the budget mostly idle: a 16-token
        prompt used 16 of 512, so a burst of short prompts prefilled one per
        forward. That is the shape a synthetic-data workload has.

        Admission is a whole pass, not one request per tick: a burst of B
        submissions used to need B ticks just to reach the running queue, so
        every tick of the ramp ran a batch smaller than the one that was
        asked for. vLLM/sglang schedule the whole waiting queue each pass.
        """
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
            # Rows are padded to a shared width, so packing a 16-token prompt
            # beside a 512-token one would compute 512 wide for both. The
            # width is bucketed anyway: pack only within one bucket and the
            # padding costs nothing.
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
                "spec_drafted": self._spec_drafted,
                "spec_accepted": self._spec_accepted,
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

    def _make_kv(self, reqs: list[_Req], seq_q: list[int], keep_steps: int = 0) -> BatchKv:
        # Fixed width = pool size: the kernels bake the table width into the
        # compiled kernel (Mb is a compile const), so a per-tick max-blocks
        # width recompiles on every block growth. Rows zero-pad; the kernels
        # index by position bounded by seq_lens, so padding is never read.
        bt = torch.zeros(len(reqs), self._kv.num_blocks, dtype=torch.long, pin_memory=self._pin)
        sl = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        ss = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        sql = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        for i, r in enumerate(reqs):
            bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
            # Materialized length AFTER this forward: a prefill row completes
            # its chunk (prefill_from is the chunk start); a decode row writes
            # its chain at [seq_len-1, seq_len-1+seq_q), so the post-write
            # length is seq_len-1+seq_q (== seq_len for a plain T=1 decode).
            sl[i] = (
                r.prefill_from + seq_q[i]
                if r.phase == _PHASE_PREFILL
                else r.seq_len - 1 + seq_q[i]
            )
            ss[i] = r.state_slot
            sql[i] = seq_q[i]
        if self._pin:
            # Move once here, not once per layer inside every kernel's _dev:
            # these four are built on the host and read by ~4 kernels per layer,
            # and an unpinned H2D copy is SYNCHRONOUS. A 64-layer prefill issued
            # 971 pageable copies, which is host stall the GPU-busy total cannot
            # see (2058 tok/s GPU-bound against 1836 end to end).
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
        """Run one mixed/decode/prefill forward over the planned rows.

        Rows are left-aligned valid tokens padded to a shared T (decode rows:
        1 token, or a 1+depth draft chain on a verify tick; the prefill row:
        its chunk), with per-row ``seq_q_lens`` so the kernels touch only valid
        positions.
        """
        # Speculate on pure-decode ticks only: a mixed tick's width is the
        # bucketed prefill chunk, which the step-state buffers cannot cover.
        chains = (
            [[r.output[-1], *r.drafts] for r in decodes]
            if self._draft is not None and decodes and not prefills
            else None
        )
        if chains is not None and max(map(len, chains)) == 1:
            chains = None  # the policy kept nothing: a plain decode tick
        elif chains is not None and self._decode_graph_on:
            # One graph per (B, width): pad every row to the widest chain so the
            # captured shape is fixed. verify_lens' per-row trim cannot survive
            # capture, and the marginal cost of a wider row inside a replay is
            # far below the 7.9x the graph itself is worth.
            w = max(map(len, chains))
            for c in chains:
                # Any pad token is correct: a draft is only ever accepted when
                # it equals what the trunk sampled there, so a repeat is simply
                # a draft that gets rejected.
                c.extend([c[-1]] * (w - len(c)))
        q_dec = [len(c) for c in chains] if chains else [1] * len(decodes)
        growth = sum(
            max(0, (r.seq_len + q - 1 + BLOCK_TOKENS) // BLOCK_TOKENS - len(r.blocks))
            for r, q in zip(decodes, q_dec)
        )
        if growth:
            evict = getattr(self._prefix, "evict_until_free", None)
            if evict is not None:
                evict(growth)
        for r, q in zip(decodes, q_dec):
            # Blocks must cover the chain's last position, r.seq_len-1+q.
            # Exhaustion raises -> step() finishes the running requests.
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
        # Bucket the forward width: tilelang kernels specialize on the shape,
        # so a width equal to the prompt length compiles a kernel set per
        # distinct prompt (MMLU: 662 variants in 20 min, GPU idle). Padding
        # rows are masked by seq_q_lens everywhere; the true chunk indexes the
        # logits below.
        # A verify width is exact (a handful of shapes, 1+depth at most); only
        # a real prefill chunk is bucketed.
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
            # Per-row valid lengths: a mixed tick's decode rows end at 1
            # while the prefill row spans the width, and lm_head must not run
            # over the padding. A verify tick needs every chain position, so
            # it opts out entirely.
            self._backend, hidden_out=hid,
            last_only=False if chains else seq_q,
        )
        if hid is not None:
            for i, r in enumerate(rows):  # the draft's fc input, whole width
                # hidden_out is appended before last_only's slice: always full width
                r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
                r.hidden = hid[-1][i : i + 1, : seq_q[i]]
                r.hidden_from = int(positions[i, 0])
        if chains:
            self._verify(decodes, chains, logits, hid[-1])
        else:
            self._sample_commit([(r, logits[i, 0], len(r.output)) for i, r in enumerate(decodes)])
        if prefills:
            self._prefill_forwards += 1
            # last_only may have collapsed the T axis to the final token.
            self._finish_prefills(prefills, chunks, logits, len(decodes))
        if decodes:
            self._decode_forwards += 1
        if decodes and prefills:
            self._mixed_forwards += 1
        if self._draft is not None:
            # Every tick that materialized tokens, not just pure-decode ones: a
            # chunked prefill would otherwise leave the chunks it did not draft
            # over with no KV, and the draft's attention would read them.
            self._draft_step(rows)

    def _finish_prefills(self, prefills: list[_Req], chunks: list[int], logits, base: int) -> None:
        """Advance every prefill row by its chunk and, for the ones that
        completed their prompt, sample the first token in ONE batched call and
        publish the prefix."""
        done = []
        for k, (pf, c) in enumerate(zip(prefills, chunks)):
            pf.prefill_from += c
            pf.seq_len = pf.prefill_from
            if pf.prefill_from >= len(pf.tokens):
                done.append((pf, logits[base + k, min(c, logits.shape[1]) - 1], 0))
        if not done:
            return
        self._sample_commit(done)
        for pf, _, _ in done:
            # Publish the prompt prefix at a block boundary: the state slot
            # still covers exactly the prompt tokens, so the snapshot is exact.
            prompt_len = len(pf.tokens) - len(pf.output)
            if pf.phase != _PHASE_DONE and prompt_len % BLOCK_TOKENS == 0:
                self._publish_prefix(pf, prompt_len)
            if pf.phase != _PHASE_DONE:
                if len(pf.output) >= pf.params.max_new_tokens:
                    self._finish(pf)
                else:
                    pf.phase = _PHASE_DECODE

    def _run_decode_graph(self, reqs: list[_Req], chains=None) -> bool:
        """Captured decode for a pure-decode tick, one graph per SIZE BUCKET.

        Capturing per exact batch size meant a batch draining from 32 to 1
        captured up to 32 graphs, each holding its own CUDA graph memory pool
        - tens of GiB that never come back, and the reason B=64 hit OOM
        during the drain while steady-state decode peaked at 39 GiB. vLLM and
        sglang capture a fixed ladder and pad up to it; same here.

        Returns False (and flips the flag off) when capture failed, so the
        caller runs eager.
        """
        n, W = len(reqs), len(chains[0]) if chains else 1
        B = next((c for c in _GRAPH_BUCKETS if c >= n), None)
        if B is None or self.limits.max_batch < B:
            B = n  # above the ladder: one exact-size graph rather than none
        if n < B and self._pad_slot is None:
            try:
                self._pad_slot = self._states.alloc_slot()
                self._pad_block = self._kv.alloc_block()
            except RuntimeError:
                B = n  # no spare capacity to park padding rows on: exact size
        g = self._decode_graphs.get((B, W))
        if g is None:
            try:
                if self._graph_pool is None:
                    self._graph_pool = torch.cuda.graph_pool_handle()
                g = _DecodeGraph(self._model, self._backend, self._kv, self._states, B,
                                 width=W, pool=self._graph_pool, keep=W if chains else 0)
            except Exception as exc:
                warnings.warn(
                    f"decode graph capture failed for B={B} W={W} ({exc}); eager fallback"
                )
                self._decode_graph_on = False
                return False
            self._decode_graphs[(B, W)] = g
        pad = None if self._pad_slot is None else (self._pad_slot, self._pad_block)
        logits = g.run(reqs, chains, pad=pad)
        self._decode_forwards += 1
        if chains:
            # No hidden from a replay: the next tick's draft reads it from the
            # graph's own static buffer, which the model wrote during capture.
            self._verify(reqs, chains, logits, g.hidden)
        else:
            if self._draft is not None and g.hidden is not None:
                for i, r in enumerate(reqs):  # keep the draft's fc input current
                    r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
                    r.hidden, r.hidden_from = g.hidden[i : i + 1], r.seq_len - 1
            self._sample_commit([(r, logits[i, -1], len(r.output)) for i, r in enumerate(reqs)])
        if self._draft is not None:
            self._draft_step(reqs)
        return True

    def _draft_step(self, rows: list[_Req]) -> None:
        """Draft over every position a row materialized but has not drafted yet.

        Position q consumes the trunk hidden at q-1 and the token at q, so the
        run spans ``[draft_pos+1 .. seq_len-1]`` and its LAST position is the
        draft for the next token: the KV fill and the draft are ONE forward,
        and the draft's attention never meets a gap. Runs at the END of a tick,
        while the trunk's hiddens are still live; the chain it leaves in
        ``r.drafts`` is what the next tick verifies.
        """
        dev = self._backend.device
        plan = []
        for r in rows:
            if r.hidden is None or r.phase == _PHASE_DONE:
                continue
            lo, hi = max(1, r.draft_pos + 1), r.seq_len - 1
            if hi < lo:
                continue
            plan.append((r, lo, hi))
        if not plan:
            return
        w = max(hi - lo + 1 for _, lo, hi in plan)
        nb = max(len(r.blocks) for r, _, _ in plan)
        n = len(plan)
        ids = np.zeros((n, w), dtype=np.int64)
        pos = np.zeros((n, w), dtype=np.int64)
        bt = torch.zeros(n, nb, dtype=torch.long)
        hs, sl, sq = [], [], []
        for i, (r, lo, hi) in enumerate(plan):
            q = hi - lo + 1
            ids[i, :q] = r.tokens[lo : hi + 1]
            pos[i, :q] = np.arange(lo, hi + 1)
            bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
            sl.append(hi + 1)
            sq.append(q)
            # hidden at [lo-1 .. hi-1]; hidden_prev supplies the one position
            # that belongs to the previous forward.
            h, base = r.hidden, r.hidden_from
            if r.hidden_prev is not None:
                h, base = torch.cat([r.hidden_prev, r.hidden], dim=1), base - 1
            off = (lo - 1) - base
            hs.append(torch.nn.functional.pad(h[:, off : off + q], (0, 0, 0, w - q)))
        kv = BatchKv(
            block_table=bt.to(dev), seq_len=torch.tensor(sl, device=dev),
            state_slot=torch.zeros(n, dtype=torch.long, device=dev),
            kv_pool=self._draft_kv, state_pool=None,
            seq_q_lens=torch.tensor(sq, device=dev),
        )
        dh: list = []
        logits = self._draft.forward(torch.cat(hs, dim=0), ids, pos, kv, self._backend,
                                     hidden_out=dh)
        last = torch.tensor([q - 1 for q in sq], device=dev)
        rng = torch.arange(n, device=dev)
        tok, prob = self._backend.greedy(logits[rng, last].unsqueeze(1))
        h = dh[-1][rng, last].unsqueeze(1)
        confs: list[list[float]] = [[] for _ in plan]
        if self._spec_depth > 1:
            conf = self._draft.confidence(h, prob, self._backend)
            for i, c in enumerate(conf[:, -1].tolist()):
                confs[i].append(float(c))
        chains = [[int(t)] for t in tok[:, -1].tolist()]
        for i, (r, _, hi) in enumerate(plan):
            if r.draft_pos == 0:
                # Position 0 is never drafted (a draft at q reads the hidden at
                # q-1), but attention over [0, seq_len) still reads its page —
                # which a recycled block leaves holding another request's.
                b = r.blocks[0]
                self._draft_kv.k_pool[:, b, :, 0, :] = 0
                self._draft_kv.v_pool[:, b, :, 0, :] = 0
            r.draft_pos = hi

        # Remaining chain steps, one position each. Bounded by the blocks the
        # row already owns: this runs after the tick's allocation, and a write
        # past them would land on another request's page.
        # ponytail: clamps the chain instead of allocating; a row at a block
        # boundary drafts shorter for one tick.
        for j in range(1, self._spec_depth):
            live = [i for i, (r, _, hi) in enumerate(plan)
                    if hi + j < len(plan[i][0].blocks) * BLOCK_TOKENS]
            if not live:
                break
            li = torch.tensor(live, device=dev)
            kv = BatchKv(
                block_table=bt[live].to(dev),
                seq_len=torch.tensor([plan[i][2] + 1 + j for i in live], device=dev),
                state_slot=torch.zeros(len(live), dtype=torch.long, device=dev),
                kv_pool=self._draft_kv, state_pool=None,
                seq_q_lens=torch.ones(len(live), dtype=torch.long, device=dev),
            )
            dh = []
            logits = self._draft.forward(
                h[li], np.array([[chains[i][-1]] for i in live], dtype=np.int64),
                np.array([[plan[i][2] + j] for i in live], dtype=np.int64),
                kv, self._backend, hidden_out=dh,
            )
            tok, prob = self._backend.greedy(logits)
            conf = self._draft.confidence(dh[-1], prob, self._backend)
            for k, c in enumerate(conf[:, -1].tolist()):
                confs[live[k]].append(float(c))
            for k, t in enumerate(tok[:, -1].tolist()):
                chains[live[k]].append(int(t))
            h = h.index_copy(0, li, dh[-1])

        keep = verify_lens([survival(c) for c in confs]) if self._spec_depth > 1 \
            else [1] * len(plan)
        for i, (r, _, _) in enumerate(plan):
            p = r.params
            if p.thinking_budget is not None and p.end_think_ids and not r.thought_closed:
                keep[i] = 0  # a forced end-think token is not the sampler's
            r.drafts = chains[i][: keep[i]]

    def _verify(self, rows, chains, logits, hidden) -> None:
        """Accept the leading run of drafts the trunk agrees with, adopt the
        recurrent state at that length, and commit the accepted prefix plus the
        trunk's own bonus token.

        Sampling is per (row, chain position) with the same per-generated-index
        seed a T=1 rollout would use, so an accepted token is bit-identical to
        the unspeculated one. Rejected drafts leave KV past the new length,
        which the next tick overwrites."""
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
        """Batched sampling for a decode tick: one sort/softmax over all rows
        instead of B per-row calls (8.2% of the B=8 slice tick was 8 separate
        sorts + D2H syncs). Per-row seeds keep the draws identical to the
        per-row path. Returns the sampled tokens; the caller commits them (a
        verify tick samples every chain position and commits only the
        accepted prefix)."""
        if not rows:
            return []
        logits = torch.stack([_restrict(l, r.params) for r, l, _ in rows])
        dev = logits.device
        # Host-side: sample_batch branches on them, and shipping them to the
        # device only to read them back was 2 syncs a tick plus one per row.
        temps = [r.params.temperature for r, _, _ in rows]
        top_ps = [r.params.top_p for r, _, _ in rows]
        seeds = [_step_seed(r.params.seed, g) for r, _, g in rows]
        toks = self._backend.sample_batch(logits, temps, top_ps, seeds).tolist()
        if any(r.params.logprobs for r, _, _ in rows):
            # From the SAME logits the draw used. Temperature > 0 scores under
            # the SAMPLING distribution, which is what a policy gradient needs.
            # Temperature 0 scores under the model's own (t=1) distribution:
            # the greedy point mass would report log p = 0 for every token,
            # which is true and useless — and greedy is exactly the eval case
            # that wants the model's real score.
            t = torch.tensor([x if x > 0 else 1.0 for x in temps], device=dev).reshape(-1, 1)
            lp = torch.log_softmax(logits.float() / t, dim=-1)
            idx = torch.tensor(toks, device=dev).reshape(-1, 1)
            self._last_logprobs = lp.gather(1, idx).reshape(-1).tolist()
        else:
            self._last_logprobs = None
        return toks

    def _sample_commit(self, rows: list[tuple]) -> None:
        """Sample one token per row and commit it (every non-verify path)."""
        toks = self._sample_batch(rows)
        lps = self._last_logprobs
        for i, ((r, _, _), tok) in enumerate(zip(rows, toks)):
            self._commit(r, [tok], None if lps is None else [lps[i]])

    def _commit(self, req: _Req, toks: list[int], lps: list[float] | None = None) -> None:
        """Append sampled tokens in order, stopping at the first one the
        request did not take verbatim (finished, or a forced end-think
        rewrite).

        Only the last token of a chain may publish its prefix: the snapshot
        holds the recurrent state at the END of the commit, so a boundary
        crossed earlier in the chain is skipped (a missed cache entry, never a
        poisoned one)."""
        p = req.params
        n, last = len(p.end_think_ids), len(toks) - 1
        for i, raw in enumerate(toks):
            tok = raw
            if (
                p.thinking_budget is not None
                and n
                and not req.thought_closed
                and len(req.output) >= p.thinking_budget
            ):  # budget spent: close the reasoning block instead of sampling
                tok = p.end_think_ids[len(req.output) - p.thinking_budget]
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
            self._prefix_state[key] = (
                self._states.states[req.state_slot].clone(),
                self._states.window_snapshot(req.state_slot),
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
            if req.params.logprobs:
                self._finished_logprobs[req.req_id] = req.logprobs
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
    draft: Any = None,
    spec_depth: int = 4,
) -> "Engine":
    """Wire a model + backend into a running Engine (pools + prefix store).

    ``cfg`` is a :class:`~tilerl.config.ModelConfig`; the factory derives the
    pool shapes from it. Pass ``prefix_store`` to inject a test double (e.g. a
    never-match store for the miss path). ``decode_graph`` auto-enables the
    captured decode tick on CUDA (design-engine.md); pass False to force the
    eager path. ``draft`` (a :class:`~tilerl.spec.DraftHead`) turns speculative
    decoding on at ``spec_depth``, which sizes the state pool's per-chain-step
    planes.
    """
    n_linear = cfg.num_layers - len(cfg.full_attn_layers)
    model.params = backend.materialize(model.params)
    kv_pool = PagedKvPool(
        num_blocks,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=backend.device,
        layer_map=cfg.full_attn_layers,
    )
    state_pool = LinearStatePool(
        num_slots,
        n_linear,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        device=backend.device,
        # sm90's fused GDN kernel is f32-IO: a bf16 pool cost two 1.5MB casts
        # per layer per tick. +1.2 GiB at 16 slots on the 27B.
        dtype=torch.float32 if backend.device.type == "cuda" else torch.bfloat16,
        conv_window=cfg.linear_conv_kernel_dim - 1,
        conv_dim=cfg.linear_qkv_dim,
        spec_steps=1 + spec_depth if draft is not None else 0,
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
        draft=draft,
        spec_depth=spec_depth,
    )
