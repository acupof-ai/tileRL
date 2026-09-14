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
output. The engine is tokenizer-free unless a caller asks for text stop
sequences: ``decode=`` is the one place ids become text, for ``stop_texts``.
# ponytail: no preemption/swap — a row holds its slot from submit to finish.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kv_cache import BLOCK_TOKENS, BatchKv, NoPrefixStore
from .kv_tiers import SpillWriteError
from .spec import _PREFILL_BUCKET, LADDER_WIDTHS


def _last_prefill_boundary(n: int) -> int:
    """Where an UNINTERRUPTED walk ends the final aligned chunk of an `n`-token
    ragged prompt; 0 if aligned. The true walk is schedule-dependent -- decode
    rows sharing a tick shrink the budget -- so this predicts the common case,
    and `_finish_prefills` holds a snapshot for walks that walk past it."""
    tail = n % BLOCK_TOKENS
    if not tail:
        return 0
    end = (n // BLOCK_TOKENS) * BLOCK_TOKENS
    return end - BLOCK_TOKENS if tail == 1 else end


def _decode_extra_blocks(seq_len: int, q: int, held: int) -> int:
    """New blocks a decode tick must grow. The verify forward rewrites position
    seq_len-1 (the anchor), so its last PHYSICAL write is seq_len+q-2; covering
    seq_len+q-1 demanded one block that is never written and killed saturated
    final ticks (~1/16, trigger (prompt+max_new)%16==1)."""
    return max(0, (seq_len + q - 2 + BLOCK_TOKENS) // BLOCK_TOKENS - held)


#: Set after the one-time sm70 graph warning, so the three _graph_on callers
#: (Engine init, build_engine pad sizing, the CLI slot fit) do not repeat it.
_sm70_graph_warned = False


def _graph_on(backend, decode_graph: bool | None) -> bool:
    """The captured decode tick is on by default on CUDA only. One definition:
    ``build_engine`` sizes the pools for the pad row from the same answer the
    engine reserves it on.

    sm70 is excluded from the AUTO path: dense decode capture fails there
    mid-kernel (torch 2.5.1, V100), and torch's ``graph.__exit__`` calls
    capture_end before popping the allocator's capture state — the failed
    capture leaves the caching allocator poisoned, so a later empty_cache in
    the same process INTERNAL-ASSERT-fails. There is no Python API to clear
    that state, so the doomed capture must not start. An explicit
    ``decode_graph=True`` is still honoured (informed opt-in for capture
    debugging)."""
    if decode_graph is not None:
        return decode_graph
    if backend.device.type != "cuda":
        return False
    if getattr(backend, "arch", "") == "sm70":
        global _sm70_graph_warned
        if not _sm70_graph_warned:
            warnings.warn(
                "decode graph capture auto-disabled on sm70: dense decode "
                "capture fails on this arch and a failed capture poisons "
                "torch's caching allocator for the process (a later "
                "empty_cache asserts). Running eager; pass decode_graph=True "
                "explicitly to attempt capture anyway.",
                stacklevel=2,
            )
            _sm70_graph_warned = True
        return False
    return True


#: Decode-graph size ladder: a tick pads up to the first bucket >= its row count.
_GRAPH_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

_PHASE_PREFILL = 1
_PHASE_DECODE = 2
_PHASE_DONE = 3

_HASH_MASK = 0x7FFFFFFF

#: Store stats deliberately not on the wire; the seam gate forbids any OTHER unforwarded key.
_STORE_STATS_INTERNAL = ("lookups_matched", "lookups_missed")


def _quantize_draft(
    params: dict[str, torch.Tensor], skip: tuple[str, ...] = (), fp4: bool = False
) -> dict[str, torch.Tensor]:
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
        _quantize_draft(draft.params, skip=draft.no_quant, fp4=not backend.has_kernel("linear_fp8"))
    )
    draft.params.clear()  # in place: the head's Model holds THIS dict
    draft.params.update(served)


def _step_seed(seed: int, generated: int) -> int:
    # Full-width hashes: a shift-then-mask collapsed seeds 1/2049/16385 to one stream.
    return ((int(seed) * 2_654_435_761) ^ (generated * 2_246_822_519)) & _HASH_MASK


class RequestFailed(RuntimeError):
    """A request ended in failure. ``reason`` is a stable tag for the failure
    class (None = untagged); a caller that tolerates one class catches on it.
    ``poll``/``take`` raise this instead of handing the failure back as data."""

    def __init__(self, request_id: int, reason: str | None, message: str):
        super().__init__(f"request {request_id} failed: {message}")
        self.request_id = request_id
        self.reason = reason


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = off; Qwen's generation_config ships top_k=20
    max_new_tokens: int = 16
    seed: int = 0
    stop_token_ids: tuple[int, ...] = ()
    #: text stop sequences; generation ends at the token that completes one. Needs
    #: ``Engine(decode=...)`` -- ``submit`` refuses these without it rather than
    #: accepting a stop that can never fire.
    stop_texts: tuple[str, ...] = ()
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


def _stop_hit(decode: Any, reply: list[int], stops: tuple[str, ...]) -> str | None:
    """The stop sequence the newest token just completed, or None.

    ``reply`` is the tokens a stop may match against -- the output PAST the reasoning
    block, never the reasoning itself: a stop like "\n\n" would otherwise fire on the
    first paragraph break inside <think> and end the request with a truncated thought
    and no answer at all.

    Decodes only the last ``k`` tokens, ``k`` = the longest stop in characters: a
    token carries at least one character, so k of them always cover a k-character
    match, and the per-token cost is one decode of a short id list instead of the
    whole output. Ties go to the earliest occurrence, which is where the caller cuts.
    """
    tail = decode(reply[-max(len(s) for s in stops) :])
    hits = [(tail.find(s), s) for s in stops if s in tail]
    return min(hits)[1] if hits else None


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
    state_slot: int | None  # None until `_admit` takes one
    seq_len: int  # == len(tokens); the logical materialized length
    phase: int  # _PHASE_PREFILL | _PHASE_DECODE | _PHASE_DONE
    prefill_from: int  # prefix-reuse offset for the prefill forward
    own_blocks: int  # blocks the engine allocated (vs adopted from a hit)
    #: this row's live decode-boundary entry length, retired when the next lands; 0 = none.
    decode_entry: int = 0
    #: interior prefill boundaries this row has already published. Only the first and the last
    #: land, so a row's publishes stay at 2 whatever the prompt length -- see `_finish_prefills`.
    interior_published: int = 0
    #: exact snapshot at the deepest aligned prefill chunk end inside the ragged tail
    #: window; inserted at completion. See `_finish_prefills`.
    pending_prefix: tuple[int, Any] | None = None
    #: A request failed mid-flight (e.g. cold spill): release its frames without
    #: trying to publish a prefix snapshot whose cold blobs may already be gone.
    failed: bool = False
    #: hybrid sparse engine: fixed at submit from prompt length vs --sparse-min-tokens.
    #: Dense rows pin their whole context and never touch the sparse tracker.
    sparse_on: bool = False
    output: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    thought_closed: bool = False  # the reasoning block ended (model's or forced)
    stop_text: str | None = None  # the stop sequence that ended it, for the caller to cut at
    reply_from: int = 0  # index in `output` past the reasoning closer; stops match only here on
    #: trunk hidden [1,w,H] at positions [hidden_from, hidden_from+w): the draft's fc input
    hidden: torch.Tensor | None = None
    hidden_prev: torch.Tensor | None = None  # [1,1,H] at hidden_from-1
    hidden_from: int = 0
    draft_pos: int = 0  # highest position whose draft KV belongs to a committed token
    drafts: list[int] = field(default_factory=list)  # next tick's chain, minus its first token
    #: sparse-KV demoted pages. Two flows, two shapes: the automatic path stores
    #: bare logical-page ints (the host blob is keyed (req_id, logical page), so
    #: no phys is carried); the #500 sparse_retier seam stores
    #: ``(logical page, phys)`` in ascending order. ``blocks`` holds the LIVE pages
    #: in sequence order either way; a promotion splices back at logical position.
    cold_pages: list = field(default_factory=list)
    #: sparse prefix hit: token length adopted from the shared index (0 = full miss)
    sparse_matched: int = 0
    #: sparse + spec: the DRAFT pool's dense block ids. The trunk tier demotes
    #: ``blocks`` every tick, but the draft head stays dense for the whole
    #: context, so under sparse it gets its own id space (empty in dense mode,
    #: where the draft table reuses ``blocks``).
    draft_blocks: list[int] = field(default_factory=list)
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


class _SparseDecodeGraph:
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
            keep_steps=int(W > 1),
            page_base=sf.page_base,
            sparse=sf,
        )
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
        B, W = self._b, self._w
        sf = self.sf
        sf.fill(srows)
        for i, rw in enumerate(srows):
            r = rw["req"]
            chain = chains[i] if chains else (r.output[-1],)
            self._ids[i, : len(chain)] = torch.tensor(chain, device=self._ids.device)
            self._pos[i, : len(chain)] = torch.arange(
                r.seq_len - 1, r.seq_len - 1 + len(chain), device=self._ids.device
            )
            self._sl[i] = r.seq_len - 1 + W
            self._slots[i] = r.state_slot
            self._sql[i] = len(chain)
        n = len(srows)
        if pad is not None and n < B:
            pad_slot, pad_block = pad
            for i in range(n, B):
                self._ids[i, :] = 0
                self._pos[i, :] = 0
                self._sl[i] = W
                self._slots[i] = pad_slot
                self._sql[i] = W
                sf.own_table[i, 0] = pad_block
        self._graph.replay()
        return self._logits


class _CpuSparseGraph:
    """CPU test seam for _SparseDecodeGraph: fills the same persistent
    SparseForward staging buffers and runs the model eagerly, so the
    selection/packed-table math the captured graph replays is exercised without a
    CUDA card. The tensors the selection reads are the captured-sf staging ones;
    a test asserts they are reused (stable data_ptr) and no host sync fires."""

    def __init__(self, model, backend, kv_pool, state_pool, sf, B, W):
        self.sf = sf
        self._model, self._backend, self._kv, self._states = (model, backend, kv_pool, state_pool)
        self._b, self._w = B, W
        self.hidden = None

    def run(self, srows, chains, pad=None):
        sf = self.sf
        sf.fill(srows)
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
            keep_steps=int(W > 1),
            page_base=sf.page_base,
            sparse=sf,
        )
        hid: list = []
        logits = self._model.forward(ids, pos, kv, self._backend, hidden_out=hid, last_only=False)
        self.hidden = hid[-1] if hid else None
        return logits


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
        decode: Any = None,
        sparse_tracker: Any = None,
        sparse_k: int = 0,
        boot_store: Any = None,
        sparse_device_select: bool = False,
        draft_num_blocks: int | None = None,
        sparse_min_tokens: int = 0,
        sparse_prefill_tokens: int = 0,
    ) -> None:
        self._model = model
        self._backend = backend
        # Text stop sequences need ids->str, and only here: a stop string does not
        # have to be a token boundary, so a tail-of-ids comparison would miss the
        # match whenever the model merged the last character into a wider token.
        # Still tokenizer-FREE by default -- `decode=None` disables `stop_texts`.
        self._decode = decode
        self._kv = kv_pool
        self._states = state_pool
        self._sparse = sparse_tracker
        self._sparse_k = sparse_k
        # Hybrid mode (sparse engine only): prompts longer than this go sparse;
        # shorter ones run dense on the captured graph and pin their whole context.
        # 0 = every request is sparse, the pre-hybrid behavior.
        self._sparse_min_tokens = sparse_min_tokens if sparse_tracker is not None else 0
        # Hybrid wall-time fairness. A ROLLING window: since the last sparse
        # tick, track wall time each mode actually received while BOTH had a
        # runnable row. Sparse owns the next tick only when dense has already
        # received at least as much wall time as sparse in that window. This
        # serves a newly-arrived dense row immediately (it arrives with dense
        # behind in the window) instead of making it pay for sparse ticks that
        # ran while no dense row existed -- that global-debt design gave ~0.08
        # dense/sparse tick ratio on the V100 (short min 1.5 tok/s, 29 s TTFT).
        self._hybrid_sparse_wall = 0.0  # sparse wall time in the current window
        self._hybrid_dense_wall = 0.0   # dense wall time in the current window
        #: test seam: (sparse_dt, dense_dt) replaces the perf_counter measurement
        self._hybrid_fake_dt: tuple[float, float] | None = None
        self._hybrid_t0 = 0.0
        self._dense_mode_ticks = 0
        self._sparse_mode_ticks = 0
        # A sparse prefill tick is capped so its wall time stays ~1 s on the
        # target card: V100 solo sparse prefill measured 65536 tokens / 343.2 s =
        # 191 tok/s (errors/2026-09-13-v100-256k), so 192 tokens ~= 1 s and is a
        # whole 3x64-token bucket above the forced 8-page (128-token) window.
        # Applied to sparse rows in HYBRID mode only (pure sparse is unchanged).
        self._sparse_prefill_cap = sparse_prefill_tokens or (
            192 if sparse_min_tokens else 0)
        #: decode ticks build the packed table with pure device selection (no host
        #: sync) — the capture-ready path; valid only with the pin steady state.
        self._sparse_device_select = sparse_device_select and sparse_tracker is not None
        #: decode ticks since the last eager full-candidate refresh; device ticks
        #: score only resident candidates, so every R-th decode tick goes eager to
        #: score ALL candidates and promote the ones the hot set is missing.
        self._sparse_ticks_since_refresh = 0
        self._sparse_graphs: dict = {}
        self._prefix = prefix_store
        #: KvBootStore for cold-start KV (--kv-store); None = no on-disk boot context.
        self._boot = boot_store
        self.limits = limits

        self._decode_graph_on = _graph_on(backend, decode_graph)
        self._decode_graphs: dict = {}
        # Sparse decode ticks use a separate capture (packed [selected;own] table,
        # not the dense table). On when sparse device selection is enabled; on a
        # CUDA backend that also requires the decode graph on (sm70 excluded by
        # _graph_on), on CPU the device-select path runs the eager CPU test seam.
        # Hybrid mode runs sparse ticks EAGER on purpose: it needs only the dense
        # precaptured graph, so the sparse capture (and its warmup-frame hazard)
        # is not required; eager sparse is token-exact on sm70.
        self._sparse_graph_on = (
            sparse_tracker is not None
            and not self._sparse_min_tokens
            and self._sparse_device_select
            and (self._decode_graph_on or backend.device.type != "cuda")
        )
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
        # is the real concurrency ceiling, and _build_plan's max_batch is unreachable.
        # The excess QUEUES: `submit` has no slot check and `_admit` returns False on
        # `free_slots < 1`, so a B=8 submit into 4 usable slots runs two waves of 4 --
        # no raise, no drop, a table with twice the ticks and half the rows per tick.
        # Warn rather than clamp, because a test that submits two rows into a 2-slot
        # pool with the default max_batch=8 is a legitimate config, not a mistake.
        if self.usable_slots < limits.max_batch:
            # The remedy names `num_slots`, the parameter the reader passes. Naming the
            # pool instead is what made the old "+ 1 for the pad row" get applied to a
            # build_engine call that already adds it -- the misread this message caused.
            remedy = f"pass num_slots >= {limits.max_batch} to build_engine"
            if self._pad_slot is not None:
                remedy += " (it adds the decode graph's pad row itself)"
            warnings.warn(
                f"{self.usable_slots} usable state slots against max_batch="
                f"{limits.max_batch}: a slot is held from submit to finish, so "
                f"concurrency is capped at {self.usable_slots} and the excess queues "
                f"into later ticks rather than raising -- twice the ticks at half the "
                f"width, not an error. To run {limits.max_batch} rows at once, "
                f"{remedy}, or size a LinearStatePool for "
                f"{limits.max_batch + (self._pad_slot is not None)} directly "
                f"(this one holds {self._states.num_slots})",
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
                raise ValueError(f"verify width must be in (1, {BLOCK_TOKENS}], got {self._width}")
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
            # .dtype, not k_pool.dtype: under kv_fp8 the latter is the fp8 store dtype, and
            # the draft pool has no scale plane, so it would hold a scale-less cast.
            # Sparse: the draft stays dense, so its pool spans a whole context per slot
            # (draft_num_blocks), not the sparse hot pool.
            draft.attach(
                backend,
                draft_num_blocks if draft_num_blocks is not None else kv_pool.num_blocks,
                dtype=kv_pool.dtype,
            )

        self._pin = backend.device.type == "cuda"
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        #: Published by the loop so `stats()` never takes the lock a forward holds.
        self._stats_snapshot: dict[str, Any] | None = None
        #: Memoized dense ledger: weights/pools/slots are static after build, and
        #: _build_stats (twice a step) used to re-walk every param tensor per tick.
        #: Keyed on len(params): add_lora attaches adapter tensors post-build, so
        #: the train manifest's ledger must recompute after an attach.
        self._mem_rows: tuple[int, list] | None = None

        self._next_id = 1
        self._waiting: deque[_Req] = deque()
        self._running: list[_Req] = []
        self._finished: dict[int, list[int]] = {}
        #: rid -> the stop sequence that ended it. Not popped with the tokens: the
        #: routes read it after `take`, and only a matched request has an entry.
        self._finished_stop: dict[int, str] = {}
        self._failed: dict[int, tuple[str | None, str]] = {}
        self._finished_count = 0

        self._blocks_used = 0  # engine allocations outstanding (retains excluded)
        self._slots_used = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        #: spec followers that adopted a WARM entry (draft K/V + boundary hidden),
        #: distinct from _prefix_hits so "0 warm adoptions" is observable, not
        #: indistinguishable from the path never running.
        self._prefix_warm_adoptions = 0
        #: cold-start KV boot store: contexts loaded from --kv-store, and load calls.
        self._boot_hits = 0
        # Matched tokens, not just hit count: a hit that matches 512 of 30826 is a miss wearing a
        # hit's label, and the count alone cannot tell the two apart (2026-09-08, #271's 2.03x).
        self._prefix_hit_tokens = 0
        self._prefix_published = 0
        self._prefill_forwards = 0
        self._prefill_tokens = 0
        self._prefill_secs = 0.0
        self._seed_rate = 2558.6 if getattr(backend, "arch", "") == "sm90" else 75.0
        self._decode_forwards = 0
        self._mixed_forwards = 0
        self._tokens_generated = 0
        self._spec_drafted = 0
        self._spec_accepted = 0
        # Per-segment spec counters for the forced-think acceptance split:
        # drafts inside the reasoning block vs after it closed, plus ticks that
        # cross the max_think cap (the forced closer makes that chain stale).
        self._spec_acc_in = self._spec_dft_in = 0
        self._spec_acc_post = self._spec_dft_post = 0
        self._spec_acc_capcross = self._spec_dft_capcross = 0
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
    def _logical_capacity_blocks(self) -> int:
        """Admission capacity in logical pages. Dense: device blocks. Sparse: the
        device hot pool plus the pages the host cold tier can hold — older pages
        demote there, so a request far longer than the device pool still admits.
        Under sparse+spec the draft head stays DENSE in its own on-device pool, so
        that pool is a tighter bind than hot+cold and wins (the device draft pool
        cannot spill to host)."""
        cap = self.usable_blocks
        if self._sparse is not None:
            cap += self._kv.cold_capacity_blocks()
            if self._draft is not None:
                cap = min(cap, self._draft.kv.num_blocks)
        return cap

    @property
    def usable_slots(self) -> int:
        return self._states.num_slots - (self._pad_slot is not None)

    @property
    def config(self) -> dict[str, Any]:
        """The fields a wall-clock number cannot be compared across runs without.

        Read off the built engine rather than the call's kwargs: `num_blocks` is
        clamped by `max_blocks` and both pools carry the graph's pad row, so the
        argument and the pool disagree. Six card sessions on the 2.6x rollout tick
        recovered their two pool sizes only because the probe logged its own flags
        (errors/2026-09-08-six-card-sessions-and-the-defect-did-not-move.md).

        ``memory`` is memory_table's rows for THIS build, the same table serve
        --dry-run prints and stats()["memory"] serves, so a run manifest and a
        dry-run report the identical occupancy surface (P5 reads it for free).
        """
        return {
            "blocks": self.usable_blocks,
            "slots": self.usable_slots,
            "max_batch": self.limits.max_batch,
            "max_total_tokens": self.limits.max_total_tokens,
            "max_num_batched_tokens": self.limits.max_num_batched_tokens,
            "decode_graph": self._decode_graph_on,
            "prefix_store": type(self._prefix).__name__,
            "spec_width": self._width,
            "memory": self._memory_rows(),
        }

    def room_for(self, prompt_tokens: int) -> int:
        """Largest ``max_new_tokens`` this prompt can ask for and still be admitted.

        The two ceilings `submit` enforces, so a caller that wants "as much as fits"
        does not re-derive them: ``max_total_tokens``, and the KV pool including the
        ``width - 1`` drafts a verify tick materializes past the last token. Returns 0
        when the prompt alone does not fit -- the caller keeps its own refusal, since
        `submit` refuses that case with a message naming which bound it hit.
        """
        by_total = self.limits.max_total_tokens - prompt_tokens
        by_pool = BLOCK_TOKENS * self._logical_capacity_blocks - prompt_tokens - self._width + 1
        return max(0, min(by_total, by_pool))

    def submit(self, input_ids: Any, params: SamplingParams | None = None) -> int:
        """Queue a request; returns its opaque id. Blocks and the state slot are taken at
        admission, not here."""
        if params is None:
            params = SamplingParams()
        tokens = [int(t) for t in input_ids]
        if not tokens:
            raise ValueError("prompt must be non-empty")
        if params.stop_texts and self._decode is None:
            raise ValueError(
                "stop_texts needs Engine(decode=tokenizer.decode): matching happens on "
                "decoded text, and accepting the field without it is a stop that can "
                "never fire"
            )
        if any(not s for s in params.stop_texts):
            # "" is in every string, so it would end the request at token 1.
            raise ValueError("stop_texts entries must be non-empty")
        if params.max_new_tokens > 0:
            total = len(tokens) + params.max_new_tokens
            if total > self.limits.max_total_tokens:
                raise ValueError(
                    f"request ({total} tokens) exceeds max_total_tokens "
                    f"({self.limits.max_total_tokens})"
                )
            # +depth: a verify tick materializes the drafts past the last token
            if self._kv.blocks_for_tokens(total + self._width - 1) > self._logical_capacity_blocks:
                raise ValueError(f"request ({total} tokens) exceeds KV pool capacity")
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            if params.max_new_tokens <= 0:
                self._finished[rid] = []
                self._finished_count += 1
                return rid

            # Unallocated: allocating here refuses permanently, since `submit` has no later
            # tick to retry on. The prefix match moves to `_admit` with the allocation.
            # Hybrid sparse engine: short prompts run dense and pin their whole context
            # (no sparse sharing); a prompt that could never fit that pin even with the
            # pool empty routes sparse instead of queueing on an impossible admit.
            sparse_on = (
                self._sparse is not None
                and (self._sparse_min_tokens == 0
                     or len(tokens) > self._sparse_min_tokens)
            )
            if (not sparse_on and self._sparse is not None and params.max_new_tokens > 0
                    and self._kv.blocks_for_tokens(
                        total + self._width - 1)
                    > self._kv.num_blocks - (self._pad_block is not None)):
                # A dense row cannot use the sparse cold tier: its pin has to fit
                # the DEVICE pool (usable_blocks counts cold for sparse rows), so a
                # prompt that would head-of-line block on a permanent _admit False
                # routes sparse instead.
                sparse_on = True
            req = _Req(
                req_id=rid,
                params=params,
                tokens=tokens,
                blocks=[],
                state_slot=None,
                seq_len=0,
                phase=_PHASE_PREFILL,
                prefill_from=0,
                own_blocks=0,
                sparse_on=sparse_on,
            )
            self._waiting.append(req)
        return rid

    @property
    def prefill_rate(self) -> float:
        """Prefill tokens/s: this engine's own measurement, or the arch seed before one."""
        if self._prefill_secs <= 0:
            return self._seed_rate
        return self._prefill_tokens / self._prefill_secs

    def poll(self) -> dict[int, list[int]]:
        """Return and clear all requests finished since the last poll."""
        with self._lock:
            if self._failed:
                rid, (reason, message) = self._failed.popitem()
                raise RequestFailed(rid, reason, message)
            out = dict(self._finished)
            self._finished.clear()
            return out

    def stop_text(self, request_id: int) -> str | None:
        """The stop sequence that ended this request, or None if none did. Pops, so
        the routes' `_finished_stop` entries do not outlive the run."""
        with self._lock:
            return self._finished_stop.pop(request_id, None)

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
            failed = self._failed.pop(request_id, None)
            if failed is not None:
                reason, message = failed
                raise RequestFailed(request_id, reason, message)
            return self._finished.pop(request_id, None)

    def step(self) -> None:
        """Run one tick: one forward over the planned rows."""
        idle = False
        with self._lock:
            decodes, prefills, chunks = self._build_plan()
            if not decodes and not prefills:
                idle = True
            else:
                # Before the forward too: without this the FIRST forward has no snapshot and
                # `stats()` falls back to the locking path.
                self._stats_snapshot = self._build_stats()
                tick_sparse = bool(
                    (decodes + prefills) and (decodes + prefills)[0].sparse_on)
                try:
                    self._hybrid_t0 = time.perf_counter()
                    self._run_forward(decodes, prefills, chunks)
                except Exception as exc:
                    for req in list(self._running):
                        self._finish(req, error=str(exc))
                    raise
                finally:
                    # `_loop` stops calling `step` once nothing runs, so this carries the last
                    # tick's state -- including a failed forward's, hence `finally`.
                    self._stats_snapshot = self._build_stats()
                self._hybrid_charge(tick_sparse)
        if idle:
            return

    def _admit(self, req: _Req) -> bool:
        """Take the slot and the blocks for one waiting request. False = it does not fit yet."""
        # Hybrid: a dense row shares the sparse engine's pools but pins its whole
        # context in the dense KV pool and never registers with the sparse tracker.
        sparse = req.sparse_on
        # Slot checked before any block allocation: a bulk boot load allocates all
        # its blocks up front, so admitting with no free slot would have to roll them back.
        if self._states.free_slots < 1:
            return False
        total_blocks = (len(req.tokens) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        # Sparse rows share through the tracker's SparsePrefixCache, never the
        # dense block-retaining store; skip the lookup entirely.
        matched, hit_blocks, snap = (
            (0, (), None) if sparse else self._match_prefix(req.tokens))
        boot_len = 0
        # A bulk boot allocates EVERY context block against the device pool up
        # front; the sparse hot pool holds only k+window+chunk per slot and grows
        # pages lazily, so that combination is refused at build time and the boot
        # lookup is skipped here for a sparse build as well.
        if not sparse and self._boot is not None:
            boot_len = (len(req.tokens) // BLOCK_TOKENS) * BLOCK_TOKENS
            if boot_len == 0 or not self._boot.exists(req.tokens[:boot_len]):
                boot_len = 0
        # Sparse pre-allocates NOTHING: needed 0. A boot hit needs the whole
        # request; a prefix hit only the residual past the store's blocks.
        # By count, not by catching `alloc_slot`'s raise: an exception out of
        # `_admit` reaches `step`'s handler, which fails EVERY running request.
        needed = 0 if sparse else (total_blocks if boot_len else total_blocks - len(hit_blocks))
        # Hybrid: a dense admit pins its whole context from the ONE pool sparse
        # rows grow into lazily. Their future own pages are not allocated yet, so
        # free_blocks overstates what the dense row may take; leave every live
        # sparse row's headroom to its hot ceiling, or that row later raises
        # "hot pool undersized" inside a tick and fails every running request.
        needed += self._sparse_hot_headroom() if not sparse and self._sparse is not None else 0
        if self._kv.free_blocks < needed:
            # Guarded: unguarded, a request waiting on a live retain would drop every entry
            # each tick and free nothing, flushing other clients' prefixes for the whole wait.
            if self._kv.free_blocks + self._prefix.reclaimable_blocks() < needed:
                return False
            self._prefix.evict_until_free(needed)
            if not boot_len:
                # Re-matched: eviction may have dropped the entry this hit came from.
                matched, hit_blocks, snap = self._match_prefix(req.tokens)
                needed = total_blocks - len(hit_blocks)
            if self._kv.free_blocks < needed:
                return False
        boot_state: Any = None
        boot_loaded = False
        if boot_len:
            loaded = self._boot.boot(req.tokens[:boot_len], self._kv)
            if loaded is None:
                return False  # corrupt/vanished between exists() and load: admit as miss next tick
            hit_blocks, boot_state = loaded["blocks"], loaded["state"]
            matched, boot_loaded = loaded["length"], True
        if matched:
            if boot_loaded:
                self._boot_hits += 1
            else:
                self._prefix_hits += 1
                self._prefix_hit_tokens += matched
        else:
            self._prefix_misses += 1
        slot = self._states.alloc_slot()
        # Sparse + spec: the draft stays dense, so its own pool must hold this
        # row's whole context. Two checks happen before any admitted state is
        # committed (returning False after the counters were bumped leaked one
        # slot per retry tick). Reserve the FULL draft span (prompt + max_new +
        # verify width-1, the same bound submit's static guard uses) at admit so
        # the shared draft pool accounts across rows: a prompt-only guard lets two
        # rows both pass and then alloc_block raises inside the forward.
        if sparse and self._draft is not None:
            draft_need = self._kv.blocks_for_tokens(
                len(req.tokens) + req.params.max_new_tokens + self._width - 1
            )
            if self._draft.kv.free_blocks < draft_need:
                self._states.free_slot(slot)
                return False
            req.draft_blocks = [self._draft.kv.alloc_block() for _ in range(draft_need)]
        # Sparse: blocks grow lazily per own span, never pre-allocate the whole
        # context; prefix-block reuse is likewise skipped (cold pages live in this
        # request's own host tier, not in the shared store).
        blocks: list[int] = [] if sparse else (list(hit_blocks) if boot_loaded else [])
        try:
            if not sparse and not boot_loaded:
                for b in hit_blocks:
                    self._kv.retain(b)  # adopt the store's blocks
                    blocks.append(b)
            target = 0 if sparse else total_blocks
            while len(blocks) < target:
                blocks.append(self._kv.alloc_block())
        except Exception:
            for b in blocks:
                self._kv.free_block(b)
            if sparse and self._draft is not None:
                for b in req.draft_blocks:
                    self._draft.kv.free_block(b)
                req.draft_blocks = []
            self._states.free_slot(slot)
            raise
        req.blocks = blocks
        req.state_slot = slot
        if sparse and self._draft is not None:
            # A dense match cannot serve a draft row (no draft KV saved); start at
            # zero and let the sparse prefix lookup below adopt a WARM entry (one
            # carrying draft K/V + boundary hidden) when it has one.
            matched = 0
        req.seq_len = matched  # materialized length (adopted prefix; 0 on a miss)
        # A bulk boot covering the WHOLE prompt leaves a zero-token residual, so no prefill
        # chunk would forward and the row stuck in PREFILL with no first-token logits (a
        # page-aligned prompt, T % BLOCK_TOKENS == 0, including 262144). Re-forward the last
        # loaded OWN page once: boot blocks are this row's fresh allocations (unlike a
        # prefix hit's shared store blocks, which must not be re-written), so overwriting the
        # last page's K/V with the same values is safe and the tail logits appear — one page
        # at cold start, the same cost as any <16-token tail.
        req.prefill_from = (
            max(0, matched - BLOCK_TOKENS)
            if boot_loaded and matched == len(req.tokens)
            else matched
        )
        if sparse:
            req.own_blocks = 0
        else:
            # A boot load is all own blocks; a prefix hit's blocks belong to the store.
            req.own_blocks = total_blocks if boot_loaded else total_blocks - matched // BLOCK_TOKENS
        self._blocks_used += req.own_blocks
        self._slots_used += 1
        if sparse:
            self._sparse.attach(req.req_id)
            if self._sparse.prefix is not None:
                self._sparse.prefix.set_request(req.req_id, len(req.tokens) // BLOCK_TOKENS)
            # Sparse prefix hit: adopt bounds + GDN snapshot + page content keys
            # WITHOUT blocks — the pages live as shared host blobs and promote
            # lazily on selection (_sparse_resolve). seq_len/prefill_from already
            # carry the matched length, so the engine prefills only the tail.
            #
            # Disabled under spec: a follower adopts the trunk prefix WITHOUT
            # forwarding it, but the draft head conditions every position on the
            # trunk hidden and builds its OWN dense KV only while forwarding — so
            # an adopted follower's draft attends over an unbuilt prefix and its
            # proposals are garbage. Warming it needs the trunk hidden at every
            # prefix position (the forward the prefix save skips) or storing those
            # hiddens; neither is cheap. Return-miss instead (never raise): the
            # follower prefills from zero, which builds trunk and draft KV correctly.
            # Spec adoption: a follower can adopt only a WARM entry whose blobs
            # carry the draft head's K/V and whose boundary snapshot carries the
            # trunk hidden at matched-1 (the first tail draft conditions on it);
            # an old/trunk-only entry is a miss (prefill from zero).
            entry = self._sparse.prefix.lookup(req.tokens) if self._sparse.prefix else None
            if entry is not None and self._draft is not None and (
                    entry.get("hidden") is None or any(
                        self._kv.cold.share_take_field(k, "dk") is None
                        for k in entry["keys"])):
                # field probes avoid pinning the whole trunk blob to check warmness
                entry = None
            if entry is not None:
                matched = len(entry["tokens"])
                req.seq_len = matched
                if (matched == len(req.tokens)
                        and matched % BLOCK_TOKENS == 0):
                    # A follower whose prompt matches a page-aligned prefix in
                    # WHOLE has zero residual tokens, so no chunk would forward
                    # and the row stuck in PREFILL with no first-token logits.
                    # Re-forward the last adopted page (its promote is a fresh
                    # private copy): the boundary hidden conditions the next
                    # step and logits at matched-1 appear. Runs with or without
                    # a draft — the no-draft follower needs those logits too.
                    req.prefill_from = matched - BLOCK_TOKENS
                else:
                    req.prefill_from = matched
                keys = list(entry["keys"])
                self._sparse.shared[req.req_id] = dict(enumerate(keys))
                if self._sparse.scorer == "bounds":
                    # bounds are read by field out of each (possibly spilled)
                    # shared blob, not pinned in the entry
                    for p, key in enumerate(keys):
                        b = self._sparse.prefix.bound_of_key(key)
                        if b is not None:
                            self._sparse.set_bounds(req.req_id, p, b)
                snap_states, snap_windows = entry["state"]
                self._states.states[slot].copy_(snap_states)
                if snap_windows is not None:
                    self._states.window_restore(slot, snap_windows)
                if self._draft is not None:
                    self._sparse_warm_draft(req, entry, matched)
                self._prefix_hits += 1
                req.sparse_matched = matched
        if matched and not sparse:
            if boot_loaded:
                self._states.states[slot].copy_(boot_state["states"].to(self._states.states.device))
                if boot_state["window"] is not None:
                    self._states.window_restore(slot, boot_state["window"])
                self._states.win_parity[slot] = boot_state["parity"]
            elif snap is not None:
                snap_states, snap_windows = snap
                self._states.states[slot].copy_(snap_states)
                if snap_windows is not None:
                    self._states.window_restore(slot, snap_windows)
        return True

    def _build_plan(self) -> tuple[list[_Req], list[_Req], list[int]]:
        """Admit the whole waiting queue up to ``max_batch``, then all running
        decodes plus as many prefill rows as the token budget and one width
        bucket allow; a longer prompt stays in PREFILL and chunks across ticks."""
        while self._waiting and len(self._running) < self.limits.max_batch:
            head = self._waiting[0]
            # break, not continue: head-of-line FIFO, else a blocked large request starves.
            if not self._admit(head):
                break
            self._running.append(self._waiting.popleft())
        decodes = [r for r in self._running if r.phase == _PHASE_DECODE]
        # Hybrid wall-time fairness: while both modes have runnable rows, sparse
        # owns a tick only when dense has spent at least as much wall time since
        # the last sparse tick as that sparse tick cost; otherwise dense owns the
        # tick. Every tick stays one mode (one BatchKv geometry).
        # ponytail: this shares the device QUEUE, not just the scheduler — a dense
        # decode tick during a long sparse prefill still syncs behind the in-flight
        # prefill kernel, so a concurrent short request measures ~9.9 tok/s vs
        # 52.6 solo (wins/2026-09-14-hybrid-*). Decoupling needs a separate fill
        # queue/stream or finer chunk interleaving; fairness ticks alone don't fix it.
        mode_sparse = self._sparse is not None
        if self._sparse is not None:
            dense_rows = [r for r in self._running if not r.sparse_on]
            sparse_rows = [r for r in self._running if r.sparse_on]
            if dense_rows and sparse_rows:
                # dense owns the tick while it is behind OR TIED: a tie at a
                # freshly-opened window (dense just arrived, both at 0) must serve
                # the dense row, not make it wait one ~1 s sparse tick.
                mode_sparse = self._hybrid_dense_wall > self._hybrid_sparse_wall
            elif dense_rows:
                mode_sparse = False
            decodes = [r for r in decodes if r.sparse_on == mode_sparse]
        prefills: list[_Req] = []
        chunks: list[int] = []
        budget = self.limits.max_num_batched_tokens - len(decodes)
        if mode_sparse and self._sparse_prefill_cap:
            budget = min(budget, self._sparse_prefill_cap)
        bucket = 0
        for r in self._running:
            if r.phase != _PHASE_PREFILL:
                continue
            if self._sparse is not None and r.sparse_on != mode_sparse:
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
            # Give up a block NOW when this chunk would leave a 1-token remainder, because
            # the back-off below only fires on the chunk that carries the tail. At n=65 the
            # 64-alignment above lands exactly on 64, so the tail is a chunk of its own:
            # `end == n` holds with `short = 64 - 64 = 0`, `short > 0` is False, and the
            # 1-token chunk ships. 14 lengths under 4000 hit this (65, 129, ... 2561, one per
            # `budget × k + 1` and per `_PREFILL_BUCKET × k + 1`). Costs no extra
            # forward: measured over n=2..4000 at budget 512, 21606 chunks before and after.
            # Schedules below BLOCK_TOKENS still ship the 1-token tail -- their deepest
            # aligned boundary is held and published at completion instead.
            # errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md
            if len(r.tokens) - (r.prefill_from + chunk) == 1 and chunk > BLOCK_TOKENS:
                chunk -= BLOCK_TOKENS
            # Cut the last chunk to a block boundary so the prompt-only publish lands at a
            # real chunk end; slicing an entry below its state snapshot is wrong.
            end = r.prefill_from + chunk
            tail = end % BLOCK_TOKENS
            short = (end // BLOCK_TOKENS) * BLOCK_TOKENS - r.prefill_from
            if end == len(r.tokens) and tail and short > 0:
                # A 1-token tail reaches the kernels with a zero block size, so back off.
                if tail == 1:
                    short -= BLOCK_TOKENS
                if short > 0:
                    chunk = short  # the <=17-token tail becomes one more forward
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
        if self._sparse is not None and self._sparse.prefix is not None:
            self._sparse.prefix.clear()  # release shared prefix blobs to the cold tier

    def stats(self) -> dict[str, Any]:
        """Lock-free while the loop thread runs; a fresh build when it does not."""
        if self._thread is None:
            # Direct-drive: no background forward to wait on, and a caller reading between
            # its own `step()` calls would get the previous tick's counters.
            return self._build_stats()
        snap = self._stats_snapshot
        if snap is None:
            # Loop started, first tick not finished yet.
            return self._build_stats()
        return snap

    def _build_stats(self) -> dict[str, Any]:
        with self._lock:
            store = self._prefix.stats()
            return {
                "waiting": len(self._waiting),
                "running": len(self._running),
                "finished": self._finished_count,
                # Runtime state, not the build setting: a failed capture flips this off and
                # /health reads stats(), so a silent eager fallback stays invisible without it.
                "decode_graph": self._decode_graph_on,
                "blocks_used": self._blocks_used,
                "blocks_total": self.usable_blocks,
                "pool_used_blocks": self._kv.used_blocks,
                "slots_used": self._slots_used,
                "slots_total": self.usable_slots,
                "prefix_hits": self._prefix_hits,
                "prefix_misses": self._prefix_misses,
                "prefix_warm_adoptions": self._prefix_warm_adoptions,
                "prefix_hit_tokens": self._prefix_hit_tokens,
                # measured prefill tokens/s for this engine (arch seed before one)
                "prefill_rate": round(self.prefill_rate, 1),
                "prefix_published": self._prefix_published,
                # Whether the store is under pressure at all: the DRAM snapshot
                # tier below it can only recover entries that were actually
                # evicted, and at 144 MiB a 27B snapshot the sm70 budget
                # (free/4 = 1417 MiB) holds 9 of them.
                "prefix_evictions": store["evictions"],
                "prefix_superseded": store["superseded"],
                # Indexed, not .get(k, 0): a default turns a store that stopped publishing the
                # key into a healthy-looking 0. Both stores publish these three.
                "prefix_blocks_freed": store["blocks_freed"],
                "prefix_entries": store["entries"],
                "prefix_capacity": store["capacity"],
                # `capacity` is the count cap; the byte budget is what binds.
                "prefix_entries_capacity": store["entries_capacity"],
                "prefix_state_bytes": store["state_bytes"],
                "prefix_state_bytes_budget": store.get("state_bytes_budget", 0),
                # Present only with a host tier; a demotion is a prefix the card could not
                # keep but did not have to lose.
                **{k: v for k, v in store.items() if k.startswith(("dram_", "ssd_"))},
                # sparse-KV cold page tier (absent when kv_cold_bytes=0)
                **(self._kv.cold.stats() if getattr(self._kv, "cold", None) is not None else {}),
                "prefix_demoted": store.get("demoted", 0),
                # cold-start KV boots from --kv-store (not a prefix-cache hit)
                "boot_hits": self._boot_hits,
                "prefill_forwards": self._prefill_forwards,
                "decode_forwards": self._decode_forwards,
                "mixed_forwards": self._mixed_forwards,
                # hybrid --sparse-min-tokens: ticks each mode ran (0 when off)
                "dense_mode_ticks": self._dense_mode_ticks,
                "sparse_mode_ticks": self._sparse_mode_ticks,
                # hybrid: live sparse residency stays visible even though the
                # memory ledger reconciles the dense view (rev-30 item 4).
                **(self._sparse_live_stats()
                  if self._sparse is not None and self._sparse_min_tokens else {}),
                "tokens_generated": self._tokens_generated,
                "spec_drafted": self._spec_drafted,
                "spec_accepted": self._spec_accepted,
                "spec_accept_in": self._spec_acc_in,
                "spec_drafted_in": self._spec_dft_in,
                "spec_accept_post": self._spec_acc_post,
                "spec_drafted_post": self._spec_dft_post,
                "spec_accept_capcross": self._spec_acc_capcross,
                "spec_drafted_capcross": self._spec_dft_capcross,
                "memory": self._memory_rows(),
            }

    def _held_storage(self) -> dict[str, int]:
        """Measured bytes of each static owner from materialized tensor storage. This is
        both the ledger's measured column and the CPU peak (off cuda there is no torch
        allocator high-water). Pool sums include both fp8 scale grids: dropping them
        makes derived != measured on the fp8 27B path by two planes."""
        kv, sp, draft = self._kv, self._states, getattr(self._draft, "kv", None)

        def pool(p) -> int:
            return sum(
                t.numel() * t.element_size()
                for t in (p.k_pool, p.v_pool, p.k_scale, p.v_scale)
                if t is not None
            )

        measured = {
            "weights": sum(t.numel() * t.element_size() for t in self._model.params.values()),
            "kv_pool": pool(kv),
            "state_slots": sum(
                t.numel() * t.element_size()
                for t in (
                    sp.states,
                    sp.conv_windows,
                    sp.step_states,
                    sp.step_windows,
                    sp.win_parity,
                )
                if t is not None
            ),
        }
        # ponytail: hybrid mode reconciles the DENSE ledger (the pool is sized for
        # dense rows pinning their whole context); sparse-only measured owners
        # (kv_hot/bounds/cold) are omitted. The pool is one container for both
        # modes, so the sparse per-row breakdown cannot reconcile against it.
        if self._sparse is not None and self._sparse_min_tokens == 0:
            # Sparse: held owners are the bounds tensors, the cross-tick pinned hot
            # pages still resident after finalize, and the host cold tier. kv_hot
            # counts the live blocks (the tick's selected set), not the pool total.
            from .memory import per_kv_block_bytes

            block_n = per_kv_block_bytes(self._model.cfg, kv.dtype, kv.kv_fp8)
            hot = sum(len(self._sparse.resident.get(r.req_id, ())) for r in self._running) * block_n
            measured.pop("kv_pool")
            measured["index_keys" if self._sparse.scorer == "index" else "page_bounds"] = (
                self._sparse.bounds_bytes()
            )
            measured["kv_hot"] = hot
            if getattr(kv, "cold", None) is not None:
                shared_n = kv.cold.shared_bytes()
                if kv.cold.bytes_held - shared_n:
                    measured["kv_cold"] = kv.cold.bytes_held - shared_n
                if shared_n:
                    measured["kv_prefix"] = shared_n
            graph_n = self._sparse_graph_bytes()
            if graph_n:
                measured["sparse_graph"] = graph_n
        if draft is not None:
            measured["draft_pool"] = pool(draft)
        return measured

    def _sparse_graph_bytes(self) -> int:
        """Held device bytes of every captured sparse graph's persistent forward
        (the fixed staging tables + packed table the graph bakes). Built lazily,
        so this is 0 until the first steady-state decode tick."""
        n = 0
        for g in self._sparse_graphs.values():
            sf = g.sf
            for t in (
                sf.table,
                sf.seq_len,
                sf.page_base,
                sf.own_table,
                sf.cand_idx,
                sf.own_log,
                sf.own_valid,
                sf.n_cand,
                sf.win,
                sf.own_len_t,
                sf.s_l2p,
            ):
                n += t.numel() * t.element_size()
            for t in getattr(sf, "s_bounds", None) or ():
                n += t.numel() * t.element_size()
        return n

    def _measured_peak_bytes(self) -> int | None:
        """The resident-device byte peak the table closes against. On cuda this is the
        torch allocator's high-water mark (reset at build); off cuda the held-storage
        sum — which equals Σ static there and keeps transient exactly zero."""
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(self._backend.device))
        return sum(self._held_storage().values())

    def _memory_rows(self) -> list[dict]:
        """The unified device ledger (memory.memory_table): every held allocation plus
        the transient residual, so ``measured peak = Σ static + transient`` is printed."""
        from .memory import memory_table, plan

        n_params = len(self._model.params)
        if self._mem_rows is not None and self._mem_rows[0] == n_params:
            return self._mem_rows[1]
        kv, sp = self._kv, self._states
        # Only the MTP DraftHead attaches a separate PagedKvPool; DFlash2 has no .kv pool.
        draft_pool = getattr(self._draft, "kv", None)
        draft_layers = 0 if draft_pool is None else self._draft.cfg.num_layers
        # device_free=0: budget rows are the dry-run path's business; the fit is done here.
        if self._sparse is None or self._sparse_min_tokens:
            derived = plan(
                self._model.cfg,
                self._model.params,
                0,
                num_slots=sp.num_slots,
                num_blocks=kv.num_blocks,
                spec_steps=0,
                state_dtype=sp.states.dtype,
                kv_io=kv.dtype,
                kv_fp8=kv.kv_fp8,
                draft_layers=draft_layers,
            )
        else:
            # Live sparse ledger from ACTUAL per-row state (post-tick every private page is
            # demoted, so kv_hot counts the blocks currently in r.blocks): bounds are count-
            # derived per complete page, hot per resident block, cold from the host tier.
            from .memory import Row, index_keys_bytes, per_kv_block_bytes

            derived = plan(
                self._model.cfg,
                self._model.params,
                0,
                num_slots=sp.num_slots,
                num_blocks=kv.num_blocks,
                spec_steps=0,
                state_dtype=sp.states.dtype,
                kv_io=kv.dtype,
                kv_fp8=kv.kv_fp8,
                draft_layers=0,
                sparse=None,
            )
            derived = [r for r in derived if r.owner != "kv_pool"]
            cfg = self._model.cfg
            block_n = per_kv_block_bytes(cfg, kv.dtype, kv.kv_fp8)
            hot_n = cold_n = 0
            # Sparse draft pool is dense-sized (a whole context per slot), unlike the
            # hot trunk pool, so its row is computed from the draft pool's own blocks.
            if draft_pool is not None:
                from .memory import draft_per_block_bytes

                derived.append(
                    Row(
                        "device",
                        "draft_pool",
                        draft_per_block_bytes(cfg, kv.dtype, draft_layers) * draft_pool.num_blocks,
                        f"{draft_layers} draft layers x {draft_pool.num_blocks} dense blocks",
                    )
                )
            pages_total = 0
            for r in self._running:
                complete = r.seq_len // BLOCK_TOKENS
                pages_total += complete
                hot_n += len(r.blocks) * block_n
            if self._sparse.scorer == "index":
                scorer_n = index_keys_bytes(cfg, pages_total, self._sparse.di)
                owner = "index_keys"
                note = f"{pages_total} complete pages, learned index scorer"
            else:
                # Actual stored bounds, not complete-page count: the newest
                # complete page's bound is written in finalize, so a stats
                # snapshot between forwards sees one fewer bound than seq_len//16.
                # measured page_bounds already counts actual (bounds_bytes()).
                scorer_n = self._sparse.bounds_bytes()
                owner = "page_bounds"
                note = f"{pages_total} complete pages, bounds scorer"
            cold_n = ssd_n = ssd_cap = prefix_n = 0
            if getattr(kv, "cold", None) is not None:
                prefix_n = kv.cold.shared_bytes()
                cold_n = kv.cold.bytes_held - prefix_n
                ssd_n = kv.cold.ssd_bytes
                ssd_cap = kv.cold.ssd_capacity_bytes
            derived.append(Row("device", owner, scorer_n, note))
            derived.append(Row("device", "kv_hot", hot_n, "resident private blocks this tick"))
            if cold_n:
                derived.append(Row("host", "kv_cold", cold_n, "demoted pages in pinned host RAM"))
            if prefix_n:
                # shared prefix blobs: trunk K/V + bounds (+ draft K/V under warm spec)
                derived.append(Row("host", "kv_prefix", prefix_n,
                                   "shared published-prefix pages in pinned host RAM"))
            if ssd_n:
                derived.append(Row("ssd", "kv_cold_ssd", ssd_n, "demoted pages spilled past the host budget"))
            if ssd_cap:
                derived.append(Row("ssd", "kv_cold_ssd_capacity", ssd_cap,
                                   "countable spill budget used by admission"))
            graph_n = self._sparse_graph_bytes()
            if graph_n:
                derived.append(
                    Row(
                        "device",
                        "sparse_graph",
                        graph_n,
                        f"captured sparse decode tick staging, {len(self._sparse_graphs)} bucket(s)",
                    )
                )
        rows = memory_table(derived, self._held_storage(), self._measured_peak_bytes())
        # Dense engine with a cold tier driven by the manual sparse_retier seam (#500):
        # its host pages are not in plan(), so append the held allocation explicitly.
        # (The sparse engine lists kv_cold in ``derived`` above.) Derived from the priced
        # cold format (per_cold_kv_block_bytes x pages), so delta catches a D2H copy of a
        # different width than the plan priced (the sm70 f16 narrowing).
        cold = getattr(kv, "cold", None)
        if self._sparse is None and cold is not None and (cold.bytes_held or cold.ssd_bytes):
            from .memory import per_cold_kv_block_bytes

            per = per_cold_kv_block_bytes(self._model.cfg, kv.dtype, kv.kv_fp8, kv.cold_dtype)
            st = cold.stats()
            if cold.bytes_held:
                pages = st["kv_cold_pages"]
                rows.append(
                    {
                        "tier": "host",
                        "owner": "kv_cold",
                        "kind": "allocation",
                        "derived": pages * per,
                        "note": f"{pages} demoted pages in host RAM",
                        "measured": cold.bytes_held,
                        "delta": pages * per - cold.bytes_held,
                    }
                )
            if cold.ssd_bytes:
                pages = st["kv_cold_ssd_pages"]
                rows.append(
                    {
                        "tier": "ssd",
                        "owner": "kv_cold_ssd",
                        "kind": "allocation",
                        "derived": pages * per,
                        "note": f"{pages} demoted pages spilled to SSD",
                        "measured": cold.ssd_bytes,
                        "delta": pages * per - cold.ssd_bytes,
                    }
                )
        # The cold-start boot store on disk: its derived == measured bytes (one
        # kv_cold(ssd) allocation per saved entry), so the plan and the filesystem agree.
        if self._boot is not None:
            ssd_bytes = self._boot.bytes_total()
            if ssd_bytes:
                rows.append(
                    {
                        "tier": "ssd",
                        "owner": "kv_cold",
                        "kind": "allocation",
                        "derived": ssd_bytes,
                        "note": f"{self._boot.entries()} saved boot entries",
                        "measured": ssd_bytes,
                        "delta": 0,
                    }
                )
        # Memoize a fully static ledger so the twice-a-step stats call does not
        # re-walk the param tensors. Plain dense with no live seams qualifies; so
        # does a HYBRID engine -- it reconciles to the dense whole-pool view, and
        # its sparse residency is reported by the separate flat stats keys, not by
        # this ledger. Without the hybrid clause an all-dense hybrid tick paid the
        # full plan twice per step (measured ~6% vs main, #586 device A-B). A pure
        # sparse engine keeps the live ledger (kv_hot tracks residency per tick).
        hybrid = self._sparse is not None and self._sparse_min_tokens
        if hybrid or (
            self._sparse is None
            and self._boot is None
            and getattr(kv, "cold", None) is None
        ):
            self._mem_rows = (n_params, rows)
        return rows

    def sparse_retier(self, keep: frozenset[int]) -> tuple[int, int]:
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
        pool = self._kv
        cold = pool.cold
        if cold is None:
            raise RuntimeError("sparse_retier: engine built without a cold page tier")
        d0, p0 = cold.demotions, cold.promotions
        remap: dict[int, int] = {}
        reqs = list(self._running) + list(self._waiting)
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

    def sparse_selection_recall(self, req_id: int, target_mass: torch.Tensor) -> dict[int, float]:
        """Recall of the LAST tick's served selection against an offline dense
        page-mass target ``[1, n_groups, nq, pages]`` (window excluded), per
        source group. Lets the card run score what the engine actually selected
        rather than only the offline teacher. Full k -> 1.0."""
        sel = self._sparse.last_selected.get(req_id)
        if sel is None:
            return {}
        out: dict[int, float] = {}
        for g, (cand, chosen) in sel.items():
            k = min(self._sparse.k_pages, len(cand))
            if k == 0:
                out[g] = 0.0
                continue
            ci = torch.tensor(cand, device=target_mass.device)
            mass = target_mass[0, g].sum(0).index_select(0, ci)
            dense = {cand[int(i)] for i in mass.topk(k).indices.tolist()}
            out[g] = len(dense & set(chosen)) / k
        return out

    def _sparse_rows(self, rows: list[_Req], seq_q: list[int], decodes: list[_Req]):
        """Build this tick's SparseForward: per-row own span (allocated/promoted),
        earlier complete candidate pages, and a resolve closure that promotes a cold
        selection. The model scores and installs page_sel mid-forward."""
        from .sparse_engine import SparseForward
        from .sparse_index import WINDOW_PAGES as _WP

        tr = self._sparse
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
                return self._sparse_resolve(r, p, reserved)

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
        if self._sparse_device_select and pure_decode:
            self._sparse_ticks_since_refresh += 1
            do_refresh = self._sparse_ticks_since_refresh >= SPARSE_REFRESH_TICKS
        else:
            do_refresh = False
        device_select = self._sparse_device_select and pure_decode and not do_refresh
        if do_refresh:
            self._sparse_ticks_since_refresh = 0
        return SparseForward(tr, srows, self._backend.device, device_select=device_select)

    def _sparse_evict_victim(self, r: _Req, reserved: set[int]) -> None:
        """Free one frame this tick does NOT need, so a promotion can allocate.

        Cross-tick pin leaves last tick's selected pages resident; when this tick's
        selection differs, a newly named page needs that frame. Demote any resident
        page of this row outside the tick's reserved set (own span + this tick's
        picks across groups). Raises if every resident page is reserved — that would
        mean the pool was undersized below the pin ceiling, a build_engine bug."""
        live = self._sparse.resident[r.req_id]
        for p, phys in live.items():
            if p in reserved:
                continue
            self._kv.demote_page(phys, key=(r.req_id, p))
            r.cold_pages.append(p)
            r.blocks.remove(phys)
            self._sparse.map_evict(r.req_id, p)
            del live[p]
            return
        raise RuntimeError(
            "sparse: no unreserved resident page to evict for a "
            "promotion; hot pool undersized below the pin ceiling"
        )

    def _sparse_resolve(self, r: _Req, page: int, reserved: set[int] | None = None) -> int:
        """Physical block for a resident, private-cold, or shared-prefix logical
        page, allocating a fresh block for a never-written own page or promoting
        the host blob. A shared prefix page promotes its read-only host blob into
        a fresh PRIVATE block (the store entry keeps the blob); promoting makes
        the page private, so it is unlinked from the shared key. Under the
        cross-tick pin the pool is full of last tick's pages, so evict one
        unreserved frame first when no block is free."""
        tr = self._sparse
        live = tr.resident[r.req_id]
        if page in live:
            return live[page]
        if self._kv.free_blocks == 0 and reserved is not None:
            self._sparse_evict_victim(r, reserved)
        # Automatic path: cold_pages are bare logical ints, blob keyed (req, page).
        # A page the publisher itself dropped also has a shared key: prefer its
        # private blob, fall back to the shared clone on byte-LRU eviction.
        shared_keys = tr.shared.get(r.req_id, {})
        if page in r.cold_pages and (r.req_id, page) in self._kv.cold:
            new = self._kv.promote_keyed((r.req_id, page))
            r.cold_pages.remove(page)
            shared_keys.pop(page, None)
        elif page in shared_keys:
            blob = self._kv.cold.share_take(shared_keys[page])
            if blob is None:
                raise RuntimeError(f"sparse prefix page {page} missing its shared blob")
            new = self._kv.shared_promote(blob)
            shared_keys.pop(page)  # the store keeps its ref; this page is now private
            if page in r.cold_pages:
                r.cold_pages.remove(page)  # transfer moved the blob to the content key
        else:
            new = self._kv.alloc_block()
        live[page] = new
        r.blocks.append(new)
        tr.map_resident(r.req_id, page, new)
        return new

    def _sparse_warm_draft(self, r: _Req, entry: dict, matched: int) -> None:
        """Restore a WARM prefix into a spec follower's dense draft pool: copy the
        publisher's per-page draft K/V into this row's reserved draft blocks,
        zero the boundary slot, and prime draft state so the first tail draft
        conditions on the saved boundary trunk hidden. Bit-equal to a cold
        follower: cold zeroed position 0 in block 0; warm zeroes position
        ``matched`` in block M, and every earlier draft slot is the publisher's
        own value (already zero at its position 0)."""
        dpool = self._draft.kv
        dev = dpool.k_pool.device
        keys = entry["keys"]
        for p, key in enumerate(keys):
            # field-only read: restore the draft planes without pinning the page's
            # trunk K/V blob into host RAM
            dk = self._kv.cold.share_take_field(key, "dk")
            dv = self._kv.cold.share_take_field(key, "dv")
            if dk is None or dv is None:
                raise RuntimeError(
                    f"warm prefix entry lost draft KV for page {p} (key {key})")
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
        self._prefix_warm_adoptions += 1

    def _sparse_offer_drop(self, r: _Req, page: int, draft_pages: dict | None = None) -> None:
        """Page ``page`` just LEFT the resident union: offer it to the prefix index.
        Drop-only — a stable pin never reaches here. No blob is copied or moved
        yet: the index buffers out-of-order drops behind the contiguous frontier
        and skips a page with no bound, so an entry never names a page it cannot
        serve. When the frontier closes, :meth:`SparsePrefixCache.publish_dropped`
        hands back the content keys and :meth:`_sparse_transfer_to_shared`
        REHOMES each private blob to its content key (one copy, not two). A
        newly published page's draft K/V is copied from the request's draft pool
        (warm spec adoption)."""
        tr = self._sparse
        if tr.prefix is None or not tr.has_bounds(r.req_id, page):
            return
        keys = tr.prefix.publish_dropped(
            r.req_id, r.tokens, tr.bounds_view(r.req_id), page, (r.req_id, page))
        # The frontier can close over MANY pages though only ``page`` dropped
        # this tick, so resolve each new page's draft block from the reserved
        # draft span, not from the one dropped page.
        written_page = ((r.draft_pos + 1) // BLOCK_TOKENS
                        if self._draft is not None and r.draft_blocks else -1)
        for p, content_key in keys.items():
            draft_block = r.draft_blocks[p] if p <= written_page else None
            self._sparse_transfer_to_shared(r, p, content_key, draft_block)
        for content_key in tr.prefix.take_freeze_refs():
            self._kv.cold.share_ref(content_key)

    def _sparse_transfer_to_shared(self, r: _Req, page: int, content_key: int,
                                   draft_block: int | None = None) -> None:
        """Publish one page under its content key: attach bounds (+ draft K/V for
        a warm spec entry) to the page's trunk K/V.

        Three states the page can be in when its frontier closes:
        - held on the host as the PRIVATE (rid,page) blob: transfer it (no copy);
        - on the private SSD: share_hold_kv lifts it into the prefix spill file;
        - still DEVICE-resident (in the own window, never dropped): build the host
          blob from its live physical frame here.
        Returning without a blob would leave a lookup entry naming a dead key."""
        tr = self._sparse
        extra = {"bounds": tr.bounds_view(r.req_id)[page].cpu()}
        if draft_block is not None and self._draft is not None:
            dpool = self._draft.kv
            # clone: .cpu() is a no-op on the CPU cell, so without it the blob
            # aliases a draft block that gets recycled and overwritten
            extra["dk"] = dpool.k_pool[:, draft_block].detach().cpu().clone()
            extra["dv"] = dpool.v_pool[:, draft_block].detach().cpu().clone()
        tr.shared.setdefault(r.req_id, {})[page] = content_key
        if (r.req_id, page) in self._kv.cold:
            n = self._kv.cold.share_hold_kv(
                (r.req_id, page), content_key, extra=extra)
            if n:
                return
            # private blob was past the host budget and consumed into the prefix
            # spill; future demotes re-home under the content key
            r.cold_pages = [
                content_key if (isinstance(p, tuple) and p == (r.req_id, page))
                else p for p in r.cold_pages]
            return
        phys = tr.resident.get(r.req_id, {}).get(page)
        if phys is None:
            raise RuntimeError(
                f"publish page {page}: neither a private host blob nor a resident "
                f"frame exists (req {r.req_id}, content key {content_key})")
        # Device-resident: snapshot the frame directly (it stays live; the page
        # did not leave the union this tick). No pool block is freed.
        blob, n = self._kv._page_blob(phys)
        blob.update(extra)
        n += sum(t.numel() * t.element_size() for t in extra.values()
                 if torch.is_tensor(t))
        self._kv.cold.share_hold(content_key, blob, n)

    def _sparse_finalize(self, sf, rows: list[_Req], hidden=None) -> list:
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
        tr = self._sparse
        pool = self._kv
        from .sparse_engine import page_bounds_one, project_index_page_keys

        # One batched D2H for the tick's departing pages across every row: each
        # demote launches non-blocking into pinned staging while its frame stays
        # live; the context syncs once before the frames return to the pool.
        # (row, dropped pages) collected in the scope, offered to the prefix index
        # after it exits — the one point every demoted blob is guaranteed held.
        dropped_offers: list[tuple[_Req, list[int]]] = []
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
                    sp = self._states
                    # vector at position q_hi-1; note_boundary moves it to host
                    boundary_h = (None if hidden is None or self._draft is None
                                 else hidden[bi, sf.rows[bi]["tq"] - 1])
                    tr.prefix.note_boundary(
                        rid, complete,
                        (sp.states[r.state_slot].clone(),
                         sp.window_snapshot(r.state_slot)),
                        boundary_h)
                kept = sf.selected_pages(bi)
                dropped = [p for p in live if p not in kept]
                for p in dropped:
                    phys = live[p]
                    pool.demote_page(phys, key=(rid, p))
                    tr.map_evict(rid, p)
                    r.blocks.remove(phys)
                    r.cold_pages.append(p)
                kept_live = {p: live[p] for p in kept if p in live}
                # r.blocks mirrors the live frames in LOGICAL page order (paged_attention
                # derives causal positions from the order), so sort the pinned set.
                r.blocks = [kept_live[p] for p in sorted(kept_live)]
                live.clear()
                live.update(kept_live)
                if dropped:
                    dropped_offers.append((r, dropped))

        return dropped_offers

    def _sparse_process_offers(self, dropped_offers) -> None:
        """Publish dropped pages AFTER ``draft.step``: this chunk's draft K/V now
        exist in the request's draft pool for the transfer to copy."""
        for r, pages in dropped_offers:
            for p in pages:
                self._sparse_offer_drop(r, p)

    def _sparse_decode_rows(self, decodes: list[_Req], q_dec: list[int]) -> list[dict]:
        """Decode-only geometry for a captured sparse tick — the decode branch of
        ``_sparse_rows`` without its refresh bookkeeping (the runner owns that).
        Own pages resolve via the same closure; candidates are the complete
        earlier pages with stored bounds."""
        from .sparse_index import WINDOW_PAGES as _WP

        tr = self._sparse
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
                return self._sparse_resolve(r, p, reserved)

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

    def _run_sparse_decode_graph(self, reqs: list[_Req], chains) -> bool:
        """Capture/replay the sparse steady-state decode tick. Returns False (caller
        runs eager) on a refresh tick (needed promotions), a bounds-scorer-only
        configuration, or a failed capture. The refresh counter is advanced ONLY
        on a captured tick; eager goes through ``_sparse_rows`` which owns it."""
        from .sparse_engine import SPARSE_REFRESH_TICKS, SparseForward, cmax_bucket
        from .sparse_index import WINDOW_PAGES as _WP

        tr = self._sparse
        if tr.scorer != "bounds":
            return False
        q_dec = [len(c) for c in chains] if chains else [1] * len(reqs)
        # Read-only peek: let _sparse_rows do the reset when this tick is a refresh.
        if self._sparse_ticks_since_refresh + 1 >= SPARSE_REFRESH_TICKS:
            return False
        rows = self._sparse_decode_rows(reqs, q_dec)
        n = len(reqs)
        B = self._graph_bucket(n)
        W = max(q_dec)
        own_w = _WP + (1 if W > 1 else 0)
        cmax = max(len(r["cand"]) for r in rows)
        key = (B, W, cmax_bucket(cmax), own_w)
        g = self._sparse_graphs.get(key)
        if g is None:
            if n < B and self._pad_slot is None:
                try:
                    self._pad_slot = self._states.alloc_slot()
                    self._pad_block = self._kv.alloc_block()
                except RuntimeError:
                    return False
            sf = SparseForward(
                tr,
                None,
                self._backend.device,
                device_select=True,
                reuse=True,
                b=B,
                cmax_cap=key[2],
                own_w_cap=own_w,
            )
            try:
                g = self._make_sparse_graph(sf, B, W)
            except Exception as exc:
                warnings.warn(
                    f"sparse decode graph capture failed for {key} ({exc}); eager fallback"
                )
                self._sparse_graph_on = False
                return False
            self._sparse_graphs[key] = g
        logits = g.run(
            rows,
            chains or [(r.output[-1],) for r in reqs],
            pad=None if self._pad_slot is None else (self._pad_slot, self._pad_block),
        )
        self._sparse_ticks_since_refresh += 1
        self._decode_forwards += 1
        # Finalize residency immediately after the forward (same order as the
        # eager path, engine _sparse_finalize before sample/verify): the pin reads
        # this tick's selection out of the captured sf and demotes the rest.
        self._sparse_finalize(g.sf, reqs)
        if chains:
            self._verify(reqs, chains, logits, g.hidden)
        else:
            if self._draft is not None and g.hidden is not None:
                for i, r in enumerate(reqs):
                    r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
                    r.hidden, r.hidden_from = g.hidden[i : i + 1], r.seq_len - 1
            self._sample_commit([(r, logits[i, -1], len(r.output)) for i, r in enumerate(reqs)])
        if self._draft is not None:
            end = self._width - 1
            for r in reqs:
                assert len(r.draft_blocks) * BLOCK_TOKENS > r.seq_len - 1 + end
            if self._draft_ms is None:
                self._draft.step(reqs)
            else:
                self._draft_step_timed(reqs)
        return True

    def _make_sparse_graph(self, sf, B: int, W: int):
        """_SparseDecodeGraph on cuda (real capture); a plain recorded forward on
        CPU so the staging-only math is testable without a card."""
        if self._backend.device.type == "cuda":
            if self._graph_pool is None:
                self._graph_pool = torch.cuda.graph_pool_handle()
            return _SparseDecodeGraph(
                self._model,
                self._backend,
                self._kv,
                self._states,
                self._sparse,
                sf,
                B,
                W,
                pool=self._graph_pool,
                aux_layers=self._aux_layers,
            )
        return _CpuSparseGraph(self._model, self._backend, self._kv, self._states, sf, B, W)

    def _make_kv(self, reqs: list[_Req], seq_q: list[int], keep_steps: int = 0, sf=None) -> BatchKv:
        sparse = sf is not None
        if sparse:
            bt = sf.own_table  # own-only table; width and page_base live on sf
        else:
            # Table width = pool size: the kernels compile it in, so a per-tick width recompiles.
            bt = torch.zeros(len(reqs), self._kv.num_blocks, dtype=torch.long, pin_memory=self._pin)
        sl = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        ss = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        sql = torch.empty(len(reqs), dtype=torch.long, pin_memory=self._pin)
        for i, r in enumerate(reqs):
            if not sparse:
                bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
            # Length after this forward; a decode row's chain starts at seq_len-1.
            sl[i] = (
                r.prefill_from + seq_q[i] if r.phase == _PHASE_PREFILL else r.seq_len - 1 + seq_q[i]
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
            page_base=None if not sparse else sf.page_base,
            sparse=sf,
        )

    def _run_forward(self, decodes: list[_Req], prefills: list[_Req], chunks: list[int]) -> None:
        # Asserted, not branched on: every row comes from `_build_plan`, so an unadmitted one
        # is a planner bug that would otherwise surface as `free_slot(None)`.
        for r in (*decodes, *prefills):
            assert r.state_slot is not None, (
                f"request {r.req_id} reached the forward unadmitted (no state slot)"
            )
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
            _decode_extra_blocks(r.seq_len, q, len(r.blocks))
            for r, q in zip(decodes, q_dec)
        )
        if growth:
            self._prefix.evict_until_free(growth)
        dead: set[int] = set()
        for i, (r, q) in enumerate(zip(decodes, q_dec)):
            if r.sparse_on:
                continue  # sparse grows its own pages lazily in _sparse_rows
            # Cover the chain's last position. By count, not by catching alloc_block's
            # raise: `_admit` does the same for the same reason -- its comment says an
            # exception out of here reaches step()'s handler and fails EVERY running
            # request. A row that does not fit fails alone and leaves the batch.
            need = _decode_extra_blocks(r.seq_len, q, len(r.blocks))
            if need > self._kv.free_blocks:
                self._finish(
                    r,
                    error=f"PagedKvPool exhausted: need {need} block(s), "
                    f"{self._kv.free_blocks} free",
                    reason="pool_exhausted",
                )
                dead.add(i)
                continue
            while len(r.blocks) * BLOCK_TOKENS <= r.seq_len - 2 + q:
                r.blocks.append(self._kv.alloc_block())
                r.own_blocks += 1
                self._blocks_used += 1
        if dead:
            decodes = [r for i, r in enumerate(decodes) if i not in dead]
            q_dec = [q for i, q in enumerate(q_dec) if i not in dead]
            if chains is not None:
                chains = [c for i, c in enumerate(chains) if i not in dead]
            if not decodes and not prefills:
                return
        tick_sparse = bool(decodes or prefills) and (decodes + prefills)[0].sparse_on
        if (
            not prefills
            and decodes
            and not tick_sparse
            and self._decode_graph_on
            and self._run_decode_graph(decodes, chains)
        ):
            self._hybrid_charge(False)
            return
        if (
            not prefills
            and decodes
            and tick_sparse
            and self._sparse_graph_on
            and self._run_sparse_decode_graph(decodes, chains)
        ):
            self._hybrid_charge(True)
            return
        rows = decodes + prefills
        seq_q = q_dec + chunks
        sparse = tick_sparse
        if sparse:
            # Sparse grows blocks lazily inside selection, so skip the dense pre-allocation
            # of the chain's tail (pages are promoted/allocated by _sparse_rows).
            # One batched H2D for the tick's promotions: own-span resolves below and the
            # selected pages resolve inside the model forward; the context retains their
            # blobs and synchronizes once, instead of one cuda sync per promoted page.
            promote_ctx = self._kv.promotions()
            sf = self._sparse_rows(rows, seq_q, decodes)
            promote_ctx.__enter__()
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
        t_fwd = time.perf_counter()
        logits = self._model.forward(
            input_ids,
            positions,
            self._make_kv(rows, seq_q, width if chains else 0, sf if sparse else None),
            self._backend,
            hidden_out=hid,
            aux_layers=self._aux_layers,
            last_only=False if chains else seq_q,  # a verify tick needs every chain position
        )
        if sparse:
            promote_ctx.__exit__(None, None, None)
            try:
                sparse_offers = self._sparse_finalize(
                    sf, rows, None if hid is None else hid[-1])
            except SpillWriteError as e:
                # A private cold page this tick demoted could not be spilled; the
                # batched demotions() exit cannot say which row owns it, so fail
                # the whole sparse tick set with a client-visible error and free
                # their slots/blocks. Returning leaves waiting requests and the
                # server serving. The wedge this replaces: the error escaped
                # step, the loop retried forever with no commit and a leaked slot
                # (V100 unwritable /data00 spill, 2026-09-14).
                for r in rows:
                    if r in self._running:
                        self._finish(r, error=str(e), reason="cold_spill_failed")
                return
            # retain the last tick's served candidates+selection so a recall probe
            # reads it after the SparseForward is discarded:
            # req -> {group: (candidate_pages, chosen_candidate_pages)}.
            self._sparse.last_selected = {
                r.req_id: {
                    g: (list(sf.rows[i]["cand"]), sf.selected(i, g)) for g in range(sf.n_groups)
                }
                for i, r in enumerate(rows)
            }
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
            # mixed ticks included: excluding them reports a rate no request sees
            self._prefill_tokens += sum(chunks)
            self._prefill_secs += time.perf_counter() - t_fwd
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
                if r.sparse_on:
                    # The dense draft KV span is fully RESERVED at admit (prompt +
                    # max_new + verify width bound), so the blocks already exist.
                    # This is a bound check, not a grow: allocating here would race
                    # another row for the shared draft pool.
                    end = r.seq_len - 1 + self._width - 1
                    assert len(r.draft_blocks) * BLOCK_TOKENS > end, (
                        f"draft needs position {end} but admit reserved "
                        f"{len(r.draft_blocks)} blocks"
                    )
                    continue
                while r.blocks and len(r.blocks) * BLOCK_TOKENS <= r.seq_len - 1:
                    r.blocks.append(self._kv.alloc_block())
                    r.own_blocks += 1
                    self._blocks_used += 1
            if self._draft_ms is None:
                self._draft.step(rows)  # every tick, or a chunked prefill leaves the draft KV empty
            else:
                self._draft_step_timed(rows)
            if sparse:
                # draft K/V for this tick's dropped pages now exist: publish them
                self._sparse_process_offers(sparse_offers)
        elif sparse:
            self._sparse_process_offers(sparse_offers)

    def _sparse_live_stats(self) -> dict:
        """Flat sparse residency counters for a hybrid engine. The memory ledger
        reconciles the dense pool view, so without these the sparse rows' hot /
        bounds / cold occupancy during a concurrent run is invisible. When no
        sparse row is running this returns the cheap zeros only -- the bounds sum
        must not tax an all-dense tick (the #586 all-dense A-B)."""
        live = [r for r in self._running if r.sparse_on and r.phase != _PHASE_DONE]
        if not live:
            return {
                "sparse_hot_pages": 0,
                "sparse_hot_bytes": 0,
                "page_bounds_bytes": 0,
                "kv_cold_bytes": 0,
                "kv_prefix_bytes": 0,
            }
        from .memory import per_kv_block_bytes

        block_n = per_kv_block_bytes(self._model.cfg, self._kv.dtype, self._kv.kv_fp8)
        hot_pages = sum(len(self._sparse.resident.get(r.req_id, ())) for r in live)
        cold = getattr(self._kv, "cold", None)
        shared_n = cold.shared_bytes() if cold is not None else 0
        return {
            "sparse_hot_pages": hot_pages,
            "sparse_hot_bytes": hot_pages * block_n,
            "page_bounds_bytes": self._sparse.bounds_bytes(),
            "kv_cold_bytes": (cold.bytes_held - shared_n) if cold is not None else 0,
            "kv_prefix_bytes": shared_n,
        }

    def _sparse_hot_headroom(self) -> int:
        """Device blocks live sparse rows can still grow toward their per-slot hot
        ceiling. A dense hybrid admit must leave this much free, or a sparse row
        raises inside a later tick and fails every running row. Conservative:
        ceiling minus currently resident, summed over live sparse rows."""
        from .memory import sparse_hot_pages_per_slot

        ceil = sparse_hot_pages_per_slot(
            self._model.cfg, self._sparse_k, self.limits.max_num_batched_tokens)
        resident = self._sparse.resident
        return sum(
            max(0, ceil - len(resident.get(r.req_id, ())))
            for r in self._running
            if r.sparse_on and r.phase != _PHASE_DONE)

    def _hybrid_charge(self, sparse: bool) -> None:
        """Per-mode counters and rolling-window wall-time accounting.

        Only ticks that run while the OTHER mode also has a runnable row enter
        the window -- time spent solo is not a debt either side owes. A sparse
        tick opens a window carrying its cost; dense ticks accrue against it and
        sparse may run again only once dense has caught up. This serves a dense
        row the instant it arrives (dense is then behind) rather than charging
        it for sparse ticks that ran while no dense row existed."""
        if self._sparse_min_tokens == 0:
            return
        other_present = any(r.sparse_on != sparse for r in self._running)
        dt = (self._hybrid_fake_dt[1 if not sparse else 0]
              if self._hybrid_fake_dt is not None
              else time.perf_counter() - self._hybrid_t0)
        if sparse:
            self._sparse_mode_ticks += 1
            if other_present:
                # New window: this sparse tick's cost is what dense must match.
                self._hybrid_sparse_wall = dt
                self._hybrid_dense_wall = 0.0
        else:
            self._dense_mode_ticks += 1
            if other_present:
                self._hybrid_dense_wall += dt

    def _finish_prefills(self, prefills: list[_Req], chunks: list[int], logits, base: int) -> None:
        done = []
        for k, (pf, c) in enumerate(zip(prefills, chunks)):
            pf.prefill_from += c
            pf.seq_len = pf.prefill_from
            if pf.prefill_from >= len(pf.tokens):
                done.append((pf, logits[base + k, min(c, logits.shape[1]) - 1], 0))
            elif pf.prefill_from % BLOCK_TOKENS == 0:
                # A chunk end is a state-pool boundary, so the snapshot is exact here.
                # The FIRST interior boundary always lands; a later one may be the
                # last, but which is last is schedule-dependent -- decode rows sharing
                # the tick shrink the budget and shift the whole walk -- so hold the
                # newest boundary's exact snapshot and insert it at completion instead
                # of predicting the walk from n alone.
                # errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md
                # The first boundary plus the last, never the ones between: a row's
                # publishes stay at 2 whatever the prompt length, where per-boundary
                # publishing emitted 62 at a 31k prompt and outran any budget a
                # pressured card has. Costs a PARTIAL sharer the intermediate prefixes
                # it would match, and that cost GROWS with prompt length. 83.3% of
                # ideal reuse at 2048 tokens, 12.5% at 16384.
                # errors/2026-09-08-the-eviction-policy-was-the-wrong-layer.md
                pf.interior_published += 1
                predicted = _last_prefill_boundary(len(pf.tokens))
                if pf.interior_published == 1:
                    if not pf.sparse_on:
                        self._publish_prefix(pf, pf.prefill_from)
                elif (not pf.sparse_on
                        and len(pf.tokens) % BLOCK_TOKENS
                        and pf.prefill_from >= predicted):
                    # Tail window [predicted, n): at most two aligned chunk ends, so
                    # this holds <=2 snapshots per ragged prompt and keeps only the
                    # deepest the actual schedule reached. Not predicted: a decode
                    # row sharing the tick shifts the walk past the n-only value.
                    # Published at completion -- at most 32 tokens later, one tick.
                    pf.pending_prefix = (
                        pf.prefill_from,
                        (self._states.states[pf.state_slot].clone(),
                         self._states.window_snapshot(pf.state_slot)),
                    )
        if not done:
            return
        self._sample_commit(done)
        for pf, _, _ in done:
            # The state slot still covers exactly the prompt, so the snapshot is exact.
            prompt_len = len(pf.tokens) - len(pf.output)
            if not pf.sparse_on and pf.phase != _PHASE_DONE and prompt_len % BLOCK_TOKENS == 0:
                self._publish_prefix(pf, prompt_len)
            elif (not pf.sparse_on and pf.phase != _PHASE_DONE
                    and pf.pending_prefix is not None):
                # Ragged prompt: the held boundary snapshot is exact and its
                # blocks are still live; insert it at completion.
                pos, snap = pf.pending_prefix
                self._prefix_published += self._prefix.insert(
                    pf.tokens[:pos], pf.blocks[: pos // BLOCK_TOKENS], snap)
            pf.pending_prefix = None
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
            g = _DecodeGraph(
                self._model,
                self._backend,
                self._kv,
                self._states,
                B,
                width=W,
                pool=self._graph_pool,
                keep=W if keep else 0,
                aux_layers=self._aux_layers,
            )
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
        return {
            (self._graph_bucket(rows), w)
            for rows in range(1, self.limits.max_batch + 1)
            for w in widths
        }

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
        """Drop everything computed under the previous weights; return casts refilled.

        An optimizer step makes both caches lie: a captured graph replays the
        forward as it was traced, and a cached prefix serves KV from the old
        policy. Both are silent -- nothing raises, the rollout is just off-policy
        -- which is why ``_require_on_policy`` refuses an engine carrying either.
        Calling this after each update is what lets a training engine keep them.

        The graphs are KEPT, because every address one baked survives the update:
        ``AdamW.step_one`` and ``Adafactor.step_one`` both end ``p.copy_()`` (in
        place), and ``materialize`` rebuilds the dict but not the tensors. The
        one thing that does not survive on its own is a cached cast --
        ``_const_f32`` refills only when something calls it, and a replay calls
        nothing -- so the refill is driven here. The prefix store is cleared: it
        holds KV, not addresses.
        """
        # returns the refill count, not len(graphs): the graphs stay
        n = self._backend.refill_const_f32()
        # The pool owns the captured memory; a new pool per invalidation would
        # leak one arena per step.
        self._prefix.clear()
        return n

    def _run_decode_graph(self, reqs: list[_Req], chains=None) -> bool:
        """Captured decode for a pure-decode tick, one graph per size bucket (a
        graph per exact size OOMed B=64 on the drain). Returns False -- caller
        runs eager -- when capture failed (flag off too) or when this tick would
        need a graph outside the `graph_keys` grid."""
        n, W = len(reqs), len(chains[0]) if chains else 1
        B = self._graph_bucket(n)
        if n < B and self._pad_slot is None:
            try:
                self._pad_slot = self._states.alloc_slot()
                self._pad_block = self._kv.alloc_block()
            except RuntimeError:
                # no pad row: an exact-size graph is off the graph_keys grid and would
                # capture mid-request
                return False
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
            drafted = len(chains[i]) - 1
            self._spec_drafted += drafted
            cap = r.params.max_think_tokens
            crosses_cap = (
                cap is not None and not r.thought_closed
                and len(r.output) < cap <= len(r.output) + drafted
            )
            if r.thought_closed:
                self._spec_acc_post += n_ok
                self._spec_dft_post += drafted
            else:
                self._spec_acc_in += n_ok
                self._spec_dft_in += drafted
            if crosses_cap:
                self._spec_acc_capcross += n_ok
                self._spec_dft_capcross += drafted
            self._states.select_step(r.state_slot, n_ok)
            r.hidden_prev = None if r.hidden is None else r.hidden[:, -1:]
            r.hidden, r.hidden_from = hidden[i : i + 1], r.seq_len - 1
            self._commit(
                r,
                got[: n_ok + 1],
                None if lps is None else lps[at - len(chains[i]) : at][: n_ok + 1],
            )

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
            logits,
            [p.temperature for p in params],
            [p.top_p for p in params],
            [_step_seed(r.params.seed, g) for r, _, g in rows],
            logprobs=want_lp,
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
                req.reply_from = len(req.output)
            req.tokens.append(tok)
            req.seq_len += 1
            self._tokens_generated += 1
            # After the append: the contract keeps the token that completed the match
            # in `output`, so the caller's decode sees it and cuts the text at the
            # match's start. Dropping it would leave a partial stop in the reply.
            # Only past the closer when the prompt opened <think> -- a stop inside the
            # reasoning would return a truncated thought and no answer.
            if (
                p.stop_texts
                and (req.thought_closed or not p.end_think_ids)
                and (hit := _stop_hit(self._decode, req.output[req.reply_from :], p.stop_texts))
            ):
                req.stop_text = hit
                self._finish(req)
                return
            materialized = req.seq_len - 1
            # Replace, not accumulate: only this row's longest decode entry can serve it again.
            # Retire after the insert -- the entries share blocks -- and only if it succeeded.
            if (
                i == last
                and not req.sparse_on
                and req.phase == _PHASE_DECODE
                and materialized % BLOCK_TOKENS == 0
                and self._publish_prefix(req, materialized)
            ):
                if req.decode_entry:
                    self._prefix.retire(req.tokens[: req.decode_entry])
                req.decode_entry = materialized
            if len(req.output) >= p.max_new_tokens:
                self._finish(req)
                return
            if tok != raw:  # a forced end-think token: the rest of the chain is stale
                return

    def save_boot(self, req: _Req) -> int:
        """Persist this request's whole block-aligned context and its recurrent snapshot to
        the --kv-store, so a later cold start boots from it instead of prefilling. Returns
        bytes written. Raises if the engine has no kv_store or the row is not block-aligned
        at the prefill/decode boundary."""
        if self._boot is None:
            raise RuntimeError("save_boot: engine built without --kv-store")
        n = (req.seq_len // BLOCK_TOKENS) * BLOCK_TOKENS
        if n == 0:
            raise RuntimeError("save_boot: nothing block-aligned to save yet")
        blocks = req.blocks[: n // BLOCK_TOKENS]
        state = {
            "states": self._states.states[req.state_slot].clone().cpu(),
            "window": (
                None
                if self._states.conv_windows is None
                else self._states.window_snapshot(req.state_slot)
            ),
            "parity": int(self._states.win_parity[req.state_slot]),
        }
        return self._boot.save(req.tokens[:n], self._kv, blocks, state)

    def _publish_prefix(self, req: _Req, length: int) -> bool:
        """Hand tokens[:length], its blocks and the linear-state snapshot at that
        boundary to the store; the store owns and evicts all three together."""
        snap = (
            self._states.states[req.state_slot].clone(),
            self._states.window_snapshot(req.state_slot),
        )
        published = self._prefix.insert(
            req.tokens[:length], req.blocks[: length // BLOCK_TOKENS], snap
        )
        self._prefix_published += published
        return published

    def _release(self, req: _Req) -> None:
        """Give back the blocks and the slot. Here, not at poll, so capacity returns now."""
        req.phase = _PHASE_DONE
        if req.state_slot is None:
            return  # never admitted; blocks and slot are taken together in `_admit`
        if req.sparse_on and self._sparse is not None:
            # pages belong to another publisher's blobs) forces its prompt-end
            # frontier closure while device frames and draft blocks are still
            # live: pages a hot pool never dropped get snapshotted from the live
            # frame here, so a same-prompt follower can still adopt the prefix.
            if self._sparse.prefix is not None and req.sparse_matched == 0 and not req.failed:
                keys = self._sparse.prefix.close_request(
                    req.req_id, req.tokens, self._sparse.bounds_view(req.req_id))
                written_page = ((req.draft_pos + 1) // BLOCK_TOKENS
                                if self._draft is not None and req.draft_blocks
                                else -1)
                for p, content_key in keys.items():
                    draft_block = req.draft_blocks[p] if p <= written_page else None
                    self._sparse_transfer_to_shared(req, p, content_key, draft_block)
                for content_key in self._sparse.prefix.take_freeze_refs():
                    self._kv.cold.share_ref(content_key)
            # Sparse: drop this request's host-held cold blobs, keyed (req, logical
            # page) and never present in req.blocks, plus its bounds store.
            cold = self._kv.cold
            if cold is not None:
                for p in req.cold_pages:
                    cold.forget((req.req_id, p) if isinstance(p, int) else p)
            self._sparse.drop(req.req_id)
        elif self._kv.cold is not None and req.cold_pages:
            # #500 manual sparse_retier seam: _sparse is None, cold_pages are (idx, phys).
            for _idx, b in req.cold_pages:
                if b in self._kv.cold:
                    self._kv.cold.forget(b)
        if req.draft_blocks:
            # Sparse+spec: dense draft KV lives in the draft pool's own id space.
            dpool = self._draft.kv
            for b in req.draft_blocks:
                dpool.free_block(b)
        for b in req.blocks:
            self._kv.free_block(b)
        self._blocks_used -= req.own_blocks
        req.pending_prefix = None  # a prefill that never completed still held a snapshot
        self._states.free_slot(req.state_slot)
        self._slots_used -= 1

    def _finish(self, req: _Req, error: str | None = None, reason: str | None = None) -> None:
        if error is not None:
            req.failed = True
        self._release(req)
        if error is None:
            self._finished[req.req_id] = req.output
            if req.stop_text is not None:
                self._finished_stop[req.req_id] = req.stop_text
            if req.params.logprobs:
                self._finished_logprobs[req.req_id] = req.logprobs
        else:
            self._failed[req.req_id] = (reason, error)
        self._finished_count += 1
        self._running.remove(req)

    def cancel(self, request_id: int) -> bool:
        """Drop a request whose reader left; True if it was still in the engine.

        Not `_finish`: that ends with `_running.remove(req)` and raises on a request
        still waiting, which owns blocks and a slot just the same. `_failed`, not
        `_finished`, so a later `take()` raises instead of returning the None that
        already means "not finished yet".
        """
        # ponytail: `_failed` grows one entry per abandoned request; TTL sweep if it bites.
        with self._lock:
            for queue in (self._running, self._waiting):
                req = next((r for r in queue if r.req_id == request_id), None)
                if req is not None:
                    # Same flag a cold-spill failure row carries: _release then skips
                    # close_request, which would publish and maybe spill a prefix for a
                    # reader that is already gone.
                    req.failed = True
                    self._release(req)
                    self._failed[request_id] = (None, "cancelled: the reader disconnected")
                    self._finished_count += 1
                    queue.remove(req)
                    # A cancel leaves no rows, so the loop idles and step()'s post-tick
                    # refresh never runs: without this /health keeps reporting the dead row.
                    if self._thread is not None:
                        self._stats_snapshot = self._build_stats()
                    return True
            return False

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


def _weight_fingerprint(cfg, kv_fp8: torch.dtype | None = None) -> str:
    """What the spilled KV was computed under, as far as the config knows.

    EVERY config field, not a hand-picked list of the ones that look load-bearing: a
    mismatch is the only thing standing between a restart and serving KV computed under
    other weights, and a field left out of the list is exactly how that happens. The
    first draft of this named `cfg.num_heads`, which does not exist -- the real field is
    `num_attention_heads` -- so the list was already wrong when it was written.

    `kv_fp8` is not a config field but IS the store's byte format, so it is appended: a
    flag flip against the same --kv-store otherwise adopts blobs of the other format, which
    is a RuntimeError in one direction and untrustworthy numerics in the other.

    It does NOT distinguish two checkpoints of the same architecture. Pass
    `ssd_fingerprint` explicitly when one boot store directory serves both.
    """
    import dataclasses

    fields = "-".join(f"{f.name}={getattr(cfg, f.name)!r}" for f in dataclasses.fields(cfg))
    return f"{fields}-block{BLOCK_TOKENS}-kv{kv_fp8 or 'io'}"


#: Card ownership prefix rules — same as scripts/card_owner.py on the pod.
#: If you change these, change card_owner.py too (and vice versa).
_OURS = re.compile(r"^\s*(tile[_-]?rl|rl[_-]?team)\b", re.IGNORECASE)
_THEIRS = re.compile(r"^\s*(granted\b|\d{4}-\d{2}-\d{2})", re.IGNORECASE)


def _is_lent(note, card: str) -> bool:
    """Whether the note records this card as lent out.

    The note is prose; lends are recorded as e.g. "cards 1 and 3 are lent to b0".
    Conservative: unparseable note → assume lent (refuse).  Empty/missing → allow.
    """
    # ponytail: lends live in prose because aupai's schema has no lend field;
    # delete this when cards[] records the lend structurally
    if not isinstance(note, str):
        return True
    if not note:
        return False
    for sentence in re.split(r"[.!?]", note):
        if re.search(r"\blent\b|\blending\b|\blend\b", sentence, re.IGNORECASE) and re.search(
            rf"\b{re.escape(card)}\b", sentence
        ):
            return True
    return False


def _stale_context(path: Path, note: str) -> str:
    """One-line context for a refusal: note excerpt + file mtime, so a stale
    assignment file is visible in the error instead of silently trusted."""
    import datetime

    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes")
    excerpt = (note or "")[:120].replace("\n", " ")
    return f"\n  note[0:120]: {excerpt!r}\n  file mtime:  {mtime}"


def card_guard() -> None:
    """Refuse to build an engine on a card not granted to tileRL, when a grant ledger exists.

    Two conditions, both must pass:
    1. ``cards[card]`` classifies as ours (same prefix rules as card_owner.py)
    2. The ``note`` field has no lend record for this card

    No card_assignment.json → no grant system on this machine → pass.
    TILERL_CARD_LEND=<ref> is the explicit escape hatch, echoed to stderr.
    """
    path = Path(os.environ.get("CARD_ASSIGNMENT_JSON", "/work/aupai/runs/card_assignment.json"))
    if not path.exists():
        return
    lend = os.environ.get("TILERL_CARD_LEND")
    if lend:
        print(f"card_guard: lend recorded — {lend}", file=sys.stderr)
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        sys.exit(
            "card_guard: CUDA_VISIBLE_DEVICES is unset — all cards visible. "
            "Set it to the card(s) you intend to use, or TILERL_CARD_LEND=<ref>."
        )
    visible = visible.strip()
    if not visible:
        return  # explicitly empty → no cards → CPU only → pass
    data = json.loads(path.read_text())
    cards = data.get("cards", {})
    note = data.get("note", "")
    for card in visible.split(","):
        card = card.strip()
        if not card:
            continue
        entry = cards.get(card, "")
        # The ledger's per-card value is either the old free-form string or the
        # current {"owner", "note"} dict. Classify by owner on a dict; the
        # per-card note and the top-level note remain lend records. A dict whose
        # owner matches neither ours nor theirs falls through to unclassified
        # (refuse), so an unknown owner never defaults to ours.
        card_note = str(entry.get("owner", "")) if isinstance(entry, dict) else entry
        if not _OURS.match(card_note):
            kind = "theirs" if _THEIRS.match(card_note) else "unclassified"
            sys.exit(
                f"card_guard: card {card} is {kind} per {path}; "
                f"a lend needs TILERL_CARD_LEND=<ledger ref>"
                f"{_stale_context(path, note)}"
            )
        if _is_lent(note, card):
            sys.exit(
                f"card_guard: card {card} is ours but lent out per {path}; "
                f"a lend needs TILERL_CARD_LEND=<ledger ref>"
                f"{_stale_context(path, note)}"
            )
