"""Serving engine: submit/poll loop with one model forward per tick.

A tick runs ONE forward over the planned rows: every running decode row (T=1,
or a 1+depth draft chain on a verify tick) plus prefill chunks up to
``max_num_batched_tokens`` — vLLM/sglang continuous batching with chunked
prefill, after agent-infer's ``build_forward_plan``. ponytail: chunked prefill
is bounded by ``max_num_batched_tokens``; a prompt over ``max_total_tokens`` is
still rejected at ``submit``. On CUDA a pure-decode tick
replays a captured decode graph per batch-size bucket; mixed ticks and every
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

import atexit
import json
import os
import pickle
import re
import sys
import threading
import time
import warnings
import weakref
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .decode_graph import GraphCapture, graph_bucket, make_decode_graph
from .kv_cache import BLOCK_TOKENS, BatchKv, NoPrefixStore
from .kv_tiers import SpillWriteError
from .memory import held_storage, measured_peak_bytes, memory_rows
from .sparse_runtime import SparseCtx, SparseRuntime
from .sparse_runtime import retier as sparse_retier_pages
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


#: Archs on which the speculated sparse captured decode graph has been
#: measured correct. sm70 only: the #805 keep_steps=W fix was verified there by
#: the 09-17 first-replay value gate (per-bucket first replay aligned to an
#: eager reference, 6/6 prompts, both widths) and by a same-window serve run at
#: 24.915 tok/s. sm90 is UNVERIFIED — keep it guarded until its own measurement.
_SPEC_SPARSE_GRAPH_VERIFIED_ARCHS = ("sm70",)


def _sparse_capture_allowed(backend, has_draft: bool, spec_depth: int | None) -> bool:
    """Whether the sparse captured decode graph may be armed for this engine.

    Guard A (#805): on CUDA the captured sparse graph is only correct on the
    d=0 single-token path. With speculation (spec_depth>=1) the width-2
    captured verify replays trunk logits/hidden that disagree with eager and
    drafts stop accepting. That was true of the pre-#805 build on every CUDA
    arch; the `keep_steps=W` fix (PR #808) was then measured correct on **sm70**
    by the 09-17 first-replay value gate against an eager reference, so sm70 may
    now be armed under speculation. Any arch not in
    ``_SPEC_SPARSE_GRAPH_VERIFIED_ARCHS`` stays guarded — unverified is not the
    same as fixed, and sm90 has not been measured. The CPU cell uses the
    CpuSparseGraph eager reference, which is token-exact at W=2 (the width-2
    oracle gate), so it stays enabled. Device-type axis as ``_graph_on``."""
    if backend.device.type != "cuda":
        return True
    if not (has_draft and (spec_depth or 0) >= 1):
        return True
    return getattr(backend, "arch", "") in _SPEC_SPARSE_GRAPH_VERIFIED_ARCHS


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


#: Accumulator keys read from torch.cuda.memory_stats on the slow-tick tail.
#: .get(k, 0) everywhere: the CPU box's memory_stats() is an EMPTY dict, and
#: older wheels can omit keys. segment.all.* count cudaMalloc/cudaFree-backed
#: segments; num_* count caching-allocator events; sync_all_streams is the
#: cross-stream event sync a forced reclaim does.
_MEM_KEYS = (
    "num_sync_all_streams",
    "num_device_alloc",
    "num_device_free",
    "num_alloc_retries",
    "num_ooms",
    "num_oom_rejections",
    "segment.all.allocated",
    "segment.all.freed",
    "reserved_bytes.all.current",
)


class _StepTiming:
    """Env-gated wall-clock timing of the segments step() holds _lock across
    (the slow-tick investigation: 1-5.6 s). TILERL_STEP_TIMING=1 enables,
    TILERL_STEP_TIMING_SLOW_MS sets the per-tick print threshold (500).

    Wall-clock, not CUDA-event time: the question is where lock-held time goes,
    host stalls included. Every slow tick prints its segments to stderr at once;
    exit prints per-segment averages. perf_counter reads stay unconditional at
    the call sites (tens of ns against a 100+ ms tick); with the env off no
    instance exists and the marks themselves are skipped. The figures quoted in
    these comments are illustrative of one machine and run, not expectations to
    compare a reading against.

    "forward" is an ENVELOPE: on the eager path it equals prep + model + sample +
    draft_offers (+ sparse_select/sparse_finalize on sparse ticks); a graph tick
    carries "graph" alone. Do not sum it together with its inner segments.

    Hollow-tick attribution: fwd_start/fwd_end bracket the whole _run_forward
    call (all four returns: eager, dense graph, sparse graph, dead rows) from
    step(). On top of the host envelope they capture an async CUDA-event device
    span and allocator-counter deltas, and the slow-tick tail prints
    fwd_host/fwd_gpu/stall + the counters plus a why= label:
      alloc_reclaim — num_alloc_retries or num_sync_all_streams fired (a
                      forced cache drain/reclaim near the VRAM edge),
      dev_malloc    — a new driver segment without a retry,
      finalize      — sparse_finalize wall dominates with flat counters,
      gpu_drain     — flat counters and the GPU was busy ~the whole span,
      sync_wait     — flat counters and the host waited with the device idle,
      host/unknown/cpu.
    stall = host-device is a LOWER bound on host-only wait: cudaEventElapsedTime
    spans stream-idle bubbles too, and this engine drains the stream inside most
    forwards (tolist/promotion/draft sync), so a mid-forward blocking wait usually
    inflates fwd_gpu and the wall SEGMENTS (model/sample/finalize) carry the
    localisation instead. A drainless tick (dense eager prefills-only mid-chunk,
    no sampling/draft D2H) has an uncompleted end event and prints fwd_gpu=pending
    why=unknown — that is honest, not a missed drain. No synchronize() ever: the
    end event is read with non-blocking query(); a completed tick's own in-forward
    drains already finished it.
    """

    __slots__ = (
        "slow_s",
        "tot",
        "count",
        "cur",
        "t0",
        "n",
        "last_total",
        "note",
        "_eng",
        "cuda",
        "ev_s",
        "ev_e",
        "mem0",
        "fwd_t0",
        "fwd_host_ms",
        "fwd_gpu_ms",
        "fwd_path",
        "fwd_sparse",
        "last_why",
        "alloc_conf",
        "phase_dec",
        "phase_pre",
    )

    def __init__(self, engine=None) -> None:
        self.slow_s = float(os.environ.get("TILERL_STEP_TIMING_SLOW_MS", "500")) / 1000.0
        self.tot: dict[str, float] = {}
        self.count: dict[str, int] = {}
        self.cur: dict[str, float] = {}
        self.t0 = 0.0
        self.n = 0
        self.last_total = 0.0
        self.note = ""
        self._eng = weakref.ref(engine) if engine is not None else lambda: None
        # cuda is lazily resolved on the first tick (is_available, never
        # torch.version.cuda): constructing an Event on a CPU wheel raises.
        self.cuda: bool | None = None
        self.ev_s = None
        self.ev_e = None
        self.mem0: dict[str, int] = {}
        self.fwd_t0 = 0.0
        self.fwd_host_ms = 0.0
        self.fwd_gpu_ms: float | None = None
        self.fwd_path = "eager"
        self.fwd_sparse = False
        self.last_why = ""
        # dec/pre row counts THIS tick, so a log parser can keep decode ticks and
        # drop the chunked-prefill ticks that would otherwise flatten the decode
        # distribution. Env-gated like the rest; off, tick_end never reads it.
        self.phase_dec = 0
        self.phase_pre = 0
        # Counter semantics change by allocator backend; printed so a reading is
        # not made under the wrong assumption (cudaMallocAsync zeros these).
        self.alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")

    def tick_start(self) -> None:
        self.cur.clear()
        self.note = ""
        self.phase_dec = 0
        self.phase_pre = 0
        self.t0 = time.perf_counter()
        self.fwd_host_ms = 0.0
        self.fwd_gpu_ms = None
        self.fwd_path = "eager"
        self.fwd_sparse = False
        self.mem0 = {}
    def mark(self, seg: str, t: float) -> None:
        self.cur[seg] = self.cur.get(seg, 0.0) + time.perf_counter() - t

    def charge_ms(self, seg: str, ms: float) -> None:
        """Add already-measured milliseconds to a segment. For a callee that
        cannot reach the tick counter (the cold tier's mmap file) and accumulates
        its own elapsed time instead."""
        self.cur[seg] = self.cur.get(seg, 0.0) + ms / 1000.0

    @staticmethod
    def _mem_snap() -> dict[str, int]:
        """Allocator counters (host-only mutex-guarded struct copy; the CPU box
        returns {} — every read is .get(k, 0))."""
        s = torch.cuda.memory_stats()
        return {k: int(s.get(k, 0)) for k in _MEM_KEYS}

    def fwd_start(self) -> None:
        """Open the forward-envelope bracket: host anchor, an async start event
        and a pre-forward allocator snapshot. record() is a non-blocking
        cudaEventRecord; no sync, no per-tick event allocation."""
        self.fwd_t0 = time.perf_counter()
        if self.cuda is None:
            self.cuda = torch.cuda.is_available()
            if self.cuda:
                self.ev_s = torch.cuda.Event(enable_timing=True)
                self.ev_e = torch.cuda.Event(enable_timing=True)
        if self.cuda:
            self.ev_s.record()
            self.mem0 = self._mem_snap()

    def fwd_end(self) -> None:
        """Close the host span and record the async end event. The device span is
        NOT read here — that happens in tick_end via non-blocking query()."""
        self.fwd_host_ms = (time.perf_counter() - self.fwd_t0) * 1000.0
        if self.cuda:
            self.ev_e.record()

    def _classify(self, dt_ms: float, dev, d: dict[str, int]) -> str:
        if not self.cuda:
            return "cpu"
        if d["num_alloc_retries"] > 0 or d["num_sync_all_streams"] > 0:
            return "alloc_reclaim"
        if d["num_device_alloc"] > 0 or d["segment.all.allocated"] > 0:
            return "dev_malloc"
        if dev is None:
            return "unknown"
        if self.cur.get("sparse_finalize", 0.0) * 1000 > 0.5 * dt_ms:
            return "finalize"
        if dev >= 0.85 * self.fwd_host_ms:
            return "gpu_drain"
        if dev < 0.5 * self.fwd_host_ms:
            return "sync_wait"
        return "host"

    def tick_end(self) -> None:
        self.last_total = time.perf_counter() - self.t0
        self.n += 1
        dt = self.last_total
        # The cold tier's mmap files measure themselves (a disk read or write has
        # no tick counter to mark against) and the engine drains them here, so
        # disk IO lands in the tick that paid it instead of hiding inside a RAM
        # bucket. Before the totals below, so a slow tick's print includes it.
        eng = self._eng()
        cold = getattr(getattr(eng, "_kv", None), "cold", None) if eng is not None else None
        if cold is not None:
            self.charge_ms("ssd_mmap", cold.drain_ssd_ms())
        for k, v in self.cur.items():
            self.tot[k] = self.tot.get(k, 0.0) + v
            self.count[k] = self.count.get(k, 0) + 1
        if dt > self.slow_s:
            parts = " ".join(f"{k}={v * 1000:.0f}ms" for k, v in self.cur.items())
            extra = f" [{self.note}]" if self.note else ""
            # dec/pre let a log analysis keep decode-only ticks and drop chunked
            # prefill ticks; with SLOW_MS=0 every tick prints, so this is the
            # phase tag for the whole-tick distribution.
            phase = f" dec={self.phase_dec} pre={self.phase_pre}"
            tail = self._slow_tail(dt * 1000)
            print(
                f"[step-timing] tick {self.n} total={dt * 1000:.0f}ms{phase} {parts}"
                f"{extra} {tail}",
                file=sys.stderr,
                flush=True,
            )

    def _slow_tail(self, dt_ms: float) -> str:
        """Device span + allocator deltas for one slow forward. Never syncs:
        elapsed_time runs only when query() says the end event already completed
        (the tick drained its stream inside the measured region)."""
        base = (
            f"path={self.fwd_path} sparse={int(self.fwd_sparse)} fwd_host={self.fwd_host_ms:.0f}ms"
        )
        if not self.cuda:
            self.last_why = "cpu"
            return f"{base} why=cpu"
        dev = None
        if self.ev_e is not None and self.ev_e.query():
            dev = self.ev_s.elapsed_time(self.ev_e)
        m1 = self._mem_snap()
        # Cumulative counters -> tick deltas. reserved_bytes.all.current is a
        # GAUGE (bytes held now), not cumulative: print its absolute post-forward
        # level next to free=, not a delta.
        d = {
            k: m1.get(k, 0) - self.mem0.get(k, 0)
            for k in _MEM_KEYS
            if k != "reserved_bytes.all.current"
        }
        seg_alloc = d["segment.all.allocated"]
        seg_free = d["segment.all.freed"]
        eng = self._eng()
        free_mib = eng._device_free_limit().get("device_free_bytes", 0) >> 20 if eng else 0
        reserved_mib = m1.get("reserved_bytes.all.current", 0) >> 20
        why = self._classify(dt_ms, dev, d)
        self.last_why = why
        self.fwd_gpu_ms = dev
        if dev is None:
            gpu = "fwd_gpu=pending"
            stall = ""
        else:
            gpu = f"fwd_gpu={dev:.0f}ms"
            stall = f" stall={max(0.0, self.fwd_host_ms - dev):.0f}ms"
        return (
            f"free={free_mib}MiB reserved={reserved_mib}MiB {base} {gpu}{stall} "
            f"d_malloc={d['num_device_alloc']} d_free={d['num_device_free']} "
            f"seg+={seg_alloc} seg-={seg_free} retries={d['num_alloc_retries']} "
            f"sync_streams={d['num_sync_all_streams']} oom={d['num_ooms']} "
            f"reject={d['num_oom_rejections']} alloc_conf={self.alloc_conf} why={why}"
        )

    def report(self) -> None:
        if not self.n:
            return
        parts = " ".join(
            f"{k}={self.tot[k] / self.count[k] * 1000:.1f}ms" for k in sorted(self.tot)
        )
        print(f"[step-timing] {self.n} ticks avg: {parts}", file=sys.stderr, flush=True)


class FatalDeviceError(RuntimeError):
    """A device error the process cannot serve through: an allocator OOM past the
    held memory fraction. The fraction is fixed for the process, so freeing the
    rows in flight cannot make room — continuing leaves a half-dead server that
    drains to idle and answers /health 200. The only recovery is a process restart.
    """


#: Exit code the #652 supervisor treats as a fatal-marker restart (its guard exits
#: 11 on a CUDA marker); printed so the grep convention matches.
FATAL_DEVICE_EXIT_CODE = 11
FATAL_DEVICE_MARKER = "FATAL device error (out of memory): restarting process"


def fatal_device_exit(exc: BaseException) -> None:
    """Terminate NOW on an unrecoverable device OOM. os._exit, not raise: the
    daemon loop's log-and-continue would otherwise swallow it, and SystemExit is
    caught by `except Exception`. The marker is the #652 supervisor's restart
    trigger. Module-level so a CPU gate monkeypatches it instead of exiting itself.
    """
    print(FATAL_DEVICE_MARKER + f": {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    os._exit(FATAL_DEVICE_EXIT_CODE)


class RequestFailed(RuntimeError):
    """A request ended in failure. ``reason`` is a stable tag for the failure
    class (None = untagged); a caller that tolerates one class catches on it.
    ``poll``/``take`` raise this instead of handing the failure back as data."""

    def __init__(self, request_id: int, reason: str | None, message: str):
        super().__init__(f"request {request_id} failed: {message}")
        self.request_id = request_id
        self.reason = reason


class EngineOverloaded(RuntimeError):
    """submit() refused because the engine already holds ``max_inflight`` live
    requests (running + waiting). Raised synchronously, before the request is
    enqueued, so the caller can back off instead of holding a slot in an
    unbounded queue until its own deadline. A RuntimeError so the existing 503
    mapping contains it; routes map it to 503 with the cap in the error body."""


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
    #: Cap on live requests (running + waiting). None = unbounded (submit never
    #: refuses for queue depth). build_engine passes the _INFLIGHT_AUTO sentinel
    #: for its serving default (twice usable_slots): one running wave plus one
    #: queued wave, enough to keep every slot saturated (a freed slot is
    #: refilled from the queue in the same step). A deeper queue cannot raise
    #: throughput and only grows per-request host RAM and hides head-of-line
    #: latency, so an over-capacity submit raises EngineOverloaded.
    max_inflight: Any = None  # int cap, None (unbounded), or _INFLIGHT_AUTO


#: Sentinel for "derive max_inflight from usable_slots (two waves)". A distinct
#: object so None (unbounded) and any int cap keep their own meaning.
_INFLIGHT_AUTO = object()


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
        device_reserve_bytes: int = 0,
        reserve_dropped_blocks: int = 0,
        sparse_headroom_bytes: int = 0,
        sparse_headroom_dropped_blocks: int = 0,
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
        self._sparse: SparseRuntime | None = None
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
        self._hybrid_dense_wall = 0.0  # dense wall time in the current window
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
        self._sparse_prefill_cap = sparse_prefill_tokens or (192 if sparse_min_tokens else 0)
        #: decode ticks build the packed table with pure device selection (no host
        #: sync) — the capture-ready path; valid only with the pin steady state.
        sparse_device_on = sparse_device_select and sparse_tracker is not None
        self._prefix = prefix_store
        #: KvBootStore for cold-start KV (--kv-store); None = no on-disk boot context.
        self._boot = boot_store

        self._decode_graph_on = _graph_on(backend, decode_graph)
        self._decode_graphs: dict = {}
        # Sparse decode ticks use a separate capture (packed [selected;own] table,
        # not the dense table). On when sparse device selection is enabled; on a
        # CUDA backend that also requires the decode graph on (sm70 excluded by
        # _graph_on), on CPU the device-select path runs the eager CPU test seam.
        # Hybrid mode runs sparse ticks EAGER on purpose: it needs only the dense
        # precaptured graph, so the sparse capture (and its warmup-frame hazard)
        # is not required; eager sparse is token-exact on sm70.
        #
        # Guard A (#805): on CUDA the captured sparse graph is only correct on
        # the d=0 single-token path. With speculation (spec_depth>=1) the
        # width-2 captured verify replayed trunk logits/hidden that disagreed
        # with eager and drafts stopped accepting; the keep_steps=W fix (#805,
        # PR #808) was then measured correct on sm70 by the 09-17 first-replay
        # value gate, so sm70 arms the capture under speculation. Other CUDA
        # archs stay guarded: unverified is not the same as fixed. Scope is the
        # arch list in _sparse_capture_allowed, not the device type alone; the
        # CPU cell uses the CpuSparseGraph eager reference recording, which is
        # token-exact at W=2 and stays enabled (the W=2 CPU gate is the oracle
        # for the width-2 root-cause triage).
        cuda_spec_guard = not _sparse_capture_allowed(backend, draft is not None, spec_depth)
        if cuda_spec_guard:
            warnings.warn(
                "sparse decode graph auto-disabled with speculation on CUDA "
                f"(spec_depth={spec_depth}, arch="
                f"{getattr(backend, 'arch', '?')!r}): the captured sparse "
                "width-2 verify is verified only on sm70 (#805, PR #808); this "
                "arch is unmeasured, so sparse decode runs eager. CPU reference "
                "is unaffected.",
                stacklevel=2,
            )
        sparse_graph_on = (
            sparse_tracker is not None
            and not self._sparse_min_tokens
            and sparse_device_on
            and (self._decode_graph_on or backend.device.type != "cuda")
            and not cuda_spec_guard
        )
        if sparse_tracker is not None:
            self._sparse = SparseRuntime(sparse_tracker, sparse_device_on, sparse_graph_on)
        # A replay's padding rows write to both pools, so they need a slot and a
        # block of their own. Reserved here, not on the first tick that pads:
        # ``build_engine`` sized the pools for this row, and taking it up front
        # keeps the capacity the caller asked for whole instead of removing one
        # request's worth of it partway through a run.
        self._graph_capture = GraphCapture(
            state_pool.alloc_slot, kv_pool.alloc_block, reserve=self._decode_graph_on
        )
        # Resolve the in-flight cap against usable_slots (known only after the
        # pad row is reserved). build_engine passes the AUTO sentinel for its
        # two-wave default; an explicit int is honored; None stays unbounded.
        if limits.max_inflight is _INFLIGHT_AUTO:
            limits = replace(limits, max_inflight=2 * self.usable_slots)
        self.limits = limits
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
            if self._graph_capture.pad_slot is not None:
                remedy += " (it adds the decode graph's pad row itself)"
            warnings.warn(
                f"{self.usable_slots} usable state slots against max_batch="
                f"{limits.max_batch}: a slot is held from submit to finish, so "
                f"concurrency is capped at {self.usable_slots} and the excess queues "
                f"into later ticks rather than raising -- twice the ticks at half the "
                f"width, not an error. To run {limits.max_batch} rows at once, "
                f"{remedy}, or size a LinearStatePool for "
                f"{limits.max_batch + (self._graph_capture.pad_slot is not None)} directly "
                f"(this one holds {self._states.num_slots})",
                stacklevel=2,
            )

        self._draft = draft
        self._aux_layers = draft.aux_layers if draft is not None else ()
        self._width = 1  # verify tick width: 1 committed token + width-1 drafts
        if draft is not None:
            if not hasattr(draft, "step"):
                raise TypeError(
                    f"draft head {type(draft).__name__} is not a drafter: it has no "
                    f"step(rows). See the contract in spec.py."
                )
            draft.set_depth(spec_depth)
            self._width = draft.width
            if not 1 < self._width <= BLOCK_TOKENS:
                raise ValueError(f"verify width must be in (1, {BLOCK_TOKENS}], got {self._width}")
            if self._width > backend.max_verify_width:
                raise ValueError(
                    f"verify width {self._width} exceeds the verify tile's "
                    f"{backend.max_verify_width}: "
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

        # Device reserve was applied in build_engine: the KV tensor was sized once
        # (smaller) before it existed, so here we only record what build cut. It is
        # a peak-live reduction, not a held reservation.
        self._device_reserve_bytes = int(device_reserve_bytes)
        self._reserve_dropped_blocks = int(reserve_dropped_blocks)
        self._sparse_headroom_bytes = int(sparse_headroom_bytes)
        self._sparse_headroom_dropped_blocks = int(sparse_headroom_dropped_blocks)

        self._pin = backend.device.type == "cuda"
        self._lock = threading.RLock()
        #: Set by an unrecoverable device OOM; once set the loop terminates and
        #: submit() refuses new rows (the process is on its way to a restart).
        self._fatal: BaseException | None = None
        self._step_timing = _StepTiming(self) if os.environ.get("TILERL_STEP_TIMING") else None
        if self._step_timing is not None:
            atexit.register(self._step_timing.report)
            # The cold tier's spill files measure themselves; hand them the timer
            # here, after build_engine created the tier without one.
            cold = getattr(self._kv, "cold", None)
            if cold is not None:
                cold.step_timing = self._step_timing
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        #: Published by the loop so `stats()` never takes the lock a forward holds.
        self._stats_snapshot: dict[str, Any] | None = None
        #: perf_counter() of the last step tick that actually advanced (a
        #: non-idle forward completed). Read by liveness(): a wedged device
        #: forward never returns, so this timestamp stops moving while requests
        #: stay running -- the only signal that distinguishes a live server from
        #: one frozen inside a kernel (stats() alone keeps serving the last
        #: snapshot and looks healthy).
        self._last_progress_ts: float = time.perf_counter()
        #: Memoized dense ledger: weights/pools/slots are static after build, and
        #: _build_stats (every tick, plus admit ticks twice) used to re-walk every
        #: param tensor per tick.
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
        #: spec followers that adopted a WARM entry (draft K/V + boundary hidden);
        #: the SparseRuntime owns the count, stats reads it through the facade.
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
        #: Diagnostic only: populated when TILERL_DRAFT_TIMING is on, None keeps
        #: the path unchanged. Entries ``(forwards_delta, gpu_ms, max_seq_len)``
        #: — the seq_len dimension tests whether draft GPU time is a fixed cost or
        #: scales with the dense prefix the one decode forward reads. Split from
        #: TILERL_STEP_TIMING: this probe synchronizes the device around every
        #: draft step, so it must not arm together with the near-zero wall timer.
        self._draft_ms: list[tuple[int, float, int]] | None = (
            [] if os.environ.get("TILERL_DRAFT_TIMING") else None
        )
        self._finished_logprobs: dict[int, list[float]] = {}
        self._taken_logprobs: set[int] = set()
        self._last_logprobs: list[float] | None = None

        if self._sparse is not None:
            # The only Engine surface a sparse tick touches: immutable pools and
            # the named callbacks that mutate engine rows/counters.
            self._sparse.ctx = SparseCtx(
                model=model,
                backend=backend,
                kv=kv_pool,
                states=state_pool,
                draft=draft,
                width=self._width,
                aux_layers=self._aux_layers,
                max_batch=limits.max_batch,
                graph_capture=self._graph_capture,
                verify=self._verify,
                sample_commit=self._sample_commit,
                draft_step=self._draft_step,
                bump_decode_forwards=self._bump_decode_forwards,
                step_timing=self._step_timing,
            )

    # ------------------------------------------------------------------ API

    @property
    def _prefix_warm_adoptions(self) -> int:
        return self._sparse.warm_adoptions if self._sparse is not None else 0

    def _sparse_prefix_stats(self) -> dict:
        """Sparse KV's own prefix-index counters, namespaced sparse_prefix_* so
        they are never confused with the dense PrefixStore prefix_* fields: a
        sparse build runs NoPrefixStore, whose dense counters stay at their empty
        values and otherwise read as a misleading +published/+entries."""
        if self._sparse is None or self._sparse.tracker.prefix is None:
            return {}
        s = self._sparse.tracker.prefix.stats()
        return {
            "sparse_prefix_published": s["published"],
            "sparse_prefix_hits": s["hits"],
            "sparse_prefix_evictions": s["evictions"],
            "sparse_prefix_entries": s["live_entries"],
            "sparse_prefix_entries_capacity": s["entries_capacity"],
            "sparse_prefix_warm_adoptions": self._sparse.warm_adoptions,
        }

    @property
    def _sparse_graphs(self) -> dict:
        return self._sparse.graphs if self._sparse is not None else {}

    @property
    def _sparse_device_select(self) -> bool:
        return self._sparse is not None and self._sparse.device_select

    @property
    def _sparse_ticks_since_refresh(self) -> int:
        return self._sparse.ticks_since_refresh if self._sparse is not None else 0

    @property
    def _sparse_graph_on(self) -> bool:
        return self._sparse is not None and self._sparse.graph_on

    @_sparse_graph_on.setter
    def _sparse_graph_on(self, value: bool) -> None:
        if self._sparse is not None:
            self._sparse.graph_on = value

    def _bump_decode_forwards(self) -> None:
        self._decode_forwards += 1

    @property
    def usable_blocks(self) -> int:
        """KV blocks a request may have. The pools are sized one larger than the
        caller asked for when the captured tick is on, and that row is the
        engine's — every capacity answer is net of it, or a request sized to the
        whole pool passes the guard and fails on the allocation behind it."""
        return self._kv.num_blocks - (self._graph_capture.pad_block is not None)

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
        return self._states.num_slots - (self._graph_capture.pad_slot is not None)

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
        if self._fatal is not None:
            # The process is exiting for a supervisor restart after a device OOM;
            # never take another row (a half-dead server must not look healthy).
            raise FatalDeviceError(f"engine is fatally failed: {self._fatal}")
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

            # Bound the queue: count rows actually occupying a slot or queued for
            # one. max_inflight None = unbounded. Refuse BEFORE enqueue so an
            # over-capacity client backs off now instead of pinning its prompt in
            # a deep queue to a deadline.
            cap = self.limits.max_inflight
            if cap is not None and len(self._running) + len(self._waiting) >= cap:
                raise EngineOverloaded(
                    f"engine is saturated: {len(self._running) + len(self._waiting)} "
                    f"in-flight requests and the cap is {cap} (running + waiting); "
                    f"retry later"
                )

            # Unallocated: allocating here refuses permanently, since `submit` has no later
            # tick to retry on. The prefix match moves to `_admit` with the allocation.
            # Hybrid sparse engine: short prompts run dense and pin their whole context
            # (no sparse sharing); a prompt that could never fit that pin even with the
            # pool empty routes sparse instead of queueing on an impossible admit.
            sparse_on = self._sparse is not None and (
                self._sparse_min_tokens == 0 or len(tokens) > self._sparse_min_tokens
            )
            if (
                not sparse_on
                and self._sparse is not None
                and params.max_new_tokens > 0
                and self._kv.blocks_for_tokens(total + self._width - 1)
                > self._kv.num_blocks - (self._graph_capture.pad_block is not None)
            ):
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
            # Idle->active edge: refresh here too, else the submit-to-first-tick
            # gap after a long idle reads as stall. Later submits must not refresh:
            # an unadmitted backlog that old is genuinely stuck.
            if not self._running and not self._waiting:
                self._last_progress_ts = time.perf_counter()
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
            _tm = self._step_timing
            slots_before = self._slots_used
            if _tm is not None:
                _tm.tick_start()
                _t = time.perf_counter()
            decodes, prefills, chunks = self._build_plan()
            if _tm is not None:
                _tm.phase_dec = len(decodes)
                _tm.phase_pre = len(prefills)
            if not decodes and not prefills:
                idle = True
            else:
                # Mark the START of an active tick too, not only its end: a long
                # idle gap must not count as stall. Without this, the first
                # request after a quiet period read active=True with stuck = the
                # whole idle gap and 503'd until its prefill finished. A forward
                # that never returns still trips: this moves only once at tick
                # start, the clock then stays frozen for the stuck duration.
                self._last_progress_ts = time.perf_counter()
                if _tm is not None:
                    _tm.mark("plan", _t)
                    _t = time.perf_counter()
                # Publish before the forward only when state changed since the last
                # tick's end snapshot: THIS tick admitted rows, or submit() queued
                # new waiters. Otherwise the end snapshot is current, and rebuilding
                # doubles the per-tick stats cost. A first tick always admits (and a
                # long multi-chunk prefill admits on its first tick), so stats() never
                # falls back to its locking path while a forward holds this lock
                # (test_health_does_not_wait_on_the_engine_lock).
                admitted = self._slots_used > slots_before
                prev_waiting = (self._stats_snapshot or {}).get("waiting")
                if admitted or prev_waiting != len(self._waiting):
                    self._stats_snapshot = self._build_stats()
                tick_sparse = bool((decodes + prefills) and (decodes + prefills)[0].sparse_on)
                if _tm is not None:
                    _tm.mark("stats", _t)
                    _t = time.perf_counter()
                    _tm.fwd_start()
                try:
                    self._hybrid_t0 = time.perf_counter()
                    self._run_forward(decodes, prefills, chunks)
                except torch.cuda.OutOfMemoryError as exc:
                    # Fatal, and deliberately NOT _finish: draining the rows would
                    # move the engine to idle so /health answered 200 while the
                    # process kept swallowing OOMs. Raise into the loop, which marks
                    # the engine failed and terminates for the supervisor to restart.
                    raise FatalDeviceError(str(exc)) from exc
                except Exception as exc:
                    for req in list(self._running):
                        self._finish(req, error=str(exc))
                    raise
                finally:
                    if _tm is not None:
                        _tm.mark("forward", _t)
                        _t = time.perf_counter()
                        _tm.fwd_end()
                    # `_loop` stops calling `step` once nothing runs, so this carries the last
                    # tick's state -- including a failed forward's, hence `finally`.
                    self._stats_snapshot = self._build_stats()
                if _tm is not None:
                    _tm.mark("stats", _t)
                    _t = time.perf_counter()
                self._hybrid_charge(tick_sparse)
                if _tm is not None:
                    _tm.mark("charge", _t)
                    _tm.tick_end()
                # A tick that ran a forward and returned is progress. Updated
                # INSIDE the lock alongside the snapshot: a forward stuck in the
                # kernel never reaches here, so liveness() sees a frozen clock.
                self._last_progress_ts = time.perf_counter()
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
        matched, hit_blocks, snap = (0, (), None) if sparse else self._match_prefix(req.tokens)
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
            # The store is loud on purpose (a corrupt k.bin/truncated aux.pt raises);
            # an admit path must turn that into a miss, or step()'s handler fails every
            # running request for one bad on-disk entry. The types are what torch.load
            # and the file/manifest/CRC reads actually raise — no broad Exception.
            try:
                loaded = self._boot.boot(req.tokens[:boot_len], self._kv)
            except (OSError, EOFError, pickle.UnpicklingError, RuntimeError):
                loaded = None
            if loaded is None:
                # Corrupt/vanished entry: admit as a normal prefill miss. Not
                # `return False` — exists() is still true, so the next _admit
                # tick would fail the same way forever.
                boot_len = 0
                needed = total_blocks - len(hit_blocks)
                if self._kv.free_blocks < needed:
                    return False
            else:
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
            if (
                entry is not None
                and self._draft is not None
                and (
                    entry.get("hidden") is None
                    or any(self._kv.cold.share_take_field(k, "dk") is None for k in entry["keys"])
                )
            ):
                # field probes avoid pinning the whole trunk blob to check warmness
                entry = None
            if entry is not None:
                matched = len(entry["tokens"])
                req.seq_len = matched
                if matched == len(req.tokens) and matched % BLOCK_TOKENS == 0:
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
                    self._sparse.warm_draft(req, entry, matched)
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

    def liveness(self, stuck_after_s: float) -> tuple[bool, float]:
        """Whether the step loop is advancing, and how long it has been stuck.

        Returns (live, stuck_secs). An IDLE engine (no running or waiting
        request) is always live: a quiet server must not report unhealthy. With
        active requests, live means a non-idle tick STARTED (or finished) within
        ``stuck_after_s`` -- the timestamp is refreshed at tick start so a long
        idle gap before the first request does not read as stall, and at tick
        end; a forward that never returns leaves it frozen past the threshold.
        Read without the lock: the timestamp is a single float write, and a
        wedged forward holds the lock anyway, so taking it here would block
        /health on the exact stall it must detect.
        """
        active = bool(self._running) or bool(self._waiting)
        if self._fatal is not None:
            # A fatally failed process must read dead even if its rows were never
            # drained, in the window before fatal_device_exit runs; never idle-200.
            return False, float("inf")
        if not active:
            return True, 0.0
        stuck = time.perf_counter() - self._last_progress_ts
        return stuck <= stuck_after_s, stuck

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

    def _device_free_limit(self) -> dict[str, int]:
        """This process's allocator free/total in bytes (torch.cuda.mem_get_info).
        Under set_per_process_memory_fraction the total is the capped process limit
        and free is the held reserve headroom; off cuda both are 0 (no device).
        Pure read; never takes the model lock path."""
        if self._backend.device.type != "cuda":
            return {"device_free_bytes": 0, "device_limit_bytes": 0}
        free, total = torch.cuda.mem_get_info(self._backend.device)
        return {"device_free_bytes": int(free), "device_limit_bytes": int(total)}

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
                # Build-time device headroom floor (--device-reserve-mib) and how
                # many KV blocks it trimmed; 0/0 means the reserve did not bind.
                "device_reserve_bytes": self._device_reserve_bytes,
                "reserve_dropped_blocks": self._reserve_dropped_blocks,
                # Sparse --device-headroom-mib: floor kept by building a smaller hot
                # pool, and how many blocks it cost; 0/0 = off (ceiling pool).
                "sparse_headroom_bytes": self._sparse_headroom_bytes,
                "sparse_headroom_dropped_blocks": self._sparse_headroom_dropped_blocks,
                # In-process allocator view, NOT nvidia-smi: under a memory
                # fraction these are capped to the process limit, so free is the
                # reserve headroom the process actually has (physical card free
                # always differs by driver/context bytes outside the fraction).
                **self._device_free_limit(),
                "slots_used": self._slots_used,
                "slots_total": self.usable_slots,
                # DENSE PrefixStore counters (self._prefix). A sparse-KV build
                # runs NoPrefixStore, so these read as the empty store (0/0) by
                # design; the sparse index's counters are the sparse_prefix_*
                # fields below, not these.
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
                # sparse-KV's OWN prefix-index counters (empty set on dense builds)
                **self._sparse_prefix_stats(),
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
                **(
                    self._sparse_live_stats()
                    if self._sparse is not None and self._sparse_min_tokens
                    else {}
                ),
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
        return held_storage(
            cfg=self._model.cfg,
            model_params=self._model.params,
            kv=self._kv,
            states=self._states,
            sparse=self._sparse,
            sparse_min_tokens=self._sparse_min_tokens,
            draft_kv=getattr(self._draft, "kv", None),
            running=self._running,
            sparse_graph_bytes=self._sparse_graph_bytes,
        )

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
        peak = measured_peak_bytes(self._backend)
        return peak if peak is not None else sum(self._held_storage().values())

    def _memory_rows(self) -> list[dict]:
        """The unified device ledger. Static dense/hybrid ledgers are memoized on
        param count (built once, #460); pure-sparse and dense-with-cold-or-boot
        rebuild every tick because they read live residency."""
        n_params = len(self._model.params)
        if self._mem_rows is not None and self._mem_rows[0] == n_params:
            return self._mem_rows[1]
        held = self._held_storage()
        peak = self._measured_peak_bytes()
        draft_pool = getattr(self._draft, "kv", None)
        rows = memory_rows(
            cfg=self._model.cfg,
            model_params=self._model.params,
            kv=self._kv,
            states=self._states,
            held=held,
            peak_bytes=peak,
            sparse=self._sparse,
            sparse_min_tokens=self._sparse_min_tokens,
            running=self._running,
            draft_kv=draft_pool,
            # A DFlash2 block drafter is attached as self._draft but has NO kv
            # pool; only a pool-carrying chain draft adds a derived draft_pool.
            draft_layers=0 if draft_pool is None else self._draft.cfg.num_layers,
            boot=self._boot,
            sparse_graph_bytes=self._sparse_graph_bytes,
            sparse_graph_count=lambda: len(self._sparse_graphs),
        )
        hybrid = self._sparse is not None and self._sparse_min_tokens
        if self._device_reserve_bytes:
            # Held by the process memory fraction set in cmd_serve, not by a KV
            # allocation, so it stays a budget row (out of the peak = Σstatic +
            # transient invariant). The fraction caps mem_get_info and turns an
            # over-fence cudaMalloc into catchable OOM; build cuts no blocks.
            rows.append(
                {
                    "tier": "device",
                    "owner": "device_reserve",
                    "kind": "budget",
                    "derived": self._device_reserve_bytes,
                    "note": "held by set_per_process_memory_fraction; no KV blocks cut",
                    "measured": None,
                    "delta": None,
                }
            )
        if hybrid or (
            self._sparse is None and self._boot is None and getattr(self._kv, "cold", None) is None
        ):
            self._mem_rows = (n_params, rows)
        return rows

    def sparse_retier(self, keep: frozenset[int]) -> tuple[int, int]:
        return sparse_retier_pages(self._kv, keep, self._running, self._waiting)

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
        return self._sparse.recall(req_id, target_mass)

    def _sparse_rows(self, rows, seq_q, decodes):
        return self._sparse.build_rows(rows, seq_q, decodes)

    def _sparse_finalize(self, sf, rows, hidden=None) -> list:
        return self._sparse.finalize(sf, rows, hidden)

    def _sparse_process_offers(self, dropped_offers) -> None:
        self._sparse.process_offers(dropped_offers)

    def _sparse_decode_rows(self, decodes, q_dec):
        return self._sparse.decode_rows(decodes, q_dec)

    def _run_sparse_decode_graph(self, reqs, chains) -> bool:
        return self._sparse.run_decode_graph(reqs, chains)

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
        _tm = self._step_timing
        _t = 0.0
        if _tm is not None:
            _t = time.perf_counter()
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
            _decode_extra_blocks(r.seq_len, q, len(r.blocks)) for r, q in zip(decodes, q_dec)
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
            if _tm is not None:
                _tm.mark("graph", _t)
                _tm.fwd_path = "graph"
            self._hybrid_charge(False)
            return
        if (
            not prefills
            and decodes
            and tick_sparse
            and self._sparse_graph_on
            and self._run_sparse_decode_graph(decodes, chains)
        ):
            if _tm is not None:
                _tm.mark("graph", _t)
                _tm.fwd_path = "graph"
                _tm.fwd_sparse = True
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
            if _tm is not None:
                _tm.mark("sparse_select", _t)
                _tm.fwd_sparse = True
                # Diagnostic: sparse geometry on the slow-tick line, to catch a
                # cmax bucket recalc or a hot-selection slide that widens the
                # attention table (the sporadic 1.3-1.5s 32k ticks). All three
                # are host-side ints, so this adds no device sync.
                _table = getattr(sf, "table", None)
                _tm.note = (
                    f"sparse cmax={getattr(sf, 'cmax', '?')} "
                    f"own_w={getattr(sf, 'own_w', '?')} "
                    f"table_w={_table.shape[1] if _table is not None else '?'}"
                )
                _t = time.perf_counter()
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
        if _tm is not None:
            _tm.mark("prep", _t)
            _t = time.perf_counter()
        logits = self._model.forward(
            input_ids,
            positions,
            self._make_kv(rows, seq_q, width if chains else 0, sf if sparse else None),
            self._backend,
            hidden_out=hid,
            aux_layers=self._aux_layers,
            last_only=False if chains else seq_q,  # a verify tick needs every chain position
        )
        if _tm is not None:
            _tm.mark("model", _t)
            _t = time.perf_counter()
        if sparse:
            promote_ctx.__exit__(None, None, None)
            try:
                sparse_offers = self._sparse_finalize(sf, rows, None if hid is None else hid[-1])
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
            if _tm is not None:
                _tm.mark("sparse_finalize", _t)
                _t = time.perf_counter()
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
        if _tm is not None:
            _tm.mark("sample", _t)
            _t = time.perf_counter()
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
            # Grow every decode row's blocks to cover the draft's post-commit
            # write BEFORE draft.step, via the one planner shared with the
            # captured-graph path (#621). Also covers a row that left prefill
            # this tick (the growth loop above only handles `decodes`).
            rows = self.ensure_draft_write_blocks(rows)
            if _tm is not None:
                _tm.mark("draft_blocks", _t)
                _t = time.perf_counter()
            self._draft_step(rows)  # every tick, or a chunked prefill leaves the draft KV empty
            if _tm is not None:
                _tm.mark("draft_step", _t)
                _t = time.perf_counter()
            if sparse:
                # draft K/V for this tick's dropped pages now exist: publish them
                self._sparse_process_offers(sparse_offers)
        elif sparse:
            self._sparse_process_offers(sparse_offers)
        if _tm is not None:
            _tm.mark("offers_pub", _t)
            # Host-only page count (sparse_offers is a list of (row, [pages]));
            # aligns offers_pub with the per-page D2H publish count, not seq_len,
            # so a linear offers_pub (page-bound) is separable from a linear
            # draft_step (prefix-bound). No device read.
            if sparse and sparse_offers is not None:
                _tm.note += f" offers_pages={sum(len(p) for _, p in sparse_offers)}"

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
            self._model.cfg, self._sparse_k, self.limits.max_num_batched_tokens
        )
        resident = self._sparse.resident
        return sum(
            max(0, ceil - len(resident.get(r.req_id, ())))
            for r in self._running
            if r.sparse_on and r.phase != _PHASE_DONE
        )

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
        dt = (
            self._hybrid_fake_dt[1 if not sparse else 0]
            if self._hybrid_fake_dt is not None
            else time.perf_counter() - self._hybrid_t0
        )
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
                elif (
                    not pf.sparse_on
                    and len(pf.tokens) % BLOCK_TOKENS
                    and pf.prefill_from >= predicted
                ):
                    # Tail window [predicted, n): at most two aligned chunk ends, so
                    # this holds <=2 snapshots per ragged prompt and keeps only the
                    # deepest the actual schedule reached. Not predicted: a decode
                    # row sharing the tick shifts the walk past the n-only value.
                    # Published at completion -- at most 32 tokens later, one tick.
                    pf.pending_prefix = (
                        pf.prefill_from,
                        (
                            self._states.states[pf.state_slot].clone(),
                            self._states.window_snapshot(pf.state_slot),
                        ),
                    )
        if not done:
            return
        self._sample_commit(done)
        for pf, _, _ in done:
            # The state slot still covers exactly the prompt, so the snapshot is exact.
            prompt_len = len(pf.tokens) - len(pf.output)
            if not pf.sparse_on and pf.phase != _PHASE_DONE and prompt_len % BLOCK_TOKENS == 0:
                self._publish_prefix(pf, prompt_len)
            elif not pf.sparse_on and pf.phase != _PHASE_DONE and pf.pending_prefix is not None:
                # Ragged prompt: the held boundary snapshot is exact and its
                # blocks are still live; insert it at completion.
                pos, snap = pf.pending_prefix
                self._prefix_published += self._prefix.insert(
                    pf.tokens[:pos], pf.blocks[: pos // BLOCK_TOKENS], snap
                )
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
        return graph_bucket(rows, self.limits.max_batch)

    def _graph_for(self, B: int, W: int, keep: bool) -> Any | None:
        """The (B, W) graph, capturing it on first use. None (and graphs off) if
        capture fails, so the caller runs eager."""
        g = self._decode_graphs.get((B, W))
        if g is not None:
            return g
        g, self._graph_capture.pool, err = make_decode_graph(
            self._model,
            self._backend,
            self._kv,
            self._states,
            B,
            W,
            keep,
            self._aux_layers,
            self._graph_capture.pool,
        )
        if g is None:
            warnings.warn(err)
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
        if n < B and not self._graph_capture.ensure_pad():
            # no pad row: an exact-size graph is off the graph_keys grid and would
            # capture mid-request
            return False
        g = self._graph_for(B, W, keep=bool(chains))
        if g is None:
            return False
        pad = self._graph_capture.pad
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
            # Grow to cover the post-commit write via the SAME planner the eager
            # path uses: pre-fork growth covers the verifier chain but is one
            # block short when a commit lands exactly on a 16-boundary (#621).
            reqs = self.ensure_draft_write_blocks(reqs)
            if not reqs:
                return True
            self._draft_step(reqs)
        return True

    def ensure_draft_write_blocks(self, rows: list[_Req]) -> list[_Req]:
        """Grow each decode row's blocks so they cover the draft's post-commit
        write ``hi = seq_len-1`` (plus, for a sparse row, the verifier tail),
        immediately before ``draft.step``. One shared planner for the eager and
        captured-graph paths: the graph path used to rely on pre-fork growth
        sized for the verifier chain, which is one block short when a commit
        lands exactly on a 16-token boundary (#621). Mutates ``rows`` in place,
        dropping rows the pool cannot fit (failed alone, ``pool_exhausted``);
        returns the same list."""
        if self._draft is None:
            return rows
        kept = rows
        for r in list(rows):
            if r.sparse_on:
                # The dense draft KV span is fully RESERVED at admit (prompt +
                # max_new + verify width bound), so the blocks already exist.
                # This is a bound check, not a grow: allocating here would race
                # another row for the shared draft pool.
                end = r.seq_len - 1 + self._width - 1
                assert len(r.draft_blocks) * BLOCK_TOKENS > end, (
                    f"draft needs position {end} but admit reserved {len(r.draft_blocks)} blocks"
                )
                continue
            need = max(0, (r.seq_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS - len(r.blocks))
            if need > self._kv.free_blocks:
                # By count, not catching alloc_block's raise: an exception out of
                # here reaches step()'s handler and fails EVERY running request.
                self._finish(
                    r,
                    error=f"PagedKvPool exhausted: need {need} block(s), "
                    f"{self._kv.free_blocks} free",
                    reason="pool_exhausted",
                )
                kept.remove(r)
                continue
            while len(r.blocks) * BLOCK_TOKENS <= r.seq_len - 1:
                r.blocks.append(self._kv.alloc_block())
                r.own_blocks += 1
                self._blocks_used += 1
        return kept

    def _draft_step(self, rows: list[_Req]) -> None:
        """One named tick-step binding for the draft head: timed when the engine
        was built with draft timing on, plain otherwise. Every forward path calls
        this so the graph and eager ticks stay one call site."""
        if self._draft_ms is None:
            self._draft.step(rows)
        elif torch.cuda.is_available():
            self._draft_step_timed(rows)
        else:
            # CPU/deviceless host with TILERL_DRAFT_TIMING on: wall-clock
            # fallback, no CUDA event. The diagnostic target is the GPU serve only.
            t0 = time.perf_counter()
            max_seq = max((r.seq_len for r in rows), default=0)
            self._draft.step(rows)
            self._draft_ms.append(
                (self._draft.forwards, (time.perf_counter() - t0) * 1000, max_seq)
            )

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
        max_seq = max((r.seq_len for r in rows), default=0)
        a.record()
        self._draft.step(rows)
        b.record()
        b.synchronize()
        gpu_ms = a.elapsed_time(b)
        self._draft_ms.append((self._draft.forwards - f0, gpu_ms, max_seq))
        print(
            f"[draft-timing] fwd={self._draft.forwards - f0} gpu={gpu_ms:.2f}ms max_seq={max_seq}",
            file=sys.stderr,
            flush=True,
        )

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
                cap is not None
                and not r.thought_closed
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
        """Give back the blocks and the slot. Here, not at poll, so capacity returns now.

        The sub-segments below split the request-end half of the step timer's
        coarse "sample" bucket: a long request ending inside one tick is what
        puts seconds there while ``model`` stays at its steady ~165 ms.
        """
        _tm = self._step_timing
        if _tm is not None:
            _t = time.perf_counter()
        req.phase = _PHASE_DONE
        if req.state_slot is None:
            return  # never admitted; blocks and slot are taken together in `_admit`
        if req.sparse_on and self._sparse is not None:
            # A SUCCESSFUL finish of a row that did NOT itself adopt a prefix
            # synchronously publishes its prompt prefix while frames/blobs/
            # snapshots are live, so the next same-prefix follower can adopt
            # (#796). A row that already adopted (sparse_matched > 0) does not
            # re-publish: it would only add a redundant frozen entry onto the
            # #793 capacity path for zero new bytes. A cancelled/failed row
            # publishes nothing (failed is set before _release).
            if not req.failed and req.sparse_matched == 0 \
                    and self._sparse.prefix is not None:
                self._sparse.publish_at_finish(req)
            # Drop this request's host-held cold blobs, keyed (req, logical
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
        if _tm is not None:
            _tm.mark("release_cold_forget", _t)
            _t = time.perf_counter()
        if req.draft_blocks:
            # Sparse+spec: dense draft KV lives in the draft pool's own id space.
            dpool = self._draft.kv
            for b in req.draft_blocks:
                dpool.free_block(b)
        for b in req.blocks:
            self._kv.free_block(b)
        if _tm is not None:
            _tm.mark("release_blocks", _t)
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
                    # Marker for a row that did not finish; request end publishes
                    # nothing regardless (publish-once, #782).
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
                except FatalDeviceError as exc:
                    # Unrecoverable allocator OOM: stop accepting work and exit so
                    # the supervisor restarts. Recorded before the seam so a gate
                    # monkeypatching fatal_device_exit still sees the engine failed.
                    with self._lock:
                        self._fatal = exc
                    self._wake.set()
                    fatal_device_exit(exc)
                    return
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
