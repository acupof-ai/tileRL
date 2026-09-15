"""Training on the hand-written tape: SFT, GRPO, on-policy distillation and
pretrain share ``_step``; serving and training share the model and weights.
# ponytail: CE is torch-eager glue; fold into a backend cross_entropy op when perf demands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from . import ledger as _ledger
from .autograd import AdamW, RecordingBackend, Tape, clip_grad_norm, cosine_warmup
from .build import QWEN38_SOURCE as _QWEN38_SOURCE
from .build import build_model
from .engine import RequestFailed, SamplingParams
from .eval import MATCHERS
from .kv_cache import BLOCK_TOKENS, LinearStatePool, NoPrefixStore, PagedKvPool
from .model import save_hf

_MAX_TICKS = 10000


def _sync(backend: Any) -> None:
    """Wait for the device before reading a clock. A phase timed without this
    bills its async kernels to whichever later phase happens to synchronize."""
    if getattr(backend, "device", None) is not None and backend.device.type == "cuda":
        torch.cuda.synchronize()


def _drain(engine: Any, ids: list[int], what: str) -> dict[int, list[int]]:
    """Tick until every id has finished. Accumulates: poll() only returns the
    requests that finished on that tick, so a single assignment loses the rest.

    A pool-exhaustion failure ends ONE rollout, not the step: the dead id gets an
    empty completion, which the live mask (``len(c) > 0`` in grpo_loop) drops, and
    the group trains on the rest. Every other failure class propagates -- catching
    it here would turn any bug into a silently missing row.
    """
    done: dict[int, list[int]] = {}
    for _ in range(_MAX_TICKS):
        engine.step()
        try:
            done.update(engine.poll())
        except RequestFailed as exc:
            if exc.reason != "pool_exhausted":
                raise
            done[exc.request_id] = []
        if all(i in done for i in ids):
            return done
    raise RuntimeError(f"{what}: did not finish within {_MAX_TICKS} ticks")


def _training_kv(model: Any, batch_size: int, seq_len: int, device: Any = None):
    """Dense KV: full-attn layers skip the paged pool so the tape's id() chain
    is unbroken; GDN layers still need a state slot per sequence."""
    cfg = model.cfg
    kv = SimpleNamespace()
    kv.dense = True
    kv.state_pool = LinearStatePool(
        num_slots=batch_size,
        num_linear_layers=cfg.num_linear_layers,
        num_heads=cfg.linear_num_value_heads,
        head_dim=cfg.linear_value_head_dim,
        device=device,
    )
    kv.state_slot = torch.arange(batch_size, dtype=torch.long)
    return kv


def _capture_kv(model: Any, seq_len: int, backend: Any):
    """A one-row PAGED pool for the frozen-base capture forward. The dense path
    materialises a [hq,t,t] score matrix (24 GiB at t=32k on the 27B); the
    serving prefill path streams key pages through PagedKvPool and keeps only
    the resident K/V (~2 GiB). Capture is a no-grad single prefill, so the tape
    does not need dense identity — paged is exact dense-causal attention."""
    cfg = model.cfg
    n_pages = (seq_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    pool = PagedKvPool(
        n_pages, cfg.num_kv_heads, cfg.head_dim,
        device=backend.device, layer_map=cfg.full_attn_layers,
        dtype=getattr(backend, "io", torch.bfloat16))
    blocks = torch.tensor([[pool.alloc_block() for _ in range(n_pages)]],
                          dtype=torch.long, device=backend.device)
    return SimpleNamespace(
        dense=False,
        kv_pool=pool,
        block_table=blocks,
        seq_len=torch.tensor([seq_len], dtype=torch.long, device=backend.device),
        seq_q_lens=torch.tensor([seq_len], dtype=torch.long, device=backend.device),
        state_pool=LinearStatePool(
            num_slots=1,
            num_linear_layers=cfg.num_linear_layers,
            num_heads=cfg.linear_num_value_heads,
            head_dim=cfg.linear_value_head_dim,
            device=backend.device,
        ),
        state_slot=torch.zeros(1, dtype=torch.long),
    )


_NO_GRAD = (
    "train_step: tape produced no parameter gradients — either the recording "
    "seam is missing (backend ops not recorded), or a trainable tensor is not "
    "the one the forward read: materialize() rebuilds any param whose "
    "device/dtype differs, and the new object has a new id()"
)

#: Check that every rank reduced its gradients in the same order. One extra
#: collective per step, so gates set it and training does not: a mismatch aborts
#: the process inside gloo, which is exactly the failure this would explain.
_CHECK_DP_ORDER = bool(os.environ.get("TILERL_CHECK_DP_ORDER"))

#: largest T measured to run with the MLP segment; above it the whole layer is the segment
#: ponytail: switches at the bottom of 1280..4352; measure both arms at 2048/3072 to move it
_MLP_SEGMENT_MAX_T = 1280


def _order_agrees(order: list[str], backend: Any) -> None:
    """Raise if the dp ranks did not all-reduce the same parameters in the same order.

    A collective pairs by call sequence, so two ranks disagreeing here meet on
    different tensors -- and gloo kills the process (``EnforceNotMet`` in
    ``pair.cc``) rather than raising, with no traceback near the loop. Comparing
    a hash costs one collective and turns that into a message.

    The hash is over the ORDERED KEY LIST, so it catches both failures: a
    different sequence, and a different key SET. ``sorted(params)`` agrees across
    ranks only because every rank holds the same keys -- a rank-conditional
    adapter breaks that, and nothing else would say so.

    Over the DP group, since that is the group the reduce ran on: ``all_gather``
    spans the tp group and would compare each rank only against its own replica.
    """
    import hashlib

    h = hashlib.sha256("\n".join(order).encode()).digest()[:8]
    mine = torch.tensor([len(order), *h], dtype=torch.float64)
    parts = backend.dp_all_gather(mine)
    if any(not bool((p == parts[0]).all()) for p in parts):
        counts = sorted({int(p[0].item()) for p in parts})
        raise RuntimeError(
            "dp ranks reduced different parameters, or in different orders "
            f"(counts across ranks: {counts}; {len(order)} here). A collective pairs by "
            "call sequence, so this would abort inside gloo rather than raise. Every rank "
            "must hold the same parameter keys and walk them in the same order.")


def _step(
    model: Any,
    input_ids: Any,
    backend: Any,
    optimizer: AdamW,
    trainable: dict[str, Any] | None,
    grad_fn: Any,
    micro: int = 0,
    timings: dict[str, float] | None = None,
    post_step: Any = None,
) -> float:
    """Forward under a tape, ``grad_fn(logits, rows, offset)`` for the logit
    gradient, backward, clip, update. ``micro`` > 0 runs that many rows at a time
    and sums the parameter gradients before one update: ``grad_fn`` normalizes by
    the whole batch, so the update is the same however the rows were split.
    The step is skipped when the loss or grad norm is non-finite."""
    input_ids = np.asarray(input_ids, dtype=np.int64)
    b = input_ids.shape[0]
    model.params = backend.materialize(model.params)
    params = model.params if trainable is None else trainable
    by_id = {id(p): p for p in params.values()}
    name_of = {id(p): k for k, p in params.items()}
    param_ids = set(by_id)
    # Which gradients this rank holds only a slice of, so clipping can use the
    # whole model's norm instead of this shard's.
    sharded_ids: set[int] = set()
    shard_dims: dict[int, int] = {}
    if getattr(backend, "tp_world", 1) > 1:
        from .tensor_parallel import shard_dim

        for k, p in params.items():
            d = shard_dim(k)
            if d is not None:
                sharded_ids.add(id(p))
                shard_dims[id(p)] = d
    rows = micro if 0 < micro < b else b
    # One name for "average this gradient over the dp replicas". Absent on a
    # backend that predates dp, and a no-op at dp_world == 1.
    dp_reduce = getattr(backend, "dp_reduce", None) if getattr(backend, "dp_world", 1) > 1 else None
    if rows != b and getattr(optimizer, "streams", False):
        raise ValueError(
            "micro-batching holds every parameter gradient until the update, which is "
            "the 50.1 GiB a streaming optimizer exists to avoid; use one row per step")

    def run(lo: int, on_grad: Any) -> tuple[float, dict[int, torch.Tensor]]:
        chunk = input_ids[lo : lo + rows]
        n, t = chunk.shape
        kv = _training_kv(model, n, t, device=backend.device)
        tape = Tape()
        t_fwd = time.perf_counter()
        with torch.no_grad(), tape:
            # A vocab-parallel head keeps its shard here: the gathered row is
            # [B, T, vocab] f32 (1.89 GiB at B=8 T=256 on the 27B) and
            # cross_entropy_loss_grad reduces it sharded instead.
            logits = model.forward(chunk, np.arange(t, dtype=np.int64), kv,
                                   RecordingBackend(backend),
                                   sharded_logits=getattr(backend, "tp_world", 1) > 1,
                                   segment="layer" if t > _MLP_SEGMENT_MAX_T else "mlp")
        # Load-bearing: kernel launches are async, so without this the forward's
        # GPU time is billed to whatever syncs first -- the backward.
        _sync(backend)
        if timings is not None:
            timings["forward_secs"] = timings.get("forward_secs", 0.0) + (
                time.perf_counter() - t_fwd)
        loss, grad_logits = grad_fn(logits, chunk, lo)
        if not math.isfinite(loss):
            return loss, {}
        return loss, tape.backward(grad_logits, on_grad=on_grad, needs=param_ids)

    if getattr(optimizer, "streams", False):
        # This optimizer clips by each tensor's own RMS and sizes its step by the
        # param's, both whole-tensor quantities -- so under TP it must reduce them
        # across the shards or each rank clips and steps by a different factor
        # (measured on tiny at tp=2: 32 tensors up to 304911 ulp from the tp=1 step).
        if sharded_ids and hasattr(optimizer, "tp_reduce"):
            optimizer.tp_reduce = backend.all_reduce
        # Every weight gradient coexisting is 50.1 GiB on the 27B: this
        # optimizer clips per update, so each gradient is applied and dropped
        # the moment backward finalizes it.
        t_update = time.perf_counter()
        optimizer.begin()
        if timings is not None:
            timings["optimizer_secs"] += time.perf_counter() - t_update
        seen = 0
        order: list[str] = []

        def _apply(tid: int, g: torch.Tensor) -> bool:
            nonlocal seen
            if tid in param_ids:
                t_update = time.perf_counter()
                if dp_reduce is not None:
                    # Safe here because the tape walks its entries in the order it
                    # recorded them, and that order is the model's own graph, not
                    # anything per-rank: measured identical on all 4 ranks of a
                    # (dp=2, tp=2) world, 27 params. _order_agrees() re-checks it
                    # at gate time so the assumption cannot rot silently.
                    order.append(name_of[tid])
                    dp_reduce(g)
                optimizer.step_one(by_id[tid], g, sharded=shard_dims.get(tid))
                if timings is not None:
                    timings["optimizer_secs"] += time.perf_counter() - t_update
                seen += 1
            return True

        loss, _ = run(0, _apply)
        assert seen or not math.isfinite(loss), _NO_GRAD
        if seen and post_step is not None:
            post_step()  # optimizer.step_one copied in place; refresh served weight faces
        if dp_reduce is not None and _CHECK_DP_ORDER:
            _order_agrees(order, backend)
        return loss

    acc: dict[int, torch.Tensor] = {}
    total = 0.0
    for lo in range(0, b, rows):
        loss, grads = run(lo, None)
        if not math.isfinite(loss):
            return loss
        total += loss * min(rows, b - lo) / b
        for tid in param_ids & set(grads):
            prev = acc.get(tid)
            # fp32 accumulation: bf16 adapter grads summed over a group lose bits.
            acc[tid] = prev.add_(grads[tid]) if prev is not None else grads[tid].float()
    assert acc, _NO_GRAD
    t_update = time.perf_counter()
    # Before the clip, not after: the clipped norm has to be the global one, and
    # clipping each replica's own gradients would scale them by different factors
    # for the same reason the tp shards did.
    #
    # By NAME, not by iterating acc: acc is keyed by id() in gradient-completion
    # order, which differs per rank, so every rank would all-reduce a different
    # tensor at each step. gloo aborts the process on the size mismatch
    # (EnforceNotMet in pair.cc) rather than raising -- measured.
    if dp_reduce is not None:
        order = [k for k in sorted(params) if id(params[k]) in acc]
        for k in order:
            dp_reduce(acc[id(params[k])])
        if _CHECK_DP_ORDER:
            _order_agrees(order, backend)
    norm = clip_grad_norm(acc, 1.0, sharded_ids, backend)
    if math.isfinite(norm):
        optimizer.step(params.values(), acc)
        if post_step is not None:
            post_step()  # refresh served weight faces after the in-place copies
    if timings is not None:
        timings["optimizer_secs"] += time.perf_counter() - t_update
    return total


def _dense_causal_mass(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Naive dense teacher O(t^2): causal softmax attention mass averaged over
    query heads (GQA: K heads repeat). ``q`` [b,t,hq,d], ``k`` [b,t,hkv,d] ->
    [b,t,t], each query row L1-normalised. The f32 parity oracle for
    :func:`dense_causal_page_mass`; never used on a long real sequence."""
    b, t, hq, d = q.shape
    rep = hq // k.shape[2]
    k_exp = k.repeat_interleave(rep, dim=2)                       # [b,t,hq,d]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k_exp) / math.sqrt(d)
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~causal[None, None], float("-inf"))
    return torch.softmax(scores, dim=-1).mean(dim=1)             # avg over heads


def dense_causal_page_mass(q: torch.Tensor, k: torch.Tensor, block: int = 16,
                           q_positions: torch.Tensor | None = None
                           ) -> torch.Tensor:
    """Dense causal attention mass pooled per key PAGE, averaged over query
    heads — the warm-up teacher for a LONG sequence (8k-32k), where the naive
    [t,t] score matrix will not fit. Streams key pages one at a time: an online
    softmax pass gets each query's log-sum-exp, a second pass normalises each
    page's numerator. Memory O(nq*block), compute O(nq*t) attention GEMM.

    ``q`` [b,t,hq,d], ``k`` [b,t,hkv,d] -> [b, nq, ceil(t/block)], each query
    row L1-normalised over pages strictly before it (window excluded later by
    :func:`page_mass_target`). ``q_positions`` (indices in [0,t)) evaluates only
    those query rows: the 27B science run samples 256 positions per span, taking
    the teacher from O(t^2) to O(256*t); None evaluates every position.
    Mathematically identical to ``_dense_causal_mass`` pooled by block at the
    evaluated rows, which is its CPU gate."""
    b, t, hq, d = q.shape
    if q_positions is None:
        q_positions = torch.arange(t, device=q.device)
    nq = int(q_positions.shape[0])
    q_eval = q.index_select(1, q_positions)                   # [b,nq,hq,d]
    rep = hq // k.shape[2]
    k_exp = k.repeat_interleave(rep, dim=2)
    scale = d ** -0.5
    # ceil, not floor: a non-block-divisible T still has a trailing partial page
    # (T=49, block=16 -> pages 0,1,2,3 with page 3 holding key 48); dropping it
    # renormalises that key's mass onto the included pages.
    n_pages = (t + block - 1) // block
    all_idx = torch.arange(t, device=q.device)
    neg_inf = torch.finfo(q.dtype).min

    def _page_scores(p):
        sl = slice(p * block, min((p + 1) * block, t))
        kj = all_idx[sl]
        s = torch.einsum("bqhd,bkhd->bhqk", q_eval, k_exp[:, sl]) * scale
        valid = q_positions[None, None, :, None] >= kj[None, None, None, :]
        return s.masked_fill(~valid, float("-inf")), valid

    # Pass 1: online softmax -> per-(b,hq,q) running max and log-sum-exp.
    run_max = torch.full((b, hq, nq), neg_inf, dtype=torch.float32, device=q.device)
    run_sum = torch.zeros((b, hq, nq), dtype=torch.float32, device=q.device)
    for p in range(n_pages):
        s, valid = _page_scores(p)
        anyv = valid.any(dim=-1)                              # [b,hq,nq]
        bmax = torch.where(anyv, s.amax(dim=-1), torch.zeros_like(run_max))
        # exp only over valid keys; a page with no key <= query contributes 0.
        bsum = torch.where(
            anyv,
            torch.where(valid, torch.exp(s - bmax[..., None]), 0.0).sum(dim=-1),
            torch.zeros_like(run_max))
        new_max = torch.maximum(run_max, torch.where(anyv, bmax, run_max))
        old = torch.where(run_max > neg_inf / 2,
                          run_sum * torch.exp(run_max - new_max), torch.zeros_like(run_sum))
        add = torch.where(anyv, bsum * torch.exp(
            torch.where(anyv, bmax, new_max) - new_max), torch.zeros_like(run_sum))
        run_sum, run_max = old + add, new_max
    lse = torch.where(run_max > neg_inf / 2,
                      run_max + run_sum.clamp_min(1e-30).log(), run_max)  # [b,hq,nq]

    # Pass 2: normalised causal mass landing on each key page, mean over heads.
    page_mass = torch.zeros(b, nq, n_pages, dtype=torch.float32, device=q.device)
    lse_safe = torch.where(lse > neg_inf / 2, lse, torch.zeros_like(lse))
    for p in range(n_pages):
        s, valid = _page_scores(p)
        mass_h = torch.where(valid, torch.exp(s - lse_safe[..., None]), 0.0).sum(dim=-1)
        page_mass[:, :, p] = mass_h.mean(dim=1)
    return page_mass


def sample_query_positions(t: int, n: int, seed: int, min_pos: int,
                           device: torch.device | str = "cpu") -> torch.Tensor:
    """``n`` distinct uniform query positions in [min_pos, t), fixed by seed.
    The 27B recall teacher scores these only: positions past 2048 leave pages the
    selector can miss. n>=t-min_pos returns every eligible position."""
    if t <= min_pos:
        raise ValueError(f"need T>{min_pos} sampled-query positions, got {t}")
    pool = t - min_pos
    # CPU RNG (not the device's): the seed reproduces the same positions on every
    # machine regardless of the CUDA RNG version; the result is small (256 ints).
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(pool, generator=gen)[: min(n, pool)]
    return (perm + min_pos).sort().values.to(torch.long).to(device)


def indexer_capture(model: Any, ids: torch.Tensor, backend: Any, block: int,
                    n_win_pages: int, q_positions: torch.Tensor | None = None,
                    return_raw: bool = False):
    """Run the frozen base once (no grad) and build the indexer warm-up inputs
    from its four source layers. Returns
    ``(H, k_pages, target, n_pages, q_eval, bounds)``:
    H [b,L,nq,hidden] at ``q_positions`` (all T positions when None), per-page
    mean K [b,L,pages,hkv,d], the streamed dense page-mass teacher [b,L,nq,pages]
    with the window excluded, the per-row page count, sampled post-rope queries
    [b,L,nq,hq,d], and per-page Quest bounds [b,L,pages,hkv,2,d] (kmin/kmax over
    the page's real tokens) for the training-free bounds recall. Shared by the
    train step and the recall eval (one forward feeds both scorers). When
    ``return_raw`` the tuple ends with the PRE-renorm page mass (window still in,
    rows sum to 1) for window-included recall controls."""
    from .sparse_index import exclude_window_renorm, index_source_groups

    b, t = ids.shape
    n_pages_tok = (t + block - 1) // block      # ceil: a tail key gets its own page
    if n_pages_tok <= n_win_pages:
        raise ValueError(f"warm-up needs > {n_win_pages} pages, got {n_pages_tok} from T={t}")
    full_layers = list(model.cfg.full_attn_layers)
    sources, _groups = index_source_groups(len(full_layers)) if len(full_layers) >= 4 \
        else (list(range(len(full_layers))), None)
    source_set = {full_layers[s] for s in sources}

    captured: list = []
    model.index_capture = captured
    model.index_capture_layers = source_set
    try:
        kv = _capture_kv(model, t, backend)
        with torch.no_grad():
            # last_only: the capture reads H/Q/K inside the layer loop, so the
            # lm_head only needs the last position — at t=32k a full [t,vocab]
            # head (plus its fp8 GEMM workspace) asks ~30 GiB and OOMs the 27B.
            model.forward(ids, torch.arange(t, device=backend.device), kv, backend,
                          last_only=True)
    finally:
        model.index_capture = None
        model.index_capture_layers = frozenset()

    hkv, d_kv = model.cfg.num_kv_heads, model.cfg.head_dim
    # The true number of valid pages per row (a trailing partial page is valid).
    n_pages = torch.full((b,), n_pages_tok, dtype=torch.long)
    pad_tok = n_pages_tok * block - t
    captured.sort(key=lambda c: c[0])
    # Indexer-Q and the teacher evaluate only q_positions (all T when None); only
    # KEYS span the full length, padded to a whole page so the reshape is legal.
    H = torch.stack([c[1] for c in captured], dim=1)
    K = torch.stack([c[3] for c in captured], dim=1)
    Q = torch.stack([c[2] for c in captured], dim=1)     # post-rope q [b,L,t,hq,d]
    if q_positions is not None:
        H = H.index_select(2, q_positions)
    q_eval = Q if q_positions is None else Q.index_select(2, q_positions)
    if pad_tok:
        K = torch.nn.functional.pad(K, (0, 0, 0, 0, 0, pad_tok))
    mass = torch.stack([dense_causal_page_mass(c[2], c[3], block, q_positions)
                        for c in captured], dim=1)
    k_blocks = K.reshape(b, len(captured), n_pages_tok, block, hkv, d_kv)

    def _page_bounds(real: torch.Tensor) -> torch.Tensor:
        """Per-page Quest kmin/kmax over the page's REAL tokens; the last page
        may be partial (real.shape[3] < block), amin/amax ignore its length."""
        return torch.stack((real.amin(dim=3), real.amax(dim=3)), dim=4)

    if pad_tok:
        # trailing page has fewer real keys: divide each page's sum by its real
        # key count so the partial page is not underweighted by zero padding;
        # its bounds are taken over the real keys only (amin would else be 0).
        counts = torch.full((n_pages_tok,), float(block), device=K.device)
        counts[-1] = block - pad_tok
        k_pages = k_blocks.sum(3) / counts[None, None, :, None, None]
        bounds_full = _page_bounds(k_blocks[:, :, :-1])
        bounds_last = _page_bounds(k_blocks[:, :, -1:, : block - pad_tok])
        bounds = torch.cat((bounds_full, bounds_last), dim=2)
    else:
        k_pages = k_blocks.mean(dim=3)
        bounds = _page_bounds(k_blocks)
    target = exclude_window_renorm(mass, n_pages, n_win_pages)
    base = (H, k_pages, target, n_pages, q_eval, bounds)
    if return_raw:
        # pre-renorm page mass still includes the window; rows sum to 1.
        return (*base, mass)
    return base


def quest_bounds_scores(q_eval: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Training-free Quest upper-bound scores from per-page kmin/kmax, max-pooled
    over the sampled queries (the unit-F scorer). ``q_eval`` [b,L,nq,hq,d],
    ``bounds`` [b,L,p,hkv,2,d] -> [b,L,p]:
    ``sum_h max_t sum_d max(q*kmin, q*kmax)`` averaged over each GQA group's
    attention heads. A page hot for ANY sampled query is selectable. Bounds are
    stored fp16 exactly as the engine's page_bounds_one, so the scored selection
    is the served one (kmin/kmax rounded before the products)."""
    return _quest(q_eval, bounds, pool_heads=True)


def quest_bounds_scores_per_head(q_eval: torch.Tensor,
                                 bounds: torch.Tensor) -> torch.Tensor:
    """Same as :func:`quest_bounds_scores` but WITHOUT the GQA head mean: score
    per attention head (repeat KV bounds over the group), max over queries, sum
    over ALL heads. The #518 item-3 unpooled variant; [b,L,p]."""
    return _quest(q_eval, bounds, pool_heads=False)


def _quest(q_eval: torch.Tensor, bounds: torch.Tensor,
           pool_heads: bool) -> torch.Tensor:
    b, l, nq, hq, d = q_eval.shape
    hkv = bounds.shape[3]
    kmin, kmax = bounds.to(torch.float16).float().unbind(dim=4)   # [b,L,p,hkv,d]
    if pool_heads:
        qi = q_eval.float().reshape(b, l, nq, hkv, hq // hkv, d).mean(dim=4)
    else:
        qi = q_eval.float()                                        # [b,L,nq,hq,d]
        rep = hq // hkv
        kmin = kmin.repeat_interleave(rep, dim=3)
        kmax = kmax.repeat_interleave(rep, dim=3)
    per = torch.maximum(
        qi[:, :, :, None, :, :] * kmin[:, :, None, :, :, :],
        qi[:, :, :, None, :, :] * kmax[:, :, None, :, :, :],
    ).sum(dim=-1)                                                  # [b,L,nq,p,kh]
    return per.amax(dim=2).sum(dim=-1)                             # [b,L,p]


def indexer_recall(model: Any, ids: torch.Tensor, backend: Any,
                   weights: dict[str, torch.Tensor], k_pages_pick: int,
                   block: int = 16,
                   q_positions: torch.Tensor | None = None) -> dict:
    """Top-k recall of dense page mass for one batch, no gradient, from ONE
    frozen forward, for BOTH scorers: the learned indexer (current weights) and
    the training-free Quest bounds scorer. The 27B run passes 256 seeded
    ``q_positions`` per span so the teacher is O(256*T), not O(T^2). Returns
    ``{"index": float, "bounds": float}``."""
    from .sparse_index import (
        WINDOW_PAGES,
        page_scores_for_selector,
        project_indexer_queries,
        project_page_keys,
        topk_page_recall,
    )

    H, k_pages, target, n_pages, q_eval, bounds = indexer_capture(
        model, ids, backend, block, WINDOW_PAGES, q_positions)
    if weights["ik"].shape[0] != model.cfg.num_kv_heads:
        raise ValueError("index heads must equal KV heads")
    with torch.no_grad():
        iq = project_indexer_queries(H, weights["iq"])
        ik = project_page_keys(k_pages, weights["ik"])
        index_sel = page_scores_for_selector(iq, ik, n_pages, WINDOW_PAGES)
        bounds_sel = quest_bounds_scores(q_eval, bounds)
        f = lambda s: float(
            topk_page_recall(s, target, n_pages, k_pages_pick, WINDOW_PAGES))
        return {"index": f(index_sel), "bounds": f(bounds_sel)}


def indexer_warmup_step(
    model: Any,
    input_ids: Any,
    backend: Any,
    weights: dict[str, torch.Tensor],
    optimizer: AdamW,
    block: int = 16,
    q_positions: torch.Tensor | None = None,
) -> float:
    """One indexer warm-up step on captured frozen-base inputs: the two indexer
    projection weights (``iq`` [ih,hidden,di], ``ik`` [ih,d_kv,di]) learn to
    match the dense page-mass teacher via the ``indexer_warmup`` tape op.
    ``q_positions`` subsamples the evaluated queries (27B: 256 per span)."""
    from .sparse_index import WINDOW_PAGES, indexer_warmup_loss

    ids = torch.as_tensor(input_ids, dtype=torch.long, device=backend.device)
    H, k_pages, target, n_pages, _, _ = indexer_capture(
        model, ids, backend, block, WINDOW_PAGES, q_positions)
    iq_w, ik_w = weights["iq"], weights["ik"]
    with Tape() as tape:
        loss = indexer_warmup_loss(H, k_pages, iq_w, ik_w, target, n_pages, WINDOW_PAGES)
    grads = tape.backward(torch.ones((), device=backend.device),
                          needs={id(iq_w), id(ik_w)})
    optimizer.step([iq_w, ik_w], grads)
    return float(loss.detach())


def init_indexer_weights(cfg: Any, gen: torch.Generator, device,
                         di: int | None = None) -> dict[str, torch.Tensor]:
    """The two trainable indexer projection weights: ``iq`` [hkv,hidden,di] and
    ``ik`` [hkv,head_dim,di], small random (V4.1 releases di=128; tiny uses 16)."""
    hkv, d_kv, hidden = cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size
    di = min(16, d_kv) if di is None else di
    scale = 0.1
    return {
        "iq": scale * torch.randn(hkv, hidden, di, generator=gen, device=device),
        "ik": scale * torch.randn(hkv, d_kv, di, generator=gen, device=device),
    }


def indexer_warmup(model: Any, backend: Any, steps: int, seed: int = 0,
                   seq_len: int = 256, batch: int = 1, lr: float = 0.02) -> list[float]:
    """Run the learned-indexer KL warm-up on the frozen base for ``steps`` over
    random ids and return the per-step loss. Tiny/CPU path; the 27B card run
    drives :func:`indexer_warmup_step` directly over real corpus spans."""
    gen = torch.Generator(device=backend.device).manual_seed(seed)
    weights = init_indexer_weights(model.cfg, gen, backend.device)
    opt = AdamW(lr=lr)
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, model.cfg.vocab_size, (batch, seq_len),
                            generator=gen, device=backend.device)
        losses.append(indexer_warmup_step(model, ids, backend, weights, opt))
    return losses


def indexer_warmup_run(model: Any, backend: Any, train_batches: list,
                       held_batches: dict, k_pages_pick: int, steps: int,
                       lr: float, seed: int = 0, di: int | None = None,
                       q_samples: int = 0, q_min_pos: int = 2048) -> dict:
    """The science run over real long-text spans. ``train_batches`` is a list of
    id tensors cycled for ``steps``; ``held_batches`` maps a length label to id
    tensors used for gradient-free recall before and after. Returns recall
    before/after per held-out length (mean and per-span min), the KL curve and
    tokens seen. ``q_samples>0`` subsamples that many seeded query positions >=
    q_min_pos per span for the dense teacher (27B: 256, turning O(T^2) into
    O(256*T)); the same positions are reused for a span before/after and across
    warm-up via ``seed`` + the span's list index."""
    gen = torch.Generator(device=backend.device).manual_seed(seed)
    weights = init_indexer_weights(model.cfg, gen, backend.device, di)

    def qpos(ids, span_seed):
        if q_samples <= 0:
            return None
        return sample_query_positions(int(ids.shape[1]), q_samples, span_seed,
                                      q_min_pos, backend.device)

    def eval_group(batches, span_seed0):
        # One forward per span returns BOTH scorers ("index" learned, "bounds"
        # training-free Quest); structure the result per scorer.
        vals = [indexer_recall(model, ids, backend, weights, k_pages_pick,
                               q_positions=qpos(ids, span_seed0 + j))
                for j, ids in enumerate(batches)]
        return {scorer: {"mean": float(np.mean([v[scorer] for v in vals])),
                         "min": float(np.min([v[scorer] for v in vals])),
                         "per_span": [float(v[scorer]) for v in vals]}
                for scorer in ("index", "bounds")}

    def recalls() -> dict:
        # held groups are sorted ctx labels, same order across before/after.
        return {label: eval_group(batches, seed + 100003 * j)
                for j, (label, batches) in enumerate(sorted(held_batches.items()))}

    before = recalls()
    opt = AdamW(lr=lr)
    curve, tokens_seen = [], 0
    for i in range(steps):
        ids = train_batches[i % len(train_batches)]
        curve.append(indexer_warmup_step(
            model, ids, backend, weights, opt,
            q_positions=qpos(ids, seed + 7919 * (i % len(train_batches)))))
        tokens_seen += int(ids.numel())
    after = recalls()
    return {"recall_before": before, "recall_after": after, "kl_curve": curve,
            "tokens_seen": tokens_seen, "steps": steps, "weights": weights,
            "k_pages": k_pages_pick, "di": weights["iq"].shape[-1],
            "q_samples": q_samples, "q_min_pos": q_min_pos}


def load_span_corpus(cdir: Any, device) -> dict:
    """Load a prepared span dir (scripts/prepare_indexer_corpus) into
    ``{length_label: [id tensors]}``, split prefix as given by the filenames."""
    import json
    from pathlib import Path

    groups: dict = {}
    for path in sorted(Path(cdir).glob("*_*.jsonl")):
        split, ctx = path.stem.split("_", 1)
        if split != "held":
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        groups[ctx] = [torch.tensor(r["ids"], dtype=torch.long, device=device).unsqueeze(0)
                       for r in rows]
    return groups


def indexer_held_recall(model: Any, backend: Any, cdir: Any,
                        weights: dict[str, torch.Tensor], k_pages_pick: int,
                        q_samples: int = 0, q_min_pos: int = 2048,
                        seed: int = 0) -> dict:
    """Gradient-free mean recall per span length over a held-only span dir, under
    the supplied (already-trained) weights — the cross-corpus control. Mirrors
    the science run's seeded sampled-teacher option."""
    groups = load_span_corpus(cdir, backend.device)
    out = {}
    for j, (label, batches) in enumerate(sorted(groups.items())):
        vals = []
        for k, ids in enumerate(batches):
            qp = (sample_query_positions(int(ids.shape[1]), q_samples,
                                         seed + 100003 * j + k, q_min_pos,
                                         backend.device)
                  if q_samples > 0 else None)
            vals.append(indexer_recall(model, ids, backend, weights, k_pages_pick,
                                       q_positions=qp))
        out[label] = {scorer: {"mean": float(np.mean([v[scorer] for v in vals])),
                               "min": float(np.min([v[scorer] for v in vals])),
                               "per_span": [float(v[scorer]) for v in vals]}
                      for scorer in ("index", "bounds")}
    return out


def train_step(
    model: Any,
    input_ids: Any,
    backend: Any,
    optimizer: AdamW,
    trainable: dict[str, Any] | None = None,
    micro: int = 0,
    post_step: Any = None,
) -> float:
    """One SFT step: causal cross-entropy on ``input_ids``. Returns the loss.

    ``post_step`` runs once after a finite optimizer update (e.g. re-packing
    served fp4 faces from the trained bf16 masters)."""
    b = np.asarray(input_ids).shape[0]

    def grad_fn(logits, chunk, lo):
        loss, grad = backend.cross_entropy_loss_grad(logits, chunk)
        # CE averages over this chunk's rows; rescale to the batch's.
        return loss, grad.mul_(len(chunk) / b)

    return _step(model, input_ids, backend, optimizer, trainable, grad_fn, micro,
                 post_step=post_step)


def group_advantages(rewards: Any, group: int, live: Any = None,
                     groups: int | None = None,
                     signal: Any = None) -> np.ndarray:
    """``(r - mean) / std`` within each group of ``group`` consecutive rollouts;
    a tied group yields zeros (no signal, no division by ~0).

    ``live`` is a per-rollout bool mask of the rows that can carry gradient;
    rows outside it set neither the mean nor the std and get advantage 0. A
    zero-length completion is the case: ``rl_step``'s mask scores 0 positions
    for it (``slen == plen``), so its own advantage reaches nothing, but as a
    reward=0 group member it shifts everyone else's. Worst at 8-of-8 correct,
    the most common group at a 91% base: the group should be tied and silent,
    and one empty row turns it into +0.378 on all seven live rows -- gradient
    for being unlike an empty string, which is not a learnable property. It
    also depresses ``tied``, which is P1's criterion.

    ``signal`` is the reward with the length term removed -- the component that
    can carry within-group ordering. A group whose signal is constant has no
    ordering left except length, and a length-only gradient is what collapsed
    the 2026-09-10 P1 run (the length term ran 1.3x the correctness term at
    step 1 and 2.1x+ after; normalisation cancels the penalty's weight, so the
    length spread fills the whole advantage). Such a group yields zeros.
    Constancy is taken over live rows: a failed rollout is not a wrong answer.
    When the judge (``tiebreak``) is on, the judge scores ARE the signal --
    correctness is constant in an all-pass group, but a judge that separates
    the rollouts is exactly the ordering GRPO should learn from. Defaults to
    ``rewards`` (no zeroing beyond the tie case).

    ``groups``, when given, is how many groups the caller believes it is passing,
    and is checked against ``len(rewards) // group``. That check is the point:
    ``reshape(-1, group)`` infers the group count from the length, so 16 rewards
    at ``group=8`` become ``(2, 8)`` whether the caller meant two prompts or one
    prompt whose rollout count doubled. Both are now legal shapes here, so the
    length alone cannot tell them apart -- only the caller's own expectation can,
    and normalising two prompts as one prompt's halves degrades GRPO to REINFORCE
    with no raise.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if group < 1 or r.size % group:
        raise ValueError(
            f"group_advantages got {r.size} rewards for group={group}: every group must "
            f"be whole, so the count has to be a multiple of the group. reshape would "
            f"have raised here too, but only because the remainder is nonzero -- it "
            f"cannot see a wrong group COUNT.")
    if groups is not None and r.size // group != groups:
        raise ValueError(
            f"group_advantages got {r.size} rewards for {groups} group(s) of {group}, "
            f"which is {r.size // group}: an advantage is normalised within ONE prompt's "
            f"rollouts, and reshape(-1, group) would have silently made "
            f"{r.size // group} groups instead, averaging across prompts.")
    r = r.reshape(-1, group)
    m = (np.ones(r.shape, dtype=bool) if live is None
         else np.asarray(live, dtype=bool).reshape(-1, group))
    n = np.maximum(m.sum(axis=1, keepdims=True), 1)
    mean = (r * m).sum(axis=1, keepdims=True) / n
    std = np.sqrt((((r - mean) * m) ** 2).sum(axis=1, keepdims=True) / n)
    adv = (r - mean) / np.where(std > 1e-8, std, 1.0)
    out = np.where((std > 1e-8) & m, adv, 0.0)
    s = r if signal is None else np.asarray(signal, dtype=np.float64).reshape(-1, group)
    hi = np.where(m, s, -np.inf).max(axis=1, keepdims=True)
    lo = np.where(m, s, np.inf).min(axis=1, keepdims=True)
    constant = (hi == lo) & (m.sum(axis=1, keepdims=True) > 0)
    return np.where(constant & m, 0.0, out).reshape(-1)


def rl_step(
    model: Any,
    input_ids: Any,
    advantages: Any,
    prompt_lens: Any,
    backend: Any,
    optimizer: AdamW,
    trainable: dict[str, Any] | None = None,
    seq_lens: Any = None,
    micro: int = 0,
    timings: dict[str, float] | None = None,
) -> float:
    """One policy-gradient step: the causal-CE gradient scaled per row by the
    advantage and zeroed on prompt/padding positions. ``input_ids`` [B,T] is
    prompt+completion right-padded; ``seq_lens`` [B] is each row's valid length
    (default T). Returns the batch cross-entropy as a diagnostic.
    # ponytail: single-update REINFORCE-with-baseline; add the PPO ratio+clip
    # when a rollout is reused for more than one step."""
    t0 = time.perf_counter()
    if timings is not None:
        timings["optimizer_secs"] = 0.0
        timings["forward_secs"] = 0.0
    ids = np.asarray(input_ids, dtype=np.int64)
    b, t = ids.shape
    adv = torch.as_tensor(np.asarray(advantages, dtype=np.float32))
    plen = np.asarray(prompt_lens, dtype=np.int64).reshape(b, 1)
    slen = (np.full((b, 1), t) if seq_lens is None
            else np.asarray(seq_lens, dtype=np.int64).reshape(b, 1))
    pos = np.arange(t)
    # Position i predicts token i+1: scored iff prompt_len <= i+1 < seq_len. Counted
    # over the WHOLE batch — a per-micro-batch normalizer reweights the rows silently.
    n = float(((pos >= plen - 1) & (pos < slen - 1)).sum())

    def grad_fn(logits, chunk, lo):
        bm, tm = chunk.shape
        loss, grad = backend.cross_entropy_loss_grad(logits, chunk)
        dev = grad.device
        p = torch.arange(tm, device=dev).reshape(1, tm)
        rows = slice(lo, lo + bm)
        keep = (p >= torch.as_tensor(plen[rows], device=dev) - 1) & (
            p < torch.as_tensor(slen[rows], device=dev) - 1)
        w = keep.float() * adv[rows].to(dev).reshape(bm, 1)
        # CE averaged over this chunk's bm*(tm-1) positions; rescale to the batch's
        # scored count so prompt, padding and micro-batch size cost nothing.
        return loss, grad.mul_(w.unsqueeze(-1) * (bm * (tm - 1) / max(n, 1.0)))

    loss = _step(model, ids, backend, optimizer, trainable, grad_fn, micro, timings)
    if timings is not None:
        _sync(backend)
        elapsed = time.perf_counter() - t0
        # backward_secs keeps its published meaning -- forward + loss + backward --
        # because manifest["metrics"] readers sum it with rollout and optimizer to
        # reconstruct the step. backward_only_secs is the new, narrower number.
        timings["backward_secs"] = elapsed - timings["optimizer_secs"]
        timings["backward_only_secs"] = (
            timings["backward_secs"] - timings["forward_secs"])
    return loss


def _require_on_policy(
    engine: Any, recapture_graph: bool = False, clear_prefix: bool = False
) -> None:
    """Refuse an engine whose caches would outlive an update.

    A cached prefix or a captured decode graph samples from an earlier policy
    without raising. Two independent caches, so two waivers: a caller that clears
    them after every update passes the matching flag and must then actually call
    ``invalidate_weights()`` -- ``grpo_loop`` does, at loop entry and after each
    step. One flag for both would waive the prefix store for a caller that only
    said "graphs": ``build_engine(decode_graph=False)`` alone still carries a live
    ``PrefixStore``. Neither flag is a capability check: every Engine has the
    method, so testing for it would make this guard pass for everyone.

    ``recapture_graph`` is still the waiver even though the graphs are now kept:
    what it asserts is that the caller calls ``invalidate_weights()`` at all, and
    that call is what refills the cached casts a replay would otherwise read
    stale.
    """
    if not recapture_graph and engine._decode_graph_on is not False:
        raise ValueError("on-policy rollouts need build_engine(decode_graph=False), or "
                         "recapture_graph=True if the loop calls "
                         "engine.invalidate_weights() after every update: a captured "
                         "graph replays a forward traced on the old weights")
    if not clear_prefix and not isinstance(engine._prefix, NoPrefixStore):
        raise ValueError("on-policy rollouts need build_engine(prefix_store="
                         "NoPrefixStore()), or clear_prefix=True if the loop calls "
                         "engine.invalidate_weights() after every update: a cached "
                         "prefix serves KV from the old policy")


def _require_group_fits(engine: Any, group: int) -> None:
    """Refuse an engine narrower than the group it will be handed.

    `grpo_loop` submits the whole group before draining, and an engine with fewer
    usable slots neither raises nor drops: the excess queues into later ticks
    (`Engine.__init__` documents this on the max_batch mismatch), so a group of 16
    into 8 slots is two waves of 8. That halves the rows per tick, and on sm90 the
    rows per tick are what fill the tensor core -- wgmma's M granularity is 16, so a
    16-wide group run as two 8s sits at 50% fill and reports the wall clock of a
    narrow batch.

    This raises rather than warns because the failure fabricates a refutation: an
    experiment asking "does a wider group help" gets "it does not", with no visible
    reason. A warning is the wrong instrument for a defect whose symptom is a wrong
    conclusion. Reads `usable_slots`, not `max_batch`, because the slot is the
    resource a request holds from submit to finish.
    """
    if engine.usable_slots < group:
        raise ValueError(
            f"group={group} into an engine with {engine.usable_slots} usable slots: "
            f"the rollout submits the whole group at once and the excess QUEUES rather "
            f"than raising, so this would run "
            f"{-(-group // engine.usable_slots)} waves of at most "
            f"{engine.usable_slots} and report a narrow batch's wall clock. Pass "
            f"num_slots >= {group} to build_engine (it adds the decode graph's pad row "
            f"itself), and size num_blocks for {group} rows, not 8")


def untruncated(sampling: Any) -> Any:
    """The sampler the policy gradient is actually taken under. ``rl_step`` scores
    with the full softmax, so a truncated or tempered rollout draws from one
    distribution and is differentiated as another, and nothing reweights them.
    Sampling untruncated makes the sampler the policy by construction, and it
    scores best on the deployed sampler too -- differentiating the truncated
    sampler instead is identically zero wherever the nucleus holds one token.
    Carrying the rollout's kept set into the gradient (DeepSeek-V3.2 3.1) is what
    would let RL train under the card's sampler; it is not a reward upgrade.
    See docs/rl-sota-parity.md 2."""
    return replace(sampling, temperature=1.0, top_p=1.0, top_k=0)


def grpo_loop(
    engine: Any,
    model: Any,
    prompts: list[Any],
    reward_fn: Any,
    steps: int,
    backend: Any,
    optimizer: AdamW | None = None,
    *,
    group: int = 8,
    sampling: Any = None,
    seed: int = 0,
    trainable: dict[str, Any] | None = None,
    micro: int = 0,
    tiebreak: Any = None,
    recapture_graph: bool = False,
    clear_prefix: bool = False,
    per_rollout: list | None = None,
    prompts_per_step: int = 1,
    decode: Any = None,
    correctness_fn: Any = None,
    length_penalty: float = 0.0,
    length_cap: int = 1,
) -> Iterator[tuple[float, float, float, float, float, dict[str, float], int, float | None]]:
    """GRPO: sample ``group`` completions per prompt in one engine batch, score
    them with ``reward_fn(prompt_ids, completion_ids) -> float``, take one
    policy-gradient step on the group-normalized advantages, in ``micro`` rows
    at a time (``0`` = the whole group at once). The engine that
    generates IS the model that trains, so it must be built with the prefix
    cache and decode graph off, and rollouts are drawn untruncated so the
    sampler is the policy the step differentiates. Yields
    ``(mean reward, cross-entropy, seconds, tied-group fraction, mean completion
    tokens, phase seconds)`` as each step finishes, so a 100-step run reports progress instead of
    printing at the end. The token count is there because ``tied_group_fraction``
    cannot fall and be bad: ``--judge`` reorders inside the all-pass subgroup by
    construction, so it drives ties toward 0 whether or not it ranks anything real.
    Length is the independent signal that separates the two.

    ``prompts_per_step`` puts several prompts in one step, each still normalised
    within its own ``group``. A step is gradient-free only when EVERY group ties,
    so the tie fraction that matters falls as ``q ** prompts_per_step`` even
    though each group ties MORE at a smaller ``group`` -- measured on 100 MATH
    level-5 problems, 2 prompts x 4 completions raises gradient-bearing steps
    from 35.0% to 41.9%, a 1.21x ceiling that 4x2 does not improve on
    (wins/2026-09-08-the-tie-is-at-the-ceiling.md). It is the only lever on
    steps-to-score rather than seconds-per-step. The batch is
    ``prompts_per_step * group`` rows, so the engine has to be sized for that.

    ``per_rollout``, when given, is extended with one dict per completion
    (``step``, ``g``, ``tokens``, ``reward``, ``advantage``, and ``text`` when
    ``decode`` is given). The yielded tuple
    carries only means, which is the axis a length-vs-reward claim cannot be made
    on: the advantage is computed within a group on one prompt, so pairing has to
    survive to the row level or prompt difficulty confounds it."""
    _require_on_policy(engine, recapture_graph, clear_prefix)
    if prompts_per_step < 1:
        raise ValueError(f"prompts_per_step must be >= 1, got {prompts_per_step}")
    rows = prompts_per_step * group
    _require_group_fits(engine, rows)
    if recapture_graph or clear_prefix:
        # Whatever the engine cached before this loop was built under other weights.
        engine.invalidate_weights()
    if optimizer is None:
        optimizer = AdamW(lr=1e-5)
    sampling = untruncated(sampling if sampling is not None
                           else SamplingParams(max_new_tokens=32))
    for step in range(steps):
        t0 = time.perf_counter()
        # Consecutive prompts, so a step's prompts differ from each other -- the whole
        # point of the lever is several INDEPENDENT groups. Wrapping by len(prompts) as
        # before, so a prompt list shorter than prompts_per_step repeats within a step
        # rather than raising; every group is then the same prompt and the step is worth
        # one, which the `distinct_prompts` yield field reports.
        picks = [np.asarray(prompts[(step * prompts_per_step + p) % len(prompts)],
                            dtype=np.int64) for p in range(prompts_per_step)]
        # Identical seeds would make the group one sample repeated, every advantage zero.
        # Offset by `rows`, not `group`: at prompts_per_step > 1 a step consumes
        # prompts_per_step * group seeds, and striding by `group` would reissue the
        # previous step's seeds to every prompt after the first.
        ids, owner = [], []
        for p, prompt in enumerate(picks):
            for g in range(group):
                ids.append(engine.submit(
                    prompt.tolist(),
                    replace(sampling, seed=seed + step * rows + p * group + g)))
                owner.append(p)
        done = _drain(engine, ids, "grpo_loop rollout")
        timings = {"rollout_secs": time.perf_counter() - t0, "invalidate_secs": 0.0}
        comps = [done[i] for i in ids]
        rewards = [float(reward_fn(picks[owner[i]], c)) for i, c in enumerate(comps)]
        rew0 = rewards
        # Binary correctness BEFORE the length term, tiebreak, and live mask: the
        # quantity comparable to a lambda=0 run's tied fraction. At lam>0 `tied`
        # above is structurally 0 (continuous rewards never exactly match), so the
        # validity gate needs this to see whether a group carries gradient signal.
        if correctness_fn is not None:
            corr = np.array([float(correctness_fn(picks[owner[i]], c)) for i, c in enumerate(comps)])
            tied_correctness = float((corr.reshape(-1, group) == corr.reshape(-1, group)[:, :1]).all(axis=1).mean())
        else:
            corr = None
            tied_correctness = None
        # A binary reward stops producing gradient once the policy clears the task.
        # `tied` is that fraction and is the run's health metric: 72% at the 256 cap,
        # 88.7% at 2048. Do not predict it with p**group -- a tie is all-SAME, not
        # all-correct, and eight rollouts on one prompt behave like 3.4 independent
        # ones (wins/2026-09-04-the-cap-was-the-gradient.md). `tiebreak` reorders
        # WITHIN the all-pass and all-fail subgroups so those steps carry signal; it
        # never crosses the two, so nothing it says can lift a wrong answer over a
        # right one.
        if tiebreak is not None:
            # Per prompt: tiebreak ranks completions of ONE prompt against each other,
            # so handing it a step's whole batch would rank across prompts.
            rewards = [
                r
                for p, prompt in enumerate(picks)
                for r in tiebreak(prompt, comps[p * group:(p + 1) * group],
                                  [x > 0.5 for x in rewards[p * group:(p + 1) * group]])
            ]
        # The non-length component of the reward: what can carry within-group
        # ordering. With the judge, the scores ARE the signal (a judge that
        # separates an all-pass group is the ordering to learn); without it,
        # strip the known length term. A group whose signal is constant is
        # zeroed inside group_advantages -- a length-only gradient is the P1
        # collapse (errors/2026-09-10).
        if tiebreak is not None:
            signal = rewards
        else:
            signal = [r - length_penalty * len(c) / length_cap
                      for r, c in zip(rew0, comps)]
        adv = group_advantages(rewards, group, live=[len(c) > 0 for c in comps],
                               groups=prompts_per_step, signal=signal)
        # Per GROUP, then averaged: `tied` has always meant "the fraction of groups with
        # no signal", and at prompts_per_step > 1 a step holds several. The step-level
        # quantity -- was this step gradient-free at all -- is `tied == 1.0`, which a
        # reader recovers from this; recording the mean keeps it comparable across runs
        # with different prompts_per_step.
        tied = float((adv.reshape(-1, group) == 0).all(axis=1).mean())
        if per_rollout is not None:
            # Per rollout, not the group means: length and reward are paired only
            # WITHIN a group, on one prompt, and a step yields one mean of each --
            # so a cross-step correlation is confounded by prompt difficulty and
            # cannot support a claim about the advantage
            # (wins/2026-09-06-what-a-length-term-can-recover.md).
            # `p` is the prompt within the step and `g` the completion within its group,
            # so a reader can still group rows by prompt at prompts_per_step > 1 -- a
            # flat index would make two prompts' rows look like one group of 16.
            per_rollout.extend(
                {"step": step + 1, "p": owner[i], "g": i % group, "tokens": len(c),
                 "reward": r, "advantage": float(a),
                 # ~0.5 MB per run (150 tok x 8 x 100 steps): cheap, and the only record
                 # of what the policy wrote -- run 86a06dc8c420 saved no text, so its
                 # step-75 collapse can never be replayed.
                 "text": decode([int(t) for t in c]) if decode is not None else ""}
                for i, (c, r, a) in enumerate(zip(comps, rewards, adv))
            )
        # Power-of-two buckets bound shape JITs (tiny: 37.7 s new width, 71 ms repeat).
        # Clamped to the cap: an odd cap would otherwise round PAST it (1500 -> 2048, a
        # 36% overshoot of padding the mask discards). The clamp costs no extra JIT --
        # the width set is {256, 512, 1024, cap} either way -- and cannot narrow below
        # the longest completion, since that never exceeds the cap.
        floor = min(256, int(sampling.max_new_tokens))
        gen = min(int(sampling.max_new_tokens),
                  1 << (max(floor, max(len(c) for c in comps), 1) - 1).bit_length())
        # Rows are left-aligned at their OWN prompt length with the padding all at the
        # end, so `plens` differs per row and each row's scored span `plen-1 .. slen-1`
        # is exactly its own completion. Not padded between prompt and completion (which
        # would put pad INSIDE the scored span) and not left-padded (which would put pad
        # tokens where causal attention lets them reach the prompt -- there is no
        # attention mask here). At prompts_per_step 1 every row has the same prompt and
        # this is the old batch exactly.
        pmax = max(len(p) for p in picks)
        width = pmax + gen
        batch = np.stack([
            np.concatenate([picks[owner[i]], np.asarray(c, dtype=np.int64),
                            np.zeros(width - len(picks[owner[i]]) - len(c), dtype=np.int64)])
            for i, c in enumerate(comps)
        ])
        plens = np.array([len(picks[owner[i]]) for i in range(len(comps))], dtype=np.int64)
        slens = np.array([len(picks[owner[i]]) + len(c) for i, c in enumerate(comps)],
                         dtype=np.int64)
        ce = rl_step(model, batch, adv, plens, backend, optimizer, trainable=trainable,
                     seq_lens=slens, micro=micro, timings=timings)
        if recapture_graph or clear_prefix:
            # After the update, not before the next rollout: a caller that stops
            # iterating must not leave the engine holding graphs traced on weights
            # that no longer exist.
            t_inval = time.perf_counter()
            engine.invalidate_weights()
            timings["invalidate_secs"] = time.perf_counter() - t_inval
        # `gen` is the padded width and the mean is the real one: run 2 could not tell
        # "a long tail" from "every completion at the cap" without both.
        secs = time.perf_counter() - t0
        # The remainder: reward_fn, tiebreak, advantages, the batch stack. Derived, and
        # with .get because rl_step is the only writer of its keys -- a caller that
        # substitutes it must still get a step time, not a KeyError.
        timings["other_secs"] = secs - sum(
            timings.get(k, 0.0)
            for k in ("rollout_secs", "backward_secs", "optimizer_secs"))
        yield (float(np.mean(rewards)), ce, secs, tied,
               float(np.mean([len(c) for c in comps])), timings, gen, tied_correctness)


def opd_loop(
    teacher_engine: Any,
    student_model: Any,
    prompts: list[Any],
    steps: int,
    backend: Any,
    optimizer: AdamW | None = None,
    seed: int = 0,
    trainable: dict[str, Any] | None = None,
    ema_decay: float = 0.999,
    sampling: Any = None,
    recapture_graph: bool = False,
) -> list[float]:
    """On-policy distillation: the teacher engine generates a completion, the
    student takes one :func:`train_step` on prompt + completion. With
    ``trainable`` (LoRA adapters) the teacher is the same model generating
    under an EMA of the adapters, so only adapter-sized memory is duplicated."""
    # Unconditional: without `trainable`, train_step updates model.params (train.py:81)
    # and the engine samples from that same object, so the no-adapter teacher is the one
    # that goes stale fastest -- measured on tiny, 27 of the teacher's parameters changed
    # within two steps. The exemption this replaces claimed the opposite.
    _require_on_policy(teacher_engine, recapture_graph)
    if recapture_graph:
        teacher_engine.invalidate_weights()
    if optimizer is None:
        optimizer = AdamW(lr=1e-3)
    if sampling is None:
        sampling = SamplingParams(max_new_tokens=8)
    # Swaps copy INTO the live adapter tensors and never rebind: the engine reads those objects.
    ema = {k: v.clone() for k, v in trainable.items()} if trainable is not None else None
    student = {k: v.clone() for k, v in trainable.items()} if trainable is not None else None
    losses: list[float] = []
    for step in range(steps):
        prompt = np.asarray(prompts[step % len(prompts)], dtype=np.int64)
        params = replace(sampling, seed=seed + step)
        if ema is not None:
            for k, v in trainable.items():
                v.copy_(ema[k])
        rid = teacher_engine.submit(prompt, params)
        finished = _drain(teacher_engine, [rid], "opd_loop teacher")
        seq = np.concatenate([prompt, np.asarray(finished[rid], dtype=np.int64)])
        if ema is not None:
            for k, v in trainable.items():
                v.copy_(student[k])
        losses.append(train_step(student_model, seq[None, :], backend, optimizer,
                                 trainable=trainable))
        if ema is not None:
            for k, e in ema.items():
                student[k].copy_(trainable[k])
                e.mul_(ema_decay).add_(trainable[k], alpha=1.0 - ema_decay)
        if recapture_graph:
            # The teacher's weights are swapped every step (ema <-> student above),
            # so a graph traced on either set replays the wrong policy.
            teacher_engine.invalidate_weights()
    return losses


class JsonlDataset:
    """JSONL ``{"text"}`` lines -> eos-joined token stream cut into ``seq_len``
    chunks (last chunk eos-padded), in file order."""

    def __init__(
        self, path: str | Path, tokenizer: Any, seq_len: int, eos_token_id: int = 0
    ) -> None:
        stream: list[int] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    stream.extend(tokenizer.encode(json.loads(line)["text"]))
                    stream.append(eos_token_id)
        if not stream:
            raise ValueError(f"JsonlDataset: no text found in {path}")
        chunks = [stream[i : i + seq_len] for i in range(0, len(stream), seq_len)]
        if len(chunks[-1]) < seq_len:
            chunks[-1].extend([eos_token_id] * (seq_len - len(chunks[-1])))
        self._sequences = [np.asarray(c, dtype=np.int64) for c in chunks]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._sequences)

    def __len__(self) -> int:
        return len(self._sequences)


def pretrain(
    model: Any,
    dataset: Any,
    backend: Any,
    optimizer: AdamW,
    steps: int,
    *,
    lr: float = 1e-3,
    warmup: int = 0,
    log_every: int = 1,
    ckpt_dir: str | Path | None = None,
    ckpt_every: int = 0,
    seed: int = 0,
) -> list[float]:
    """Causal-LM training over ``dataset`` (iterable of [T] token arrays) with a
    seeded epoch-wise shuffle and :func:`cosine_warmup`; checkpoints every
    ``ckpt_every`` steps plus ``final/``. Returns the per-step losses."""
    sequences = list(dataset)
    if not sequences:
        raise ValueError("pretrain: dataset yielded no sequences")
    rng = np.random.default_rng(seed)
    ckpt_path = Path(ckpt_dir) if ckpt_dir is not None else None
    losses: list[float] = []
    step = 0
    while step < steps:
        for idx in rng.permutation(len(sequences)):
            input_ids = np.asarray(sequences[idx], dtype=np.int64)[None, :]
            optimizer.lr = cosine_warmup(step, steps, warmup, lr)
            loss = train_step(model, input_ids, backend, optimizer)
            losses.append(loss)
            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"step {step + 1:4d}/{steps}  loss {loss:.4f}  lr {optimizer.lr:.2e}")
            step += 1
            if ckpt_path is not None and ckpt_every > 0 and step % ckpt_every == 0:
                save_hf(model, ckpt_path / f"step_{step}")
            if step >= steps:
                break
    if ckpt_path is not None:
        save_hf(model, ckpt_path / "final")
    return losses


# --- CLI orchestration helpers (moved from cli.py in the step-8a split) ---

def _progress(as_json: bool):
    if not as_json:
        return print
    return lambda *a, **k: print(*a, **{**k, "file": sys.stderr, "flush": True})


def _qwen38_tokenizer():
    from .tokenizer import get_tokenizer

    try:
        return get_tokenizer(_QWEN38_SOURCE)
    except Exception as exc:
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        sys.exit(f"error: could not load the Qwen3-27B tokenizer from "
                 f"{_QWEN38_SOURCE!r}: {first}")






def _train_dry_run(args: argparse.Namespace) -> None:
    """--dry-run on `train`: print the training rows (adapter, optimizer state,
    ISO frames, the layer-segment tape) without building anything. B is the
    micro-batch rows (--micro, else the RL group), S is the max token length the
    tape segments against (--train-seq-len, else the recipe's max_new_tokens)."""
    from . import config as config_mod
    from .memory import format_memory_table, memory_table, train_plan

    cfg = {"tiny": config_mod.tiny, "tiny-agent": lambda: config_mod.tiny(65536),
           "qwen38-27b": config_mod.qwen38_27b}[args.model]()
    b = args.batch or args.micro or args.group
    s = args.train_seq_len or args.max_new_tokens
    is_lora = args.rl or args.opd
    rows = train_plan(cfg, b, s, lora_rank=args.lora_rank if is_lora else None,
                      optim=args.optim)
    table = memory_table(rows, {}, None)
    if args.json:
        print(json.dumps(table, indent=1))
    else:
        algo = f"LoRA r{args.lora_rank} + AdamW" if is_lora else f"full SFT {args.optim}"
        print(f"tilerl train --dry-run: model={cfg.name} {algo} B={b} S={s}")
        print(format_memory_table(table))


def _train_indexer_recall(args: argparse.Namespace, backend, model, log) -> dict:
    """27B science run over prepared corpus spans (scripts/prepare_indexer_corpus):
    recall@k_pages before/after per span length, KL curve, tokens seen."""
    import time

    import torch

    from . import train as train_mod
    from .ledger import file_hash

    cdir = Path(args.indexer_corpus)

    def load(split: str, max_total: int = 0):
        groups = {}
        for path in sorted(cdir.glob(f"{split}_*.jsonl")):
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            ctx = rows[0]["ctx"]
            groups[str(ctx)] = [
                torch.tensor(r["ids"], dtype=torch.long, device=backend.device).unsqueeze(0)
                for r in rows]
        labels = sorted(groups)
        for ctx in labels:
            log(f"{split} ctx={ctx}: {len(groups[ctx])} prompts available")
        if max_total:
            # balanced round-robin across the sorted lengths: 16 over 3 -> 6/5/5,
            # so the cut run is not weighted to cheap 8k spans.
            import itertools
            picked: dict[str, int] = {ctx: 0 for ctx in labels}
            for ctx in itertools.islice(itertools.cycle(labels), max_total):
                if picked[ctx] < len(groups[ctx]):
                    picked[ctx] += 1
            groups = {ctx: groups[ctx][: picked[ctx]] for ctx in labels}
            for ctx in labels:
                log(f"{split} ctx={ctx}: using {picked[ctx]} spans (balanced cut)")
        return groups

    held = load("held", getattr(args, "held_spans", 0))
    train_groups = load("train")
    # cycle training prompts across lengths in an interleaved order
    import itertools
    train_batches = [b for row in itertools.zip_longest(*train_groups.values())
                     for b in row if b is not None]
    t0 = time.perf_counter()
    out = train_mod.indexer_warmup_run(
        model, backend, train_batches, held, args.k_pages, args.steps, args.lr,
        seed=args.seed, di=args.indexer_di,
        q_samples=args.q_samples, q_min_pos=args.q_min_pos)
    out["secs_total"] = time.perf_counter() - t0
    out["corpus"] = file_hash(str(cdir / "manifest.json")) if (cdir / "manifest.json").exists() else None
    # Optional cross-corpus control: held-only recall before/after under the SAME
    # trained weights (e.g. English cosmo 8k after training on Chinese wiki),
    # showing recall is not a tokenization/language artifact.
    if args.indexer_control_corpus:
        cdir2 = Path(args.indexer_control_corpus)
        out["control"] = train_mod.indexer_held_recall(
            model, backend, cdir2, out["weights"], args.k_pages,
            q_samples=args.q_samples, q_min_pos=args.q_min_pos, seed=args.seed)
        out["control_corpus"] = file_hash(str(cdir2 / "manifest.json")) \
            if (cdir2 / "manifest.json").exists() else None
    for i, v in enumerate(out["kl_curve"]):
        if (i + 1) % max(1, args.steps // 20) == 0 or i == 0:
            log(f"step {i + 1:4d}/{args.steps}  kl {v:.4f}")
    return out


def _train_indexer_warmup(args: argparse.Namespace) -> None:
    """Learned-indexer KL warm-up on the frozen base (sparse-KV unit D). With
    --indexer-corpus DIR this is the 27B recall science run (recall@k_pages
    before/after per 8k/16k/32k held-out length); without it, the tiny/CPU
    one-step gate path."""
    import math
    import time

    from tilerl_kernels.backend import get_backend

    from . import train as train_mod
    from .ledger import commit, format_run, gates_pass, new_manifest, now, runs_root, write_manifest

    manifest = new_manifest("train", {
        "model": args.model, "recipe": args.recipe,
        "source": _QWEN38_SOURCE if args.model == "qwen38-27b" else "tiny",
        "commit": commit(), "algo": "indexer-warmup", "steps": args.steps,
        "lr": args.lr, "seed": args.seed,
        "k_pages": getattr(args, "k_pages", None),
        "indexer_di": getattr(args, "indexer_di", None),
        "indexer_corpus": args.indexer_corpus,
        "indexer_control_corpus": args.indexer_control_corpus})
    if args.steps == 0:
        manifest["gates"] = []
        manifest["finished"] = now()
        write_manifest(runs_root(), manifest)
        print(json.dumps(manifest, indent=1) if args.json else format_run(manifest))
        return
    backend = get_backend()
    cfg, model = build_model(args.model, seed=args.seed, keep_master=False)
    log = _progress(args.json)
    log(f"tilerl train: indexer warm-up model={cfg.name} steps={args.steps}")

    if args.indexer_corpus:
        out = _train_indexer_recall(args, backend, model, log)
        acc = args.recall_threshold
        metrics = {
            "tokens_seen": out["tokens_seen"], "k_pages": out["k_pages"], "di": out["di"],
            "kl_first": out["kl_curve"][0], "kl_last": out["kl_curve"][-1],
            "secs_total": out["secs_total"], "corpus": out.get("corpus"),
            "q_samples": out.get("q_samples", 0), "q_min_pos": out.get("q_min_pos")}

        def flatten(tag, book):
            # book: {ctx: {"index"|"bounds": {mean,min,per_span}}}
            for ctx, scorers in book.items():
                for sc, st in scorers.items():
                    metrics[f"recall_{tag}_{sc}_{ctx}"] = st["mean"]
                    if tag in ("before", "after"):
                        metrics[f"recall_{tag}_{sc}_min_{ctx}"] = st["min"]

        flatten("before", out["recall_before"])
        flatten("after", out["recall_after"])
        if "control" in out:
            metrics["control_corpus"] = out.get("control_corpus")
            for ctx, scorers in out["control"].items():
                for sc, st in scorers.items():
                    metrics[f"control_recall_{sc}_{ctx}"] = st["mean"]
        manifest["recall_detail"] = {
            "before": out["recall_before"], "after": out["recall_after"]}
        manifest["metrics"] = metrics
        # Accept: mean LEARNED-indexer after-recall over held lengths clears the
        # threshold; the training-free bounds recall is a reported baseline, not a
        # gate. Per-length mean, per-span min and per-span values are recorded.
        afters = [v["index"]["mean"] for v in out["recall_after"].values()]
        mean_after = sum(afters) / len(afters)
        worst_span = min(v["index"]["min"] for v in out["recall_after"].values())
        bounds_after = [v["bounds"]["mean"] for v in out["recall_before"].values()]
        bounds_mean = sum(bounds_after) / len(bounds_after)
        manifest["gates"] = [{
            "name": "indexer_recall_at_k", "value": mean_after, "threshold": acc,
            "kind": "verdict", "skipped": False, "passed": mean_after >= acc,
            "per_span_min": worst_span,
            "training_free_bounds_recall_mean": bounds_mean}]
        manifest["finished"] = now()
        write_manifest(runs_root(), manifest)
        print(json.dumps(manifest, indent=1) if args.json else format_run(manifest))
        if not gates_pass(manifest):
            sys.exit(1)
        return

    t0 = time.perf_counter()
    losses = train_mod.indexer_warmup(model, backend, args.steps, seed=args.seed, lr=args.lr)
    for i, v in enumerate(losses):
        log(f"step {i + 1:4d}/{args.steps}  kl {v:.4f}")
    finite = all(math.isfinite(v) for v in losses)
    manifest["metrics"] = {"kl_first": losses[0], "kl_last": losses[-1],
                          "secs_total": time.perf_counter() - t0}
    # The tiny teacher (random QK) is near-uniform, so the CPU gate is that the
    # chain RUNS a step with a finite loss, not that KL halves -- learnability on
    # a real teacher is pinned separately in test_sparse_index's KL-halving gate.
    manifest["gates"] = [{
        "name": "indexer_warmup_step_runs", "value": args.steps if finite else None,
        "threshold": args.steps, "kind": "validity", "skipped": False,
        "passed": finite}]
    manifest["finished"] = now()
    write_manifest(runs_root(), manifest)
    print(json.dumps(manifest, indent=1) if args.json else format_run(manifest))
    if not gates_pass(manifest):
        sys.exit(1)


def cmd_train(args: argparse.Namespace) -> None:
    if getattr(args, "dry_run", False):
        return _train_dry_run(args)
    if getattr(args, "indexer_warmup", False):
        return _train_indexer_warmup(args)
    if args.rl or args.opd:
        if getattr(args, "served_fp4", False):
            sys.exit("error: --served-fp4 is full-parameter SFT only; LoRA keeps the frozen served faces")
        if not args.data and not (args.recipe == "grpo-tiny-smoke" and args.model == "tiny"):
            sys.exit("error: --data is required for RL/OPD training")
        return _train_adapters(args)
    _train_full(args)


def _jsonl(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    # A named file with no rows is silent otherwise: cmd_train's `or [...]` falls back
    # to random prompts, so a 100-step GRPO run trains on noise and reports a reward.
    if not rows:
        sys.exit(f"error: {path} has no rows")
    return rows


def _train_full(args: argparse.Namespace) -> None:
    """Full-parameter SFT on random tokens: Adafactor or ISO, streamed updates."""
    import torch
    from tilerl_kernels.backend import get_backend

    from . import train as train_mod
    from .autograd import Adafactor, cosine_warmup
    from .ledger import commit, new_manifest, read_manifest, runs_root
    from .model import drop_quantized

    log = _progress(args.json)
    # The ledger is per-RUN, not per-algorithm: sft-iso-27b exists to produce a
    # P3 verdict and had nowhere to record one.
    manifest = new_manifest("train", {
        "model": args.model, "recipe": args.recipe,
        "source": _QWEN38_SOURCE if args.model == "qwen38-27b" else "tiny",
        "commit": commit(), "algo": "sft", "optim": args.optim,
        "steps": args.steps, "lr": args.lr, "seed": args.seed})
    prev = read_manifest(runs_root(), manifest["id"])
    if prev and prev["finished"] and not args.force:
        log(f"run {prev['id']} already finished; --force reruns")
        return _ledger.finish_run(prev, args.json)
    if args.steps == 0:
        return _ledger.finish_run(manifest, args.json)
    manifest["metrics"] = dict.fromkeys(("ce_first", "ce_last", "secs_per_step_median"))

    backend = get_backend()
    cfg, model = build_model(args.model, seed=args.seed, keep_master=True)
    post_step = None
    if args.served_fp4:
        # Keep the served .wq/.scale/.oscale slots beside the bf16 masters and
        # refresh them after every step; otherwise full SFT frees them on sight.
        from .model import requantize_fp4

        if not cfg.fp4:
            sys.exit("error: --served-fp4 needs an fp4 config (qwen38-27b)")
        post_step = lambda: requantize_fp4(model)  # noqa: E731
    else:
        drop_quantized(model)
    # Adam's m+v on the 27B is 200.4 GiB; Adafactor is 0.03 GiB and streams its updates.
    optimizer = Adafactor(lr=args.lr, weight_decay=0.1)
    if args.optim == "iso":
        from .iso import ISO

        optimizer = ISO(optimizer)
    gen = torch.Generator().manual_seed(args.seed)
    log(f"tilerl train: model={cfg.name} layers={cfg.num_layers} "
        f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} steps={args.steps}")
    losses, secs = [], []
    for step in range(args.steps):
        # ponytail: random-token batch; a real corpus plugs in here without touching train_step.
        input_ids = torch.randint(0, cfg.vocab_size, (2, 64), generator=gen)
        optimizer.lr = cosine_warmup(step, args.steps, 5, args.lr)
        t0 = time.perf_counter()
        loss = train_mod.train_step(model, input_ids, backend, optimizer, post_step=post_step)
        secs.append(time.perf_counter() - t0)
        losses.append(loss)
        log(f"step {step + 1:4d}/{args.steps}  loss {loss:.4f}  {secs[-1]:.1f}s")
    manifest["metrics"].update(
        ce_first=losses[0], ce_last=losses[-1],
        secs_per_step_median=statistics.median(secs))
    if torch.cuda.is_available():
        manifest["metrics"]["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    # Save the trained bf16 model only on request: on 27B this writes ~54 GiB and does
    # a per-tensor .cpu().contiguous() sync, and save_hf over the fused/master keys is
    # only exercised on GPU. The merge path needs bf16 masters (merge refuses fp4), so
    # --save-model is the producer flag for a merge specialist (set by sft-iso-27b).
    if args.save_model:
        from .model import save_hf

        out_dir = Path(runs_root()) / manifest["id"] / "model"
        save_hf(model, out_dir)
        manifest["artifacts"]["out"] = str(out_dir)
    return _ledger.finish_run(manifest, args.json)


def _load_adapter(trainable: dict, path: str, log) -> None:
    """Copy a saved adapter INTO the tensors add_lora just attached.

    ``copy_``, never rebind: the forward reads the objects add_lora put in
    ``model.params``, so assigning new tensors here would load an adapter the model
    never sees and re-score the base while reporting a trained number.

    Unknown or missing keys are refused rather than skipped. An adapter saved before
    the dead-adapter fix (#98) carries ``<weight>.scale.lora_*`` and ``conv1d.lora_*``
    keys that no longer exist, and silently dropping them would load a partial adapter
    under a full adapter's name.
    """
    import torch
    from safetensors.torch import load_file

    saved = load_file(path)
    extra, missing = set(saved) - set(trainable), set(trainable) - set(saved)
    if extra or missing:
        raise SystemExit(
            f"error: {path} does not match this model's adapter\n"
            + (f"  {len(extra)} unknown key(s), e.g. {sorted(extra)[:3]}\n" if extra else "")
            + (f"  {len(missing)} missing key(s), e.g. {sorted(missing)[:3]}\n" if missing else "")
            + "  hint: an adapter saved before the dead-adapter fix carries "
              ".scale/.conv1d adapters that no longer exist; retrain or strip them")
    with torch.no_grad():
        for k, v in saved.items():
            t = trainable[k]
            if tuple(v.shape) != tuple(t.shape):
                raise SystemExit(
                    f"error: {path}: {k} is {tuple(v.shape)}, expected {tuple(t.shape)}")
            t.copy_(v.to(device=t.device, dtype=t.dtype))
    log(f"loaded adapter {sum(v.numel() for v in saved.values()) / 1e6:.1f}M params <- {path}")


def _before_eval_key(args, cfg, backend, eval_params, mmlu_set) -> str | None:
    """The cache key, or None when the base model's identity is not in it.

    ``weights`` is always present and never absent-by-omission: the 27B keys on its
    checkpoint files, `tiny` is a pure function of ``--seed`` and says so, and any
    other model REFUSES to cache rather than key on a base it cannot identify --
    a key that silently omits the weights serves one model's before-arm for another.
    """
    from .ledger import file_hash

    if args.model == "qwen38-27b":
        source = Path(_QWEN38_SOURCE)
        if not source.is_dir():
            from huggingface_hub import snapshot_download

            source = Path(snapshot_download(_QWEN38_SOURCE, local_files_only=True))
        weights = [(str(p.resolve()), s.st_size, s.st_mtime_ns)
                   for p in sorted(source.iterdir()) if p.is_file() for s in [p.stat()]]
    elif args.model.startswith("tiny"):
        weights = None  # built by build_random(seed), and the seed is in `sampling`
    else:
        return None
    inputs = {
        "version": 2, "weights": weights, "config": asdict(cfg),
        # cfg is already tp_config(cfg, tp) here, so tp reaches the key through the
        # sharded dims -- but only while that call order holds. Explicit is cheaper.
        "tp": args.tp,
        "target": backend.target, "precision": backend.precision,
        "eval_file": file_hash(args.eval_gsm8k) if args.eval_gsm8k else None,
        "eval_n": args.eval_n, "matcher": args.reward, "sampling": asdict(eval_params),
        "thinking": args.max_think_tokens > 0 if args.model == "qwen38-27b" else None,
        "mmlu": mmlu_set, "concurrency": 8,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _write_eval_rows(run_id: str, tag: str, rows: list) -> float:
    """One JSON row per problem, so two arms over the same set can be compared
    paired. Returns the mean completion length. P1 fell back to the unpaired
    interval because only totals were kept.

    Creates the run directory: `_finish` makes it, and `_finish` runs AFTER both
    eval arms, so a `not is_dir(): return` here silently wrote nothing at all --
    which is what it did on the first MATH run.
    """
    from .ledger import runs_root

    d = Path(runs_root()) / run_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"eval-{tag}.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return sum(r["tokens"] for r in rows) / max(1, len(rows))


def _eval_row_appender(run_id: str, tag: str):
    """Append scored rows to eval-<tag>.jsonl as they land: a killed eval arm keeps
    what finished -- the MATH before-arm died at 1h40m with zero rows on disk,
    because the write happened only after the whole arm (errors/2026-09-09-the-
    killed-eval-arm-kept-nothing.md). One open/close per row, so a kill loses at
    most the row in flight.

    Coverage is the GSM8K arm and the curve points: gsm8k_accuracy streams rows
    through on_row. The MMLU arm still lands whole -- mmlu_accuracy has no
    on_row -- so a kill mid-MMLU still loses that arm (about 12% of before/after
    wall time)."""
    from .ledger import runs_root

    path = Path(runs_root()) / run_id / f"eval-{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    def append(row: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    return append


def _read_eval_rows(run_id: str, tag: str) -> list:
    """Per-problem rows of one eval arm, or [] if the arm was not written."""
    from .ledger import runs_root

    f = Path(runs_root()) / run_id / f"eval-{tag}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.is_file() else []


def _mcnemar(before: list, after: list, dataset: str = "gsm8k") -> dict | None:
    """Paired significance on the per-question rows both arms wrote, or None.

    The comparison IS paired: `cli.py` scores one `eval_rows` list in both arms and
    `gsm8k_accuracy` forces temperature 0, so question i is the same question on both
    sides. Keeping only the two totals threw that away and left the unpaired interval,
    whose 80%-power one-sided MDE at n=500 is 7.70 pt -- ABOVE the roadmap's +5 pt
    target, so a real effect at the standard would have failed to register. The paired
    SE is `sqrt(b + c) / n` over the discordant counts, 1.00-1.41 pt at a 5-10% flip
    rate, which puts +5 pt at 3.5-5 SE instead.

    None when the arms cannot be paired -- different lengths, a missing `i`, or no
    discordant pairs at all. `b + c == 0` is not a failure: it means the two arms
    agreed on every question, so there is nothing for a paired test to resolve.
    """
    def by_i(rows):
        out = {}
        for r in rows:
            if r.get("dataset", dataset) == dataset and "i" in r:
                out[r["i"]] = bool(r["correct"])
        return out

    lo, hi = by_i(before), by_i(after)
    if not lo or lo.keys() != hi.keys():
        return None
    b = sum(1 for i in lo if lo[i] and not hi[i])   # was right, now wrong
    c = sum(1 for i in lo if not lo[i] and hi[i])   # was wrong, now right
    n = len(lo)
    if b + c == 0:
        return {"n": n, "b": b, "c": c, "delta": 0.0, "se": None, "z": None}
    se = (b + c) ** 0.5 / n
    return {"n": n, "b": b, "c": c, "delta": (c - b) / n, "se": se,
            "z": ((c - b) / n) / se}


#: the before-arm's mean completion must leave headroom under the rollout cap. 0.8
#: rather than 1.0 because the MEAN fitting exactly means half the rollouts do not.
_ROLLOUT_HEADROOM = 0.8


#: How many eval prompts the before/after/curve arms submit at once. A constant because
#: the KV pool is sized for the WIDER of this and the rollout group -- one engine, two
#: consumers. It was three literal 8s while the rollout width was also 8, so --group 2
#: sized the pool for 2 rows and the eval arm's 8 exhausted it mid-step.
_EVAL_CONCURRENCY = 8


def _curve_rows(eval_rows: list, n: int, seed: int) -> list:
    """The curve subset: a fixed-seed shuffle's first ``n`` rows, not the file's.

    ``gsm8k_test.jsonl`` is ordered -- its first 200 rows run 5 pt low (z=3.05,
    errors/2026-09-04-the-eval-cap-measured-itself.md) -- and a 5 pt bias is the
    size of the effect the curve measures. The seed is fixed across runs so a
    curve point is paired across steps and comparable to the historical anchor.
    """
    pool = list(eval_rows)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def _write_rollout_rows(run_id: str, rows: list, written: int = 0) -> int:
    """Append the rows not yet on disk, and return the new count.

    One JSON row per completion, so length and reward stay paired. Run 2's mechanism
    claim -- short rollouts score better, so the policy lengthens -- was made from two
    group MEANS per step, which is a cross-step correlation confounded by prompt
    difficulty; each step draws a different prompt
    (wins/2026-09-06-what-a-length-term-can-recover.md). Nothing could have been
    re-derived from that run because the pairing never reached disk.

    Per step rather than once at the end, because nothing in this package handles a
    signal: run 2 took a SIGTERM at step 45 and never reached any writer. A run killed
    that way now loses at most the step in flight.
    """
    from .ledger import runs_root

    if len(rows) <= written:
        return written
    d = Path(runs_root()) / run_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / "rollouts.jsonl").open("a" if written else "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows[written:])
    return len(rows)


def _length_aware(match, gold, tok, lam: float, cap: int):
    """The RL reward: correctness minus ``lam`` times the completion's fraction of the cap.

    A correctness-only reward is indifferent between two right answers of any two lengths, so
    an all-right group ties at zero advantage and produces no gradient -- run 2 collapsed that
    way at step 41 of 100 (errors/2026-09-06-the-rollouts-grew-into-the-cap.md).

    Here and NOT in `match`: `MATCHERS` also feeds `gsm8k_accuracy`, whose count becomes
    `manifest["metrics"]["gsm8k_*"]`, the number P1's exit criterion reads. A length term in
    the matcher contract would sit inside the gate.

    `lam` is a switch, not a dial. In an all-right group it cancels exactly -- the advantage
    divides by the group std, so `-(L_i - Lbar)/std(L)` has no `lam` in it -- and in a mixed
    group `lam <= cap/(cap-1)` keeps a short wrong answer from outranking a long right one
    (2048/2047 = 1.000488520, measured).

    An all-wrong group is the third case: every match is 0, so `r_i = -lam * L_i / cap` and
    the normalized advantage is again `-(L_i - Lbar) / std(L)` -- `lam` cancels for every
    lam > 0, so its magnitude does not tune this gradient and only lam = 0 turns it off (zero
    reward spread, and `group_advantages` zeroes a tied group). The difference is what the
    gradient says: length is the only signal, the shortest wrong answer gets the highest
    advantage, and on a problem the model cannot solve it learns "answer shorter" and nothing
    else. That buys seconds_per_step and cannot outrank a right answer (the bound above still
    holds). What neither covers is the direction itself: this pressure points at empty
    outputs, and `_refuse_short_rollouts` only reads the BASE policy's length before training
    -- `--allow-short-rollouts` disables it and the in-loop drift check, and it cannot see a
    policy that shortens mid-training -- while the `live` mask in `group_advantages` only
    keeps an empty row from polluting its group's normalization, not the live rows' gradient
    toward shorter. A 2026-09-09 run collapsed to all-empty outputs (GSM8K 0/500) with both
    guards in place; that run had lam=0, so it is not this gradient's doing, but it shows the
    path is reachable on this model.
    """
    def reward(prompt, completion):
        text = tok.decode([int(t) for t in completion])
        r = float(match(text, gold[tuple(int(t) for t in prompt)]))
        return r - lam * (len(completion) / cap)

    return reward


def _within_group_r(rows: list) -> float | None:
    """Pearson r of (tokens, reward) POOLED over within-group deviations.

    Centering per group is what removes prompt difficulty: a hard prompt shifts
    both its lengths and its rewards, and that shift is the confound. A tied group
    contributes zero deviation in reward and so cannot move r -- which is correct,
    it carries no signal, and it is also why r is None on a run where every group
    tied.

    Recorded, never gated: the consumer is a person reading a finished P1 run, not code.
    The sign says whether the length term is doing what run 2's diagnosis said it would,
    and that reading needs the number together with the run's context.
    """
    import collections

    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["step"]].append((r["tokens"], r["reward"]))
    dx: list[float] = []
    dy: list[float] = []
    for g in groups.values():
        if len(g) < 2:
            continue
        mx = sum(t for t, _ in g) / len(g)
        my = sum(v for _, v in g) / len(g)
        dx.extend(t - mx for t, _ in g)
        dy.extend(v - my for _, v in g)
    sxx = sum(a * a for a in dx)
    syy = sum(b * b for b in dy)
    if sxx <= 0 or syy <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sxx * syy) ** 0.5


def _refuse_short_rollouts(mean_len: float | None, cap: int, allow: bool = False) -> None:
    """Stop before training when the rollouts cannot reach an answer.

    The base policy's own completion length is measured by the before-arm that just
    ran, so this compares two known numbers rather than guessing. Truncated rollouts
    never emit the answer, every sample in a group scores 0, and GRPO trains on a
    reward that is constant -- 100 steps of tied-at-the-floor groups, which looks
    like a hard task rather than a misconfiguration (measured: MATH level 5 needs
    1038 tokens against a 512 cap, 5 of the first 6 steps tied at 1.00, reward 0).

    The mirror of it is the eval cap, which scores the cap instead of the policy
    (errors/2026-09-04-the-eval-cap-measured-itself.md). Same family: a length
    parameter set without measuring the length it bounds.
    """
    if not mean_len or allow or mean_len <= _ROLLOUT_HEADROOM * cap:
        return
    sys.exit(
        f"error: the base policy averages {mean_len:.0f} completion tokens but "
        f"--max-new-tokens is {cap}. Rollouts would be truncated before they answer, "
        f"so every group ties at the floor and no gradient flows. Raise the cap above "
        f"{mean_len / _ROLLOUT_HEADROOM:.0f}, pick an easier task, or pass "
        f"--allow-short-rollouts if the truncation is deliberate."
    )


def _emit_eval_records(correct: int, total: int, ntok: int, token_lens: list,
                       steps: int, backend) -> None:
    """Append the arm's two operands to the bench store. A training run with
    eval arms IS the collector — the numbers exist here and nowhere else.

    tokens/correct is the view rollout_tokens / gsm8k_pct, never stored: a
    stored ratio gets one chance to drift from its operands."""
    import math

    from .ledger import _benchrec
    benchrec = _benchrec()
    p = correct / total
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    import torch
    cuda = backend.device.type == "cuda" and torch.cuda.is_available()
    # The seven common fields are record_common's contract; building them here
    # forked the device-name decision twice (a hardcoded "H20", then a second
    # cuda predicate that diverged from record_common's on CI). The namespace is
    # the adapter: record_common eats argparse namespaces, and a server-side
    # write takes the torch-probe path with no --device-name gate.
    args = argparse.Namespace(
        target=backend.arch, model_name="27B-nvfp4",
        card=int(vis.split(",")[0]) if cuda and vis else None,
        device_name=None, new_device=False,
    )
    common = {
        "shape": {"steps": steps},
        # compiles=0 means "not applicable": accuracy and greedy length are
        # compile-invariant (JIT time enters a seconds figure, not a proportion
        # or token count), not "measured zero compiles in the window".
        "warm": {"state": "warm", "compiles": 0},
        "n": total,
        **benchrec.record_common(args, build="eager"),
    }
    acc = {
        "metric": "gsm8k_pct", "value": round(100 * p, 1), "unit": "%",
        "spread": round(100 * math.sqrt(p * (1 - p) / total), 2), **common,
    }
    acc["floor"] = benchrec.measured_best_floor(acc, lower_is_better=False)
    benchrec.append(acc)
    tok = {
        "metric": "rollout_tokens", "value": round(ntok / total, 1), "unit": "tokens",
        "spread": round(statistics.stdev(token_lens), 1) if total >= 2 else 0.0, **common,
    }
    # rollout_tokens has no monotonic direction (a shorter rollout can be a
    # better policy or a collapsed one): the floor is the measurement itself,
    # judged only alongside gsm8k_pct.
    tok["floor"] = {
        "kind": "reference", "value": tok["value"], "unit": tok["unit"],
        "derivation": "no monotonic direction; read alongside gsm8k_pct",
    }
    benchrec.append(tok)


def _train_adapters(args: argparse.Namespace) -> None:
    """GRPO or OPD: LoRA on the frozen base, the engine samples, the ledger gates."""
    if getattr(args, "eval_every", 0):
        _ledger.refuse_blind_curve(args.eval_curve_n, args.curve_target_pt)
    import torch
    from tilerl_kernels.backend import get_backend

    # Lazy: tests monkeypatch tilerl.build.build_engine to capture training kwargs;
    # a module-level import binds it before the patch.
    from . import train as train_mod
    from .autograd import AdamW
    from .build import build_engine
    from .eval import gsm8k_accuracy, mmlu_accuracy, mmlu_questions
    from .kv_cache import NoPrefixStore
    from .ledger import (
        EarlyStop,
        commit,
        curve_churn,
        file_hash,
        find_run_for_artifact,
        new_best_point,
        new_manifest,
        paired_se,
        read_manifest,
        require_paired_width,
        runs_root,
        significant_decline,
        write_manifest,
    )
    from .model import add_lora
    from .prompt import render_chat, sampling
    from .tokenizer import get_tokenizer

    real = args.model == "qwen38-27b"
    log = _progress(args.json)
    tok = _qwen38_tokenizer() if real else get_tokenizer(None)
    rows = _jsonl(args.data)
    _eval_all = _jsonl(args.eval_gsm8k)
    # Held-out means held-out: every CLI gate test once passed the SAME file to
    # --data and --eval-gsm8k, so the encoded gate was green at a contamination
    # fraction of 1.0 and no check asserted the eval was a distinct slice. Check
    # the FULL eval file, not the eval_n slice: a prompt scored later must not be
    # a prompt the policy trained on, whichever row --eval-n kept.
    if rows and _eval_all:
        train_q = {r.get("prompt") for r in rows}
        overlap = sum(1 for r in _eval_all if r.get("prompt") in train_q)
        if overlap:
            sys.exit(f"error: --eval-gsm8k shares {overlap} prompts with --data; "
                     "the eval arm must be held out from training")
    eval_rows = _eval_all[: args.eval_n]
    thinking = (args.max_think_tokens > 0) if real else None
    params = sampling(tok, thinking, args.max_new_tokens, temperature=args.temperature,
                      max_think_tokens=args.max_think_tokens, seed=args.seed)
    # A SEPARATE params for the eval arms. Sharing `params` scored the policy at the
    # ROLLOUT cap, so the eval measured the cap: 38.4% with mean completion 238.7
    # against a 256 cap, ~82.5% uncapped
    # (errors/2026-09-04-the-eval-cap-measured-itself.md). Same prompt template and
    # stop ids -- only the length differs, and gsm8k_accuracy forces temperature 0.
    eval_params = sampling(tok, thinking, args.eval_max_new_tokens,
                           temperature=args.temperature,
                           max_think_tokens=args.max_think_tokens, seed=args.seed)

    # Same inputs = same run: a finished one is returned instead of retrained.
    manifest = new_manifest("train", {
        "model": args.model, "recipe": args.recipe, "source": _QWEN38_SOURCE if real else "tiny",
        "commit": commit(), "algo": "grpo" if args.rl else "opd",
        "data": file_hash(args.data) if args.data else None, "steps": args.steps,
        "group": args.group, "prompts_per_step": args.prompts_per_step,
        "max_new_tokens": args.max_new_tokens,
        "allow_short_rollouts": args.allow_short_rollouts,
        "temperature": params.temperature, "max_think_tokens": args.max_think_tokens,
        "lr": args.lr, "lora_rank": args.lora_rank, "seed": args.seed, "eval_mmlu": args.eval_mmlu,
        # In the id: tp=1 and tp=4 are different runs, and without this the second
        # would be handed the first's finished manifest and never train.
        "tp": args.tp,
        "reward": args.reward,
        # In the id: it changes what the reward MEANS, so two runs differing only here are
        # not the same run and the second must not be handed the first's manifest. Stays a
        # float although the help calls it a switch -- narrowing the type would change every
        # already-recorded id and orphan those runs' manifests.
        "length_penalty": args.length_penalty,
        # In the id: with the judge on, judged ordering replaces the length-shaped
        # reward inside saturated groups, so a judge run is a different reward.
        "judge": args.judge,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "load_adapter": file_hash(args.load_adapter) if args.load_adapter else None,
        "eval_gsm8k": file_hash(args.eval_gsm8k) if args.eval_gsm8k else None,
        "eval_n": args.eval_n,
        # In the id: it selects which problems the curve scores, so two runs differing
        # only here are not the same run.
        "eval_curve_seed": args.eval_curve_seed,
        # In the id: they decide which problems the curve scores, how dense it is, and
        # where the run stops. A patience on/off pair sharing an id would hand the
        # second run the first's finished manifest -- silent, and the pair is the
        # evidence for the default-flip decision.
        "eval_every": args.eval_every, "eval_curve_n": args.eval_curve_n,
        "patience": args.patience})
    prev = read_manifest(runs_root(), manifest["id"])
    if prev and prev["finished"] and not args.force:
        log(f"run {prev['id']} already finished; --force reruns")
        return _ledger.finish_run(prev, args.json)
    # Lineage: a continued adapter run descends from the run that produced the file
    # loaded through --load-adapter. Resolved before this manifest is written, so the
    # search can never match the current run; an unlinked file contributes no parent.
    if args.load_adapter:
        parent = find_run_for_artifact(runs_root(), args.load_adapter)
        if parent is not None:
            manifest["parents"] = [parent]
    manifest["metrics"] = dict.fromkeys((
        "mmlu_before", "mmlu_after", "gsm8k_before", "gsm8k_after",
        "gsm8k_before_tokens", "gsm8k_after_tokens", "peak_gib"))

    backend = get_backend()
    # LoRA on a frozen base needs no bf16 master (~27 GB on the 27B).
    cfg, model = build_model(args.model, seed=args.seed, keep_master=False,
                              tp=args.tp, backend=backend)
    log(f"tilerl train: model={cfg.name} layers={cfg.num_layers} "
        f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} steps={args.steps}")
    gen = torch.Generator().manual_seed(args.seed)
    prompts = [tok.encode(render_chat([("user", r["prompt"])], thinking)) for r in rows] or [
        torch.randint(0, cfg.vocab_size, (16,), generator=gen).tolist() for _ in range(8)]
    draft = None
    if args.opd and args.draft:
        from .spec import load_draft

        draft = load_draft(model, args.draft)
    # The pool holds every in-flight row's whole sequence, so a flat 512 blocks is
    # 8192 tokens across 8 slots -- 1024 each. Past that the rollout dies mid-step on
    # "PagedKvPool exhausted" (kv_cache.py:80), so --max-new-tokens above ~1024 was
    # unreachable however the recipe was written. Size the pool from the ask instead.
    from .kv_cache import BLOCK_TOKENS

    # 1024 floor = the old flat 512 blocks. max_total_tokens only guards one request and
    # costs no memory, so it never drops below the 8192 default.
    # Hand-computed, so `_fit_blocks` never runs here: passing num_blocks truthy is what
    # skips it (engine.py:1645), and it is the only path that measures free memory instead
    # of deriving a pool from context. That is deliberate for now -- training also holds
    # gradients, the tape and the optimizer state, which `_fit_blocks` does not model, so
    # its two-thirds rule is calibrated for serve. Whether training should use it is a
    # card-pending question, not an oversight.
    #
    # Two consumers, each priced on its OWN rows and its OWN length, then max(). Crossing
    # the axes instead -- widest rows x longest sequence -- costs `--group 16` twice the
    # blocks the eval needs, and over-allocation here is not slack: this path also holds the
    # gradients, the tape and the optimizer state.
    #
    # #320 took the max on the ROW axis alone and left per-row length at the rollout's cap,
    # which still exhausted at the default --eval-max-new-tokens 2048 (measured: `--group 8`,
    # 520 blocks, needs 1099). Its arms passed only because `max(2, 8)` handed the narrow
    # group 4x the rows it used, absorbing the length shortfall on the wrong axis.
    rollout_ctx = max(map(len, prompts)) + args.max_new_tokens + 64
    # 515: MMLU's longest rendered prompt, the figure the 1024 floor was chosen for. GSM8K's
    # 183 sits under any floor, so the MMLU arm is the only eval prompt that can exceed the
    # training prompts -- and this function never sees either.
    eval_ctx = max(max(map(len, prompts)), 515 if args.eval_mmlu else 0) \
        + args.eval_max_new_tokens + 64
    # Sized from --group, not a literal 8: grpo_loop submits the whole group at once
    # (train.py, one submit per g), so a group wider than the engine runs in waves and
    # every rollout in the second wave decodes at a batch the tensor core underfills.
    # The three used to be 8 while --group was a settable flag defaulting to 8, so
    # --group 16 quietly became two waves of 8.
    # A step is --prompts-per-step groups, all submitted at once, so the rollout's width is
    # their product, not --group. At the default 1 this is `max(args.group, 1)` exactly.
    rollout_batch = max(args.group, 1) * max(args.prompts_per_step, 1)
    # min(), not _EVAL_CONCURRENCY: the eval arms ask for _EVAL_CONCURRENCY rows but only
    # num_slots of them hold blocks at once, since a submit past the slots queues inside the
    # engine. Measured on the discriminating case -- `--group 4`, 520 blocks, eval cap 1500:
    # 4 rows need 384 and pass, 8 would need 768 -- and `tilerl-0a` predicted the group-16
    # exhaustion (1033 blocks) from the same model before `tilerl-48` hit it.
    eval_rows_in_flight = min(rollout_batch, _EVAL_CONCURRENCY)
    blocks = max(rollout_batch * -(-rollout_ctx // BLOCK_TOKENS),
                 eval_rows_in_flight * -(-eval_ctx // BLOCK_TOKENS)) + 8
    ctx = max(rollout_ctx, eval_ctx, 1024)
    engine = build_engine(cfg, model, backend, num_slots=rollout_batch,
                          max_batch=rollout_batch, draft=draft,
                          num_blocks=blocks,
                          max_total_tokens=max(ctx, 8192),
                          spec_depth=args.depth,
                          decode_graph=not args.deterministic,
                          sparse_k=0,  # on-policy training needs the dense full-context tape
                          prefix_store=NoPrefixStore())
    # Not in `inputs`: the id is a hash of it, so recording the pool there would make
    # every pool change a different run and hand nothing back on a rerun. It is beside
    # `metrics` because it is a property of the run, and read off the built engine
    # because the kwargs and the pool disagree (max_blocks clamps, the graph adds a row).
    manifest["engine"] = engine.config
    # After build_engine: it materializes the params an adapter must point at.
    trainable = add_lora(model, rank=args.lora_rank)
    if args.load_adapter:
        _load_adapter(trainable, args.load_adapter, log)
    optimizer = AdamW(lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    mean_len: dict[str, float | None] = {}
    mmlu_set = mmlu_questions(args.eval_mmlu) if args.eval_mmlu else None
    cache = None
    if (eval_rows or mmlu_set) and not args.load_adapter and not args.draft:
        key = _before_eval_key(args, cfg, backend, eval_params, mmlu_set)
        if key is not None:
            cache = Path(runs_root()) / "eval-cache" / f"{key}.json"
            manifest["eval_before_cache"] = {"key": key, "cache_hit": cache.is_file()}

    def evals(tag):
        # Timed on BOTH paths, so the cache's payoff is a recorded number rather than an
        # argument: a hit writes ~0 s here and a miss writes what the arm cost, and the
        # difference is what wins/2026-09-05-before-eval-cache.md has owed since it landed
        # `pending-remote` -- 55 lines of mechanism plus 129 of test is worth it at 15 min
        # per hit and is not at 40 s.
        t_eval = time.perf_counter()
        if tag == "before" and cache is not None and cache.is_file():
            saved = json.loads(cache.read_text())
            manifest["metrics"].update(saved["metrics"])
            _write_eval_rows(manifest["id"], tag, saved["rows"])
            mean_len[tag] = saved["mean_len"]
            manifest["eval_before_cache"]["cache_hit"] = True
            manifest["metrics"][f"eval_{tag}_secs"] = time.perf_counter() - t_eval
            log(f"eval before: cache hit {cache.stem}")
            return
        rows_out: list = []
        append = _eval_row_appender(manifest["id"], tag)
        if args.eval_mmlu:
            # Per-arm, because `eval_{tag}_secs` is the SUM of both arms and no historical run
            # can be decomposed into them -- not even by subtraction, since the gsm8k arm was
            # never timed either. MMLU is prefill-dominated (1000 questions x ~515 prompt
            # tokens, 1 token generated), so its cost does not follow from any decode figure.
            t_mmlu = time.perf_counter()
            c, n, conc = mmlu_accuracy(engine, tok, args.eval_mmlu, concurrency=_EVAL_CONCURRENCY,
                                       questions=mmlu_set, per_problem=rows_out)
            for r in rows_out:
                append(r)  # mmlu first, then the gsm8k stream: same order the cache replays
            manifest["metrics"][f"mmlu_{tag}_secs"] = time.perf_counter() - t_mmlu
            manifest["metrics"][f"mmlu_{tag}"] = c / n
            manifest["metrics"][f"mmlu_{tag}_concurrency"] = conc
            manifest["metrics"][f"mmlu_{tag}_correct"] = c
            manifest["metrics"][f"mmlu_{tag}_total"] = n
            log(f"mmlu 0-shot {c}/{n} = {100 * c / n:.1f}% (seed 0, concurrency {conc}) "
                f"in {manifest['metrics'][f'mmlu_{tag}_secs']:.1f}s")
        if eval_rows:
            gsm_rows: list = []
            t_gsm = time.perf_counter()
            c, n, ntok = gsm8k_accuracy(engine, tok, eval_rows, eval_params, concurrency=_EVAL_CONCURRENCY,
                                        thinking=thinking,
                                        match=MATCHERS[args.reward],
                                        per_problem=gsm_rows,
                                        on_row=lambda r: append(dict(r, dataset="gsm8k")))
            manifest["metrics"][f"gsm8k_{tag}_secs"] = time.perf_counter() - t_gsm
            mean_len[tag] = sum(r["tokens"] for r in gsm_rows) / max(1, len(gsm_rows))
            rows_out.extend(dict(r, dataset="gsm8k") for r in gsm_rows)
            manifest["metrics"][f"gsm8k_{tag}"] = c
            manifest["metrics"][f"gsm8k_{tag}_tokens"] = ntok
            manifest["metrics"][f"gsm8k_{tag}_total"] = n
            # tokens/correct, not tokens: the ratio is what a length claim compares
            # on, and it cannot be improved by getting fewer questions right.
            per = f"  {ntok} tokens ({ntok / c:.1f}/correct)" if c else f"  {ntok} tokens"
            log(f"gsm8k greedy {c}/{n} = {100 * c / n:.1f}%{per}")
            if real:
                _emit_eval_records(c, n, ntok, [r["tokens"] for r in gsm_rows],
                                   0 if tag == "before" else args.steps, backend)
        # rows_out (mmlu + gsm8k, prompt order) feeds the before-arm cache payload;
        # the file itself was streamed above, mmlu rows in-block and gsm8k per row.
        # Read before the cache write so a hit's cost excludes the write only a miss pays,
        # but stored after it, because a duration is not a cacheable result: it belongs to
        # the run that paid it. Inside the payload it would replay a past cost onto a hit
        # -- 0.74 s where 0.0013 s was spent -- and `_secs` matches the `_before` filter.
        elapsed = time.perf_counter() - t_eval
        if tag == "before" and cache is not None:
            # `_secs` excluded, not just `eval_before_secs` by ordering: a duration belongs to
            # the run that paid it, and the per-arm timings added beside the scores DO match
            # the `_before` filter, so caching them would replay a miss's minutes onto a hit.
            saved = {"metrics": {k: v for k, v in manifest["metrics"].items()
                                 if "_before" in k and not k.endswith("_secs")},
                     "rows": rows_out, "mean_len": mean_len.get(tag)}
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=cache.parent, delete=False) as f:
                json.dump(saved, f)
            os.replace(f.name, cache)
        manifest["metrics"][f"eval_{tag}_secs"] = elapsed

    drift = {"name": "rollouts_within_cap", "value": None,
             "threshold": _ROLLOUT_HEADROOM * args.max_new_tokens,
             # Pre-seeded rather than built in `_finish`, so it carries its own `kind`:
             # a gate without one would read as `verdict` to any consumer that defaults.
             "kind": "validity",
             "skipped": True, "passed": None}
    if args.rl:
        manifest["gates"].append(drift)
    # Before the eval arms, and for BOTH algos: `write_manifest` otherwise runs only
    # inside `_finish`, so a run killed anywhere earlier left no manifest and
    # `tilerl ledger` could not see it. Measured on cpu: SIGTERM at step 6 of a grpo
    # run left 12 rollout rows and no manifest; the same kill on an opd run left no
    # run DIRECTORY at all. `_finish` overwrites this with the finished manifest.
    write_manifest(runs_root(), manifest)
    # No eval configured: the eval gates have nothing to measure and must be
    # explicitly skipped, not left to the vacuous-pass rule.
    if not args.eval_mmlu and not args.eval_gsm8k:
        manifest["gates_skip_after"] = True
    evals("before")  # LoRA B is zero at init: the base model's score
    if args.steps == 0:
        evals("after")
        manifest["engine"] = engine.config  # re-read: see the comment at the other _finish
        return _ledger.finish_run(manifest, args.json)
    _refuse_short_rollouts(mean_len.get("before"), args.max_new_tokens,
                           args.allow_short_rollouts)
    # The weights behind the best curve point. Every intermediate policy is otherwise
    # destroyed: `AdamW.step_one` ends in `p.copy_()` (in place, which is what lets the
    # engine keep its captured graphs), so a run that peaks mid-way can neither stop there
    # nor roll back to it. Measured 2026-09-08: score 87.4 -> 93.2 -> 93.4 -> 82.4 -> 91.2,
    # so the run shipped 91.2 and the 93.4 it had reached was gone. Out here, not in the RL
    # branch, because the save site below is shared with opd.
    best: dict = {}
    if args.rl:
        if rows:
            gold = {tuple(p): r["answer"] for p, r in zip(prompts, rows)}
            match = MATCHERS[args.reward]
            reward = _length_aware(match, gold, tok, args.length_penalty,
                                   max(int(args.max_new_tokens), 1))

            # binary correctness before the length term — the tied_correctness input
            def correctness(prompt, completion):
                text = tok.decode([int(t) for t in completion])
                return float(match(text, gold[tuple(int(t) for t in prompt)]))
        else:
            # No length term: this reward is a RATE, so its expectation does not grow with
            # length and the defect above is absent by construction -- a longer completion
            # earns no more, so nothing pressures the policy to lengthen. True even if this
            # path stops being the smoke-test one.
            half = cfg.vocab_size // 2

            def reward(prompt, completion):
                return sum(1 for t in completion if t < half) / max(len(completion), 1)

        tiebreak = _judge_tiebreak(engine, tok, params) if args.judge else None

        # `steps_to_score x seconds_per_step` needs the step at which a score was crossed,
        # and gsm8k_before/after cannot say which step that was. So: score a fixed held-out
        # subset every `--eval-every` steps and keep (step, score, cumulative_secs).
        # cumulative_secs is summed rather than `secs_per_step_median x step` because the
        # step length is not constant within a run -- it changes once a run hits the rollout
        # cap. The threshold stays at the READING end: the curve records the scores, and
        # which one counts as "the score" is not the ledger's business.
        #
        # `secs` is TRAINING time and excludes this scoring: grpo_loop stops its clock
        # at `train.py:496`, before the yield, so the probe's own cost is outside every
        # point. That is the quantity `time_to_score` wants -- production does not pay the
        # probe -- but it means the curve's last point is `secs_total`, not wall clock.
        # Scoring at the yield is also the only correct place: grpo_loop calls
        # `invalidate_weights()` before yielding (`:492`), so the eval sees the policy the
        # step just produced, with the decode graph already dropped.
        # Sliced from `eval_rows`, which `--eval-n` has already capped, so asking for more
        # curve rows than eval rows quietly scores fewer. `curve["n"]` records the real
        # size, but a reader looking at the run WHILE it happens sees only this line.
        curve_rows = _curve_rows(eval_rows, args.eval_curve_n, args.eval_curve_seed)
        if curve_rows and args.eval_every and len(curve_rows) < args.eval_curve_n:
            log(f"curve subset is {len(curve_rows)} rows, not the {args.eval_curve_n} asked "
                f"for: --eval-n {args.eval_n} caps it")
        # Early stopping is a verdict about the curve: with no curve points the switch can
        # never fire, so refuse at startup instead of running to --steps behind dead code.
        if args.patience and not (args.eval_every and curve_rows):
            sys.exit("--patience early-stops on the eval curve, but this run produces no "
                     "curve points: pass --eval-every and an eval set (--eval-curve-n rows).")
        curve: list[dict] = []
        # `train_secs`, not `elapsed`: the timings loop below rebinds `elapsed` on every
        # step, so an accumulator by that name silently became the last timing value --
        # measured, the curve read 0.143 s at step 4 against 0.148 at step 2, a
        # cumulative figure going DOWN.
        train_secs = 0.0
        # Patience is in curve POINTS, not steps: one unit is one `--eval-every` interval.
        # 0 never stops -- the default; flipping it on needs the seed-1 verdict.
        early = EarlyStop(args.patience)

        def score_curve(step: int) -> bool:
            nonlocal best
            # `eval_secs` per point, so the "keep the scoring under 5% of a step" criterion
            # is a fact checkable AFTER a run rather than a guess before one. Estimating it
            # from another config's eval would extrapolate across n, generation length and
            # batch shape -- and an estimated default is harder to overturn than no default,
            # because it looks calibrated. Same idiom as `eval_{tag}_secs` (#309).
            t_eval = time.perf_counter()
            # `per_problem` for the LENGTHS, not just the count. A score is not
            # interpretable without them: the same 60% can be a policy answering in 300
            # tokens or one being cut off, and 2026-09-04 shipped a 39.0% that was the
            # cap's number rather than the policy's. `at_cap` is the reading that
            # distinguishes them, so it travels with every point.
            per: list = []
            append = _eval_row_appender(manifest["id"], f"curve-{step}")
            c, n, ntok = gsm8k_accuracy(engine, tok, curve_rows, eval_params,
                                        concurrency=_EVAL_CONCURRENCY, thinking=thinking,
                                        match=MATCHERS[args.reward], per_problem=per,
                                        on_row=lambda r: append(dict(r, dataset="gsm8k")))
            eval_secs = time.perf_counter() - t_eval
            at_cap = sum(p["tokens"] >= args.eval_max_new_tokens for p in per)
            # The rows stream to disk as they land (on_row above), because the whole
            # point of the curve is comparing its points to each other and that
            # comparison is PAIRED: every point scores the same `curve_rows`. Unpaired,
            # adjacent points carry a 1.90 pt difference SE at n=500; paired at 5%
            # discordant it is 1.00 pt, and "has it stopped rising" is exactly a
            # question about a difference smaller than the arms. P1 fell back to the
            # unpaired interval for want of these rows, and `per` was being built here
            # and dropped. Streaming also means a killed run keeps the points that
            # finished.
            # Churn vs the previous point: the run's own instrument reading, recorded per
            # point so a "these two points differ by N questions" claim has N's measurement
            # beside it. Zero new evals -- these rows were just written and the previous
            # point's are in the same run dir. First point: null, not 0.
            churn = churn_dir = None
            if curve:
                prev_rows = _read_eval_rows(manifest["id"], f"curve-{curve[-1]['step']}")
                pair = curve_churn(prev_rows, per)
                if pair is None:
                    log(f"  curve step {step}: churn null -- {len(prev_rows)} rows at step "
                        f"{curve[-1]['step']} vs {len(per)} now, not comparable")
                else:
                    churn, churn_dir = pair[0] + pair[1], list(pair)
            # The first point compiles the eval's shapes and every later one hits the cache,
            # so its eval_secs is 5.6x the steady state and --eval-curve-n is calibrated off
            # point two -- recorded, because the curve is a list of equal-looking dicts.
            #
            # `tied` is the RUN's tie fraction over the steps since the previous point, not
            # anything about the eval. It is the only way to tell a plateau where the policy
            # stopped improving from one where its groups stopped disagreeing: score flat
            # with tied rising is the usable set self-consuming, both flat is another cause.
            # A whole-run mean cannot separate them -- the 2026-09-05 P1 run went 0.50 ->
            # 0.87 across its own steps while reward rose with it, so the two are confounded
            # in any single aggregate.
            since = [h[3] for h in hist[curve[-1]["step"] if curve else 0:]]
            curve.append({"step": step, "correct": c, "total": n, "score": c / max(n, 1),
                          "secs": round(train_secs, 3), "eval_secs": round(eval_secs, 3),
                          "mean_len": round(ntok / max(n, 1), 1), "at_cap": at_cap,
                          "tied": round(statistics.mean(since), 4) if since else None,
                          "churn": churn, "churn_dir": churn_dir,
                          "jit": not curve})
            # SIGNIFICANTLY greater, not merely greater. Measured 2026-09-08: re-scoring
            # one fixed set of weights across processes at temperature 0.0 moved 438/500
            # to 437/500, so this eval's own floor is 0.2 pt -- and the run's step-50 point
            # led step 25 by exactly one question. Taking the numerically higher point
            # would have bought 501.2 s of extra training for a reading inside the
            # instrument. A tie goes to the earlier point, which is not an arbitrary
            # tie-break: `time_to_score` is the objective, so when two options are the same
            # score the cheaper one wins, and "the same" is defined by the measured floor.
            # The criterion lives in `ledger.new_best_point` next to the SE formulas, so
            # the run and its post-hoc readers cannot drift apart -- its `__main__` check
            # runs this exact curve, plus the one-question case, the paired-vs-unpaired
            # case, and each one's negative control.
            #
            # The width is PAIRED: every curve point scores the same `curve_rows`, and the
            # best point's per-problem rows are on disk from when it was scored. The
            # unpaired width is 1.9x wider here (8.6% discordant, measured 2026-09-08), so
            # it would make this criterion never fire on a slow rise -- a selection that
            # always keeps the first point and does not say so. Rows missing (old runs)
            # fall back to the conservative width, marked in `se_kind` so nobody reads a
            # conservative "not greater" as "the two points are the same". With --patience
            # on, a missing width refuses instead: stopping on a width-less curve decides
            # without a sampling width, and the unpaired fallback is too wide to ever fire -- both silent.
            if best:
                rows = _read_eval_rows(manifest["id"], f"curve-{best['step']}")
                se = paired_se(rows, per)
                se_kind = "paired" if se is not None else "unpaired (conservative)"
                require_paired_width(se, args.patience, best["step"])
            else:
                se = se_kind = None  # first point: no comparison installed it
            replaced = new_best_point(curve[-1], best or None, se)
            if replaced:
                # `mean_len` and `tok_per_correct` ride with the snapshot so a downstream
                # consumer can trade score against answer cost -- the 2026-09-05 run bought
                # most of its +6% as 2.74x shorter answers, and score alone cannot show that.
                best = {"step": step, "score": curve[-1]["score"],
                        "mean_len": curve[-1]["mean_len"],
                        "tok_per_correct": round(ntok / c, 1) if c else None,
                        "se_kind": se_kind,
                        "tensors": {k: v.detach().to("cpu", copy=True)
                                    for k, v in trainable.items()}}
            # A significant decline stops immediately and spends no patience: the collapse
            # is the event this feature exists for, and stopping never loses anything --
            # the best snapshot is kept either way. seed 0's recovery (412 -> 456) still
            # ended below the peak (467), so waiting for it bought less than keeping it.
            reason = early.update(replaced, significant_decline(curve[-1], best, se))
            if reason:
                manifest["early_stopped"] = {"at_step": step, "kept_step": best["step"],
                                             "patience": args.patience, "reason": reason}
                why = ("a significant decline" if reason == "decline"
                       else f"{args.patience} curve points without a significant gain")
                log(f"  early stop ({reason}) at step {step}: {why}; keeping step "
                    f"{best['step']} ({100 * best['score']:.1f}%)")
                return True
            log(f"  curve step {step}: {c}/{n} = {100 * c / max(n, 1):.1f}% "
                f"tied {curve[-1]['tied']} "
                f"at {train_secs:.1f}s cumulative, mean {ntok / max(n, 1):.0f} tok, "
                # tokens/correct, the ratio the before/after arms already log (`per`, :718).
                # It separates two things a score cannot: the 2026-09-05 run moved
                # tokens/correct 394.0 -> 143.8 (2.74x) while accuracy moved 88.0 -> 93.6
                # (+6%), so most of what that RL bought was shorter answers. A curve read on
                # score alone records that as "the rate of learning to be right".
                # Derived, not stored: it is mean_len * total / correct from fields already
                # in the point, and a second copy in the dict could disagree with them.
                f"{ntok / c if c else float('nan'):.1f} tok/correct, "
                f"{at_cap}/{n} at cap, scored in {eval_secs:.1f}s")
            return False

        hist = []
        rollouts: list = []
        written = 0
        for i, (r, ce, secs, tied, ntok, timings, width, tied_c) in enumerate(
                train_mod.grpo_loop(engine, model, prompts, reward, args.steps, backend, optimizer,
                                    group=args.group, prompts_per_step=args.prompts_per_step,
                                    sampling=params, seed=args.seed,
                                    trainable=trainable, micro=args.micro,
                                    tiebreak=tiebreak, recapture_graph=True,
                                    per_rollout=rollouts, decode=tok.decode,
                                    correctness_fn=correctness if rows else None,
                                    length_penalty=args.length_penalty,
                                    length_cap=max(int(args.max_new_tokens), 1))):
            hist.append((r, ce, secs, tied, ntok, tied_c))
            train_secs += secs
            written = _write_rollout_rows(manifest["id"], rollouts, written)
            if (curve_rows and args.eval_every and (i + 1) % args.eval_every == 0
                    and score_curve(i + 1)):
                break
            for phase, elapsed in timings.items():
                manifest["metrics"][phase] = manifest["metrics"].get(phase, 0.0) + elapsed
            tied_c_str = f"  tied_c {tied_c:.2f}" if tied_c is not None else ""
            log(f"step {i + 1:4d}/{args.steps}  reward {r:.4f}  ce {ce:.4f}  "
                f"tied {tied:.2f}{tied_c_str}  tok {ntok:.0f}  width {width}  {secs:.1f}s  "
                f"rollout {timings['rollout_secs']:.3f}s  "
                # .get: rl_step writes these, and a test or caller that substitutes it
                # still gets a log line rather than a KeyError mid-run.
                f"fwd {timings.get('forward_secs', 0.0):.3f}s  "
                f"bwd {timings.get('backward_only_secs', 0.0):.3f}s  "
                f"optimizer {timings['optimizer_secs']:.6f}s  "
                f"other {timings.get('other_secs', 0.0):.3f}s", flush=True)
            if len(hist) >= 5 and not args.allow_short_rollouts:
                mean = statistics.mean(h[4] for h in hist[-5:])
                drift.update(value=mean, step=i + 1, skipped=False,
                             passed=mean <= drift["threshold"])
                manifest["metrics"]["rollout_window_mean"] = mean
                if not drift["passed"]:
                    drift["reason"] = (
                        f"error: at step {i + 1} the last 5 steps average {mean:.1f} "
                        f"completion tokens but --max-new-tokens is {args.max_new_tokens}. "
                        f"Rollouts risk truncation before they answer. Raise the cap above "
                        f"{mean / _ROLLOUT_HEADROOM:.0f}, pick an easier task, or pass "
                        f"--allow-short-rollouts if the truncation is deliberate.")
                    log(drift["reason"], flush=True)
                    break
        # Windowed means, not hist[0] vs hist[-1]: per-step reward moves with the
        # sampled prompt, so two single steps compare two draws, not two policies
        # (tests/test_rl.py::test_grpo_loop_raises_reward uses the same windows).
        w = max(1, len(hist) // 4)
        tc = [h[5] for h in hist if h[5] is not None]
        manifest["metrics"].update(
            steps_completed=len(hist),
            reward_first=statistics.mean(h[0] for h in hist[:w]),
            reward_last=statistics.mean(h[0] for h in hist[-w:]),
            ce_last=hist[-1][1],
            secs_per_step_median=statistics.median(h[2] for h in hist),
            secs_total=sum(h[2] for h in hist),
            tied_group_fraction=statistics.mean(h[3] for h in hist),
            # Binary-correctness tie fraction, pre-length-term. `tied` is structurally
            # 0 at lam>0; this is the validity gate's real input.
            tied_correctness=statistics.mean(tc) if tc else None,
            # --judge drives tied_group_fraction toward 0 by construction, so it
            # cannot report a bad judge. Length is the signal that can.
            tokens_first=statistics.mean(h[4] for h in hist[:w]),
            tokens_last=statistics.mean(h[4] for h in hist[-w:]))
        manifest["metrics"]["length_reward_r"] = _within_group_r(rollouts)
        # Its own top-level key, not a metric: `format_run` prints every metric inline on
        # one `tilerl ledger` row, so a curve in there would push the row past a screen.
        if curve:
            manifest["eval_curve"] = {"n": len(curve_rows), "every": args.eval_every,
                                      "points": curve}
    else:
        losses = train_mod.opd_loop(engine, model, prompts, args.steps, backend, optimizer,
                                    seed=args.seed, trainable=trainable, sampling=params,
                                    recapture_graph=True)
        for i, loss in enumerate(losses):
            log(f"step {i + 1:4d}/{args.steps}  loss {loss:.4f}")
        manifest["metrics"]["ce_last"] = losses[-1]
    if torch.cuda.is_available():  # the number the group size is really bounded by
        manifest["metrics"]["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        log(f"peak allocated {manifest['metrics']['peak_gib']:.2f} GiB")
    # Before the after-eval, not after it: a gsm8k_after that beats its own baseline is
    # the run's whole claim, and without the weights that produced it nobody can check
    # whether the metric moved or the reward was gamed. An eval that dies still leaves
    # the adapter behind.
    from safetensors.torch import save_file

    d = Path(runs_root()) / manifest["id"]
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in trainable.items()},
              str(d / "adapter.safetensors"))
    manifest["artifacts"]["adapter"] = "adapter.safetensors"
    log(f"adapter {sum(v.numel() for v in trainable.values()) / 1e6:.1f}M params -> {d}")
    # `best_curve_point`, never `best_step`: the snapshot is taken inside `score_curve`, so
    # its resolution is `--eval-every`. At 25 a true peak at 40 is recorded as 50. This is
    # the best point we LOOKED AT, and a name promising the best step would be read as an
    # optimum.
    if args.rl and best:
        save_file({k: v.contiguous() for k, v in best.pop("tensors").items()},
                  str(d / "adapter-best.safetensors"))
        manifest["artifacts"]["adapter_best"] = "adapter-best.safetensors"
        manifest["best_curve_point"] = {**best, "every": args.eval_every}
        log(f"adapter-best step {best['step']} score {100 * best['score']:.1f}% "
            f"(best of {len(curve)} curve points, resolution {args.eval_every} steps)")
    if drift["passed"] is not False:
        evals("after")
    else:
        # The after-arm never ran, so `mmlu_after`/`gsm8k_after` are None -- and
        # `_finish`'s `v is None or ...` would score both gates PASS on a run that
        # measured neither. Mark them skipped so the manifest says "not measured".
        manifest["gates_skip_after"] = True
    # Re-read, not the build-time copy: `_graph_for` sets `_decode_graph_on = False`
    # in its `except` on a capture failure, so a snapshot taken at build time can
    # record graph-on for a run that decoded eagerly -- and the whole point of this
    # block is that a wall clock is read against it.
    manifest["engine"] = engine.config
    return _ledger.finish_run(manifest, args.json)


def _judge_tiebreak(engine, tok, params):
    """Rank rollouts the binary reward cannot separate, using the policy as its own judge.

    `answer_match` decides first and the judge only reorders inside the all-pass or
    all-fail subgroup (judge.py enforces that split), so no judgement can lift a wrong
    answer over a right one. All C(K,2) pairs are generated in ONE batch and looked up,
    because judge_rewards asks pair by pair and 56 sequential round trips per step
    would cost more than the training step itself.
    """
    from dataclasses import replace

    from .eval import generate
    from .judge import judge_rewards
    from .prompt import render_chat

    sp = replace(params, temperature=0.0, max_new_tokens=4, max_think_tokens=0)

    def ask(q, a, b):
        return render_chat([("user",
            f"Problem:\n{q}\n\nTwo worked solutions.\n\n[A]\n{a}\n\n[B]\n{b}\n\n"
            "Which shows the better reasoning: clearer steps, no unjustified leaps, "
            "no wasted work? Reply with exactly one token: A or B or tie.")], False)

    def pick(t):
        t = (t or "").strip().upper()
        return "A" if t.startswith("A") else "B" if t.startswith("B") else "tie"

    def tiebreak(prompt, comps, passed):
        q = tok.decode([int(t) for t in prompt])
        texts = [tok.decode([int(t) for t in c]) for c in comps]
        pairs = [(i, j) for i in range(len(comps)) for j in range(i + 1, len(comps))]
        # Both orders for every pair: pair_verdict abstains unless the swapped call
        # agrees, which is the position-bias control and is not optional.
        prompts = [ask(q, texts[i], texts[j]) for i, j in pairs] + \
                  [ask(q, texts[j], texts[i]) for i, j in pairs]
        out = generate(engine, tok, prompts, sp, 8)
        n = len(pairs)
        seen = {(i, j): (pick(out[k]), pick(out[k + n])) for k, (i, j) in enumerate(pairs)}
        scores, _ = judge_rewards(list(range(len(comps))), passed,
                                  lambda a, b: seen[(a, b)] if (a, b) in seen
                                  else tuple(reversed(seen[(b, a)])))
        return scores

    return tiebreak
