"""Training on the hand-written tape: SFT, GRPO, on-policy distillation and
pretrain share ``_step``; serving and training share the model and weights.
# ponytail: CE is torch-eager glue; fold into a backend cross_entropy op when perf demands."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .autograd import AdamW, RecordingBackend, Tape, clip_grad_norm, cosine_warmup
from .engine import RequestFailed, SamplingParams
from .kv_cache import LinearStatePool, NoPrefixStore
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

    import torch.distributed as dist

    h = hashlib.sha256("\n".join(order).encode()).digest()[:8]
    mine = torch.tensor([len(order), *h], dtype=torch.float64)
    parts = [torch.empty_like(mine) for _ in range(backend.dp_world)]
    dist.all_gather(parts, mine, group=backend._dp_pg)
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


def dense_causal_page_mass(q: torch.Tensor, k: torch.Tensor, block: int = 16
                           ) -> torch.Tensor:
    """Dense causal attention mass pooled per key PAGE, averaged over query
    heads — the warm-up teacher for a LONG sequence (8k-32k), where the naive
    [t,t] score matrix will not fit. Streams key pages one at a time: an online
    softmax pass gets each query's log-sum-exp, a second pass normalises each
    page's numerator. Memory O(t*block), compute the same GEMM as attention.

    ``q`` [b,t,hq,d], ``k`` [b,t,hkv,d] -> [b, t, t//block], each query row
    L1-normalised over pages strictly before it (window excluded later by
    :func:`page_mass_target`). Mathematically identical to
    ``_dense_causal_mass`` pooled by block, which is its CPU gate."""
    b, t, hq, d = q.shape
    rep = hq // k.shape[2]
    k_exp = k.repeat_interleave(rep, dim=2)
    scale = d ** -0.5
    n_pages = t // block
    q_idx = torch.arange(t, device=q.device)
    neg_inf = torch.finfo(q.dtype).min

    def _page_scores(p):
        sl = slice(p * block, min((p + 1) * block, t))
        kj = q_idx[sl]
        s = torch.einsum("bqhd,bkhd->bhqk", q, k_exp[:, sl]) * scale  # [b,hq,t,blk]
        valid = q_idx[None, None, :, None] >= kj[None, None, None, :]
        return s.masked_fill(~valid, float("-inf")), valid

    # Pass 1: online softmax -> per-(b,hq,q) running max and log-sum-exp.
    neg_inf = torch.finfo(q.dtype).min
    run_max = torch.full((b, hq, t), neg_inf, dtype=torch.float32, device=q.device)
    run_sum = torch.zeros((b, hq, t), dtype=torch.float32, device=q.device)
    for p in range(n_pages):
        s, valid = _page_scores(p)
        anyv = valid.any(dim=-1)                              # [b,hq,t]
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
                      run_max + run_sum.clamp_min(1e-30).log(), run_max)  # [b,hq,t]

    # Pass 2: normalised causal mass landing on each key page, mean over heads.
    page_mass = torch.zeros(b, t, n_pages, dtype=torch.float32, device=q.device)
    lse_safe = torch.where(lse > neg_inf / 2, lse, torch.zeros_like(lse))
    for p in range(n_pages):
        s, valid = _page_scores(p)
        mass_h = torch.where(valid, torch.exp(s - lse_safe[..., None]), 0.0).sum(dim=-1)
        page_mass[:, :, p] = mass_h.mean(dim=1)
    return page_mass


def indexer_warmup_step(
    model: Any,
    input_ids: Any,
    backend: Any,
    weights: dict[str, torch.Tensor],
    optimizer: AdamW,
    block: int = 16,
) -> float:
    """One indexer warm-up step. The frozen base supplies the teacher: a
    no-grad dense forward captures each source layer's input H and post-rope
    Q/K. Dense causal attention mass is pooled per page and the two indexer
    projection weights (``iq`` [ih,hidden,di], ``ik`` [ih,d_kv,di]) learn to
    match it via the ``indexer_warmup`` tape op. Returns the scalar KL."""
    from .sparse_index import (
        WINDOW_PAGES,
        exclude_window_renorm,
        index_source_groups,
        indexer_warmup_loss,
    )

    ids = torch.as_tensor(input_ids, dtype=torch.long, device=backend.device)
    b, t = ids.shape
    n_pages_tok = t // block
    if n_pages_tok <= WINDOW_PAGES:
        raise ValueError(f"warm-up needs > {WINDOW_PAGES} pages, got {n_pages_tok} from T={t}")

    full_layers = list(model.cfg.full_attn_layers)
    sources, _groups = index_source_groups(len(full_layers)) if len(full_layers) >= 4 \
        else (list(range(len(full_layers))), None)
    source_set = {full_layers[s] for s in sources}

    captured: list = []
    model.index_capture = captured
    try:
        kv = _training_kv(model, b, t, device=backend.device)
        with torch.no_grad():
            model.forward(ids, torch.arange(t, device=backend.device), kv, backend)
    finally:
        model.index_capture = None

    hkv, d_kv = model.cfg.num_kv_heads, model.cfg.head_dim
    ih = weights["ik"].shape[0]
    if ih != hkv:
        raise ValueError(f"index heads {ih} must equal KV heads {hkv} on the tiny warm-up")
    n_pages = torch.full((b,), n_pages_tok, dtype=torch.long)

    # Stack captured SOURCE layers on the L_src axis (one entry per source layer).
    cap = [c for c in captured if c[0] in source_set]
    cap.sort(key=lambda c: c[0])
    H = torch.stack([c[1][:, :n_pages_tok * block] for c in cap], dim=1)   # [b,L,t,hid]
    K = torch.stack([c[3] for c in cap], dim=1)                            # [b,L,t,hkv,d]
    # Long-sequence teacher streamed per page (never a [t,t] matrix), then the
    # window excluded and the page distribution renormalised.
    mass = torch.stack([dense_causal_page_mass(c[2], c[3], block) for c in cap], dim=1)

    # Per-page K = mean of the block's token Ks: [b,L,pages,hkv,d].
    k_pages = K.reshape(b, len(cap), n_pages_tok, block, hkv, d_kv).mean(dim=3)
    target = exclude_window_renorm(mass, n_pages, WINDOW_PAGES)

    iq_w, ik_w = weights["iq"], weights["ik"]
    with Tape() as tape:
        loss = indexer_warmup_loss(H, k_pages, iq_w, ik_w, target, n_pages, WINDOW_PAGES)
    grads = tape.backward(torch.ones((), device=backend.device),
                          needs={id(iq_w), id(ik_w)})
    optimizer.step([iq_w, ik_w], grads)
    return float(loss.detach())


def indexer_warmup(model: Any, backend: Any, steps: int, seed: int = 0,
                   seq_len: int = 256, batch: int = 1, lr: float = 0.02) -> list[float]:
    """Run the learned-indexer KL warm-up on the frozen base for ``steps`` and
    return the per-step loss. Tiny/CPU path; the 27B card run is pending-remote."""
    gen = torch.Generator(device=backend.device).manual_seed(seed)
    hkv, d_kv, hidden = model.cfg.num_kv_heads, model.cfg.head_dim, model.cfg.hidden_size
    di = min(16, d_kv)  # small indexer head on tiny
    weights = {
        "iq": (0.1 * torch.randn(hkv, hidden, di, generator=gen, device=backend.device)),
        "ik": (0.1 * torch.randn(hkv, d_kv, di, generator=gen, device=backend.device)),
    }
    opt = AdamW(lr=lr)
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, model.cfg.vocab_size, (batch, seq_len),
                            generator=gen, device=backend.device)
        losses.append(indexer_warmup_step(model, ids, backend, weights, opt))
    return losses


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
