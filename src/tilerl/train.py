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
from .engine import SamplingParams
from .kv_cache import LinearStatePool, NoPrefixStore
from .model import save_hf

_MAX_TICKS = 10000


def _drain(engine: Any, ids: list[int], what: str) -> dict[int, list[int]]:
    """Tick until every id has finished. Accumulates: poll() only returns the
    requests that finished on that tick, so a single assignment loses the rest."""
    done: dict[int, list[int]] = {}
    for _ in range(_MAX_TICKS):
        engine.step()
        done.update(engine.poll())
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
    if getattr(backend, "tp_world", 1) > 1:
        from .tensor_parallel import is_sharded

        sharded_ids = {id(p) for k, p in params.items() if is_sharded(k)}
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
        with torch.no_grad(), tape:
            # A vocab-parallel head keeps its shard here: the gathered row is
            # [B, T, vocab] f32 (1.89 GiB at B=8 T=256 on the 27B) and
            # cross_entropy_loss_grad reduces it sharded instead.
            logits = model.forward(chunk, np.arange(t, dtype=np.int64), kv,
                                   RecordingBackend(backend),
                                   sharded_logits=getattr(backend, "tp_world", 1) > 1)
        loss, grad_logits = grad_fn(logits, chunk, lo)
        if not math.isfinite(loss):
            return loss, {}
        return loss, tape.backward(grad_logits, on_grad=on_grad, needs=param_ids)

    if getattr(optimizer, "streams", False):
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
                optimizer.step_one(by_id[tid], g)
                if timings is not None:
                    timings["optimizer_secs"] += time.perf_counter() - t_update
                seen += 1
            return True

        loss, _ = run(0, _apply)
        assert seen or not math.isfinite(loss), _NO_GRAD
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
    if timings is not None:
        timings["optimizer_secs"] += time.perf_counter() - t_update
    return total


def train_step(
    model: Any,
    input_ids: Any,
    backend: Any,
    optimizer: AdamW,
    trainable: dict[str, Any] | None = None,
    micro: int = 0,
) -> float:
    """One SFT step: causal cross-entropy on ``input_ids``. Returns the loss."""
    b = np.asarray(input_ids).shape[0]

    def grad_fn(logits, chunk, lo):
        loss, grad = backend.cross_entropy_loss_grad(logits, chunk)
        # CE averages over this chunk's rows; rescale to the batch's.
        return loss, grad.mul_(len(chunk) / b)

    return _step(model, input_ids, backend, optimizer, trainable, grad_fn, micro)


def group_advantages(rewards: Any, group: int) -> np.ndarray:
    """``(r - mean) / std`` within each group of ``group`` consecutive rollouts;
    a tied group yields zeros (no signal, no division by ~0)."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group)
    std = r.std(axis=1, keepdims=True)
    adv = (r - r.mean(axis=1, keepdims=True)) / np.where(std > 1e-8, std, 1.0)
    return np.where(std > 1e-8, adv, 0.0).reshape(-1)


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
        # Includes the recorded forward, loss and gradient accumulation.
        timings["backward_secs"] = time.perf_counter() - t0 - timings["optimizer_secs"]
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
) -> Iterator[tuple[float, float, float, float, float, dict[str, float]]]:
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
    # ponytail: recapture the graph and drop the prefix entries after each
    # update instead of disabling both, once a rollout's decode cost matters."""
    _require_on_policy(engine, recapture_graph, clear_prefix)
    if recapture_graph or clear_prefix:
        # Whatever the engine cached before this loop was built under other weights.
        engine.invalidate_weights()
    if optimizer is None:
        optimizer = AdamW(lr=1e-5)
    sampling = untruncated(sampling if sampling is not None
                           else SamplingParams(max_new_tokens=32))
    for step in range(steps):
        t0 = time.perf_counter()
        prompt = np.asarray(prompts[step % len(prompts)], dtype=np.int64)
        # Identical seeds would make the group one sample repeated, every advantage zero.
        ids = [
            engine.submit(prompt.tolist(), replace(sampling, seed=seed + step * group + g))
            for g in range(group)
        ]
        done = _drain(engine, ids, "grpo_loop rollout")
        timings = {"rollout_secs": time.perf_counter() - t0}
        comps = [done[i] for i in ids]
        rewards = [float(reward_fn(prompt, c)) for c in comps]
        # A binary reward stops producing gradient once the policy clears the task.
        # `tied` is that fraction and is the run's health metric: 72% at the 256 cap,
        # 88.7% at 2048. Do not predict it with p**group -- a tie is all-SAME, not
        # all-correct, and eight rollouts on one prompt behave like 3.4 independent
        # ones (wins/2026-09-04-the-cap-was-the-gradient.md). `tiebreak` reorders
        # WITHIN the all-pass and all-fail subgroups so those steps carry signal; it
        # never crosses the two, so nothing it says can lift a wrong answer over a
        # right one.
        if tiebreak is not None:
            rewards = tiebreak(prompt, comps, [r > 0.5 for r in rewards])
        adv = group_advantages(rewards, group)
        tied = float((adv.reshape(-1, group) == 0).all(axis=1).mean())
        # One rectangle of a FIXED width: TileLang specializes per shape, so a
        # data-dependent width JITs a fresh kernel set per step (measured on tiny:
        # 37.7 s for a new width against 71 ms for a repeat). prompt + max_new_tokens
        # is known before the rollout; seq_lens carries the true lengths and the
        # advantage mask keeps the padding out of the gradient.
        gen = max(int(sampling.max_new_tokens), max(len(c) for c in comps))
        batch = np.stack([
            np.concatenate([prompt, np.asarray(c, dtype=np.int64),
                            np.zeros(gen - len(c), dtype=np.int64)])
            for c in comps
        ])
        plens = np.full(group, len(prompt), dtype=np.int64)
        slens = np.array([len(prompt) + len(c) for c in comps], dtype=np.int64)
        ce = rl_step(model, batch, adv, plens, backend, optimizer, trainable=trainable,
                     seq_lens=slens, micro=micro, timings=timings)
        if recapture_graph or clear_prefix:
            # After the update, not before the next rollout: a caller that stops
            # iterating must not leave the engine holding graphs traced on weights
            # that no longer exist.
            engine.invalidate_weights()
        yield (float(np.mean(rewards)), ce, time.perf_counter() - t0, tied,
               float(np.mean([len(c) for c in comps])), timings)


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
