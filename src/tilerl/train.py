"""Training on the hand-written tape: SFT, GRPO, on-policy distillation and
pretrain share ``_step``; serving and training share the model and weights.
# ponytail: CE is torch-eager glue; fold into a backend cross_entropy op when perf demands."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch

from .autograd import AdamW, RecordingBackend, Tape, clip_grad_norm, cosine_warmup
from .engine import SamplingParams
from .kv_cache import LinearStatePool, NoPrefixStore
from .model import save_hf


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


def _step(
    model: Any,
    input_ids: Any,
    backend: Any,
    optimizer: AdamW,
    trainable: dict[str, Any] | None,
    grad_fn: Any,
    micro: int = 0,
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
    param_ids = set(by_id)
    rows = micro if 0 < micro < b else b
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
            logits = model.forward(chunk, np.arange(t, dtype=np.int64), kv,
                                   RecordingBackend(backend))
        loss, grad_logits = grad_fn(logits, chunk, lo)
        if not math.isfinite(loss):
            return loss, {}
        return loss, tape.backward(grad_logits, on_grad=on_grad, needs=param_ids)

    if getattr(optimizer, "streams", False):
        # Every weight gradient coexisting is 50.1 GiB on the 27B: this
        # optimizer clips per update, so each gradient is applied and dropped
        # the moment backward finalizes it.
        optimizer.begin()
        seen = 0

        def _apply(tid: int, g: torch.Tensor) -> bool:
            nonlocal seen
            if tid in param_ids:
                optimizer.step_one(by_id[tid], g)
                seen += 1
            return True

        loss, _ = run(0, _apply)
        assert seen or not math.isfinite(loss), _NO_GRAD
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
    norm = clip_grad_norm(acc, 1.0)
    if math.isfinite(norm):
        optimizer.step(params.values(), acc)
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


def group_advantages(rewards: Any, group: int) -> "np.ndarray":
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
) -> float:
    """One policy-gradient step: the causal-CE gradient scaled per row by the
    advantage and zeroed on prompt/padding positions. ``input_ids`` [B,T] is
    prompt+completion right-padded; ``seq_lens`` [B] is each row's valid length
    (default T). Returns the batch cross-entropy as a diagnostic.
    # ponytail: single-update REINFORCE-with-baseline; add the PPO ratio+clip
    # when a rollout is reused for more than one step."""
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

    return _step(model, ids, backend, optimizer, trainable, grad_fn, micro)


def _require_on_policy(engine: Any) -> None:
    # A cached prefix or a captured decode graph samples from an earlier policy without raising.
    if engine._decode_graph_on is not False or not isinstance(engine._prefix, NoPrefixStore):
        raise ValueError("on-policy rollouts need build_engine(decode_graph=False, "
                         "prefix_store=NoPrefixStore()): a captured graph or a cached "
                         "prefix samples from an earlier policy")


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
) -> list[tuple[float, float, float, float]]:
    """GRPO: sample ``group`` completions per prompt in one engine batch, score
    them with ``reward_fn(prompt_ids, completion_ids) -> float``, take one
    policy-gradient step on the group-normalized advantages, in ``micro`` rows
    at a time (``0`` = the whole group at once). The engine that
    generates IS the model that trains, so it must be built with the prefix
    cache and decode graph off, and rollouts are drawn untruncated so the
    sampler is the policy the step differentiates. Returns per step
    ``(mean reward, cross-entropy, seconds, tied-group fraction)``.
    # ponytail: recapture the graph and drop the prefix entries after each
    # update instead of disabling both, once a rollout's decode cost matters."""
    _require_on_policy(engine)
    if optimizer is None:
        optimizer = AdamW(lr=1e-5)
    sampling = untruncated(sampling if sampling is not None
                           else SamplingParams(max_new_tokens=32))
    out: list[tuple[float, float, float, float]] = []
    for step in range(steps):
        t0 = time.perf_counter()
        prompt = np.asarray(prompts[step % len(prompts)], dtype=np.int64)
        # Identical seeds would make the group one sample repeated, every advantage zero.
        ids = [
            engine.submit(prompt.tolist(), replace(sampling, seed=seed + step * group + g))
            for g in range(group)
        ]
        done: dict[int, list[int]] = {}
        for _ in range(10000):
            engine.step()
            done.update(engine.poll())
            if all(i in done for i in ids):
                break
        else:  # pragma: no cover - engine bug, not a training path
            raise RuntimeError("grpo_loop: rollout did not finish within 10000 ticks")
        comps = [done[i] for i in ids]
        rewards = [float(reward_fn(prompt, c)) for c in comps]
        adv = group_advantages(rewards, group)
        tied = float((adv.reshape(-1, group) == 0).all(axis=1).mean())
        # Right-pad to one rectangle; the advantage mask keeps padding out of the gradient.
        width = max(len(c) for c in comps)
        batch = np.stack([
            np.concatenate([prompt, np.asarray(c, dtype=np.int64),
                            np.zeros(width - len(c), dtype=np.int64)])
            for c in comps
        ])
        plens = np.full(group, len(prompt), dtype=np.int64)
        slens = np.array([len(prompt) + len(c) for c in comps], dtype=np.int64)
        ce = rl_step(model, batch, adv, plens, backend, optimizer, trainable=trainable,
                     seq_lens=slens, micro=micro)
        out.append((float(np.mean(rewards)), ce, time.perf_counter() - t0, tied))
    return out


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
) -> list[float]:
    """On-policy distillation: the teacher engine generates a completion, the
    student takes one :func:`train_step` on prompt + completion. With
    ``trainable`` (LoRA adapters) the teacher is the same model generating
    under an EMA of the adapters, so only adapter-sized memory is duplicated."""
    if trainable is not None:  # a frozen teacher with no adapters cannot go stale
        _require_on_policy(teacher_engine)
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
        finished: dict[int, list[int]] = {}
        for _ in range(10000):
            teacher_engine.step()
            finished = teacher_engine.poll()
            if rid in finished:
                break
        else:  # pragma: no cover - engine bug, not a training path
            raise RuntimeError("opd_loop: teacher did not finish within 10000 ticks")
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
