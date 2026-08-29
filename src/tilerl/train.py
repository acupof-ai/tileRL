"""Training: train_step + opd_loop on top of the hand-written tape.

Same model, same engine, same weights as serving (训练推理一体): the student is
a ``tilerl.model.Model`` trained in place; the teacher is the same model class
driven by a ``tilerl.engine.Engine`` — frozen, generating on-policy.

``train_step`` runs the model forward under a :class:`~tilerl.autograd.Tape`
(via a :class:`~tilerl.autograd.RecordingBackend` proxy so the backend stays
tape-unaware), computes causal cross-entropy and its logit gradient outside
the tape, replays the tape backward, clips grads to max-norm 1.0, and applies
AdamW — skipping the step when the loss or the global grad norm is non-finite
(agent-infer's ``finite_optimizer_step``). ``opd_loop`` is on-policy
distillation: the frozen teacher generates a completion for a prompt, the
student does one ``train_step`` on the full (prompt + completion) sequence.
EMA self-teacher is day-2 — see ``agent-infer/crates/train/src/ema_self_teacher.rs``.

CE is torch-eager glue (softmax through the backend op; one-hot/mean in torch
— no backend cross_entropy op exists yet).
# ponytail: fold CE into a backend cross_entropy op when perf demands.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .autograd import AdamW, RecordingBackend, Tape, clip_grad_norm, cosine_warmup

__all__ = ["train_step", "rl_step", "group_advantages", "grpo_loop",
           "opd_loop", "pretrain", "JsonlDataset"]


# ---------------------------------------------------------------------------
# Training KV (dense, no paged pool)
# ---------------------------------------------------------------------------


def _training_kv(
    model: Any, batch_size: int, seq_len: int, device: "torch.device | str | None" = None
):
    """Dense-layout KV for a training forward.

    The model's full-attn layers call ``backend.attention`` (dense GQA) when
    ``kv.dense`` is set — no paged pool indirection, so the tape's id()-based
    grad chain is unbroken. The GDN layers still need a LinearStatePool for
    the recurrent state (one slot per sequence).

    ``device`` must match the backend that will run the forward (the pool
    stores states the kernel reads/writes); it defaults to the global backend
    device for callers that do not thread one through.
    """
    from types import SimpleNamespace

    from .kv_cache import LinearStatePool

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


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


def _step(
    model: Any,
    input_ids: Any,
    backend: Any,
    optimizer: AdamW,
    tape: Tape | None,
    trainable: dict[str, Any] | None,
    grad_fn: Any,
) -> float:
    """Forward under a tape, ``grad_fn`` for the logit gradient, backward, clip,
    AdamW. SFT and RL differ ONLY in ``grad_fn``; everything else — the tape,
    the frozen-base filter, the finite-step rejection — is shared.

    The step is SKIPPED (params untouched) when the loss or the pre-clip grad
    norm is non-finite (agent-infer's ``finite_optimizer_step``).
    """
    input_ids = np.asarray(input_ids, dtype=np.int64)
    b, t = input_ids.shape
    positions = np.arange(t, dtype=np.int64)
    model.params = backend.materialize(model.params)
    kv = _training_kv(model, b, t, device=backend.device)
    if tape is None:
        tape = Tape()

    with torch.no_grad(), tape:
        logits = model.forward(input_ids, positions, kv, RecordingBackend(backend))

    loss, grad_logits = grad_fn(logits, input_ids)
    if not math.isfinite(loss):
        return loss

    grads = tape.backward(grad_logits)
    #: ``trainable`` = the subset that gets gradients (LoRA adapters); the rest
    #: is frozen, which is what keeps the 27B inside one card.
    params = model.params if trainable is None else trainable
    param_ids = {id(p) for p in params.values()}
    assert param_ids & set(grads), (
        "train_step: tape produced no parameter gradients — either the "
        "recording seam is missing (backend ops not recorded), or a trainable "
        "tensor is not the one the forward read: materialize() rebuilds any "
        "param whose device/dtype differs, and the new object has a new id()"
    )
    # The GDN initial state is a tape leaf: its grad is not a parameter and
    # must not enter the clip norm.
    grads = {k: v for k, v in grads.items() if k in param_ids}
    norm = clip_grad_norm(grads, 1.0)
    if math.isfinite(norm):
        optimizer.step(params.values(), grads)
    return loss


def train_step(
    model: Any,
    input_ids: Any,
    backend: Any,
    optimizer: AdamW,
    tape: Tape | None = None,
    trainable: dict[str, Any] | None = None,
) -> float:
    """One SFT step: causal cross-entropy on ``input_ids``. Returns the loss."""
    return _step(model, input_ids, backend, optimizer, tape, trainable,
                 lambda logits, ids: backend.cross_entropy_loss_grad(logits, ids))


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------


def group_advantages(rewards: Any, group: int) -> "np.ndarray":
    """Group-relative advantages: within each group of ``group`` consecutive
    rollouts of the same prompt, ``(r - mean) / std``.

    This is what lets GRPO drop the critic — the group's own mean is the
    baseline, so there is no value network and no second set of weights.
    A group whose rewards are all equal has no signal and yields zeros rather
    than a division by ~0.
    """
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
    tape: Tape | None = None,
    trainable: dict[str, Any] | None = None,
    seq_lens: Any = None,
) -> float:
    """One policy-gradient step on sampled sequences.

    The objective is ``-mean_t A_row * log p(a_t | a_<t)`` over COMPLETION
    tokens only. Its logit gradient is the causal-CE gradient scaled per row by
    the advantage and zeroed on prompt positions — so this is ``train_step``
    with one elementwise multiply, not a second training path.

    ``input_ids`` [B,T] is prompt+completion per row, right-padded to a common
    T; ``advantages`` [B]; ``prompt_lens`` [B]; ``seq_lens`` [B] is the valid
    length of each row (default T — pass it whenever rows are padded, or the
    padding gets gradient).

    Returns the batch cross-entropy: a DIAGNOSTIC (how likely the sampled
    tokens were), not the objective — the objective is the advantage-weighted
    one whose gradient this step applies.

    Strictly on-policy: one update per rollout, so the importance ratio is 1
    and there is no clipping term.
    # ponytail: single-update REINFORCE-with-baseline; add the PPO ratio+clip
    # when a rollout is reused for more than one step.
    """
    adv = torch.as_tensor(np.asarray(advantages, dtype=np.float32))
    plen = np.asarray(prompt_lens, dtype=np.int64)
    slen = None if seq_lens is None else np.asarray(seq_lens, dtype=np.int64)

    def grad_fn(logits, ids):
        b, t, _ = logits.shape
        loss, grad = backend.cross_entropy_loss_grad(logits, ids)
        dev = grad.device
        # Position i predicts token i+1: scored iff that token is a real
        # completion token, i.e. prompt_len <= i+1 < seq_len.
        pos = torch.arange(t, device=dev).reshape(1, t)
        keep = pos >= torch.as_tensor(plen, device=dev).reshape(b, 1) - 1
        if slen is not None:
            keep = keep & (pos < torch.as_tensor(slen, device=dev).reshape(b, 1) - 1)
        w = keep.float() * adv.to(dev).reshape(b, 1)
        # CE averages over b*(t-1) positions; rescale to the scored count so the
        # step size does not depend on how much prompt or padding is in the batch.
        n = float(keep[:, :-1].sum().item())
        grad = grad * w.unsqueeze(-1) * (b * (t - 1) / max(n, 1.0))
        return loss, grad

    return _step(model, input_ids, backend, optimizer, tape, trainable, grad_fn)


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
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    seed: int = 0,
    trainable: dict[str, Any] | None = None,
) -> list[tuple[float, float]]:
    """GRPO: sample ``group`` completions per prompt, score them, take one
    policy-gradient step on the group-normalized advantages.

    The engine that generates IS the model that trains — same weights, same
    runtime (训练推理一体), so a rollout costs a serving batch and nothing is
    duplicated. ``reward_fn(prompt_ids, completion_ids) -> float``.

    ``engine`` must be built with the prefix cache and the captured decode graph
    OFF (``prefix_store=NoPrefixStore(), decode_graph=False``). Both cache work
    across steps — a stored prefix holds KV from an earlier policy, and a
    captured graph holds f32 casts of weights the optimizer updates in place —
    so either one samples from a policy that is not the current one, without
    ever raising. The price is the eager decode path for rollouts, which is
    ~8x slower than a replay at B=1.
    # ponytail: recapture the graph and drop the prefix entries after each
    # update instead of disabling both, once a rollout's decode cost matters.

    Returns per step ``(mean reward, cross-entropy of the sampled tokens)``.
    Rollouts within a step are one engine batch: the group is what continuous
    batching is for.
    """
    from .engine import SamplingParams

    if optimizer is None:
        optimizer = AdamW(lr=1e-5)
    out: list[tuple[float, float]] = []
    for step in range(steps):
        prompt = np.asarray(prompts[step % len(prompts)], dtype=np.int64)
        # Distinct seeds, one batch: identical seeds would make the group a
        # single sample repeated, and every advantage exactly zero.
        ids = [
            engine.submit(prompt.tolist(), SamplingParams(
                temperature=temperature, max_new_tokens=max_new_tokens,
                seed=seed + step * group + g))
            for g in range(group)
        ]
        done: dict[int, list[int]] = {}
        for _ in range(10000):  # one forward per tick; bounded guard
            engine.step()
            done.update(engine.poll())
            if all(i in done for i in ids):
                break
        else:  # pragma: no cover - engine bug, not a training path
            raise RuntimeError("grpo_loop: rollout did not finish within 10000 ticks")
        comps = [done[i] for i in ids]
        rewards = [float(reward_fn(prompt, c)) for c in comps]
        adv = group_advantages(rewards, group)
        # Right-pad to one rectangle: padding sits past every prompt_len, and
        # the advantage mask is what keeps it out of the gradient.
        width = max(len(c) for c in comps)
        batch = np.stack([
            np.concatenate([prompt, np.asarray(c, dtype=np.int64),
                            np.zeros(width - len(c), dtype=np.int64)])
            for c in comps
        ])
        plens = np.full(group, len(prompt), dtype=np.int64)
        slens = np.array([len(prompt) + len(c) for c in comps], dtype=np.int64)
        ce = rl_step(model, batch, adv, plens, backend, optimizer,
                     trainable=trainable, seq_lens=slens)
        out.append((float(np.mean(rewards)), ce))
    return out


# ---------------------------------------------------------------------------
# opd_loop
# ---------------------------------------------------------------------------


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
) -> list[float]:
    """On-policy distillation: frozen teacher generates, student SFTs.

    Each step: the teacher generates ``max_new_tokens`` tokens for one prompt
    (cycled), then the student does one :func:`train_step` on the full
    (prompt + completion) sequence. The teacher is frozen by construction —
    only ``submit``/``poll``/``step`` are ever called on it, never
    ``train_step``.

    ``trainable`` (the LoRA adapters) turns this into the self-teacher loop of
    ``agent-infer/crates/train/src/ema_self_teacher.rs``: the teacher engine
    drives the SAME model, generating with an EMA of those adapters instead of
    a second copy of the weights. Only the adapters are duplicated, so the
    self-teacher costs adapter-sized memory, not model-sized.
    """
    from .engine import SamplingParams

    if optimizer is None:
        optimizer = AdamW(lr=1e-3)
    ema = {k: v.clone() for k, v in trainable.items()} if trainable is not None else None
    losses: list[float] = []
    for step in range(steps):
        prompt = np.asarray(prompts[step % len(prompts)], dtype=np.int64)
        params = SamplingParams(temperature=1.0, top_p=1.0, max_new_tokens=8, seed=seed + step)
        if ema is not None:  # generate with the teacher weights, train the student
            student_model.params.update(ema)
        rid = teacher_engine.submit(prompt, params)
        finished: dict[int, list[int]] = {}
        for _ in range(10000):  # one forward per tick; bounded guard
            teacher_engine.step()
            finished = teacher_engine.poll()
            if rid in finished:
                break
        else:  # pragma: no cover - engine bug, not a training path
            raise RuntimeError("opd_loop: teacher did not finish within 10000 ticks")
        seq = np.concatenate([prompt, np.asarray(finished[rid], dtype=np.int64)])
        if ema is not None:
            student_model.params.update(trainable)
        losses.append(train_step(student_model, seq[None, :], backend, optimizer,
                                 trainable=trainable))
        if ema is not None:
            for k, e in ema.items():
                e.mul_(ema_decay).add_(trainable[k], alpha=1.0 - ema_decay)
    return losses


# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------


class JsonlDataset:
    """JSONL corpus -> packed fixed-length token sequences.

    Each line must have a ``"text"`` field. Texts are tokenized and joined
    with ``eos_token_id`` separators, then cut into ``seq_len`` chunks (the
    last chunk is eos-padded). Iteration is file order; :func:`pretrain`
    shuffles with its own seeded RNG.
    """

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
    """Pretrain ``model`` on ``dataset`` (iterable of [T] token-id arrays).

    Causal-LM loss via :func:`train_step`; LR follows :func:`cosine_warmup`.
    Checkpoints land in ``ckpt_dir`` every ``ckpt_every`` steps plus a final
    ``final/`` save. Returns the per-step losses. Deterministic for a fixed
    ``seed`` (epoch-wise shuffle of the dataset).
    """
    from .model import save_hf

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
            loss = train_step(model, input_ids, backend, optimizer, Tape())
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
