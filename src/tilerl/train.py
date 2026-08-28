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

__all__ = ["train_step", "opd_loop", "pretrain", "JsonlDataset"]


# ---------------------------------------------------------------------------
# Causal cross-entropy
# ---------------------------------------------------------------------------


def _ce_loss_grad(logits: torch.Tensor, input_ids: Any, backend: Any) -> tuple[float, torch.Tensor]:
    """Stable shifted causal CE and matching logit gradient via backend ops."""
    return backend.cross_entropy_loss_grad(logits, input_ids)


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


def train_step(
    model: Any,
    input_ids: Any,
    backend: Any,
    optimizer: AdamW,
    tape: Tape | None = None,
    trainable: dict[str, Any] | None = None,
) -> float:
    """One SFT step: forward under a tape, causal CE, backward, clip, AdamW.

    Returns the scalar loss. The step is SKIPPED (params untouched) when the
    loss or the pre-clip grad norm is non-finite — agent-infer's
    ``finite_optimizer_step`` semantics.
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

    loss, grad_logits = _ce_loss_grad(logits, input_ids, backend)
    if not math.isfinite(loss):
        return loss

    grads = tape.backward(grad_logits)
    #: ``trainable`` = the subset that gets gradients (LoRA adapters); the rest
    #: is frozen, which is what keeps the 27B inside one card.
    params = model.params if trainable is None else trainable
    param_ids = {id(p) for p in params.values()}
    assert param_ids & set(grads), (
        "train_step: tape produced no parameter gradients — the recording "
        "seam is missing (backend ops not recorded?)"
    )
    # The GDN initial state is a tape leaf: its grad is not a parameter and
    # must not enter the clip norm. ponytail: backward still computes and
    # frees it mid-pass; mark the state input non-differentiable if its
    # allocation ever shows up in a peak-memory profile.
    grads = {k: v for k, v in grads.items() if k in param_ids}
    norm = clip_grad_norm(grads, 1.0)
    if math.isfinite(norm):
        optimizer.step(params.values(), grads)
    return loss


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
) -> list[float]:
    """On-policy distillation: frozen teacher generates, student SFTs.

    Each step: the teacher generates ``max_new_tokens`` tokens for one prompt
    (cycled), then the student does one :func:`train_step` on the full
    (prompt + completion) sequence. The teacher is frozen by construction —
    only ``submit``/``poll``/``step`` are ever called on it, never
    ``train_step``.

    EMA self-teacher (teacher weights = EMA of student) is day-2 — see
    ``agent-infer/crates/train/src/ema_self_teacher.rs``.
    """
    from .engine import SamplingParams

    if optimizer is None:
        optimizer = AdamW(lr=1e-3)
    losses: list[float] = []
    for step in range(steps):
        prompt = np.asarray(prompts[step % len(prompts)], dtype=np.int64)
        params = SamplingParams(temperature=1.0, top_p=1.0, max_new_tokens=8, seed=seed + step)
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
        losses.append(train_step(student_model, seq[None, :], backend, optimizer))
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
