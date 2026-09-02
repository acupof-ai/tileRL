"""Hand-written reverse-mode autograd tape.

Mirrors ``agent-infer/crates/autograd/src/tape.rs``: a :class:`Tape` records op
executions while active and replays them in reverse on :meth:`Tape.backward`.
No ``torch.autograd`` and no ``torch.optim`` anywhere — gradients are plain
tensor arithmetic dispatched through the backend's ``*_bwd`` ops.

Recording seam: :class:`RecordingBackend` wraps any backend and calls
:func:`maybe_record` after every op with a backward handler; the backend stays
tape-unaware. Rules: ``output`` is the single differentiable output (multi-
output ops record only the first); ``args`` leads with the differentiable
tensor inputs in the order the op's ``*_bwd`` returns their grads; params are
passed verbatim (no views — grads accumulate by ``id()``); fp4's ``master``
kwarg carries the bf16 weight the STE grad lands on.

Backward replay: entries replay in reverse insertion order (a valid reverse
topological order). Branches accumulate into the same ``id()`` bucket; entries
whose output received no grad are dead branches and skipped; tensors consumed
but never produced by a recorded op are the leaves (the model params), and
``backward()`` returns exactly their grads.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from . import precision

__all__ = [
    "Tape",
    "AdamW",
    "Adafactor",
    "cosine_warmup",
    "clip_grad_norm",
    "maybe_record",
    "RecordingBackend",
    "reshape",
    "slice",
]

#: The active tape, or ``None``. A ContextVar so concurrent threads/sessions
#: never cross-record. Set by ``Tape.__enter__``, cleared during backward so
#: backward ops cannot accidentally record (mirrors tape.rs ``enabled=false``).
_current_tape: ContextVar["Tape | None"] = ContextVar("tilerl_current_tape", default=None)


def maybe_record(op_name: str, output: torch.Tensor, *args: Any, **kwargs: Any) -> None:
    """Recording hook for backend ops. No-op when no tape is active.

    ``_backend`` (keyword-only, private) names the backend that owns the
    matching ``*_bwd`` calls; :class:`RecordingBackend` passes it so the tape
    replays backward against the same backend that ran the forward.
    """
    backend = kwargs.pop("_backend", None)
    tape = _current_tape.get()
    if tape is not None:
        if backend is not None:
            tape.bwd_backend = backend
        tape.record(op_name, args, kwargs, output)


# --- Structural ops (reshape / slice) ---
# These are torch container ops, not backend kernels — but the tape must see
# them or the id()-based grad chain breaks wherever a view sits between two
# backend ops (e.g. ``q = reshape(linear(x, w_q), B, T, H, D)``). Model code
# uses these helpers instead of the raw torch methods.


def reshape(x: torch.Tensor, *shape: int) -> torch.Tensor:
    y = x.reshape(*shape)
    maybe_record("reshape", y, x, shape=tuple(shape))
    return y


def slice(x: torch.Tensor, *key: Any) -> torch.Tensor:
    """Recorded slice (``x[key]``). Backward scatters into a zeros tensor."""
    y = x[key]
    maybe_record("slice", y, x, key=key)
    return y


class RecordingBackend:
    """Proxy that records differentiable ops onto the active tape.

    The backend stays tape-unaware: this proxy is the recording seam. Wrap any
    backend (TileLang or torch-eager reference) and pass the wrapper where a
    backend is expected — forwards record onto the tape, backwards pass through
    unwrapped (the tape calls them directly, against the same backend)::

        with Tape() as tape:
            logits = model.forward(ids, pos, kv, RecordingBackend(backend))
        grads = tape.backward(grad_logits)

    Only ops with a backward handler (``_BWD``) are recorded; ``sample``,
    ``softmax`` and friends pass through untouched. Multi-output
    ops (``linear_attn_chunk``) record their first (differentiable) output.
    # ponytail: if backend.py grows its own maybe_record calls, drop this
    # proxy — the two together would double-record.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.name = getattr(backend, "name", "recording")
        self.target = getattr(backend, "target", "cpu")
        self.device = getattr(backend, "device", torch.device("cpu"))

    def __getattr__(self, name: str) -> Any:
        if name in ("linear_fp4", "linear_fp8"):

            def master_linear(x: torch.Tensor, *args: Any, master=None, **kwargs: Any) -> Any:
                if master is None:  # frozen base (LoRA / OPD): quantized kernel, dX only
                    out = getattr(self._backend, name)(x, *args, **kwargs)
                    maybe_record(name + "_frozen", out, x, *args, _backend=self._backend,
                                 **kwargs)
                    return out
                out = self._backend.linear(x, master)
                maybe_record("linear", out, x, master, _backend=self._backend)
                return out

            return master_linear
        attr = getattr(self._backend, name)
        if name not in _BWD:
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            out = attr(*args, **kwargs)
            recorded = out[0] if isinstance(out, tuple) else out
            maybe_record(name, recorded, *args, _backend=self._backend, **kwargs)
            return out

        return wrapper


@dataclass
class _Entry:
    op_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: torch.Tensor


# --- Backward handlers ---
# Each handler yields (slot, grad) pairs: slot is an int (index into the
# entry's args) or ("kw", name) (a kwargs entry). The default handler covers
# any op whose `<name>_bwd(grad, *args, **kwargs)` returns grads for the
# differentiable args in order.

_Handler = Callable[[Any, torch.Tensor, tuple, dict], Iterator[tuple]]


def _default(op_name: str) -> _Handler:
    def handler(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
        yield from enumerate(getattr(backend, op_name + "_bwd")(g, *args, **kw))

    return handler


def _linear(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    kw = dict(kw)
    bias = kw.pop("bias", None)
    if bias is not None:
        # ponytail: no bias grads — the model has no biases; add g.sum(axes)
        # and a third return if a biased linear ever appears.
        raise NotImplementedError("linear bias gradient not supported")
    gx, gw = backend.linear_bwd(g, *args, **kw)
    yield 0, gx
    yield 1, gw


def _rope(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # rope_bwd(grad, positions, theta[, rotary_dim]) -> gx: x itself is not an input.
    positions = args[1]
    theta = args[2] if len(args) > 2 else kw["theta"]
    rotary_dim = kw.get("rotary_dim")
    yield 0, backend.rope_bwd(g, positions, theta, rotary_dim=rotary_dim)


def _embedding(backend: Any, g: torch.Tensor, args: tuple, kw: dict, wants: Any):
    # embedding_bwd(grad, idx, num_rows) -> gtable: num_rows comes from the
    # saved table's shape, the table itself is not an input to the kernel.
    # The table gradient is DENSE [vocab, hidden] f32 — 4.7 GiB on the 27B —
    # so a frozen embedding (LoRA, OPD) must not pay for one nobody reads.
    idx, table = args[0], args[1]
    if wants(table):
        yield 1, backend.embedding_bwd(g, idx, table.shape[0])


_embedding.wants = True  # takes the extra `wants` argument


def _attention(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Dense causal GQA attention (training path). q/k/v [B,T,H,D].
    # With gate: out = attn_out * sigmoid(gate); recompute attn_out for the
    # gate grad (the tape context is off during backward, so no re-record).
    q, k, v = args[0], args[1], args[2]
    scale = args[3] if len(args) > 3 else kw.get("scale", 1.0)
    gate = kw.get("gate")
    if gate is not None:
        attn_out = backend.attention(q, k, v, scale)
        g_attn, g_gate = backend.attention_gate_bwd(g, attn_out, gate)
        gq, gk, gv = backend.attention_bwd(g_attn, q, k, v, scale)
        yield 0, gq
        yield 1, gk
        yield 2, gv
        yield ("kw", "gate"), g_gate
    else:
        gq, gk, gv = backend.attention_bwd(g, q, k, v, scale)
        yield 0, gq
        yield 1, gk
        yield 2, gv


def _paged_attention(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Paged attention is never recorded in training (the model uses dense
    # attention there). If an inference forward ever runs under a tape, fail
    # loudly instead of replaying a dense backward over paged args.
    raise NotImplementedError(
        "paged_attention has no tape backward — training runs use dense "
        "attention (kv.dense=True); do not record the inference path"
    )


_GDN_KW = ("z", "conv1d_weight", "dt_bias", "a_log", "norm_weight")


def _linear_attn_chunk(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # 6-arg form: linear_attn_bwd returns 6 grads (q,k,v,g,beta,state).
    # Full-GDN form (kwargs present): gdn_backward returns 11 — the extra 5
    # map to the GDN kwargs in order.
    results = backend.linear_attn_bwd(g, *args, **kw)
    for i, gi in enumerate(results):
        if i < len(args):
            yield i, gi
        else:
            yield ("kw", _GDN_KW[i - len(args)]), gi


def _reshape(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    yield 0, g.reshape(args[0].shape)


def _slice(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Scatter the grad into a zeros tensor at the slice key.
    gx = torch.zeros_like(args[0])
    gx[kw["key"]] = g
    yield 0, gx


def _add(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Residual add: the grad flows to both inputs unchanged.
    yield 0, g
    yield 1, g


def _frozen(fp8: bool) -> _Handler:
    # Frozen quantized base: the weight has no master, so only dX flows.
    def handler(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
        yield 0, backend.linear_frozen_bwd(g, args[1], args[2], oscale=kw.get("oscale"), fp8=fp8)

    return handler


_BWD: dict[str, _Handler] = {
    "linear_fp4_frozen": _frozen(False),
    "linear_fp8_frozen": _frozen(True),
    "rmsnorm": _default("rmsnorm"),
    "rope": _rope,
    "linear": _linear,
    "attention": _attention,
    "paged_attention": _paged_attention,
    "linear_attn_chunk": _linear_attn_chunk,
    "silu_mul": _default("silu_mul"),
    "embedding": _embedding,
    "reshape": _reshape,
    "slice": _slice,
    "add": _add,
}


class Tape:
    """Records backend ops while active, replays them in reverse on backward.

    Usage::

        with Tape() as tape:
            logits = model.forward(...)      # every backend op records itself
        grads = tape.backward(grad_logits)   # {id(param): grad}

    ``grad_output`` is the gradient of the LAST recorded op's output (the
    logits in a training step). One backward per tape.
    """

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._token: Any = None
        self._consumed = False
        #: Backend for the ``*_bwd`` calls; set by RecordingBackend while
        #: recording. Falls back to the TileLang singleton when unset.
        self.bwd_backend: Any = None

    def __enter__(self) -> "Tape":
        if self._token is not None or self._entries or self._consumed:
            raise RuntimeError("Tape is already active (nested/reused tape)")
        self._token = _current_tape.set(self)
        return self

    def __exit__(self, *exc: Any) -> bool:
        _current_tape.reset(self._token)
        self._token = None
        return False

    def record(
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: torch.Tensor,
    ) -> None:
        """Called by ``maybe_record``; appends one entry."""
        self._entries.append(_Entry(op_name, tuple(args), dict(kwargs), output))

    def _first_use(self) -> dict[int, int]:
        """``{id(tensor): lowest entry index that consumes it}``.

        Walking the tape in reverse, a tensor's gradient is final once that
        entry has run — no earlier entry can add to it. This is what lets a
        gradient be handed out and dropped the moment it is done.
        """
        first: dict[int, int] = {}
        for i, e in enumerate(self._entries):
            for a in e.args:
                if isinstance(a, torch.Tensor):
                    first.setdefault(id(a), i)
            for a in e.kwargs.values():
                if isinstance(a, torch.Tensor):
                    first.setdefault(id(a), i)
        return first

    def backward(
        self, grad_output: torch.Tensor, on_grad: Any = None, needs: set[int] | None = None
    ) -> dict[int, torch.Tensor]:
        """Reverse replay. Returns ``{id(param): grad}`` for every leaf
        (consumed-but-never-produced tensor) that received gradient.

        ``needs`` — when given, the ids of the leaves whose gradient the caller
        will actually read. A handler that can skip an expensive gradient for
        an unwanted leaf does so: the 27B's frozen embedding table gradient is
        a dense [248320, 5120] f32, 4.7 GiB, computed and discarded on every
        LoRA step without it.

        ``on_grad(tensor_id, grad) -> bool`` — when given, each gradient is
        offered as soon as it is final; returning True means the callback took
        it and this dict DROPS it, so gradients never all coexist. That is what
        full fine-tuning needs: with masters the 27B OOMs at 95.09 of 95.22 GiB
        inside the backward's ``gemm_tn`` while holding every weight gradient
        (errors/2026-08-29-full-finetune-blocker.md). It also releases the ~9
        GiB of intermediate-activation gradients this dict otherwise keeps past
        their last use, which LoRA pays as well.
        """
        if not self._entries:
            raise RuntimeError(
                "backward on an empty tape — no backend op recorded. The "
                "recording hook (maybe_record) is missing from the backend."
            )
        last = self._entries[-1]
        if grad_output.shape != last.output.shape:
            raise ValueError(
                f"grad_output shape {tuple(grad_output.shape)} does not match "
                f"the last op ({last.op_name}) output shape {tuple(last.output.shape)}"
            )

        # Lazy import: backend.py imports maybe_record from this module at top
        # level, so importing it here (not at module top) breaks the cycle.
        backend = self.bwd_backend
        if backend is None:
            from tilerl_kernels.backend import get_backend

            backend = get_backend()
        grads: dict[int, torch.Tensor] = {id(last.output): grad_output}

        # Backward ops must not record onto this tape.
        entries = self._entries
        produced = {id(e.output) for e in entries}
        # Index i is where each tensor is FIRST consumed, so walking in reverse
        # its gradient is final once entry i has run. Intermediates are dropped
        # by their PRODUCER instead: an activation's gradient is consumed when
        # the entry that produced it replays, and releasing it at its own
        # first-use index deletes it before that — which severs the chain and
        # leaves every downstream parameter without a gradient.
        first_use = self._first_use() if on_grad is not None else {}
        produced_at = ({id(e.output): i for i, e in enumerate(self._entries)}
                       if on_grad is not None else {})
        taken: set[int] = set()

        def wants(t: torch.Tensor) -> bool:
            """``needs`` names the leaves the caller will read. An intermediate
            is always wanted — dropping one severs the chain below it."""
            return needs is None or id(t) in needs or id(t) in produced

        token = _current_tape.set(None)
        try:
            for i in range(len(entries) - 1, -1, -1):
                entry = entries[i]
                g_out = grads.get(id(entry.output))
                if g_out is None:
                    # A dead branch still ends tensors' lifetimes at this index.
                    if on_grad is not None:
                        self._release(grads, first_use, produced_at, taken, i, on_grad)
                    continue  # dead branch: output feeds nothing differentiable
                handler = _BWD.get(entry.op_name)
                if handler is None:
                    raise KeyError(
                        f"no backward handler for recorded op {entry.op_name!r}; "
                        f"known ops: {sorted(_BWD)}"
                    )
                call = ((backend, g_out, entry.args, entry.kwargs, wants)
                        if getattr(handler, "wants", False)
                        else (backend, g_out, entry.args, entry.kwargs))
                for slot, g_in in handler(*call):
                    if isinstance(slot, int):
                        target = entry.args[slot]
                    else:
                        target = entry.kwargs[slot[1]]
                    tid = id(target)
                    if tid in grads:
                        grads[tid] = grads[tid] + g_in
                    else:
                        grads[tid] = g_in
                # The entry that just ran was the last chance to add to any
                # tensor first consumed here: those gradients are final.
                if on_grad is not None:
                    grads.pop(id(entry.output), None)  # this entry consumed it
                    self._release(grads, first_use, produced_at, taken, i, on_grad)
        finally:
            _current_tape.reset(token)
            self._entries.clear()
            self._consumed = True
        return {tid: g for tid, g in grads.items()
                if tid not in produced and tid not in taken}

    @staticmethod
    def _release(grads, first_use, produced_at, taken, idx, on_grad) -> None:
        """Offer every LEAF gradient that entry ``idx`` finalized, and drop the
        ones the callback takes.

        Only leaves: an intermediate is released by the caller when the entry
        that produced it replays, because that is when its gradient stops being
        needed. Keying an intermediate off its own first-use index deletes it
        one step too early and breaks the chain.
        """
        for tid, g in list(grads.items()):
            if first_use.get(tid) != idx or tid in produced_at:
                continue
            if on_grad(tid, g):
                taken.add(tid)
                del grads[tid]


class AdamW:
    """AdamW with decoupled weight decay, mirroring ``optim.rs`` step_host.

    Moments live in fp32 keyed by ``id(param)``; params may be bf16 masters —
    the update is computed in fp32 and cast back. Call :meth:`step` once per
    training step with the model's params (stable identity for the run).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}
        self._step = 0

    def step(self, params: Any, grads: dict[int, torch.Tensor]) -> None:
        """Apply one update. ``params`` is an iterable of param tensors;
        ``grads`` maps ``id(param)`` to its gradient. Params without a grad
        entry are skipped."""
        self.begin()
        for p in params:
            g = grads.get(id(p))
            if g is not None:
                self.step_one(p, g)

    def begin(self) -> None:
        self._step += 1

    def step_one(self, p: torch.Tensor, g: torch.Tensor) -> None:
        b1, b2 = self.betas
        bc1 = 1.0 - b1**self._step
        bc2 = 1.0 - b2**self._step
        g = g.to(torch.float32)
        pid = id(p)
        m = self._m.get(pid)
        if m is None:
            m = torch.zeros(p.shape, dtype=precision.dtype("optimizer_state"), device=p.device)
            v = torch.zeros_like(m)
            self._m[pid] = m
            self._v[pid] = v
        else:
            v = self._v[pid]
        m.mul_(b1).add_(g, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
        p32 = p.to(torch.float32)
        if self.weight_decay > 0.0:
            p32.mul_(1.0 - self.lr * self.weight_decay)
        denom = v.div(bc2).sqrt_().add_(self.eps)
        p32.addcdiv_(m, denom, value=-self.lr / bc1)
        p.copy_(p32.to(p.dtype))


class Adafactor:
    """Adafactor: the second moment of a 2D param is stored as a row vector and
    a column vector instead of the full matrix, and there is no first moment.

    This is what makes full fine-tuning arithmetically possible on one card.
    Adam's m+v for the 27B is 200.4 GiB of fp32 against 50.1 GiB of bf16
    weights; the factored form is 0.03 GiB. Same ``step(params, grads)``
    signature as :class:`AdamW`, so the training loop does not care which it
    holds.

    Follows Shazeer & Stern 2018 with the paper's defaults: relative step size
    scaled by the parameter's own RMS (so ``lr`` is a multiplier, not an
    absolute rate), update clipping at RMS 1.0, and beta2 growing as
    ``1 - t^-0.8``. ``beta1 > 0`` restores momentum at the cost of one fp32
    tensor per param — off by default, which is the whole point.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        beta1: float = 0.0,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip: float = 1.0,
        decay_power: float = -0.8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.eps = eps
        self.clip = clip
        self.decay_power = decay_power
        self.weight_decay = weight_decay
        self._state: dict[int, Any] = {}
        self._step = 0

    #: This optimizer needs no global gradient norm — its own update clipping
    #: bounds each step — so a parameter's gradient can be consumed and freed
    #: the moment backward finalizes it. That is what lets full fine-tuning
    #: run: holding all 26.9B gradients at once is 50.1 GiB of bf16 that
    #: clip_grad_norm would otherwise force to coexist.
    streams = True

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        """RMS as a DEVICE tensor. ``float()`` here would be a host sync per
        parameter, and ``step_one`` runs interleaved with backward, so each one
        drains the pipeline mid-step — the same cost clip_grad_norm's on-device
        accumulation already removed."""
        return t.norm() / math.sqrt(t.numel())

    def begin(self) -> None:
        """Advance the step counter once, before any :meth:`step_one`."""
        self._step += 1

    def step(self, params: Any, grads: dict[int, torch.Tensor]) -> None:
        self.begin()
        for p in params:
            g = grads.get(id(p))
            if g is not None:
                self.step_one(p, g)

    def step_one(self, p: torch.Tensor, g: torch.Tensor) -> None:
        """Update ONE parameter. Independent of every other parameter, which is
        exactly why the gradient can be released straight after."""
        b2 = 1.0 - self._step**self.decay_power
        g = g.to(torch.float32)
        u = g.mul(g).add_(self.eps[0])
        st = self._state.get(id(p))
        factored = g.dim() == 2
        if st is None:
            if factored:
                st = (
                    torch.zeros(g.shape[0], dtype=precision.dtype("optimizer_state"), device=g.device),
                    torch.zeros(g.shape[1], dtype=precision.dtype("optimizer_state"), device=g.device),
                )
            else:
                st = (torch.zeros_like(g),)
            if self.beta1 > 0.0:
                st = st + (torch.zeros_like(g),)
            self._state[id(p)] = st
        if factored:
            r, c = st[0], st[1]
            r.mul_(b2).add_(u.mean(dim=1), alpha=1.0 - b2)
            c.mul_(b2).add_(u.mean(dim=0), alpha=1.0 - b2)
            #: outer(r / mean(r), c) is the rank-1 reconstruction of v
            upd = g * torch.rsqrt(r.div(r.mean()).unsqueeze(1) * c.unsqueeze(0))
        else:
            v = st[0]
            v.mul_(b2).add_(u, alpha=1.0 - b2)
            upd = g * torch.rsqrt(v)
        #: clip to RMS <= self.clip. nan_to_num is what replaces the old early
        #: return on a non-finite gradient: a nan or inf RMS makes the factor
        #: nan or 0, and either way the update lands as zeros.
        upd = torch.nan_to_num(upd.mul_(self.clip / self._rms(upd).clamp(min=self.clip)))
        p32 = p.to(torch.float32)
        #: relative step: the update is scaled by the parameter's own size,
        #: so one lr works across tensors of very different magnitude.
        step = self._rms(p32).clamp(min=self.eps[1]).mul(self.lr)
        if self.weight_decay > 0.0:
            p32.mul_(1.0 - step * self.weight_decay)
        if self.beta1 > 0.0:
            m = st[-1]
            m.mul_(self.beta1).add_(upd, alpha=1.0 - self.beta1)
            upd = m
        p.copy_(p32.sub_(upd.mul(step)).to(p.dtype))


def cosine_warmup(step: int, total: int, warmup: int, lr: float) -> float:
    """Linear warmup to ``lr`` over ``warmup`` steps, then half-cosine decay
    to 0 at ``total``. Mirrors ``lr_schedule.rs`` CosineWithWarmup with
    ``min_lr=0``. Step 0 under warmup returns 0."""
    if warmup > 0 and step < warmup:
        return lr * step / warmup
    if step >= total:
        return 0.0
    span = total - warmup
    if span <= 0:
        return lr
    progress = (step - warmup) / span
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_grad_norm(grads: dict[int, torch.Tensor], max_norm: float) -> float:
    """Scale grads in place so their global L2 norm <= ``max_norm``. Returns
    the PRE-clip global norm (fp64 accumulation, like grad_clip.rs).

    A non-finite norm is returned as-is: the caller (train_step) rejects the
    step, matching ``finite_optimizer_step``. ``max_norm <= 0`` or non-finite
    disables clipping. Accumulation is fp64 on the grads' own device, then ONE
    sync — the per-grad ``.to("cpu")`` it replaces cost 126 device-to-host
    copies a step (7% of the 27B LoRA step). MPS has no float64 kernel, so
    that target still sums on the host."""
    dev = next(iter(grads.values())).device if grads else torch.device("cpu")
    host = dev.type == "mps"
    total_sq = torch.zeros((), dtype=torch.float64, device="cpu" if host else dev)
    for g in grads.values():
        # Two-step move on MPS: ``.to("cpu", torch.float64)`` still hits its
        # missing float64 kernel (torch converts dtype on the source device).
        gd = g.detach()
        total_sq += (gd.to("cpu") if host else gd).to(torch.float64).pow(2).sum()
    total = float(total_sq.sqrt().item())
    if not math.isfinite(total) or total == 0.0:
        return total
    if max_norm > 0.0 and math.isfinite(max_norm) and total > max_norm:
        scale = max_norm / total
        for g in grads.values():
            g.mul_(scale)
    return total
