"""Hand-written reverse-mode tape (mirrors ``agent-infer/crates/autograd``):
no ``torch.autograd``, no ``torch.optim``; gradients are the backend's ``*_bwd``
ops replayed in reverse, accumulated by ``id()`` — leaves are the params."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch

from . import precision

# ContextVar so concurrent sessions never cross-record; cleared during backward.
_current_tape: ContextVar[Tape | None] = ContextVar("tilerl_current_tape", default=None)


def maybe_record(op_name: str, output: torch.Tensor, *args: Any, **kwargs: Any) -> None:
    """Record one op on the active tape (no-op without one). ``args`` leads with
    the differentiable inputs in the order the op's ``*_bwd`` returns their
    grads, params passed verbatim (no views — grads accumulate by ``id()``);
    ``_backend`` is the backend the tape replays ``*_bwd`` against."""
    backend = kwargs.pop("_backend", None)
    tape = _current_tape.get()
    if tape is not None:
        if backend is not None:
            tape.bwd_backend = backend
        tape.record(op_name, args, kwargs, output)


def checkpoint(fn: Callable[..., torch.Tensor], *args: Any) -> torch.Tensor:
    """Recompute instead of store: run ``fn(*args)`` with recording off and
    record one entry that replays it under a sub-tape during backward, so the
    segment's activations never coexist with the rest of the forward. ``fn``
    must be pure — a segment that advances the KV or GDN state pool would
    recompute against state its own forward already moved."""
    tape = _current_tape.get()
    if tape is None or not tape.recompute:
        return fn(*args)
    token = _current_tape.set(None)
    try:
        out = fn(*args)
    finally:
        _current_tape.reset(token)
    tape.record("checkpoint", args, {"fn": fn}, out)
    return out


# Recorded views: a raw torch view between two backend ops breaks the id() chain.
def reshape(x: torch.Tensor, *shape: int) -> torch.Tensor:
    y = x.reshape(*shape)
    maybe_record("reshape", y, x, shape=tuple(shape))
    return y


def slice(x: torch.Tensor, *key: Any) -> torch.Tensor:
    y = x[key]
    maybe_record("slice", y, x, key=key)
    return y


class RecordingBackend:
    """The recording seam: wraps a tape-unaware backend so ops in ``_BWD``
    record their (first) output; everything else passes through.
    # ponytail: if backend.py grows its own maybe_record calls, drop this proxy."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

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


# A handler yields (slot, grad): slot is an arg index, ("kw", name), or ("id", id(t)).

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
    positions = args[1]
    theta = args[2] if len(args) > 2 else kw["theta"]
    rotary_dim = kw.get("rotary_dim")
    yield 0, backend.rope_bwd(g, positions, theta, rotary_dim=rotary_dim)


def _embedding(backend: Any, g: torch.Tensor, args: tuple, kw: dict, wants: Any):
    # The table grad is dense [vocab, hidden] f32 (4.7 GiB on the 27B): skip it when frozen.
    idx, table = args[0], args[1]
    if wants(table):
        yield 1, backend.embedding_bwd(g, idx, table.shape[0])


_embedding.wants = True


def _attention(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # With gate: out = attn_out * sigmoid(gate); attn_out is recomputed for the gate grad.
    q, k, v = args[0], args[1], args[2]
    scale = args[3] if len(args) > 3 else kw.get("scale", 1.0)
    gate = kw.get("gate")
    # The forward's mask, or the backward recomputes softmax over a different one.
    pos = {"q_pos": kw.get("q_pos"), "k_pos": kw.get("k_pos")}
    if gate is not None:
        attn_out = backend.attention(q, k, v, scale, **pos)
        g_attn, g_gate = backend.attention_gate_bwd(g, attn_out, gate)
        gq, gk, gv = backend.attention_bwd(g_attn, q, k, v, scale, **pos)
        yield 0, gq
        yield 1, gk
        yield 2, gv
        yield ("kw", "gate"), g_gate
    else:
        gq, gk, gv = backend.attention_bwd(g, q, k, v, scale, **pos)
        yield 0, gq
        yield 1, gk
        yield 2, gv


def _paged_attention(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    raise NotImplementedError(
        "paged_attention has no tape backward — training runs use dense "
        "attention (kv.dense=True); do not record the inference path"
    )


_GDN_KW = ("z", "conv1d_weight", "dt_bias", "a_log", "norm_weight")


def _linear_attn_chunk(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Grads beyond the 6 positional (q,k,v,g,beta,state) map to the GDN kwargs in order.
    results = backend.linear_attn_bwd(g, *args, **kw)
    for i, gi in enumerate(results):
        if i < len(args):
            yield i, gi
        else:
            yield ("kw", _GDN_KW[i - len(args)]), gi


def _checkpoint(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    sub = Tape()
    sub.bwd_backend = backend
    with sub:
        kw["fn"](*args)
    # A leaf inside the segment (a weight, a LoRA adapter) is not an arg of this
    # entry, so it comes back addressed by id.
    inputs = {id(a): i for i, a in enumerate(args)}
    for tid, gi in sub.backward(g).items():
        yield inputs.get(tid, ("id", tid)), gi


def _reshape(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    yield 0, g.reshape(args[0].shape)


def _slice(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    gx = torch.zeros_like(args[0])
    gx[kw["key"]] = g
    yield 0, gx


def _add(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    yield 0, g
    yield 1, g


def _tp_fork(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Identity forward, all-reduce backward. Each rank's column-parallel linears
    # contribute their own share of dX for the one replicated activation they
    # all read; the sum is the true dX.
    yield 0, backend.all_reduce(g)


def _all_reduce(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Row-parallel forward: Y = sum_r Y_r, so dY_r = dY on every rank already.
    # The entry exists so the tape's id() chain crosses the collective; without it
    # RecordingBackend passes all_reduce through and the residual add below it
    # reads a tensor no entry produced.
    yield 0, g


def _all_gather(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # Forward concatenates each rank's slice; the backward keeps this rank's.
    dim = kw.get("dim", -1)
    yield 0, g.chunk(backend.tp_world, dim=dim)[backend.tp_rank].contiguous()


def _cp_gather(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
    # NOT a slice, unlike _all_gather above. There each rank owns a distinct
    # slice of the output; here every rank's queries read every rank's K/V, so
    # each rank's incoming gradient covers the whole sequence and the chunks must
    # be summed before this rank keeps its own.
    yield 0, backend.cp_reduce_scatter(g, dim=kw.get("dim", 1))


def _frozen(fp8: bool) -> _Handler:
    # No master: only dX flows.
    def handler(backend: Any, g: torch.Tensor, args: tuple, kw: dict):
        yield 0, backend.linear_frozen_bwd(g, args[1], args[2], oscale=kw.get("oscale"), fp8=fp8)

    return handler


_BWD: dict[str, _Handler] = {
    "linear_fp4_frozen": _frozen(False),
    "linear_fp8_frozen": _frozen(True),
    "rmsnorm": _default("rmsnorm"),
    # _default's argument names the BACKWARD it resolves (`<name>_bwd`), not the key:
    # "rmsnorm" here means rmsnorm_bwd, which is right because a norm's gradient does
    # not depend on the dtype the forward stored. Omitting the key entirely records NO
    # entry and silently drops the q/k norm gradient (measured: 1 entry vs 0).
    "rmsnorm_f32": _default("rmsnorm"),
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
    "all_reduce": _all_reduce,
    "all_gather": _all_gather,
    "cp_gather": _cp_gather,
    "tp_fork": _tp_fork,
    "checkpoint": _checkpoint,
}


class Tape:
    """Records backend ops while active; ``backward(grad_of_last_output)``
    replays them in reverse and returns ``{id(leaf): grad}``. One backward per
    tape. With ``recompute`` (default on), :func:`checkpoint` segments store
    their input instead of their activations and are replayed during backward."""

    def __init__(self, recompute: bool = True) -> None:
        self.recompute = recompute
        self._entries: list[_Entry] = []
        self._token: Any = None
        self._consumed = False
        self.bwd_backend: Any = None  # set by RecordingBackend; else the TileLang singleton

    def __enter__(self) -> Tape:
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
        # An in-place op that returns its own input is invisible here: backward
        # writes the gradient under id(output), then pops it as consumed, and the
        # input's gradient vanishes with no error anywhere. all_reduce did exactly
        # this -- dX came back None. A view is enough to separate them.
        if any(a is output for a in args) or any(v is output for v in kwargs.values()):
            raise RuntimeError(
                f"{op_name} returned one of its own inputs; the tape addresses "
                "tensors by id(), so its gradient would be dropped silently. "
                "Return a distinct object (x.view_as(x) for an in-place op)."
            )
        self._entries.append(_Entry(op_name, tuple(args), dict(kwargs), output))

    def _first_use(self) -> dict[int, int]:
        """``{id(tensor): first entry index consuming it}``: in reverse replay,
        a gradient is final once that entry has run."""
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
        """Reverse replay; returns ``{id(leaf): grad}``. ``needs``: leaf ids the
        caller will read, so handlers can skip an unwanted expensive gradient.
        ``on_grad(tensor_id, grad) -> bool``: offered each leaf gradient as soon
        as it is final; returning True takes it out of the returned dict, so the
        27B's weight gradients (50 GiB) never all coexist."""
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

        backend = self.bwd_backend
        if backend is None:
            from tilerl_kernels.backend import get_backend  # backend.py imports this module

            backend = get_backend()
        grads: dict[int, torch.Tensor] = {id(last.output): grad_output}

        entries = self._entries
        produced = {id(e.output) for e in entries}
        first_use = self._first_use() if on_grad is not None else {}
        produced_at = ({id(e.output): i for i, e in enumerate(self._entries)}
                       if on_grad is not None else {})
        taken: set[int] = set()

        def wants(t: torch.Tensor) -> bool:
            # An intermediate is always wanted: dropping one severs the chain below it.
            return needs is None or id(t) in needs or id(t) in produced

        token = _current_tape.set(None)
        try:
            for i in range(len(entries) - 1, -1, -1):
                entry = entries[i]
                g_out = grads.get(id(entry.output))
                if g_out is None:  # dead branch; it still ends tensor lifetimes at this index
                    if on_grad is not None:
                        self._release(grads, first_use, produced_at, taken, i, on_grad)
                    continue
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
                        tid = id(entry.args[slot])
                    elif slot[0] == "kw":
                        tid = id(entry.kwargs[slot[1]])
                    else:  # ("id", tid): a leaf inside a checkpointed segment
                        # The segment replayed whole, so its leaf grads are final here.
                        tid = slot[1]
                        first_use.setdefault(tid, i)
                    if tid in taken:
                        raise RuntimeError(
                            f"gradient for leaf {tid} arrived after it was streamed "
                            "out: a tensor is read in two checkpointed segments")
                    grads[tid] = grads[tid] + g_in if tid in grads else g_in
                grads.pop(id(entry.output), None)  # consumed: nothing below reads it
                if on_grad is not None:
                    self._release(grads, first_use, produced_at, taken, i, on_grad)
        finally:
            _current_tape.reset(token)
            self._entries.clear()
            self._consumed = True
        return {tid: g for tid, g in grads.items()
                if tid not in produced and tid not in taken}

    @staticmethod
    def _release(grads, first_use, produced_at, taken, idx, on_grad) -> None:
        # Leaves only: an intermediate released at its own first-use index goes
        # one entry too early and severs the chain below it.
        for tid, g in list(grads.items()):
            if first_use.get(tid) != idx or tid in produced_at:
                continue
            if on_grad(tid, g):
                taken.add(tid)
                del grads[tid]


class _Optimizer:
    """``step`` is the same loop for every optimizer; ``step_one`` is the update."""

    def begin(self) -> None:
        self._step += 1

    def step(self, params: Any, grads: dict[int, torch.Tensor]) -> None:
        self.begin()
        for p in params:
            g = grads.get(id(p))
            if g is not None:
                self.step_one(p, g)


class AdamW(_Optimizer):
    """AdamW (decoupled weight decay); fp32 moments keyed by ``id(param)``,
    the update computed in fp32 and cast back to the param's dtype."""

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

    def step_one(self, p: torch.Tensor, g: torch.Tensor, key: Any = None) -> None:
        b1, b2 = self.betas
        bc1 = 1.0 - b1**self._step
        bc2 = 1.0 - b2**self._step
        g = g.to(torch.float32)
        pid = id(p) if key is None else key
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


class Adafactor(_Optimizer):
    """Adafactor (Shazeer & Stern 2018, paper defaults): factored second moment,
    no first moment unless ``beta1 > 0``, relative step size (``lr`` is a
    multiplier on the param's RMS), update clipping at RMS ``clip``. Adam's m+v
    on the 27B is 200.4 GiB; this is 0.03 GiB."""

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

    streams = True  # per-update clipping, no global norm: grads can be freed one at a time

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        # Stays on device: a float() here is one host sync per parameter, mid-backward.
        return t.norm() / math.sqrt(t.numel())

    def step_one(self, p: torch.Tensor, g: torch.Tensor, key: Any = None) -> None:
        b2 = 1.0 - self._step**self.decay_power
        g = g.to(torch.float32)
        u = g.mul(g).add_(self.eps[0])
        key = id(p) if key is None else key
        st = self._state.get(key)
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
            self._state[key] = st
        if factored:
            r, c = st[0], st[1]
            r.mul_(b2).add_(u.mean(dim=1), alpha=1.0 - b2)
            c.mul_(b2).add_(u.mean(dim=0), alpha=1.0 - b2)
            # outer(r / mean(r), c) is the rank-1 reconstruction of v
            upd = g * torch.rsqrt(r.div(r.mean()).unsqueeze(1) * c.unsqueeze(0))
        else:
            v = st[0]
            v.mul_(b2).add_(u, alpha=1.0 - b2)
            upd = g * torch.rsqrt(v)
        # A non-finite gradient makes the factor nan or 0; nan_to_num lands it as zeros.
        upd = torch.nan_to_num(upd.mul_(self.clip / self._rms(upd).clamp(min=self.clip)))
        p32 = p.to(torch.float32)
        step = self._rms(p32).clamp(min=self.eps[1]).mul(self.lr)
        if self.weight_decay > 0.0:
            p32.mul_(1.0 - step * self.weight_decay)
        if self.beta1 > 0.0:
            m = st[-1]
            m.mul_(self.beta1).add_(upd, alpha=1.0 - self.beta1)
            upd = m
        p.copy_(p32.sub_(upd.mul(step)).to(p.dtype))


def cosine_warmup(step: int, total: int, warmup: int, lr: float) -> float:
    """Linear warmup to ``lr`` over ``warmup`` steps, then cosine decay to 0 at ``total``."""
    if warmup > 0 and step < warmup:
        return lr * step / warmup
    if step >= total:
        return 0.0
    span = total - warmup
    if span <= 0:
        return lr
    progress = (step - warmup) / span
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_grad_norm(
    grads: dict[int, torch.Tensor], max_norm: float, sharded: set[int] | None = None,
    backend: Any = None,
) -> float:
    """Scale grads in place to global L2 norm <= ``max_norm`` (``<= 0`` disables);
    returns the pre-clip norm, non-finite as-is for the caller to reject.
    fp64 accumulation on device with ONE sync: the per-grad ``.to("cpu")`` this
    replaced cost 126 device-to-host copies a step, 7% of the 27B LoRA step.
    MPS has no float64, so it sums on the host.

    Under TP the norm must be the WHOLE model's or each rank scales its shard by
    a different factor. ``sharded`` is the ids whose gradient this rank holds only
    a slice of; those squares are summed across the tp group, the replicated ones
    are already identical on every rank and are added once. Measured on tiny at
    tp=2: local norms 2399.73 and 2288.83 (4.85% apart, so two different scales),
    against the true 2641.80 -- which naive summing overshoots to 3316.24 by
    counting every replicated gradient twice.
    """
    if not grads:  # _foreach_norm rejects an empty list; the norm of nothing is 0
        return 0.0
    dev = next(iter(grads.values())).device
    host = dev.type == "mps"
    ids = list(grads)  # one order for the norms and the sharded mask below
    gl = [grads[t].detach() for t in ids]
    if host:
        gl = [g.to("cpu") for g in gl]
    # One fused norm kernel over the whole list, then one f64 reduction and one
    # sync: measured 141 -> 17.5 ms at 27B-LoRA scale (1092 f32 adapters), the
    # norm agreeing to 7.7e-07 relative. _foreach_norm squares in the INPUT
    # dtype, so anything narrower than f32 is widened first — raw bf16 costs
    # 1.3 decimal digits, raw f16 costs 0.9. The rescale below still walks
    # grads.values(), so it scales the real gradients, not these copies.
    norms = torch._foreach_norm(
        [g if g.dtype.itemsize >= 4 else g.float() for g in gl], 2)
    sq = torch.stack(norms).to(torch.float64).pow(2)
    if sharded and getattr(backend, "tp_world", 1) > 1:
        keep = torch.tensor([t in sharded for t in ids], device=sq.device)
        part = torch.stack([sq.mul(keep).sum(), sq.mul(~keep).sum()])
        backend.all_reduce(part)  # [sum of sharded squares, world x the replicated ones]
        total = float((part[0] + part[1] / backend.tp_world).sqrt().item())
    else:
        total = float(sq.sum().sqrt().item())
    if not math.isfinite(total) or total == 0.0:
        return total
    if max_norm > 0.0 and math.isfinite(max_norm) and total > max_norm:
        scale = max_norm / total
        for g in grads.values():
            g.mul_(scale)
    return total
