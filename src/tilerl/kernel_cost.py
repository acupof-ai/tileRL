"""Per-kernel roofline declarations for one 27B decode tick.

Each launched kernel declares the bytes it moves and the flops it performs at a
concrete shape, priced through :func:`tilerl.precision.nbytes` — the ONLY byte
arithmetic (docs/design-cost-model.md, "Kernel cost"). Weight bytes come from
:func:`tilerl.model.param_specs`, or per linear from the checkpoint's device
faces (``TickShape.faces``), so a declaration never restates a matrix dim.

The table is the ground-truth side of ``tilerl bench --kernels``. GPU timing is a
separate calibration; without a card the measured columns render
``pending-remote``. A declaration that quietly disagrees with what a kernel
reads is caught by the attention-decode byte gate.

Scope: the kernels one 27B DECODE tick launches. The nvfp4 linears are
enumerated from param_specs, not listed by hand. Prefill/backward are later
rows; a kernel without a grounded shape is left undeclared rather than costed
against one it does not see.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import precision as P
from .precision import Format, nbytes


@dataclass(frozen=True)
class TickShape:
    """The shape coordinates one decode tick is costed at.

    ``b`` rows decode one token each against ``s`` pooled context tokens. Layer
    counts and widths live on cfg; this carries only the batch/context axis and
    the formats the served path uses.
    """

    b: int
    s: int
    kv: Format
    weight: Format
    #: param key -> (shape, device face) from a real checkpoint; when present every
    #: linear is priced at its OWN face (the 27B mixes nvfp4 and fp8 linears).
    faces: dict | None = None


def _gemv(spec: tuple[int, int], fmt: Format, t: TickShape) -> tuple[int, int]:
    """Decode linear y[b,out] = x[b,in] @ W^T[out,in]: weight streamed once,
    plus the b-row input/output."""
    out, inn = spec
    bytes_ = nbytes(fmt, spec) + nbytes(P.bf16, (t.b, inn)) + nbytes(P.bf16, (t.b, out))
    return bytes_, 2 * t.b * out * inn


def _paged_attention_decode(cfg, t: TickShape) -> tuple[int, int]:
    """One full-attn layer's decode attention.

    Reads that layer's pooled K and V — ``[b, kv_heads, s, d]`` each, priced in
    the pool's own format (fp8 plane + its per-head_dim f32 scale) — scores q
    against the s keys and combines with V, writing ``[b, q_heads, d]``.
    """
    hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    kv_bytes = 2 * nbytes(t.kv, (t.b, hkv, t.s, d))
    q_bytes = nbytes(P.bf16, (t.b, hq, d))
    o_bytes = q_bytes
    flops = 4 * t.b * hq * t.s * d  # q.k then score.v, 2 mul+add each
    return kv_bytes + q_bytes + o_bytes, flops


def _gdn_decode_fused(cfg, t: TickShape) -> tuple[int, int]:
    """The delta-recurrence kernel for one GDN layer at decode (chunk n=1).

    Carries state ``s [b, nvh, dk, dv]``: ``d = U - W s`` and
    ``s' = decay s + R^T d``. The state plane is updated in place, so it is read
    and written (2x). ``W``/``R``/``U`` are in-kernel intermediates recomputed
    from the f32 q/k/v/z/g/beta activations — NOT HBM reads (the projections'
    weights are the nvfp4 in_proj linears costed separately); only those
    activations are streamed in.
    """
    nvh, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    state = nbytes(P.f32, (t.b, nvh, dk, dv))  # one [dk,dv] state buffer
    # q,k,z [nvh,dk], v [nvh,dv], g,beta [nvh], all f32
    act = nbytes(P.f32, (t.b, nvh, 3 * dk + dv + 2))
    flops = 4 * t.b * nvh * dk * dk * dv  # W s and R^T d, two [dk,dk]@[dk,dv]
    return 2 * state + act, flops


def _rmsnorm(cfg, t: TickShape, width: int) -> tuple[int, int]:
    x = nbytes(P.bf16, (t.b, width))
    return x + x + nbytes(P.f32, (width,)), 4 * t.b * width  # read + write, ~4 flop/elt


def _silu_mul(cfg, t: TickShape) -> tuple[int, int]:
    """SiLU(gate) * up over the MLP intermediate: gate+up read, product write."""
    inter = cfg.intermediate_size
    return 3 * nbytes(P.bf16, (t.b, inter)), t.b * inter


def tick_rows(cfg, t: TickShape) -> list[dict]:
    """Every kernel one decode tick launches, with per-launch bytes/flops and the
    number of launches per tick. Linears are enumerated from ``param_specs`` (one
    row per distinct weight shape); the fused kernels one per layer they appear
    on. Sums over rows times ``count`` are the tick's cost.

    Each row: ``name, shape, count, bytes, flops, face`` (bytes/flops PER launch).
    """
    from .model import param_specs

    specs = param_specs(cfg)
    n_full = len(cfg.full_attn_layers)
    n_gdn = cfg.num_layers - n_full
    rows: list[dict] = []

    def add(name: str, shape: str, count: int, by: int, fl: int,
            face: Format | None = None) -> None:
        rows.append(dict(name=name, shape=shape, count=count, bytes=by, flops=fl, face=face))

    # Enumerate every layer's actual launches, then collapse equal
    # (key, shape, face) GEMVs to one row with the launch count. Without t.faces
    # every launch takes t.weight, so the collapse yields one representative row
    # per linear (e.g. gate_proj x 64); with faces each layer is priced at the
    # face its OWN checkpoint weights carry (the 27B mixes nvfp4 and fp8).
    faces = t.faces or {}
    full_layers = set(cfg.full_attn_layers)
    launches: list[tuple[str, tuple, Format]] = []
    for i in range(cfg.num_layers):
        keys = (
            ("q_proj", "k_proj", "v_proj", "o_proj")
            if i in full_layers
            else ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
        )
        keys += ("gate_proj", "up_proj", "down_proj")
        for k in keys:
            full = f"layers.{i}.{k}"
            shape, fmt = faces.get(full, (tuple(specs[full]), t.weight))
            launches.append((k, tuple(shape), fmt))
    grouped: dict[tuple, int] = {}
    order: list[tuple] = []
    for sig in launches:
        if sig not in grouped:
            grouped[sig] = 0
            order.append(sig)
        grouped[sig] += 1
    for k, shape, fmt in order:
        by, fl = _gemv(shape, fmt, t)
        add(k, f"{shape[0]}x{shape[1]}", grouped[(k, shape, fmt)], by, fl, fmt)

    # The lm_head multiplies the full [vocab, hidden] matrix EVERY decode step
    # (the sampled row is the embedding lookup, a separate smaller op).
    if "lm_head" in specs:
        shape, fmt = faces.get("lm_head", (tuple(specs["lm_head"]), t.weight))
        by, fl = _gemv(shape, fmt, t)
        add("lm_head", f"{shape[0]}x{shape[1]}", 1, by, fl, fmt)

    ab, af = _paged_attention_decode(cfg, t)
    add(
        "paged_attention_decode",
        f"b{t.b} kv{n_full and cfg.num_kv_heads} s{t.s} d{cfg.head_dim}",
        n_full,
        ab,
        af,
    )
    gb, gf = _gdn_decode_fused(cfg, t)
    add(
        "gdn_decode_fused",
        f"b{t.b} nvh{cfg.linear_num_value_heads} dk{cfg.linear_key_head_dim}",
        n_gdn,
        gb,
        gf,
    )
    # three norms per layer (input/post at hidden width; tiny q/k norms omitted),
    # plus the final norm once.
    rb, rf = _rmsnorm(cfg, t, cfg.hidden_size)
    add("rmsnorm", f"b{t.b} h{cfg.hidden_size}", 2 * cfg.num_layers + 1, rb, rf)
    sb, sf = _silu_mul(cfg, t)
    add("silu_mul", f"b{t.b} inter{cfg.intermediate_size}", cfg.num_layers, sb, sf)
    return rows


def tick_totals(cfg, t: TickShape) -> tuple[int, int]:
    """(bytes, flops) summed over every kernel launch in one decode tick."""
    return (
        sum(r["bytes"] * r["count"] for r in tick_rows(cfg, t)),
        sum(r["flops"] * r["count"] for r in tick_rows(cfg, t)),
    )
