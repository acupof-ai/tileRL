"""Per-kernel roofline declarations for one 27B decode tick.

Each launched kernel declares the bytes it moves and the flops it performs at a
concrete shape, priced through :func:`tilerl.precision.nbytes` — the ONLY byte
arithmetic (docs/design-cost-model.md, "Kernel cost"). Weight bytes come from
:func:`tilerl.model.param_specs`, so a declaration never restates a matrix dim.

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


def _gemv(spec: tuple[int, int], t: TickShape) -> tuple[int, int]:
    """Decode linear y[b,out] = x[b,in] @ W^T[out,in]: weight streamed once,
    plus the b-row input/output."""
    out, inn = spec
    bytes_ = nbytes(t.weight, spec) + nbytes(P.bf16, (t.b, inn)) + nbytes(P.bf16, (t.b, out))
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
    ``s' = decay s + R^T d``. At n=1, ``W`` and ``R`` are the per-token
    ``[b, nvh, dk, dv]`` gate-weighted key projections read once each; the state
    is read and written. Projections themselves are the nvfp4 linears costed
    separately, so this is only the recurrence sweep.
    """
    nvh, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    state = nbytes(P.f32, (t.b, nvh, dk, dv))
    wr = 2 * nbytes(P.bf16, (t.b, nvh, dk, dv))  # W and R
    flops = 4 * t.b * nvh * dk * dk * dv  # W s and R^T d, each [dk,dk]@[dk,dv]-ish
    return 2 * state + wr, flops


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

    Each row: ``name, shape, count, bytes, flops`` (bytes/flops PER launch).
    """
    from .model import param_specs

    specs = param_specs(cfg)
    n_full = len(cfg.full_attn_layers)
    n_gdn = cfg.num_layers - n_full
    full_layer = f"layers.{cfg.full_attn_layers[0]}" if n_full else None
    gdn_layer = next(
        (f"layers.{i}" for i in range(cfg.num_layers) if i not in set(cfg.full_attn_layers)), None
    )
    rows: list[dict] = []

    def add(name: str, shape: str, count: int, by: int, fl: int) -> None:
        rows.append(dict(name=name, shape=shape, count=count, bytes=by, flops=fl))

    # nvfp4 linears, streamed once per launch; weight bytes priced through nbytes.
    groups = [
        (n_full, full_layer, ("q_proj", "k_proj", "v_proj", "o_proj")),
        (n_gdn, gdn_layer, ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")),
        (cfg.num_layers, gdn_layer, ("gate_proj", "up_proj", "down_proj")),
    ]
    for count, layer, keys in groups:
        if not count or layer is None:
            continue
        for k in keys:
            spec = tuple(specs[f"{layer}.{k}"])
            by, fl = _gemv(spec, t)
            add(k, f"{spec[0]}x{spec[1]}", count, by, fl)

    # The lm_head multiplies the full [vocab, hidden] matrix EVERY decode step
    # (the sampled row is the embedding lookup, a separate smaller op). Untied on
    # the 27B and nvfp4-packed; ~0.64 GB, a larger stream than the fp8 KV scale.
    if "lm_head" in specs:
        spec = tuple(specs["lm_head"])
        by, fl = _gemv(spec, t)
        add("lm_head", f"{spec[0]}x{spec[1]}", 1, by, fl)

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
