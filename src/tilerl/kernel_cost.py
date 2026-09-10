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
enumerated from param_specs, not listed by hand. Prefill rows live next to them
(``prefill_rows``); backward is a later row. A kernel without a grounded shape is
left undeclared rather than costed against one it does not see.
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


#: chunkwise-WY prefill chunk length — mirrors backend._WY_CHUNK, grounded here so
#: a declaration never imports a backend scheduling constant.
_PREFILL_CHUNK = 64


def _gdn_chunk_matmul_dims(dk: int, dv: int, n: int) -> tuple[tuple[int, int, int], ...]:
    """The eight batched matmuls of ``kernels.reference._gdn_chunk_fwd`` per
    (value head, chunk of n tokens), as (M, N, K). Derived from the model's
    ``dk``/``dv`` — never hardcoded — so the table is right on any config:

    KK^T/QK^T (n,n,dk), M@bV/A@d (n,n,dv), M@beK (n,n,dk),
    W@S/P@S (n,dk,dv), R^T@d (dk,n,dv).
    """
    return (
        (n, n, dk), (n, n, dv), (n, n, dk), (n, dk, dv),
        (n, n, dk), (n, dk, dv), (n, n, dv), (dk, n, dv),
    )


def _linear_prefill(spec: tuple[int, int], t: TickShape) -> tuple[int, int]:
    """Prefill linear y[m,out] = x[m,in] @ W^T over m = b*S token rows: weight
    streamed once for the whole chunk, plus the m-row input and output."""
    out, inn = spec
    m = t.b * t.s
    bytes_ = nbytes(t.weight, spec) + nbytes(P.bf16, (m, inn)) + nbytes(P.bf16, (m, out))
    return bytes_, 2 * m * out * inn


def prefill_kv_write_bytes(cfg, t: TickShape) -> int:
    """K and V one prefill writes to the paged pool for ``b`` sequences of ``s``
    tokens, across every full-attention plane, priced in the pool's own format
    (fp8 plane + its per-head_dim f32 scale). This is the byte gate's side."""
    return 2 * nbytes(t.kv, (t.b, cfg.num_kv_heads, t.s, cfg.head_dim)) * len(
        cfg.full_attn_layers
    )


def _paged_attention_prefill(cfg, t: TickShape) -> tuple[int, int]:
    """One full-attn layer's causal prefill attention over s tokens.

    Charged per HBM direction crossed: K and V are WRITTEN to the paged pool for
    the first time and READ back as the attention's operands (the write half is
    what :func:`prefill_kv_write_bytes` pins to the pool). Q is read and O written
    once. Causal masking halves the pairs, not the loaded K/V blocks, so bytes are
    the full K/V both directions while flops count only the lower triangle.
    """
    hq, d = cfg.num_attention_heads, cfg.head_dim
    kv_one_way = nbytes(t.kv, (t.b, cfg.num_kv_heads, t.s, d))  # K xor V
    kv_bytes = 4 * kv_one_way  # write K+V, read K+V
    qo = nbytes(P.bf16, (t.b, hq, t.s, d)) * 2
    pairs = t.s * (t.s + 1) // 2  # lower triangle including the diagonal
    flops = 4 * t.b * hq * pairs * d
    return kv_bytes + qo, flops


def _gdn_chunk_forward(cfg, t: TickShape) -> tuple[int, int]:
    """The GDN chunked-forward recurrence over a prefill of s tokens, one layer.

    Flops are the eight batched matmuls of ``reference._gdn_chunk_fwd`` (dimensions
    derived from ``dk``/``dv``) over every ``_PREFILL_CHUNK``-token chunk of EVERY
    sequence and every value head. Chunking is per sequence — ``b * ceil(s/n)``,
    not ``ceil(b*s/n)`` — because the recurrence resets at a sequence boundary.

    Bytes charge only what crosses HBM: the f32 activations the in_proj linears
    hand in and the bf16 output, plus the recurrent state. Each sequence carries
    one state per head: written by every chunk and read by every chunk but that
    sequence's first, hence ``b * (2*chunks_per_seq - 1)``. The per-chunk M/W/U
    intermediates are in-kernel recomputation and move nothing. The triangular
    solve inside chunk assembly is ~0.07% and is omitted.
    """
    nvh, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    n = _PREFILL_CHUNK
    chunks_seq = -(-t.s // n)  # ceil(s/n) per sequence
    nchunks = t.b * chunks_seq
    matmuls = _gdn_chunk_matmul_dims(dk, dv, n)
    flops_per_chunk_head = sum(2 * m * nn * k for m, nn, k in matmuls)
    flops = nchunks * nvh * flops_per_chunk_head
    # state is per (sequence, value head): b*nvh [dk,dv] planes, carried across chunks.
    state = nbytes(P.f32, (t.b, nvh, dk, dv))
    act = nbytes(P.f32, (t.b * nvh, t.s, 3 * dk + dv + 2))
    out = nbytes(P.bf16, (t.b, nvh, t.s, dv))
    bytes_ = state * (2 * chunks_seq - 1) + act + out
    return bytes_, flops


def _rmsnorm_prefill(cfg, t: TickShape, width: int) -> tuple[int, int]:
    m = t.b * t.s
    x = nbytes(P.bf16, (m, width))
    return x + x + nbytes(P.f32, (width,)), 4 * m * width


def _silu_mul_prefill(cfg, t: TickShape) -> tuple[int, int]:
    m = t.b * t.s
    inter = cfg.intermediate_size
    return 3 * nbytes(P.bf16, (m, inter)), m * inter


def prefill_rows(cfg, t: TickShape) -> list[dict]:
    """Every kernel one prefill of ``b`` sequences of ``s`` tokens launches.

    Linears process all ``b*s`` token rows in one GEMM (weight streamed once);
    ``lm_head`` scores only the last token per sequence (m=b), since prefill
    hidden states are not all projected. Same row schema as :func:`tick_rows`.
    """
    from .model import param_specs

    specs = param_specs(cfg)
    n_full = len(cfg.full_attn_layers)
    n_gdn = cfg.num_layers - n_full
    gdn_layer = next(
        (f"layers.{i}" for i in range(cfg.num_layers) if i not in set(cfg.full_attn_layers)), None
    )
    rows: list[dict] = []

    def add(name: str, shape: str, count: int, by: int, fl: int) -> None:
        rows.append(dict(name=name, shape=shape, count=count, bytes=by, flops=fl))

    groups = [
        (n_full, f"layers.{cfg.full_attn_layers[0]}",
         ("q_proj", "k_proj", "v_proj", "o_proj")),
        (n_gdn, gdn_layer,
         ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")),
        (cfg.num_layers, gdn_layer, ("gate_proj", "up_proj", "down_proj")),
    ]
    for count, layer, keys in groups:
        if not count or layer is None:
            continue
        for k in keys:
            spec = tuple(specs[f"{layer}.{k}"])
            by, fl = _linear_prefill(spec, t)
            add(k, f"M{t.b*t.s} {spec[0]}x{spec[1]}", count, by, fl)

    if "lm_head" in specs:
        spec = tuple(specs["lm_head"])
        last = TickShape(b=t.b, s=1, kv=t.kv, weight=t.weight)
        by, fl = _linear_prefill(spec, last)
        add("lm_head", f"M{t.b} {spec[0]}x{spec[1]}", 1, by, fl)

    ab, af = _paged_attention_prefill(cfg, t)
    add(
        "paged_attention_prefill",
        f"b{t.b} s{t.s} kv{cfg.num_kv_heads} d{cfg.head_dim} causal",
        n_full, ab, af,
    )
    gb, gf = _gdn_chunk_forward(cfg, t)
    add(
        "gdn_chunk_forward",
        f"b{t.b} s{t.s} nvh{cfg.linear_num_value_heads} chunk{_PREFILL_CHUNK}",
        n_gdn, gb, gf,
    )
    rb, rf = _rmsnorm_prefill(cfg, t, cfg.hidden_size)
    add("rmsnorm", f"M{t.b*t.s} h{cfg.hidden_size}", 2 * cfg.num_layers + 1, rb, rf)
    sb, sf = _silu_mul_prefill(cfg, t)
    add("silu_mul", f"M{t.b*t.s} inter{cfg.intermediate_size}", cfg.num_layers, sb, sf)
    return rows


def prefill_totals(cfg, t: TickShape) -> tuple[int, int]:
    """(bytes, flops) summed over every kernel launch in one prefill."""
    return (
        sum(r["bytes"] * r["count"] for r in prefill_rows(cfg, t)),
        sum(r["flops"] * r["count"] for r in prefill_rows(cfg, t)),
    )


def tick_rows(cfg, t: TickShape) -> list[dict]:
    """Every kernel one decode tick launches, with per-launch bytes/flops and the
    number of launches per tick. Linears are enumerated from ``param_specs`` (one
    row per distinct weight shape); the fused kernels one per layer they appear
    on. Sums over rows times ``count`` are the tick's cost.

    Each row: ``name, shape, count, bytes, flops`` (bytes/flops PER launch).
    ``lm_head`` is priced for SERVING prefill, which projects only the last token
    per sequence; training's full-sequence lm_head is a different row, later.
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
