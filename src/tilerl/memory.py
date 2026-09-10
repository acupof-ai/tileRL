"""The memory ledger: every byte the engine HOLDS is derived in ONE place.

``plan`` is the allocator's INPUT, not a second computation of a number the pool also
computes: the fit in build_engine calls :func:`fit_num_blocks`, the KV pool row calls
:func:`per_kv_block_bytes`, and LinearStatePool's tensors are the shapes in
:func:`state_shapes`. A CPU test that merely shows the two agree today cannot stop them
drifting tomorrow, so there is only one formula for each number.

GPU-measured columns are pending-remote; the budget arithmetic is not — ``plan`` takes
``device_free`` as a parameter, so on CPU you pass the number and the free*2/3 pool
fit and free/4 snapshot budget rows are exercised without a card. On the CPU tiny model
the weights, KV pool and state rows are exact (tests/test_memory_ledger.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .kv_cache import BLOCK_TOKENS
from .precision import Format, bf16, f32, kv_format, nbytes

#: budget rules build_engine uses (docs/design-cost-model.md); each is the row it makes.
POOL_FRACTION = 2 / 3  # the KV pool takes 2/3 of post-weights free
STATE_FRACTION = 1 / 4  # the resident-snapshot store takes 1/4 of what remains

_DTYPE_FMT = {16: bf16, 32: f32}


def _dtype_fmt(dtype) -> Format:
    return dtype if isinstance(dtype, Format) else _DTYPE_FMT.get(dtype.itemsize * 8, bf16)


@dataclass(frozen=True)
class Row:
    """One allocation or budget."""

    tier: str  # device | host | ssd
    owner: str
    n: int
    note: str = ""


def per_kv_block_bytes(cfg, kv_io, kv_fp8=None) -> int:
    """Bytes of ONE trunk KV block: K+V over every full-attn plane, the fp8 per-head_dim
    scale plane included. MAIN pool only — the draft's pool is a separate pool and a
    separate row (:func:`draft_per_block_bytes`); folding it in here would count it twice
    (once in kv_pool, once in draft_pool)."""
    planes = 2 * len(cfg.full_attn_layers)
    io_fmt = _dtype_fmt(kv_io)
    kv_fmt = kv_format(cfg.head_dim) if kv_fp8 is not None else io_fmt
    return nbytes(kv_fmt, (planes, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim))


def draft_per_block_bytes(cfg, kv_io, draft_layers: int) -> int:
    """The draft's per-block K+V pool: plain IO dtype, no fp8 scale plane (DraftHead.attach
    builds PagedKvPool without kv_fp8), one pair per draft layer."""
    return draft_layers * nbytes(_dtype_fmt(kv_io),
                                  (2, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim))


def fit_num_blocks(cfg, device_free: int, kv_io, kv_fp8=None, draft_layers: int = 0,
                   cap: int = 0, floor: int = 64) -> int:
    """The KV blocks that fit ``device_free`` under the pool rule. The fit's denominator is
    main + draft per block (the draft pool mirrors num_blocks), but the fitted count feeds
    BOTH pools — it is one number. Pure arithmetic: on CPU the caller passes device_free."""
    per_block = per_kv_block_bytes(cfg, kv_io, kv_fp8) + draft_per_block_bytes(
        cfg, kv_io, draft_layers)
    n = max(floor, int(device_free * POOL_FRACTION) // per_block)
    return min(n, cap) if cap else n


def state_shapes(cfg, num_slots: int, spec_steps: int = 0) -> list[tuple[tuple, object]]:
    """The (shape, dtype-kind) of every LinearStatePool tensor — the one list both the pool
    and the state rows price. kind 'state' uses the pool storage dtype, 'parity' is int32."""
    n_lin = cfg.num_layers - len(cfg.full_attn_layers)
    nvh, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    conv_k = cfg.linear_conv_kernel_dim - 1
    out: list[tuple[tuple, object]] = [((num_slots, n_lin, nvh, dk, dv), "state")]
    if n_lin and conv_k:
        out.append(((num_slots, n_lin, 2, conv_k, cfg.linear_qkv_dim), "state"))
    if spec_steps and n_lin:
        out.append(((num_slots, n_lin, spec_steps, nvh, dk, dv), "state"))
        if conv_k:
            out.append(((num_slots, n_lin, spec_steps, conv_k, cfg.linear_qkv_dim), "state"))
    out.append(((num_slots,), "parity"))
    return out


def _state_bytes(cfg, num_slots: int, state_fmt: Format, spec_steps: int = 0) -> int:
    return sum(nbytes(f32 if kind == "parity" else state_fmt, shape)
               for shape, kind in state_shapes(cfg, num_slots, spec_steps))


def weight_row(params: dict) -> Row:
    """Served weights summed from the materialized tensors. A quantized model holds
    .wq/.scale/.oscale sidecars (and deletes the bf16 master), so this is the storage
    truth, not param_specs — the gate asserts this equals the live params."""
    return Row("device", "weights", sum(t.numel() * t.element_size() for t in params.values()))


def plan(cfg, params: dict | None, device_free: int, *, num_slots: int, num_blocks: int,
         spec_steps: int = 0, state_dtype=f32, kv_io=bf16, kv_fp8=None,
         explicit_state_budget: int = 0, dram_budget: int = 0,
         draft_layers: int = 0) -> list[Row]:
    """The device rows the engine holds plus the budget rules. ``num_blocks`` is what
    build_engine built (fitted via fit_num_blocks or explicit). Weights need the
    materialized params; None leaves them pending. ``draft_layers>0`` adds the draft pool
    row (its share of every KV block, priced in the block formula)."""
    rows: list[Row] = []
    if params is not None:
        rows.append(weight_row(params))
    rows.append(Row("device", "state_slots",
                    _state_bytes(cfg, num_slots, _dtype_fmt(state_dtype), spec_steps)))
    rows.append(Row("device", "kv_pool",
                    per_kv_block_bytes(cfg, kv_io, kv_fp8) * num_blocks,
                    f"{num_blocks} blocks"))
    if draft_layers:
        rows.append(Row("device", "draft_pool",
                        draft_per_block_bytes(cfg, kv_io, draft_layers) * num_blocks,
                        f"{draft_layers} draft layers x {num_blocks} blocks"))
    # Budget rules as *_budget rows so a reader summing allocations skips them.
    if device_free:
        rows.append(Row("device", "kv_pool_budget", int(device_free * POOL_FRACTION),
                        "rule free*2/3"))
        rows.append(Row("device", "prefix_entries_budget",
                        explicit_state_budget or int(device_free * STATE_FRACTION),
                        "rule free/4"))
    if dram_budget:
        rows.append(Row("host", "prefix_entries_budget", dram_budget, "dram tier budget"))
    return rows
