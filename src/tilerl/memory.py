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

#: owners the plan emits. prefix_entries is a row only when demoted to host/ssd; the
#: trainer's tape is outside the engine (docs/design-cost-model.md).
OWNERS = ("weights", "kv_pool", "draft_pool", "state_slots", "graph_pad",
          "prefix_entries", "staging")

_DTYPE_FMT = {16: bf16, 32: f32}


def _dtype_fmt(dtype) -> Format:
    return dtype if isinstance(dtype, Format) else _DTYPE_FMT.get(dtype.itemsize * 8, bf16)


@dataclass(frozen=True)
class Row:
    """One allocation or budget. ``bytes`` is direct when a tensor doesn't map to one
    fmt/shape (the served quantized weight sidecars, a budget)."""

    tier: str  # device | host | ssd
    owner: str
    n: int
    fmt: object = None
    shape: tuple[int, ...] = ()
    count: int = 1
    note: str = ""

    def as_dict(self) -> dict:
        return {"tier": self.tier, "owner": self.owner, "bytes": self.n,
                "shape": list(self.shape), "count": self.count, "note": self.note}


def _kv_shape(cfg, num_blocks: int) -> tuple[int, ...]:
    """The PagedKvPool K-or-V tensor shape: [planes, blocks, kv_heads, BLOCK, head_dim]."""
    return (len(cfg.full_attn_layers), num_blocks, cfg.num_kv_heads,
            BLOCK_TOKENS, cfg.head_dim)


def per_kv_block_bytes(cfg, kv_io, kv_fp8=None, draft_layers: int = 0) -> int:
    """Bytes of ONE KV block, the fp8 scale plane and the draft pool included.

    THE single formula — build_engine's fit and the plan's kv_pool row both call it, so
    they cannot disagree. Mirrors PagedKvPool's k_pool/v_pool/(k|v)_scale allocation:
    K+V (count 2) over every full-attn plane, plus the draft's plain-IO pool (no scale).
    """
    planes = 2 * len(cfg.full_attn_layers)
    io_fmt = _dtype_fmt(kv_io)
    kv_fmt = kv_format(cfg.head_dim) if kv_fp8 is not None else io_fmt
    pair_shape = (2, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim)
    return (nbytes(kv_fmt, (planes, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim))
            + draft_layers * nbytes(io_fmt, pair_shape))


def fit_num_blocks(cfg, device_free: int, kv_io, kv_fp8=None, draft_layers: int = 0,
                   cap: int = 0, floor: int = 64) -> int:
    """The KV blocks that fit ``device_free`` under the pool rule. Pure arithmetic: on CPU
    the caller passes device_free (build_engine passes mem_get_info()*POOL_FRACTION)."""
    n = max(floor, int(device_free * POOL_FRACTION)
            // per_kv_block_bytes(cfg, kv_io, kv_fp8, draft_layers))
    return min(n, cap) if cap else n


def state_shapes(cfg, num_slots: int, spec_steps: int = 0) -> list[tuple[tuple, object, str]]:
    """The (shape, dtype, note) of every LinearStatePool tensor — the one list both
    LinearStatePool allocates and the state rows price."""
    n_lin = cfg.num_layers - len(cfg.full_attn_layers)
    nvh, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    conv_k = cfg.linear_conv_kernel_dim - 1
    out = [((num_slots, n_lin, nvh, dk, dv), "state", "gdn state")]
    if n_lin and conv_k:
        out.append(((num_slots, n_lin, 2, conv_k, cfg.linear_qkv_dim), "state", "conv windows"))
    if spec_steps and n_lin:
        out.append(((num_slots, n_lin, spec_steps, nvh, dk, dv), "spec", "spec states"))
        if conv_k:
            out.append(((num_slots, n_lin, spec_steps, conv_k, cfg.linear_qkv_dim),
                        "spec", "spec conv"))
    out.append((((num_slots,),), "parity", "win_parity"))
    return out


def weight_row(params: dict) -> Row:
    """Served weights summed from the materialized tensors. A quantized model holds
    .wq/.scale/.oscale sidecars (and deletes the bf16 master), so this is the storage
    truth, not param_specs — the gate asserts this equals the live params."""
    return Row("device", "weights", sum(t.numel() * t.element_size() for t in params.values()),
                note="sum of materialized params")


def plan(cfg, params: dict | None, device_free: int, *, num_slots: int, num_blocks: int,
         spec_steps: int = 0, state_dtype=f32, kv_io=bf16, kv_fp8=None,
         explicit_state_budget: int = 0, dram_budget: int = 0,
         draft_layers: int = 0) -> list[Row]:
    """The device rows the engine holds, in allocation order, plus the budget rules as
    rows. ``num_blocks`` is what build_engine built (fitted via fit_num_blocks or explicit).
    Weights need the materialized params (quantized sidecars); None leaves them pending."""
    rows: list[Row] = []
    if params is not None:
        rows.append(weight_row(params))
    fmt = _dtype_fmt(state_dtype)
    for shape, kind, note in state_shapes(cfg, num_slots, spec_steps):
        rows.append(Row("device", "state_slots",
                        4 * num_slots if kind == "parity" else nbytes(fmt, shape),
                        Format(32) if kind == "parity" else fmt, shape, note=note))
    rows.append(Row("device", "kv_pool",
                    per_kv_block_bytes(cfg, kv_io, kv_fp8, draft_layers) * num_blocks,
                    note=f"{num_blocks} blocks"))
    # Budget rules as rows (owner _budget so a reader summing allocations skips them): the
    # pool rule's input is the same free memory fit_num_blocks consumes; the snapshot store
    # then takes STATE_FRACTION of that.
    if device_free:
        rows.append(Row("device", "kv_pool_budget", int(device_free * POOL_FRACTION),
                        note="rule free*2/3"))
        rows.append(Row("device", "prefix_entries_budget",
                        explicit_state_budget or int(device_free * STATE_FRACTION),
                        note="rule free/4"))
    if dram_budget:
        rows.append(Row("host", "prefix_entries_budget", dram_budget, note="dram tier budget"))
    return rows
