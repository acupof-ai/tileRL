"""The attention-decode kernel's declared bytes_moved equals the KV bytes the
pool actually hands it.

This is the one place the roofline declaration can silently drift from the
thing it costs: both price a gathered K/V read of ``[b, kv_heads, s, d]``, but
one lives in ``kernel_cost`` and the other is the ``PagedKvPool``'s real
allocation. The gate crosses the two so a changed head count, a switched KV
format, or a missed scale plane fails here instead of shipping a table that
disagrees with the kernel.

Independent derivation: the pool tensors' OWN ``element_size`` (the storage
truth) priced for the gathered span, against the declaration priced through
``precision.nbytes``. bf16 and fp8 (with its per-head_dim f32 scale) are both
checked.
"""

from __future__ import annotations

import pytest
import torch

from tilerl.config import tiny
from tilerl.kernel_cost import TickShape, _paged_attention_decode, tick_rows, tick_totals
from tilerl.kv_cache import PagedKvPool
from tilerl.precision import bf16, kv_format, nbytes


def _pool_kv_bytes_per_token(pool: PagedKvPool) -> int:
    """K+V bytes for ONE token across the pool's planes, from the real tensors.

    Independent of nbytes: payload element_size plus the scale grids.
    """
    kv = pool.num_layers * pool.num_kv_heads * pool.head_dim
    total = 2 * kv * pool.k_pool.element_size()
    if pool.k_scale is not None:
        # one scale per (layer, head, token): kv // head_dim scales, times 2 for K/V
        total += 2 * (kv // pool.head_dim) * pool.k_scale.element_size()
    return total


def _pool(cfg, kv_dtype):
    # KV IO dtype: bf16 off fp8 (the served/paged path), f32 storage only with the
    # fp8 planes — mirror what the declaration prices.
    io_dtype = torch.float32 if kv_dtype is not None else torch.bfloat16
    return PagedKvPool(
        num_blocks=8,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        num_layers=len(cfg.full_attn_layers),
        dtype=io_dtype,
        kv_fp8=kv_dtype,
    )


@pytest.mark.parametrize("use_fp8", [False, True])
def test_attn_decode_bytes_equal_pool_kv_read(use_fp8):
    cfg = tiny()
    kv_dtype = torch.float8_e4m3fn if use_fp8 else None
    kv_fmt = kv_format(cfg.head_dim) if use_fp8 else bf16
    b, s = 2, 37
    pool = _pool(cfg, kv_dtype)

    # The declaration's per-layer total minus its q/o bf16 terms leaves the KV read.
    tick = TickShape(b=b, s=s, kv=kv_fmt, weight=bf16)
    declared, _ = _paged_attention_decode(cfg, tick)
    qo = 2 * nbytes(bf16, (b, cfg.num_attention_heads, cfg.head_dim))
    declared_kv = declared - qo

    # Independent: per token the pool stores _pool_kv_bytes_per_token, and one
    # layer's decode reads s tokens for each of b rows.
    per_token_per_layer = _pool_kv_bytes_per_token(pool) // pool.num_layers
    actual_kv = per_token_per_layer * b * s

    assert declared_kv == actual_kv, (
        f"attn-decode declares {declared_kv} KV bytes but the pool hands the kernel "
        f"{actual_kv} for b={b} s={s} ({per_token_per_layer}/token/layer)"
    )


def test_pool_bytes_per_token_matches_nbytes_formula():
    """The pool's own bytes_per_token (element_size truth) equals nbytes on fp8."""
    cfg = tiny()
    pool = _pool(cfg, torch.float8_e4m3fn)
    elt_per_token = pool.num_layers * pool.num_kv_heads * pool.head_dim * 2
    assert pool.bytes_per_token == nbytes(kv_format(cfg.head_dim), (elt_per_token,))


def test_tick_rows_cover_the_launched_set_and_total_is_positive():
    cfg = tiny()
    tick = TickShape(b=1, s=64, kv=bf16, weight=bf16)
    names = {r["name"] for r in tick_rows(cfg, tick)}
    assert {"paged_attention_decode", "gdn_decode_fused", "rmsnorm", "silu_mul"} <= names
    bytes_, flops = tick_totals(cfg, tick)
    assert bytes_ > 0 and flops > 0
    for r in tick_rows(cfg, tick):
        assert r["count"] >= 1 and r["bytes"] > 0 and r["flops"] >= 0
