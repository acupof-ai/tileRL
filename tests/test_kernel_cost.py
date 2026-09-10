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
from tilerl.kernel_cost import (
    TickShape,
    _gdn_chunk_forward,
    _gdn_chunk_matmul_dims,
    _paged_attention_decode,
    prefill_kv_write_bytes,
    prefill_rows,
    prefill_totals,
    tick_rows,
    tick_totals,
)
from tilerl.kv_cache import PagedKvPool
from tilerl.precision import bf16, f32, kv_format, nbytes


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


@pytest.mark.parametrize("use_fp8", [False, True])
def test_attn_prefill_kv_write_bytes_equal_pool_kv_write(use_fp8):
    """The prefill declaration's K/V WRITE half equals what the pool stores for
    b sequences of s tokens across every full-attn plane — the prefill twin of
    the decode byte gate, priced from the real tensors independently."""
    cfg = tiny()
    kv_dtype = torch.float8_e4m3fn if use_fp8 else None
    kv_fmt = kv_format(cfg.head_dim) if use_fp8 else bf16
    b, s = 2, 53
    pool = _pool(cfg, kv_dtype)

    declared = prefill_kv_write_bytes(cfg, TickShape(b=b, s=s, kv=kv_fmt, weight=bf16))

    # Independent truth: per token the pool stores _pool_kv_bytes_per_token
    # (K+V across all planes); prefill writes b*s fresh tokens, all planes.
    actual = _pool_kv_bytes_per_token(pool) * b * s

    assert declared == actual, (
        f"prefill declares {declared} KV-write bytes but the pool stores {actual} "
        f"for b={b} s={s}"
    )


def test_prefill_rows_weight_streamed_once_and_lm_head_scores_last_token():
    """Each prefill linear streams its WEIGHT once (that term is M-independent);
    the M-row input/output activations grow with M and flops scale with M. lm_head
    costs only the b last tokens, not b*s."""
    cfg = tiny()
    t32 = TickShape(b=1, s=32, kv=bf16, weight=bf16)
    t64 = TickShape(b=1, s=64, kv=bf16, weight=bf16)
    rows32 = {r["name"]: r for r in prefill_rows(cfg, t32)}
    rows64 = {r["name"]: r for r in prefill_rows(cfg, t64)}

    # down_proj bytes at s=64 minus s=32 equals ONLY the added activation bytes
    # (weight streamed once, unchanged), and flops double.
    from tilerl.model import param_specs
    from tilerl.precision import nbytes as _nb
    gdn_layer = next(f"layers.{i}" for i in range(cfg.num_layers)
                     if i not in set(cfg.full_attn_layers))
    out, inn = tuple(param_specs(cfg)[f"{gdn_layer}.down_proj"])
    added_act = _nb(bf16, (32, inn)) + _nb(bf16, (32, out))
    d32, d64 = rows32["down_proj"], rows64["down_proj"]
    assert d64["bytes"] - d32["bytes"] == added_act
    assert d64["flops"] == 2 * d32["flops"]

    # lm_head, when untied, is independent of s: only the last token/sequence scored.
    if "lm_head" in rows32:
        assert rows32["lm_head"]["bytes"] == rows64["lm_head"]["bytes"]
        assert rows32["lm_head"]["flops"] == rows64["lm_head"]["flops"]
    else:
        # tied embed_tokens model: lm_head row is absent by construction.
        assert "lm_head" not in rows64


def test_gdn_prefill_flops_derived_from_cfg_on_tiny():
    """Value gate (not positivity): the gdn_chunk_forward flops equal an
    independent cfg-derived recomputation of the eight batched matmuls. A
    hardcoded 128-dim shape is ~15x wrong on tiny's dk=dv=16 and a positivity
    check would stay green."""
    cfg = tiny()
    dk, dv, nvh = cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_num_value_heads
    n_gdn = cfg.num_layers - len(cfg.full_attn_layers)
    n = 64
    b, s = 1, n  # exactly one chunk per sequence
    _, declared = _gdn_chunk_forward(cfg, TickShape(b=b, s=s, kv=bf16, weight=bf16))

    # Independent: eight matmuls spelled out for tiny's dk=dv=16, NOT read back
    # from the helper under test (a hardcoded-128 helper would otherwise be the
    # source on both sides and the equality could never fail).
    n = 64
    d = 16
    literal = [
        (n, n, d), (n, n, d), (n, n, d), (n, d, d),
        (n, n, d), (n, d, d), (n, n, d), (d, n, d),
    ]
    per_head_chunk = sum(2 * m * nn * k for m, nn, k in literal)
    expected = n_gdn * b * nvh * 1 * per_head_chunk
    assert declared == expected
    # And the helper itself must equal that literal at tiny dims.
    assert _gdn_chunk_matmul_dims(dk, dv, n) == tuple(literal)


def test_gdn_prefill_chunks_are_per_sequence_not_raveled():
    """Value gate: b=2, s=65 (one token past 64) must cost FOUR chunks — two per
    sequence — not three from ceil(2*65/64). Chunking resets at the sequence
    boundary, and the carried state is per sequence."""
    cfg = tiny()
    dk, dv, nvh = cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_num_value_heads
    n_gdn = cfg.num_layers - len(cfg.full_attn_layers)
    n = 64
    b, s = 2, 65
    declared_bytes, declared_flops = _gdn_chunk_forward(
        cfg, TickShape(b=b, s=s, kv=bf16, weight=bf16))

    chunks_seq = 2  # ceil(65/64)
    literal = _gdn_chunk_matmul_dims(dk, dv, n)
    per_head_chunk = sum(2 * m * nn * k for m, nn, k in literal)
    expected_flops = n_gdn * b * nvh * chunks_seq * per_head_chunk
    assert declared_flops == expected_flops, "flops raveled batch into sequence"

    state = nbytes(f32, (b, nvh, dk, dv))
    act = nbytes(f32, (b * nvh, s, 3 * dk + dv + 2))
    out = nbytes(bf16, (b, nvh, s, dv))
    expected_bytes = state * (2 * chunks_seq - 1) + act + out
    assert declared_bytes == expected_bytes, "GDN bytes must use per-sequence chunk count"


def test_tick_rows_cover_the_launched_set_and_total_is_positive():
    cfg = tiny()
    tick = TickShape(b=1, s=64, kv=bf16, weight=bf16)
    names = {r["name"] for r in tick_rows(cfg, tick)}
    assert {"paged_attention_decode", "gdn_decode_fused", "rmsnorm", "silu_mul"} <= names
    bytes_, flops = tick_totals(cfg, tick)
    assert bytes_ > 0 and flops > 0
    for r in tick_rows(cfg, tick):
        assert r["count"] >= 1 and r["bytes"] > 0 and r["flops"] >= 0
    pnames = {r["name"] for r in prefill_rows(cfg, tick)}
    assert {"paged_attention_prefill", "gdn_chunk_forward"} <= pnames
    pb, pf = prefill_totals(cfg, tick)
    assert pb > 0 and pf > 0
