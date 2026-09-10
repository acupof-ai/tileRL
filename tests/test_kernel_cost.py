"""The attention-decode kernel's declared bytes_moved equals the KV bytes the
pool actually hands it, and a checkpoint's mixed weight faces reach the tick
table per linear.

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

import os

import pytest
import torch

from tilerl.config import qwen38_27b, tiny
from tilerl.kernel_cost import TickShape, _paged_attention_decode, tick_rows, tick_totals
from tilerl.kv_cache import PagedKvPool
from tilerl.model import checkpoint_weight_faces, param_specs
from tilerl.precision import bf16, fp8_block_dev, kv_format, nbytes, nvfp4, nvfp4_dev


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


def _mixed_tiny_checkpoint(cfg, tmp_path):
    """Write a tiny checkpoint whose served linears carry all four faces, using
    real HF tensor names, and return the canonical linear param keys it contains."""
    from safetensors.torch import save_file

    specs = param_specs(cfg)

    def t(name_key, hf_name, face, n=None, k=None):
        n, k = n or specs[name_key][0], k or specs[name_key][1]
        base = f"model.layers.{hf_name}"
        if face == "fp4":  # official NVFP4: uint8 nibbles + scale + scale_2
            tensors[base + ".weight"] = torch.zeros((n, k), dtype=torch.uint8)
            tensors[base + ".weight_scale"] = torch.zeros((n, k // 16), dtype=torch.uint8)
            tensors[base + ".weight_scale_2"] = torch.zeros(1, dtype=torch.float32)
        elif face == "fp8blk":
            tensors[base + ".weight"] = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
            tensors[base + ".weight_scale_inv"] = torch.zeros((1, 1), dtype=torch.float32)
        elif face == "fp8row":
            tensors[base + ".weight"] = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
            tensors[base + ".weight_scale"] = torch.zeros((n,), dtype=torch.float32)
        else:
            tensors[base + ".weight"] = torch.zeros((n, k), dtype=torch.bfloat16)

    tensors = {}
    # layer 0 is full-attn, layer 1 is GDN; every face appears at least once.
    t("layers.0.q_proj", "0.self_attn.q_proj", "fp4")
    t("layers.0.k_proj", "0.self_attn.k_proj", "fp8blk")
    t("layers.0.v_proj", "0.self_attn.v_proj", "fp8row")
    t("layers.0.o_proj", "0.self_attn.o_proj", "bf16")
    t("layers.1.in_proj_qkv", "1.linear_attn.in_proj_qkv", "bf16")
    t("layers.1.in_proj_z", "1.linear_attn.in_proj_z", "fp4")
    t("layers.1.in_proj_b", "1.linear_attn.in_proj_b", "fp8blk")
    t("layers.1.in_proj_a", "1.linear_attn.in_proj_a", "fp8row")
    t("layers.1.out_proj", "1.linear_attn.out_proj", "bf16")
    for i in range(cfg.num_layers):
        t(f"layers.{i}.gate_proj", f"{i}.mlp.gate_proj", "fp4")
        t(f"layers.{i}.up_proj", f"{i}.mlp.up_proj", "fp8blk")
        t(f"layers.{i}.down_proj", f"{i}.mlp.down_proj", "fp8row")
    # non-linear served tensors: priced nowhere in the tick, must not break mapping
    tensors["model.embed_tokens.weight"] = torch.zeros(specs["embed_tokens"], dtype=torch.bfloat16)
    tensors["model.layers.0.input_layernorm.weight"] = torch.zeros(
        specs["layers.0.input_norm"], dtype=torch.bfloat16)
    tensors["model.layers.1.linear_attn.conv1d.weight"] = torch.zeros(
        specs["layers.1.conv1d"], dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    return {
        "layers.0.q_proj", "layers.0.k_proj", "layers.0.v_proj", "layers.0.o_proj",
        "layers.1.in_proj_qkv", "layers.1.in_proj_z", "layers.1.in_proj_b",
        "layers.1.in_proj_a", "layers.1.out_proj",
        *(f"layers.{i}.{m}" for i in range(cfg.num_layers)
          for m in ("gate_proj", "up_proj", "down_proj")),
    }


def test_checkpoint_faces_price_each_linear_at_its_own_device_face(tmp_path):
    cfg = tiny()
    linear_keys = _mixed_tiny_checkpoint(cfg, tmp_path)
    faces = checkpoint_weight_faces(cfg, tmp_path)

    # every served linear key is mapped; embed (excluded from the tick stream) and
    # 1-D norms map or not, but never as a priced linear row.
    assert linear_keys <= set(faces)
    seen = {r["face"] for r in tick_rows(
        cfg, TickShape(b=1, s=64, kv=kv_format(cfg.head_dim), weight=nvfp4, faces=faces))}
    assert {nvfp4_dev, fp8_block_dev} <= seen  # mixed population reaches the table

    b = 1
    rows = tick_rows(cfg, TickShape(b=b, s=64, kv=kv_format(cfg.head_dim),
                                   weight=nvfp4, faces=faces))
    # weight-only bytes per linear row, activations stripped with the same _gemv terms
    got = 0
    for r in rows:
        if r["face"] is None:
            continue  # fused kernels are not weight rows
        out, inn = (int(x) for x in r["shape"].split("x"))
        weight_only = r["bytes"] - 2 * b * (out + inn)
        assert weight_only == nbytes(r["face"], (out, inn))
        got += weight_only * r["count"]
    # independent total: each enumerated linear priced once at its checkpoint face
    # (faces may also carry embed_tokens/conv1d, which the tick table does not stream)
    want = sum(nbytes(fmt, shape) for key, (shape, fmt) in faces.items() if key in linear_keys)
    assert got == want
    # and the mixed table is strictly heavier than the all-nvfp4 pricing it replaces
    all4 = tick_rows(cfg, TickShape(b=b, s=64, kv=kv_format(cfg.head_dim), weight=nvfp4))
    old = sum((r["bytes"] - 2 * b * sum(int(x) for x in r["shape"].split("x"))) * r["count"]
              for r in all4 if r["name"] not in
              {"paged_attention_decode", "gdn_decode_fused", "rmsnorm", "silu_mul"})
    assert got > old


def test_bf16_linear_is_reported_as_the_face_load_hf_serves_under_fp4(tmp_path):
    """cfg.fp4 packs bf16-shipped fp4_param_keys linears at load, so the SERVED
    face classification must report nvfp4_dev, not the bf16 disk face."""
    from dataclasses import replace

    from safetensors.torch import save_file

    from tilerl.precision import fp8_dev

    cfg = replace(tiny(), fp4=True)
    specs = param_specs(cfg)
    n, k = specs["layers.0.o_proj"]  # bf16 on disk in every config, packable at load
    tensors = {"model.layers.0.self_attn.o_proj.weight": torch.zeros((n, k), dtype=torch.bfloat16)}
    n2, k2 = specs["layers.1.in_proj_b"]
    tensors["model.layers.1.linear_attn.in_proj_b.weight"] = torch.zeros(
        (n2, k2), dtype=torch.float8_e4m3fn)
    tensors["model.layers.1.linear_attn.in_proj_b.weight_scale"] = torch.zeros(
        (n2,), dtype=torch.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    faces = checkpoint_weight_faces(cfg, tmp_path)
    assert faces["layers.0.o_proj"][1] == nvfp4_dev  # packed by load_hf
    assert faces["layers.1.in_proj_b"][1] == fp8_dev  # explicit fp8 is untouched


@pytest.mark.skipif(not os.environ.get("TILERL_27B_CKPT"),
                    reason="set TILERL_27B_CKPT to a 27B checkpoint dir")
def test_27b_checkpoint_faces_cover_every_linear_and_mix_formats():
    from tilerl.precision import fp8_dev

    cfg = qwen38_27b()
    faces = checkpoint_weight_faces(cfg, os.environ["TILERL_27B_CKPT"])
    specs = param_specs(cfg)
    linears = {k for k, shp in specs.items()
               if len(shp) == 2 and k != "embed_tokens" and not k.endswith("conv1d")}
    assert linears <= set(faces), f"unpriced linears: {sorted(linears - set(faces))[:5]}"
    face_names = {fmt for _, fmt in faces.values()}
    assert nvfp4_dev in face_names  # the 27B mixes populations
    assert fp8_block_dev in face_names or fp8_dev in face_names
    for b in (1, 8):
        tick = TickShape(b=b, s=4096, kv=kv_format(cfg.head_dim), weight=nvfp4, faces=faces)
        print(f"B={b} {tick_totals(cfg, tick)}")
