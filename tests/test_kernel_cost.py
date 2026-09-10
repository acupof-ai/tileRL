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


# --- checkpoint device faces ------------------------------------------------
def _mixed_tiny_checkpoint(cfg, tmp_path):
    """Write a tiny checkpoint whose served linears carry all four faces, using
    real HF tensor names, and return the canonical linear param keys it contains."""
    from safetensors.torch import save_file

    from tilerl.model import param_specs

    specs = param_specs(cfg)

    def t(name_key, hf_name, face):
        n, k = specs[name_key]
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


def _weight_dims(shape: str) -> tuple[int, int]:
    # decode shapes are "NxK"; prefill shapes are "M<rows> NxK" — the matrix is last.
    n, k = shape.split()[-1].split("x")
    return int(n), int(k)


def test_checkpoint_faces_price_each_linear_at_its_own_device_face(tmp_path):
    from tilerl.model import checkpoint_weight_faces
    from tilerl.precision import fp8_block_dev, nvfp4, nvfp4_dev

    cfg = tiny()
    linear_keys = _mixed_tiny_checkpoint(cfg, tmp_path)
    faces = checkpoint_weight_faces(cfg, tmp_path)

    # every served linear key is mapped; the mixed population reaches both tables.
    assert linear_keys <= set(faces)
    t0 = TickShape(b=1, s=64, kv=kv_format(cfg.head_dim), weight=nvfp4, faces=faces)
    assert {nvfp4_dev, fp8_block_dev} <= {r["face"] for r in tick_rows(cfg, t0)}
    assert {nvfp4_dev, fp8_block_dev} <= {r["face"] for r in prefill_rows(cfg, t0)}

    # weight-only bytes per row equal nbytes(face, shape); the total equals an
    # independent per-linear sum (decode and prefill share the face plumbing;
    # a prefill GEMM streams its weight once, independent of the M token rows).
    want = sum(nbytes(fmt, shape)
               for key, (shape, fmt) in faces.items() if key in linear_keys)
    for rows in (tick_rows(cfg, t0), prefill_rows(cfg, t0)):
        got = 0
        for r in rows:
            if r["face"] is None:
                continue  # fused kernels are not weight rows
            got += nbytes(r["face"], _weight_dims(r["shape"])) * r["count"]
        assert got == want

    # the mixed table is strictly heavier than the all-nvfp4 pricing it replaces
    all4 = tick_rows(cfg, TickShape(b=1, s=64, kv=kv_format(cfg.head_dim), weight=nvfp4))
    old = sum((r["bytes"] - 2 * sum(_weight_dims(r["shape"]))) * r["count"]
              for r in all4 if r["face"] is not None)
    new = sum((r["bytes"] - 2 * sum(_weight_dims(r["shape"]))) * r["count"]
              for r in tick_rows(cfg, t0) if r["face"] is not None)
    assert new > old


def test_bf16_linear_is_reported_as_the_face_load_hf_serves_under_fp4(tmp_path):
    """cfg.fp4 packs bf16-shipped fp4_param_keys linears at load with pack_fp4's
    block 32, so the served face must price the ACTUAL packed tensors (nibbles +
    f32/32 + f32/row), not the disk bf16 and not the block-16 nvfp4_dev face.
    The assertion is bytes against pack_fp4+renorm storage, so a block mismatch
    fails it (block 16 would price 2x the real scale plane)."""
    from dataclasses import replace

    from safetensors.torch import save_file
    from tilerl_kernels.reference import pack_fp4, renorm_fp4_scale

    from tilerl.model import checkpoint_weight_faces, param_specs
    from tilerl.precision import fp8_dev, nbytes, nvfp4_dev, nvfp4_dev_b32

    cfg = replace(tiny(), fp4=True)
    specs = param_specs(cfg)
    n, k = specs["layers.0.o_proj"]
    tensors = {"model.layers.0.self_attn.o_proj.weight": torch.zeros((n, k), dtype=torch.bfloat16)}
    # 3-D disk conv1d [C,1,Kc]: load_hf reshapes it 2-D; the face map must too.
    ck, kc = specs["layers.1.conv1d"]
    tensors["model.layers.1.linear_attn.conv1d.weight"] = torch.zeros(
        (ck, 1, kc), dtype=torch.bfloat16)
    n2, k2 = specs["layers.1.in_proj_b"]
    tensors["model.layers.1.linear_attn.in_proj_b.weight"] = torch.zeros(
        (n2, k2), dtype=torch.float8_e4m3fn)
    tensors["model.layers.1.linear_attn.in_proj_b.weight_scale"] = torch.zeros(
        (n2,), dtype=torch.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    faces = checkpoint_weight_faces(cfg, tmp_path)
    assert faces["layers.0.o_proj"][1] == nvfp4_dev_b32  # pack_fp4 default block 32
    assert faces["layers.1.in_proj_b"][1] == fp8_dev  # explicit fp8 is untouched
    assert faces["layers.1.conv1d"][0] == (ck, kc)  # [C,1,Kc] flattened as load_hf does

    # Independent served storage for o_proj: pack_fp4(block=32) widens to f32,
    # renorm splits out the per-row epilogue — exactly nvfp4_dev_b32's planes.
    wq, scale = pack_fp4(torch.zeros((n, k), dtype=torch.bfloat16))
    scale, oscale = renorm_fp4_scale(scale)
    served = wq.numel() + scale.numel() * 4 + oscale.numel() * 4
    assert nbytes(faces["layers.0.o_proj"][1], (n, k)) == served
    assert served != nbytes(nvfp4_dev, (n, k))  # the block-16 face would be wrong


def test_27b_served_weight_faces_equal_load_hf_resident_exact():
    """Pending-remote byte-exact oracle: the served-face map is exhaustive over
    param_specs and its nbytes sum equals, to the integer, BOTH the recorded
    load_hf resident total AND a live load_hf model's tensor storage.

        TILERL_27B_CKPT=/work/Qwen3.8-27B-NVFP4 pytest tests/test_kernel_cost.py -k 27b

    Recorded by running load_hf on the pod (1845 tensors): 24,436,981,888 B.
    """
    import os

    import pytest

    ckpt = os.environ.get("TILERL_27B_CKPT")
    if not ckpt:
        pytest.skip("set TILERL_27B_CKPT to a 27B checkpoint dir")
    from tilerl.config import qwen38_27b
    from tilerl.model import checkpoint_weight_faces, param_specs
    from tilerl.precision import fp8_block_dev, fp8_dev, kv_format, nbytes, nvfp4, nvfp4_dev

    cfg = qwen38_27b()
    faces = checkpoint_weight_faces(cfg, ckpt)
    specs = param_specs(cfg)
    assert set(faces) == set(specs)  # exhaustive: one served face per param key
    total = sum(nbytes(fmt, shape) for shape, fmt in faces.values())
    assert total == 24_436_981_888, f"served faces {total} != load_hf resident 24,436,981,888"

    # Independent second derivation on a live load (real weight bytes this once):
    # its resident storage must be the same integer. Header-only is the shipping
    # path; this is the equality the headers-only formula rests on.
    from tilerl.model import load_hf

    live = sum(t.numel() * t.element_size() for t in load_hf(cfg, ckpt).params.values())
    assert live == total

    face_names = {fmt for _, fmt in faces.values()}
    assert nvfp4_dev in face_names  # on-disk block-16 NVFP4
    assert fp8_block_dev in face_names or fp8_dev in face_names
    for b in (1, 8):
        tick = TickShape(b=b, s=4096, kv=kv_format(cfg.head_dim), weight=nvfp4, faces=faces)
        print(f"B={b} {tick_totals(cfg, tick)}")
