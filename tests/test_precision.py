"""The precision table is total over its roles, and the call sites read it."""

import torch

from tilerl import precision


def test_policy_is_total_and_device_aware():
    for r in precision.roles():
        assert isinstance(precision.dtype(r, "cpu"), torch.dtype)
    assert precision.dtype("recurrent_state", "cuda") == torch.float32
    assert precision.dtype("recurrent_state", "cpu") == torch.bfloat16
    assert precision.dtype("optimizer_state") == torch.float32


def test_nbytes_derived_pool_equals_allocated_storage_bf16_and_fp8():
    """nbytes over the whole pool shape equals every tensor torch allocated, to the byte.

    Includes num_blocks (5f's kernel_cost gate prices only a gathered span, so it cannot
    catch a wrong block factor). Measured by summing storage of k_pool/v_pool and the two
    scale planes; bf16 (no scale) and fp8 (per-head_dim f32 scales) both checked.
    """
    from tilerl.kv_cache import PagedKvPool
    from tilerl.precision import kv_format, nbytes

    blocks, layers, heads, head_dim = 7, 3, 2, 16
    for label, kw, store in (
        ("bf16", {}, torch.bfloat16),
        ("fp8", {"kv_fp8": torch.float8_e4m3fn}, torch.float8_e4m3fn),
    ):
        pool = PagedKvPool(blocks, heads, head_dim, num_layers=layers, device="cpu", **kw)
        shape = (2 * layers, blocks, heads, 16, head_dim)
        fmt = (
            kv_format(head_dim) if pool.kv_fp8 is not None else precision.Format(store.itemsize * 8)
        )
        measured = sum(
            t.numel() * t.element_size()
            for t in (pool.k_pool, pool.v_pool, pool.k_scale, pool.v_scale)
            if t is not None
        )
        assert nbytes(fmt, shape) == measured, label
        assert pool.bytes_per_token * 16 * blocks == measured, label


def test_nbytes_nvfp4_matches_checkpoint_loader_packing():
    """nvfp4 formula == the ModelOpt tensors the loader consumes.

    No CPU path PRODUCES ModelOpt nvfp4 (only the external checkpoint does; the local
    pack_fp4 makes a different e2m1/block-32 layout), so this pins the loader layout
    (model.py weight_packed / weight_scale / weight_global_scale; _native_fp4 reshape(1),
    renorm reduces dim=1 -> one e4m3 scale per 16 along K, one f32 per tensor).
    """
    from tilerl.precision import nbytes, nvfp4

    n, k = 8, 64
    packed = torch.zeros((n, k // 2), dtype=torch.uint8)  # nibbles, 2/byte
    block_scale = torch.zeros((n, k // 16), dtype=torch.float8_e4m3fn)
    global_scale = torch.zeros(1, dtype=torch.float32)
    measured = sum(t.numel() * t.element_size() for t in (packed, block_scale, global_scale))
    assert nbytes(nvfp4, (n, k)) == measured
    assert nbytes(nvfp4, (n, k)) == 256 + 32 + 4  # nibbles + e4m3 blocks + one f32


def test_device_faces_equal_the_served_tensor_storage():
    """nvfp4_dev / fp8_dev price exactly what the loader puts on the card.

    nvfp4_dev: nibbles + f32 scale per block + one f32 per output row. The block is
    whatever the served scale shape says (32 from build_random's pack_fp4, 16 from a
    ModelOpt checkpoint); both checked. fp8_dev: e4m3 weight + the [N/128,K/128] f32
    grid + one f32 per row.
    """
    from tilerl_kernels.reference import pack_fp4, renorm_fp4_scale

    from tilerl.precision import fp8_block_dev, fp8_dev, nbytes, nvfp4_dev

    n, k = 64, 128
    w = torch.randn(n, k)
    for blk in (16, 32):
        wq, scale = pack_fp4(w, block=blk)
        scale, osc = renorm_fp4_scale(scale)  # served device tensors, both f32
        fmt = precision.Format(4, ((blk, "f32"), ((None,), "f32")))
        measured = wq.numel() + scale.numel() * 4 + osc.numel() * 4  # wq is uint8
        assert nbytes(fmt, (n, k)) == measured
    # The named constant is the block-16 ModelOpt face.
    assert nbytes(nvfp4_dev, (n, k)) == n * k // 2 + n * (k // 16) * 4 + n * 4

    # fp8 block-only face (weight_scale_inv): e4m3 weight + f32 block grid, NO row scale.
    w8 = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
    grid = torch.zeros((-(-n // 128), -(-k // 128)), dtype=torch.float32)
    assert nbytes(fp8_block_dev, (n, k)) == sum(
        t.numel() * t.element_size() for t in (w8, grid))
    # fp8 face with a resident per-row scale (plain .weight_scale branch).
    row = torch.zeros(n, dtype=torch.float32)
    assert nbytes(fp8_dev, (n, k)) == sum(
        t.numel() * t.element_size() for t in (w8, grid, row))


def test_weight_specs_classifies_a_checkpoint_header_without_weight_bytes():
    """The three loader branches and activation-quant sidecars, one row per base weight."""
    from tilerl.precision import weight_specs

    header = {
        "a.weight_packed": {"shape": (17408, 2560), "dtype": "U8"},  # nvfp4 logical 17408x5120
        "a.weight_scale": {"shape": (17408, 320), "dtype": "F8_E4M3FN"},
        "a.weight_global_scale": {"shape": (1,), "dtype": "F32"},
        "b.weight": {"shape": (48, 5120), "dtype": "F8_E4M3FN"},  # weight_scale_inv: grid only
        "b.weight_scale_inv": {"shape": (1, 40), "dtype": "F32"},
        "c.weight": {"shape": (64, 5120), "dtype": "F8_E4M3FN"},  # plain scale: grid + row
        "c.weight_scale": {"shape": (64,), "dtype": "F32"},
        "embed.weight": {"shape": (248320, 5120), "dtype": "BF16"},
        "norm.weight": {"shape": (5120,), "dtype": "BF16"},
        # activation quantization: never a priced resident-weight row
        "c.input_scale": {"shape": (1,), "dtype": "F32"},
        "c.input_global_scale": {"shape": (1,), "dtype": "F32"},
    }
    rows = {name: (shape, fmt) for name, shape, fmt in weight_specs(header)}
    assert set(rows) == {"a.weight_packed", "b.weight", "c.weight",
                        "embed.weight", "norm.weight"}
    assert rows["a.weight_packed"][0] == (17408, 5120)
    assert rows["a.weight_packed"][1] == precision.nvfp4_dev
    assert rows["b.weight"][1] == precision.fp8_block_dev
    assert rows["c.weight"][1] == precision.fp8_dev
    assert rows["embed.weight"][1].bits == 16


def test_checkpoint_weight_specs_round_trips_a_real_mixed_safetensors(tmp_path):
    """Write a mixed fp4/fp8-block/fp8-row/bf16 checkpoint and read it back header-only."""
    from safetensors.torch import save_file

    from tilerl.precision import (
        checkpoint_weight_specs,
        fp8_block_dev,
        fp8_dev,
        nbytes,
        nvfp4_dev,
    )

    N, K = 128, 64
    tensors = {
        # nvfp4 (ModelOpt naming): packed nibbles + e4m3 scale + global
        "m.weight_packed": torch.zeros((N, K // 2), dtype=torch.uint8),
        "m.weight_scale": torch.zeros((N, K // 16), dtype=torch.float8_e4m3fn),
        "m.weight_global_scale": torch.zeros(1, dtype=torch.float32),
        # fp8 block-only: w8 + grid inv, no row
        "b.weight": torch.zeros((N, K), dtype=torch.float8_e4m3fn),
        "b.weight_scale_inv": torch.zeros((1, 1), dtype=torch.float32),
        # fp8 with a plain per-channel scale: w8 + row (grid is ones at load)
        "c.weight": torch.zeros((N, K), dtype=torch.float8_e4m3fn),
        "c.weight_scale": torch.zeros((N,), dtype=torch.float32),
        "c.input_scale": torch.zeros(1, dtype=torch.float32),  # must be excluded
        "embed.weight": torch.zeros((100, K), dtype=torch.bfloat16),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    rows = {name: (shape, fmt) for name, shape, fmt in checkpoint_weight_specs(tmp_path)}
    assert set(rows) == {"m.weight_packed", "b.weight", "c.weight", "embed.weight"}
    assert rows["m.weight_packed"] == ((N, K), nvfp4_dev)
    assert rows["b.weight"] == ((N, K), fp8_block_dev)
    assert rows["c.weight"] == ((N, K), fp8_dev)
    # Derived total is the exact sum of the three device faces plus bf16 embed.
    expected = (nbytes(nvfp4_dev, (N, K)) + nbytes(fp8_block_dev, (N, K))
                + nbytes(fp8_dev, (N, K)) + 100 * K * 2)
    single_total = sum(nbytes(fmt, shape) for shape, fmt in rows.values())
    assert single_total == expected

    # Same population split across TWO shards with an index.json weight_map -- the only
    # layout the sharded 27B takes. Classification and byte total must be identical.
    import json

    shard1 = {k: tensors[k] for k in
              ("m.weight_packed", "m.weight_scale", "m.weight_global_scale", "b.weight",
               "b.weight_scale_inv")}
    shard2 = {k: tensors[k] for k in
              ("c.weight", "c.weight_scale", "c.input_scale", "embed.weight")}
    save_file(shard1, str(tmp_path / "model-00001.safetensors"))
    save_file(shard2, str(tmp_path / "model-00002.safetensors"))
    (tmp_path / "model.safetensors").unlink()
    weight_map = {name: "model-00001.safetensors" for name in shard1}
    weight_map.update({name: "model-00002.safetensors" for name in shard2})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    srows = {name: (shape, fmt) for name, shape, fmt in checkpoint_weight_specs(tmp_path)}
    assert srows == rows
    assert sum(nbytes(fmt, shape) for shape, fmt in srows.values()) == single_total


def test_27b_resident_weight_bytes_match_the_measured_24_44gb(tmp_path):
    """The checkpoint-derived resident total equals the 24.44 GB measured on H20.

    Pending-remote: needs the Qwen3.8-27B-NVFP4 checkpoint's safetensors headers (no
    weight bytes are read). Point TILERL_27B_CKPT at the dir to run it. The oracle is
    errors/2026-09-03-fp4-param-keys-is-not-the-fp4-tensors.md: 24.44 GB resident from a
    MIXED population (264 nvfp4 + 233 fp8), which config alone cannot reproduce.
    """
    import os

    ckpt = os.environ.get("TILERL_27B_CKPT")
    if not ckpt:
        import pytest

        pytest.skip("set TILERL_27B_CKPT to the 27B NVFP4 dir; headers only, no weights")
    from tilerl.precision import checkpoint_weight_specs, nbytes

    rows = checkpoint_weight_specs(ckpt)
    by_face: dict[str, int] = {}
    for _, shape, fmt in rows:
        by_face[fmt] = by_face.get(fmt, 0) + nbytes(fmt, shape)
    total = sum(by_face.values())
    # The errors entry records 24.44 GB = 22.76 GiB (decimal GB). Assert to the measured
    # 0.01 GB: header-derived bytes must equal what memory_allocated read on H20.
    assert abs(total - int(24.44e9)) < int(0.01e9), f"resident {total / 1e9:.3f} GB != 24.44 GB"


def test_iso_frames_follow_the_policy():
    from tilerl.iso import ISO

    u, _, _ = ISO().frames(torch.randn(6, 4, dtype=torch.bfloat16))
    assert u.dtype == precision.dtype("frame")


def test_on_policy_guard_refuses_cached_engines():
    """Both arms, because only one of them was reachable from this test.

    The guard refuses a captured decode graph OR a live prefix store. Measured by
    mutation: with the `_decode_graph_on` half deleted this test still passed — on cpu
    `_graph_on` returns False, so the engine it builds trips the prefix arm and the
    graph arm never fires. That arm is the one that matters on the pod, where
    `_graph_on` defaults to True for CUDA and a `grpo_loop` call that forgot
    `decode_graph=False` lands on exactly the untested half.

    Deleting either half now fails: prefix arm CAUGHT before, graph arm CAUGHT after.
    """
    import pytest

    from tilerl.cli import _build_model
    from tilerl.engine import build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.testing import RefBackend
    from tilerl.train import grpo_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    run = lambda e: list(grpo_loop(e, model, [[1, 2, 3]], lambda p, c: 0.0, 1, RefBackend()))

    # prefix cache on, graph off
    cached = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=4)
    with pytest.raises(ValueError, match="on-policy"):
        run(cached)

    # graph on, prefix off — decode_graph=True is honoured on cpu, so this is testable
    # here and not a CUDA-only path.
    graphed = build_engine(
        cfg,
        model,
        RefBackend(),
        num_blocks=32,
        num_slots=4,
        decode_graph=True,
        prefix_store=NoPrefixStore(),
    )
    assert graphed._decode_graph_on is True, "decode_graph=True was not honoured"
    with pytest.raises(ValueError, match="on-policy"):
        run(graphed)


def test_opd_refuses_a_cached_engine_with_no_adapters_too():
    """The guard must be unconditional, not conditional on `trainable`.

    `opd_loop` used to skip it when `trainable is None`, reasoning that "a frozen teacher
    with no adapters cannot go stale". The teacher is not frozen: with no `trainable`,
    `train_step` updates `model.params` (train.py:81) and the engine samples from that same
    object (engine.py:320). Measured on tiny, 27 of the teacher's parameters changed within
    two steps -- the exempt path was the one that went stale fastest.

    Asserts the no-adapter call is refused, because the tree has no caller passing None
    today and "nobody passes it" is a fact that a future caller silently reverses. The
    other arm (with adapters) is covered above.
    """
    import pytest

    from tilerl.cli import _build_model
    from tilerl.engine import build_engine
    from tilerl.testing import RefBackend
    from tilerl.train import opd_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    # Prefix store on: the same cached-engine condition the adapter arm is refused for.
    cached = build_engine(cfg, model, RefBackend(), num_blocks=32, num_slots=4)
    with pytest.raises(ValueError, match="on-policy"):
        opd_loop(cached, model, [[1, 2, 3]], 1, RefBackend(), trainable=None)


def test_the_cached_cast_keeps_its_address_across_an_optimizer_step():
    """`_const_f32`'s buffer must survive a `_version` bump at the same address.

    A captured CUDA graph bakes the ADDRESS of every tensor its kernels read.
    `AdamW.step_one` ends `p.copy_(...)`, which is in place — the parameter's
    address never moves, so on that count a replay would read the new weights.
    The cast was the thing that moved: `copy_` bumps `t._version`, the cache
    missed, and `self._dev(t, dtype)` allocated the converted copy somewhere new.
    That is what `engine.py`'s `# ponytail: no recapture after training — the
    graph bakes the f32 embed cast` names, and it is 27 call sites, not one.

    Four arms, because the refill is guarded and an unguarded arm is the one
    production takes: the address holds, the values are those of a fresh cast,
    and a `pad_to` or `dtype` change must still allocate rather than write into a
    buffer of the wrong shape.
    """
    from tilerl_kernels.backend import get_backend

    b = get_backend()
    p = torch.randn(8, 4, dtype=torch.bfloat16, device=b.device)

    first = b._const_f32(p)
    addr = first.data_ptr()
    assert first.dtype == torch.float32, "nothing was cast; pick a dtype that converts"

    p.copy_(torch.randn(8, 4, dtype=torch.bfloat16, device=b.device))  # the optimizer
    second = b._const_f32(p)
    assert second.data_ptr() == addr, (
        "the cached cast moved across an optimizer step: a captured graph baked "
        f"{addr:#x} and would replay stale bytes"
    )
    # Address stability is worthless if the buffer kept the OLD values.
    assert torch.equal(second, p.to(torch.float32)), "the refill did not land"

    # Guard arm 1: a different pad_to is a different cache key, so it must allocate
    # rather than write into the unpadded buffer. 1-D because that is what the call
    # sites pass -- `pad_to` compares shape[0] but F.pad fills the LAST dim, so the
    # two agree only for a vector (every real caller passes a per-row scale).
    v = torch.randn(8, dtype=torch.bfloat16, device=b.device)
    plain = b._const_f32(v)
    padded = b._const_f32(v, pad_to=12)
    assert padded.shape[0] == 12 and padded.data_ptr() != plain.data_ptr()
    assert torch.equal(padded[:8], v.to(torch.float32))
    assert torch.equal(padded[8:], torch.zeros_like(padded[8:]))

    # Guard arm 2: a different dtype likewise, and the f32 entry keeps its own
    # address afterwards -- the refill must not be confused by a neighbouring key.
    p.copy_(torch.randn(8, 4, dtype=torch.bfloat16, device=b.device))
    half = b._const_f32(p, dtype=torch.float16)
    assert half.dtype == torch.float16 and half.data_ptr() != addr
    assert b._const_f32(p).data_ptr() == addr, "the f32 entry lost its address"


def test_kernel_io_is_keyed_on_arch_not_on_being_cuda():
    """No dtype decision in backend.py may read `target.startswith("cuda")`.

    sm70 has no bf16 load, so its cells are compiled f32 and `Backend.io` says so
    per-arch. Three call sites re-derived it as `bf16 if cuda else f32`, which is
    right on sm90 and wrong on every other CUDA arch: gdn_prep then got f16 tensors
    against an f32 signature and died with "input Q dtype mismatch, expected
    float32" -- mid-run, after the weights were resident, pointing at the kernel
    rather than at the marshalling.

    Structural because there is no sm70 in CI: the arch that breaks cannot run the
    numeric gate, so the gate has to read the source. Negative control: restore any
    of the three `io = torch.bfloat16 if self.target.startswith("cuda")` lines and
    this fails naming its line number.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent
        / "packages"
        / "tilerl-kernels"
        / "src"
        / "tilerl_kernels"
        / "backend.py"
    ).read_text()
    bad = []
    for node in ast.walk(ast.parse(src)):
        # An `x = <bf16/f16> if <...cuda...> else <...>` anywhere: the dtype is the
        # tell, since target.startswith("cuda") is legitimate for control flow.
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp)):
            continue
        whole, cond = ast.dump(node.value), ast.dump(node.value.test)
        if "cuda" in cond and ("bfloat16" in whole or "float16" in whole):
            bad.append(node.lineno)
    assert not bad, (
        f"backend.py:{bad}: a dtype chosen by `target.startswith('cuda')`. Use "
        f"self.io / self.gemv_io / self.scale_io, which are keyed on arch -- sm70 "
        f"is CUDA and takes f32."
    )


if __name__ == "__main__":  # runnable check
    test_policy_is_total_and_device_aware()
    test_kernel_io_is_keyed_on_arch_not_on_being_cuda()
    print("precision: policy OK")
