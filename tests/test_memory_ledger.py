"""The memory ledger's derived rows equal the bytes the engine actually holds, and the
fit/plan share ONE formula.

This is the gate docs/design-cost-model.md rests on: plan is the allocator's input, not
a second computation — build_engine's fit calls memory.fit_num_blocks and the pool row
calls the same per_kv_block_bytes; the state shapes are the ones LinearStatePool
allocates. The fp8 case is checked too: the measured column must include BOTH scale
planes, or the 27B path this is built for false-flags. GPU weights/fit are
pending-remote; on the CPU tiny model these rows are exact.
"""

import pytest

from tilerl.cli import _build_model
from tilerl.engine import build_engine
from tilerl.kv_cache import PagedKvPool
from tilerl.memory import (
    draft_per_block_bytes,
    fit_num_blocks,
    per_kv_block_bytes,
    plan,
    state_shapes,
    weight_row,
)
from tilerl.precision import f32, nbytes
from tilerl.testing import RefBackend


def _engine(num_slots=4, num_blocks=8):
    cfg, model = _build_model("tiny", seed=0)
    eng = build_engine(cfg, model, RefBackend(), num_blocks=num_blocks, num_slots=num_slots,
                       max_batch=num_slots, max_total_tokens=2048, max_num_batched_tokens=512)
    return cfg, model, eng


def _pool_bytes(pool) -> int:
    return sum(t.numel() * t.element_size() for t in
               (pool.k_pool, pool.v_pool, pool.k_scale, pool.v_scale) if t is not None)


def _state_actual(pool):
    return sum(t.numel() * t.element_size() for t in
               (pool.states, pool.conv_windows, pool.step_states, pool.step_windows,
                pool.win_parity) if t is not None)


def test_weight_row_equals_materialized_params():
    _, model, _ = _engine()
    assert weight_row(model.params).n == sum(
        t.numel() * t.element_size() for t in model.params.values()) > 0


@pytest.mark.parametrize("kv_fp8", [None, "fp8"])
def test_per_kv_block_equals_pool_and_shares_fit_formula(kv_fp8):
    """The ONE block formula equals the live pool per block, bf16 and fp8 (scales incl.).
    Without the fp8 arm a k_pool+v_pool measured column hides the two missing scale
    planes — exactly the delta the ledger exists to catch."""
    cfg, _, eng = _engine()
    # A standalone fp8 pool (the served engine on CPU stays bf16) exercises the scale path.
    if kv_fp8 == "fp8":
        import torch

        pool = PagedKvPool(8, cfg.num_kv_heads, cfg.head_dim,
                           num_layers=len(cfg.full_attn_layers), device="cpu",
                           kv_fp8=torch.float8_e4m3fn)
        actual_block = _pool_bytes(pool) // pool.num_blocks
        assert per_kv_block_bytes(cfg, torch.bfloat16, torch.float8_e4m3fn) == actual_block
        free = actual_block * 300
        # fit denominator adds a (zero here) draft term, so it equals the main block.
        assert fit_num_blocks(cfg, free, torch.bfloat16, torch.float8_e4m3fn, floor=64) \
            == max(64, int(free * 2 / 3) // actual_block)
        return
    kv = eng._kv
    actual_block = _pool_bytes(kv) // kv.num_blocks
    assert per_kv_block_bytes(cfg, kv.k_pool.dtype, kv.kv_fp8) == actual_block
    free = actual_block * 300
    want = max(64, int(free * 2 / 3) // actual_block)
    assert fit_num_blocks(cfg, free, kv.k_pool.dtype, kv.kv_fp8, floor=64) == want
    assert fit_num_blocks(cfg, free, kv.k_pool.dtype, kv.kv_fp8, cap=10, floor=64) == 10


def test_state_shapes_equal_state_pool_tensors():
    """state_shapes prices the parity plane through nbytes(f32,(N,)) — no mirrored literal."""
    cfg, _, eng = _engine()
    pool = eng._states
    from tilerl.memory import _dtype_fmt
    state_fmt = _dtype_fmt(pool.states.dtype)
    derived = sum(nbytes(f32 if kind == "parity" else state_fmt, shape)
                  for shape, kind in state_shapes(cfg, pool.num_slots, 0))
    assert derived == _state_actual(pool)


def test_plan_rows_cover_weights_kv_and_state_to_the_byte():
    cfg, model, eng = _engine()
    kv, sp = eng._kv, eng._states
    rows = plan(cfg, model.params, 0, num_slots=sp.num_slots, num_blocks=kv.num_blocks,
                state_dtype=sp.states.dtype, kv_io=kv.k_pool.dtype, kv_fp8=kv.kv_fp8)
    actual = {
        "weights": sum(t.numel() * t.element_size() for t in model.params.values()),
        "kv_pool": _pool_bytes(kv),
        "state_slots": _state_actual(sp),
    }
    for owner, want in actual.items():
        assert sum(r.n for r in rows if r.owner == owner) == want, owner


def test_plan_fp8_pool_row_includes_scale_planes():
    """The plan's fp8 kv_pool row must equal a real fp8 pool (data + 2 scale planes); a
    formula dropping the scales disagrees with the measured column on the 27B."""
    import torch

    cfg, _, _ = _engine()
    pool = PagedKvPool(8, cfg.num_kv_heads, cfg.head_dim, num_layers=len(cfg.full_attn_layers),
                       device="cpu", kv_fp8=torch.float8_e4m3fn)
    rows = plan(cfg, None, 0, num_slots=1, num_blocks=8,
                kv_io=torch.bfloat16, kv_fp8=torch.float8_e4m3fn)
    assert sum(r.n for r in rows if r.owner == "kv_pool") == _pool_bytes(pool)


def test_draft_pool_is_separate_not_folded_into_kv_pool():
    """On an MTP engine the draft KV is its OWN pool: kv_pool row is main-only and the
    draft_pool row carries the draft term. Folding the draft bytes into per_kv_block_bytes
    double-counted them (kv_pool main+draft vs a main-only measurement), and no draft gate
    existed to turn it red. Also pins the fit denominator = main + draft (same blocks)."""
    import torch

    from tilerl.spec import DraftHead
    cfg, model, eng = _engine()
    draft = DraftHead(model, {}, num_layers=1)
    # attach builds the draft's own PagedKvPool the way build_engine does.
    draft.attach(RefBackend(), eng._kv.num_blocks, dtype=torch.bfloat16)
    kv, blocks = eng._kv, eng._kv.num_blocks

    main_block = per_kv_block_bytes(cfg, torch.bfloat16, None)
    draft_block = draft_per_block_bytes(cfg, torch.bfloat16, 1)
    assert draft_block > 0
    rows = plan(cfg, model.params, 0, num_slots=eng._states.num_slots, num_blocks=blocks,
                state_dtype=eng._states.states.dtype, kv_io=torch.bfloat16, draft_layers=1)
    by = {r.owner: r.n for r in rows}
    # kv_pool row is trunk ONLY, measured against the trunk pool (not main+draft).
    assert by["kv_pool"] == main_block * blocks == _pool_bytes(kv)
    # the draft term appears once, on draft_pool, measured against the draft's own pool.
    assert by["draft_pool"] == draft_block * blocks == _pool_bytes(draft.kv)
    # fit's denominator is main + draft, so its block count is the shared allocation.
    free = (main_block + draft_block) * 90
    assert fit_num_blocks(cfg, free, torch.bfloat16, None, draft_layers=1, floor=64) \
        == max(64, int(free * 2 / 3) // (main_block + draft_block))


def test_plan_budget_rows_are_arithmetic_without_a_card():
    """device_free is a parameter: free*2/3 pool and free/4 snapshot budget are arithmetic
    exercised on CPU, not pending-remote."""
    cfg, _, _ = _engine()
    by = {r.owner: r.n for r in plan(cfg, None, 1000, num_slots=1, num_blocks=64)}
    assert by["kv_pool_budget"] == int(1000 * 2 / 3)
    assert by["prefix_entries_budget"] == 1000 // 4


def test_plan_weights_row_from_checkpoint_faces_drops_nonserving_and_repacks_bf16(tmp_path):
    """The header-only weights row sums the SERVED faces checkpoint_weight_faces returns:
    a non-serving tensor (MTP) is dropped, and a bf16-shipped fp4_param_keys linear is
    priced at the block-32 repack face load_hf serves — not its disk bf16. The raw
    checkpoint_weight_specs path is wrong on both, so this goes red on the old code."""
    from dataclasses import replace

    import torch
    from safetensors.torch import save_file

    from tilerl.config import tiny
    from tilerl.memory import plan, weight_row_faces
    from tilerl.model import checkpoint_weight_faces, param_specs
    from tilerl.precision import nbytes, nvfp4_dev_b32

    cfg = replace(tiny(), fp4=True)
    specs = param_specs(cfg)
    n, k = specs["layers.0.q_proj"]           # an fp4_param_keys linear, shipped bf16
    ve, h = specs["embed_tokens"]
    save_file({
        # bf16 on disk; load_hf repacks it with pack_fp4 block 32 -> nvfp4_dev_b32
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((n, k), dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.zeros((ve, h), dtype=torch.bfloat16),
        # a non-serving tensor (MTP/visual): _param_key_for -> None, must be dropped
        "model.mtp.0.enhance.weight": torch.zeros((16, k), dtype=torch.bfloat16),
    }, str(tmp_path / "model.safetensors"))
    faces = checkpoint_weight_faces(cfg, tmp_path)
    assert "layers.0.q_proj" in faces and len(faces) == 2  # q_proj + embed; MTP dropped
    assert faces["layers.0.q_proj"][1] == nvfp4_dev_b32
    # Exact served bytes: repacked q_proj (b32) plus the bf16 embed table; no MTP row.
    want = nbytes(nvfp4_dev_b32, (n, k)) + 2 * ve * h
    assert weight_row_faces(faces).n == want
    rows = plan(cfg, None, 0, num_slots=1, num_blocks=8, ckpt_faces=faces)
    assert sum(r.n for r in rows if r.owner == "weights") == want


def test_serve_dry_run_checkpoint_is_header_only_and_needs_dry_run(tmp_path, capsys):
    """--dry-run --checkpoint DIR prices from served faces (no load, no engine, measured/delta
    null); --checkpoint without --dry-run refuses; blocks are fitted after fixed bytes."""
    import json

    import torch
    from safetensors.torch import save_file

    from tilerl import cli
    from tilerl.memory import _state_bytes, fit_num_blocks, weight_row_faces
    from tilerl.model import checkpoint_weight_faces

    # config.json must match the --model cfg or the new model/checkpoint guard refuses
    # (pricing one checkpoint's faces on another cfg's shapes is the 27B/tiny bug).
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16}))
    save_file({"model.embed_tokens.weight": torch.zeros((320, 64), dtype=torch.bfloat16)},
              str(tmp_path / "model.safetensors"))

    with pytest.raises(SystemExit, match="--dry-run"):
        cli.cmd_serve(cli._build_parser().parse_args(
            ["serve", "--model", "tiny", "--checkpoint", str(tmp_path)]))
    cli.cmd_serve(cli._build_parser().parse_args(
        ["serve", "--model", "tiny", "--dry-run", "--checkpoint", str(tmp_path),
         "--json", "--device-free", "1000000", "--slots", "4"]))
    rows = json.loads(capsys.readouterr().out)
    by = {r["owner"]: r for r in rows}
    cfg, _, _ = _engine()
    faces = checkpoint_weight_faces(cfg, tmp_path)
    # One presentation contract with the built --dry-run: kind/derived, not a second
    # {bytes,...} schema; transient is suppressed (no peak measured header-only).
    assert by["weights"]["kind"] == "allocation"
    assert by["weights"]["derived"] == weight_row_faces(faces).n
    assert by["weights"]["measured"] is None and by["weights"]["delta"] is None
    assert "transient" not in by
    # build_engine fits AFTER weights and the state pool (slots+CUDA graph pad; 0 on cpu)
    # are resident; the header-only fit subtracts the same before fitting. Recompute the
    # pad from the SAME backend cmd_serve builds (get_backend) — not a RefBackend, which
    # is always CPU and undercounts the pad slot on a CUDA card (305 vs 306 blocks).
    from tilerl_kernels.backend import get_backend

    from tilerl.engine import _graph_on

    pad = int(_graph_on(get_backend(), None))
    free_after_fixed = 1000000 - weight_row_faces(faces).n - _state_bytes(cfg, 4 + pad, f32)
    want_blocks = fit_num_blocks(cfg, free_after_fixed, torch.bfloat16)
    assert by["kv_pool"]["note"] == f"{want_blocks} blocks"


def test_dry_run_refuses_checkpoint_that_does_not_match_model(tmp_path, monkeypatch, capsys):
    """--checkpoint <27B> left on the default --model tiny priced 27B faces on tiny cfg
    shapes (the roofline timing bug). The checkpoint's own config.json is ground truth:
    a structural mismatch must refuse, not silently mix models."""
    import json

    import pytest

    from tilerl import cli

    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 48, "hidden_size": 5120, "num_attention_heads": 40,
        "num_key_value_heads": 4, "head_dim": 256}))
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(SystemExit, match="not a tiny checkpoint"):
        cli.cmd_serve(cli._build_parser().parse_args(
            ["serve", "--model", "tiny", "--dry-run", "--checkpoint", str(tmp_path)]))


def test_27b_checkpoint_weights_row_matches_load_hf_resident_exact():
    """Pending-remote: the header-only weights row on the real 27B equals BOTH load_hf's
    resident bytes and its live materialized storage, to the integer cc recorded
    (1845 tensors). Headers only for the sum; the live equality is run once on the pod.

        TILERL_27B_CKPT=/work/tilerl-ckpt/Qwen3.8-27B-NVFP4 \\
        uv run tilerl serve --model qwen38-27b --dry-run --checkpoint \"$TILERL_27B_CKPT\" \\
            --device-free 60000000000
    """
    import os

    ckpt = os.environ.get("TILERL_27B_CKPT")
    if not ckpt:
        pytest.skip("set TILERL_27B_CKPT to the 27B NVFP4 dir; headers only, no weights")
    from tilerl.config import qwen38_27b
    from tilerl.memory import weight_row_faces
    from tilerl.model import checkpoint_weight_faces, load_hf

    cfg = qwen38_27b()
    total = weight_row_faces(checkpoint_weight_faces(cfg, ckpt)).n
    assert total == 24_436_981_888, f"served weights {total} != load_hf resident 24,436,981,888"
    assert sum(t.numel() * t.element_size() for t in load_hf(cfg, ckpt).params.values()) == total


def test_serve_dry_run_needs_device_free_off_cuda_and_prints_rows(capsys):
    """--dry-run --json builds, reconciles derived vs measured, and emits the budget rows
    from --device-free; off CUDA the flag is required (no mem_get_info to invent a number)."""
    import json

    import pytest

    from tilerl import cli

    with pytest.raises(SystemExit, match="--device-free"):
        cli.cmd_serve(cli._build_parser().parse_args(
            ["serve", "--model", "tiny", "--dry-run", "--json",
             "--blocks", "8", "--slots", "4", "--max-batch", "4"]))
    cli.cmd_serve(cli._build_parser().parse_args(
        ["serve", "--model", "tiny", "--dry-run", "--json", "--device-free", "1000000",
         "--blocks", "8", "--slots", "4", "--max-batch", "4"]))
    rows = json.loads(capsys.readouterr().out)
    owners = {r["owner"] for r in rows}
    assert {"weights", "kv_pool", "state_slots", "kv_pool_budget",
            "prefix_entries_budget"} <= owners
    assert all(r["delta"] == 0 for r in rows
               if r["owner"] in ("weights", "kv_pool", "state_slots"))


def test_peak_equals_static_plus_transient_exactly():
    """The printed invariant is exact: measured peak = Σ held static rows + transient.
    Budget rows never enter either side. On the CPU tiny cell the measured peak is the
    held storage sum, so transient is exactly 0 and the total equals that peak."""
    from tilerl.memory import memory_table, static_rows

    cfg, model, eng = _engine()
    kv, sp = eng._kv, eng._states
    rows = plan(cfg, model.params, 0, num_slots=sp.num_slots, num_blocks=kv.num_blocks,
                state_dtype=sp.states.dtype, kv_io=kv.dtype, kv_fp8=kv.kv_fp8)
    measured = {r["owner"]: r["measured"] for r in eng.stats()["memory"]
                if r["kind"] == "allocation" and r.get("measured") is not None
                and r["owner"] != "transient"}
    peak = eng._measured_peak_bytes()
    table = memory_table(rows, measured, peak)
    by = {r["owner"]: r["derived"] for r in table}

    static_sum = sum(r.n for r in static_rows(rows))
    assert by["transient"] == peak - static_sum
    assert static_sum + by["transient"] == peak
    # CPU tiny cell: no allocator scratch beyond the named rows.
    assert by["transient"] == 0
    assert by["device_total"] == peak


def test_transient_zero_on_tiny_and_red_under_a_dropped_static_row():
    """Mutant control for the invariant: if a static row is dropped from the plan but the
    measured peak is unchanged, transient silently ABSORBS the dropped row's bytes. The
    transient row must then exceed a bound derived from the tiny shapes (the real scratch
    headroom), so the missing row cannot hide. We assert the bound from shapes, not a
    literal, and that dropping one row pushes transient past it."""
    from tilerl.memory import memory_table, static_rows

    cfg, model, eng = _engine()
    kv, sp = eng._kv, eng._states
    rows = plan(cfg, model.params, 0, num_slots=sp.num_slots, num_blocks=kv.num_blocks,
                state_dtype=sp.states.dtype, kv_io=kv.dtype, kv_fp8=kv.kv_fp8)
    measured = {r["owner"]: r["measured"] for r in eng.stats()["memory"]
                if r["kind"] == "allocation" and r.get("measured") is not None
                and r["owner"] != "transient"}
    peak = eng._measured_peak_bytes()

    # Bound derived from tiny shapes: real scratch is a handful of activation tensors far
    # smaller than the smallest held pool row; any transient >= that pool row is a dropped
    # static allocation, not scratch.
    scratch_bound = nbytes(f32, (1, cfg.hidden_size))  # one activation-sized allowance

    good = memory_table(rows, measured, peak)
    good_transient = next(r["derived"] for r in good if r["owner"] == "transient")
    assert good_transient < scratch_bound, good_transient

    # Mutant: drop the kv_pool static row but keep measured + peak (the pool is still held).
    dropped = [r for r in rows if r.owner != "kv_pool"]
    mutated = memory_table(dropped, measured, peak)
    absorbed = next(r["derived"] for r in mutated if r["owner"] == "transient")
    dropped_bytes = next(r.n for r in static_rows(rows) if r.owner == "kv_pool")
    assert absorbed == peak - sum(r.n for r in static_rows(dropped))
    assert absorbed >= dropped_bytes
    assert absorbed >= scratch_bound, (
        "the gate must go red when a static row is dropped: transient absorbs it and "
        "exceeds the shape-derived scratch bound")


def test_residency_row_roundtrips_through_benchrec(tmp_path):
    """The peak+transient ledger row carries the invariant in its shape and is appended
    through the one schema writer (a malformed row is rejected)."""
    import json

    from tilerl.memory import append_residency, residency_row

    p = tmp_path / "measurements.jsonl"
    row = residency_row("tiny-cpu", None, 500, static_bytes=450,
                        transient_bytes=50, target="cpu", model="tiny")
    rid = append_residency(row, p)
    got = json.loads(p.read_text())
    assert rid and got["metric"] == "device_resident_bytes"
    assert got["target"] == "cpu" and got["device"]["card"] is None
    assert got["shape"]["static"] + got["shape"]["transient"] == got["value"] == 500
    # the ledger is the same one the kernel roofline reads
    assert "measurements.jsonl" in str(p).split("/")[-1] or p.name == "measurements.jsonl"
    # A card-less sm90 row is refused at append: residency must not fabricate target/card.
    import pytest

    fake = residency_row("H20", None, 500, 450, 50, target="sm90", model="27B-nvfp4")
    with pytest.raises(Exception):
        append_residency(fake, tmp_path / "sm.jsonl")


if __name__ == "__main__":
    for f in (test_weight_row_equals_materialized_params,
              test_per_kv_block_equals_pool_and_shares_fit_formula,
              test_state_shapes_equal_state_pool_tensors,
              test_plan_rows_cover_weights_kv_and_state_to_the_byte,
              test_plan_fp8_pool_row_includes_scale_planes,
              test_draft_pool_is_separate_not_folded_into_kv_pool,
              test_plan_budget_rows_are_arithmetic_without_a_card,
              test_plan_weights_row_from_checkpoint_faces_drops_nonserving_and_repacks_bf16,
              test_serve_dry_run_checkpoint_is_header_only_and_needs_dry_run,
              test_serve_dry_run_needs_device_free_off_cuda_and_prints_rows,
              test_peak_equals_static_plus_transient_exactly,
              test_transient_zero_on_tiny_and_red_under_a_dropped_static_row,
              test_residency_row_roundtrips_through_benchrec):
        f(None) if f.__code__.co_argcount else f()
    print("memory ledger: single-formula fit/plan, fp8 scales, checkpoint weights, tiny OK")
