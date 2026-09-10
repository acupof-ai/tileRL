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


def test_plan_weights_row_from_checkpoint_specs_matches_header_bytes(tmp_path):
    """The header-only weights row prices the real device-face population (nvfp4 packed
    + fp8-block + bf16) through nbytes — no model load, no params dict."""
    import torch
    from safetensors.torch import save_file

    from tilerl.memory import plan, weight_row_specs
    from tilerl.precision import checkpoint_weight_specs, fp8_block_dev, nbytes, nvfp4_dev

    N, K = 128, 64
    save_file({
        "m.weight_packed": torch.zeros((N, K // 2), dtype=torch.uint8),
        "m.weight_scale": torch.zeros((N, K // 16), dtype=torch.float8_e4m3fn),
        "m.weight_global_scale": torch.zeros(1, dtype=torch.float32),
        "b.weight": torch.zeros((N, K), dtype=torch.float8_e4m3fn),
        "b.weight_scale_inv": torch.zeros((1, 1), dtype=torch.float32),
        "embed.weight": torch.zeros((100, K), dtype=torch.bfloat16),
    }, str(tmp_path / "model.safetensors"))
    specs = checkpoint_weight_specs(tmp_path)
    want = (nbytes(nvfp4_dev, (N, K)) + nbytes(fp8_block_dev, (N, K)) + 100 * K * 2)
    assert weight_row_specs(specs).n == want
    cfg, _, _ = _engine()
    rows = plan(cfg, None, 0, num_slots=1, num_blocks=8, ckpt_specs=specs)
    assert sum(r.n for r in rows if r.owner == "weights") == want


def test_serve_dry_run_checkpoint_is_header_only_and_needs_dry_run(tmp_path, capsys):
    """--dry-run --checkpoint DIR prices from headers (no load, no engine, measured/delta
    null); --checkpoint without --dry-run refuses; blocks are fitted, not built."""
    import json

    import torch
    from safetensors.torch import save_file

    from tilerl import cli
    from tilerl.memory import fit_num_blocks, weight_row_specs
    from tilerl.precision import checkpoint_weight_specs

    save_file({"embed.weight": torch.zeros((100, 64), dtype=torch.bfloat16)},
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
    specs = checkpoint_weight_specs(tmp_path)
    assert by["weights"]["bytes"] == weight_row_specs(specs).n
    # Header-only: nothing was built, so there is no measured column or delta.
    assert by["weights"]["measured"] is None and by["weights"]["delta"] is None
    # build_engine fits AFTER weights and the state pool are resident; the header-only fit
    # must subtract both before fitting, or it prices blocks the card cannot hold.
    from tilerl.memory import _state_bytes
    free_after_fixed = 1000000 - weight_row_specs(specs).n - _state_bytes(cfg, 4, f32)
    want_blocks = fit_num_blocks(cfg, free_after_fixed, torch.bfloat16)
    assert by["kv_pool"]["note"] == f"{want_blocks} blocks"


def test_27b_checkpoint_weights_row_matches_24_44gb():
    """Pending-remote: the header-derived weights row on the real 27B equals the measured
    24.44 GB (errors/2026-09-03-...: 264 nvfp4 + 233 fp8, config cannot see the split).
    Headers only, no weight bytes. The header-only CLI is exercised the same way:

        TILERL_27B_CKPT=/data00/Qwen3.8-27B-NVFP4 \\
        uv run tilerl serve --model qwen38-27b --dry-run --checkpoint \"$TILERL_27B_CKPT\" \\
            --device-free 60000000000
    """
    import os

    ckpt = os.environ.get("TILERL_27B_CKPT")
    if not ckpt:
        pytest.skip("set TILERL_27B_CKPT to the 27B NVFP4 dir; headers only, no weights")
    from tilerl.memory import weight_row_specs
    from tilerl.precision import checkpoint_weight_specs

    total = weight_row_specs(checkpoint_weight_specs(ckpt)).n
    assert abs(total - int(24.44e9)) < int(0.01e9), f"weights {total / 1e9:.3f} GB != 24.44 GB"


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


if __name__ == "__main__":
    for f in (test_weight_row_equals_materialized_params,
              test_per_kv_block_equals_pool_and_shares_fit_formula,
              test_state_shapes_equal_state_pool_tensors,
              test_plan_rows_cover_weights_kv_and_state_to_the_byte,
              test_plan_fp8_pool_row_includes_scale_planes,
              test_draft_pool_is_separate_not_folded_into_kv_pool,
              test_plan_budget_rows_are_arithmetic_without_a_card,
              test_plan_weights_row_from_checkpoint_specs_matches_header_bytes,
              test_serve_dry_run_checkpoint_is_header_only_and_needs_dry_run,
              test_serve_dry_run_needs_device_free_off_cuda_and_prints_rows):
        f(None) if f.__code__.co_argcount else f()
    print("memory ledger: single-formula fit/plan, fp8 scales, checkpoint weights, tiny OK")
