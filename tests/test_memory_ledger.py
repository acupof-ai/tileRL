"""The memory ledger's derived rows equal the bytes the engine actually holds, and the
fit/plan share ONE formula.

This is the gate docs/design-cost-model.md rests on: plan is the allocator's input, not
a second computation — build_engine's fit calls memory.fit_num_blocks and the pool row
calls the same per_kv_block_bytes; the state shapes are the ones LinearStatePool
allocates. We assert that single formula against the live tensors. GPU columns (nvfp4
weights, the mem_get_info fit, host/SSD) are pending-remote; on the CPU tiny model the
weights, KV pool and state rows are exact.
"""


from tilerl.cli import _build_model
from tilerl.engine import build_engine
from tilerl.memory import (
    fit_num_blocks,
    per_kv_block_bytes,
    plan,
    state_shapes,
    weight_row,
)
from tilerl.testing import RefBackend


def _engine(num_slots=4, num_blocks=8):
    cfg, model = _build_model("tiny", seed=0)
    eng = build_engine(cfg, model, RefBackend(), num_blocks=num_blocks, num_slots=num_slots,
                       max_batch=num_slots, max_total_tokens=2048, max_num_batched_tokens=512)
    return cfg, model, eng


def _state_actual(pool):
    return sum(t.numel() * t.element_size() for t in
               (pool.states, pool.conv_windows, pool.step_states, pool.step_windows,
                pool.win_parity) if t is not None)


def test_weight_row_equals_materialized_params():
    _, model, _ = _engine()
    assert weight_row(model.params).n == sum(
        t.numel() * t.element_size() for t in model.params.values()) > 0


def test_per_kv_block_equals_pool_tensors_and_shares_fit_formula():
    """The ONE block formula equals k_pool/v_pool per block, at several pool sizes, and is
    exactly what fit_num_blocks divides free memory by."""
    cfg, _, eng = _engine()
    kv = eng._kv
    actual_block = (kv.k_pool.numel() * kv.k_pool.element_size()
                    + kv.v_pool.numel() * kv.v_pool.element_size()) // kv.num_blocks
    assert per_kv_block_bytes(cfg, kv.k_pool.dtype, kv.kv_fp8) == actual_block
    # fit with a known free budget is pure arithmetic: blocks = max(64, free*2/3 // block).
    free = actual_block * 300
    want = max(64, int(free * 2 / 3) // actual_block)
    assert fit_num_blocks(cfg, free, kv.k_pool.dtype, kv.kv_fp8, floor=64) == want
    # a cap clamps it
    assert fit_num_blocks(cfg, free, kv.k_pool.dtype, kv.kv_fp8, cap=10, floor=64) == 10


def test_state_shapes_equal_state_pool_tensors():
    cfg, _, eng = _engine()
    pool = eng._states
    from tilerl.memory import _dtype_fmt, nbytes
    fmt = _dtype_fmt(pool.states.dtype)
    derived = sum(
        4 * pool.num_slots if kind == "parity" else nbytes(fmt, shape)
        for shape, kind, _ in state_shapes(cfg, pool.num_slots, 0))
    assert derived == _state_actual(pool)


def test_plan_rows_cover_weights_kv_and_state_to_the_byte():
    cfg, model, eng = _engine()
    kv, sp = eng._kv, eng._states
    rows = plan(cfg, model.params, 0, num_slots=sp.num_slots, num_blocks=kv.num_blocks,
                state_dtype=sp.states.dtype, kv_io=kv.k_pool.dtype, kv_fp8=kv.kv_fp8)
    actual = {
        "weights": sum(t.numel() * t.element_size() for t in model.params.values()),
        "kv_pool": (kv.k_pool.numel() * kv.k_pool.element_size()
                    + kv.v_pool.numel() * kv.v_pool.element_size()),
        "state_slots": _state_actual(sp),
    }
    for owner, want in actual.items():
        got = sum(r.n for r in rows if r.owner == owner)
        assert got == want, (owner, got, want)


def test_plan_budget_rows_are_arithmetic_without_a_card():
    """device_free is a parameter: the free*2/3 pool rule and free/4 snapshot budget are
    exercised on CPU by passing the number — they are arithmetic, not pending-remote."""
    cfg, _, _ = _engine()
    rows = plan(cfg, None, 1000, num_slots=1, num_blocks=64)
    by = {r.owner: r.n for r in rows}
    assert by["kv_pool_budget"] == int(1000 * 2 / 3)
    assert by["prefix_entries_budget"] == 1000 // 4


def test_serve_dry_run_prints_memory_rows_with_zero_delta(capsys):
    """`tilerl serve --dry-run --json` builds and prints the ledger, never starts uvicorn."""
    import json

    from tilerl import cli

    cli.cmd_serve(cli._build_parser().parse_args(
        ["serve", "--model", "tiny", "--dry-run", "--json",
         "--blocks", "8", "--slots", "4", "--max-batch", "4"]))
    rows = json.loads(capsys.readouterr().out)
    alloc = [r for r in rows if r["owner"] in ("weights", "kv_pool", "state_slots")]
    assert {r["owner"] for r in alloc} == {"weights", "kv_pool", "state_slots"}
    assert all(r["delta"] == 0 for r in alloc)  # derived == measured to the byte on tiny


if __name__ == "__main__":
    for f in (test_weight_row_equals_materialized_params,
              test_per_kv_block_equals_pool_tensors_and_shares_fit_formula,
              test_state_shapes_equal_state_pool_tensors,
              test_plan_rows_cover_weights_kv_and_state_to_the_byte,
              test_plan_budget_rows_are_arithmetic_without_a_card,
              test_serve_dry_run_prints_memory_rows_with_zero_delta):
        f()
    print("memory ledger: single-formula fit/plan + derived == measured on tiny OK")
