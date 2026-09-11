# Flaky wall-clock / race tests in the CPU+CUDA suite — inventory, 2026-09-11

Surveyed every test that reads a clock (`time.time`, `perf_counter`,
`monotonic`) or drives a background thread. A test is flaky on a shared CI
runner only when its PASS/FAIL line compares a measured wall duration to a
threshold. A clock used only as a *wait bound* (`join(timeout=)`, a spin-until
condition, a poll loop) cannot flake: it bounds the test, it does not judge the
product.

## The one flaky gate

**`tests/test_kv.py::test_a_yielded_gil_runs_a_background_load_promptly`** —
the only test whose verdict is a ratio of two wall durations:

```python
assert busy > yielded * 1.5
assert yielded < target_ms * 4
```

It measures whether `time.sleep(0)` yields the GIL to a background
`torch.load`. On a contended Mac dev box busy/yielded ≈ 29×; on CI macos-14 it
has measured 2.92× and, on 2026-09-11, 91.9/62.4 = **1.47× → red**, and
62.4 ms also grazes the absolute 4×-target ceiling. The effect size is a
property of how loaded the shared runner is, not of the code. This is the same
shape as the SSD fetch-hold test hardened in #471: an environmental timing
ratio standing in for a control-flow property.

### Fix (recommended)

The property is "a `sleep(0)` in the main loop hands the GIL to a queued
background thread." That is deterministically observable WITHOUT a wall ratio:
instrument a `threading.Event` (or an injected "did the worker run?" seam) the
background loader sets on its first GIL acquisition, run the busy loop a fixed
number of iterations with a `sleep(0)` each, and assert the worker ran; then
the same fixed iterations with NO yield and the worker not having run. The
existing KvTier already has exactly this seam (`hold_fetches_for_test`,
`_fetch_gate`/`_fetch_started`), which is why `test_a_*fetch*` gates are not on
this list. If a wall number is still wanted for the wins entry, keep the
measurement as a recorded print (it already prints the ratio) and move the
threshold to a card-only / manual job, never the merge gate.

## Clock uses that are NOT flaky (audited, leave as-is)

- `tests/test_e2e.py::_drain_clock` — the wall clock bounds a spin
  (`while time < end: step()`); the assertions are on counts/state after.
- `tests/test_e2e.py` `fetch_deadline = time.perf_counter()` — a value passed
  into the break-even arithmetic, not compared to a measured duration.
- `tests/test_iso.py:83` — computes `dt` per step but NEVER asserts on it
  (dead measurement). The real gate is loss + 2D-weight movement.
- `tests/test_kv.py:641`, `tests/test_server.py:711`, `test_api_sdk.py:93`,
  `test_serve_v100_sh.py` — `sleep`/`join(timeout=)` polling/wait bounds; the
  assertions are on outputs/counts. `test_serve_v100_sh` is a subprocess
  integration test that already waits on a condition with a timeout rather than
  asserting a duration.
- SSD `KvTier` gates (`test_kv.py`) — count bytes/calls and use the
  Event-controlled fetch seam; the durable/ page-cache ms numbers live in the
  errors entry, not a test threshold.

## Rule

A merge gate must not decide pass/fail by dividing one shared-runner wall
duration by another. Time a product in a controlled (card, dedicated) job and
record the number; gate the control flow with an Event/injected seam on CPU.
The two fetch-hold gates and `hold_fetches_for_test` are the template.
