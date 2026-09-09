# A test patched a seam the CUDA tick never crosses — 2026-09-10

## Context

`test_submit_rollback_and_terminal_failure` passed on CPU and failed on the pod
with `DID NOT RAISE RuntimeError` (first observed ≤ 2026-09-10, card 0, tree
`ccf1efde`). The test patches `engine._model.forward` to raise, then asserts
`step()` raises, `take(1)` raises, and the engine rolls blocks and slots back
to zero.

## Root Cause

A pure-decode tick on CUDA replays a captured graph through
`Engine._run_decode_graph` and never calls `_model.forward` — the graph was
captured in an earlier tick, before the patch. The patched seam was entered 0
times on the card; the graph seam was entered 1 time. `step()` did not raise,
`_failed` stayed empty, the request kept running, and the rollback assertions
were never reached. On CPU `decode_graph_on` is False, the eager path calls
`_model.forward`, and the test passed — the injection worked on exactly one
backend.

Measured on card 1, tree `73e095cb`: CUDA `{forward: 0, graph: 1}`, no raise,
blocks/slots 1/1; CPU `{forward: 1, graph: 0}`, raise, rollback to 0/0.

## Fix

The test patches both seams. Each backend crosses exactly one: the eager path
calls `_model.forward`, the graph path calls `_run_decode_graph`. The
assertions (raise, `take(1)` raises, rollback to zero) are unchanged.

## Rule

A test that injects a failure must inject it at the seam every backend it runs
on crosses — count the injection on the card, not only on CPU. A mutant that is
never entered is the same shape as one that is ignored: the test stays green
while the behavior it claims to gate is untested.
