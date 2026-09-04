# One of the on-policy guard's two arms was never tested, and it is the one CUDA uses

**Date:** 2026-09-05 · **Class:** error · **Where:** `src/tilerl/train.py:211`,
`tests/test_precision.py:23`

## Context

`_require_on_policy` is the refusal that keeps GRPO from sampling with an earlier
policy. Two conditions, one `if`:

```python
if engine._decode_graph_on is not False or not isinstance(engine._prefix, NoPrefixStore):
    raise ValueError("on-policy rollouts need build_engine(decode_graph=False, "
                     "prefix_store=NoPrefixStore()): ...")
```

One test covered it, building an engine with the prefix cache on and `decode_graph`
left at its default.

## Root cause

`_graph_on(backend, None)` returns `backend.device.type == "cuda"`, so on the CPU
target the default is **False** — the engine that test builds satisfies the graph half
and trips only the prefix half. Measured by deleting each half in turn and re-running
the only test that exercises the guard:

| mutation | result |
|---|---|
| `_decode_graph_on` half deleted | **1 passed** — untested |
| prefix half deleted | 1 failed — caught |

So a rename of `_decode_graph_on`, a change from `is not False` to truthiness, or a
refactor that dropped that clause would all have shipped green.

**It is the half that matters on the pod.** On CUDA the default is True, so a
`grpo_loop` call that forgot `decode_graph=False` lands on exactly the clause with no
coverage — and what it produces is not a crash but rollouts sampled from a captured
graph of an earlier policy, which is the silent failure the guard exists to prevent.

## Fix

The test now builds both engines. `decode_graph=True` is honoured on the CPU target
(verified: `_decode_graph_on is True` after `build_engine(..., decode_graph=True)`), so
the graph arm is testable here rather than being a CUDA-only path.

Mutation control after the change: **both arms CAUGHT**, restored green.

## Why the probe came first

The four-combination probe before the mutation was not redundant. It established that
`decode_graph=True` survives on cpu — if `_graph_on` had forced False there, widening
the test would have produced a second engine identical to the first and a mutation run
that still passed, which reads as "the guard is fine" rather than "the test is inert".
`PrefixStore` is also not a `NoPrefixStore` subclass, so the `isinstance` arm is a real
type test rather than an accidental pass.

## Rule

A guard with `or` between two conditions needs one test per operand. The default that
makes one arm unreachable is often environment-dependent — here the CPU/CUDA split in
`_graph_on` — so the arm that goes untested is the one the *other* target relies on,
and CI reports full coverage of the guard by running the half that cannot fire in
production.
