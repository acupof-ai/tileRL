# Two CUDA-only tests are red on clean main, and CI cannot see either

## Context

A second-hand report said `test_decode_graph_matches_eager` fails on CUDA
because eager and captured decode produce different greedy tokens at width 1 —
which, if true, would be a shipped-default correctness bug, since `engine.py`
turns the decode graph on whenever the backend device is CUDA.

Checked on a clean `origin/main` (a702c9a) synced to its own pod directory, no
local changes carried in, H20 GPU 2 idle, tilelang 0.1.13 / torch 2.11.0.

Both named tests are red. Neither is the reported bug.

## Root Cause

**`test_decode_graph_matches_eager` — a stale guard, not a divergence.** The
token assertion passes; eager and captured both emit
`[284, 309, 14, 48, 206, 155]`, identical at every index. What fails is the
line after it:

```python
assert captured._decode_graph_on and captured._decode_graphs.get(1) is not None
```

`3a81039 feat(engine): capture speculative ticks, one graph per (batch, chain
width)` re-keyed the cache from a scalar batch size to `(B, W)`
(`engine.py:689`, `:702`) and did not update this guard, written back in
`27dd42f`. The probe shows the graph is there and healthy —
`_decode_graph_on: True`, `_decode_graphs keys: [(1, 1)]`, `get((1,1))` returns
a `_DecodeGraph`, `get(1)` returns `None`. The lookup can never succeed, so the
assertion fails on a *working* graph. It fails safe — a guard that always fires
beats one that passes vacuously — but it has been red on CUDA ever since.

**`test_paged_attention_vs_naive` — a real missing migration.**
`RuntimeError: kernel paged_attention_decode input KCache device_type
mismatch`. The test builds `k_cache`/`v_cache` with a bare `torch.randn`, so
they are CPU tensors, and the decode arm of `Backend.paged_attention`
(`backend.py:481`) hands them straight to the kernel. The prefill arm two lines
below does `self._dev(self._c(q), torch.bfloat16)`; the decode arm does not.

The engine is not affected: the decode-graph run above drove six full decode
ticks through that same arm on CUDA and produced correct tokens, because
`PagedKvPool` is already on device. This is the test's construction meeting a
backend arm that trusts its caller.

**Why nobody saw either.** CI is `TILERL_TARGET=cpu` on ubuntu-latest and
macos-14 only, and both tests are CUDA-gated (`skipif(not
torch.cuda.is_available())` / the sm90 kernel branch). The deterministic set CI
calls "exactly the tests that block" has never executed either line.

## Fix

Not applied here — the agent that reported this has fixes in flight and two
people editing one line is worse than one. Both are one-liners:

- `tests/test_decode_graph.py:61` — `_decode_graphs.get(1)` →
  `_decode_graphs.get((1, 1))`, or better, assert the cache is non-empty so the
  next re-keying does not silently repeat this.
- `packages/.../backend.py:481` — migrate `k_cache`/`v_cache` through
  `self._dev` on the decode arm, matching the prefill arm.

## Rule

A CUDA-gated test is not covered by a green CI badge; it is only covered by
someone running it on a card. When a cache key changes, grep the tests for the
old key shape — a guard that reads a stale key fails on correct code, and the
failure text points at the feature, not at the guard.

Second-hand failure reports get reproduced on a clean tree before they get
escalated. "Captured decode diverges from eager" and "an assertion after the
token check uses a stale dict key" have the same red output and opposite
severity.
