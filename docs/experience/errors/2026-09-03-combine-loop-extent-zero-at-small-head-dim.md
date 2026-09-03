---
question: Why did every CUDA speculation test fail on main while the same tests passed on CPU?
status: measured
source: H20 sm90, tilelang 0.1.13, clean origin/main at a702c9a vs the same tree with a three-line fix
---

# A `D // 32` loop is a no-op when D is 16, and the output it never writes is `T.empty`

`paged_attention_combine` merges the split-KV partials into the attention
output. One warp owns a row; each lane owns one of 32 columns:

```python
for i in T.unroll(D // 32):
    ...
    Out[bb, hkv, g, i * 32 + lane] = T.cast(acc[0] / l[0], "bfloat16")
```

At `head_dim = 256` that is 8 passes. At `head_dim = 16` it is **zero**, and
`Out = T.empty(...)` is returned exactly as allocated — whatever the caching
allocator last left in those bytes.

Production is unaffected: the 27B has `head_dim = 256`. But `config.tiny()`
has `head_dim = 16`, so every CUDA test that decodes through the split-KV path
read garbage attention output whose value depended on allocation history.

## What it broke

`test_speculation_reproduces_greedy_decode` compares a run with a random draft
head against a run without one, and requires the tokens to be identical — a
random draft must be rejected at position 0 every tick. Attaching a draft
changes the allocation pattern, so the two runs read *different* garbage and
diverged at token 2. Four `test_engine_draft_matches_full_context_draft`
parametrizations failed the same way. All five passed on CPU, where the
reference kernel runs instead.

The assertion was correct, present, and had never been allowed to run. CI is
`TILERL_TARGET=cpu` on ubuntu and macos, so every CUDA-gated test skips there,
and a red CUDA gate can sit red across arbitrarily many commits.

## Root cause

A loop bound computed by truncating division, on a dimension whose small value
only ever appears in tests. `D // 32` silently encodes "D is a multiple of 32
and at least 32"; nothing in the kernel says so and nothing checks it.

## Fix

Round the bound up and guard the lane, the way a tail is normally handled:

```python
for i in T.unroll(T.ceildiv(D, 32)):
    if i * 32 + lane < D:
        ...
```

Two other pre-existing defects surfaced behind it, each hidden by the failure
in front of it:

- the decode arm of `paged_attention` handed `k_cache`/`v_cache` to the kernel
  unmigrated while the prefill arm two lines below went through `_dev` —
  invisible in serving, where `PagedKvPool` is already on device, and a device
  mismatch in any test that builds a CPU cache;
- `_OracleDraft.forward` in `tests/test_e2e.py` returned `torch.zeros(...)` on
  CPU, so indexing it with device indices raised. That arm of the test had
  never been reached, because the random-draft assertion above it failed first.

After all three: `tests/test_e2e.py -k "specul or draft"` is 5 passed on CUDA,
was 5 failed.

## Rule

A shape that only tests exercise is still a shape the kernel must handle. When
a loop bound is a division, write the tail — `T.ceildiv` plus a guard — rather
than assuming the divisor divides.

And a gate that only runs on hardware CI does not have is not a gate. Either
someone runs the CUDA suite on every merge, or its red state means nothing
when it is finally read.
