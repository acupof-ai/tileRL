# GDN state comparison failed on CUDA: chunk-size-dependent rounding

## Context

`test_a_ragged_prompt_publishes_and_its_state_matches_no_store` failed on
CUDA (H20, card 1, 2026-09-10) with `max|delta| 3.586e-04, norms 43.6672
vs 43.6672`. CPU passed. The test compares the GDN recurrent state
restored from a prefix-store snapshot against a store-less engine's
state, both at the same `prefill_from`. The identical norms and tiny
delta read as a distribution difference — "the output can be
byte-identical while this is wrong."

First observed ≤ 2026-09-10.

## Root Cause

The two engines computed the state at the restore point with different
prefill chunk sizes. `_build_plan` cuts the first prefill chunk at the
largest `_PREFILL_BUCKET` (64) boundary that fits the prompt:

- Warm-up (100 tokens): first chunk = 64. The snapshot is the state at
  the **end** of a 64-token chunk.
- Reference (164 tokens): first chunk = 128. The state at token 64 is
  an **intermediate point** inside a 128-token chunk.

The GDN kernel's parallel scan rounds differently for different chunk
lengths. The state at token 64 is not bit-identical between a 64-token
chunk and a 128-token chunk, and the deterministic ~3e-04 difference
propagates through the continuation to token 164. Both arms are
individually deterministic (bit-identical across reruns); the difference
is between them, not within them.

The state restore itself is exact: the snapshot is a `clone()`, the
restore is a `copy_()`. The defect is in the test's design, not the
restore path.

## Fix

Match the first-chunk sizes: `short, long = 66, 100`. Both prompts
produce a first chunk of 64 tokens, so the state at the restore point is
computed with the same chunk length in both arms. With matching chunks,
the states are bit-identical (`max|delta| = 0.0`).

## Rule

When comparing GDN (or any parallel-scan) states across two engines,
both must use the same prefill chunk size at the comparison point. A
difference in chunk length produces a deterministic rounding delta that
`allclose(rtol=1e-2)` rejects at near-zero elements, even though the
restore is exact.
