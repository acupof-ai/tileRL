# Prefill width = prompt length ⇒ a kernel set per distinct prompt — cuda(H20), 2026-08-28

## Context

First MMLU run through the engine (1000 prompts of ~120–400 tokens, one
generated token each) had made no progress after 16 minutes: GPU at 0%,
nine `nvcc` processes, 662 new kernel variants in the tilelang cache.

## Root Cause

`Engine._run_forward` sized the batch's forward at `width = max(seq_q)`,
i.e. exactly the prefill chunk length. tilelang kernels specialize on every
shape, so each distinct prompt length compiled a fresh set of prefill kernels
(~10 s each). The harness never saw it: its prompts are all 512/2048/8192.
Any real traffic with varied prompt lengths hits this.

## Fix

Pad the forward width to a multiple of `_PREFILL_BUCKET = 64`; the padding
rows were already masked by `seq_q_lens` in every kernel (mixed batches used
the same mechanism), and the logits index keeps the true chunk. Bounded
shapes: ≤ 8 prefill variants per batch size.

## Rule

Every shape a kernel specializes on must come from a bounded bucket, never
from user data. The perf harness only covers the bucket boundaries; run a
varied-length workload (MMLU is a good one) before calling serving done.
