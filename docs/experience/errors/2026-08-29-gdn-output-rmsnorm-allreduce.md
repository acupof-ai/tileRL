# The GDN prefill kernel's output RMSNorm is still a serial thread-0 reduce, and I could not move it

## Context

`us/step` in `make_gdn_chunk_fused` is **flat at 3.08-3.28 across T = 64, 128,
256, 512** (`scripts/occ_gdn.py --lens`). A serial scan whose per-step cost does
not amortise is latency-bound, and the biggest thing on that critical path is
visible in the source:

```python
out_s[tv] = acc_o[0]
T.tvm_storage_sync("shared")
if tv == 0:                       # one thread, 127 idle
    for j in T.serial(V):         # 128 dependent shared-load + FMA
        acc_sq[0] += out_s[j] * out_s[j]
T.tvm_storage_sync("shared")
```

~128 dependent shared-memory FMAs is roughly 3840 cycles, ~2.7 us at 1.4 GHz,
against a measured 3.15 us per step.

The same kernel's q/k L2-norms were already moved onto
`T.tvm_thread_allreduce` for exactly this reason — its comment says "Thread 0
alone summing K=128 twice is 256 dependent FMAs on the critical path of EVERY
token — at T=512 roughly half this kernel". The output norm was left behind.

## What Failed

Three attempts, all rejected by tilelang's layout inference on CUDA:

1. `po = alloc_local`, fed from `acc_o` — "local vs local.fragment" mismatch.
2. `po = alloc_fragment` to match `acc_o` — same conflict, now reported against
   the `StepStates` loop.
3. Stage through shared and read it back, copying the `pq`/`pk` pattern
   verbatim — still rejected.

Attempt 3 is the informative one. It looks identical to the working q/k code,
but the data flows the other way: `q_s`/`k_s` are shared buffers filled BEFORE
the token loop and read by a local, whereas `acc_o` is a fragment written
INSIDE the loop, and exporting it to shared changes the layout the whole loop
must satisfy. Copying the shape of a working idiom is not copying the idiom.

Reverted at three attempts, per the rule set before starting.

## What Is Still True

- The cost is per-step, not launch overhead (flat us/step across T).
- The serial reduce is ~86% of that step by arithmetic, unverified by
  experiment — no variant compiled, so no A/B exists.
- The other measured lever on this kernel stands: it is SM-limited, and a V
  split is worth 1.67x on the kernel / +13.4% on prefill
  ([2026-08-29-gdn-48-blocks-78-sms.md](2026-08-29-gdn-48-blocks-78-sms.md)).
  That change needs the same epilogue moved out of the kernel entirely, which
  sidesteps this layout problem rather than solving it.

## Rule

Set the attempt limit before the first attempt, and spend the attempts on
different hypotheses rather than three variants of one. All three here were
"make the allreduce inputs type-check"; none tested whether the reduce is
actually the 2.7 us the arithmetic claims. A profiler run inside the kernel, or
a variant that simply deletes the norm to see the floor, would have been worth
more than attempts 2 and 3.
