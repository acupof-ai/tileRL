# The GDN kernel's serial RMSNorm is 3-15% of a step, not the 86% I calculated

## Context

`us/step` in `make_gdn_chunk_fused` is flat at 3.08-3.28 across T = 64..512, so
the cost is per-token, not amortised launch overhead. The most visible thing on
that critical path was the gated RMSNorm: thread 0 summing V=128 serially,
inside the token loop, with 127 threads idle between two syncs.

I sized it by arithmetic — 128 dependent shared-memory FMAs at ~30 cycles each
is ~3840 cycles, ~2.7 us at 1.4 GHz, against a measured 3.15 us — and concluded
it was ~86% of the step. Three attempts to replace it with the block allreduce
that the same kernel's q/k norms already use were all rejected by layout
inference.

## The Measurement

Moving the epilogue OUT of the kernel entirely (raw core out; `silu_mul(z,
rmsnorm(core, NormW, 1e-6))` in the backend, two existing kernels) compiles on
the first try, which the allreduce never did. It buys:

| T | before | after | |
|---:|---:|---:|---:|
| 128 | 3.17 us/step | 2.71 | -15% |
| 512 | 3.15 us/step | 3.06 | **-3%** |

**3-15%, not 2-3x.** The 30-cycles-per-dependent-shared-load assumption was
wrong; the loop pipelines far better than that.

## What This Rules Out, And In

Ruled out: the per-step cost is not synchronisation or the reduce. It is the
recurrence itself — two 128-long FMA chains per token (`k·S` then `q·S`), which
no rearrangement of the epilogue touches.

Ruled in, and it now explains the whole history: **the arithmetic is not the
bottleneck, the dependency chain is.** That is why three chunkwise-WY rewrites
lost despite doing strictly less arithmetic, and why the occupancy measurement
(2x blocks for 1.20x time) shows so much headroom — more resident blocks hide
each other's chains, which is the only lever that has ever measured positive on
this kernel.

Reverted: 3% at T=512 is 1% of prefill, and it broke
`test_gdn_chunk_matches_decode` (the decode kernel still fuses its epilogue) and
the full-scale parity gate.

## Rule

Sizing a hot spot by cycle arithmetic is a hypothesis, not a measurement, and
mine was off by 6x. The experiment that settles it — delete the suspect code
and read the floor — was cheaper than the three attempts to optimise it, and I
ran it fourth instead of first.
