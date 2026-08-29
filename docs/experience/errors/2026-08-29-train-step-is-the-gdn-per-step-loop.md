# The 27B train step is 491K kernels, and 62% of them are one Python loop — 2026-08-29

> Status: root cause measured; the fix (a chunked GDN backward) is not written.

## Context

Training had no perf work at all. The baseline row is
`train/27B-lora-b1t256 = 41.2 tok/s`, against a **forward** that runs at 2218
tok/s on the same model — a fwd+bwd should be within ~3x of the forward, and
this is 54x.

## What the step actually is

`scripts/profile_train.py`, 64 layers, 1x256, LoRA on the frozen fp4 base:

```
GPU-busy 2296.2 ms, 491113 kernels
  torch elementwise            128644   4.8 us   618.2 ms  26.9%
  linear_fp4_bwd (ours)           168  2104.7    353.6     15.4%
  torch vectorized elementwise 129467   2.5      317.7     13.8%
  Memcpy DtoD                  112756   1.4      154.3      6.7%
  cuBLAS gemv (two kernels)     73728   ~3       230.4     10.0%
  torch reduce                  25877   4.1      107.2      4.7%
```

**~434,000 micro-op launches, 62% of the step.** The real backward work — the
fp4 weight GEMMs — is 354 ms of the 2296.

They come from `reference.gdn_backward`, which runs two Python loops over the
time dimension: a forward re-scan (~10 launches per step) and a reverse scan
(~18). 48 GDN layers x 256 steps x ~28 = ~344K, plus the einsums that dispatch
to cuBLAS gemv. The count depends on (layers x T) and NOT on batch, so B=1 —
the only batch the bench measured — is the worst shape the number can be taken
at.

## Memory, and a claim retracted

I claimed the same loop also costs 39 GB, from `states[b, t+1, 48, 128, 128]`
f32 = 808 MB per layer held across 48 layers. **Wrong.** `states` is a local of
`gdn_backward`, built during its own forward re-scan and freed when that layer's
backward returns — one layer's worth is live, never 48. Measured
(`scripts/probe_train_mem.py`):

| T | peak GiB | over base | GiB/token |
|---:|---:|---:|---:|
| 64 | 47.0 | 23.8 | 0.372 |
| 128 | 50.5 | 27.3 | 0.213 |
| 256 | 57.5 | 34.3 | 0.134 |

Base (weights + adapters) is 23.2 GiB. Fitting the two ends: **20.3 GiB fixed +
0.055 GiB/token**. The cost is dominated by a T-independent 20.3 GiB, not by
anything the scan stores. B=2 x T=256 still OOMs a 95 GiB H20, so ~256 tokens
per step is the practical cap — but the reason is the fixed term plus f32
activations, not the state tensor.

## The fix, not written

Chunked backward: fla's `chunk_gated_delta_rule_bwd_dhu`, ported in
`/Users/bytedance/code/tilelang/examples/gdn/example_chunk_delta_bwd.py`. Its
state scan runs `S/64` iterations of matmuls instead of `S` steps of elementwise
ops — at T=256 that is 4 iterations instead of 256.

The forward half already exists here and is proven: `reference.gdn_chunk_core`
(chunkwise-WY, equal to the serial scan to 3.7e-07). What is missing is the
adjoint: the per-step quantities the current backward indexes (`states[:, step]`
for the gate gradient, `ps`/`deltas` for beta and k) do not exist in the chunked
form, so this is a port of fla's decomposition, not a loop rewrite.

Oracle for it is already in the tree: `test_gdn_bwd` gradchecks
`reference.gdn_backward` against `torch.autograd` on `gdn_forward`, at any T.

## Rule

A component with no perf coverage is not "fine", it is unmeasured. Training had
a correctness gate and a throughput row and neither said that 62% of the step
was Python-loop launch overhead. The profile took one run.
