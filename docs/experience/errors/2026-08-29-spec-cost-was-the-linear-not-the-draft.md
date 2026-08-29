# The speculative tick's missing 19 ms was in the linear kernels, not the draft head — 2026-08-29

> Status: root cause found; the fix (an M-row GEMV) is a separate entry.

## Context

`wins/2026-08-29-spec-decode-net-win.md` rejected speculative decoding at
0.43-0.76x of graph decode and closed with: "the remaining work is to find
where ~10 ms per draft step actually goes." That 10 ms was never measured. It
was **inferred by subtraction** — a 35.4 ms captured spec tick at B=1 depth 2,
minus an assumed ~13 ms verify replay, divided by two draft steps.

Both terms of that subtraction were wrong.

## Root Cause

Two direct measurements (`scripts/probe_draft_step.py`,
`scripts/profile_verify_replay.py`), B=1, 64 layers, H20 GPU 7:

```
draft step stages          verify replay, per width
  embedding       0.053      W=1 (plain decode)  11.27 ms
  norm embed      0.041      W=2                 29.59
  norm hidden     0.041      W=3                 29.70
  fc              0.059      W=5                 29.99
  full_attn       0.917
  mlp             0.273    per-kernel, W=1 -> W=2
  final norm      0.040      linear_fp4  gemv  5.18 -> mma8 13.49
  lm_head         0.441      linear_fp8  gemv  4.25 -> mma8  8.74
  greedy          0.071      gdn  decode_fused 0.39 -> chunk_fused 0.48
  --------------------       paged attention   0.22 -> 0.28
  sum of stages   1.967
  DraftHead.forward 2.062
```

- **A draft step costs 2.06 ms, not ~11.** The stages sum to 1.97 against a
  2.06 ms whole — there is no unexplained overhead in the draft head at all.
- **The verify replay costs 29.6 ms, not 13**, and it is FLAT in width: W=2,
  W=3 and W=5 are within 1.4% of each other. A flat cost is not per-token
  work; it is a fixed penalty for leaving the W=1 route.
- That penalty is **entirely the two linear kernels** (+13.5 and +8.7 ms).
  GDN, which was the obvious suspect (48 of 64 layers, and the chunk kernel is
  the most expensive one in prefill), contributes +0.08 ms.

`Backend._plan` buckets M=1 to a GEMV and 2 <= M <= 16 to a decode GEMM, and
`linear_*_mma8` pads M to 8 rows unconditionally. A verify of W=2 therefore
does eight rows of tensor-core work for two rows of result, at 2.6x (fp4) and
2.1x (fp8) the GEMV's cost for the same weight bytes.

## The same defect outside speculation

The M-padding is not specific to verify. Plain decode, W=1, varying batch:

| B | replay | aggregate tok/s |
|---:|---:|---:|
| 1 | 11.27 ms | 88 |
| 2 | 27.27 ms | 73 |
| 4 | 27.61 ms | 145 |
| 8 | 27.18 ms | 294 |

**B=2 and B=8 cost the same tick, and batching two requests is slower in
aggregate than serving one.** Nothing in the bench suite covered B=2 or B=4,
so a whole loss region sat unmeasured behind two rows that both looked fine.

## Rule

A cost attributed by subtraction is not a measurement. Both terms here were
off by more than 2x, and the difference happened to land on the one component
(the draft head) that five A/B experiments then failed to make faster —
because nothing was wrong with it. Time the stages before naming the suspect.

Corollary for the bench suite: measure the interior of a swept parameter, not
only its endpoints. B=1 and B=8 were both on the baseline; the defect lived at
B=2.
