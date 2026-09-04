# The 64% roofline rested on a byte count nobody derived — 2026-09-03

## Context

`README.md` stated that a decode tick reads 22.8 GB in 11.0 ms — 2.1 TB/s, 64%
of the H20's measured 3312 GB/s — and used it to argue the remaining gap is
kernel efficiency, and to frame the whole sglang comparison. The 22.8 comes from
`docs/analysis/2026-08-28-vs-sglang-h20.md:22`, which records it in a table
with no derivation.

Two sessions tried to rebuild it from the checkpoint and neither reached it.

## Root Cause

`param_specs(qwen38_27b())` gives 26.90 B parameters, 25.62 B of them in
`fp4_param_keys`. At 4 bits that is 12.81 GB of nibbles. The block scales are
where the number moves:

| scale storage | bytes | tick total |
|---|---:|---:|
| as the checkpoint holds it — fp8, one per 16 | 1.60 GB | 14.41 GB |
| as the kernel reads it — f32 | 6.41 GB | **19.22 GB** |
| recorded in the entry | — | 22.80 GB |

The widening is at load, not at dispatch: `model.py:141` calls
`renorm_fp4_scale(weight_scale.float(), ...)`, so the f32 scale is resident and
streamed every tick. `backend.py:367`'s `self._f32(scale)` is then a no-op and
`kernels_linear.py:546` declares `Scale: T.Tensor((N, K // block), "float32")`.

`embed_tokens` is 2.54 GB of the 2.55 GB non-fp4 total and is a one-row gather
per token, so it does not belong in a streamed figure either.

The residual 3.6 GB between 19.22 and 22.80 is still unexplained. Ruled out:
`_pad2d` padding, since every fp4 weight's K is in {5120, 6144, 17408} and all
are `% 256 == 0` against the plan's K pad of 256, and every N is `% 4 == 0`
against the GEMV plan's N cap — padded and unpadded both compute 19.22 GB.

## Fix

The README no longer quotes 64%. It states the range the two derivations bracket
(53–63%), names the missing measurement, and says the conclusion that survives
either figure: decode is bandwidth-bound and the gap is kernel efficiency. A
DRAM-read counter on one steady-state tick closes it and needs no arithmetic
anyone can dispute; it is `pending-remote`.

## Rule

A byte count in a bench entry is a claim like any other and needs its derivation
written next to it. This one was quoted in the README, used to rank the memory
system against the kernels, and carried a head-to-head against another engine —
for six days, and neither of the two people who tried could reproduce it.
