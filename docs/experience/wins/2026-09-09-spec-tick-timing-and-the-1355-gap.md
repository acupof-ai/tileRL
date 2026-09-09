# The 135.5 gap is scheduling overhead, not the decode tick — and 135.5 never ran on main

**Date:** 2026-09-09
**Arch:** H20 (sm90) card 6, 27B NVFP4 + DFlash2 block drafter, per-tick wall timing with
sync on both sides, `scripts/acc_spec_tick_timing.py` / `scripts/acc_spec_divergence_logits.py`
**Task:** locate the 6.6% throughput gap between the recorded 135.5 tok/s B=1 W=8 arm
(2026-09-03 entry) and the current sha's 126.5

> Status: Shipped

## Context

The reproduction (same day, [the repro entry](2026-09-09-accspec-b1-repro-w8-block-drafter.md))
found the drafter's algorithmic behavior intact — 6.18 vs 6.12 tok/decode-fwd, 6.19 of 8
blocks accepted, 6.16x fewer trunk forwards — but the spec arm 6.6% slower on wall clock.
A reverse derivation (tok/s ÷ tok/decode-fwd) put the W=8 tick at 48.9 ms against the
recorded 45.2, an 8% regression. That derivation attributes the whole arm wall to decode
forwards; this entry times each tick directly instead.

## What Worked

**The instrument.** A class-level patch on `Engine._run_forward` times every pure-decode
tick with `torch.cuda.synchronize()` on both sides, recording (width, ms). At B=1 the
engine is already synchronous per tick (poll waits on the event), so the sync is near-free:
the identity `tok/s = (tok/decode-fwd) / tick × (decode_s/wall_s)` closes to 0.1%
(126.6 computed vs 126.5 measured).

**The tick did not regress — it improved.**

| sha | W=1 tick mean | W=8 tick mean | spec tok/s | decode_s/wall_s |
|---|---:|---:|---:|---:|
| 09657c0 (first main sha that can run B=1) | 11.77 ms | 43.52 ms | 131.7 | 0.936 |
| f80e894 (current) | 11.56 ms | 41.96 ms | 126.5 | 0.860 |

The W=8 tick is 3.6% *faster* on the current sha. The reverse-derived 48.9 ms was an
artifact: decode forwards are only 86% of the arm wall, and the derivation spread the
other 14% across them. The whole 6.6% throughput gap is the ratio: non-decode time grew
from 31.6 s to 71.9 s per 200-question run (0.158 s to 0.360 s per question) — a 2.3x
growth in scheduling/eval overhead, with prefill only ~0.04 s/question of it. The decode
kernels are innocent; the regression lives in the host path around the tick.

**135.5 has never run on a main sha.** 09657c0 is the first commit on main whose
`acc_spec_arms.py` can run B=1 at all (the `--concurrency` flag landed in #58; the recorded
sha 40bc83c is hardwired to B=8 — see
[the provenance error entry](../errors/2026-09-09-the-recorded-sha-was-the-tip-not-the-tree.md)).
At 09657c0 the spec arm reads 131.7 tok/s, 2.9% short of the recorded 135.5. The number
most likely came from the #58 branch (f49e006), which never sat on main.

**The 38/200 base-vs-spec completion gap is by-design drift, not a verify bug.** A
spec-vs-spec control (two identical spec runs, same process) differs on 0/200 completions
with score Δ = 0 — the gap is specific to the base-vs-spec pair. At the first diverging
position of three diverging questions, both arms commit their own trunk logits' argmax:
the verify logic is correct. The two paths' logits differ by ~1e-1 (max 2.27), not the
~1e-6 last-mile tile rounding first guessed — the target model computes 8 positions per
forward in the spec path and 1 in the base path, different kernels down the whole path,
and the argmax flips at near-ties (top-2 gap as small as 0.002). `_verify`'s docstring
already states bit-identity is not guaranteed off the CPU reference. The noise-band
observation this adds: on 200 greedy problems, run-to-run score Δ = 0 (n=1, greedy only;
the graph-replay nondeterminism cc measured is on sampled paths).

## Rule

A throughput gap between two spec runs is not a tick gap until the tick is timed
directly. The identity `tok/s = (tok/decode-fwd) / tick × (decode_s/wall_s)` has three
factors; a reverse derivation that defaults the unmeasured one to a constant invents the
regression (it did, twice: 8.1% and 45.2 ms). Here the tick improved 3.6% and the
scheduling overhead around it grew 2.3x — the regression to bisect is in the host path,
not the kernels. And a bench number whose recorded sha cannot produce it is a provenance
bug first and a performance question second: 135.5's sha pointed at a B=8-hardwired tree.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-09 | f80e894 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 + DFlash2 | — | W=1 11.56 / W=8 41.96 | 126.5 spec / 79.5 base (B=1, 200 GSM8K, 512 cap) |
| 2026-09-09 | 09657c0 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 + DFlash2 | — | W=1 11.77 / W=8 43.52 | 131.7 spec / 81.9 base (B=1, 200 GSM8K, 512 cap) |

Raw artifacts: `/work/acctick.log` (current sha) and `/work/acctick0.log` (09657c0) on
the pod, both 0-compile; per-arm JSONs under `/work/accspec_tick/` and `/work/accspec_tick0/`.
