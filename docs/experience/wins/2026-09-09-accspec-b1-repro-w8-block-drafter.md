# W=8 block-drafter B=1 reproduction — 1.59x, not the recorded 1.728x

**Date:** 2026-09-09
**Arch:** H20 (sm90) card 6, 27B NVFP4 + DFlash2 block drafter, `scripts/acc_spec_arms.py`
**Task:** reproduce the 2026-09-04 W=8 arm (commit 40bc83c) on the current sha — the
README's 135.5 tok/s / 1.728x headline rests on it, and its 78.4 baseline disagreed with
the 94.6 measured for B=1 decode on the serving build

> Status: Shipped

> **Superseded on the tick claim (2026-09-09):** the "W=8 verify tick is 7.3% slower"
> claim below was reverse-derived from wall tok/s; directly timed, the W=8 tick improved
> 3.6% (43.52 → 41.96 ms). See [the tick-timing entry](2026-09-09-spec-tick-timing-and-the-1355-gap.md).
> The reproduction numbers below stand.

## Context

`2026-09-03-batched-selector-walk.md` recorded B=1 base 78.4 tok/s and B=1 spec W=8
135.5 tok/s — 1.728x — on 200 GSM8K, greedy, `max_new_tokens=512`, decode graph on, both
arms in one process. The README headline quotes that spec number. Two open questions:
does it reproduce on the current sha (f80e894), and why is the baseline 78.4 against the
94.6 the serving build measures for B=1 decode elsewhere.

Config copied verbatim: `--gsm8k-n 200 --mmlu-n 0 --max-new-tokens 512 --concurrency 1
--width 8 --decode-graph`, same harness, same model and draft weights, one process.
Pre-registered as `accspec-b1-repro`: spec arm ≥130 tok/s **and** base within ±5% of 78.4
(74.5–82.3).

## What Worked

Two runs on card 6: a cold run that warms the JIT cache, then the warm re-run both arms
are reported from. Per-arm compile verification on the warm log: **0 compiles in either
arm** (the cold run compiled 26 kernels, some inside the spec arm's timed window — its
spec wall is 529.3s against the warm 512.0s). This is the second time the warm-cache
0-compile rule proved itself on an H20 spec run: the cold spec arm reads 122.4 tok/s
against the warm 126.5, a 3.3% contamination that would have landed in the verdict.

| | wall | tok/s | tok/decode-fwd | block accepted | GSM8K |
|---|---:|---:|---:|---:|---:|
| recorded 2026-09-04 (40bc83c) base | 823.5s | 78.4 | 1.00 | — | 168/200 |
| recorded 2026-09-04 spec w8 | 477.4s | 135.5 | 6.12 | 6.14 of 8 | 167/200 |
| reproduced base (warm) | 813.4s | **79.5** | 1.00 | — | 169/200 |
| reproduced spec w8 (warm) | 512.0s | **126.5** | 6.18 | 6.19 of 8 | 168/200 |

**The baseline reproduces** (78.2 cold / 79.5 warm against 78.4, within ±5%) and **the
spec arm does not** (122.4 cold / 126.5 warm against 135.5, −6.7%). The measured ratio is
**1.591x**, not 1.728x. The pre-registered hypothesis is falsified on the spec side.

The drafter itself behaves identically: tok/decode-fwd 6.18 against 6.12, block accepted
6.19 of 8 against 6.14 (54463 of 73423 drafts, 74.2%), and the trunk runs 6.16x fewer
forwards (64612 → 10489) against the recorded 6.11x. What does not reproduce is the spec
arm's wall clock: 512.0s against 477.4s, a 7.3% slower spec tick at unchanged acceptance,
while the base arm matches to 1.2%. The 1.728x was priced on a faster spec tick than the
current sha produces; the cause is unread.

The 78.4-vs-94.6 baseline gap is workload, not regression: same card, same sha, same
build — 200 real GSM8K prompts with context growing past 512 tokens gives 79.5, the
30-tick synthetic-prompt microbenchmark gives 94.6. Both are `decode_tok_s` on
sm90/27B-nvfp4/fused+graph; only the shape differs, by 1.19x.

**Equality is not exact.** The spec arm's claim is string-identical greedy output; 38 of
200 completions differ between the arms (accuracy within 1 point, 169 vs 168). The
2026-09-03 entry did not record a diff count, so whether this is new is unknown. Recorded
here, not explained.

## Rule

The README's 135.5 tok/s and 1.728x do not reproduce on the current sha: the W=8 block
drafter at B=1 measures 126.5 tok/s and 1.59x, with the drafter's acceptance and
forward-reduction intact and only the spec tick slower. The headline was taken down in
#354; 135.5 stands in its dated 2026-09-03 entry, not the README. A spec arm's equality claim must be checked per completion — 38/200
divergences at unchanged accuracy is invisible to the accuracy row.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-09 | f80e894 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 + DFlash2 | — | — | 79.5 base / 126.5 spec-w8 (B=1, 200 GSM8K, 512 cap) |

Raw artifacts: `/work/accspecb1w.log` (warm, reported) and `/work/accspecb1.log` (cold,
26 compiles) on the pod; per-arm JSONs under `/work/accspec_b1/`.
