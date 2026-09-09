# The 135.5 gap: the 09657c0 baseline was void — the tick/prefill/base-arm comparisons are unmeasured

> **VOID — 2026-09-09 correction.** Every 09657c0 number in this entry was
> measured on a pod tree that was byte-identical to HEAD (see
> [the contamination error entry](../errors/2026-09-09-pod-09657c0-tree-was-head-contaminated.md)).
> The "09657c0 vs HEAD" differences below were run-to-run noise between
> identical-code runs. The tick improvement, the flat prefill kernel, the flat
> base-arm wall, the 4% spec regression, and the fp8 dispatch audit are all
> **unmeasured** — there was no valid 09657c0 baseline. The only conclusion
> that survives is that 135.5 has never run on a main sha (a provenance fact
> about the recorded number, not a measurement). The 09657c0 columns are kept
> and marked VOID rather than deleted, so the evidence of the mistake stays
> checkable. The line below is preserved as it stood: the mechanism of the
> two-instrument disagreement was never explained from the code, and no story
> was invented for it — that stance was correct even though the disagreement
> itself is now known to be noise.

**Date:** 2026-09-09
**Arch:** H20 (sm90) card 6, 27B NVFP4 + DFlash2 block drafter, per-tick wall timing with
sync on both sides, `scripts/acc_spec_tick_timing.py` / `scripts/acc_spec_divergence_logits.py`
/ `scripts/acc_spec_overhead.py` / `scripts/acc_spec_prefill_profile.py`
**Task:** locate the 6.6% throughput gap between the recorded 135.5 tok/s B=1 W=8 arm
(2026-09-03 entry) and the current sha's 126.5

> Status: **Corrected — the 09657c0 baseline was void; the tick/prefill/base-arm
> comparisons are unmeasured. See the VOID banner above.**

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
| 09657c0 (first main sha that can run B=1) | **VOID** 11.77 ms | **VOID** 43.52 ms | **VOID** 131.7 | **VOID** 0.936 |
| f80e894 (current) | 11.56 ms | 41.96 ms | 126.5 | 0.860 |

The W=8 tick "3.6% faster" claim is **VOID** — both rows ran on HEAD. The
"unattributed improvement" framing is moot: there was no second sha to compare.

The reverse-derived 48.9 ms was an artifact: decode forwards are only 86% of the arm
wall, and the derivation spread the other 14% across them. The throughput gap is two
gaps with different causes:

| segment | size | explained? |
|---|---:|---|
| 135.5 (f49e006, never on main) → 131.7 (09657c0) | −2.8% | **VOID** — 131.7 was HEAD, not 09657c0; the segment is noise |
| 131.7 (09657c0) → 126.5 (current) | −3.9% | **VOID** — both endpoints were HEAD; the "4% regression" was noise |

**The prefill kernel is flat — VOID.** A five-bucket profile
(`scripts/acc_spec_prefill_profile.py`, same instrument at both
shas, B=1, 50 questions, sync on both sides of `Model.forward`, 0 compiles, no graph
fallback at either sha):

| per question | 09657c0 | current |
|---|---:|---:|
| 1. tokenize + render | **VOID** 0.4 ms | 0.4 ms |
| 2. admit (block alloc) | **VOID** 0.1 ms | 0.1 ms |
| **3. kernel (Model.forward, synced)** | **VOID 290.9 ms** | **288.0 ms** |
| 4a. plan excl admit | **VOID** 0.0 ms | 0.0 ms |
| 4b. forward host | **VOID** 2.8 ms | 2.7 ms |
| 4c. step remainder | **VOID** 0.1 ms | 0.1 ms |
| wall | **VOID 211.0 s** | 209.9 s |
| kernel first step | **VOID** 629 ms | 613 ms |
| kernel rest median per chunk | **VOID** 123.8 ms | 118.4 ms |

The "kernel is 1% faster per question and 4% faster per chunk at HEAD" claim is
**VOID** — the 09657c0 column was HEAD. The base-arm wall "flat" claim is VOID
for the same reason. An earlier decomposition (a synced wrap of
`Engine._run_forward`, `scripts/acc_spec_overhead.py`) had put prefill at
0.143s → 0.298s per question (2.1x) and named it the regression. The two
instruments **agree at HEAD** (14.7 vs 14.4 s per 50 questions) and **disagree
2x at "09657c0"** (7.2-9.1 vs 14.55 s) — but the "09657c0" tree was HEAD, so
the disagreement was HEAD-vs-HEAD noise, not a timing-boundary move. The
mechanism of the disagreement was never explained from the code — it is
recorded here as an open question, not a conclusion, and no story was invented
for it. That stance was correct even though the phenomenon itself is now known
to be noise.

**The fp8→bf16 hypothesis is dead — VOID test.** A proposed explanation was that the prefill
activation path fell back from fp8 to bf16 kernels (~2x). Runtime dispatch at "09657c0"
is all-fp8 (`linear_fp4` → `linear_fp4_fp8`, `linear_fp8` → `linear_fp8` /
`linear_fp8_gemv`, zero bf16), and the phase derivation (`m==1` gemv / `m<=16` decode /
else prefill), the `_MX=8` / `_MGEMV=3` thresholds, and the `_CUDA_PLAN` table are
byte-identical at the two shas. **But both trees were HEAD**, so "all-fp8 at both shas"
was trivially true and proved nothing about 09657c0. The hypothesis is neither
confirmed nor refuted; it is untested. The hypothesis predicted a ~2x kernel at HEAD;
the measurement is −1% — but the measurement's baseline was void.

**What remains is unmeasured — the "spec-arm-specific ~4%" was noise.** The
apples-to-apples comparison was 09657c0 → current: 131.7 → 126.5 tok/s on 200
questions (−3.9%), and 123.1s → 128.4s on the 50-question overhead harness
(+4.3% wall). **Both endpoints were HEAD** — the 4% was run-to-run noise
between identical-code runs. There is no evidence for a spec-arm regression,
and no evidence against one; the question is unmeasured pending a true 09657c0
baseline. The draft-coupled prefill path (`hidden_out` / `aux_layers` /
`draft.step`) remains the right place to look *if* a real gap reappears once a
valid baseline exists.

**The 09657c0 arm gap is an arm-order artifact — survives as a HEAD conclusion.**
Base prefill (9.1s) exceeded spec (7.2s) at "09657c0"; a reversed-order run
(spec first) flips it — spec 10.2s, base 7.1s. Whichever arm runs first pays a
~2-3s one-time cost in its prefill bucket (warm JIT cache, 0 compiles — not
compilation). This is a within-tree comparison, so it is real — but it is a
conclusion about HEAD, not about 09657c0, since both trees were HEAD. The
comparable second-arm numbers are 7.1-7.2s vs 14.9s, but both are HEAD.

**The prefix-reuse hypothesis is dead by construction.** A proposed explanation for
the convergence was that prefix reuse broke: GSM8K prompts share a chat-template
prefix, so hits would both lower prefill and make arms unequal. It cannot be tested
in this harness because prefix reuse is off at both shas by construction — the engine
raises on `draft.aux_layers` with a real prefix store (engine.py:445, identical at
09657c0's engine.py:333) and both shas' harnesses pass `NoPrefixStore`
(acc_spec_arms.py:103 at 09657c0). A stats() run confirms `prefix_hits = 0` at both
arms. The serving path's prefix reuse — a README flagship (19x cross-turn) — has
never been exercised by any of this; `bench_chat_reuse.py --turns 6` is the
instrument for that question.

encode/detokenize is genuinely negligible, not unmeasured: the wrappers fire exactly
50+50 times per arm (once per prompt each, eval.py:48/60), totaling 0.016-0.020s encode
and 0.003-0.012s detokenize per 50 questions — the 0.0 in earlier tables was one-decimal
rounding.

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
already states bit-identity is not guaranteed off the CPU reference. Same-sha
run-to-run is 0/200 (the noise floor, n=1 greedy); cross-sha is 167/200 at 09657c0 vs
168/200 at the current sha — deterministic within each sha, a 1-question drift between
shas, not noise.

## Rule

A wall bucket is not a measurement of the thing in the bucket. A bucket is "wall minus
the parts I accounted for", so any unaccounted time falls into some bucket, and the
bucket does not tell you it over-collected. The 09657c0 prefill bucket under-counted by
half and nothing would have found it for two months — except a second instrument
measuring the same quantity. **Any number derived from a remainder must have a
direct-measurement control.** The per-chunk line in the table above was that control
this time, and it was worth the whole profile.

A throughput gap between two spec runs is not a tick gap until the tick is timed
directly. The identity `tok/s = (tok/decode-fwd) / tick × (decode_s/wall_s)` has three
factors; a reverse derivation that defaults the unmeasured one to a constant invents the
regression (it did, twice: 8.1% and 45.2 ms). An arm gap that flips when the arm order
flips is a first-use cost charged to the first arm, not a difference between the arms —
run the reversed order before explaining a two-arm difference (it killed the 9.1-vs-7.2
here). A hypothesis about prefix reuse is untestable in a harness that builds with
`NoPrefixStore` — check the construction before proposing the counter run. A bench
number whose recorded sha cannot produce it is a provenance bug first and a performance
question second: 135.5's sha pointed at a B=8-hardwired tree. And a hypothesis priced on
"this file changed 765 lines" is not evidence — the fp8→bf16 guess died on the first
measurement.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-09 | f80e894 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 + DFlash2 | — | W=1 11.56 / W=8 41.96 | 126.5 spec / 79.5 base (B=1, 200 GSM8K, 512 cap) |
| 2026-09-09 | 09657c0 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 + DFlash2 | — | **VOID** W=1 11.77 / W=8 43.52 | **VOID** 131.7 spec / 81.9 base — tree was HEAD |
| 2026-09-09 | f80e894 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 | 288.0 ms/q kernel | — | base wall 209.9 s / 50q (five-bucket profile) |
| 2026-09-09 | 09657c0 | H20 card 6 | cuda/sm90 decode-graph | Qwen3.8-27B-NVFP4 | **VOID** 290.9 ms/q kernel | — | **VOID** base wall 211.0 s — tree was HEAD |

Raw artifacts: `/work/acctick.log` (current sha) and `/work/acctick0.log` (09657c0) on
the pod, both 0-compile; per-arm JSONs under `/work/accspec_tick/` and `/work/accspec_tick0/`;
five-bucket profiles under `/work/accpf3/` (current) and `/work/accpf0/` (09657c0),
logs `/work/accpf3.log` / `/work/accpf0.log`.
