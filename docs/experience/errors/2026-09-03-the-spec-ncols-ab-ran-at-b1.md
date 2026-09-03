# The spec-ncols A/B ran at B=1 — both arms were the same kernel, V100 sm70, 2026-09-03

> Status: **#36's "spec decode is a wash under the gate" is WITHDRAWN.** The A/B behind it
> ran `bench_ctx_decode.py`, which submits **one** request, so a depth-3 verify tick was
> **4 rows on the 4 rung** — below `_NCOLS_MIN_M=32`, so `ncols=2` was **off in both arms**.
> The flat 0.995 / 1.000 / 0.988 / 0.996 / 0.998× was one code path measured twice. Re-run
> at B=4 in flight; the gate's threshold is unaffected because it was chosen from the M=1
> and M=512 measurements, which are real.

## Context

`_NCOLS_MIN_M` keys on the **compiled rung** `Mk`, and I wrote three entries reasoning
about which rung a verify tick lands on. The reasoning was right about serving:
`max_batch=4` × width 4 = 16 rows, which `_sm70_chunks` rounds up to 32, so `ncols=2` is
on. What I never checked is whether the **bench** produced that batch.

It does not. `measure()` calls `e.submit` once and times the resulting ticks:

```python
rid = e.submit(list(range(10, 10 + ctx)), SamplingParams(...))
```

`max_batch=4` is an upper bound, not a batch size. One request in flight means **B=1**:

| B | W=4 rows | rung | `Mk >= 32`? |
|---:|---:|---:|:--:|
| **1** | **4** | **4** | **no — ncols off** |
| 2 | 8 | 8 | no |
| **4** (serving) | **16** | **32** | **yes** |
| 8 | 32 | 32 | yes |

So `TILERL_NCOLS=2` and `TILERL_NCOLS=1` compiled and ran the **identical** 1-column
kernel. The env var was read, the dispatch consulted it, and the row count never reached
the threshold that makes it matter.

## What should have caught it

The flatness itself. **0.995 / 1.000 / 0.988 / 0.996 / 0.998× across five contexts** is
tighter than this harness's own run-to-run noise on *different* code, and I recorded it as
"worst 1.2%, inside the threshold" — reading a null result as a passed test. A/B arms that
agree to 0.5% over a 128× context sweep are usually the same binary, and that reading was
available without any new measurement.

That table now has exactly one legitimate use: it is **the harness's noise floor**, since
it is one kernel measured twice. Worst deviation **1.16%**, spread 1.16 points. So the 2%
threshold committed for the corrected run sits at 1.7× the noise floor, and a real ncols
effect at 16 rows has to clear ~1.2% to be distinguishable at all — which is the number
that makes "the arms agreed too well" a quantitative claim rather than a hunch.

Second time today an A/B compared a kernel against itself. The first was `ncols` passed
positionally onto `abl`
([`errors/2026-09-03-the-ab-measured-abl-not-ncols.md`](2026-09-03-the-ab-measured-abl-not-ncols.md)),
fixed by making the flags keyword-only. That fix was at the right depth for *that* bug and
useless against this one: here the flag arrives correctly and the **shape** of the work
never reaches the gate. A parameter-passing fix cannot protect a threshold that keys on
data.

## Blast radius

Three published numbers, and only one dies:

1. **#36's spec wash — withdrawn.** Never measured. Re-running at B=4.
2. **The cost line `0.670 + 0.5265·W` (#37) — survives, with its axis corrected.** It is a
   B=1 fit, and at B=1 rows *equal* W exactly (2/4/8 rows on the 2/4/8 rungs, no rounding),
   so the four points are internally consistent and the depth predictions (5/5 correct)
   stand. But its `W` is a **chain width, not a launched-row count**, and I had been reading
   it as rows. That misreading is how this was found: pricing the ladder's rungs with the
   line produced a trim that chose W=2 at every acceptance including 0.95, because at B=4
   widths 3 and 4 launch the same 32 rows and cannot have the 2.78 / 4.75 costs the line
   assigns them.
3. **The `ncols` rung gate itself — unaffected.** Its threshold came from dense decode at
   M=1 (−4.9%) and prefill at M=512 (1.52×), both correctly measured, and the gate ships
   `nc = _NCOLS if Mk >= 32 else 1` either way.

A fourth, smaller: I described the W=8 arm as landing on "the 32-row rung, where `ncols=2`
also turns on". At B=1, W=8 is **8 rows on the 8 rung**, and `ncols` was off there too.

## Fix

`--batch N` on `bench_ctx_decode.py`, submitting N concurrent requests, with the reason in
the docstring so the next reader does not have to rediscover that `max_batch` is a ceiling.
Getting it to actually run took three more fixes, each found before it produced a number:
`num_slots` had to exceed `max_batch` because a padded tick permanently keeps one state slot
([`2026-09-03-num-slots-equals-max-batch-is-one-short.md`](2026-09-03-num-slots-equals-max-batch-is-one-short.md)),
`precapture()` had to be called because B=4 needs **12** decode graphs (102 s) against B=1's
4, and `--max-ctx` had to exist because B=4 **OOMs from ctx≥512** on this 32 GB card. The
corrected A/B therefore runs at ctx=32 — which is where the question is sharpest anyway,
since dense decode is fastest there and depth 3 already measured a 12% loss.

The guard is a test, because this failure is silent by construction — a too-narrow tick
produces plausible numbers rather than an error.
`tests/test_e2e.py::test_a_verify_tick_submits_batch_times_width_rows` spies on
`_run_forward` and asserts the widest pure-decode tick exceeds 4 rows with 4 concurrent
requests. **Negative control verified**: with one request the widest tick is exactly 4
rows and the assertion fails, which is the harness this entry is about. `measure()` carries
the same check inline, against `batch * width`.

## Rule

**An A/B whose arms agree to a fraction of a percent has probably not been run.** Two
different kernels do not track each other to 0.5% across a 128× context sweep. Treat
suspicious agreement as suspiciously as a suspicious difference — I have now twice read a
null result as a confirmation, and both times the tell was in the numbers already printed.

Second: **a benchmark's batch size is part of the kernel it selects.** `max_batch=4` on the
engine and 4 rows in the tick are different facts, and on a machine where the GEMV picks a
schedule by row count, the second one is what compiles. Any perf claim about a
row-count-gated path has to state the rows it actually ran.

Third: **fixing a bug class at the right depth does not immunize the next instance.**
Keyword-only flags killed positional-argument A/B failures for good and did nothing here,
because this instance routes through data shape rather than argument order.

## Gate

188 tests pass (the new row-count test included), ruff clean. The B=4 re-run of the
spec-ncols A/B is in flight with its 2% threshold committed in the script before launch;
this entry will carry its numbers.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 659b745 | V100 | cuda sm70 | qwen38-27b | rows in a depth-3 verify tick, B=1 | **4 (rung 4) — ncols=2 off** |
| 2026-09-03 | 659b745 | V100 | cuda sm70 | qwen38-27b | rows in a depth-3 verify tick, B=4 | 16 (rung 32) — ncols=2 on |
| 2026-09-03 | 659b745 | V100 | cuda sm70 | qwen38-27b | **#36's spec-ncols "wash"** | **withdrawn — one path measured twice** |
| 2026-09-03 | 659b745 | V100 | cuda sm70 | qwen38-27b | cost line's variable | **chain width at B=1, not launched rows** |
| 2026-09-03 | 659b745 | V100 | cuda sm70 | qwen38-27b | new test, negative control | B=1 widest tick 4 rows → assertion fails |
| 2026-09-03 | ae268d0 | V100 | cuda sm70 | qwen38-27b | **harness noise floor** (one kernel, twice, 5 ctx) | **worst 1.16% — so the 2% threshold is 1.7× it** |
| 2026-09-03 | ae268d0 | V100 | cuda sm70 | qwen38-27b | B=4 tok/forward @ctx32 vs B=1 | 9.84 vs 2.44 = **2.46 vs 2.44 per request** (batching only) |
| 2026-09-03 | ae268d0 | V100 | cuda sm70 | qwen38-27b | decode graphs to capture, B=4 vs B=1 | **12 (102 s) vs 4** — why the first row read UNWARMED |
| 2026-09-03 | ae268d0 | V100 | cuda sm70 | qwen38-27b | B=4 memory ceiling on 32 GB | **OOMs from ctx≥512**; 1.50 GiB wanted, 0.69 free |
