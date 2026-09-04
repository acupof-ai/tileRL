# The post-precapture "ramp" was prompt variance — V100 (sm70), 2026-09-02

> Status: no defect, and the JIT hypothesis was refuted before it cost any pod
> time. With graphs precaptured, decode is **flat from request 0** — 32.9 / 31.8 /
> 24.4 / 31.2 / 39.4 ms/token, reserved memory unchanged, zero allocator retries.

## Context

After `precapture` landed, served throughput still read 13.6 → 21.2 → 24.3 →
25.5 tok/s over four HTTP requests. I opened a task for it with a named suspect:
the fp4 kernels JIT per prefill shape, so the first requests would compile
widths `precapture` never touches.

## Root Cause

**The suspect was refutable from a log already on the pod.** `grep -c "begins to
compile" /tmp/serve.out` returns **3**, and all three are timestamped 21:15:25-32
— inside precapture, before the first request. Zero compiles during requests 1-4.
Whatever the ramp was, it was not JIT.

Instrumenting properly (decode-only windows, distinct prompts, equal token
counts, torch's own allocator counters) leaves nothing to explain:

| request | ms/token | tok/s | reserved MiB | Δ reserved | retries |
|---:|---:|---:|---:|---:|---:|
| 0 | 32.9 | 30.4 | 28062 | 6 | 0 |
| 1 | 31.8 | 31.4 | 28064 | 2 | 0 |
| 2 | 24.4 | 40.9 | 28064 | 0 | 0 |
| 3 | 31.2 | 32.0 | 28064 | 0 | 0 |
| 4 | 39.4 | 25.4 | 28064 | 0 | 0 |

Request 0 is already at 30.4 tok/s. The spread is ±23% with no trend, and it goes
*both ways* — request 2 is the fastest of the five and request 4 the slowest. That
is speculation acceptance varying with the prompt (tok/forward moves 1.6-3.8
across prompts), not a warmup curve.

Reserved memory moves 6 MiB total and `num_alloc_retries` stays 0, so the
allocator hypothesis dies on the same table.

The HTTP numbers that started this were measuring something real but different:
each of those requests included its own prefill, and the *first* prefill in a
process pays kernel-cache loads and buffer allocation that no later one repeats.
Burning the prefill before timing — which is what `bench_ctx_decode` has always
done — removes it.

## Rule

**Grep the log before writing the profiler.** The compile-count refutation took
one command against a file that was already sitting on the pod from the previous
run. I wrote a 100-line attribution script first and only checked the log while
it was running.

Second: **a monotone-looking sequence of four points is not a trend.** Five points
of the same measurement span 24.4-39.4 in no order at all. Before attributing a
ramp, establish the noise floor — repeat the *same* condition and see how wide it
is.

Third, and this one is now twice in two days on the same investigation: **make the
window contain only what you are attributing.** Prefill inside a decode
measurement produced both the phantom 6.5× gap and this phantom ramp.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---:|
| 2026-09-02 | 5a6a824 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | request 0, precaptured | **32.9 ms/tok** |
| 2026-09-02 | 5a6a824 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | requests 0-4 spread | 24.4-39.4 |
| 2026-09-02 | 5a6a824 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | kernel compiles, req 1-4 | **0** |
| 2026-09-02 | 5a6a824 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | precapture, warm cache | 6 s |

Precapture with a warm `TILELANG_CACHE_DIR` costs **6 s**, not the 19 s measured
on a colder one — the 8 graphs are all that is left to build.

Raw artifact: `scripts/prof_serve_ramp.py`. Related:
`wins/2026-09-02-precapture-the-decode-graphs.md` (the fix this was probing) and
`errors/2026-09-02-a-rate-needs-equal-work.md` (the same window error, first
occurrence).
