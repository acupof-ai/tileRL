# Precapture the decode graphs instead of hoping traffic warms them — V100 (sm70), 2026-09-02

> Status: shipped and confirmed on the pod. Warmup **208 s → 19 s** for all 8
> graphs, and the first served request goes **0.9 → 3.23 → 13.62 tok/s** across
> no warmup, a token-generating warmup, and this.

## Context

`serve` never warmed up. `bench_ctx_decode.timed` has warmed since three
benchmarks were ruined by unwarmed readings, but the product surface went
straight from weight load to the first real request — which then paid for the
captures. Measured decode-only, distinct prompts, equal token counts:

| request | ms/token | tok/s |
|---:|---:|---:|
| 0 | 1088.2 | 0.9 |
| 5 | 26.2 | 38.1 |

## What Worked

**Enumerate the grid; do not sample it.** The first fix ran throwaway requests
and generated tokens, on the theory that decode would exercise the widths. It got
part way — `warm in 208s`, then 3.23 → 3.47 → **26.19** → 25.39 tok/s. Requests 1
and 2 still paid 14.0 s and 11.7 s over the plateau, i.e. two captures the warmup
never triggered.

It cannot be fixed by generating more. A speculative tick's chain width is
whatever the draft's confidence leaves after truncation, so no token count
*guarantees* a given width appears; the warmup was a lottery with decent odds and
no floor. `Engine.precapture()` walks `graph_keys()` — every (bucket, width) the
limits admit — and captures each directly. Deterministic, and it needs no
sampling luck.

`serve` prints `N decode graphs in Ms` before its URL. `--no-warmup` opts out.

**The marker was already there.** `_DecodeGraph`'s docstring said
`# ponytail: captured lazily on the first decode tick (first token pays JIT +
capture); capture at engine build is the upgrade` — the upgrade path, named by
whoever wrote the shortcut. It also said it twice, in two consecutive lines; both
are gone now that it is done.

**One key function, not two.** `precapture` and `_run_decode_graph` must agree
about which graphs exist, or warming succeeds, reports N graphs, and a real
request captures anyway. They now share `_graph_bucket(rows)`, so disagreement is
unrepresentable rather than merely tested for.

That last point cost a test. The first version of it recomputed the bucket ladder
inside the test file, and its negative control **passed** — breaking
`precapture` to build only bucket 1 did not fail a test that had its own copy of
the logic. A test that duplicates the code under test verifies nothing. The
shipped version calls `e.graph_keys()` and `e._graph_bucket()`, and its negative
control fails as it should:

    AssertionError: max_batch=2: a 2-row tick keys on (2, 1),
                    which precapture would not build

## Also

`--max-batch` drops 8 → 2 for `serve`. One person does not need eight rows, and
the ceiling is also the size of the grid to capture and hold.

## Rule

**A warmup must enumerate what it warms.** If the thing being warmed is keyed on
a value the sampler chooses, generating more tokens raises the odds and never
reaches certainty. Ask the code what keys exist.

Second, and it is the reason the negative control matters: **a test that
reimplements its subject tests only itself.** The give-away is the one that
should always be run — break the code deliberately and watch the test fail. Mine
passed, which is how I learned the test was decorative.

Third: **when a shortcut carries a `ponytail` marker naming its upgrade, the
marker is the design.** This one said "capture at engine build is the upgrade"
and I spent a round building a probabilistic warmup instead of reading it.

## Results

Four equal-length requests through HTTP, cold server each time:

| warmup | cost | req 1 | req 2 | req 3 | req 4 |
|---|---:|---:|---:|---:|---:|
| none | — | 0.9 | 2.6 | 2.9 | 8.3 |
| generate throwaway tokens | 208 s | 3.23 | 3.47 | 26.19 | 25.39 |
| **precapture (8 graphs)** | **19 s** | **13.62** | 21.20 | 24.29 | 25.45 |

Precapture is **11× cheaper than the warmup it replaces and 4.2× better on the
first request**, which is the number a person actually experiences. It is cheaper
*because* it is direct: generating tokens spent 208 s producing widths it had
already captured, then missed two.

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---:|
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | warmup, 8 graphs | **19 s** |
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | request 1 | **13.62 tok/s** |
| 2026-09-02 | (generate-and-hope) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | warmup / request 1 | 208 s / 3.23 |
| 2026-09-02 | (no warmup) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | request 0 / 5 | 0.9 / 38.1 |

**Still ramping, and it is not graph capture.** 13.62 → 25.45 over four requests
with all 8 graphs already resident. The remaining cost is the fp4 kernel JIT,
which specializes per prefill shape — a different mechanism with a different fix
(bucket the prefill widths, or precompile them too), and worth its own
measurement rather than a guess.

Gate: `tests/test_decode_graph.py::test_graph_keys_covers_what_a_decode_tick_keys_on`
(runs off CUDA — it checks keys, not captures; capture parity is the CUDA test
beside it). Related: `errors/2026-09-02-a-rate-needs-equal-work.md`, which is how
this curve was misread as a regression first.
