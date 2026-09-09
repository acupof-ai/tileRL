# A timing-dependent test goes red under xdist — 2026-09-09

## Context

`test_the_tier_read_rate_keeps_moving_after_the_first_fetch` guards that
`read_bytes_per_s()` divides running totals (the counters accumulate, it
divides), so a warm first fetch cannot permanently over-permit prefixes. The
original implementation asserted a rate ratio: do a fast fetch, do a second
fetch with a 50 ms `torch.load` delay, assert `second < first / 2`.

Under xdist (`-n auto`, 8+ workers hitting the same disk), the first fetch is
already slow. The CI failure on #378:

```
AssertionError: B barely moved on a fetch made 50 ms slower (0.0 -> 0.1 MB/s)
```

The mocked-slow fetch was *faster* than the contended baseline. The 50 ms
delay, significant against a serial disk, is noise against a disk eight
workers are already loading.

## Root cause

The test used the environment's natural speed as its control. The property
it guards — "B is cumulative" — is algebraic: the counters accumulate and B
divides the running totals. Writing it as a temporal assertion (rate dropped
by 2x) made the test depend on a quantity (disk speed) the test does not
control.

## Fix

Replace the rate-ratio assertion with algebraic assertions:

- `fetch_ms > fast_ms` and `fetch_bytes > fast_bytes` — the counters
  accumulated across fetches.
- `second == fetch_bytes / (fetch_ms / 1000)` — B divides the running totals,
  not a cached first-fetch value.

The 50 ms delay is kept (it makes the second fetch's contribution distinct)
but no assertion depends on the absolute rate. Both mutants verified red:

| mutant | caught by |
|---|---|
| frozen counter (`+=` → `=`) | `fetch_ms > fast_ms` |
| frozen B (cache at first call) | `second == fetch_bytes / (fetch_ms / 1000)` |

## Rule

Parallelism does not create bugs; it turns timing-dependent tests from
occasionally correct into frequently wrong. A test that fails under
`-n auto` was only accidentally green serially. Test the algebraic property,
not the environmental timing.
