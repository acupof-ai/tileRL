# B=8 fills the 32 rung: 1.79× over `serve`'s default max_batch=2, V100 sm70, 2026-09-03

> Status: **A/B closed; default flip blocked on memory.** `B=8` with `ncols=2` reads **75.0 /
> 74.7 tok/s** (two arms, 0.40% apart — inside the 1.16% noise floor) at ctx=32 depth 3,
> against **51.6** with `ncols=1` in the same process: **1.451×**, inside the band committed
> before the run. Against **41.8** — what `tilerl serve --max-batch 2` actually gets — B=8 is
> **1.79×**. `tok/forward` is **17.52 in every arm**, so this is pure kernel time. **Shipped:
> a guard that warns when `B·W` falls between rungs** — the old one only fired past the top,
> leaving the whole 3..7 band silent.

## Context

[The batching sweep](2026-09-03-batching-is-non-monotone-padding-rows-cost-3x.md) found the
tick is dominated by *launched* rows, and that a padding row costs **3.3×** the useful work
layered on it. `max_batch=4` at depth 3 launches 32 rows to do 16 rows of work — the one
config in the sweep whose useful rows are not a rung — so it is the worst point measured.
`B=8` fills the same 32-row launch completely.

The open question was what `ncols=2` is worth once the rung is saturated. It measured
**1.498×** at B=4, where half the rows are padding, and the microbench reads **1.82×** at
M=32. Both bracket were plausible: more useful rows means more arithmetic per byte (argues
up), but padding rows are cheap arithmetic the 2-column kernel also halves (argues down).

**Committed before the run: between 1.0× and 1.5×.** Explicitly not 51.6 × 1.498 = 77 —
that multiplies a measurement by a ratio taken at a different occupancy.

## Results

`bench_ctx_decode.py --depth 3 --batch 8 --max-ctx 32 --tokens 64`, arms run **nc2 / nc1 /
nc2** so drift shows as the two nc2 readings disagreeing rather than as a gain.

| arm | tok/s | ms/token | tok/forward |
|---|---:|---:|---:|
| **B=8 `ncols=2`** | **75.0** | 13.3 | 17.52 |
| B=8 `ncols=1` (in-process control) | **51.6** | 19.4 | **17.52** |
| **B=8 `ncols=2`** (confirm) | **74.7** | 13.4 | 17.52 |
| B=8 `ncols=1`, separate process | 51.6 | 19.4 | 17.52 |

**1.451×** on the two-arm mean, inside the committed band near its top edge. Three
independent checks that the delta is real rather than drift:

- the two `nc2` arms agree to **0.40%**, inside the 1.16% noise floor;
- the `ncols=1` control reads **51.6 / 19.4 / 17.52 — identical to three decimals** to the
  separate-process run, so B=8's baseline has reproduced across two processes and two scripts;
- **`tok/forward` is 17.52 in all four arms** — same tokens, same forwards, 31% less time
  each, so this is kernel time and not acceptance.

## The ncols win is a property of the rung, not of the occupancy

| config | useful / launched | `ncols=2` |
|---|---:|---:|
| dense decode M=1 | 1 / 1 | 0.951× (loss, gated off) |
| spec verify, B=4 | 16 / 32 | **1.498×** |
| spec verify, B=8 | 32 / 32 | **1.451×** |
| prefill M=512 | 512 / 512 | 1.520-1.600× |

The two 32-rung points agree to **3%**, which is inside two arms' worth of the 1.16% noise
floor. So `ncols=2` is worth **~1.45-1.50× on the 32 rung regardless of how much of that
rung is padding** — the win keys on the *compiled* rung, exactly like the
`_NCOLS_MIN_M = 32` gate that ships it.

I predicted the opposite twice and both times the numbers on the line above refuted it:
first that the effect would grow with useful rows (it is flat, and 1.451 < 1.498), then that
B=8 with ncols "can only widen" its lead over a hypothetical 16-rung B=4 — a direction
asserted for an unmeasured quantity one step after writing down that doing so is forbidden.

## The guard that was one band too late

`engine.py` warned when `B·W` **exceeded** the top rung, and said nothing about the far more
expensive case of landing *between* rungs. At depth 3 that silent band is the whole of
`max_batch=3..7`:

| `max_batch` | rows | rung | padding |
|---:|---:|---:|---:|
| 2 | 8 | 8 | — |
| 3 | 12 | 32 | **62%** |
| **4** | **16** | **32** | **50% ← the shipped default** |
| 5 | 20 | 32 | 38% |
| 6 | 24 | 32 | 25% |
| 7 | 28 | 32 | 12% |
| 8 | 32 | 32 | — |

At 7.53 ms per launched row, `max_batch=4` spends **120 ms of every tick** on rows that do
nothing. The guard now warns with the padding count and names the batch that fills the rung;
the past-the-top branch is unchanged.
`tests/test_e2e.py::test_a_batch_between_rungs_warns_about_its_padding` checks the whole 3..7
band lands on 32, keeps two negative controls so the warning never fires on `max_batch=2` or
`8` (the configs it recommends), and asserts the guard's own source. **Negative control
verified**: stubbing the new branch to `elif False` fails the test at its assertion, and
restoring it passes. Pure arithmetic over `LADDER_WIDTHS`, so it runs on the CPU target where
the sm70 dispatch it describes never executes.

## What it means for the default

Checking the code rather than my assumption — and the assumption was wrong. **No shipped
default sits in the padding band**; `B=4` was *my* bench parameter:

| path | default | rows @depth 3 | rung | padding |
|---|---:|---:|---:|---:|
| `tilerl serve --max-batch` | **2** | 8 | 8 | — (but below the ncols gate) |
| `StepLimits` / `build_engine` | **8** | 32 | 32 | — |
| `tilerl generate --max-batch` | **32** | 128 | 4 × 32 | — |
| `bench_ctx_decode.py --batch` | 4 *(mine)* | 16 | 32 | **50%** |

So the **42.7 I twice called "what serving gets today" is a number nothing ships.** The real
serve baseline is `--max-batch 2`: it *fills* the 8 rung, but 8 rows is below
`_NCOLS_MIN_M = 32`, so it gets no `ncols=2` at all.

| config | tok/s | note |
|---|---:|---|
| B=1 | 32.4 | one request |
| **B=2, ncols off** | **41.8** | **what `tilerl serve` ships** |
| B=4, ncols=2 | 42.7 | bench-only; 50% of its launch is padding |
| **B=8, ncols=2** | **74.85** | **1.79× over the serve default** |

Per-request rate at B=8 is 9.4 tok/s; aggregate 2.31× over a single request. The flip is
`serve`'s `--max-batch 2 → 8`, worth **1.79×** — *larger* than the 1.75× I first published,
because the serve default is worse than my bench's B=4, not better.

**Not yet a default flip.** The blocker is memory, not throughput: B=8 peaked at **31110 of
32768 MiB** at ctx=32, and B=4 already OOMs from ctx≥512. A `--max-ctx 2048` sweep is
running. A batch size that only works at ctx=32 is a bench artifact, not a serving default —
and `serve`'s docstring gives a second, independent reason for 2 ("a decode graph is captured
per bucket × chain width, so a lower ceiling is fewer captures"). That cost is countable:
graphs = `{c for c in _GRAPH_BUCKETS if c <= max_batch}` × widths `1..1+depth`, which
reproduces both measured precapture lines exactly (**12 at B=4, 16 at B=8**), so B=2 is **8**.
Raising the ceiling to 8 doubles the captures — **122-155 s of startup** a single-user
endpoint would pay for concurrency it never uses.

## The 16 rung, priced and set aside

`_sm70_chunks` builds the ladder in one expression (`backend.py:51`) and the extern is
templated on M with no upper bound, so adding a 16 rung is a one-line change. Priced with the
plane from the batching entry, B=4's tick would fall 303 → 183 ms, worth **1.11×** over
today's 42.7 — but `_NCOLS_MIN_M = 32` keys on the compiled rung, so a 16 rung would turn
`ncols=2` **off** at B=4 and the gate would have to move with it.

Set aside anyway, because **B=8 measures 75.0 against that predicted 47.3**: the rung fix
helps only the batch size that B=8 dominates, and cannot help B=8 at all (32 useful rows take
the 32 rung either way). Recorded rather than discarded — it is a real 1.11× if B=4 ever has
to ship for a memory reason.

(Working that out, I printed "the 16 rung would LOSE 0.90×" from an inverted ratio, then
narrated the inverted version — 47.3 > 42.7 is a win. The verdict survived for a different
reason than the one I first gave.)

## Rule

**A committed prediction band has to be read against its own edges.** I called 1.451×
"above" a 1.0-1.5 band and built a mechanism story on it in the same breath. The band existed
precisely so the answer would not need interpreting, and I interpreted it anyway.

Second: **when two configurations of the same kernel agree inside noise, that is the finding
— resist the trend.** 1.498 and 1.451 invite a story about occupancy; three percent across
two arms supports none, and "the win is a property of the rung" is both truer and more useful
because it matches what the gate keys on.

Third, and it cost a 20-minute run: **a guard whose false-positive rate scales with the
thing it guards is worse than no guard.** My own row-count check from earlier today asserted
`max(rows) >= batch * width`, which requires all B requests to keep every draft in the *same*
tick — B independent events, since `verify_lens` trims per request. It caught a real bug at
B=4 and then killed a healthy B=8 run at 28 rows. Assert the invariant (B requests decoded
together), not a quantity that legitimate behaviour varies.

## Gate

`tok/forward` identical across all four ncols arms (17.52) — the acceptance control. The
`ncols=1` control reproduced across two processes to three decimals. nc2/nc1/nc2 ordering so
drift is visible. GPU verified idle before launch, `timeout 1500` per arm. Batch spy in
`measure()` now asserts requests rather than rows (`6fbe738`); the old form is what killed the
first ceiling attempt.

**Still open: the context ceiling.** A stepped sweep (32 → 512 → 1024, one process per arm so
a failure at 512 keeps the lower rows) is running.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **spec d3 @ctx32, B=8, ncols=2** | **75.0 / 74.7 tok/s (0.40% apart)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **vs `serve --max-batch 2` (41.8)** | **1.79× — the default-flip number** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | vs bench-only B=4 (42.7) | 1.75× |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | vs one request (32.4) | 2.31× aggregate, 9.4 per-request |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | ncols=2 at B=8, in-process control 51.6 | **1.451×** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | tok/forward, all four arms | **17.52 — identical** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **ncols=2 on the 32 rung, 50% vs 100% full** | **1.498× vs 1.451× — flat to 3%** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | `serve` default vs what I assumed | **2, not 4 — 42.7 ships nowhere** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | decode graphs = buckets ≤ B × widths | 8 / 12 / 16 at B=2 / 4 / 8 |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=8 peak memory @ctx32 | 31110 / 32768 MiB |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=8 context ceiling | **pending — stepped sweep running** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | 16 rung for B=4 (predicted) | 1.11×, **set aside — B=8 dominates** |
| 2026-09-03 | 6fbe738 | V100 | cuda sm70 | qwen38-27b | my row-count guard at B=8 | **false positive at 28 rows — fixed** |
