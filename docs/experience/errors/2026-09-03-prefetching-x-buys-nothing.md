# Prefetching X buys nothing: the cost is the loads, not their latency — V100, 2026-09-03

> Status: **PIPELINE rejected at 1.00×** (threshold ≥1.15×, committed in the script
> before the run, relerr 0 as required). This is the measurement that splits
> `X_REUSE`'s 8.75×, and it lands on traffic + issue slots, not on latency exposure.

## Context

`X_REUSE` (every row reads row 0's X) was 8.75× at M=32, but it changed three things
at once — X's traffic, X's latency, and, because a loop-invariant address is
hoistable, **85% of the LDG instructions** (SASS: 363 → 53). So it bounds X-related
cost from above without saying which part to attack, and SMEM staging only removes
traffic.

`abl=4` PIPELINE separates them. It issues row m+1's two loads *before* row m's FMA
block, so the loads are in flight across arithmetic. Same addresses, same count —
verified: still 363 LDG — and unlike `abl=1..3` it only reorders, so its output must
be bit-exact and the harness checks `relerr`.

## Results

`scripts/ab_gemv_ablate.py`. No relerr warning fired, so PIPELINE is bit-exact.

| shape | M | X_REUSE | PIPELINE | of X_REUSE's headroom |
|---|---:|---:|---:|---:|
| gate_up | 8 | 3.61× | 1.01× | 0.4% |
| down | 8 | 3.71× | 1.00× | 0.0% |
| qkvz | 8 | 3.28× | 1.00× | 0.0% |
| gdn out | 8 | 1.57× | 1.01× | 1.8% |
| qkv | 8 | 3.27× | 1.00× | 0.0% |
| attn o | 8 | 1.54× | 1.00× | 0.0% |
| gate_up | 32 | 9.05× | 0.99× | −0.1% |
| down | 32 | 10.69× | 1.00× | 0.0% |
| qkvz | 32 | 8.74× | 0.99× | −0.1% |

**Prefetching recovers 0.0-1.8% of the headroom, and ~0% at M=32.**

Per-pass totals weighted by launches/token, with the previous run's X_REUSE beside
it — the two runs bracket the noise floor:

| M | base | X_REUSE | NO_SCALE | NO_DECODE | **PIPELINE** |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.3 ms | 1.01× (was 0.97×) | 1.07× | 0.94× | 1.02× |
| 8 | 73.2 | 3.19× (was 3.05×) | 0.99× | 0.99× | **1.00×** |
| 32 | 271.0 | 8.80× (was 8.75×) | 0.92× | 0.98× | **0.99×** |

The M=1 row moves 0.97 → 1.01× between runs on the same kernel, so **±4% is the
noise floor there** and no sub-1.05× M=1 reading in this family of experiments means
anything. M=8 and M=32 reproduce to 2% and 0.6%, which is what makes the 1.00×
PIPELINE result trustworthy at the M values that matter.

## What it decides

**The cost is the loads themselves, not their exposure.** If X's per-row loads were
stalling on latency, putting them in flight across eight FMAs would recover a real
share of 9×. It recovers none. So the 8.75× is X's *traffic* plus the *issue slots*
its 310 extra LDGs occupy — both of which scale with the number of loads, and
neither of which reordering can touch.

**That makes SMEM staging the right shape of fix after all** — for the traffic half.
Staging the tile's X slice once per block and reading it from SMEM removes global
loads rather than rescheduling them. The honest caveat: SMEM has the same 128
B/cycle/SM port as L1, so staging converts global-load instructions into shared-load
instructions and only wins on the traffic and on L2/L1 pressure, not on the issue
slots. The `X_REUSE` bound (8.75×) is above what staging can deliver, and the LDG
arithmetic says how much: staging replaces 310 of 363 LDG with LDS, so the ceiling
is whatever those 310 global loads cost *minus* what 310 shared loads cost.

**A second, cheaper lever the numbers now favour:** load X once per tile into
registers shared across the m loop is impossible (that is what the 150× register
cliff forbids), but `n_partition` currently makes 4 column-groups in a block each
issue their own X loads for the *same* X — 4× redundant global traffic that a single
SMEM stage would collapse. That is the same fix, and it is the reason it should pay
more than "X fits L2 anyway" suggests.

## Rule

**Reordering and removing are different fixes, and an ablation that does both cannot
choose between them.** `X_REUSE` deleted loads *and* cached them *and* hoisted them;
its 8.75× was a ceiling with three mechanisms inside. One more variant — same
addresses, same count, different schedule — cost one pod run and cut the space in
half.

Second: **a bit-exact reorder that measures 1.00× is a load-bound proof.** Latency
hiding is the cheapest thing a scheduler can do for you; when it does nothing, the
constraint is throughput, not latency. That is worth more than the 1.00× looks,
because it is what makes staging worth writing after two entries of guessing.

## Gate

`abl=4` is bit-exact and the harness asserts it (`relerr` printed per shape, warning
on nonzero). It stays a factory flag defaulting to 0; 182 tests unchanged.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 63bab34 | V100 | cuda sm70 | GEMV M=32 | PIPELINE (abl=4) | **0.99-1.00× — reject** |
| 2026-09-03 | 63bab34 | V100 | cuda sm70 | GEMV M=8 | PIPELINE (abl=4) | 1.00-1.01× |
| 2026-09-03 | 63bab34 | V100 | cuda sm70 | GEMV M=32 | headroom recovered | **~0%** |
| 2026-09-03 | 63bab34 | V100 | cuda sm70 | GEMV | PIPELINE relerr | 0 (bit-exact) |
