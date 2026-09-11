# The kernel roofline divides by a measured card, not a datasheet — CPU, 2026-09-11

> Status: Shipped (CPU ledger/arithmetic + cuda measurement path); the first rows land
> when a card returns and runs `tilerl bench --calibrate`.

## Context

`bench --kernels` declared every decode/prefill kernel's bytes and flops, but the ms and
%-of-bound columns printed `pending` because dividing them needs a bandwidth and a compute
ceiling. Substituting a datasheet HBM/peak number is exactly what earlier entries warned
against (an assumed denominator turns a derived byte number into a false % of bound). The
floor has to be measured on THIS card.

## What worked

`tilerl bench --calibrate --card N` measures two ceilings with CUDA events and appends
one ledger row each to the existing bench store
(`docs/experience/bench/measurements.jsonl`, `$TILERL_BENCH_STORE` override):

- **HBM bandwidth** — a ≥1 GiB device-to-device copy, read+write bytes over the event
  median (the same HBM-direction rule as the kernel byte ledger). Metric `hbm_bw_gbs`.
- **bf16 tensor peak** — one large square bf16 GEMM, `2n³` flops over the median. Metric
  `bf16_peak_tflops`.

`bench --kernels` then reads the newest non-superseded pair keyed on the **exact device
name** and fills the `bound` column with `max(bytes/bw, flops/peak)`; on cuda the ms column
times the actual registry GEMM kernel and %bound is measured, off cuda both stay
`pending-remote`. The command refuses off cuda and prints the card command, rather than
emitting a number.

The testable core is cuda-free and exact: `bound_seconds` is pure arithmetic; the floor
lookup keys on full device name (a V100 row is not an H20 floor); a missing or
half-present calibration renders pending-remote, never a fallback. Gates pin the bound to
an exact fixture row, the name join to refuse a different/typo'd card, and the empty
store to print pending.

## Rule

A roofline % is only as real as its denominator: measure bandwidth and peak on the card
(large copy + large GEMM, CUDA-event median), key the floor to the exact device name, and
render pending-remote — never a datasheet number — when the row is absent. The bound
arithmetic stays CPU-pure so the join and the division are gated without a GPU.

## Results — first measured H20 floors (cards 0 and 1, 2026-09-11)

`bench --calibrate` at commit `a08e546ad836246de272d599e557d56d0aadbb87`, four rows
in `measurements.jsonl` (ids `1570734f7f15`, `c5037c831df7`, `619d9cdd0d0e`,
`e627533d787c`). Under one visible device the in-process index is always card 0
(`CUDA_VISIBLE_DEVICES` masks it), so the row's `shape.card` reads 0 for both; the
physical card is fixed by the append order (the `CUDA_VISIBLE_DEVICES=0` run, then
the `=1` run):

| physical card | HBM GB/s | bf16 TFLOP/s |
|---:|---:|---:|
| 0 | 3292.07 | 136.39 |
| 1 | 3292.39 | 137.76 |

Placement control: two idle H20s agree to **0.01% on bandwidth** and **1.0% on
bf16 peak**, so a roofline divided by either card is the same number; cards 2/3
(separate calibration PRs) read 3316/3311 GB/s and ~136.5 TFLOP/s — the same
device population under the exact name "NVIDIA H20". The store is append-only;
`latest_floor` resolves the floor to the NEWEST non-superseded row for that
exact device name (not the max), so the row order across the calibration PRs
sets which same-name measurement the roofline divides by.

