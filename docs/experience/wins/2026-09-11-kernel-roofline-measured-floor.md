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
