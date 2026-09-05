# GRPO phase timing — CPU, 2026-09-05

> Status: pending-remote (CPU correctness passed; GPU timing unmeasured)

## Context

GRPO already yielded each step and the CLI flushed each log line. The step
seconds did not distinguish sampling, tape work and optimizer updates.

## What Worked

Each yielded step now carries `rollout_secs`, `backward_secs` and
`optimizer_secs`. The CLI prints them immediately and saves their totals beside
`secs_total` and the existing median in the run manifest.

`rollout_secs` covers prompt preparation, submission and engine sampling.
`backward_secs` includes the recorded forward, loss and gradient accumulation;
`optimizer_secs` covers clipping and updates, including streamed updates.
Reward evaluation, batch packing and cache invalidation remain in step seconds
but outside these phases. All clocks use `time.perf_counter`; GPU work is not
synchronized, so these are host wall intervals, not GPU kernel durations.

The CPU tiny smoke recipe ran 12 steps. Phase totals were 1.270699 s rollout,
1.784430 s backward and 0.006222 s optimizer against 3.063391 s total step time
(99.9334% coverage). This is an accounting check, not a throughput comparison.
The manifest test fails on base `91977a8` with `KeyError: 'rollout_secs'`.

## Rule

Compare phase totals with summed step seconds, not the per-step median.

## Results

| date | base commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-05 | 91977a8 | local Mac | cpu | tiny | n/a | n/a | n/a |

Raw artifacts: `/private/tmp/codex-timing-runs/e06276f730b2/manifest.json`,
`/private/tmp/codex-timing-live.log`, `/private/tmp/codex-timing-red.log`.
