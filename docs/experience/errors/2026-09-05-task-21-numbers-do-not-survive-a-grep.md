# Task #21's numbers do not survive a grep, and the task was already withdrawn — 2026-09-05

**Date:** 2026-09-05
**Arch:** sm70 (Tesla V100-SXM2-32GB) — no run made; this is an archive check
**Task:** #21, sm70 GEMV small-N shapes

## What #21 asks for

Three figures, carried in the task description: `attn o` at N=1024 running at **5% of
peak**, `gdn out`/`gdn z` at **32%**, and **144 GEMV launches/token**. The instruction is
to measure in the captured graph first, because the microbench has a ~60 µs eager launch
floor the graph path does not pay, and to find out how much of the 5% is real before
touching `n_partition` or fusing shapes.

Before running anything I looked for where those three numbers came from. None of them
survives.

## The two levers were dropped three days ago

`errors/2026-09-02-per-shape-gap-was-a-wrong-shape-table.md:62-69` retires both by name:

> - **Raise `n_partition`** — the shape it was aimed at (N=1024) is not launched. The
>   smallest real shape is `gdn ab` at N=96, and it is 11.8 MB/token, 0.1% of the stream.
>   Nothing to win.
> - **Fuse the small shapes** — `gdn out`/`gdn z` were the target. `gdn z` is ALREADY
>   fused into `qkvz`; the table listed the pre-fusion members.

So `N=1024` is not a shape this engine launches, and the second lever's target was fused
before the lever was written. The 5% and the 32% are per-row figures from a shape table
whose `hq*d` used a head_dim of 64 instead of 256 and whose count of 32 double-counted a
per-layer projection that exists 16 times. The table passed its own assert
(`abs(nib_tot/1e9 - 12.81) < 0.02`, at 12.799 GB) because the per-row errors cancel —
`attn o` 168 MB short against `gdn out` 252 MB long — and every conclusion drawn from it
was about the distribution the assert cannot see.

## 144 launches/token contradicts both in-graph measurements by 2.1-2.3x

| source | figure | instrument |
|---|---:|---|
| #21's description | 144 /token | none found |
| `errors/2026-09-02-…-wrong-shape-table.md:75` | **305** /token | graph profile, `linear_fp4_gemv_sm70_m_kernel` |
| `wins/2026-09-04-the-rung-step-is-93-percent-gemv.md:34` | **313.4** /forward (d1) | `prof_decode_budget.py`, printed beside the per-class ms |

The two measured counts are independent — different day, different script — and agree to
2.8%. 144 disagrees with both by 2.1-2.3x, past the 2x line at which the instrument is the
suspect rather than the system.

**144 does appear in the archive, twice, and neither instance is a launch count:**

- `wins/2026-09-04-depth-1-wins-and-block-parallel-is-rejected.md:34` — `r4 x144` is the
  number of **ticks** that landed on rung 4 in the depth-3 row.
- the same file `:329` — `ceil(2176/16)·1 + 8 = 144` is the **KV block count** `NB`, a
  shape parameter.

And the one place it was used as a launch count is already withdrawn:
`errors/2026-09-03-batching-is-non-monotone-padding-rows-cost-3x.md:130` lists under *What
this withdraws* — "At 144 launches/token and a tiny KV the tick looked launch-dominated" —
retracted because a marginal launched row costs 7.53 ms, so rows are the bill, not
launches.

## Verdict

**#21 is closed as already-answered, with no run.** Both levers were priced against a 42%
gap that does not exist; the shape they aimed at is not launched; and the launch count in
its premise is 2.1-2.3x off two agreeing in-graph measurements and traces to a figure that
was withdrawn two days before the task was restated.

The V100 could not have run it this tick anyway — the resident 27B server holds 22.3 of
32 GB and the sweep needs ~20 GB — but that is not why this is closed.

**What the same profile did find, and what should carry #21's number if anything does**
(`errors/2026-09-02-…:73-84`): **305 f32→f16 casts of X, 1.64 ms/token, 5.9% of the
token**, one per GEMV launch, from `backend.py:554`'s `.to(torch.float16)` inside the chunk
loop. The bytes are 14.5 MB/token = 0.016 ms at peak, so 1.64 ms is **102x the byte cost**
— a launch-count problem with a real number attached, which is what #21 was meant to be.
That was filed separately and is the live descendant.

## Rule

**Check the premise's provenance before building the instrument to test it.** The
instruction here was methodologically right — measure in the graph, distrust the eager
launch floor — and would have produced a careful measurement of a shape that is never
launched. A grep for each figure cost minutes; the run would have cost a GPU window and a
service outage.

**A number restated in a task description is not a measurement.** All three of #21's
figures had been superseded in `docs/experience/`, two of them explicitly under a *What
this withdraws* heading. The task text was the only place they still stood, and a task
description does not carry an instrument.

**When a figure disagrees with a measurement by more than 2x, grep for the figure before
re-deriving it.** 144 was in the tree three times — twice as a different quantity
entirely, once as a retraction — and no time as a live launch count.
