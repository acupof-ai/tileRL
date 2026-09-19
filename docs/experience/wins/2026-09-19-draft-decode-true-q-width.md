# A decode-only draft step keeps its true query width — pending-remote, 2026-09-19

> Status: pending-remote. The T-shape change and its CPU parity are measured
> (CPU target); the served sm70 gain is not measured and is the point of the PR.

## Context

The draft head's `step` rounded its forward width `w` up to `_PREFILL_BUCKET`
(64) for every non-unit `w`. That bucket exists for chunked PREFILL: an
unbucketed widening prompt length gave the draft a new `(n, w)` per prompt and
recompiled the two `seq_q_lens` kernels — 14 compiles / 15.5 s inline on a served
first visit, 4.4 tok/s against 45.0 on the repeat (spec.py's own comment on the
rule).

A spec decode tick is not that shape. At d1 its row carries q = n_ok + 1 = 2 on
an accept (the common tick) and 1 on a reject, and it carries no prefill at all,
so the bucket protects it from nothing while costing it 32x:

- **sm70 attention.** `paged_attention_split` carries both the history split and
  **S in its grid** (`kernels.py:970`, `T.Kernel(KVSPLIT, S * H, B)`). Its work is
  S query rows x n history each, so T=64 over q=2 does 32x the work of rows whose
  output is never read. The grid drops 16x, halved again because `sm70_kvsplit`
  raises KVSPLIT 16 -> 32 below S=8 — a win at these widths, so the net grid
  effect is not the 32x the row count alone suggests.
- **Projections.** fc / gate / up / down run at M = B*T, so B*64 -> B*q.

This is the `[S]` lever from the 2026-09-18 root-cause synthesis (附录 B 第 1 条),
scoped to its lowest-risk form: a host-side shape decision, no kernel change.

## What Worked

`TILERL_DRAFT_TRUE_Q_WIDTH` (default **off**). When on, a plan in which every row
is a decode-phase verify tail takes T = max(q) instead of the bucket.

The predicate is the one `_windowed_read_kv` already applies, extracted to a
module-level `draft_step_is_decode_only(sq, width, decode=None)` and **called from
both sites** rather than restated. The two have opposite failure modes and must
not drift: a narrow forward with a full-prefix window is a silent numerics change,
and a bucketed forward with a truncated window is a wasted launch. Two exclusions,
both required: a row still PREFILLING (`decode=False`) and a decode-phase
CATCH-UP row (q > `width`, the hidden-gap case after chunked prefill advanced
without drafting) keep the bucket.

Measured on CPU (`tests/test_draft_q_width.py`, forced-acceptance so verify ticks
really carry q=2):

| arm | T histogram over one served run |
|---|---|
| flag off | `{64: 12}` |
| flag on | `{2: 11, 64: 1}` |

The one surviving 64 is a catch-up row (q=13 > width=2); the prefill tick
(q=127) also keeps it. Both exclusions are exercised by the run, not just by the
unit table. Generated tokens are **identical** across the two arms (24/24) — the
padded columns were never read, so this is a launch-shape change and not a
numerics change.

Negative controls: with the flag forced off the gate is red on its own positive
control; with the predicate forced true (exclusions removed) it is red on the
exclusion assertions.

## Rule

A prefill-shape bucket must not be applied to a decode tick. When a width decision
and a read-view decision depend on the same row classification, they must call one
function — two restatements of "is this a decode-only tail" is exactly the pair
that diverges silently.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| | | | | | | | |

Raw artifacts: pending the next V100 window. The A/B is flag on vs off on the same
serve command (`TILERL_DRAFT_TRUE_Q_WIDTH=1`), reading `draft_step` GPU ms at
9k/16k/32k from `TILERL_STEP_TIMING`; the predicted effect is on the 32k
draft_step term (~100 ms, 38% of the tick) whose measured slope is 3.57-3.64
µs/token, in-kernel.
