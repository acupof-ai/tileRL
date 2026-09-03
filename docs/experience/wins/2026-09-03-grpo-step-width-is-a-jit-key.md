# The GRPO batch width is a JIT cache key — cpu, 2026-09-03

> Status: Shipped (cpu and sm90)

## Context

`uv run tilerl train --recipe grpo-tiny-smoke` was the designated CPU smoke
recipe and it exited 1 on every run. Three separate problems sat behind that one
symptom, and the expensive one had nothing to do with the gate: `grpo_loop` sized
its training rectangle from the rollout's own output, so the shape TileLang
compiles for changed whenever a completion length changed.

The metric that matters is seconds per training step, and the failure mode is
that it is not a constant — it is ~0.2 s or ~37 s depending on whether this
step's width has been compiled before.

## What Worked

**One change:** `train.py` `grpo_loop` padded to `max(len(c) for c in comps)`;
it now pads to `max(sampling.max_new_tokens, max(len(c)))`, a width known before
the rollout starts. `seq_lens` already carried each row's true length and the
advantage mask already zeroed padding, so the update is unchanged — only the
compiled shape is.

Measured on the tiny model, warm cache, `rl_step` at group 6, one call per width:

| arm | widths exercised | total | per call |
|---|---|---:|---:|
| data-dependent (old) | 8 distinct | **239.2 s** | 35.8 / 34.8 / 36.9 / 33.1 / 37.9 / 16.8 / 39.4 / 4.5 s |
| fixed (new) | 1 | **0.6 s** | 0.08 s × 8 |

**398×** on that workload. Isolated, a single new width costs 37.7 s against
71 ms for a width already compiled — a 530× penalty paid per novel length.

Two smaller fixes in the same pass, both measured:

- `clip_grad_norm` replaced its per-gradient f64 loop with one
  `torch._foreach_norm` plus one f64 reduction: **141 → 17.5 ms** at 27B-LoRA
  scale (1092 f32 adapters), norm agreeing to 7.7e-07 relative. The documented
  one-sync property is preserved. `_foreach_norm` squares in the input dtype, so
  anything narrower than f32 is widened first — raw bf16 loses 1.3 decimal
  digits, raw f16 loses 0.9.
- The recipe itself was mis-parameterized: at `max_new_tokens=4` every group
  ties (12/12, whatever the reward's shape — see
  `errors/2026-09-03-tied-groups-are-the-rewards-shape.md`), and 2 steps of a
  per-step reward compares two draws rather than two policies. At steps 12,
  group 6, `max_new_tokens` 8, lr 0.05 the recipe passes its own gates on 4 of 4
  seeds with `tied_group_fraction` 0. `tests/test_recipes.py` now asserts
  `gates_pass`; zeroing `group_advantages` makes it fail, so the assertion has a
  negative control.

## Rule

A shape that a rollout decides is a JIT cache key. Pad training batches to a
width the *configuration* fixes, never to one the data chooses — and read
per-step wall clock as a distribution, because a mean over a run that compiled
once hides a 500× tail.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-03 | pending | Mac (M-series, no GPU) | cpu | tiny | n/a | n/a | n/a |
| 2026-09-03 | pending | H20 card 6 | cuda sm90 | qwen38-27b | n/a | n/a | n/a |

**sm90 constant, measured (n=5, private cache, LoRA r16, group 8, T≈384):** a
novel width costs **11.05 s on a 22.23 s `rl_step`, ratio 1.5x** -- novel
34.7/33.3/33.8/31.2/31.0 s against repeats 22.1/22.0/22.4/23.0/22.2 s. Cold
warm-up alone is 170.9 s. So the mechanism transfers and the constant does not:
tiny's 530x is an artifact of a 71 ms step, where compile time is the whole
measurement. Two earlier runs of the same probe reported 1.3x and 1.0x from a
shared `TILELANG_CACHE_DIR` that labelled cache hits `novel`
(errors/2026-09-03-probe-rebuilt-the-setup-in-the-wrong-order.md).

This also bounds what the fix is worth against the P1 entry's
`secs_per_step_median = 60.45` at gen 256: `rl_step` is ~22 s of that, leaving
~38 s of rollout, and compile can add 11 s only to the steps that hit a new
width. **The median step carried no surcharge**, so the recorded 60.45 describes
the step rather than the harness -- the opposite of what the tiny number
suggested.

Training-step timings, not serving throughput — the serving columns do not apply
to this entry. The sm90 constant is now measured (table above); it is 1.5x, not
tiny's 530x, because the 27B step has 22 s of real compute for the compile to be
a surcharge on.

Raw artifacts: measured inline in this session; reproduce with
`TILERL_TARGET=cpu uv run tilerl train --recipe grpo-tiny-smoke --force` (12
steps, expect one warmup step then a flat tail).
