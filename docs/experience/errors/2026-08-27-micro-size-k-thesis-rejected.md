# The micro_size_k load-width thesis is dead — measured on H20, 2026-08-27

> Status: REJECTED by same-process sweep on the pod. The lever the whole
> tilelang-vs-native strategy brief was built on makes the GEMV **slower**,
> not faster. Both kill-criterion gates fail.

## Context

The brief (`docs/analysis/2026-08-27-tilelang-vs-native.md`) named
`micro_size_k = 8 → 32` as "the main event": the fp4 decode GEMV issues
`LDG.32` (4 B/thread) where its bf16/fp8 siblings issue `LDG.128` (16 B), so
widening the load was predicted to buy 1.5–1.85× on the GEMV. The kill
criterion: sweep `(micro_size_k, GROUP)` on `gate_up` (34816×5120) and `down`
(5120×17408) at M=1, same process; Gate A = any arm ≥1.05× the shipped (8,4)
on **both** shapes.

## What was measured (H20, cuda/sm90, mean of 50 iters/arm, same process)

| arm | down TB/s | vs (8,4) | gate_up TB/s | vs (8,4) |
|---|---|---|---|---|
| **micro=8 GROUP=4 (shipped)** | **1.483** | 1.000× | **1.696** | 1.000× |
| micro=16 GROUP=2 | 1.335 | 0.90× | 1.495 | 0.88× |
| micro=16 GROUP=1 | 1.246 | 0.84× | 1.466 | 0.87× |
| micro=32 GROUP=1 | 0.972 | 0.66× | 1.051 | 0.62× |
| micro=32 GROUP=4 | 0.981 | 0.66× | 1.055 | 0.62× |

- **Gate A: FAIL.** No arm beats the shipped (8,4) on either shape. Wider loads
  are monotonically **slower**: micro=16 loses ~10%, micro=32 loses ~35%.
- **Gate B: FAIL.** Best down_proj (the shipped arm) is 45.1 µs vs Marlin's
  38.9 µs = 1.16×. `micro` cannot touch the 6 µs gap — it is the f32
  block-scale overhead (0.1875 B/elem Marlin does not pay), as the brief
  predicted.

## Root cause of the wrong prediction

The Mac codegen saw the load **width** (`LDG.32` vs `LDG.128`) but not the
**occupancy cost**. A wider micro-tile spends more registers per thread and
runs fewer resident blocks; on H20 the net is a loss. This is the mirror image
of [[read-source-before-benchmarking]] — codegen reveals instruction selection,
it says nothing about occupancy, and occupancy was the deciding variable here.
The shipped (8,4) was already the tuned point; the "tested worse" note struck
from the docstring was, in fact, correct.

## What this changes

1. **The load-width lever is closed. Do not reopen it.** micro_size_k stays 8.
   The sweep knobs on `make_linear_fp4_gemv` can stay (harmless, default
   unchanged) but no config above (8,4) ships.
2. **scale dtype f32 → bf16 is now the #1 GEMV lever.** At block 16 the f32
   scale is 0.1875 B/elem = a quarter of the 0.75 B/elem fp4 stream; it is also
   exactly Gate B's 6 µs gap. This was recommendation 1 in the brief; it is now
   the head of the queue.
3. **The register-resident SR (Marlin-shaped) kernel is now the main path, not
   a fallback.** The CAN verdict stands
   (`docs/analysis/2026-08-27-tilelang-vs-native.md`); it is the only route
   to Marlin's 38.9 µs, and the sweep proved there is no cheaper one inside the
   current kernel.

## Rule

A load-width change is an occupancy change. Measure occupancy (registers/thread
× resident blocks), not just the emitted load instruction, before predicting a
GEMV speedup. Codegen on a GPU-less host cannot see it.
