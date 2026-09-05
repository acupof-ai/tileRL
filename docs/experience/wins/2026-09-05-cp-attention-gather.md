# CP attention: the gather's backward is a sum, not a slice — cpu, 2026-09-05

> Status: Shipped (attention half; CP unusable on the 27B until GDN lands)

## Context

First context-parallel op: split the sequence across ranks, all-gather K and V
per full-attention layer. `docs/design-parallel.md` specified it as
"`reduce_scatter`-free since the backward of a gather is a slice", which would
have made the backward a local chunk read with no collective at all.

## What Worked

**The design's backward was wrong, and measuring it first cost one probe.** The
slice rule holds for the vocab-parallel `lm_head` gather, where each rank owns a
distinct slice of the *output*. Under CP every rank's queries read every rank's
keys, so each rank's dK is a partial sum over the whole sequence. Against a dense
single-rank reference:

| cp | slice only | sum across ranks | dQ (local) | scale |
|---:|---:|---:|---:|---:|
| 2 | 0.649 | 3.0e-08 | — | \|dK\| 0.46 |
| 4 | 1.032 | 2.4e-07 | 3.0e-07 | \|dK\| 1.78 |

Slicing is off by 58% of full scale at cp=4. The direct evidence is which
positions each rank writes dK to: contiguous chunks at cp=4 give rank 0 positions
0..7 and rank 3 positions 0..31, so a slice keeps rank r's own contribution to
its own chunk and discards the other `cp-1-r` ranks'. Rank `cp-1`'s chunk is the
one case where the slice is right, which is how a spot check passes and hides it.

**Chunks are assigned zigzag.** A causal mask makes contiguous chunking lopsided
— rank r scores against chunks 0..r — and the gather is a barrier, so the slowest
rank is the step time. Rank r takes chunks `r` and `2cp-1-r`. Scored q·k pairs
per rank at T=32:

| assignment | cp=2 | cp=4 |
|---|---|---|
| contiguous | [10, 26] (2.6x) | [36, 100, 164, 228] (**6.3x**) |
| zigzag | [18, 18] (1.0x) | [132, 132, 132, 132] (**1.0x**) |

That reorders the gathered K/V into **rank order, not sequence order**, so the
mask can no longer come from a tensor index. `dense_attention` takes `q_pos` and
`k_pos`; the `triu` it used assumed q and k are the same length, which stops
being true the moment the queries are one chunk.

**Gate** `tests/cp_world2.py`, two gloo ranks, forward and all three gradients
against the unsplit reference: **3.0e-07 / 4.8e-07 / 4.8e-07 / 3.6e-07** on
out/dQ/dK/dV. Two controls, each failing with its own signature rather than
generically:

| control | out | dQ | dK | dV |
|---|---:|---:|---:|---:|
| `--slice-bwd` (the doc's claim) | 3.0e-07 | 4.8e-07 | **2.216** | **2.156** |
| `--seq-mask` (mask by index) | **2.941** | **2.306** | **1.449** | **3.069** |

`--slice-bwd` leaving the forward and dQ exact is what distinguishes this defect
from a broken harness.

**Doc arithmetic recomputed, not adjusted.** Counting the backward doubles the
bytes and the calls, but for *both* schemes, so the ring-vs-all-gather ratio is
unchanged (cp=8: 224 MiB, ring 4816 µs vs all-gather 688 µs, still 7x). Memory
moves instead: +12.5% flat, because the gathered K/V are retained for all 16
layers while dK/dV live one layer at a time. The crossover goes from ~64K to
**~58K** tokens (58254 at a 4 GiB budget, B=1).

## Not established

- **CP is unusable on Qwen3.8-27B.** 48 of its 64 layers are GDN, and `_gdn`
  refuses `cp>1` rather than silently start a chunk from a zero state. This is
  the attention half; the model-level claim waits on the GDN `(A, B)` all-gather.
- world=2 on CPU/gloo is the largest exercised. No NCCL, no multi-GPU, no 27B run
  (`pending-remote`).
- No perf number of any kind. The collective is unbucketed and unoverlapped, and
  the exposed-cost measurement the ring decision rests on still does not exist.
- The paged path refuses `cp>1` too — serving holds whole sequences.

## Rule

The backward of an all-gather is a slice only when each rank owns a distinct
slice of the **output**. When every rank reads the whole gathered tensor, each
rank's gradient is a partial sum and the backward is a `reduce_scatter`. Check
which one it is by measuring against a dense reference, not by reusing the rule
from another gather.
