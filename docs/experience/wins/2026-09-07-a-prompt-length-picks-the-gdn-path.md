# A prompt length picks the GDN forward path — H20 sm90, 2026-09-07

> Status: measurement. Closes task-board row 45. The fix (pad T) ships with the C=64 recompute
> PR as its precondition, not on a step-time claim.

## The condition

`grpo_loop` builds the training batch as `len(prompt) + gen` (`train.py:459-462`), where `gen`
is bucketed to a power of two, floored at `min(256, cap)` and clamped to the cap
(`:456-458`). Every power of two ≥ 64 is a multiple of 64, so:

```
T % 64 == len(prompt) % 64
```

`linear_attn_chunk` takes the chunkwise-WY kernels only when `t % _WY_CHUNK == 0`
(`_wy_eligible`, `backend.py:1147`). So **the GDN forward reaches the WY kernels only when the
prompt length is a multiple of 64**, and 63 of 64 prompt lengths fall through to
`gdn_chunk_fused` (`:1169`) instead. Prompts arrive unconstrained from the caller's list
(`train.py:418`); nothing in `train.py` or `model.py` mentions `_WY_CHUNK` or pads to it.

An odd cap breaks it independently: `gen = cap` is not rounded, so a cap of 1500 gives
`1500 % 64 = 44`.

**Every number we have was measured on the multiple-of-64 arm.** #192, #217 and #219 all ran at
prompt 256, T=1280.

## What the two arms show

Card 6, uncontended, sequential, `--gen 1024 --group 8`, warm step. Arm labels are read from the
backend's own dispatch, printed by the probe.

| | arm A | arm B |
|---|---|---|
| prompt / T | 256 / 1280 | 300 / 1324 |
| T % 64 | 0 | 44 |
| GDN forward arm | `wy_kernels` | `gdn_chunk_fused` |
| checkpoint segment | `mlp` | **`layer`** |
| `backward_secs` (warm) | 84.940 | 90.685 |
| chunk calls | 30720 = 384 × 80 | 31872 = 384 × 83 |

**The backward is the same code on both arms.** 80 = `ceil(1280/16)` and 83 = `ceil(1324/16)`:
`gdn_backward` always runs the reference chunk loop at C=16 (`_GDN_CHUNK`, `reference.py:592`)
regardless of which forward arm ran. Per-call the two arms agree within 5%:

| row | A total | B total | per-call B/A |
|---|---:|---:|---:|
| `_gdn_chunk_bwd` — the adjoint | 31.036 | 30.744 | 0.955x |
| `_gdn_chunk_fwd` — the recompute | 18.988 | 18.441 | 0.936x |
| `solve_triangular` — inside the recompute | 2.161 | 2.138 | 0.954x |
| attributed | 57.908 | 57.068 | |
| residual | 27.033 (31.8%) | 33.617 (37.1%) | |

So the +5.745 s warm difference sits in the residual — `Tape.backward`'s own loop plus this
probe's per-call sync — and not in any measured row.

**Row 45's finding, stated for what it is:** `T % 64` selects the GDN *forward* dispatch, and
therefore which recompute path a future C=64 switch can use. The forward arm's own cost is
**unmeasured**, because nothing times the forward: `train_secs − backward_secs` is 0.09 s in arm
A and 0.075 s in arm B, so `rl_step`'s timing does not expose it either. Measuring it needs a
timing point around the forward pass — that is the instrument if anyone wants the number.

## Why the pad ships anyway

`_wy_eligible` gates the WY forward *and* any recompute routed through `_gdn_wy_core`. Without
the pad, 63 of 64 prompt lengths take the fused forward and keep `gdn_backward`'s 80-iteration
python recompute, so the C=64 switch would apply to one prompt length in 64. The pad is at most
63 tokens — 4.92% of T=1280, 20 tokens (1.51%) at T=1324 — and it is loss-neutral: `rl_step`
masks on `slen` (`train.py:310-311`, `keep = (p >= plen-1) & (p < slen-1)`), a different variable
from the WY predicate's `seq_q_lens`, so padded positions carry zero gradient weight.

That is a coverage argument, not a perf one, and it is the whole justification. It ships inside
the recompute PR.

## Three retractions

**A cold step quoted against a warm one.** I reported arm B at 1.47x arm A from arm B's *step 1*
(122.038) against arm A's *step 2* (84.940). Arm B's warm step is 90.685; the real ratio is
**1.068x**. Step 1 pays the JIT for a new shape and ran 1.35x the warm step. The probe prints
both steps and I had written "step 2 is the one I will quote" before quoting step 1.

**A two-shape comparison that moved two variables.** `_MLP_SEGMENT_MAX_T = 1280` and the selector
is `segment="layer" if t > _MLP_SEGMENT_MAX_T else "mlp"` (`train.py:71`, `:162`) — strict `>`,
so T=1280 is the last `mlp` shape and T=1324 is `layer`. Arm B therefore changed the checkpoint
segment as well as the GDN arm, and #211 measured that segment alone at 1.079x backward with a
forward peak of 54.0 → 14.9 GiB. The 1.068x A→B ratio is *below* the segment's own 1.079x, and
the 44 GB vs 67 GB memory difference is the checkpoint, not the GDN path. The probe now prints
`segment` beside the arm so this cannot recur.

**An instrument aimed at the wrong side.** `--inside-gdn` splits the backward; the question was
about the forward's dispatch. A third arm at T=1344 (WY arm, layer segment) was planned to
isolate `T % 64` at equal segment and was cancelled: it would have compared two shapes whose
backwards are the same C=16 code, producing a number that reads like an answer to a question it
cannot answer.

The label also said `fused_or_serial`, conflating `gdn_chunk_fused` with the per-step reference —
two different implementations. It now names which one the dispatch selects, and the field is
`gdn_forward_arm` rather than `gdn_arm`, because the old name implied it described the layer.

## Not established

- The forward arm's cost. Unmeasured, and no existing timing exposes it.
- The absolute seconds are not the shipped path's: a device sync around every timed call puts
  `backward_secs` ~11 s above #217's registry-mode reading of the same work.
- One warm step per arm, one process, one card.

## Rule

Before comparing two shapes, enumerate every threshold between them. T=1280 and T=1324 differ in
`T % _WY_CHUNK` and in `t > _MLP_SEGMENT_MAX_T`, and the second was invisible in the probe's
output until it was printed. A comparison that moves two variables measures neither.

Ask what an instrument can see before booking time on it. A backward profile cannot answer a
question about forward dispatch, however carefully the arms are chosen.
