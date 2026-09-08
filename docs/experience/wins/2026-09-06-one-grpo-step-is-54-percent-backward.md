# One GRPO step is 54% backward and 43% decode ticks — the next RL lever is backward, not the rollout

**Status:** closed
**Date:** 2026-09-06
**Commit:** a16ff9c (the tree the run executed; probe lands with this entry)
**Card:** H20 card 6
**Config:** `qwen38-27b`, group 8, gen 1024, `--blocks 2304`, micro 1, LoRA rank 16,
`decode_graph=True`, `NoPrefixStore()`

## Context

The step's total was reported and three of its parts already were: `rl_step` writes
`backward_secs` and `optimizer_secs` into a `timings` dict and `grpo_loop` seeds it with
`rollout_secs` (`train.py:286-315, 415`). Nothing split the rollout, the biggest bucket,
because `_drain` is `engine.step()` in a loop. So "what is the next RL lever" had no number
behind it.

## The decomposition

Warm means over 2 steps (step 0 excluded, it pays every JIT):

| bucket | seconds | share |
|---|---:|---:|
| **backward** | **71.529** | **54.4%** |
| rollout | 59.965 | 45.6% |
| — decode ticks (1023) | 56.484 | 42.9% |
| — mixed ticks (6) | 3.181 | 2.4% |
| — prefill tick (1) | 0.275 | 0.2% |
| — unattributed (submit/poll/python) | 0.025 | 0.02% |
| optimizer | 0.085 | 0.06% |
| reward | ~0 | ~0% |
| **step** | **131.579** | |

**The decomposition closes to 0.0006 s of 131.579** and **0 ticks were unexplained** — every
tick moved exactly one forward counter, so nothing is hiding in a residual.

**Backward is the lever.** It is 1.193x the entire rollout and 54.4% of the step, against a
decode path that has already had graph capture, the recapture waiver and the cast work spent
on it. Optimizer and reward are noise (0.06% and ~0). Per-tick decode is **55.21 ms** at
B=8 W=1, which is **6.902 ms per row-token**.

Cold step 0 was 183.6 s, **1.396x** the warm mean, all of it in backward (115.1 → 71.5,
1.59x): the `linear_fp4_bwd` and `gemm_tn` compiles land there.

## The sync convention, measured rather than assumed

Every tick is followed by `torch.cuda.synchronize()` so a bucket is device-inclusive and the
residual is host-only. That convention costs **0.0381 s over 1030 ticks = 37 µs/tick, 0.028%
of the step** — negligible *here* because a tick is 55 ms. It is not transferable: at a
shorter tick the same 37 µs is a large fraction, which is why
[the launch floor](2026-09-06-the-launch-floor-is-ten-microseconds.md) measured 10.10 vs
21.23 µs on one kernel purely from sync convention. Any comparison against an unsynced
measurement subtracts this number first.

## Two OOMs, and the first fix was wrong in a way worth recording

The run this entry reports is the third attempt.

**Attempt 1** at gen 4096, `--blocks 4096`: OOM, 94.42 GiB in use, failing allocation
**1.69 GiB** inside `reference.py:418` (`dense_attention`, the training forward).

**Attempt 2** cut the pool to `--blocks 2304`, freeing **3.63 GiB** against a 1.69 GiB need.
It OOMed at the same 1.69 GiB — and with **94.78 GiB in use, 0.36 GiB MORE than before.**
That is the finding: the freed pool did not become headroom, so the pool was never the
binding term, and "reduce the KV pool" is the obvious wrong fix that would have looked
plausible indefinitely.

**What actually binds:** the training forward materializes a full T² attention matrix per
full-attn layer. At the 27B's 24 heads and T = 4352 (4096 gen + 256 prompt), f32:
**24 × 4352² × 4 B = 1.69 GiB each**, matching the failing allocation exactly, quadratic in
generated length.

**Corrected 2026-09-06 (was `× 16 full-attn layers = 27.1 GiB`).** That figure assumed the
tape retains one matrix per layer. It does not: `autograd.py:149-166`'s `_attention` handler
saves q, k and v and never saves `att`, so each T² tensor dies when its call returns.
tilerl-48 measured the live bytes by refcount death through a `TorchDispatchMode`: the peak
is **3 concurrent matrices in forward (5.08 GiB) and 6 in backward (10.16 GiB)**, not 16.
The OOM at 94.42 GiB in use is real; the attribution to a 27 GiB per-layer tensor family
was not. gen 4096 → 1024 divides each matrix by 16 (0.094 GiB) and it fits with room.

**So this bounds the rollout cap from the training side**, at 10.16 GiB rather than 27.1: the
run-3 recipe's 4096 cap is not reachable through this training path on one 95 GiB card at
group 8, since the card was already at 94.42 GiB in use when a single 1.69 GiB matrix failed
to allocate. The decomposition above is therefore at gen 1024, and a longer-generation step
needs the attention matrix not to be materialized rather than a bigger card.

## Rule

**A memory hypothesis needs the arm where the freed bytes fail to appear.** Cutting the pool
by 3.63 GiB and observing 0.36 GiB *more* in use is what killed the pool hypothesis in one
run; without that arm the 1.69 GiB allocation and a large pool are perfectly consistent with
each other and the wrong fix ships.

**A per-call tensor size times a layer count is not a live-bytes figure.** The 27.1 GiB in
the first version of this entry was 1.69 × 16 — arithmetic over the config, with retention
assumed. Which tensors the tape *saves* is a property of the handler, and `_attention` saves
q/k/v only, so 13 of those 16 matrices never coexist. The number that binds is concurrent
live bytes at the peak, and only a refcount- or allocator-level measurement gives it.

**A residual has to be named and checked, not assumed small.** `unattributed_secs` is 0.025 s
and `unexplained_ticks` is 0 — but the probe credits a tick that moves no forward counter to
its own bucket rather than to a phase, because a bucket that absorbs unattributed ticks reads
as a clean decomposition. Verified by folding those ticks into `decode` and confirming the
selfcheck goes red on the counts assertion.

## Results

| date | commit | card | metric | value |
|---|---|---|---|---|
| 2026-09-06 | a16ff9c | H20 6 | step, warm mean of 2 | **131.579 s** |
| 2026-09-06 | a16ff9c | H20 6 | backward | **71.529 s, 54.4%** |
| 2026-09-06 | a16ff9c | H20 6 | rollout | 59.965 s, 45.6% |
| 2026-09-06 | a16ff9c | H20 6 | decode ticks (1023) | 56.484 s, **55.21 ms/tick** |
| 2026-09-06 | a16ff9c | H20 6 | per row-token at B=8 | **6.902 ms** |
| 2026-09-06 | a16ff9c | H20 6 | optimizer / reward | 0.085 s / ~0 |
| 2026-09-06 | a16ff9c | H20 6 | step − sum of parts | **0.0006 s** |
| 2026-09-06 | a16ff9c | H20 6 | unexplained ticks | **0** |
| 2026-09-06 | a16ff9c | H20 6 | sync overhead | 37 µs/tick, 0.028% |
| 2026-09-06 | a16ff9c | H20 6 | cold step 0 | 183.62 s, 1.396x warm |
| 2026-09-06 | a16ff9c | H20 6 | T² attention, T=4352 f32 | **1.69 GiB each; peak 3 live fwd = 5.08, 6 bwd = 10.16 GiB** |

## Limitations

- **gen 1024, not the recipe's 4096**, forced by the T² retention above. The backward share
  is expected to *rise* with generation length since backward carries the quadratic term, so
  54.4% is a floor for longer generations, not a constant.
- **2 warm steps.** Backward 72.49 and 70.57, decode 56.50 and 56.46 — tight, but 2 samples
  give no useful spread.
- **`micro=1` with a LoRA adapter**, so `acc` holds only adapter gradients. A full-parameter
  step is the 50.1 GiB path `train.py:141` describes and is a different measurement.
- The rollout runs on the **ordinary paged engine** — `engine.submit` plus `_drain`
  (`train.py:442-445`), the same path serving uses. An earlier version of this line said it was
  "the training path (`_training_kv` builds a dense pool, no draft plane, no `spec_depth`)", and
  both halves were wrong: `_training_kv` is called at `train.py:160`, in the forward/backward, not
  in the rollout, and the serving arm this was contrasted against was itself W=1 no-draft. Two
  differences that are real: the rollout serves a LoRA-attached model, and it builds with
  `fuse_projections` defaulting to `False` (`cli.py:44`) where the serving arms pass `True`.
  Priced, those are ~4% and 2.9% of the 2.4x gap to a serving tick — so **55.21 ms/tick against a
  serving arm's 23.34 is 93% unexplained**, not explained by the path. Open item, not a caveat.
- Measured on a16ff9c, **before #188** changed the decode-graph fallback. The changed branch
  was unreachable here — no draft, so `graph_keys` is `{(bucket(rows), 1)}` and every tick is
  a full B=8 W=1, on-grid; the log has 0 capture-failure lines.
