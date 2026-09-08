# Three OOMs, and the memory was never where the traceback pointed

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55

## Context

The ISO-RL arm OOMed three times on the 27B before reaching step 2. Each traceback named a
real allocation, and each one led me somewhere wrong.

| attempt | asked for | site | what I concluded |
|---|---|---|---|
| 1 | 9.47 GiB | my Σ gate, `svdvals(...double())` | f64 is wasteful — true |
| 2 | 4.74 GiB | my Σ gate, `.float().cpu()` | the cast order matters — true |
| 3 | 4.74 GiB | `autograd.step_one`, `p.to(float32)` | the embedding cast is the bottleneck — **false** |

On the third I proposed three levers to a peer: cut KV blocks (frees 0.38 GiB), restrict
`trainable` to 2D params (frees ~0), or exclude `embed_tokens`/`lm_head` (frees exactly the
4.74 GiB peak, and changes what ISO is measured on — it would have required rewording P3's
exit criterion).

## Root cause

The peer refused to choose and asked one question instead: on a 95.2 GiB card, why is only
3.38 GiB free? A serving arm of the same model on the same card type peaks at 25.99 GiB.

`scripts/probe_rl_budget.py`, read after `build_engine` and before any step:

```
after _build_model(keep_master=True)   allocated  0.00  free 94.91   <- params still on host
after build_engine                     allocated 72.09  free 22.51

  bf16 masters        50.10 GiB (851 tensors)
  fp8 served bytes     9.90 GiB (233)
  uint8 fp4+scales     6.98 GiB (264)
  f32                  3.51 GiB (994)
  params 70.48, other 1.61 (KV pool, state pool, workspaces)

largest param: embed_tokens (248320, 5120) bf16
  its f32 copy: 4.74 GiB   free right now: 22.51 GiB   -> fits: True
```

**The allocation that failed fits with 17.8 GiB to spare.** So all three levers were
subtraction inside a budget nobody had read, and the one I preferred would have permanently
narrowed ISO's scope for a memory problem that does not exist.

`scripts/probe_rl_phases.py` then attributed the rest by phase. The rollout costs **0.17 GiB**;
the forward+backward costs **33.24 GiB**. And `micro` is not a lever against it: `_step:152`
refuses micro-batching outright for any optimizer declaring `streams`, because holding every
parameter gradient until the update is the 50.1 GiB streaming exists to avoid. Adafactor
streams, so micro and streaming are mutually exclusive by construction — a fact I would have
discovered by running it, and did.

What the arm was actually missing is the third mechanism in
[wins/2026-08-29-full-finetune-fits.md](../wins/2026-08-29-full-finetune-fits.md):
`drop_quantized()` frees the served bytes once a bf16 master exists, because the tape routes
every linear through `master_linear` and never reads them. `cli.py` calls it at both training
entry points (`:275`, `:1094`). This script builds its engine directly, so nothing called it,
and 20.38 GiB of unreadable bytes sat on the card through all three attempts.

## Fix

`drop_quantized(base)` before the host snapshot, so the snapshot holds masters only. Measured:

```
drop_quantized: 2342 -> 851 tensors
after build_engine     allocated 51.56  free 43.29   (params 50.10)
after rollout          allocated 51.58  peak 51.74  free 42.98
after rl_step          allocated 51.61  peak 84.98  free  9.31   <- the step completed
```

**One number disagrees with the record and I am not smoothing it.** The 08-29 entry says 14.9
GiB for this call; I measure **20.38** (1.37×, under the 2× suspect-the-instrument bar but not
equal). Internally consistent: my dtype totals give fp8 9.90 + uint8 6.98 = 16.88, and
16.88 + 3.50 GiB of f32 scale keys = 20.38. The likely reason for the gap is that 14.9 predates
the `.w8`/`.wscale` suffixes that fp8 checkpoints carry — the same uncounted-suffix shape
already in this log. Someone should re-derive the entry's figure rather than assume mine.

## Rule

A traceback says where an allocation **failed**, not where the memory **went**. The failing
site is whatever ran next after the budget was already spent, so it is systematically the
wrong place to optimize — and it is convincing precisely because the number it names is real.

Before choosing between levers, read the total. If the failing allocation would fit in the
budget a correct configuration leaves, the lever list is answering the wrong question, and the
most appealing lever on that list was the one that would have changed the measurement.

And check whether the mechanism you need already exists on the path you bypassed. This arm
reimplemented a training entry point and inherited none of its memory discipline; the fix was
one call that had been written, measured, and documented ten days earlier.
