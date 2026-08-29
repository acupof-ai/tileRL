# Adafactor synced to the host twice per parameter

**Date:** 2026-08-30 · **Scope:** `train` · **Status:** CPU-verified, GPU `pending-remote`

## Context

Training profile: `linear_fp4_bwd` 29.7% and ~90k elementwise launches 29.2%
of a 1189 ms step. Chasing the elementwise half without a GPU, an aten-dispatch
count over one `train_step` (tiny model, CPU) showed something the launch count
alone hides — **55 device-to-host syncs per step**, of which 54 came from one
call site.

## What worked

Counting `aten._local_scalar_dense` with a `TorchDispatchMode` and attributing
each one to its Python frame. All 54 were `Adafactor._rms` (`autograd.py:561`),
called twice per parameter:

```python
rms = self._rms(upd)                              # float(t.norm()/sqrt(n)) -> sync
step = self.lr * max(self.eps[1], self._rms(p32)) # -> sync
```

27 parameters × 2 = 54. The 27B has **851 parameter tensors → 1702 syncs a
step**. Worse than the count: `streams=True` runs `step_one` *interleaved with
backward*, so each sync drains the pipeline in the middle of the backward pass
rather than at a step boundary.

The repo had already learned this exact lesson one function down — `clip_grad_norm`
carries the note that its 126 per-grad `.to("cpu")` copies cost 7% of the 27B
LoRA step, and accumulates on-device because of it. `Adafactor` was doing 13x
that count.

**Fix:** keep both RMS values as device tensors. The early `return` on a
non-finite gradient (which needed the value on the host) becomes a zero scale;
`add_(upd, alpha=-step)` becomes `sub_(upd.mul(step))` because `alpha` will not
take a tensor. Two extra elementwise launches, zero syncs.

## Numbers

| | before | after |
|---|---|---|
| host syncs / step (tiny, 27 params) | 55 | **1** (the loss finite-check) |
| host syncs / step (27B, 851 params) | 1702 | **1** |
| compute launches / step (tiny) | 1603 | **1441** |
| `max abs` param drift vs old formula, 5 steps | — | 1.2e-07 |

The launch count fell because the replacement clip factor is one chain —
`clip / rms.clamp(min=clip)` with `nan_to_num` standing in for the early
return — rather than an `isfinite` / `reciprocal` / `where` triple.

Step time on the 27B: **pending-remote** — no GPU on this host, and the win is
a sync/overlap effect that a CPU run cannot show at all.

## What is still there

`step_one` costs **27 launches for a 2D parameter, 20 for a 1D one** — about
**22,500 a step** for the 27B's 851 tensors, roughly a quarter of the ~90k
elementwise launches in the training profile. Batching them the way
`torch.optim`'s `foreach=True` does is only half available here: Adafactor's
per-shape `mean(dim=)` reductions and rank-1 reconstruction have no `_foreach_`
form, and `streams=True` deliberately holds one gradient at a time, so a
foreach path would have to bucket by bytes to keep that bound.

# ponytail: 27 launches/param, byte-bucketed _foreach_ when the profile says
# the optimizer is the elementwise bottleneck on a real GPU step.

## Rule

A launch-count profile does not show syncs, and syncs are the expensive kind of
"elementwise op". Count `aten._local_scalar_dense` per step before optimizing
kernel counts. Any `float(tensor)` / `.item()` in a per-parameter loop is a bug
by construction — `tests/test_e2e.py::test_train_step_does_not_sync_per_parameter`
now caps a step at 2.
