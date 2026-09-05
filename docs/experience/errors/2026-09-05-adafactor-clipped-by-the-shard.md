# Adafactor clipped by the shard's statistics, not the tensor's, 2026-09-05

## Context

#126 fixed `clip_grad_norm` to span the whole model under TP. That function is on
the **accumulating** path. The 27B trains full-parameter SFT with `Adafactor`
(`cli.py:240`) whose `streams = True` path returns before it, and Adafactor was
never using a global norm at all — it clips each *update* by that update's own
RMS. A peer asked whether that per-update clip is TP-global or local. It was
local.

## Root cause

Adafactor's arithmetic is full of **whole-tensor** statistics, and under TP each
rank holds a slice, so every one of them was that slice's:

```
Adafactor, tp=2 vs tp=1 on tiny (same gradients, same seed):
  32 tensors off by >1 bf16 ulp, worst 304911 ulp
AdamW, same comparison:            0 tensors, worst 1.0 ulp
```

AdamW is elementwise, so it has no such statistic and cannot exhibit this. That
contrast is what proves the defect is the optimizer's, not the gradients'.

## The audit, which is wider than the two RMS calls

The obvious sites are `_rms(upd)` (the update clip) and `_rms(p32)` (the relative
step size). A peer pushed for the rest, and the factored second moment turns out
to depend on **which axis** is sharded. Measured by splitting a tensor and
comparing, not read off the code:

| statistic | dim-0 shard (column-parallel) | dim-1 shard (row-parallel) |
|---|---|---|
| `u.mean(dim=1)` → `r` | whole | **needs reduce** |
| `u.mean(dim=0)` → `c` | **needs reduce** | whole |
| `r.mean()` | **needs reduce** | whole (`r` is whole) |
| `_rms(upd)` | **needs reduce** | **needs reduce** |
| `_rms(p32)` | **needs reduce** | **needs reduce** |
| `v` (unfactored) | elementwise | elementwise |
| `weight_decay` | via `step` | via `step` |

A mean along the sharded axis has only this rank's terms; a mean along the other
axis is complete on every rank. `r.mean()` is a scalar over all rows, so a dim-0
shard holds only some of them. This is why `sharded` is the shard **dim**, not a
bool: a bool cannot express "reduce `c` but not `r`".

`shard_dim(key)` in `tensor_parallel.py` returns it, and the `__main__` selfcheck
checks it against what `shard_params` actually did for all 59 params — including
the LoRA pairs, whose narrow side flips with the base.

## Result

```
Adafactor tp=2 vs tp=1, reduce on:   2 tensors off by >1 ulp, worst 4.0 ulp
```

Not zero, and the reason is measured rather than waved at: the reduced inputs
agree to **2.9e-07** (gradient sum-of-squares) and **6.1e-08** (parameter
sum-of-squares), and Adafactor divides by `rsqrt` of those, so a relative
difference that small comes out as a few bf16 rounding steps on 2 of 27 tensors.
The gate's Adafactor arm allows 4 ulp for that; the defect it exists for is
304911 ulp, so the margin is five orders either way.

## Gates

`tests/tp_world2.py` gained an Adafactor arm — tp=2 vs tp=1, weights after the
update — because the dp gate structurally cannot see this: both its arms are
tp=2, so a per-shard statistic cancels. Its control `--local-stats` restores the
per-shard behaviour and fails on **both** axes (`down_proj`/`o_proj` are dim-1 at
478 and 526 ulp; `gate_proj`/`k_proj` are dim-0 at 195 and 171).

## ISO is a third streaming optimizer, and it cannot be fixed this way

Changing `step_one`'s signature broke `tests/test_iso.py`, which is how I found
that `--optim iso` is also `streams = True`. Its arithmetic is not a statistic
that can be summed: it reparameterizes each 2D weight by **its own SVD**
(`iso.py:69`) and reconstructs `p = (u * s) @ v.T`. A shard's singular vectors
are not a slice of the full matrix's, so no all-reduce recovers them.

So ISO **refuses** a sharded param, and this time the refusal is a measured
statement about a factorization rather than a guess about an iteration order.
`test_iso_refuses_a_sharded_param` holds it; deleting the check fails the test.

## What this does NOT establish

world=2 on CPU/gloo; no NCCL, no 27B run (`pending-remote`). The reduce adds two
small collectives per sharded tensor per step (one for each RMS) plus one for the
factored mean — unmeasured against the step time, and not bucketed.

## Rule

**A fix aimed at one path stops at that path's edge.** `clip_grad_norm` was made
TP-global and the entry said so; the 27B does not call it, because a streaming
optimizer returns before it and clips its own way. When a defect is "this
quantity is per-shard", the fix is not one call site — it is an audit of every
whole-tensor quantity in the reachable code, listed with what a shard does to
each, because the next reader needs the table and not the conclusion. And ask
which axis: a matrix has two, and a statistic can be correct along one and
partial along the other.
