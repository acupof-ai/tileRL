# The dp gradient average, and a collective issued in per-rank order, 2026-09-05

## Context

`Mesh.dp_group()` shipped with #123 and nothing called it. `--tp` (#126) refused
`dp>1` outright, because no gradient all-reduce across replicas existed anywhere
in the tree: each replica would train on its own rows and diverge, with finite
plausible losses and no error.

## What worked

`Backend.dp_reduce(g)` — one all-reduce over the dp group, then divide by
`dp_world`. **Mean, not sum:** each replica's loss already averages over its own
rows, so summing scales the update by `dp_world`, which reads as a convergence
difference rather than an error.

It runs in `_step` **before the clip**, so the clipped norm is the whole batch's.
Clipping first would scale each replica's gradients by a different factor, the
same defect the tp shards had (see
[the clip norm was this shard's own](../errors/2026-09-05-the-clip-norm-was-this-shards-own.md)).

`init_tp` now takes `dp_groups` beside `tp_groups`, and `_shard` builds both from
the mesh. Every rank builds every group, **tp first then dp, in one order**:
`new_group` is collective, so the loops' order is part of the contract, not a
detail.

## The bug this turned up: a collective in per-rank order

The first version was `for g in acc.values(): dp_reduce(g)`. Every rank aborted:

```
libc++abi: terminating due to uncaught exception of type gloo::EnforceNotMet:
[enforce fail at gloo/transport/uv/pair.cc:253] op.nread == op.preamble.nbytes
```

`acc` is keyed by `id(param)` and iterated in insertion order, which is the order
the tape finalized each gradient — a per-rank property. So rank 0 was all-reducing
`down_proj` while rank 1 all-reduced `embed_tokens`, and gloo hit two different
message sizes on the same pair.

It **aborts the process**, it does not raise: no Python traceback points at the
loop, and `mp.spawn` reports only `process 1 terminated with signal SIGABRT`.

Fixed by iterating `sorted(params)` — parameter names, which every rank agrees
on — and looking each gradient up by id.

Two wrong diagnoses came first, both discarded by measurement rather than
argument: that a second `mp.spawn` in a gloo-initialized process was the cause
(the arm aborts alone, with one spawn), and that the two arms collided on
`MASTER_PORT` (distinct ports changed nothing).

**The streaming optimizer cannot do this at all.** It applies each gradient the
moment backward finalizes it, which is exactly the order that differs per rank,
so `_step` raises for `dp>1` with `streams=True` rather than shipping a hang.

## The gate

`tests/dp_world4.py`: four gloo ranks as (dp=2, tp=2), each replica training a
real `train_step` on its own half of a 2-row batch, compared against a **dp=1
tp=2 step on the whole batch**. Built through `cli._shard`, so the flag's own
group plumbing is what gets gated.

| arm | worst deviation from the dp=1 step |
|---|---|
| gate | **1.0 ulp** |
| `--no-dp` (averaging removed, layout kept) | **83671 ulp** |

In ulps, not absolutely, for the reason
[the clip entry](../errors/2026-09-05-the-clip-norm-was-this-shards-own.md)
records: bf16 spacing at magnitude 1 is 7.8e-3, so any absolute tolerance is
either tighter than one rounding step or four steps wide.

Its own file, not an arm of `mesh_world4.py`: that file's spawn had already run,
and two arms in one process were the first thing I suspected — keeping them apart
also keeps each gate's failure attributable to one thing.

## What this does NOT establish

world=4 on CPU/gloo is the largest exercised; no NCCL run and no 27B run
(`pending-remote`). The dp all-reduce is one call per gradient with no bucketing
or overlap with backward — at the 27B's parameter count that is the obvious next
cost, and it is unmeasured here. `dp>1` with the streaming optimizer is refused,
not solved.

## Rule

**A collective's argument order is part of its contract, and Python's natural
iteration order is usually per-process.** Anything keyed by `id()`, by allocation,
or by completion is a different sequence on every rank; only names, sorted keys,
or an explicit index are shared. When a collective goes wrong this way the
process aborts inside the transport rather than raising, so the traceback points
at `mp.spawn`, not at the loop — expect to find it by reading the ordering, not
by reading the error.
