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

**The streaming optimizer needed measuring, not refusing.** My first read was
that it applies each gradient "in completion order, which differs per rank", and
I made `dp>1` with `streams=True` raise. That reasoning was wrong and untested:
the tape walks its entries in the order it *recorded* them, which is the model's
graph, and a probe over four ranks found the sequence **identical on all of them**
(27 params, same order). The per-rank order in the abort above came from
`param_ids & set(grads)` — a set of `id()` ints, iterated by hash — not from the
tape. So the streaming path reduces each gradient where it applies it, and the
refusal is gone.

This mattered: `Adafactor` with streamed updates is the 27B's optimizer
(`cli.py:240`, Adam's m+v is 200.4 GiB there), so a refusal would have made dp
unusable on the only model actually trained.

`_order_agrees()` re-checks the assumption instead of trusting it: each rank
hashes its own apply order and all-gathers the hash **over the dp group**, once
per step, behind `TILERL_CHECK_DP_ORDER` so gates pay for it and training does
not. It turns a process abort into a message naming the cause.

## The gate

`tests/dp_world4.py`: four gloo ranks as (dp=2, tp=2), each replica training a
real `train_step` on its own half of a 2-row batch, compared against a **dp=1
tp=2 step on the whole batch**. Built through `cli._shard`, so the flag's own
group plumbing is what gets gated.

| arm | worst deviation from the dp=1 step |
|---|---|
| gate (AdamW) | **1.0 ulp** |
| gate (Adafactor, streamed updates) | **1.0 ulp** |
| `--no-dp` (averaging removed, layout kept) | **83671 ulp** |
| `--scramble` (one dp rank's order reversed) | reports the mismatch |

The streamed arm compares against a **streamed** dp=1 reference: Adafactor and
AdamW produce different weights from the same gradients, so a cross-optimizer
comparison would measure the optimizers rather than the reduce.

**The scramble control passed at first, and its own bug is the reason.** It
reversed the order on `rank % 2`, but the check compares within the dp group,
which for rank r is {r, r+2} — both members always have the same parity, so it
reversed both sides of every comparison and they still agreed. Keyed on
`dp_rank` instead, it reports.

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
cost, and it is unmeasured here. The tape's order is measured identical across
ranks on `tiny` at world=4; `_order_agrees` is what would catch a model or a tape
change where it is not.

## Rule

**A collective's argument order is part of its contract, and Python's natural
iteration order is usually per-process.** Anything keyed by `id()`, by allocation,
or by completion is a different sequence on every rank; only names, sorted keys,
or an explicit index are shared. When a collective goes wrong this way the
process aborts inside the transport rather than raising, so the traceback points
at `mp.spawn`, not at the loop — expect to find it by reading the ordering, not
by reading the error.

**And the second rule, which cost more: I turned one diagnosis into a refusal
without testing it.** Having found `id()` order to be per-rank, I assumed the
tape's order was too and made the streaming path raise — which would have made dp
unusable on the 27B, since Adafactor is its optimizer. The probe that settled it
took one run. A refusal is a claim about what does not work; it needs the same
evidence as a claim about what does.

