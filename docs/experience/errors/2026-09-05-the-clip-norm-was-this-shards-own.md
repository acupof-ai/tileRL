# The clip norm was this shard's own, 2026-09-05

## Context

TP-2 training passed its gradient gate: all 54/56 gradient tensors matched the
unsharded ones on both head layouts. The gate stopped at `tape.backward`. The
next thing `_step` does is clip, and clipping reads a number that spans the whole
model.

## Root cause

`clip_grad_norm` summed the squares of the gradients **this rank holds**. Under
TP each rank holds a different slice, so the totals differ and each rank scales
its shard by a different factor. Measured on tiny at tp=2:

```
world=1  2641.795815
rank0    2399.731442      ratio 1.0485
rank1    2288.828278
```

`max_norm=1.0` against a norm of 2.4e3 means clipping fires every step, so the
two shards of one weight matrix were scaled 4.85% apart on every update.

Summing the local totals across ranks does not fix it: the replicated tensors
(norms, `embed_tokens`, biases) are identical on every rank, so a plain
all-reduce counts each of them `world` times and overshoots to **3316.24**. The
sharded and replicated squares have to be reduced differently — sum the first,
average the second — which needs to know which is which.

## Fix

`is_sharded(key)` in `tensor_parallel.py`, and `clip_grad_norm(grads, max_norm,
sharded, backend)` reducing a 2-vector `[sharded squares, replicated squares]`
in one all-reduce. Both ranks now compute **2641.796301** against world=1's
2641.795815 — 1.8e-7 relative, and identical to each other.

The classifier is by name, not by config: `model.cfg` after `tp_config` no longer
knows the original head counts, and reading `num_kv_heads` off it calls a sharded
`k_proj` replicated (tiny shards kv 2 -> 1, and 1 reads as "one head everywhere").

**A LoRA pair splits on one side only, and the side flips.** On a column-parallel
base, `lora_b` is `[n/w, r]` and `lora_a` is the full `[r, k]`; on a row-parallel
base it is the mirror image. The first version of `is_sharded` answered "sharded"
for both halves of every adapter — measured wrong on `q_proj.lora_a`,
`o_proj.lora_b` and `down_proj.lora_b`, which is 3 of the 6 adapter tensors on
one layer. `--tp` with `--rl`/`--opd` trains exactly these.

## The gate, and two things that nearly made it vacuous

`tests/tp_world2.py` gained a whole `train_step` and compares the **weights after
the update**; the gradient arms cannot see this bug because they stop before the
optimizer. Its control (`--local-clip`) restores the local-shard norm: 8 tensors
wrong, up to **261 ulp** on `embed_tokens`, against 0 for the fixed arm.

**Comparing weights absolutely does not work.** bf16 spacing at magnitude 1 is
7.8e-3, so the two summation orders land one rounding step apart and any absolute
tolerance is either tighter than one ulp (rounding reads as failure — 2 tensors
did) or four steps wide. The comparison is in ulps.

**A control that hangs reports nothing.** `_refusals()` runs in the parent after
the spawned arms have set `MASTER_ADDR`, so a `_shard` that wrongly accepted dp=2
would block in `init_process_group(world_size=4)` waiting for three ranks that do
not exist — not print "ACCEPTED". The probe backend raises on `init_tp` instead of
joining, so reaching it at all is the failure, and the control prints its reason.

`tensor_parallel.py`'s `__main__` checks `is_sharded` against what `shard_params`
actually did, for all 59 params including the 32 adapters. Control: making the
LoRA branch ignore which side splits fails it on `down_proj.lora_b`.

## What this does NOT establish

`--tp` refuses dp>1: no gradient all-reduce across dp replicas exists anywhere in
the tree, and `DataParallelEngine` is a serving object (N engines on N devices in
one process), not this axis. Without the refusal each replica would train a
different model on its own data with no error and plausible losses.

The K/V-replica case (`replicas > 1`, where ranks sharing a KV head may hold
partial sums rather than copies) is **unmeasured**: tiny at tp=4 is refused by
`linear_num_value_heads=2`, and the 27B needs tp=8 on real hardware. cpu/gloo
world=2 is the largest exercised. No 27B run (`pending-remote`).

## Rule

**A gate that stops one call short of the bug passes.** The gradient comparison
was correct and thorough — 54 tensors, both head layouts, its own negative
control — and the defect was in the next statement, where a per-shard quantity
gets used as a global one. When a change adds a collective, list every step
downstream that reads across the whole model (clip, norm, loss scaling,
early-stop thresholds) and check each is reduced, rather than checking the thing
the change was about.
