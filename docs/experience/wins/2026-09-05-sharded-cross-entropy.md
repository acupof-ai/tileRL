# The sharded cross-entropy: two scalars per row instead of a 1.89 GiB gather, 2026-09-05

## Context

A vocab-parallel `lm_head` leaves each rank holding `[B, T, V/world]`. The
unsharded `cross_entropy_loss_grad` needs the whole row, so the obvious move is
to all-gather the logits and reuse it. This is the op that does not.

The cost, computed for this op rather than borrowed: one f32 logits row at the
27B's `V = 248320` is **0.947 MiB**, so a gather over the positions TP training
actually runs is

| batch | positions | gathered logits |
|---|---:|---:|
| B=8, T=256 | 2048 | **1.89 GiB** |
| B=8, T=512 | 4096 | **3.79 GiB** |

per rank, per step, on top of the weights and the tape. (The 08-30 OOM is often
cited here at 8.5 GiB; that was three `(3072, 248320)` f32 tensors from `lm_head`
over every position at B=32, not a TP gather — a different failure with a similar
smell, and the argument above stands without it.)

## What worked

Each rank reduces its own shard to two scalars per row, and three all-reduces
combine them:

```
m = max_r m_r                    one all_reduce (max), [B*T]
s = sum_r exp(m_r - m) * s_r     one all_reduce (sum), [B*T]
x[target] = sum_r picked_r       one all_reduce (sum), [B*T]
loss = m + log(s) - x[target]
```

The third one is the target term: the column lives on exactly one rank, so
`picked` is that rank's value and zero everywhere else, and summing selects it
without anyone needing to know who owns it. The design called for two — it missed
this one. The gradient (`softmax - onehot`) is produced already sharded, which is
what the sharded weight wants, and `scatter_add_` writes the `-1` only on the
owning rank.

Three `[B*T]` all-reduces replace one `[B, T, V]` gather. At the 27B's
`V = 248320`, that is a 248320:1 ratio in bytes moved.

## The memory half needed its own gate, and bytes do not work

`tracemalloc` reports **80 bytes for a 4 MB torch tensor** — torch's CPU
allocator is not Python's, so an "assert peak allocated bytes" gate on the CPU
target measures nothing. Measured before relying on it.

What is enforceable is the shape. A `TorchFunctionMode` records the widest last
dimension any op produces during the call; a `[B, T, V]` intermediate is exactly
the thing that must not exist, and its last dim is `V` against `V/world` for
every legitimate tensor.

```
$ python3 tests/ce_sharded_world2.py
sharded CE matches unsharded (loss 4.395447), widest tensor 32 == V/world

$ python3 tests/ce_sharded_world2.py --gather
rank 0: formed a tensor 64 wide, shard is 32
rank 1: formed a tensor 64 wide, shard is 32
memory control: correctly FAILED
```

The `--gather` arm is the all-gather implementation: **numerically correct**, and
it fails on the memory assertion alone. That is what makes the assertion a
control rather than a restatement — a gate that only checked the numbers would
pass the implementation this op exists to avoid.

## Two things the wiring needed

**`Backend.tp_rank` had to come back.** #106's vulture sweep deleted it as unread
— correctly, at the time. The sharded CE needs it to locate its slice of the
vocabulary. It carries a comment saying so, so the next sweep does not repeat the
deletion.

**`forward(sharded_logits=True)`.** Serving gathers because it needs a full row
to sample from; training must not. One flag, set by `_step` when `tp_world > 1`,
because the two callers genuinely want different things from the same forward.

## What this does NOT establish

No 27B number — `pending-remote`. This is the reference implementation; there is
no TileLang kernel, and the op is marked accordingly. The mesh is still not
written, so nothing yet *selects* a `(dp, tp, cp)` layout: this makes TP training
correct, not configurable.

The gate runs at `V=64, world=2`. The arithmetic is world-agnostic but only
world=2 is exercised.

## Rule

**When an op exists for a resource reason, gate the resource, not just the
answer.** The correct-but-wasteful implementation passes every numeric check —
it has to, or it would not be the tempting alternative. Find the property that
separates them (here: the widest tensor formed), assert on that, and run the
wasteful version once to watch the assertion fire.
