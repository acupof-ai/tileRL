# A GRPO group of 8 fits the card — sm90, 2026-09-03

> Status: Shipped (the levers; `recipes.py` keeps `grpo-gsm8k-27b` at
> `pending-remote` until the recipe itself runs end to end)

## Context

`tilerl train --recipe grpo-gsm8k-27b` (group 8, `max_new_tokens=256`, LoRA
rank 16) reached the training step and died: `torch.OutOfMemoryError` on a
146 MiB buffer with 95.15 of the H20's 95.22 GiB already used. Weights are
23.2 GiB, so the rest was activations the tape was holding.

Shrinking the group was ruled out: the group mean *is* the GRPO baseline, so a
smaller group is a different training signal. TRL rejects `auto_find_batch_size`
for exactly this reason. The metric here is peak allocated bytes for one
training step, at a fixed group of 8.

## What Worked

Three levers, all of them what TRL and AReaL already ship
([rl-sota-parity.md §5](../../rl-sota-parity.md)):

1. **Recompute the MLP block instead of storing it.** `autograd.checkpoint`
   records one tape entry that replays a pure segment under a sub-tape during
   backward; `Model._mlp` wraps `_mlp_body` in it. The MLP is the layer's
   largest pure block — attention and GDN advance their pools, so replaying
   either would recompute against state its own forward moved.
   TRL overrides HF's default to on; every AReaL example sets it.
2. **Micro-batch the group with gradient accumulation.** `rl_step(micro=1)`
   runs one row per backward and sums the parameter gradients before one
   update. The normalizer is the *whole batch's* scored-token count, computed
   before the split, so the update is identical however the rows were chopped
   up — proved by `test_micro_batching_is_the_same_update`, not argued.
3. **Write the logit gradient into the logits.** `cross_entropy_loss_grad` held
   four `[B,T,248320]` f32 tensors at once and `rl_step` allocated a fifth;
   `softmax - onehot` needs no second buffer, which is AReaL's vocab-parallel
   trick and removes TRL's reason to chunk the softmax.

Measured with `scripts/probe_tape_mem.py` on one idle H20, one process per arm.
27B, group 8, T=275, LoRA rank 16 — a probe of the training step, not a full
recipe run, so the 95.04 below is the probe's OOM and the 95.15 above is the
recipe's.

**64 layers — the shipping shape:**

| recompute | micro | CE | weights GiB | step peak GiB | activations GiB |
|---|---|---|---:|---:|---:|
| off | 8 | 5-buffer | — | **OOM at 95.04 in use** | — |
| on | 8 | 5-buffer | 23.22 | 83.28 | 60.05 |
| on | 8 | in-place | 23.22 | 81.24 | 58.01 |
| on | 1 | 5-buffer | 23.22 | 33.78 | 10.56 |
| on | 1 | in-place | 23.22 | **33.54** | **10.32** |

**16 layers — truncated so the pre-fix arm completes and the levers separate:**

| recompute | micro | CE | activations GiB |
|---|---|---:|---:|
| off | 8 | 5-buffer | 41.14 |
| off | 8 | in-place | 41.16 |
| on | 8 | in-place | 23.87 |
| on | 1 | 5-buffer | 5.20 |
| on | 1 | in-place | 4.95 |

Attribution: MLP recompute is **−42%** (41.16 → 23.87), micro-batching a
further **−79%** (23.87 → 4.95), and the in-place cross-entropy is worth
**0.25 GiB**.

**The survey's headline does not hold here.** Both TRL and AReaL name the
vocab-sized logits tensor as the true peak. Ours is not: at 16 layers with the
activations stored, replacing four vocab tensors with one moved the peak by
0.02 GiB, because the peak sits inside the layer-stack backward, after the
cross-entropy buffers are already freed. The in-place CE is worth 2.04 GiB at
64 layers with the whole group in one backward and 0.24 GiB at `micro=1` — real,
and two orders of magnitude smaller than the other two levers. Our vocab is
248320, larger than either stack's, and it still is not the peak. A fused
chunked LM head (Liger's answer) would buy nothing on top.

## Rule

On this model the training peak is stored layer activations, not the logits.
Reach for gradient checkpointing and micro-batching first; the vocab-sized
tensors are worth low single-digit GiB. And micro-batching is the only way to
shrink a GRPO step without changing the training signal — never the group.

## Results

| date | commit | machine | target | model | group | micro | peak GiB | activations GiB |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-03 | a702c9a | H20 (95.22 GiB) | cuda sm90 | qwen38-27b | 8 | — | OOM | — |
| 2026-09-03 | this | H20 (95.22 GiB) | cuda sm90 | qwen38-27b | 8 | 1 | 33.54 | 10.32 |

Raw artifacts: `scripts/probe_tape_mem.py` (the four-arm harness),
`/work/memab.log`, `/work/memab2.log`, `/work/grpo100b.log` on the pod.
