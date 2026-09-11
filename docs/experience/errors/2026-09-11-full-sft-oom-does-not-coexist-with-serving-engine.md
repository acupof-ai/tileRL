---
question: Why did the P2.0 recapture probe OOM at the training backward on a free 96 GiB H20, and what actually fits?
status: measured
source: P2.0 recapture-after-update probe on H20 card 4, 27B NVFP4, 2026-09-11, three runs (scripts/recapture_update.py)
---

# Full-parameter SFT does not coexist with the captured serving engine on one H20; LoRA does

## Context

P2.0's gate keeps the captured decode graph across an optimizer step and proves
post-step greedy tokens equal a fresh eager engine on identical weights. The
first probe ran it as a full-parameter SFT step on the 27B served model, in the
same process that held the serving engine (weights, KV pool, GDN state pool,
captured graph memory pool). The backward pass of the first `train_step` OOMed
on card 4 — not at launch, not at capture, but inside the tape backward. Three
runs narrowed it; none was a fragmentation or block-pool sizing problem.

The card is 95.22 GiB total. Verbatim from the three runs:

| run | optimizer | KV blocks | alloc conf | tried to allocate | free at OOM | held by process |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| recapupd3 | plain AdamW | 2048 | default | 340 MiB | 63.56 MiB | 95.15 GiB |
| recapada | Adafactor | 2048 | default | 4.74 GiB | 1.51 GiB | 93.70 GiB |
| recapada16 | Adafactor | 64 | expandable_segments | 4.74 GiB | 3.69 GiB | 91.53 GiB |

## Root cause

A full-parameter step on the 27B needs far more resident than the served engine
leaves, and shrinking the KV pool cannot close the gap:

- **Served nvfp4 weights ≈ 23.3 GiB**, but full SFT builds with `keep_master=True`,
  retaining a **bf16 master per trained linear ≈ 54.6 GiB** on top of the served
  faces the engine forwards against — ~78 GiB of weights before any activation.
- **Plain AdamW is the LoRA-only optimizer for a reason.** It holds two f32
  moments per trained parameter; the code comment in `autograd.py` prices that at
  **200.4 GiB on the 27B**. The non-streaming `train_step` path also holds every
  parameter gradient until the update (the `train.py` comment prices "every weight
  gradient coexisting" at **50.1 GiB**). recapupd3 died holding both.
- **Adafactor removed the moments** (0.03 GiB state, streams one grad at a time),
  and that alone dropped the held footprint from 95.15 to 93.70 GiB — but the
  backward still wanted 4.74 GiB of activation/working tensors with only 1.51
  GiB free. Cutting the KV pool from 2048 blocks (~4.3 GiB) to 16 and turning on
  `expandable_segments` raised free memory to 3.69 GiB and still left a
  ~1.05 GiB shortfall. The remaining pressure is the f32 backward tape over the
  forward activations against ~78 GiB of weight masters + served faces + graph
  pool — a footprint property, not a tunable.

The memory ledger already names the parts: `autograd.py` ("Adam's m+v on the 27B
is 200.4 GiB; Adafactor is 0.03 GiB") and `train.py` (50.1 GiB of held
gradients), and `cli.py` routes full SFT to Adafactor with `drop_quantized`
served faces precisely because the two do not share a card.

## Fix (the gate that fits)

Run the recapture gate on the **LoRA path**, which is what P1 actually trains:
a frozen fp4 base (`keep_master=False`, no 54.6 GiB masters) plus a rank-16
adapter (~124.8M params) updated by the same AdamW whose `step_one` ends in the
in-place `p.copy_()` the kept graph depends on. Two ordering facts the probe
exposed:

1. **`add_lora` must run AFTER `build_engine`.** Building materializes
   `model.params` onto the device; an adapter attached before points at the
   pre-materialize CPU tensors, the forward never reads those objects, and
   `train_step` asserts "tape produced no parameter gradients." Production
   `cli.py` attaches in the post-build order.
2. The step has to be strong enough to move greedy tokens while the token-
   equality property is what is gated. rank-16 / seq-64 / lr 1e-4 trained but did
   not change the greedy argmax; seq-256 / lr 0.02 did.

Result on card 4 (`/work/recapture_lora4.json`, 27B nvfp4):

```json
{
  "graphs_held_before": 1, "graphs_held_after": 1,
  "no_step_unchanged": true, "update_changed_tokens": true,
  "recaptured_equals_eager": true, "verdict": "PASS"
}
```

The kept decode graph replayed updated adapter values; post-step captured tokens
were bit-equal to a fresh eager engine carrying identical post-step adapters;
the no-step control was unchanged and the graph was held, not rebuilt.

## Rule

**Full-parameter SFT cannot coexist with the captured serving engine on one
H20** — 54.6 GiB of bf16 masters plus the f32 backward tape against 23.3 GiB of
served faces and the graph pool exceed the card even with Adafactor, a 16-block
pool, and expandable segments. **LoRA coexists** (frozen base, 124.8M adapter),
and that is the P1/P2.0 training path. Full-SFT weight refresh is a separate
process: ISO merge or the served-fp4 repack, not an in-place step beside the
engine. When a training step OOMs at the backward on a coexistence gate, first
ask which optimizer and whether the bf16 masters were built — not whether the
KV pool was too big.
