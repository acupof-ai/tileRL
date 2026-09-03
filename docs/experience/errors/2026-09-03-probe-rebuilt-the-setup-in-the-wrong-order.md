---
question: Why did a standalone 27B probe produce zero parameter gradients when the same call works inside `tilerl train`?
status: measured
source: tileRL, 27B on card 6, measured 2026-09-03 while pricing the GRPO batch width
---

# A probe that rebuilds the training setup in its own order gets a different object graph

`scripts/probe_width_jit_27b.py` calls `rl_step` on the real checkpoint to price
one thing: how much a novel batch width costs in TileLang compile time. It does
what `_train_adapters` does — load, attach LoRA, optimize — and it failed twice
before producing a number, both times for reasons that have nothing to do with
what it was measuring.

## Root Cause

**`add_lora` before `backend.materialize` silently detaches every adapter.**

`Backend.materialize` rebuilds any param whose device or dtype differs from the
backend's, and the rebuilt tensor is a **new object with a new `id()`**. The tape
accumulates gradients by `id()`. So adapters attached to the pre-materialize
tensors point at objects the forward never reads: the forward runs, the backward
runs, and not one parameter gradient lands.

`_train_adapters` never hits this because `build_engine` materializes first and
the ordering is load → `build_engine` → `add_lora`, with a comment at
`cli.py:237` saying exactly why. A standalone probe has no `build_engine`, so
the ordering has to be explicit — and getting it wrong is invisible except for
one assert.

Second failure, same run: **`load_hf(..., keep_master=True)` OOMs the backward**
at 95.01 of 95.22 GiB. LoRA on a frozen base needs no bf16 masters (~27 GB on the
27B); `_train_adapters` passes `keep_master=False` for that reason. I copied the
flag from `_train_full`, which is full-parameter SFT and does need them.

## Fix

```python
# materialize BEFORE add_lora: the new object has a new id()
model.params = be.materialize(model.params)
trainable = add_lora(model, rank=16)
```

and `keep_master=False` for any LoRA path.

What caught it was `_step`'s assertion, whose message names the cause outright:

    train_step: tape produced no parameter gradients — either the recording seam
    is missing (backend ops not recorded), or a trainable tensor is not the one
    the forward read: materialize() rebuilds any param whose device/dtype
    differs, and the new object has a new id()

That is the second time in one day this assert paid for itself, and it is the
only thing standing between this mistake and a probe that reports plausible
step times while training nothing.

## A third defect in the same probe, found only by disagreeing with itself

The first two runs shared `TILELANG_CACHE_DIR` with each other. Widths compiled
by run 1 were cache **hits** in run 2, which my probe still labelled `novel`:

| run | reported compile cost | why it is invalid |
|---|---|---|
| n=3 | 7.32 s (ratio 1.3x) | one of three "novel" widths was a hit (T=383, 23.3 s) |
| n=6 | **0.60 s (ratio 1.0x)** | three of six were hits; the median is mostly hits |
| clean cache | cold warm-up alone is **170.9 s** vs 24.2 s warm | — |

The n=6 run's `ratio 1.0x` reads as "the effect does not exist on the 27B",
which is the conclusion a shared cache manufactures. Only the two uncontaminated
points (T=380, T=379: 32.3 / 32.5 s against a 23.1 s repeat) carry a number, and
it is **~9.3 s on a 23.1 s step, about 1.4x** — not the 530x the same mechanism
produces on tiny, where the step's real compute is 71 ms and compile time is
therefore the whole measurement.

## Rule

A probe that reconstructs a training setup outside the CLI must reproduce the
CLI's **order**, not just its calls: materialize before attaching anything the
tape must see, and copy the flags from the path you are imitating
(`keep_master=False` for LoRA), not from the neighbouring one. And give a
compile-cost probe a **private cache directory** — a shared one turns a hit into
a "novel" measurement and manufactures the conclusion that compilation is free.
Two runs of the same probe disagreeing 1.3x vs 1.0x is the tell; the quantity
being varied was not the one in the label.
