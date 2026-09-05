---
question: Does every LoRA adapter `add_lora` creates actually receive a gradient?
source: cpu (this Mac), tiny model fp4 and bf16, rank 4, one forward + `tape.backward`
---

# 32 of 64 LoRA adapters received no gradient, and rode in the optimizer anyway

## Context

Chasing an unrelated question — whether the tape backwards through the fused
`.ab` projection — meant listing which parameters `add_lora` attaches to. Two of
them had no business being there.

## Root cause

`add_lora` identified a linear weight by **shape**: any 2-D parameter that is not
itself an adapter and owns no `.wq`/`.w8` (`model.py:415-421`). Two kinds of
tensor pass that test without being a linear weight:

- **A quantized weight's scale sidecar.** An fp4 `.scale` is `[N, K/32]` — 2-D,
  and `layers.0.q_proj.scale.wq` does not exist, so it looked like a dense base.
- **`conv1d`.** `[qkv, 4]`, 2-D, and it goes to the GDN kernel, not a linear.

Nothing read either one. `_linear` resolves `lora_a` from the **base** key
(`model.py:161`), so an adapter parked on `q_proj.scale` is never looked up.

Measured, tiny model, rank 4, one forward and one `tape.backward`:

| | adapters | receiving a gradient |
|---|---:|---:|
| fp4 | 64 | 32 |
| bf16 | 34 | 32 |

The dead ones were returned as `trainable`, handed to `AdamW` (two moments
each), and written into `adapter.safetensors`. The sidecar half exists only
under fp4; **`conv1d` is dead in both precisions**, which is why the gate runs
both.

## Fix

One condition, in the same shape test: a key whose **parent** owns a `.wq`/`.w8`
is a sidecar, and `.conv1d` is excluded by name.

```python
and not any(k.endswith(x) for x in (".lora_a", ".lora_b", ".conv1d"))
and f"{k}.wq" not in model.params
and f"{k}.w8" not in model.params
and f"{base}.wq" not in model.params
and f"{base}.w8" not in model.params
```

fp4 tiny goes 64 adapters → 32, all 32 receiving a gradient. bf16 goes 34 → 32.
No live adapter is lost: the count of adapters *with* a gradient is 32 before and
after, in both precisions.

Gate: `test_every_adapter_receives_a_gradient` counts gradients rather than
matching names, so a new dead target nobody anticipated fails it too. Two
negative controls, each failing on its own — reverting the sidecar guard leaves
30 dead, reverting only the `.conv1d` exclusion leaves 2.

## The 27B number, reconciled

Counted from a real run's `adapter.safetensors`
(`/work/tilerl-p1/runs/1fa1e58388a2`, 341 741 400 bytes, 2086 keys), bucketed by
the stem the LoRA hangs off:

| | predicted | measured |
|---|---:|---:|
| live | 132.7 M | **125.25 M** (1462 keys) |
| dead `.scale` | — | 37.65 M (528 keys) |
| dead `.conv1d` | — | 7.87 M (96 keys) |
| dead, total | 68.1 M | **45.52 M** (624 keys) |
| total | 200.8 M | **170.76 M** |

The thinking cap's 170.8 M / 341 MB was right; the whole 30.0 M gap was in the
prediction. Dead share 26.6% measured against 33.9% predicted.

**The estimate was biased, not noisy.** Both buckets missed in the same
direction — live by 7.5 M, dead by 22.6 M — because the formula rounded up per
key and multiplied by a key count nobody had verified against a checkpoint. A
component-wise error that is positive everywhere is a method fault; 200.8 M is
struck rather than kept as one end of a range.

This run launched from `91977a8`, before the fix landed at `863a257`, so those
624 dead adapters are **in this checkpoint** — carried through 100 steps, each
with two AdamW moments. That is what makes the loader gate behavioural rather
than structural: the file parsing proves nothing when 528 of its keys were never
going to attach, so `test_a_loaded_adapter_actually_changes_the_output` decodes
with and without and requires the tokens to differ.

Nothing in the tree loaded `adapter.safetensors` back when this was written; the
loader and that gate landed in #104.

## Rule

**A parameter's shape does not identify its role.** `ndim == 2` selected a
quantization sidecar and a conv kernel alongside the linears it was aiming at.
Where a rule picks parameters to train, gate it on what the forward actually
resolves — count the gradients that arrive, which fails for the case you did not
imagine, instead of asserting the names you did.
