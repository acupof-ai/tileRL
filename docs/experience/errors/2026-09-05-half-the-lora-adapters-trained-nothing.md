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
(`/work/tilerl-p1/runs/1fa1e58388a2`, 341 741 400 bytes, 2086 keys), grouped by
the suffix of the stem each LoRA hangs off:

| bucket | keys | params |
|---|---:|---:|
| live | **996** | **124.838 M** |
| dead `.scale` | 528 | 37.650 M |
| dead `.wscale` | 466 | 0.409 M |
| dead `.conv1d` | 96 | 7.867 M |
| dead, total | **1090** | **45.926 M** |
| total | 2086 | **170.76 M** |

**1090 of 2086 keys are dead — 52%.** The param share stays small because the
466 `.wscale` sidecars carry 0.409 M between them: tiny tensors, and they are
what blocked the load.

The prediction was 200.8 M (132.7 M live, 68.1 M dead). The thinking cap's
170.8 M / 341 MB was right and the whole 30.0 M gap was in the prediction, which
missed high in every component — a method fault, not noise, from rounding up per
key and multiplying by a key count never checked against a checkpoint. 200.8 M is
struck rather than kept as one end of a range.

**The first two measurements of this file agreed with each other and were both
wrong.** Each bucketed on `.scale` and `.conv1d` and took live as the complement,
so `.wscale` fell into live and the split read 1462/624. The second measurement
was run on the pod specifically to avoid transcribing the first, but it reused
the first's category list — a re-execution of one partition on the same bytes,
which can only agree. Enumerating every suffix present takes the same one query
and shows `.wscale` immediately.

What actually found it: `--load-adapter` refused this checkpoint with **1090
unknown keys**. #104's gate caught a partition error that two people measuring by
hand did not.

This run launched from `91977a8`, before the fix landed at `863a257`, so those
1090 dead adapters are **in this checkpoint** — carried through 100 steps, each
with two AdamW moments. That is what makes the loader gate behavioural rather
than structural: the file parsing proves nothing when half its keys were never
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
