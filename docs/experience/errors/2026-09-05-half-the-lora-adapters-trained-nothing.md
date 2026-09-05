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

## What this does NOT establish

**The 27B number is unreconciled.** Applying the pre-fix rule to `param_specs` at
rank 16 predicts 200.8 M adapter params (132.7 M live, 68.1 M dead), but
[the thinking cap](../wins/2026-09-04-the-thinking-cap.md) records the real run's
adapter as **170.8 M / 341 MB** — a 30.0 M gap. The formula reproduces
`add_lora` exactly on a tiny build (15028 == 15028), so the gap is in the 27B
parameter set, not the arithmetic: `load_hf` may not produce what `param_specs`
describes. No checkpoint is reachable from this machine, so the dead share on the
27B is **an estimate, not a measurement**, and every published adapter-params
number stays suspect until someone counts the keys of a real run's
`adapter.safetensors` and buckets them by `.scale`/`.conv1d`.

Nothing in the tree loads `adapter.safetensors` back — only a test reads it — so
no loader was written to reject the stale files. A run's adapter from before this
fix simply carries keys that a post-fix run does not.

## Rule

**A parameter's shape does not identify its role.** `ndim == 2` selected a
quantization sidecar and a conv kernel alongside the linears it was aiming at.
Where a rule picks parameters to train, gate it on what the forward actually
resolves — count the gradients that arrive, which fails for the case you did not
imagine, instead of asserting the names you did.
