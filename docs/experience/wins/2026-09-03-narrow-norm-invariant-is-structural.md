# The narrow-norm invariant is now structural, not remembered — 2026-09-03

## Context

`backend.rmsnorm(..., narrow=True)` makes the kernel write `gemv_io` (f16 on sm70)
instead of f32. That is free when the next op is the fp4 GEMV, which narrows anyway,
and it bought +4.0-4.3% dense decode
([wins/2026-09-02-elementwise-writes-the-gemv-dtype.md](2026-09-02-elementwise-writes-the-gemv-dtype.md)).

It is not free for any other consumer. rope, attention and the GDN scan are f32, so
a f32 → f16 → f32 round trip there drops 13 mantissa bits for nothing.

**This already went wrong once on this branch.** The first version narrowed
`rmsnorm_apply` for all of sm70, which silently degraded q_norm/k_norm — and every
gate passed, because `rope` calls `_f32()` defensively. The defensive widening
restores the dtype, not the bits. The fix was to scope the flag to four call sites
where the only consumer is a linear, and that scoping was left as a fact someone
would have to remember.

## What Worked

A **structural** gate, not a numeric one. Nothing numeric can catch this — the
downstream `_f32()` makes the wrong version type-correct and quietly less accurate.
So the test parses `model.py` and asserts a dataflow property:

```python
ALLOWED = {"_linear", "_base_linear", "_add_via"}

for name, line, fn in _narrow_targets(ast.parse(MODEL_PY.read_text())):
    bad = sorted(set(_consumers(name, line, fn)) - ALLOWED)
    assert not bad, f"model.py:{line}: narrowed `{name}` flows into {bad}"
```

Four assertions in `tests/test_narrow_norm.py`:

1. every `narrow=True` output reaches nothing but a linear (the invariant itself),
2. q_norm/k_norm are never narrowed (the specific regression that shipped),
3. `backend.rmsnorm` gates the narrow kernel on `gemv_io != float32`, not on the
   flag alone — otherwise sm90 and cpu lose precision at all four sites,
4. `reference.rmsnorm` ignores `narrow` and stays f32, which is what makes the CPU
   twin a real parity target rather than f16-vs-f16.

It runs on this GPU-less machine in 0.9 s, because it reads source, not a card.

## Negative controls, both run

| edit | result |
|---|---|
| `h2 = backend.rope(h, ...)` after the narrowed `post_attn_norm` | **fails**, naming `['rope']` |
| `backend.rmsnorm(qkv, params["...q_norm"], ..., narrow=True)` | **fails**: `model.py:209: q_norm/k_norm narrowed` |

The first control is exactly the shape of the original bug — a second, f32 consumer
of a narrowed tensor. Tree restored and clean after both.

## Rule

**When a defensive cast makes the wrong version type-correct, the gate has to be
structural.** `_f32()` in rope and attention is good defensive code and it is also
what hid a 13-mantissa-bit loss through an entire test suite. Any invariant whose
violation is *absorbed* downstream cannot be tested by comparing numbers; test the
dataflow that the invariant is about.

Second: **a per-call-site invariant with no gate is a fact someone has to remember,
and this branch has already proved nobody does** — including me, one day after
fixing it.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this change) | Mac | cpu | — | suite | 182 passed, 4 skipped |
| 2026-09-03 | (this change) | Mac | cpu | — | negative controls | 2/2 fail correctly |

Docs/test-only: no runtime code changed, so there is nothing to bench.
