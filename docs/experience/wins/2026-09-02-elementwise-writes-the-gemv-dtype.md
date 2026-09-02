# Elementwise ops write the GEMV's dtype — V100 (sm70), 2026-09-02

> Status: shipped. Dense decode **+4.0-4.3%** at every context (4096: 37.6 →
> 39.1 tok/s), by deleting 193 of the 305 f32→f16 casts a dense token pays.
> Predicted 3.7% before measuring; measured 4.0-4.3%.

## Context

The sm70 fp4 GEMV reads X in f16 — that is what took it from 127 µs/row flat to
24-45 (`wins/2026-09-01-sm70-gemv-packed-x-f16.md`). But X arrived as f32, so the
dispatch cast it at every launch:

```python
_pad2d(x2[m : m + Mr], Mk, Kp).to(torch.float16)
```

305 launches per dense token, one cast each, over bytes that `rmsnorm_apply` and
`silu_mul` had just written in f32 — two passes over the same values, the second
producing exactly what the first could have.

## What Worked

**The producers write f16 directly.** Both kernels already had a bf16 twin for
sm90 (whose GEMV is bf16-IO), so the fix is the out dtype as a factory argument
and one registry line each — no new kernel source:

```python
"rmsnorm_apply_narrow": lambda t: kernels.make_rmsnorm_apply_bf16(t, out_dtype="float16"),
"silu_mul":             lambda t: kernels.make_silu_mul_bf16(t, out_dtype="float16"),
```

`Backend.gemv_io` names the dtype the GEMV wants, `_rows(x, keep_f16=...)` stops
widening an X that is already right, and past the twiddled ladder one widen
restores f32 for the four branches whose kernels are f32-IO.

**Which casts this can remove is arithmetic, not a guess.** Only the launches fed
by a narrowing producer:

| producer | consumer launches/token |
|---|---:|
| `rmsnorm(input_norm)` | 16 qkv + 48 qkvz |
| `rmsnorm(post_attn_norm)` | 64 gate_up |
| `rmsnorm(final_norm)` | 1 lm_head |
| `silu_mul` | 64 down |
| **covered** | **193 of 305** |

The other 112 (`o_proj`, `out_proj`, `ab`) are fed by attention and the GDN scan,
both f32-IO kernels — narrowing those is a separate change with its own parity
risk. At the task's 1.64 ms/token for 305 casts, 193 of them is 1.04 ms = **3.7%
of a 27.8 ms token**. Measured **4.0-4.3%**, slightly better than predicted
because a removed cast also frees the write bandwidth of the f32 buffer it used
to produce. Predicting within half a point is what makes this a measurement
rather than a hope.

## The mistake worth recording

The first version replaced `rmsnorm_apply` outright for sm70. That also narrowed
`q_norm`/`k_norm` — whose consumers are `rope` and `paged_attention`, both f32.
`rope` calls `_f32()` on its input, so nothing broke and the bench still read
+3.2%. It was silently **worse than doing nothing** on those two calls: a
f32→f16→f32 round trip that drops 13 mantissa bits *and* adds 32 casts rather
than removing them.

Nothing would have caught it. Parity passed (4.93e-04, unchanged), the bench
improved, the suite was green. It is visible only by asking, per call site, what
reads the output — and the scoped version then measured **faster** at every
context (42.3 → 42.7 at 512, 38.8 → 39.1 at 4096), because the round trip was
adding work as well as losing bits. The correct version being the faster one is a
coincidence here, not a rule; the reason to scope it was precision.

The fix is a `narrow=True` flag at the four sites that feed a linear, off by
default, and a separate registry key so the plain `rmsnorm_apply` still exists
for everyone else.

## Rule

**Narrowing a dtype is a property of the edge, not the op.** "sm70's GEMV wants
f16" is true; "sm70's rmsnorm should emit f16" does not follow, because rmsnorm
has consumers that are not the GEMV. Ask what reads the output at each call site
before changing what the op produces.

Second: **a downstream `_f32()` hides a lossy round trip.** Defensive casts at
the top of every op are what let this land green — they guarantee correctness and
therefore guarantee silence. When a change is invisible to the tests, the review
has to be the reasoning, not the run.

Third: **write the predicted number down before the measurement.** 193/305 × 1.64
ms = 3.7% was on record before the bench came back at 4.0-4.3%. Agreement
validates the model; a 2× disagreement would have meant the instrument or the
cast-count was wrong, and I would have known which question to ask.

## Results

| date | commit | machine | target | model | ctx | tok/s before | tok/s after | |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 32 | 41.6 | **43.4** | +4.3% |
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 512 | 41.0 | **42.7** | +4.1% |
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 1024 | 40.4 | **42.1** | +4.2% |
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 2048 | 39.4 | **41.0** | +4.1% |
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 4096 | 37.6 | **39.1** | +4.0% |
| 2026-09-02 | (unscoped, q/k_norm too) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 4096 | 37.6 | 38.8 | +3.2% |

Gates: compile gate 34/34 on the pod, `parity_sm70_gemv_m` worst 4.93e-04 against
a 1e-2 gate (unchanged — the GEMV sees the same f16 values, just produced one step
earlier), CPU suite 177 passed.

Remaining: the 112 launches fed by attention and the GDN scan, worth ~0.60
ms/token (2.2%) on the same arithmetic. Their producers are f32-IO kernels, so
each needs its own narrow variant and its own parity check.

Raw artifact: `scripts/bench_ctx_decode.py`.
