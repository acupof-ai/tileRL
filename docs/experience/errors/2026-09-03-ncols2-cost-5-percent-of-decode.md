# ncols=2 cost 4.9% of dense decode — a default flip measured on one path only, V100 sm70, 2026-09-03

## Context

`ncols=2` shipped as a **default flip on the whole sm70 fp4 ladder** (`_NCOLS = 2`,
applied at every M in `Backend.linear_fp4`) on the strength of a prefill A/B: 1.59×
end to end, with text parity checked on real prompts. Decode was never run.

## What it cost

Dense decode, `bench_ctx_decode.py`, arms nc2 / nc1 / nc2:

| ctx | nc1 | nc2 | nc2/nc1 |
|---:|---:|---:|---:|
| 32 | 43.1 | 41.0 | 0.951× |
| 512 | 42.7 | 40.4 | 0.946× |
| 1024 | 42.1 | 39.9 | 0.948× |
| 2048 | 41.1 | 38.9 | 0.946× |
| 4096 | **39.1** | **37.2** | **0.951×** |

**A uniform 4.9% loss at every context**, and the confirm arm reproduced arm 1 to the
decimal at all five points (41.0 / 40.4 / 39.9 / 38.9 / 37.2), so it is not drift.

## Why the mechanism reverses below the top rung

`ncols=2` wins by raising HFMA2 per LDG — arithmetic per load. That only pays where
the kernel is **compute-bound**. At M=1 the GEMV is bandwidth-bound (84% of its own
byte roofline, `wins/2026-09-02-f16-block-scales.md`), so there is no arithmetic to
win and only the cost remains: the 2col kernel grids `ceildiv(N/2, n_partition)`, so
**the grid halves**.

That lands on shapes with no room. Blocks per SM under the 2col kernel:

| shape | N | 1col blocks | 2col | blk/SM |
|---|---:|---:|---:|---:|
| gate_up | 34816 | 8704 | 4352 | 54.4 |
| qkvz | 16384 | 4096 | 2048 | 25.6 |
| gdn out | 6144 | 1536 | 768 | 9.6 |
| down / attn o | 5120 | 1280 | 640 | **8.0** |

Task #21 had already recorded these small-N decode shapes at **5-33% of peak** —
grid-starved before the change. Halving their grid is exactly the wrong direction.

## Why the microbench said nothing

The accept run read M=1 at **1.05×**, and the same entry states ±4% is the M=1 noise
floor. So the reading was *inside its own error bar* — "no signal", which I read as
"no risk". Two further reasons it could not have caught this: six shapes in isolation
are not the 144-launch decode path, and a microbench at fixed N cannot show grid
starvation interacting with the other 143 launches competing for the same SMs.

## Fix

Gate on the rung, not on a global flag:

```python
nc2 = _NCOLS if Np == N and N % 2 == 0 else 1     # padding gate, unchanged
for m, Mr, Mk in chunks:
    nc = nc2 if Mk >= _NCOLS_MIN_M else 1        # _NCOLS_MIN_M = 32
```

`_sm70_chunks` compiles M=512 as sixteen 32-row chunks, so prefill keeps `ncols=2`
everywhere it matters, while M=1 decode gets the 1-column kernel. One expression; the
ladder already carried the rung.

> **Correction (same day).** This entry first added "and a verify tick (M=B·W≤32, which
> takes the 8 rung)" to that sentence. **That is false.** The sm70 ladder is
> `1/2/4/8/32` with **no rung between 8 and 32**, so any M in 9..32 rounds *up* to 32 —
> and the engine's defaults (`max_batch=4`, `spec_depth=3` → W=4) submit **B·W = 16
> rows**, which take the 32 rung and **keep `ncols=2`**. The gate turns it off for
> dense decode only. Spec decode is a third path, measured separately in
> [`2026-09-03-the-ncols-gate-left-spec-decode-on.md`](2026-09-03-the-ncols-gate-left-spec-decode-on.md);
> the claim was written from the phrase "top rung = prefill" rather than from
> `_sm70_chunks`, which prints the answer in one line.

`tests/test_ncols_contract.py` grew a fifth assertion that walks `_sm70_chunks` for
every rung decode and verify actually take and requires `ncols` off below 32 — plus
the dispatch line itself, since **this is a silent failure mode**: the wrong rung
costs throughput and nothing raises. Negative control verified (deleting the gate
fails the test).

## Verified on both paths

Decode returns to the control exactly — `bench_ctx_decode.py`, tok/s:

| ctx | control (nc1) | ncols everywhere | **gated** | gated/control |
|---:|---:|---:|---:|---:|
| 32 | 43.1 | 41.0 | 43.4 | 1.007× |
| 512 | 42.7 | 40.4 | 42.7 | 1.000× |
| 1024 | 42.1 | 39.9 | 42.1 | 1.000× |
| 2048 | 41.1 | 38.9 | 41.0 | 0.998× |
| 4096 | **39.1** | 37.2 | **39.1** | **1.000×** |

Prefill keeps the win — `bench_prefill.py`, ms/token, against that harness's own HEAD
baseline measured earlier the same day (7.88 / 8.03 / 8.33 / 8.91):

| ctx | HEAD | **gated** | gain |
|---:|---:|---:|---:|
| 512 | 7.88 | 4.91 | 1.60× |
| 1024 | 8.03 | 5.05 | 1.59× |
| 2048 | 8.33 | 5.31 | 1.57× |
| 4096 | 8.91 | **5.86** | **1.52×** |

Compare within one harness only: the 1.59× first reported came from
`ab_prefill_ncols.py`, whose nc1 arm reads 7.81 / 8.17 / 8.66 where `bench_prefill`
reads 7.88 / 8.03 / 8.91 — 1-3% apart, enough to fake a regression if the numerator
and denominator come from different scripts. And gating cannot cost prefill by
construction: `_sm70_chunks(512)` is sixteen chunks of `Mk=32`, all ≥ 32, so every
prefill launch still takes `ncols=2`; the only M=1 call in that window is the closing
decode tick, which now gets the *faster* kernel.

## Rule

**A default flip is a claim about every path, so it needs a number from every path.**
Prefill and decode take the same dispatch and the same kernel family at different M,
and the mechanism that pays at M=32 *inverts* at M=1 — compute-bound versus
bandwidth-bound is not a detail, it is the whole reason the optimization exists.
Before flipping a default, list the paths it reaches and measure each.

Second: **a reading inside its own noise floor is not a green light.** M=1 at 1.05×
against a ±4% floor says the instrument could not see the effect, not that the effect
is absent. When the upside is structurally ~0 (bandwidth-bound) and a real cost is
structurally present (halved grid), the null reading should have prompted a better
instrument, not a ship.

Third: **compare within one harness.** `bench_prefill.py` and `ab_prefill_ncols.py`
disagree by 1-3% on identical code, so a gain computed from one script's numerator and
another's denominator can invent or hide a regression of exactly the size being
measured.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | f6d0805 | V100 32GB | cuda sm70 | qwen38-27b | dense decode @4096, ncols on all M | 37.2 tok/s |
| 2026-09-03 | f6d0805 | V100 32GB | cuda sm70 | qwen38-27b | dense decode @4096, ncols off | **39.1 tok/s** |
| 2026-09-03 | f6d0805 | V100 32GB | cuda sm70 | qwen38-27b | cost of ncols at M=1, all contexts | **0.946-0.951× (uniform 4.9%)** |
| 2026-09-03 | f6d0805 | V100 32GB | cuda sm70 | qwen38-27b | confirm arm vs arm 1 | identical at all 5 contexts |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | dense decode @4096, gated to M>=32 | **39.1 tok/s (1.000× of control)** |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | prefill ms/token @4096, gated | 8.91 → **5.86 (1.52×)** |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | prefill ms/token @512, gated | 7.88 → **4.91 (1.60×)** |

Reproduce: `bench_ctx_decode.py` under `TILERL_NCOLS=2` and `=1`; `bench_prefill.py`
for the prefill column.
