# Prefill's growing share is attention, and it runs at 0.75 TFLOPS, V100 sm70, 2026-09-03

> Status: **claim verified, one number in it corrected.** The Amdahl inversion in
> [`wins/2026-09-03-ncols2-raises-loads-per-fma.md`](2026-09-03-ncols2-raises-loads-per-fma.md)
> said the GEMV's share falls with context and that what grows is attention. A profile
> at two contexts confirms the identity — attention is the only class that scales
> superlinearly, 5.07× per token from 512 to 4096 — but the "18% non-GEMV" it quoted
> was the pre-fix figure; post-`ncols=2` it is **27.1%**, and that is what caps
> further GEMV work.

## Context

The inversion was arithmetic on three end-to-end gains, not a measurement. It named
attention because attention is the only quadratic term in a prefill, which is a
story about the model, not an observation about this binary. It was also
load-bearing: it is what priced `ncols=4` and pointed the next tick at attention.

## The instrument was wrong first, and would have shown nothing

`prof_prefill_budget.py` windowed `e.step()` **once**. Above the 512-token chunk
budget that is chunk *one* — tokens 0..512 — whose attention window is identical to a
512-token prompt's. Profiling 512 against 4096 would have compared **the same tick
twice** and reported no context dependence, which reads exactly like "the claim is
false". `--all` now steps until the request reaches decode, and the per-token and
FLOP denominators use the rows actually inside the window.

## Results

Whole-prefill windows, `ncols=2` shipped, 27B NVFP4.

| class | ms/token @512 | ms/token @4096 | ratio |
|---|---:|---:|---:|
| fp4 GEMV | 3.983 | 3.986 | **1.00×** |
| **attention** | 0.211 | **1.069** | **5.07×** |
| GDN | 0.226 | 0.226 | 1.00× |
| elementwise | 0.084 | 0.084 | 1.00× |
| rmsnorm | 0.050 | 0.050 | 1.00× |
| other | 0.047 | 0.047 | 1.00× |

**Every class is flat per token except attention.** Nothing else in prefill grows
with context — not GDN (48 of 64 layers, but its chunk scan is linear in T), not the
elementwise tail, not the norms. The identity in the claim is right, and it is the
*only* candidate: the six other classes reproduce to within 0.1% per token across an
8× context change, which also says the instrument is measuring what it claims.

Shares: attention **4.6% → 19.6%** of GPU; GEMV 86.5% → 72.9%.

## The inversion checks out, once both arms are reconstructed

The profile runs *post*-fix, so its GEMV share is not the one the inversion inferred.
Scaling the measured GEMV back by 1.82× recovers the pre-fix mix:

| ctx | pre-fix GEMV share | predicted gain | measured |
|---:|---:|---:|---:|
| 512 | 0.921 | 1.710× | 1.694× |
| 4096 | 0.831 | 1.598× | 1.585× |

Within 1% at both ends. The inversion was sound; what it could not know was the
*identity* of the remainder, and the number it quoted for the remainder (18%) was
the pre-fix share, not the post-fix one.

## What it decides

**Post-fix, non-GEMV at 4096 is 27.1%.** So further GEMV work is capped:

| further GEMV gain | end-to-end at 4096 |
|---|---:|
| 2× | 1.57× |
| 4× | 2.21× |
| ∞ | 3.70× |

`ncols=4` (ratio 1 : 32, ~+82 registers from 254) would need to be worth ~2× in the
kernel to buy 1.57×, and it is likelier to spill. Meanwhile attention is a **single
kernel, 4310.9 ms of a 22382 ms prefill**, and the roofline says it is not close to
anything:

- **FLOPs**: 3.3 TFLOP of attention math in 4378 ms = **0.75 TFLOPS**, against 31.4
  scalar-FMA peak. 42× off.
- **Bytes**: the same window re-reads **6912 GiB** of f32 K+V — 1695 GB/s against
  900 GB/s HBM, so it is served from cache and is not HBM-bound either.

Neither roofline binds at 0.75 TFLOPS, which points at the kernel's shape rather than
its arithmetic: `paged_attention_split` grids `(KVSPLIT, S*H, B)` and each thread walks
its slice with `block_N=16` **serially**, one query row per thread, f32 throughout on a
card whose f32 has no packed-FMA path.

> **The "next lever" clause that stood here is withdrawn.** The shape diagnosis is
> confirmed — every query row re-reads the window, and batching 512 rows captures only
> 1.56× of a possible ~512× — but attention at *infinite* speed is worth **1.243×** of
> prefill, against 1.57× for a further 2× on the GEMV. Distance from peak is not prize
> size. See
> [`errors/2026-09-03-attention-ceiling-is-1.24x-priority-withdrawn.md`](../errors/2026-09-03-attention-ceiling-is-1.24x-priority-withdrawn.md).

## Rule

**An inversion gives you a magnitude, never an identity.** The arithmetic that said
"18% is not GEMV" was right to 1%; the clause that said "the 18% is attention" was a
guess that happened to be correct, and the guess is the part that steers the next
kernel. Profile before spending a tick on what a share is *made of*.

Second: **a profiler that windows one tick cannot see a context trend.** The defect
here would have produced a confident refutation of a true claim, because two
different `--ctx` values profiled byte-identical work. Check that the two runs of a
comparison actually differ in the variable named.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | attention ms/token, 512 → 4096 | 0.211 → **1.069 (5.07×)** |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | every other class, ms/token | flat to 0.1% |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | attention share of GPU @4096 | **19.6%** (4378 ms of 22382) |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | GEMV share of GPU @4096, post-fix | 72.9% |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | attention TFLOPS @4096 | **0.75** of 31.4 peak (42× off) |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | attention K+V bytes read @4096 | 6912 GiB = 1695 GB/s (not HBM-bound) |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | ceiling on further GEMV work @4096 | 3.70× (non-GEMV 27.1%) |

Reproduce: `scripts/prof_prefill_budget.py --source $CKPT --ctx {512,4096} --all`.
