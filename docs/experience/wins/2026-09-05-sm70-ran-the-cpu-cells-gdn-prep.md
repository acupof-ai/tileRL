# sm70 ran the CPU cell's gdn_prep, and every thread computed the same 128 columns — V100 sm70, 2026-09-05

> Status: Shipped. Parity PASS on sm70, and the end-to-end number is measured on
> the live server.

## Context

Profiling an 11K prefill on the V100 while looking for what a prefix cache would
be worth. `gdn_prep` came back as **53.5% of a prefill tick's GPU time** — 3121 ms
of 5831 across 48 calls, 65 ms per layer at T=2048.

That was not the expected answer. sm70 has no `gdn_state_scan`, so
`_gdn_chunk_wy` falls into its else branch and the chunkwise-WY core runs as
`reference.gdn_chunk_core`, a torch bmm loop. I predicted **that** would dominate.
It is **1.5%** — 87 ms, of which `aten::bmm` is 53.6 and the triangular solve 10.3.
The missing WY schedule costs almost nothing, because those bmms are large matrix
multiplies the GPU is good at.

## Root Cause

**sm70 was running the CPU cell's kernel.** `_resolve` showed the two makers are
different objects: sm70 got `kernels.make_gdn_prep` (the f32 CPU-lineage one),
sm90 got `kernels_gdn.make_gdn_prep_bf16`.

The CPU source wraps its work in `T.serial(DK)` / `T.serial(DV)` loops. The launch
passes `threads=vd` — 128 — and the body **binds no thread index**. So all 128
threads execute the same 128-column loop.

Measured rather than reasoned, three arms at T=2048, NVH=48, DK=128:

| arm | ms | vs shipped |
|---|---:|---:|
| KER=4, threads=128 (shipped) | 264.33 | — |
| KER=2, threads=128 | 174.60 | 1.51x |
| KER=4, threads=32 | 68.08 | 3.88x |
| **KER=4, threads=1** | **54.04** | **4.89x** |
| KER=4, threads=128, DK=64 | 72.45 | 3.65x |

Fewer threads doing **identical work** is 4.89x faster, while halving the conv taps
moves it only 1.51x. So it is redundancy, not read traffic — which decides the fix.

## What Worked

Point sm70 at sm90's one-thread-per-column schedule, with the IO dtype
parameterised:

```python
"gdn_prep": lambda t: kernels_gdn.make_gdn_prep_bf16(t, "float32"),
```

Parameterised rather than copied because the two cells differ in **six dtype
literals and nothing else** — the schedule is the entire point. The CPU cell stays
as the CPU twin.

| | before | after | ratio |
|---|---:|---:|---:|
| kernel, T=512 | 66.49 ms | **0.17 ms** | **397.9x** |
| prefill tick, T=2048 | 5830.7 ms | **2773.5 ms** | **2.10x** |
| tick per token | 2.847 ms | 1.354 ms | |
| **server, 11019-token request (warm)** | **163.1 s** | **100.2 s** | **1.63x** |
| server, first request (incl. cold JIT) | 182.5 s | 100.7 s | 1.81x |
| one-time fp4 JIT | 19.4 s | 0.5 s | |
| ms per prompt token, end to end | 14.68 | **9.00** | |
| peak VRAM | 27036 MiB | 26668 MiB | |

Parity on sm70 against `reference.gdn_prep`, all six outputs:
`max|delta|` **1.192e-07** worst (Ko), `allclose(rtol=1e-2)` True. Against the old
kernel: **1.907e-06** worst. Same math, so they must agree, and they do.

**398x is the shape of a kernel that did not run, so that was checked**: all six
outputs are 100% non-zero with sane magnitudes (`Vo` mean 7.48e-01, `Go` 8.95e-01).
It ran.

The tick's 2.10x lands against a **2.15x Amdahl bound** from prep's 53.5% share —
1.354 ms/token measured against 1.323 predicted. That agreement is the evidence the
gain is real and not an artefact of the profiler window. `fp4 gemv` is now **77.1%**
of the tick and is the next thing to look at.

## Rule

A shared kernel name is not a shared kernel. `_resolve` the maker per arch and
compare the objects — sm70 and sm90 both had `gdn_prep` and they were different
functions, one of them a CPU schedule launched with 128 threads.

And when a kernel is slow, separate redundancy from traffic before rewriting:
holding the work fixed while varying `threads` answered it in one run, and the
answer chose the fix.

## Results

| date | commit | machine | target | model | metric | before | after |
|---|---|---|---|---|---|---:|---:|
| 2026-09-05 | (this) | V100-SXM2-32GB | cuda sm70 | 27B NVFP4, draft d1 | `gdn_prep` T=512 | 66.49 ms | **0.17 ms** |
| 2026-09-05 | (this) | V100-SXM2-32GB | cuda sm70 | 27B NVFP4, draft d1 | prefill tick T=2048 | 5830.7 ms | **2773.5 ms** |
| 2026-09-05 | (this) | V100-SXM2-32GB | cuda sm70 | 27B NVFP4, draft d1 | 11019-tok request, warm | 163.1 s | **100.2 s** |

**Not reconciled:** the tick profile says 2.10x and the server says 1.63x. The gap
is real work outside the profiled tick — HTTP, template render, tokenize, sampling,
chunking overhead — and I have not attributed it. Both numbers are reported as
measured rather than one derived from the other.

**Also not claimed:** anything about sm90. It already had this schedule; only the
registration for sm70 changed, and `make_gdn_prep_bf16`'s default stays `bfloat16`.

Raw artifacts: `torch.profiler` self-CUDA tables before and after at T=2048;
parity and A/B script output; `~/serve70c.log` boot at `2026-09-05T11:38:33+08:00`.
