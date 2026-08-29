# Prefill: the last cheap lever is closed; the gap is two kernel projects — 2026-08-29

> Status: the cheap levers are closed. Prefill is 2237 tok/s. What that is
> "behind" depends on which sglang row you pick, and the answer is not one
> number — see "What the target actually is" below.

## Why this was re-measured

`gdn_chunk_core` — the torch-level chunkwise-WY path behind
`TILERL_GDN_CHUNKWISE` — was rewritten today to call the new `_gdn_chunk_fwd`,
the same helper the chunked backward uses. The earlier rejection of torch
chunkwise measured DIFFERENT code, so quoting it at the new code would have
been assuming rather than knowing.

Prefill suite, 27B, H20 GPU 7, three lengths each:

| GDN path | len 512 | len 2048 | len 8192 | vs fused |
|---|---:|---:|---:|---:|
| **fused kernel (shipped)** | **2237** | 2218 | 2143 | 1.00x |
| torch chunkwise C=16 | 493 | 489 | 496 | 0.22x |
| torch chunkwise C=32 | 841 | 838 | 834 | 0.38x |
| torch chunkwise C=64 | 1123 | 1085 | 1105 | 0.50x |

It loses 2-4.5x and improves monotonically with chunk size — the launch-bound
signature, since a bigger chunk is fewer launches for the same work. Even at
C=64 it is half the fused kernel. **The rejection holds for the rewritten code.**

(The same helper is a 1.63x WIN in the backward. Nothing contradictory: the
backward it replaced ran ~28 launches per TIME STEP, so chunking cut launches
there; the forward it competes with is already one fused kernel per layer.)

## What the target actually is

`2026-08-28-vs-sglang-h20.md` is the source of the 4022, and reading it again
changes the framing twice:

| | prefill tok/s (len 512, B=1) |
|---|---:|
| tileRL, native NVFP4+FP8 | **2237** |
| sglang bf16 | 2512 |
| sglang online fp8 | 4022 |

- **Against sglang's bf16 we are at 0.89x, not 0.56x.** I have been quoting the
  fp8 row as "the" target without saying which row it is.
- sglang **cannot run NVFP4 on Hopper at all** (`NotImplementedError: Current
  platform does not support w4a4 nvfp4`), so both of its rows are a
  dequantized-to-bf16 checkpoint — and that checkpoint **emits garbage**
  (MMLU 0/1000, completions like `'束'`). That entry says so itself and asks
  for the numbers to be re-confirmed once the conversion is fixed. Our 2237 is
  on weights that score correctly.

So it is a kernel-shape comparison, which is still worth having, but "we are
1.80x off SOTA" overstates what was measured.

## The gap, as arithmetic

Taking the fp8 row at face value anyway, since it is the fastest thing measured
on this card:

- Prefill is 913.9 ms where 4022 tok/s implies 509 ms.
- GDN is 27.6% of it. **Delete the GDN kernel entirely — set it to zero — and
  prefill reaches 3160 tok/s, still short of 4022.**
- So closing it needs a near-perfect linear-attention kernel AND about 1.36x
  more from the GEMMs, which run at 59% of fp8 peak. sglang's own fp8 prefill
  is 1.6x its bf16 prefill on identical math, which is where to look: the gap
  is concentrated in fp8 GEMM efficiency, not spread everywhere.

Every cheap lever on the GDN kernel has now been measured and rejected:
chunkwise-WY (three times, two implementations), the V split (duplicates the
per-block q/k work), shared-memory state (2.5x slower — the old note was
right), the q/k prologue (2-3%, under the gate), the ptxas register level, and
the K split (its cost model is verified free; three attempts could not make
KSP>1 compile). The kernel is register-limited at 255 registers/thread with
6.25% occupancy and 4.5M local-load sectors per launch — it spills, and every
cheap fix moves which resource binds without moving the time.

## One more probe, which opened nothing (and a counter trap)

ncu on the prefill kernels themselves, len 512, the two that matter:

| | DRAM | tensor (hmma) | occupancy | issue |
|---|---:|---:|---:|---:|
| `linear_fp8` | 18.1% | 17.3% | 11.4% | 11.0% |
| `gdn_chunk_fused` | 0.23% | 0.13% | 6.25% | 19.2% |

The 17.3% looks like 5x of headroom and is **not usable**:
`sm__pipe_tensor_op_hmma_cycles_active` counts the fp16 HMMA pipe, and Hopper's
fp8 GEMM issues wgmma/QGMMA, which it does not count. The 59% figure quoted
throughout comes from measured throughput instead (176 of 296 TFLOP/s,
[wins/2026-08-25-fp8-prefill-n64-tile.md](../wins/2026-08-25-fp8-prefill-n64-tile.md)),
and a wgmma kernel at 59% of FLOP peak on 11% occupancy is unremarkable — wgmma
is async and does not need resident warps to feed the tensor pipe.

`gdn_chunk_fused` is the one thing this confirmed rather than assumed: DRAM and
tensor at ~0 with 19.2% issue and 6.25% occupancy is a kernel bound by the
dependency chain of its serial scan, which earlier entries had inferred.

## The one lever left, scoped rather than gestured at

GDN is 27.6% of prefill and the counters now say WHY rather than suggest it:
`gdn_chunk_fused` runs at **6.25% occupancy, 19.2% issue, ~0% DRAM and tensor**
— bound by the dependency chain of the serial scan it runs inside the kernel.

Two things are true today that were not before:

- `reference._gdn_chunk_fwd` is a chunked formulation **proven equal to the
  serial scan to 1e-15 in f64**, written for the training backward. It is an
  executable spec for a chunked forward KERNEL.
- All three previous rejections of chunkwise-WY were **torch-level** and all
  three lost for the same reason — launch overhead (0.22x / 0.38x / 0.50x at
  chunk 16 / 32 / 64, measured again today against the rewritten code). None of
  them tested the chunked algebra INSIDE one TileLang kernel, which is a
  different thing: matmul shapes with one launch.

Reference to port: `tilelang/examples/gdn/example_chunk_delta_h.py`. Its
interface matches ours more than expected — `(B, S, H, DK/DV)` with DK=DV=128,
and `use_g` / `use_initial_state` / `store_final_state` are all there.

Two mismatches to plan for, not discover:

1. It covers only the **inter-chunk state pass**; `W` and `U` are its INPUTS.
   The WY/UT transform that produces them (`M = (I+L)^-1`, `U = M(bV)`,
   `W = M(beK)`) is upstream and not in that example.
2. It assumes **one head count**, `(B,S,H,DK)`. We have 16 key heads to 48
   value heads. `_gdn_chunk_fwd` broadcasts with `repeat_interleave`, so a
   literal port reads K and W 3x. That is exactly the trap in
   [v-split-duplicates-qk](2026-08-29-v-split-duplicates-qk.md).

Even at best this is bounded: halving GDN's share takes prefill 2237 -> ~2650,
and zeroing it entirely leaves 3160 against 4022. It is the largest single item
and it does not on its own reach the target.

## Rule

State a gap as arithmetic, not as a verdict — and name which measurement the
target came from, including which counter. Reading fp8 utilisation off the HMMA
pipe would have manufactured a 5x lever that does not exist. "Not SOTA" invites another round of tuning; "zeroing the
single largest kernel still leaves you 21% short" names the size of what is
left. But I also wrote that this needs "an algorithm or a machine, not tuning",
and that was overreach: two hard kernel projects (a linear-attention kernel
that is not register-limited, and fp8 GEMMs at 80% of peak instead of 59%)
are hard, not impossible. A comparison whose baseline runs a checkpoint that
emits garbage deserves the caveat carried with the number, every time it is
quoted.
