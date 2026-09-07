# The fp4 backward was running Ampere MMA on a Hopper card: 2.77x on the kernel, 1.32x on the backward — H20 sm90, 2026-09-07

> Status: ACCEPT. Two values at one call site (`backend.py:1126-1134`): the dX tile 64 -> 128 and
> its thread count 64 -> 128. `backward_secs` **31.339 -> 23.755** (`--no-instrument`, C=128
> recipe), **1.319x**. The kernel itself goes 5.47-5.62x a same-shape bf16 GEMM down to
> **1.96-2.05x**. `_THREADS` is left alone.

## Context

`linear_fp4_bwd` is 18.82% of the GRPO backward — 12.981 s over 2112 calls
([where-the-backward-goes](2026-09-07-where-the-backward-goes.md)) — and it ran at 24.3-24.8
TFLOP/s where `torch.matmul` on bf16 weights at the same shape reaches 135-137. Reading the
kernel explained part of that and not the rest: the dequant sits inside the pipelined loop, so
every M-tile re-decodes the whole weight (1.78 G element-decodes against 0.09 G unique, 20x,
exactly the M-tile count), but priced at 4-12 ops/element against a 20-45 TOP/s ALU that is
0.16-1.07 ms of a 9.39 ms call. HBM traffic 4.57 GiB = 1.2-1.5 ms. GEMM floor 1.66 ms. Those
overlap rather than add, so **at least 5.9 ms had no mechanism**.

Two probes went looking in the wrong place before the third looked at the emitted code.

## The two knobs, and what they were worth

Card 6, solo box, M=1280 (`micro=1` splits the group, so one backward sees one row's tokens),
bN=64, 12 reps, median of `perf_counter` with a sync per rep, wq/scale sliced from the loaded
checkpoint's own packed tensors. Both arms built through the shipped
`make_linear_fp4_bwd_mma` — a hand-copied kernel body is a second kernel sharing a name.

| shape | calls | bM64/t64 (shipped) | bM64/t128 | bM128/t64 | **bM128/t128** | bf16 |
|---|---:|---:|---:|---:|---:|---:|
| N=17408 K=5120 (gate, up) | 896 | 9.451 | 4.837 | 7.523 | **3.439** | 1.692 |
| N=5120 K=17408 (down) | 448 | 9.258 | 4.892 | 7.061 | **3.309** | 1.681 |

Priced on the real call counts, the 1344 MLP calls only:

| cell | seconds | vs shipped |
|---|---:|---:|
| bM=64, threads=64 (shipped) | 12.616 | — |
| bM=128, threads=64 | 9.904 | 1.274x |
| bM=64, threads=128 | 6.525 | **1.933x** |
| bM=128, threads=128 | **4.563** | **2.765x** |

**The thread count is worth more than the tile, and neither is worth what both are.** 1.274 x
1.933 = 2.463 if they were independent; measured 2.765, so they compound. The tile halves the
dequant redundancy; the threads change the instruction. `vs_bf16` goes 5.47-5.62x -> 1.96-2.05x.

## The mechanism, read rather than inferred

`linear_fp4_bwd_kernel` at `_THREADS=64` contains **zero `wgmma`**. It runs
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — the Ampere warp-level MMA — loaded by
`ptx_ldmatrix_x4` into registers. It is doing Hopper *data movement* (TMA descriptors, mbarrier,
warp specialization, `warpgroup_reg_dealloc`) and then Ampere *math*.

The rule, read per file across `/work/tilelang_cache` (~16k kernels):

| shape of kernel | MMA emitted | files |
|---|---|---:|
| warp-specialized, 128-thread producer + **64-thread consumer** | `mma.sync` | 25/25 |
| `launch_bounds` 256 | `wgmma` | 85/91 |
| not warp-specialized (no producer guard), threads 64 or 128 | `mma.sync` | 8/8 |

Hopper's `wgmma` issues from a 128-thread warpgroup (4 warps). At `_THREADS=64` this kernel's
consumer partition is 2 warps, so **no bM and no `num_stages` could ever have produced `wgmma`**.
The probe's own per-cell read of the compiled source, once it worked (below), is unambiguous:

| threads | `launch_bounds` | consumer | `wgmma` | `mma_sync` | cells |
|---:|---:|---:|---|---|---:|
| 64 | 192 | 64 | no | yes | 4/4 |
| 128 | 256 | 128 | **yes** | no | 4/4 |

**Aggregate counts hid this.** Counting per kernel family, `linear_fp8` and `gemm_tn` both appear
at `launch_bounds=192` *with* `wgmma`, so "192 is the tell" is false and was reported to the
coordinator in that wrong form before the per-file read corrected it. 192 is a consequence of the
producer/consumer split, not a cause. The cause is the consumer width.

## End to end

| arm | `backward_secs` | peak GiB |
|---|---:|---:|
| C=128, `--no-instrument`, warm, before ([c128 entry](2026-09-07-c128-backward-chunk.md)) | 31.339 | 63.8 |
| C=128, `--no-instrument`, warm, this branch | **23.194** | 63.8 |

**1.351x**, against **1.346x** predicted from the microbench's 8.052 s saving — 0.4% apart, so the
kernel table and the step agree without a fitted term between them.

The cold step on this branch reads 23.755 s and is **not** the comparison:
`prof_backward_ops.py` prints `step: step + 1`, so `step: 1` is the JIT-paying first step while the
baseline above is warm. Using it would credit this change with the cold-vs-warm difference on top
of its own effect — 1.319x instead of 1.351x here, wrong in the flattering direction by less than
the error it hides.

For scale: the chunk ladder that preceded this took `backward_secs` 80.207 -> 41.446 -> 34.717
by changing one constant twice. This is 1.351x from one call site, and it is the first change in
that sequence that made the kernel faster rather than calling it fewer times.

For scale: the chunk ladder that preceded this took `backward_secs` 80.207 -> 41.446 -> 34.717
by changing one constant twice. This is 1.319x from one call site, and it is the first change in
that sequence that made the kernel faster rather than calling it fewer times.

## Why the change is local

`_THREADS = 64` is module-global (`backend.py:25`) and reaches 20+ kernel launches, none of them
measured at 128. Widening it globally would be a 20-kernel change justified by a 1-kernel
measurement. So the tile and the thread count are literals at the `linear_fp4_bwd` call site.

The wide warpgroup is also **gated on the full tile**: `thr = 128 if bM == 128 else _THREADS`.
Below 128 rows `_snap_mma_tile` returns 16 or 32, and 128 threads on a 16-row tile is a cell
nothing measured — most of the warpgroup idle. Training's dX calls are all M=1280 (verified: the
frozen-shapes table records `M=1280` for all 2112 fp4 calls), so the shipped path always takes
the wide cell; the narrow branch exists for correctness at small M, not for speed.

## The probe printed the opposite of the truth, and my first two diagnoses of that were wrong

The sweep records, per cell, whether the compiler emitted `wgmma` — so the mechanism would be
measured rather than inferred from a ratio. It read nothing, and the summary line evaluated
`if hi["wgmma"] and not lo["wgmma"]` — False on a failed read — and printed the else branch:

```
#   bM=64: threads 64->128 is 1.933x -- still mma.sync, so this is a tile effect
```

All eight cells said `wgmma: null` and all four comparisons printed that sentence. **A failed
read rendered as a confident wrong verdict, 180 degrees from the truth**, and it survived into the
JSON. It was caught only because a 1.933x from a thread count cannot be a tile effect, so the two
statements could not both be true.

**The cause took three attempts to name, and the first two were published before being tested.**

1. "The handler swallowed an exception" — true but not the cause.
2. "`get_kernel_source` is a property on tilelang 0.1.13, not a method" — **false.** This shipped
   in the first version of this entry, in the commit message, and in the PR body, stated
   confidently three times. The next run still read nothing.
3. The actual cause, from inspecting the object: the factory returns a `JITImpl` that **compiles
   per shape**, so `get_kernel_source()` with no arguments raises
   `TypeError: missing a required argument: 'block_M'`. There is no single source to ask for. It
   needs the same arguments as the call: `_emitted(kern, *call)`.

```
factory returns: JITImpl tilelang.jit
  kernel_source:     ABSENT
  get_kernel_source: method -> raised TypeError: missing a required argument: 'block_M'
```

The `verified` flag is what makes the difference visible. On the run with diagnosis (2) it printed
"MMA class unread, so the cause is not established here" four times instead of a verdict — the
guard held while the fix was still wrong, which is the only reason the second wrong diagnosis cost
nothing. An instrument that cannot read its subject must say so; the failure mode is not a wrong
number, it is a **default that looks like a finding**.

The mechanism claim in this entry never rested on that column: it was established by reading the
cache's `.cu` files directly. The column now agrees with them, 4/4 both ways.

## A CPU pass is not a gate for this change

`test_frozen_bwd_wide_warpgroup_parity` covers M=64/128/192 against
`reference.linear_frozen_bwd`. On the CPU target it is a **tautology** — no `linear_fp4_bwd`
kernel exists there, so both sides are the same eager function. Verified by mutation: breaking
the tile arithmetic (`bN` 64 -> 32) still passed all CPU tests. The docstring says so, and the
gate that has meaning is the same test on sm90 where the dispatch reaches the kernel.

On card 6, the full parity suite: **42 passed, 1 failed** —
`test_paged_attention_prefill_tiled_vs_naive`, with
`kernel paged_attention_prefill input Q device_type mismatch, expected: 1, got: 2`. A device-index
bug, and this diff touches no attention code. That reasoning is not evidence, and no `errors/`
entry records the failure as known, so it was settled by a revert control in one process on the
card: main's two values, then this branch's, on that test alone.

```
### MAIN values: rc=1 :: 1 failed in 2.97s
### MY values:   rc=1 :: 1 failed in 2.88s
```

Pre-existing, unrelated, and now recorded so the next agent does not have to re-derive it. It is
worth a separate look — a kernel that demands `cuda:1` while the tensors are on `cuda:2` will
fail on any card but one — but it is not this branch's to fix.

## Two ways a check reported nothing and read as a pass

Both cost a card claim and neither produced a number, so they are here rather than in an errors
entry of their own.

`pytest -k "frozen_bwd or linear_fp4"` through `pod_run.sh` became
`ERROR: file or directory not found: or` / `no tests ran in 0.00s` — the launcher builds its
command with `CMD="$*"`, which drops the quotes and word-splits the selector. A selector that
selects nothing exits after collecting zero tests, and "no failures" is what that looks like from
outside.

The first revert control **claimed card 6 and executed a 0-byte file**: the script was written
through `tn exec` with a heredoc, which arrives empty — the gotcha `pod_sync.sh`'s own header
documents ("a heredoc through `tn exec` arrives empty", which is why it base64s). Checking
`wc -c` on the far side is what caught it; the job's own exit status was clean.

## Rule

Ask what instruction the kernel actually emitted before pricing what it does. A 5.6x gap that
survives two rounds of arithmetic about arithmetic (dequant ops, HBM bytes, GEMM floor) is a
sign the model is of the wrong machine — the generated `.cu` in `TILELANG_CACHE_DIR` answers
"which MMA" for free, no card and no profiler, and it answered here after two card-consuming
probes had missed it. Aggregate counts over a kernel family can hide the correlation that
matters; read per file.

Second rule, from the probe rather than the kernel: **a diagnosis of an instrument's failure is a
claim, and it needs the same evidence as a number.** "It is a property, not a method" was
plausible, cost nothing to write, shipped in three places, and was wrong. Inspecting the object
took one run and gave the real answer. What kept that from mattering was the `verified` flag —
the guard reports "unread", so a wrong fix reads as *still unread* rather than as a finding.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-07 | pending | H20 pod (GPU 6) | cuda/sm90 | Qwen3.8-27B NVFP4, GRPO backward | — | — | — |

Backward-only change; no serving path touched (`linear_fp4_bwd` is reached only from
`autograd.py:250`). `backward_secs` 31.339 -> 23.755 is the metric, above.
