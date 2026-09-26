# A long-prompt tail stall was TileLang JIT recompile, not candidate-page growth — V100, 2026-09-26

Production sm70 (07502a34, ThinkingCap-orig, sparse window 1024, refresh R32,
depth-1 draft, decode graph on, 4 slots, 8 GiB host cold tier): submit a fresh
~5.2k-token prompt while another row streams, and the answer's last ~100 frames
show repeated >1 s gaps (p99 5.5 s, max 9.7–53 s). It was filed as
"prompt-tail candidate-page inflation." It is not a scoring, residency, or bandwidth cost —
it is **synchronous TileLang JIT recompilation of attention kernels for a new prefill
length/shape**.

## The discriminating shape of the gap

`TILERL_STEP_TIMING=1` puts >90% of the slow wall in the `model`
segment, but a `torch.profiler` trace of a 17.2 s mixed tick shows the GPU
was mostly idle: all CUDA kernels sum to ~1.0 s, aten cpu ops to ~2.8 s,
`cudaStreamSynchronize` to ~0.7 s. The other ~15 s is three gaps of ~5 s
each: **the GPU is idle and there is no aten op on the CPU**, each sitting
right after a `cudaLaunchKernel` of an HtoD `direct_copy`.

That shape — GPU idle, CPU running Python/nvcc/codegen but emitting no aten
op, bounded by a launch — is JIT compilation. Check it before theorizing about
compute or memory:

```
find ~/.tilelang_cache/0.1.13/cuda-binaries -name '*.cubin' | wc -l
```

Poll that count every 2 s during the slow run. During the slow run a new
`.cubin` appeared every 4–6 s (six over 30 s), each landing exactly on a ~5 s
gap.

## The decisive control: same prompt, same process, second time

Run the same cold prompt twice in one server process. The first pass compiles; the
second must add zero cubins:

- run 1: cache 4669 → 4675 cubins, max frame gap **28.1 s**
- run 2 (identical prompt): cubin count unchanged 4675 → 4675, max gap
  **564 ms**, p99 5747 → 162 ms, `err=None`

An earlier different-length fresh prompt showed the same pattern (40 new cubins on
first pass, zero on the second). A prefix-cache hit does not stall for the same
reason — the shapes are already compiled.

`strings` on the new cubins (`tvm_kernels.cu`) names the residual
recompiled kernels:

- **`paged_attention_split_kernel`** — the dominant one, one specialization per
  prefill length/shape; every cubin added in the late-prompt slow window was this
- **`write_tokens_f32_kernel`** — two variants

A new prompt length yields a new attention shape yields a new cubin, compiled
serially and synchronously, blocking token emission.

## Exhausted-and-excluded (measured, do not re-walk)

All measured on the slow mixed ticks themselves:

- Quest scoring: 68 calls in the slow window, **sum 49.5 ms, max 1.4 ms**
  (candidate count Cp already grown to 475)
- `select_pages` (sm70 routes to the torch reference with topk/`.item()`):
  ~5 ms device per tick
- the whole eager `_select` wall (48-layer python loop + per-page resolve):
  only 0.4–0.5 ms per tick
- forward-internal cold-page promote: 192 calls, cuda-event device time max
  **8 ms** per tick; `evict_victim` D2H = 0
- `sparse_finalize` 20–150 ms; `ssd_mmap=0`; `offers_pages` mostly 0
- packed attention table width stayed ~225 — attention width does not grow with
  context
- control arm routing every H2D/D2H in `_select` through pinned staging with
  `non_blocking=True`: max gap still 12.1 s — the pageable-D2H
  hypothesis is falsified

## Probe trap

`torch.profiler.profile(..., with_stack=True)` on this torch (this venv,
V100) raises
`!stack.empty() INTERNAL ASSERT FAILED at "torch/csrc/autograd/profiler_python.cpp:981"`
and turns the request into a 500. You do not need python stacks to localise a
JIT stall — count cubin files over time.

## Follow-up

Same root cause as the cold-JIT work ("component 1"): after fixing the common
static shapes, a residual 3–6 s recompile remains per new prompt length. The lever is
to converge prefill-chunk/attention shapes onto fixed buckets (stop specializing per
length) or pre-warm the `paged_attention_split_kernel` /
`write_tokens_f32_kernel` shape set. Distinct from the R32 periodic refresh stall
(eager refresh every 32 ticks, ~218 ms, strictly 32-frame spacing).
