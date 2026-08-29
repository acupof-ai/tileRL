# One backward op held 14.2 GiB — chunking it bought 14.7% throughput too

## Context

After the embedding fixes, a 27B LoRA step at B=1 T=64 still peaked 14.2 GiB
above what the forward held live, on a sequence of 64 tokens. The stored
activations could not explain it: the tape's own accounting says they are
**2.7 GiB** at that size (`add` 1.15 + `linear` 1.22 + the rest), and the two
large rows in that table — `linear_fp4_frozen` 11.0 GiB, `linear_fp8_frozen`
10.4 GiB — are the QUANTIZED WEIGHTS held as entry args, resident params rather
than activations.

So the gap is transient allocation inside a backward handler. Attributing the
peak per op (`probe_train_ledger.py --peaks`, replaying the tape entry by entry
with `reset_peak_memory_stats` around each handler) named one:

```
  -- worst backward peak, by op
      14.209 GiB  linear_fp8_frozen
       0.118 GiB  add
       0.076 GiB  linear_attn_chunk
       0.059 GiB  linear
```

Every other op in the model is under 0.12 GiB. One call was the whole gap.

## What Worked

`linear_frozen_bwd`'s fp8 branch has no tilelang kernel (fp4 does), so it runs
the eager reference, which materialized the entire dequantized weight —
`lm_head` is [248320, 5120], **4.74 GiB in f32** — and then made a second copy
of it to fold `oscale` in.

Two changes, no new kernel:

- **`oscale` folds into the gradient, not the weight.** It scales weight row n,
  so it multiplies the [M, N] gradient instead of the [N, K] weight. This is
  where the fp4 kernel path already puts it. One 4.74 GiB copy gone.
- **Dequantize a slice at a time.** dX contracts over N, so the weight has to be
  materialized — but only 512 MiB of it at once, accumulating into the output.

The slice boundary is the trap: an fp4 scale covers one weight row, but an fp8
scale covers a **128-row block**, so slicing on plain row indices reads the
wrong scale for every chunk after the first. The chunk step is a multiple of
the scale's own row granularity, and `test_frozen_bwd_chunking_matches_whole`
compares chunked against one-shot for both formats — a defect no throughput
number would have shown.

## Measured (27B LoRA, H20 GPU 7, loadavg 9.1)

| B x T | tok/s before | after | ratio | peak GB before | after |
|---|---:|---:|---:|---:|---:|
| 1x64 | 50.3 | **57.7** | 1.147x | 47.0 | **31.0** |
| 1x128 | 80.5 | **90.5** | 1.125x | 50.6 | **36.2** |
| 1x256 | 113.5 | **124.3** | 1.096x | 57.5 | **46.6** |
| 2x256 | 178.2 | **194.7** | 1.093x | 76.5 | **67.7** |

(The "before" column is this morning's baseline, so it also carries the
embedding-copy win measured separately.) The op's own peak: **14.209 -> 2.054
GiB**. Throughput rises because the removed work was not only memory —
materializing 4.74 GiB of f32 twice is bandwidth the matmul never needed.
4x256 still OOMs.

## Rule

Attribute a peak to an op before deciding what to build. The plausible story
here was activation checkpointing, a large change touching the tape's
representation; the actual cause was one eager fallback with no kernel behind
it, fixed in ten lines. The tape's own byte histogram was what killed the
activation hypothesis — and it was misleading at first glance too, since its
two biggest rows are weights, not activations.
