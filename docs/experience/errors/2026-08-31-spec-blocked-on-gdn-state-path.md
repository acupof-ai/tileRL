# Speculation on sm70: measured to the bottom, blocked on the GDN state path — 2026-08-31

> Status: **ROOT CAUSE SUPERSEDED.** The step is `linear_fp4_gemv_sm70_m`, not
> the GDN state path — per-kernel profiling puts GDN at 1.9-2.9 ms/tick against
> 252 ms of GEMV. See
> [2026-08-31-m8-gemv-occupancy-not-reuse.md](2026-08-31-m8-gemv-occupancy-not-reuse.md).
> The measurements below (replay flat in W, the MTP quality numbers, the wrong
> turns) all stand; only the attribution to gather/scatter is wrong.

## What works

The checkpoint's MTP head is real and good. `load_draft` already handled the
`mtp.` prefix and the Qwen3_5RMSNorm +1 fold; all 15 `mtp.*` keys live in one
shard (`model-00018-of-00018.safetensors`) and map cleanly with nothing missing.

| metric | value |
|---|---|
| top-1 agreement with trunk | 62% |
| median trunk-rank of draft's pick | 0 |
| draft pick in trunk top-5 | 84% |
| accept rate in serving | 97-100% |
| tokens committed per forward | 2.35-5.33 |

The speculation *policy* is not the problem and never was.

## The measurement that settles it

Pure graph replay, no observer effect (timing `g._graph.replay()` directly):

| graph | replay |
|---|---:|
| (B=1, W=1) | 39 ms |
| (B=1, W=3) | 266 ms |
| (B=1, W=4) | 267 ms |

**Flat in W.** 266 and 267 are the same number, so the cost is not per query
position — crossing W>1 switches paths, and the new path is 6.8× slower at any
width. That single fact killed three plausible diagnoses.

## Root cause

`backend.py:917` sends any `q.shape[1] > 1` to `gdn_chunk_fused`. On sm70
`gdn_decode` returns None (it is sm90-only), so `model.py:421-439` takes the
generic route for all 48 GDN layers:

```python
state, window = backend.state_gather(pool.states, pool.conv_windows, ...)
out, new_state, new_window = backend.linear_attn_chunk(...)
backend.state_scatter(pool.step_states if ks else pool.states, ...)
```

`reference.state_gather` is `states[slots, layer_idx]` — torch advanced
indexing, so **3 MiB copied out and 3 MiB back, per layer**: 288 MiB of round
trip across 48 layers, in torch ops that cannot fuse into the kernel. Plus the
chunk kernel's own two epilogue launches (gated RMSNorm + z-gate).

Delta is 227 ms over 48 layers = 4.73 ms/layer. Bandwidth for the traffic is
~1 ms total, so this is launch and indexing overhead, not bytes.

Note W=1 pays the same path — which is why 39 ms is already 2.5× the 15.6 ms
weight roofline.

## Wrong turns, and what each one cost

Recorded because the pattern matters more than the individual errors: every one
was inferred from end-to-end throughput and refuted by a direct measurement.

1. **"fp8 quantization has no sm70 kernel"** — true, and worth 0.7 → 6.0 tok/s.
   `linear_fp8` is sm90-only, so `_quantize_draft` sent every draft projection
   to the generic path. Still a net loss, so not the blocker.
2. **"then fp4, sm70's fused format"** — measured *worse* (3.1 tok/s) and I
   concluded fp4 was wrong. It was my profiler: it deleted the dense keys
   including `fc`, silently reverting to dense. Fixed, fp4 is **4.98 ms/step vs
   120.91 dense (24×)** — the original instinct was right and the self-correction
   was the error.
3. **"the draft runs outside the captured graph"** — true but minor. Cost is
   attention/GDN width, not launch count.
4. **"split-KV only covers s==1, so verify takes the serial kernel"** — true and
   now fixed (9.5× at S=4, n=1024). Attention was 2.7% of the tick.
5. **"`linear_fp4` is 69% of the tick"** — an artifact of my own op wrapper: it
   called `torch.cuda.synchronize()` per op, which broke capture and measured
   the eager path. Graphs were fine (`[(1,3), (1,4)]` captured, `graph_on=True`).

## What would fix it

Extend `gdn_decode_fused` to T>1 with in-place pool state, so a verify forward
keeps the decode path's fused state handling instead of the gather/scatter
round trip. That removes the 227 ms step-change *and* takes W=1 below 39 ms,
since both pay it today.

Projected if the verify forward cost ~W×(decode tick): depth 3 at 62% acceptance
clears 60 tok/s. Not attempted — it is a kernel project, not a fix.

## Capacity facts found on the way

- `step_states` is `[num_slots, L, spec_steps, heads, K, V]`, sized by SLOT
  COUNT: 16 slots at depth 6 is `16 × 7 × 144 MiB = 15.75 GiB` and OOMs a 32 GB
  card outright. 4 slots (3.94 GiB) fits.
- Tree verification is separately blocked: `kernels_gdn.py:500-520` evolves
  `state_local` across the `t` loop, so node t builds on t−1, not on its parent.
  Not needed anyway — a linear chain tops out at `1 + p/(1−p) = 2.63` tokens =
  67.5 tok/s at p=0.62.

## Rule

When a cost is flat in the parameter you think drives it, you are on the wrong
axis — look for a path switch, not a scaling term. And never attribute cost with
an instrumentation wrapper that synchronizes: on a graph-captured path the
wrapper measures the fallback it just caused. Time the replay itself.
