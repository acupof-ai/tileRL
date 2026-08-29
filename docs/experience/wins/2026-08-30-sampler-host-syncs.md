# The sampler shipped its own arguments to the device to read them back — 2026-08-30

**Date:** 2026-08-30 · **Scope:** `engine` · **Status:** CPU-verified, GPU `pending-remote`

## Context

A peer working the sm70 bring-up traced their CUDA-graph capture failures to
host syncs INSIDE the captured forward, from targets that lack the fused
kernels and fall back to torch loops. That is a useful frame for a second
question: what else syncs in a decode tick?

The CPU target takes the same fallbacks a GPU without those kernels takes, so
counting `aten._local_scalar_dense` per tick on CPU enumerates them — no GPU
needed.

## What worked

One decode tick, tiny model, each sync attributed to its Python frame:

```
host syncs in one decode tick: 7
  1  kv_cache.py:223 write_tokens     <- fallback only
  1  kv_cache.py:224 write_tokens     <- fallback only
  1  kv_cache.py:227 write_tokens     <- fallback only
  1  reference.py:738 <listcomp>      <- fallback only (eager gdn)
  1  reference.py:1092 sample_batch   <- EVERY target
  1  reference.py:1094 sample_batch   <- EVERY target
  1  engine.py:982 <listcomp>         <- EVERY target
```

The split is the finding. Four are the un-fused paths a peer was already
fixing by registering the kernels. **Three run on every target, sm90
included**, and they are pure waste:

```python
temps = torch.tensor([r.params.temperature for r, _, _ in rows], device=dev)  # host -> device
...
sample_rows = temperatures > 0
if not bool(sample_rows.all()):     # device -> host
if bool(sample_rows.any()):         # device -> host
    ...
    gen = ...manual_seed(int(seeds[idx[i]]))   # device -> host, PER SAMPLED ROW
```

`temperature`, `top_p` and `seed` are Python scalars on `SamplingParams`. The
engine put them on the device and `sample_batch` read them back to decide which
rows are greedy. At temperature > 0 with B=8 that is 2 + 8 + 1 = 11 syncs a
tick.

**Fix:** keep them on the host. `sample_batch` splits greedy from sampled rows
in Python, and only the sliced temperature/top_p vectors for the sampled subset
go to the device.

## Numbers

| | before | after |
|---|---:|---:|
| host syncs / decode tick (greedy) | 7 | **4** |
| of which run on a target with fused kernels | **3** | **0** |
| host syncs / decode tick (B=8, temperature > 0) | 15 | **4** |

Tick time: **pending-remote**. This host has no GPU, and a sync costs nothing
measurable on CPU.

Draws are unchanged — `test_sample_batch_matches_per_row` compares the batched
sampler against per-row `sample()` for the same seeds and still passes.

## Rule

Count `aten._local_scalar_dense` per tick, not just kernel launches. And when a
value reaches the device from a Python scalar, nothing downstream should ever
read it back — that round trip is always removable, and it is the second time
today it was worth several hundred syncs a step (see
[2026-08-30-adafactor-host-sync-per-parameter.md](2026-08-30-adafactor-host-sync-per-parameter.md)).

Corollary: the CPU target is a free enumerator for the fallback paths a
partially-ported GPU target will take.
