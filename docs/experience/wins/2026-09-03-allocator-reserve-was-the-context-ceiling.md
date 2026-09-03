# The allocator's load reserve was the context ceiling — sm70, 2026-09-03

> Status: Shipped

## Context

`serve` on the 32 GB V100 was launched `--blocks 512`, an 8192-token context, and
the number had never been derived — it was the largest value that did not OOM.
Raising KV capacity at B=1 was the question; the f16-pool path (136 → 68 KiB per
token, a 2x) was the plan.

The card said 25728 MiB resident with only 1088 MiB of that in blocks. Arithmetic
over the pool shapes said the fixed footprint was 16.11 GiB, so 8 GiB was
unattributed — flagged instrument-suspect rather than reported, since 1.5x is past
the point where a derived ceiling is worth printing.

## What Worked

It was not fragmentation and not the pools. Loading and quantizing the 27B leaves
the caching allocator holding **29.02 GiB reserved against 15.96 allocated**, and
`torch.cuda.mem_get_info` counts every byte of that 13.06 GiB as used. Both
`build_engine`'s `PrefixStore` budget and any block sizing read that number.

One `torch.cuda.empty_cache()` after `materialize`, measured as an A/B in one
process each (`scripts/probe_kv_ceiling.py`):

| | control (shipped) | reclaim |
|---|---:|---:|
| free before sizing | 2.36 GiB | 15.17 GiB |
| KV pool | 0.13 GiB | 8.42 GiB |
| **context at B=1** | **1024 tok** | **64912 tok** |

Then `num_blocks=0` fits the pool from that corrected number. Two ordering
constraints, both found by getting them wrong:

- The fit must run **after** `materialize` and the reclaim. Sized one call earlier
  in `cli._build_engine`, it asked for 10.21 GiB with 4.96 free and OOMed inside
  `PagedKvPool`.
- The fit must run **after** the GDN state pool, which is 2.94 GiB at slots=3
  depth=3 — 79% of it `step_states`, which scales `slots × width` and **not**
  `max_batch`. Allocating the state pool first turns the estimate into a
  measurement.

End to end, no flags: `kv=4046 blocks = 64736 tokens`, and the server answers.
That is **7.9x** the 8192 it was serving, and the f16 pool is now the *second*
lever, not the first — it needs a kernel change for 2x where this needed none.

Two bugs the probe surfaced on the way, both real and both shipped broken:

- `spec.py` could not read the only draft head in production. The NVFP4 shard's
  readout is `lm_head.wq/scale/oscale`; `_param_key_for` names none of them, so a
  skip keyed on `mapped == "lm_head"` sent all three to `unknown` and raised. Every
  test used an f16 head with one `lm_head.weight` and passed. Now keyed on
  `_is_lm_head`, with a negative control that an unrelated unmapped tensor still
  raises.
- Three sites in `backend.py` read `io = bf16 if target.startswith("cuda")`. sm70
  is CUDA and has no bf16 load, so `gdn_prep` got f16 against an f32 signature and
  died mid-run with "input Q dtype mismatch". `Backend.io` is keyed on arch and
  already documents this hazard; all three now use it, with a source gate
  (`test_precision.py`) since the arch that breaks cannot run in CI.

## Rule

Free memory on a freshly-loaded model is not free memory. `mem_get_info` reports
the allocator's reserve as used, so anything sized from it after a big load is
sized against a number that is 13 GiB wrong on this card — reclaim first, and
allocate the pools whose cost you would otherwise estimate before you measure what
is left.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-03 | pending | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | — | — | — |

Capacity, not rate: this entry changes the context ceiling (8192 → 64736 tokens at
B=1, spec depth 3, slots 3) and leaves the tick untouched. Decode rate at ctx=1024
stands at 45.9 tok/s (`2026-09-03-single-stream-b1-baseline.md`); the rate across
the newly reachable contexts is not measured yet.

Raw artifacts: `$HOME/tilerl-logs/kv1.log` (reclaim), `kv2.log` (control),
`srv3_out.log` (fitted serve), on the V100.
