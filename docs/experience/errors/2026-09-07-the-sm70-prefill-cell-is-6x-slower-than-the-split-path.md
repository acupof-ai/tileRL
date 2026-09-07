# The sm70 prefill cell is at least 5.97x slower than the split path it replaces

**Status:** closed — #232 closed unmerged 2026-09-07T13:40:06Z; main never carried it
**Date:** 2026-09-07
**Branch:** `kernels/sm70-prefill-cell` at `e1e38c4` (#232)
**Card:** one Tesla V100-SXM2-32GB, real `qwen38-27b` NVFP4, serve child stopped
**Tree:** `/data00/home/chenkailun.c/tl-s-v100sm70`
**Verdict:** REJECT. Gate 3 of the accepted approach note required TTFT at 16,384
tokens to fall. It rose to **at least 5.97x** of it.

## Context

The cell was accepted on a measurement:
[wins/2026-09-07-the-sm70-prefill-n2-is-one-kernel-at-30x-its-floor.md](../wins/2026-09-07-the-sm70-prefill-n2-is-one-kernel-at-30x-its-floor.md)
put `paged_attention` at 100.5 s of a 197.59 s 16,384-token prefill, 29.9x its
arithmetic floor, rising 0.0889 → 6.7584 s (75.98x) from first to last chunk
because `paged_attention_split` opens one query row per thread block. A 64-row
query tile was the fix. #232 wrote it. The acceptance run is this one:

```
scripts/prof_prefill_ops.py --model qwen38-27b --tokens 8192,16384,21727 \
    --prefill-kv-dtype f32 --json prefill_f32.json
```

**It never reached a timed arm.** The log spans 19:08:47 to 19:42:36 (**2029 s**)
and its last line is the model banner, so all of it is the untimed 21,727-token JIT
warm-up arm. The baseline entry's `224.7 s / 329.1 s` is two runs — the profiler
prints one `%.1f s` figure per invocation (`prof_prefill_ops.py:317`) — and neither
JSON records the warm-up, so which figure belongs to which run is **unverified**;
the profiler warms at `max(--tokens)` (`:316`), so on arm size the comparable
baseline is 329.1 s. 2029 s is 6.17x it, and the model load is inside that 2029 s,
so >= 6x is the safe reading.

## Compilation is excluded, three ways

- **0** `begins to compile` lines in the log (`tilelang/jit/kernel.py:137`). The
  print is on by default and reaches the log — `TILELANG_PRINT_ON_COMPILATION` is
  `"1"` and `tilelang/__init__.py` calls `set_log_level("INFO")` at import, on the
  pod's tilelang 0.1.13. Its negative control is red: other logs on the same pod
  and interpreter carry 2, 2 and 9 of these lines.
- `find ~/.tilelang/cache -newermt` found **0** files touched during the run
  window (re-run here; 1610 cubins predate it).
- No child compiler process.

The cache was warm and holds the cell's 21 MB cubin from an earlier window. So the
33 minutes are execution.

## The binding number, from measured values only

Three `py-spy dump --locals` reads of the profiler's own loop variables, spanning
208 s:

| read | `ticks` | `prefix` |
|---|---:|---:|
| t | 21 | 10752 |
| t + 90 s | 22 | 11264 |
| t + 208 s | **22** | 11264 |

The 512-token advance is the loop's own `chunk: 512`, so the counters are the
scheduler's chunks and not an artifact. The third read did not move, so **the
single chunk at prefix 11264 cost >= 118 s.** The first chunk of the same arm
(prefix 0) cost 8.607 s of `tick_secs`.

A 16,384-token arm contains the chunks at prefix 11264, 11776, …, 15872 — **10
chunks** — and each costs at least as much as the one at 11264, because cost
rises monotonically with prefix; the baseline measured that rise as 75.98x
first-to-last on this same arm, so monotonicity is measured, not assumed.
Therefore

**TTFT at 16,384 >= 10 x 118 s = 1180 s, >= 5.97x the baseline's 197.59 s.**

That is a lower bound over 10 of the arm's 32 chunks, and it already exceeds the
baseline's entire prefill by 5.97x, so no slope improvement in the other 22 chunks
can offset it. Gates 1 (per-chunk slope) and 2 (ratio to the compute floor) were
never measured: no timed arm ran.

The run was then SIGTERMed at the fd-verified pid, rc=143, rather than run to
completion. **Extrapolations, not measurements:** taking the warm-up rate as
quadratic gives 16,384 ≈ 3796 s (19.2x) and 21,727 ≈ 6911 s (22.0x), the whole run
≈ 4.6 h. Only the 1180 s bound carries the verdict.

## Root cause

At `block_N=16` the K/V loop is serial with no prefetch.
`kernels_attn.py:294` `make_paged_attention_prefill_sm70`, loop at 360-397:

```python
for k in T.Pipelined(T.ceildiv(upper, block_N), num_stages=1):
```

`num_stages=1` puts every statement in one stage:
`pipeline_planning.cc:1232` assigns `pinfo.stage = num_stages` and copy stages
stage 0, the rewrite that hoists a copy ahead of its consumer is gated on
`num_stages >= 2` (`pipeline_planning.cc:1270`), and `inject_pipeline.cc:1219`
multi-versions a buffer only when it needs more than one version. So the
global→shared K/V load of iteration `k+1` cannot start before iteration `k`'s math
finishes.

**Raising `num_stages` would not have fixed it on this card, for two independent
reasons.** Volta has no `cp.async`: `TargetCudaHasAsyncCopy` returns `arch >= 80`
(`tilelang/src/cuda/target_utils.cc:89`, reached through `TargetHasAsyncCopy` at
`src/backend/common/target_utils.cc:15`), so
every async-copy annotation path is skipped at sm70 — `pipeline_planning.cc:860`,
`:932` and `:1097` all gate on `TargetHasAsyncCopy(target_) && use_async_copy_`.
And double-buffering the tiles does not fit — the tile was chosen to sit exactly at
Volta's shared-memory limit, `(64 + 2·16)·256·4 = 96 KiB`
([wins/2026-09-07-the-sm70-prefill-tile-is-64x16-f32.md](../wins/2026-09-07-the-sm70-prefill-tile-is-64x16-f32.md)),
and a second version of `Ks` and `Vs` adds 32 KiB (derived).

The launch config is `block_M=64, block_N=16, threads=256`, f32 shared tiles,
f32 accumulators, `KVSPLIT` removed, grid `(ceildiv(S, 64), H, B)` = 192 blocks
at S=512, H=24 against 80 SMs. At `block_N=16` the serial loop runs
`ceildiv(21727, 16)` = **1358** iterations at the longest arm (derived), each one
a synchronous shared-memory load followed by a `T.Parallel(block_M, block_N)`
whose every element is `for d in T.serial(256)` of scalar f32 MACs over shared
operands — 1024 length-256 dot products per iteration spread over 256 threads.

**What the replaced path does that this does not.**
`kernels.make_paged_attention_split` (`kernels.py:868`) puts `KVSPLIT` and `S*H`
in the grid — `T.Kernel(KVSPLIT, S*H, B)` — so at S=512, H=24, KVSPLIT=16 it
launches 196,608 blocks whose K/V ranges are disjoint slices of the history, and
each block's serial loop is `ceildiv((p1 - p0), block_N)`, i.e. 1/KVSPLIT of the
history rather than all of it. That hides the load latency by occupancy. The new
cell removes both — one query tile per block, 192 blocks, the full history serial
in each — and its 64x fewer K/V bytes do not pay for that.

## Fix

`backend.py:962-984` is the branch that sends `s > 8` on sm70 to the cell; without
it the same call falls to the `paged_attention_split` `elif` at 985, the measured
baseline. Main never carried that branch — its sm70 attention call goes straight to
the split `elif` (`backend.py:935` on `origin/main`) — so no serve path was ever
exposed and nothing in the tree needs changing. **#232 was closed unmerged**, which
is the fix; a later revival must strip the routing commit first. The kernel, its CPU
twin, the tile arithmetic and the predicate can stay.
`tests/test_sm70_prefill_cell.py` cannot: its `expect="paged_attention_prefill"`
arm asserts the cell ran, so it goes with the branch. That file is sm70-gated
(`pytestmark ... arch != "sm70"`), so CI on cpu never ran it either way.

A next attempt needs a schedule that hides the K/V load, and on Volta that means
occupancy rather than `cp.async`: fp16 tiles (48 KiB, two blocks per SM), a
larger `block_N` to cut the iteration count, or a split of the history across the
grid as split does, keeping the query tile. Each changes what the card executes, so
each needs its own arm.

## The routing proof was not a proof

Independent of the speed result: #232's entry claims the cell is routed because
py-spy put `backend.py:969` in **191 of 191** samples. Three things are wrong with
that, all verified:

**py-spy cannot see it in the profiler process.** Measured: 40 samples, **40/40**
in `torch.cuda.synchronize` (`torch/cuda/__init__.py:954`) called from
`prof_prefill_ops.py:87`, **0/40** on any `backend.py` line. The profiler syncs
after every op (`_Timer.timed`), so the CPU returns from the dispatch in
microseconds and blocks in the sync for essentially the whole wall clock. The
191/191 must have come from the serve process, which has no such sync — a
different process from the one the timed arms measure.

**"The `elif` at 985 took none" is entailed by the branch order, not measured.**
The split branch is the `elif` after the prefill branch, and every
`paged_attention` call in a prefill chunk arrives at the chunk width — 512, so
`is_prefill_width` (`s > 8`, `backend.py:36`) is true for all 16 of them, one per
full-attention layer (measured, `prefill_ops_small.json` chunk 0). The two
branches are mutually exclusive, so the second must read zero whenever the first
is taken: it restates the first count instead of testing it. Probed on main, where
that branch does not exist, the same `s=512` call dispatches
`paged_attention_split` + `paged_attention_split_combine` — so deleting the branch
does **not** leave the zero unchanged.

**The guard tests a name, and the name is shared.** `backend.py:965` reads
`"paged_attention_prefill" in _resolve(self.precision, self.arch)`, and
`_CPU_KERNELS` already registers `"paged_attention_prefill":
kernels.make_paged_attention_prefill` (`registry.py:57`), so the membership test
is true on the cpu target too. It asks whether *some* maker is registered under
that name, never which one. The cell branch's own support matrix records the shift
this causes: sm70's row goes from `24 | 2 | 8 | 14` to `24 | 3 | 8 | 13`, so
`paged_attention_prefill` becomes sm70's third override and the count of keys that
are still the CPU object by identity drops to 13.

The profiler's own provenance print does not close the gap either: `_provenance`
reads `_REGISTRY` at startup, so it prints the sm70 maker name for a run with
zero prefill dispatches.

**The fix exists and is in the tree.** `scripts/prove_kernel_routing.py`, 82
lines: it wraps `Backend._kernel`, counts every dispatch keyed by the *resolved
maker* plus factory args, and runs the target script unchanged in-process through
`runpy`. Four controls, all run — the third is the one that matters, being exactly
the case `backend.py:965` cannot distinguish:

| control | result |
|---|---|
| cpu target, expecting the sm70 maker | `the sm70 cell did not run: no dispatch resolved to …`, **0 of 418 dispatches**, exit 1 (re-run here, reproduced) |
| same run, `--expect kernels.make_paged_attention` | `PROVEN: … carried 11 dispatch(es)`, exit 0 |
| `pytest -k prefill_tiled` — op name present, maker is the CPU twin | red, naming `kernels.make_paged_attention_prefill` — the arm a name-based check passes |
| a script that calls no kernel | `INSTRUMENT SAW NOTHING` |

## A label-only difference in the identity lines

The baseline entry's identity line reads `precision=fp4` and this run's reads
`precision=bf16`. Comparability is unaffected: `self.precision` is assigned the
literal `"bf16"` at all five shas checked (`c905bb3`, `d3cdd30`, `14f206c`,
`57b06ea`, `e1e38c4`) and `git log -S'precision = "fp4"'` over `backend.py` finds
no commit, and `registry.py` registers the same `_SM70_KERNELS` dict under both
`("bf16", "sm70")` and `("fp4", "sm70")`, so the same kernels resolve either way.
It does mean the baseline entry's header line did not come from an unmodified run
of that code.

## Rule

**A kernel's cost is its schedule, and a tile is only one term of it.** The
accepted analysis priced the K/V traffic a 64-row query tile removes — 20.16 s of
bandwidth floor down to 0.31 s — and it was right about that term. It never
priced what the tile costs: 192 blocks instead of 196,608, and a 1358-iteration
serial dependent loop instead of many short independent ones. On a card with no
async copy, the exposed latency is not recoverable by a flag.

**A routing proof must name the maker that ran, not the key that resolved.** A
name shared between a cell and its CPU twin makes `name in _resolve(...)` true
everywhere, and a probe whose process syncs after every op cannot see the branch
at all.
