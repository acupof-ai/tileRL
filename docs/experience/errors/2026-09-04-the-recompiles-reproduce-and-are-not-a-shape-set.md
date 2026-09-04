# The 51 recompiles reproduce on demand, and they are not a shape set being enumerated

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + draft, B=8, ctx=1024, `--prompts 24`
**Task:** #72 (the B=8 arm), closing #60's open recompile question to a *measurement*

## What #60 left open

#60 measured `write_tokens_f32` and `paged_attention_split` recompiling **51 times each**
in one run, 3-5 s apiece, and could not say why. Every mechanism proposed since has been
wrong, and the wrongness is recorded rather than deleted:

- **`NB = T.const("NB")` bakes the block-table width, so pool growth recompiles.** Two
  errors in one: `NB` is the block *count* (`Mb` is the table width), and the pool never
  grows — `PagedKvPool` sets `num_blocks` once (`kv_cache.py:64`) and `alloc_block`
  *raises* on exhaustion (`:81`). Killed by one `grep`.
- **A full disk starving the JIT cache.** `/` is at 100% on this pod, but
  `~/.tilelang/cache` resolves onto `/data00` with 46 GB free, 315 kernel dirs, 7460
  `.cu` files, written the same day.
- **The `_kernel` factory-arg cache.** Ruled out correctly: `write_tokens` passes no
  factory args (`backend.py:901`), so `self._kernels` returns the same wrapper every call.
- **`@tilelang.jit` shape specialization over `(B, S, H, D, NB, Mb)`.** The right *layer* —
  both kernels declare `B, S = T.const(...)` from tensor shapes
  (`kernels_mma.py:57`, `kernels_attn.py:36`) — but I bounded the reachable set at **~7**
  and reported the bound, which was the last mechanism standing.

## The measurement that ends the shape hypothesis

The B=8 arm of #72 reproduces it live, and this invocation *has* the TileLang INFO logger
that the ctx=2048 run lacked (there, "0 compiles" measured the logging, not the cache).

```
t+0:00   first compile
t+6:56   138 compiles, still going, ZERO depth rows emitted
```

Parsed over the first 273 s, by kernel:

| kernel | compiles | mean | total |
|---|---:|---:|---:|
| `paged_attention_split` | 20 | 3.5 s | 71 s |
| `write_tokens_f32` | 20 | 2.8 s | 56 s |
| `rmsnorm_apply` | 9 | 0.0 s | 0 s |
| `rmsnorm_partial` | 8 | 0.1 s | 1 s |
| `rope` | 4 | 0.2 s | 1 s |
| `embedding_f16`, `silu_mul` | 3 each | 0.0 s | 0 s |
| `paged_attention_split_combine` | 2 | 1.0 s | 2 s |
| `gdn_chunk_fused` | 1 | 1.0 s | 1 s |

**Two kernels carry 127 of 132 s — 96% of all compile time — and 48% of the run's wall
clock is spent compiling.** Every other kernel compiles a handful of times and stops.

Three facts kill the shape hypothesis:

1. **The rate is steady**, ~1 pair per 12 s, flat across every 30 s bucket for the whole
   span. A finite shape set exhausts; this does not.
2. **The count exceeds the enumerable space.** For this exact invocation the reachable
   `(prefill_rows, S_padded)` set is **1 pair** — `budget = max_num_batched_tokens −
   len(decodes)` = 512−n and `_PREFILL_BUCKET = 64` round every chunk of a 1152-token
   prompt to 512. Twenty compiles against one shape.
3. **It is the same two kernels every time**, matched 20/20. They are the only two that
   take `seq_q_lens`, and they are also the two slowest to compile — but the pairing says
   they share whatever varies.

So the shapes are not what varies. Either the disk cache is not consulted for these two,
or their cache key includes something that changes per call. **I am not naming which.**
This entry records a reproducible symptom with its rate and its share, not a mechanism —
the difference between this and the four wrong attributions above is that a bound got
reported as an answer before, and here the count climbed past my bound while I watched.

## What it costs, and where it does not

**Bench-only, eager-path.** The graph path forces the JIT to finish before capture
(`engine.py:223`, "Warmup on a side stream: tilelang JIT (host work) must finish before
capture"), and `B` is bucketed to `_GRAPH_BUCKETS` (`engine.py:823`), so a served tick
replays a graph whose compiles were paid once at capture. At the shipped `--max-batch 8`
the whole reachable `(B, S)` space is 20 pairs.

**It does bound what the harness can measure.** At B=1 the sweep produced eight clean
rows because a compile has to land inside a *timed window* to contaminate it, and B=1
visits few enough shapes that warmup covers them. At B=8 the compile never stops, so
**no B=8 row from this harness is usable** — 138 compiles in 7 minutes with no row
emitted, and the sm90 peer's 1-group B=8 run put a 15693 ms compile inside a 70-tick mean
and reported 190.9 tok/s where the clean answer was 320.

## Consequence for #72

The B=8 arm cannot be measured with `ab_draft_depth.py` as it stands, on either arch.
That is not a reason to flip or not flip the depth default; it means the batch-dependence
question needs either a harness that pre-visits every shape a sweep will touch, or the
graph path, where the compile is paid at capture.

The B=1 verdict is unaffected: depth 1 beats the shipped depth 3 by **1.2522x at
ctx=1024** and **1.1186x at ctx=2048**, on rows whose rung cross-checks agree to
0.16-0.26%.

## Separately: the sm70 memory ceiling makes the peer's B=8 config unreachable here

`num_slots = max(batches) + 1 = 9` is hardcoded at `ab_draft_depth.py:267` — no flag.
`step_states` is `[slots, linear_layers, spec_steps, heads, kd, vd]` f32
(`kv_cache.py:215`), and on the 27B geometry (48 linear layers, 48 value heads, 128×128)
that is **0.141 GiB per slot per step**:

```
slots=9 depth=4:  states 1.27 + step_states 6.33 = 7.59 GiB
KV at ctx=2048 B=8, x2 for the draft mirror       = 4.28 GiB
weights                                            = 24.4 GiB
                                                    ------
                                                     36.3 GiB on a 32 GiB card
```

The peer's ctx=2048 B=8 arm runs only because H20 has the memory. **On sm70 it does not
fit at any depth**, which is a constraint on #72 rather than a launch detail. This run
uses ctx=1024, where the estimate leaves margin at depth 1-2 — and the estimate is known
**1.39x low** (#55 measured 2.94 GiB where this model says 2.11), so the deep rows may
OOM; that outcome is reportable.

## The run ended in OOM at depth 1, and the failing shape is 8x what the plan permits

After 269 compiles and 10 minutes it died — **at depth 1**, where my estimate said there
was margin:

```
torch.OutOfMemoryError: Tried to allocate 272.00 MiB.
  31.74 GiB total, 186.38 MiB free, 31.55 GiB in use (30.33 allocated by PyTorch)
  backend.py:1202 in silu_mul  <-  self._c(self._f32(gate).reshape(-1))
```

So the estimate was wrong in the direction already flagged (#55 measured 2.94 GiB where
the model says 2.11, 1.39x low) — depth 1 does not fit either, and depths 2-4 were never
reached. **That is the reportable row for #72: on sm70 the B=8 arm does not run at any
depth, at ctx=1024 or ctx=2048.**

**One number in it I cannot account for.** The failing allocation is 272 MiB =
`4096 × 17408` f32 — 4096 rows into `silu_mul`. But `_build_plan` decrements a *shared*
token budget (`engine.py:591`, `:608`): `budget = max_num_batched_tokens − len(decodes)`
= 512, the first prefill row takes `min(1152, 512) = 512` and drives the budget to 0, so
the next row breaks out. One tick should therefore prefill **512 tokens → 512 rows → 102
MiB**. The observed tensor is **8x** that, as if all eight rows each took a full 512-wide
chunk.

Either my reading of the budget is wrong or something else feeds `silu_mul` a
`[8, 512, ·]` activation. I have not established which, and the whole point of this entry
is not to fill that kind of gap with a plausible story — the four dead mechanisms above
are what that costs.

## Rule

A steady rate refutes an enumeration. Before attributing repeated work to a finite set of
inputs, check whether the rate decays — if it is flat, the set is not what is being
enumerated, and no amount of counting the inputs will explain it.

And a bound is not an answer. "~7 reachable shapes against 51 measured" was true and
useless: it named the gap as the finding and then let the shape story stand as the
explanation anyway. The honest form is what this entry does — the symptom, its rate, its
share of wall time, and an explicit refusal to name the cause.
