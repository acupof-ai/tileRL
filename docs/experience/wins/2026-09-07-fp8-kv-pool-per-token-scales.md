# fp8 KV pool and fp8 attention readers, per-token scales — sm90 + cpu, 2026-09-07

> Status: Shipped (flag, default off) — writers and readers both fp8, gated on card.
> **No tok/s claim yet**: the 27B decode arms are pending-remote.

## Context

`--kv-fp8` stores the KV planes in e4m3 instead of bf16. The plane is what a decode
tick re-reads per token at long context, so halving it is the reason to want this.

**The pool alone was a regression, not a partial win, and that is what forced the
readers into the same change.** With the pool fp8 and the attention kernels still
reading bf16, `kv_layer()` dequantized the *whole plane* per call — every block,
including ones the request never touches. One tick's 16 calls at the 27B's pool shape:

| accessor | bf16 pool | fp8 pool |
|---|---:|---:|
| `kv_layer()` | 0.01 ms | **64.49 ms** |
| `kv_operands()` (raw planes + scales) | 0.07 ms | **0.04 ms** |

The cost scaled with `num_blocks`, not with the sequence: 31 / 102 / 491 ms at 128 /
512 / 2048 blocks, against `kv_operands` flat at 0.020 / 0.018 / 0.021. A capacity flag
that gets *worse* as the pool grows is backwards, so this entry covers both halves.

What it settles: the scale geometry, the append rule, the dtype, the reader conversion,
three defects the tree already had, and five claims that were wrong in the design note
or in my own earlier messages.

## The bound any later tok/s claim sits under

Stated before any speedup, per the note. At the 27B's 16 full-attn planes x 4 kv
heads x 256 head_dim, measured off a real pool rather than computed:

| pool | bytes/token | one block | 8k seq | 32k seq |
|---|---:|---:|---:|---:|
| bf16 | 65536 | 1.0 MiB | 512 MiB | 2048 MiB |
| e4m3 + f32 per-token scale | 33280 | 0.508 MiB | 260 MiB | 1040 MiB |

**1.969x, not 2.000x** — the missing 0.031 is the scale plane at 1.56% of the fp8
bytes. And a decode tick reads the 27B's weights every token regardless, so even a
fully converted reader path makes this a fraction of the tick, not a halving of it.

The weights are **22.76 GiB resident** (24436981888 bytes, summed off the loaded model),
not the ~12.6 GiB the NVFP4 checkpoint occupies on disk. Every ceiling below is against
the measured figure; my first set used the on-disk number and overstated all of them:

| cell | KV share of a decode tick, bf16 | ceiling | I first said |
|---|---:|---:|---:|
| B=1 ctx=8k | 2.1% | **1.011x** | 1.019x |
| B=1 ctx=32k | 8.1% | **1.041x** | 1.072x |
| B=8 ctx=8k | 14.9% | **1.079x** | 1.13x |
| B=8 ctx=32k | 41.3% | **1.255x** | 1.380x |
| B=32 ctx=32k | 73.8% | **1.570x** | 1.699x |

A resident footprint is not a file size, and a ratio built on the wrong denominator is
wrong in the direction that flatters the change.

A byte figure carries its dtype or it is half an answer: the same expression over the
same config gives 1.0 MiB on the H20's bf16 pool and 2.0 MiB on the V100's f32 one
(`Backend.io` is f32 for cpu/metal/sm70). Two sessions each stated one of those as
"the block size" today, in opposite directions.

## What Worked

**One f32 scale per `(plane, block, kv_head, token)`** — amax over head_dim only.
Two independent reasons, neither of which is accuracy:

*The launch shape.* `make_write_tokens` runs `T.Kernel(B*S, H)` — one thread block
per `(b, t, h)`, head_dim inner. A per-`(block, head)` absmax spans the 16 thread
blocks that share a pool block, so a single-launch fused writer cannot compute it
without atomics or a second pass. A per-token absmax is a reduction over head_dim
inside one block. Per-token is the only grid this writer can produce in one launch.

*The append rule.* A per-block scale changes as the block fills, so every append must
re-quantize the tokens already stored. Three arms, one growing-magnitude fixture
(token `t` at `(1+t)x` token 0's scale), max relative error and token 0's own:

| append rule | max rel | token 0 |
|---|---:|---:|
| dequantize from the fp8 block, requantize — per-block scale | 0.3376 | 0.3376 |
| bf16 staging buffer per open block, requantize staging — per-block scale | 0.0629 | 0.0584 |
| per-token scale, write only the token written | 0.0588 | 0.0579 |

Token 0 is the worst-hit element in the compounding arm — it is the one rounded 16
times, which is the signature that distinguishes compounding from ordinary
quantization error. Staging fixes it, at 32 KiB per open block per plane. Per-token
needs neither buffer nor read-modify-write: it is idempotent by construction.

**Accuracy separates the grids too, in the metric that bounds the logit — and an earlier
draft of this entry said it did not.** Error over the row's amax, measured on the pool a real
prefill filled:

| grid | K | V |
|---|---:|---:|
| per `(block, head, token)` — shipped | 0.0357 | 0.0354 |
| per `(block, head)` | 0.0585 | 0.0574 |

**1.64x.** The earlier draft read them as identical at 5.882e-2, from a per-element relative
statistic — and that statistic *cannot* distinguish them, because both grids saturate e4m3's
near-amax bound, so it reads the same number for any grid. Two lessons, not one: the grid
choice was already settled by the writer's launch shape, so the wrong number changed no
decision; but it was quoted to three sessions as evidence, and evidence that cannot come out
differently is not evidence.

**5.882e-2 is not e4m3's floor**, which an earlier draft also claimed. It is the bound near a
row's absmax. One e4m3 code is worth up to 100% relative at 3 mantissa bits, so a small
element in a wide-range row exceeds it: over 8 seeds two fixtures read 0.268 and 0.067.
Restricted to elements above 1% of their row's amax the worst is back at 0.0586. Per-element
relative error on an fp8 grid is unbounded near zero by construction, which is why the
accuracy column here is absolute error over the row's amax.

**e4m3 over e5m2 on measurement.** 1.89x better on the worst element (5.882e-2 vs
1.111e-1), 2.14x on the typical, and `zeroed_frac` 0.000% in all eight arms at an
in-block dynamic range up to 3.1e4 — nothing underflowed, so e5m2's extra range buys
nothing it could be paid for with. Confirmed on real 27B-shaped KV at both grids: e5m2
reads 0.0714/0.1094 where e4m3 reads 0.0357/0.0574. The note had left this open on the
grounds that picking now would be picking by analogy with the weight path; it is picked by
measurement.

**`attn_prep_fp8` takes its K amax post-RoPE.** RoPE is a rotation: it preserves the
pairwise norm but not the per-element absmax. Over 2000 random rows at D=256, RD2=64,
the post-RoPE absmax reaches **1.383x** the pre-RoPE one — against a pre-RoPE scale
that quantizes the worst element to 620 and saturates e4m3's 448, in the kernel that
is the sm90 hot path. K is staged through the rotation in shared memory before the
reduction. `write_tokens_fp8` has no such hazard and needs no staging.

Both writers take the **raw** plane plus the scale plane. `kv_layer()` hands back a
dequantized copy, and a scatter into that copy is discarded with no error — which is
what the previous refusal existed to prevent. The refusal now fires only where a
fused writer has no fp8 twin, so sm70 raises rather than dropping writes (verified:
sm90 accepts, sm70 refuses on `write_tokens`, cpu/metal reach the torch path).

## Three defects the audit found, none where the design pointed

Enumerating every site that touches the pool — 67 of them — rather than the ones the
design note named:

**A one-directional guard that fails every running request.** The SSD tier caught
fp8-pool-reads-bf16-spill and returned False cleanly; the reverse, a bf16 pool
adopting an fp8 spill, reached `index_copy_` and raised `RuntimeError` on the dtype
mismatch. That escapes `load_kv` through `_fault_in` (whose `try` wraps only a
`finally`) into `PrefixStore.lookup` and out of `Engine._admit`, whose own comment
says an exception there fails **every** running request. Reachable because `kv_fp8`
is not a `ModelConfig` field and so was absent from `_weight_fingerprint`: flipping
`--kv-fp8` against the same `--ssd-path` left the fingerprint identical. Fixed at both
levels — the fingerprint now carries the format, and the guard is symmetric. Fixing
only the guard would have left the *silent* direction open: a store whose dtype a run
can read but whose numerics it should not trust.

**`fetch_bytes` under-reported the bytes it moved** by 1.56%, summing only `k` and
`v` and never the scale — biasing `read_bytes_per_s` high and `break_even_tokens`
low. Folded into `_blob_bytes` so there is one formula, not two.

**`_fit_blocks` charged the draft pool at fp8+scale rates** while `spec.py` allocates
it at the IO dtype with no scale plane, under-asking ~2x on that term.

KV bytes are computed in **three** places, not one — `bytes_per_token`, `_fit_blocks`
(which must reimplement it from `cfg`, since it runs before the pool exists), and
`scripts/probe_block_bytes.py`. They agree by hand-maintained duplication.

## What the card measured

Two writer arms and two reader arms, all on H20 card 0. The reader pair is two numbers
because the obvious single number was misleading:

| arm | number | what it says |
|---|---:|---|
| scalar fp8 load + scale → bf16 tile, vs torch | **0.0** | the reader change lowers at all, exactly |
| writers vs `reference.quant_kv_fp8`, 8 seeds | byte delta **1** on 7, **0** on 1 | see below |
| readers: kernel dequant vs torch's, same values | **0.0** | the in-kernel multiply is right |
| readers: fp8 pool vs **bf16** pool | **3.16%** of output amax | what fp8 KV actually costs |

The first reader number was the whole gate in my first draft, reported as
`max_abs_err: 0.0`. It is exact and it is nearly meaningless: the oracle read the
*dequantized* pool, so both arms saw identical numbers, and 0.0 only says the multiply
is right. The accuracy question needs the original bf16 pool as the reference, which is
the fourth row — 0.0117 absolute against an output amax of 0.371.

**The writers are not bit-identical, and the difference is measured to be ties.** Byte
delta is exactly 1 on seven of eight seeds, never 2; the scale agrees to 1.19e-07 (one
f32 ulp); and every differing element sits **9.5e-07** from the midpoint between its two
candidate e4m3 codes. So `T.cast` and torch's `.to()` round opposite ways on exact
midpoints — a tie-break, not a disagreement. That claim needed the midpoint distance:
"byte delta is 1" alone is equally consistent with a real error.

The fixture had to be seeded to say any of this. Unseeded, `bitexact` read false on one
run and true on the next, and **the difference was the draw** — neither reading meant
anything, and I reported the second one as a correction of the first before noticing.

## What the 27B measured

The same flag on `Qwen3.8-27B-NVFP4`, H20 card 0, run
`tilerl-kvfp8-27b-s3c`. `scripts/probe_kv_fp8_27b.py`, 2048-token prompt, 24 new tokens.

| arm | number |
|---|---:|
| next-token agreement, fp8 pool vs bf16, same engine | **24 / 24**, no divergence |
| bytes per token | 65536 → **33280** = **1.9692x** |
| K error over row amax, e4m3 per-token | **0.0357** |
| K error over row amax, e4m3 block_head (not chosen) | **0.0588** |
| K error over row amax, e5m2 per-token | **0.1111** |
| elements zeroed, per-token vs block_head | 9.0e-06 vs 1.16e-05 |

Two things carry over from tiny unchanged: the grid gap is **1.65x** here against 1.64x
there, and e4m3 beats e5m2 by 3.1x. The 27B's real KV has amax p50 5.66 and max 22.9, so
it sits well inside e4m3's range — the format is not the constraint, the grid is.

Agreement being exactly 24/24 is the accept condition, not a strong result on its own: 24
greedy tokens is a short window, and this entry claims nothing about quality over a long
run.

### The capacity claim, demonstrated — and the throughput it costs

B=32 x 32k on one H20, both pools fitted by `_fit_blocks` on the same free card:

| | bf16 | fp8 |
|---|---:|---:|
| blocks fitted | 45294 | **89179** (1.9689x) |
| peak requests resident of 32 | 22 | **32** |
| wall clock for the same 32 requests | **702.4 s** | 1215.8 s |

The block ratio matches the byte ratio to four digits — the fit is exactly proportional,
with no per-block overhead unaccounted for. And bf16 could hold only 22 of the batch, so
it serialized the rest, which is the shape the capacity claim predicted.

**Then fp8 took 1.73x the wall clock anyway.** More concurrency and half the KV bytes, and
it still finished the same work slower. So the honest end-to-end reading at this shape is a
capacity win that does not convert: the readers dequantize at every gather — a cast and a
multiply per KV element — and at 32k context attention reads enough KV for that ALU cost to
exceed the bytes it saves.

The two arms did not run at equal concurrency (22 vs 32 rows), so this does not isolate
per-tick cost; it is a whole-run comparison at equal total work. The per-tick decode split
that would isolate it is the pending number.

## Traps, each a rule for the next kernel in this tree

- `T.alloc_fragment((2,), ...)` then `part[1]` is rejected: *"Only fragment[0] access is
  allowed."* A fragment is per-thread, so a two-quantity reduction needs two 1-element
  fragments, not one of size 2.
- A **closure local** in a `T.Tensor` annotation is a `NameError` at build time — the
  eager builder re-executes the body with only its own kwargs bound. A jit **parameter**
  in the same position works, tested directly, which is what lets one body serve both
  dtypes instead of the duplication `write_tokens_f32` needed for exactly this reason.
- The off-fp8 dummy scale must be `(1, Hkv, block_size)`, not `(1,)`: `Hkv` and
  `block_size` come from `T.const` and are bound from the real operands, so a mismatched
  dummy fails the packed-ABI check.
- `build_engine`'s `num_blocks` defaults to **64**, and `_fit_blocks` runs only
  `if not num_blocks`. A nonzero default means "measure free memory" never happens
  unless a caller passes 0 — the 27B probe ran a day of arms against a 64-block,
  1024-token pool, which refused its own 2048-token prompt before any forward.
- `_admit` returns **False** when the pool is short; it does not raise, by design
  (`engine.py:672` — a raise there reaches `step`'s handler and fails every running
  request). So an over-large batch is admitted as far as it fits and the rest queues.
  A capacity claim of the form "bf16 OOMs and fp8 serves" is therefore unreachable
  through the engine: the observable is how many requests are resident at once.
- An **uncapped** fit is the other half of the same trap. `_fit_blocks` takes 2/3 of what
  is free, which on a nearly-empty card is ~54 GiB of KV — and then the prefill
  transients at B=8 have nowhere to go (OOM with 94.27 GiB in use). The cap has to come
  from what the batch can address (`max_blocks`, as `cli.py:132` passes), not from what
  is free. Fixing the 64-block bug is what exposed this: a pool small enough to be wrong
  was also small enough to leave room.
- A number known **before** the step that might fail has to be recorded there. The
  boundary arm's `blocks_ratio` was computed alongside the peak-resident count, so an
  arm that OOMed in the transients lost the capacity answer it had already measured.

## The gate, and its negative control

`paged_attention` raises on a raw fp8 plane with no scale. `_dev` would have up-cast it
— 448x too small, finite, and plausible enough that **both** the token-agreement and
max-logit gates pass with the treatment entirely absent. This is the third instance of
that class in this work; the first two were the bare `.to(fp8)` cast and the fused
writers scattering into a dequantized copy. The refusal sits **above** the arm dispatch,
because the sm70 split arm reached `self._f32(k_cache)` with the same exposure — a
per-arm guard would have covered the arm I was looking at.

Token agreement cannot see a scale-less cast at all: e4m3 is a float format, and the
tiny model's KV sits inside its range, so the mutant generates the identical 6 tokens.
The gate therefore asserts three things a passing token match cannot: the scale-shift
mutant changes the output, the scale plane's shape, and that writing 16 tokens one at
a time is **bit-identical** to quantizing the block once. That last assert was
verified non-vacuous by reverting the write path to the compounding form — it fails
at `tests/test_e2e.py:1425`.

## Rule

A quantization grid is chosen by the writer's launch shape and its append rule, not
by round-trip error. Measure whether an append re-rounds what is already stored,
because that error compounds where the round-trip number does not — and check the
FIRST element written, which is where compounding shows and where a max-over-tensor
statistic hides it.

Second rule, from four instrument failures in one day: **a control is evidence only if
you can name the mechanism by which it would fail.** An unseeded fixture, a marker
written before the seed, the wrong directory, and a range arm handed an already-quantized
pool each produced a plausible number that would have shipped. Three of them read as a
*pass*.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-07 | 0d14ab2 | this Mac | cpu | tiny | — | — | — (parity only) |
| 2026-09-07 | e2d30c3 | H20 card 0 | sm90 | (kernel arms) | — | — | — (correctness only) |
| 2026-09-07 | 9497f92 | H20 card 0 | sm90 | qwen38-27b | — | — | — (accuracy + range only) |
| 2026-09-07 | 3eed93f | H20 card 0 | sm90 | qwen38-27b | ~7% slower | — | 1492.9 -> 862.4 tok/s over the run (0.578x) |

The cpu row is parity and byte accounting: 473 tests pass, the fp8 gate green with its
negative control, no timing claimed — the C backend cannot codegen fp8 at all, so both
the writers and the readers are card-only.

The sm90 kernel row is the four arms above, run under `tilerl-kvfp8-gate3` … `-read4`.
Kernels compile, parity is ties-only, and the reader path costs 3.16% of the output amax.

The 27B row is agreement and range (the table above), run under `tilerl-kvfp8-27b-s3c`.
No rate, because the run OOMed after those two arms for the reason in the traps list.

The B=32 row is the whole-run rate at equal total work and unequal concurrency (bf16 22
rows, fp8 32), so it is an end-to-end reading, not a per-tick one.

Still pending, and it needs a card window with the 42 GB checkpoint:

1. **decode ms/tick at B=8 ctx=8k**, fp8 against bf16 at equal concurrency, whose ceiling
   is 1.079x. The one number that could still show a gain, and the only one that isolates
   the readers' per-gather dequant from the writers' quantize-on-write. B=1 is not the
   cell: its ceiling is 1.011x at 8k and 1.041x at 32k, both under run-to-run variance.

Raw artifacts: `scripts/probe_kv_fp8_kernels.py` (the four card arms, JSON on stdout)
and `scripts/probe_kv_fp8_27b.py` (the 27B arms). The append-rule and RoPE-absmax
numbers were measured in one-off scripts and are restated in the reference self-check
and the `attn_prep_fp8` docstring rather than kept as files.

## What this does not claim

- **No speedup anywhere, and a measured slowdown in both phases.** Prefill ~7% slower at
  B=8 ctx=8k; the B=32 x 32k run 1.73x slower end to end despite holding 10 more requests
  resident. A per-tick decode ratio is still unmeasured, and it is the only place a gain
  could still be.
- The 1.969x is a **capacity** figure — bytes resident per token, and now also blocks
  fitted on one card — not a throughput one. The throughput form it implies at saturation
  did **not** appear: fp8 admitted the whole batch that bf16 had to queue and was still
  slower. Capacity is what this flag buys; it does not buy speed at these shapes.
- Multi-session hit rates are not an fp8 measurement. A miss publishes ~62 interior
  entries at a 31k prompt and each evicts another session's shared head (v100's H20
  cell, entry at 541a37c); `_entries_capacity` divides by the GDN snapshot, never KV
  bytes, so fp8 KV is neutral to that cascade in both directions.
