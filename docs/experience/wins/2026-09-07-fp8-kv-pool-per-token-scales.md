# fp8 KV pool, per-token scales — cpu (parity) + sm90 (pending-remote), 2026-09-07

> Status: Shipped (flag, default off) — **no tok/s claim**; the card arms are pending-remote

## Context

`--kv-fp8` stores the KV planes in e4m3 instead of bf16. The plane is what a decode
tick re-reads per token at long context, so halving it is the reason to want this.

**This entry claims no speedup, and the reason is structural rather than pending.**
The pool and both writers are fp8; the attention *readers* still take a dequantized
plane, because `kv_layer()` dequantizes the whole plane per call. So the bytes moving
through the gather on a decode tick are unchanged, and a tok/s number measured today
would be measuring nothing. Converting the readers is the next PR and is where the
bandwidth win lives.

What this entry does settle is the scale geometry, the append rule, the dtype, and
four things that were wrong in the design note or the tree.

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

**Accuracy does not separate the grids.** Measured on tiny, both give an identical
**5.882e-2** worst element — that is e4m3's 3-mantissa-bit floor, not a grid property.
Per-token moves only the typical element, 2.17e-2 → 2.04e-2, 6%. The design note had
claimed the coarse grid's accuracy cost was "the number the accuracy gate has to
produce"; it was not, and the note is corrected. The coarse grid was a write-path
hazard, not an accuracy one.

**e4m3 over e5m2 on measurement.** 1.89x better on the worst element (5.882e-2 vs
1.111e-1), 2.14x on the typical, and `zeroed_frac` 0.000% in all eight arms at an
in-block dynamic range up to 3.1e4 — nothing underflowed, so e5m2's extra range buys
nothing it could be paid for with. The note had left this open on the grounds that
picking now would be picking by analogy with the weight path; it is now picked by
measurement, with the 27B gate free to overturn it.

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

## The gate, and its negative control

`paged_attention` raises `TypeError` on a raw fp8 plane. `_dev` would have up-cast it
with no scale — 448x too small, finite, and plausible enough that **both** the
token-agreement and max-logit gates pass with the treatment entirely absent. This is
the third instance of that class in this work; the first two were the bare
`.to(fp8)` cast and the fused writers scattering into a dequantized copy.

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

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-07 | 0d14ab2 | this Mac | cpu | tiny | — | — | — (parity only) |
| pending-remote | | H20 card 0 | sm90 | qwen38-27b | | | |

The cpu row is a parity and byte-accounting row: 471 tests pass, the fp8 gate green
with its negative control, no timing claimed — the C backend cannot codegen fp8 at
all, so the writers are card-only.

The pending sm90 arms, each its own `pod_run.sh` invocation under its own claim name:

1. **`tilerl-kvfp8-gate`** — the two writers compile and the parity gate passes
   against the bf16 pool through the same kernels. Nothing else can be believed until
   this runs; the kernels have never been compiled.
2. **`tilerl-kvfp8-range`** — the 27B's real per-block dynamic range and round-trip
   error, both grids, e4m3 and e5m2. Confirms or overturns the tiny-model e4m3 verdict
   on the model that ships. The coarse column is the record of what was not chosen.
3. **decode tok/s at 8k and 32k** — deferred to the reader-conversion PR, because with
   the readers on a dequantized plane there is nothing for it to measure.

Raw artifacts: none yet for sm90. The tiny-model grid and range numbers reproduce with
`uv run python scripts/probe_kv_fp8_range.py --model tiny --prompt-tokens 256`; the
append-rule and RoPE-absmax numbers were measured in one-off scripts and are restated
in the reference self-check and the `attn_prep_fp8` docstring rather than kept as
files.

## What this does not claim

- No decode or prefill speedup, at any context length, on any card.
- The 1.969x is a **capacity** figure — bytes resident per token — not a throughput one.
- Multi-session hit rates are not an fp8 measurement. A miss publishes ~62 interior
  entries at a 31k prompt and each evicts another session's shared head (v100's H20
  cell, entry at 541a37c); `_entries_capacity` divides by the GDN snapshot, never KV
  bytes, so fp8 KV is neutral to that cascade in both directions.
