# A speculative rate divided by a dense roofline — 2026-09-06

> Status: closed 2026-09-06. The four wrong denominators are fixed here, and the
> **32-42% bracket collapses to 38.2% / 34.3%** now that a draft forward's streamed
> bytes are measured: **1.065 GB**, against the 1.65 the wide end charged.
> Removed from OPEN.md.

## Context

Four sites cite the retired `64 tok/s` weight roofline as a live denominator. The
obvious repair is to swap in the current figure. Doing the arithmetic first
found the wrong defect: the denominator's **value** is the smaller problem, and
updating it alone moves every conclusion in the wrong direction.

## Root Cause

**The numerator is a speculative rate and the denominator is a dense
single-token roofline.** A weight roofline is bytes-streamed-per-forward over
bandwidth, so it bounds **forwards per second**. Speculation exists to emit more
than one token per forward — 2.95 tok/forward measured beside the 52.7 — so a
speculative tok/s is not the quantity that bound applies to. Dividing them
compares a rate against a ceiling that is `tok/fwd` times too low.

The correction is one identity, and it needs no byte count at all:

```
% of the speculative ceiling  =  (% of the dense roofline) / (tok per forward)
```

Both sites, with the roofline that was current **when each was measured**
(16.04 GB → 56.1 tok/s; the 14.44 GB → 62.3 f16-scale figure postdates them by a
day, so using it would be a second era error):

| site | published | numerator is | tok/fwd | era-correct vs dense | **vs the speculative ceiling** |
|---|---|---|---|---|---|
| `wins/2026-09-01-sm70-gemv-packed-x-f16.md:83` | 52.7 is **82%** of 64 | depth 3 | 2.95 | 94% | **32%** |
| `wins/2026-09-01-sm70-attention-thread-redundancy.md:93` | 46.5 is **73%** of 64 | depth 3 | 2.90 | 83% | **29%** |
| `LOG-v100-sm70.md:211` | 52.7 is **82%** of 64 | depth 3 | 2.95 | 94% | 32% |
| `CHANGELOG.md:333` | 52.7 is **82%** of 64 | depth 3 | 2.95 | 94% | 32% |

**Fixing only the stale value makes it worse.** Against the f16-era 62.3, 52.7
reads **85%** and 46.5 reads **75%** — both *higher* than the figures they
replace, so the repair that looks like diligence publishes a stronger claim on
the same broken comparison. That is what makes this worth an entry rather than a
sed: the stale-citation framing predicts a correction downward, and the tell
that the framing is wrong is that the correction goes up.

**The upper bound is what is claimable; the exact share is not.** 32% assumes the
draft forwards stream **zero** bytes, which is false — it is the ceiling's
ceiling, so 32% is an upper bound on the share and the true share is lower.
Charging the draft head plus lm_head at f32-era bytes for all three draft steps
(3 × (0.85 + 0.80) = 4.95 GB, from
[roofline is the streamed subset](2026-09-02-roofline-is-the-streamed-subset.md))
gives 126.5 tok/s and 42%. The two ends bracket the answer at **32–42%**, and
both reject 82%. Which end is right depends on the draft's streamed bytes, which
nothing in the tree measures — the 0.85 GB is the head's *resident* size, and a
draft step's streamed subset has never been bucketed the way
`check_scale_f16.py` buckets the trunk's.

## The missing operand, measured — 2026-09-06

`scripts/draft_streamed_bytes.py` buckets the served head the way
`check_scale_f16.py` buckets the trunk: per tensor, at the dtype the engine
actually serves, keeping only what a forward multiplies by. The shard is
`model-00018-of-00018.safetensors`, 18 tensors, 1.645 GB as stored.

| bucket | GB | why |
|---|---:|---|
| head as stored (bf16) | 0.849 | the file's own tensors — this is the 0.85 the bracket used |
| head as **served** (fp4 on sm70) | **0.265** | `_quantize_draft` re-packs every [N,K] ≥128×128 to 4-bit + one scale per 32, charged f32 for these sites' era (see the f16 note below) |
| trunk `lm_head` it reads out via | 0.800 | `read_head_params` **skips** the head's own `lm_head`/`embed_tokens`/`final_norm` — warns "the trunk's are shared" |
| **one draft forward streams** | **1.065** | 0.265 + 0.800 |

Three tensors were skipped by the loader and the probe printed which:
`lm_head.wq`, `lm_head.scale`, `lm_head.oscale`. So the readout is charged once
at the trunk's 0.800, not twice.

**0.849 → 0.265 is the whole correction.** The bracket's wide end charged the
head at its *resident bf16* size while the engine serves it fp4 — the same class
of error as counting `embed_tokens` in the trunk roofline, one level down. 1.065
against the 1.65 charged is 1.55x.

Depth 3 is 1 trunk forward + 3 draft forwards = 16.04 + 3 × 1.065 = **19.24 GB**,
so **46.79 forward-sets/s** at 900 GB/s. Applying the entry's own identity with
each site's measured acceptance:

| site | published | tok/fwd | ceiling | **share** |
|---|---:|---:|---:|---:|
| `wins/2026-09-01-…-gemv-packed-x-f16.md:83` | 52.7 | 2.95 | 138.0 | **38.2%** |
| `wins/2026-09-01-…-attention-thread-redundancy.md:93` | 46.5 | 2.90 | 135.7 | **34.3%** |

Both land inside the bracket, and the probe reproduces both of its ends from the
same code path — 0 bytes → 165.5 tok/s → 31.8%, and 1.65 GB → 126.5 → 41.7% —
which is what makes the 38.2% a reading of the system rather than of a new
script. At perfect acceptance the ceiling would be 187.1 tok/s; that variant is
not what 32-42% was computed against and is printed separately so the two are not
confused.

The fp4 byte model is asserted against `reference.pack_fp4`'s own output before
any total is printed: 81,920 modelled against 82,944 real on a 256×512, 1.2%
apart (the gap is the oscale plane, one f32 per row).

**What this does not measure:** the KV plane. A draft forward reads its own KV,
which grows with context and is not part of a weight roofline — the same
exclusion every other roofline here makes, stated so the 1.065 is not read as
total traffic.

### Reproduced independently, and the one operand the model rounds up — 2026-09-07

Re-run on the pod from a second reading of the shard header, no card taken:
18 tensors / 1.6450 GB stored, 15 kept after the loader's skip, 0.8494 stored →
0.2655 served, **1.0655 GB per forward, 38.2%** — every published figure to the
digit, and both bracket ends (31.8% / 41.7%) off the same path.

One term in the byte model is deliberately the wrong era, and it belongs written
down rather than found later. `_served_bytes` charges the block scale at **4 B per
32 weights**, but sm70 serves that plane at **f16**: `Backend.scale_io` is
`torch.float16` on this arch (`backend.py:356`, since `f709df8` on 09-02) and
`materialize`'s `narrow` branch (`backend.py:891`) casts every `.scale` during the
device move. The draft head goes through the same `materialize`, so its served
plane is 2 B per 32, not 4.

Recomputed at f16 scales: 0.2655 → **0.2389** GB served, 1.0655 → 1.0389 per
forward, ceiling 138.0 → 138.6, share 38.2% → **38.0%**.

**0.2pp, and the f32 figure is the right one to publish here** — the two sites are
09-01, a day before the f16 flip, so charging f32 is the same era discipline that
made this entry pair 16.04 with 56.1 rather than the newer 14.44 with 62.3. The
number is not wrong; it was un-named, which is what makes it worth a line: an
operand carried at a retired width silently becomes an error the moment someone
re-uses the probe on a current measurement. A **post-09-02 draft rate must pass
`--scale-bytes 2`**, and the flag exists now so that is a switch rather than an
edit.

## Four other citations of 64 are correct and are not touched

The same grep returns four more, and the criterion that separates them is
whether the number is used as a **live** denominator:

| site | why it stands |
|---|---|
| `wins/2026-08-31-sm70-split-kv-decode-attention.md:96` | already says it "originally cited a remembered 14 GB / 64 tok/s" and links the correction |
| `wins/2026-08-29-sm70-volta-fp4-cell.md:90` | reports that day's *estimate*, immediately followed by "Measured 19.9" |
| `errors/2026-09-01-spec-warmup-one-width.md:35` | a sanity check — 289 tok/s exceeded it, and a dense bound rejects 289 at any of these values |
| `scripts/bench_b1_decode.py:50` | the same check, in the comment that keeps the warmup honest |

The last two are the interesting pass: they divide a *speculative* 289 by a dense
bound too, and the comparison survives anyway because a bound only has to be
exceeded to fire. A one-sided test tolerates a denominator an equality cannot.

## Fix

Replace the four live sites with the ceiling the numerator belongs to, stated as
a bracket, and keep the era-correct dense figure beside it since the dense
comparison is the one the kernel work was judged on. No runtime change.

## Rule

Before dividing by a roofline, check that the numerator counts the same events
the roofline bounds. A weight roofline bounds forwards; speculation, batching,
and any form of multi-token emission put a factor between forwards and tokens,
and that factor is exactly the overstatement.

Second, from how this was nearly mis-repaired: **a stale operand and a wrong
operand need different fixes, and the stale framing hides the wrong one.**
"Update the number" is a mechanical edit that never re-derives the comparison, so
it cannot find a dimensional error — and here it would have raised 82% to 85%
while looking like a correction. When a repair moves a number *away* from
conservative, stop and re-derive instead of committing it.

Third: a bound used one-sidedly can survive a denominator that an equality
cannot. Do not read the four surviving citations as evidence the number was fine.

Fourth, from closing the operand: **a served size is not a stored size, and a
bracket's wide end is where that difference hides.** 0.85 GB was read off the
checkpoint; the engine serves the same tensors fp4 at 0.265. Both ends of a
bracket deserve the scrutiny a point estimate gets — the wide end reads as the
conservative choice, which is exactly why nobody re-derived it.

## Results

| date | commit | host | target | model | shape | metric | value |
|---|---|---|---|---|---|---|---|
| 2026-09-06 | dc612c6 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 draft shard | depth 3 | draft head served GB | **0.265** |
| 2026-09-06 | dc612c6 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 draft shard | depth 3 | one draft forward GB | **1.065** |
| 2026-09-06 | dc612c6 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | depth 3 | 52.7 tok/s share of spec ceiling | **38.2%** |
| 2026-09-06 | dc612c6 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | depth 3 | 46.5 tok/s share of spec ceiling | **34.3%** |

No runtime change — a probe, this section, and one OPEN.md row removed.

| date | commit | quantity | published | corrected |
|---|---|---|---|---|
| 2026-09-06 | (this) | depth-3 52.7 tok/s vs its own ceiling | 82% of 64 dense | **38.2%**, 94% of era dense |
| 2026-09-06 | (this) | depth-3 46.5 tok/s vs its own ceiling | 73% of 64 dense | **34.3%**, 83% of era dense |
| 2026-09-06 | (this) | draft step streamed bytes | — | **1.065 GB** (0.265 served + 0.800 trunk lm_head) |
