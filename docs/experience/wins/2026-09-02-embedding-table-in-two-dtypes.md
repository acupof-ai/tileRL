# The embedding table lived on the card twice — V100 (sm70), 2026-09-02

> Status: fixed. **2.37 GiB back**, and it is the difference between `serve`
> answering and `serve` dying on its first token. First working end-to-end
> completion from the real 27B on this card.

## Context

`serve --model qwen38-27b` reached `/health`, reported `status: ok`, and showed
30082 of 32510 MiB used. The first chat completion then died:

```
CUDA out of memory. Tried to allocate 4.74 GiB.
  4.14 GiB is free ... backend.py line 1015, in embedding
```

`248320 × 5120 × 4 = 4.74 GiB` — the f32 embedding table, allocated on the first
forward, long after startup had declared everything fine.

## What Worked

`materialize` had already moved the bf16 table to the device (2.37 GiB). Then
`embedding` cast it to f32 *and cached the result*, so the same tensor occupied
**7.11 GiB in two dtypes at once**.

The cast was not gratuitous — sm70 cannot codegen a bf16 load, which is why
`Backend.io` is f32 on this arch. But it loads **f16** natively, and that was
never used here. The fix is two lines of policy:

1. `materialize` casts the table to `embed_io` during the device move it was
   already doing — `bfloat16` on sm90, **`float16` on sm70**, f32 on the C target.
   Untied heads only: a tied table is also the `lm_head` weight, and that linear
   wants f32.
2. `embedding` reads whatever narrow dtype it finds and widens in the gather
   (a new `embedding_f16` kernel body). It never narrows a table itself — that
   would be the same second copy, one dtype smaller.

One place decides the dtype; the other obeys. That is what makes a second copy
unrepresentable rather than merely absent.

Numerically this drops the 2 mantissa bits f16 has fewer of than bf16, on a value
whose next operation is a rmsnorm. f16's exponent range easily covers embedding
weights; the gather still emits f32.

## The budget, which is the real finding

| what | GiB |
|---|---:|
| fp4 body (64 layers, nibbles + f16 scales) | 14.17 |
| lm_head (fp4) | 0.74 |
| embedding table (f16; f32 was 4.74) | 2.37 |
| KV pool @ `--blocks 2048`, f32 | 4.00 |
| GDN recurrent states, 4 slots | 0.56 |
| spec `step_states` @ depth 3 | 1.69 |
| conv + step windows | 0.53 |
| draft head (fp4) | 0.25 |
| **accounted** | **24.31** |

The OOM message itself closes the rest, and the agreement is the check on the
model — I did not have to guess at the slack:

| torch / driver | GiB |
|---|---:|
| PyTorch allocated | 23.37 |
| reserved but unallocated (allocator slack) | 3.86 |
| process total | 27.59 |
| `nvidia-smi` at `/health` | 29.38 |

23.37 allocated against 24.31 accounted is 4% high — close enough that no
allocation is unexplained, and the residual is the two fused projections' scale
planes being narrower than my uniform `/32` assumption. The gap from 27.59 to
29.38 is the CUDA context plus the loaded tilelang modules.

Every line is individually defensible. The failure only existed in the sum: 23.37
allocated plus 3.86 slack left 4.14 GiB free, and the f32 table wanted 4.74.

## Rule

**A memory budget is a sum, so audit it as a sum.** Nothing here was oversized;
the OOM lived in the total, which no single allocation site can see. The audit
that catches it is arithmetic over numbers the config already holds, and it costs
one line — cheaper than one 5-minute weight load.

Second: **a dtype conversion that caches its result has doubled the tensor.** The
cache is what turns a transient cast into a permanent second copy, and it is
exactly the cache that makes the cast look free in a profile.

Third: **`/health` returning ok proves startup, not capacity.** Allocation
continues into the first forward — the activations, the widened constants, the
graph capture. A server that answers `/health` has not yet demonstrated it can
answer a request.

## Results

| date | commit | machine | target | model | table dtype | table GiB | serve first token |
|---|---|---|---|---|---|---:|---|
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | f16 | **2.37** | **works** |
| 2026-09-02 | a317f61 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | bf16 + f32 | 7.11 | OOM |

The first real completions this card has produced, all correct:

| prompt | reply | finish |
|---|---|---|
| `Reply with exactly: tunnel works` (thinking on) | `<think>The user is asking me to reply with exactly "tunnel works". This is a simple, direct request.` | `length` @ 24 tok |
| `What is 17 times 23?` (thinking off) | `391` | `stop` |
| `Name three primary colors.` | red, blue, yellow — plus an unprompted note that RGB differs for screens | `stop` |

391 is right, and it is worth saying why that matters more than the string: it is
three tokens of arithmetic through 64 layers of 4-bit weights, an f16 scale plane,
an f16 embedding table and a speculative draft head — every lossy choice on this
branch, composed, on a question with one correct answer.

Decode and prefill throughput are unchanged — the table is read once per token by
a gather that was never on the critical path. This buys capacity, not speed.

The served rate itself turned out to be fine: it warms up over ~6 requests
(1088 → 26.2 ms/token) and lands within 1.27× of the bench. I first reported that
as a 6.5× regression that degraded per request; see
`errors/2026-09-02-a-rate-needs-equal-work.md` for why that reading was wrong.
`serve` now warms up at startup, which it never did.

Raw artifact: an ad-hoc curl probe on the pod against `serve`, whose working
invocation `docs/serve-v100.md` documents. Related:
`errors/2026-09-02-serve-never-sized-its-kv-pool.md`, the same demo rehearsal and
the three walls in front of this one.
