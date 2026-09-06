# V100 prefill is per token and quadratic, not per chunk — 2026-09-07

> The 512-token chunk budget is not the lever. Prefill cost is the sm70 attention
> kernel, and it grows as n², which is 68% of the cost of a 21.7k-token prompt.

## Context

v100-sm70-fp4-55's agent trial measured a Claude Code turn at **314 s, 294 of them
before the first token**, for a 21,727-token prompt — 43 prefill ticks at
`max_num_batched_tokens = 512` (`engine.py:183`, no CLI flag), about 6.8 s per
chunk. Two explanations predict that equally well: a per-token kernel cost, or a
fixed per-chunk overhead. They imply opposite fixes — raise the chunk budget, or
fix the sm70 prefill kernel — so the question had to be settled before either.

## The measurement problem, first

**A plain length sweep cannot answer this.** `chunks = ceil(tokens/512)`, so
tokens and chunks are collinear and both models fit the same curve; a linear
TTFT-vs-length result is equally consistent with either, including the 294 s /
43 ticks that prompted the question. The brief's grid (256/512/1024/2048/4096/8192)
is exactly that shape.

What separates them is **pairs with the same chunk count and different token
counts**, found by replaying the scheduler's own cut (`engine.py:678-698`, which
truncates the first chunk to a 64 multiple) rather than assuming `n/512`.

My first attempt at such a pair was **wrong in the direction that would have
reversed the conclusion**: I picked 512 vs 544 believing they shared a chunk
count. They do not — 512 is one chunk, 544 is two — so a flat result across that
pair would have been read as "per-chunk" when it was measuring an extra chunk.
Replaying the cut in a scratch script caught it before the run.

## Result

One client, fresh text per arm (`prefix_hits +0` on every row, asserted), TTFT
timed client-side from the first content frame.

| ask | prompt_tokens | prefill_forwards | TTFT (s) | s/token | tok/s |
|---|---|---|---|---|---|
| 512 | 460 | 2 | 2.59 | 0.0056 | 178 |
| 544 | 495 | 2 | 2.57 | 0.0052 | 193 |
| 1024 | 1011 | 2 | 5.16 | 0.0051 | 196 |
| 1568 | 1596 | 4 | 8.44 | 0.0053 | 189 |
| 2048 | 2116 | 5 | 11.80 | 0.0056 | 179 |
| 3616 | 3801 | 8 | 22.58 | 0.0059 | 168 |
| 4096 | 4316 | 9 | 26.15 | 0.0061 | 165 |
| 8192 | 9483 | 19 | 77.68 | 0.0082 | 122 |

Repeats at 1024: 5.16 / 5.25 / 5.22 s — **sd 0.04 s, 0.8%**. Every difference
discussed below is far outside that.

**The discriminating pair.** 544 → 1024 holds `prefill_forwards` at **2** while
prompt tokens go **+104%** and TTFT goes **+101%**. Cost tracks tokens, not
chunks. A per-chunk model predicts those two arms are equal; they differ by 2x.

**The chunk count adds nothing once tokens are known.** Residuals of the
token-only fit against `forwards` are ≤0.47 s across a 2.57–77.68 s range, with
no structure — 2-forward arms scatter both signs, the 19-forward arm sits at
+0.08 s.

## It is quadratic, and that is the real finding

`s/token` is not flat: 178 tok/s at 460 tokens, **122 tok/s at 9483**, a 1.45x
slowdown across 20x length. A straight line hides this — R² 0.9827 looks fine and
the intercept is **−3.87 s**, which is not a physical quantity.

```
ttft = 0.56 + 0.00422·n + 4.117e-07·n²        R² 0.9999
```

The n² term is attention over the prompt. Its share of the total:

| n | linear part | quadratic part | quadratic share |
|---|---|---|---|
| 1,000 | 4.2 s | 0.4 s | 9% |
| 4,000 | 16.9 s | 6.6 s | 28% |
| 9,483 | 40.0 s | 37.0 s | 48% |
| 21,727 | 91.7 s | 194.3 s | **68%** |

**Out-of-sample check.** At n=21,727 the quadratic predicts **287 s** against
v100's independently measured **294 s** — 2.4% off, from a fit whose largest arm
was 9,483 tokens. The linear model predicts 174 s, off by 41%. That number was
measured by a different session, on a different day, through a different client,
and was not used to build the fit.

## What this means for the fix

**Raising `max_num_batched_tokens` buys nothing.** It changes how the same tokens
are grouped, and grouping is not what costs. It would make each tick longer and
each tick blocks the engine lock, so it would make the `/health` problem below
worse for no throughput gain.

**The sm70 prefill kernel is the target.** A prompt of this size should not be
paying n² at 4.1e-07 s per token². Worth checking against the shared-kernel-name
history — sm70 has previously run the CPU schedule in prefill — before assuming
the schedule is what it looks like.

## `/health` blocks under prefill, and it corrupted an earlier instrument

`Engine.stats()` takes `self._lock` (`engine.py:731`) and `step()` holds that lock
across each forward. Measured in a dedicated arm, run last so it could not
contaminate a timed row:

```
idle:             1 ms
during prefill:   median 20,206 ms   max 55,337 ms    (17,156x)
```

**My first probe polled `/health` once a second from inside every timed arm.**
That poll waits on the same lock as the work being timed, so it would have
inflated the TTFT it was measuring and I would have fitted a model to my own
instrument. Caught by v100's warning, not by me; removed from every timed arm.

Two further instrument corrections, both of which produced a plausible wrong
answer rather than an error:

- **`prompt_tokens: None` on every row.** The streaming route only attaches usage
  when `stream_options.include_usage` is set (`server.py:475`); without it the
  per-token axis of the whole measurement silently does not exist. The probe now
  raises instead of fitting on None.
- **A quiet gate that was already satisfied.** `running == 0 and waiting == 0`
  is true while the previous request's last tick is still in flight — v100 caught
  the counter rising after `running` hit 0. The gate now requires
  `prefill_forwards` and `decode_forwards` to hold still across a second.

## Caveats

- The 256-token arm came back at 1190 tokens: it ran before the words-to-tokens
  ratio was calibrated. Excluded from every fit and from the table.
- Two intended pairs (1568/2048, 3616/4096) came out with different forward
  counts because measured tokens drifted past a boundary. The probe prints them
  as NOT a pair rather than fitting them. Only 544/1024 is a valid pair — one is
  enough to kill the per-chunk model, but a second would have been better.
- Single card, single request, sm70, NVFP4 27B. Nothing here is claimed for H20
  or for a batched endpoint.

## Rule

When two models predict the same curve, a better fit to that curve chooses
neither. Find the pair of points where they disagree, derive it from the code
that decides rather than the formula you believe, and check the derivation can
fail — mine did, in the direction that would have flipped the answer.
