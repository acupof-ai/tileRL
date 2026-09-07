# The sm70 prefill n² is one kernel, at 30x its floor — 2026-09-07

> Verdict: **write the cell.** At 16,384 tokens on the real 27B,
> `paged_attention` is 100.5 s of a 197.6 s prefill and runs 29.9x its
> arithmetic floor. A query-tiled cell recovers 97.1 s — **49.2% of TTFT**,
> against a 15% threshold set before the numbers arrived.

No number in this entry comes from `tiny`. Every share previously quoted for
sm70 prefill — mine and the peer's — was a 2-layer random model and is withdrawn
(`errors/2026-09-07-a-profile-of-the-wrong-model-printed-a-clean-table.md`).

## The run

Real `qwen38-27b` on the V100, serving child down, #218's gated profiler. The
identity line is the first thing to read, because it is what the withdrawn run
failed:

```
arch=sm70 target=cuda precision=fp4
model=qwen38-27b layers=64 full_attn=16 H=24 D=256
JIT warm-up (untimed, real shapes): 224.7 s / 329.1 s
```

| n | profile | fit | ratio | attention share | compute floor | × compute | bandwidth floor (tile 1) | × bandwidth |
|---|---|---|---|---|---|---|---|---|
| 2,048 | 13.26 s | 10.37 s | 1.28x | 8.5% | 0.05 s | — | 0.38 s | — |
| 8,192 | 69.45 s | 62.20 s | 1.12x | — | 0.84 s | — | 5.19 s | — |
| 16,384 | 197.59 s | 179.66 s | 1.10x | **100.5 s (50.9%)** | 3.36 s | **29.9x** | 20.16 s | **4.99x** |
| 21,727 | 314.49 s | 286.04 s | 1.10x | **185.5 s (59.0%)** | 5.91 s | **31.4x** | 35.18 s | **5.27x** |

All four reconcile inside the 0.5–2.0x band, and **the ratio falls monotonically
1.28 → 1.12 → 1.10 → 1.10**. That is the direction the fit predicts rather than
a coincidence: `_reconcile` compares against `c1·n + c2·n²` and drops the fit's
constant, so the per-chunk sum should overshoot most at small n, where the
constant is the largest fraction. It does, and it flattens where the constant
stops mattering.

The 21,727 arm is out-of-sample twice over. #213's fit was built on arms up to
9,000 tokens and predicts 286.04 s here — 2.4x past its largest input, and the
profile lands 1.10x from it. Separately, 314.485 s is 0.05% from the 314.33 s
measured end-to-end for a Claude Code turn (#212), two instruments that share
no code. **Agreement in magnitude, not precision** — the second pair especially,
where four significant figures of coincidence is what it looks like.

## The n² is one kernel

16,384-token arm, first of 32 chunks → last, prefix 0 → 15,872:

| op | first chunk | last chunk | ratio |
|---|---|---|---|
| `paged_attention` | 0.0889 s | 6.7584 s | **75.98x** |
| `linear_fp4` | 2.4895 s | 2.4707 s | 0.99x |
| `linear_attn_chunk` | 0.4375 s | 0.4405 s | 1.01x |
| `rmsnorm` | 0.0345 s | 0.0349 s | 1.01x |
| `rope` | 0.0090 s | 0.0089 s | 0.99x |

One op rises with the prefix. Everything else is flat within 1%. `linear_fp4` is
a flat per-token cost in all 32 chunks — it is the linear term, `paged_attention`
is the quadratic one.

**The 16,384 arm is the one to quote, because 16,384 = 32 × 512 exactly.** At
21,727 the script's printed trend reads `paged_attention` 51.31x, `linear_fp4`
**0.51x**, `linear_attn_chunk` 0.17x — the last chunk there is 223 tokens, 43.6%
of a full one, so every op is measured on a short chunk and two of them appear to
get *cheaper* with prefix. Against the last **full** chunk (prefix 20,992, 512
tokens) the same arm reads `paged_attention` 0.0904 → 9.128 s = **101.0x** and
`linear_fp4` 2.4890 → 2.477 = **1.00x**, which is the 16,384 picture continued.
`_report` prints this artifact for any n that is not a multiple of 512; the fix
is to trend against the last full chunk, and it rides this PR.

**The share is a function of n, so no single share is quoted.** At 2,048 tokens
`linear_fp4` is 75.5% and attention 8.5%; at 16,384 attention is 51.1%; at 21,727
it is 59.0%. They cross at **prefix 7,168** and attention is still climbing at
the longest arm — no plateau in range.

## Why 30x, and what a tiled cell recovers

Attention over a prompt is genuinely O(n²), so the term cannot be removed — only
its constant. The constant is the kernel's shape.

`backend.py:933` routes every sm70 attention call to `paged_attention_split`,
whose grid is `T.Kernel(KVSPLIT, S*H, B)` and whose body opens
`Qf[d] = Q[bb, tt, hh, d]` — **one query row per thread block**. That is a query
tile of 1, and it is correct for what the kernel was built for: its docstring
says "S enters the GRID, so W verify positions run concurrently", where W is a
speculative verify width of at most 8 (`_MAX_VERIFY_W`). A prefill chunk is 512.

At tile 1 the K/V bytes are re-read once per query row. The two floors at 16,384
tokens:

| | tile 1 (today) | tile 64 |
|---|---|---|
| compute (causal QK<sup>T</sup>+PV, 15.7 TFLOP/s) | 3.36 s | 3.36 s |
| bandwidth (K/V per query tile, f32, 900 GB/s) | **20.16 s** | 0.31 s |
| **binds** | **bandwidth** | **compute** |

**The tile size is the binding term.** Today's kernel is bandwidth-bound at
20.16 s; a 64-query tile moves it into the compute-bound regime at 3.36 s. That
is the whole argument for a new cell, and it is why the gap is 30x rather than
2x.

The two arms carry a second, independent signature of the same thing. Between
16,384 and 21,727 the **compute** ratio climbs 29.9x → 31.4x while the
**bandwidth** ratio is nearly flat, 4.99x → 5.27x. A kernel whose cost tracks the
bandwidth term and not the compute term is what "bandwidth-bound at tile 1"
predicts, and that pair of ratios is stronger than either number alone. (The
peer's independent arithmetic gives 5.0x and 5.4x for the same two points.)

The f32 KV pool is part of the size of the win, not a caveat against it.
`backend.py:353` sets `io = float32` for sm70 and `engine.py:1503` routes it into
the pool, overriding the bf16 default — so each avoided re-read is 4 bytes, not
2. The tile removes the per-row re-read; the f32 pool makes each removed read
twice as expensive to have made.

Both floors above were computed twice, from separate code by two people, and
agree to **2%** (tile-1 bandwidth 20.16 / 35.18 s here against 34.4 s at the long
arm independently; ratios 4.99x / 5.27x against 5.0x / 5.4x). The difference is
the chunk loop — this version sums `ceil(c/tile)` tile-passes over each chunk's
own `prefix + c`, which is what the scheduler does, rather than over the whole
prompt at once. That agreement is the only independent check on the traffic model
itself, which is otherwise arithmetic no measurement touches.

Measured 100.5 s against a tiled floor of 3.36 s leaves **97.1 s recoverable,
49.2% of TTFT** — over three times the 15% threshold. At 21,727 the same
arithmetic gives 179.5 s, 57.1%.

Two independent confirmations that the floor is not an artifact of one method:
the profiler's own efficiency table reads 27.6x at prefix 0, 19.6x near 3,072 and
32.7x at 15,872, against 29.9x from the whole-arm arithmetic here; and the
last-full-chunk figure at 21,727 (9.128 s against a 0.2725 s floor) is **33.5x**,
the same quantity computed per chunk rather than per arm.

## The linears are already near peak

`linear_fp4` is the other 34–40%, and it has no schedule win. Per token the 27B
spends `mlp 3.423e10 + gdn_proj 1.107e10 + attn_proj 3.355e9 = 4.865e10` FLOP, so
a 512-token chunk is 2.491e13 FLOP — a **1.587 s** floor at fp32 against **2.47 s
measured, 1.56x**. (A first estimate of 1.4x counted only the full-attention
projections; the GDN `in_proj`/`out_proj` across all 48 GDN layers are 23% of the
per-token FLOP.)

On sm70 the linears run f32 with no tensor cores, so 1.56x off fp32 peak is close
to the ceiling of that path. Their only remaining rung is fp16 tensor cores
(`mma.sync.m8n8k4`, 125 TFLOP/s → a 0.199 s floor, 12.4x of headroom) at a
precision cost a parity gate would have to accept. **Named, not chased.**

## Caveats

**The instrument's syncs serialize what would otherwise overlap**, so the per-op
sum is an upper bound on an unsynced total and what this measures well is the
*split* between ops. `sync_secs` is not additive overhead — `_Timer.timed`
records `t2 - t0` spanning the call and the sync, while `sync_secs` is `t2 - t1`,
the trailing slice of that same window. The check: per-op sum 196.858 s against
an arm total of 197.589 s, ratio **0.9963**; additive, the arm would have needed
343.5 s. This does not touch the comparison that matters, since 75.98x and 0.99x
are two ops measured identically.

**The fit and the profile measure different things** and are not merged into one
number: `c1·n + c2·n²` is wall-clock TTFT through the route, the profile is a
device-inclusive per-op sum with the instrument's syncs inside it. They agree to
1.10–1.28x anyway, which is what makes the reconciliation an out-of-sample check
rather than a tautology.

Single card, single request, sm70, one prompt shape.

## Rule

A quadratic term being most of the wall clock does not by itself justify a
kernel — causal attention *is* quadratic. What justifies one is the measured
distance from the arithmetic floor, and the floor has to be computed for the
kernel you would write, not the one you have: the same work is 20.16 s bound at
one query row per block and 3.36 s bound at 64.

And a per-chunk trend is only readable where the chunks are the same size. At
n = 21,727 the last chunk is 223 tokens and the script's own trend line says
`linear_fp4` gets cheaper with prefix, which is false. Compare like-sized
chunks, or normalize per token.
