# expandable_segments is load-bearing, and three buffers now OOM at the same padded shape, V100 sm70, 2026-09-03

> Status: **findings, no runtime change.** Three separate results. (1) The `88.5 tok/s at
> B=8 ctx=512` number needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the tree
> sets **nowhere** — without it the same build still OOMs, so that entry's claim is scoped to
> "with the flag". (2) **ctx=1024 does not run**; the B=8 depth-3 ceiling sits between 512 and
> 1024. (3) Three different allocations have now failed at **exactly 7 × 512 rows** — the
> attention partials, the fused `gate_up` output, and `silu_mul`'s contiguous copy — because
> `engine.py:729` gives every row in a tick one bucketed width. The padding, not any one
> buffer's dtype, is the mechanism.
>
> Cost of getting here: **two instrument faults in my own guards**, both firing on healthy runs.

## Why this was worth an arm

The ctx=512 win was measured with a flag my pod runner exported. `grep` over `src/`,
`packages/` and `scripts/*.py` finds `expandable_segments` **nowhere**, so `tilerl serve`
cannot reach the configuration the entry describes. That is a claim about the shipped product
resting on a variable outside it, so it gets measured rather than assumed.

## Arm 1: without the flag, ctx=512 still fails

```
torch.OutOfMemoryError: Tried to allocate 476.00 MiB.
  31.74 GiB total, 214.38 MiB free, 1.22 GiB reserved but unallocated
  model.py:335 _mlp -> model.py:162 _linear -> model.py:174 _base_linear
  -> backend.py:500 linear_fp4
      y2 = torch.empty(M, N, dtype=torch.float32) if len(chunks) > 1 else None
```

**The flag is load-bearing**: with it, this config runs at 88.5 tok/s; without it, it OOMs.
`expandable_segments` was separately measured to take free memory from 396 MiB to 1.11 GiB,
and the reserved-but-unallocated figure here (**1.22 GiB**) is the memory the flag would have
made available.

## 476 MiB is a shape, and it names the same padding

`476 MiB / 4 B = 124,780,544` elements. Line 335 is `gu = self._linear(backend, h, gu_key)` —
the **fused** gate_up, so `N = 2 × intermediate_size = 2 × 17408 = 34816`:

```
124,780,544 / 34,816 = 3584 exactly  =  7 rows x 512
```

**The same 7 rows × 512 the attention partials died on.** The tick's shape never changed; only
which buffer overflows first did. And the buffer is padding-driven twice over:

| shape | `y2` (f32) | useful rows only |
|---|---:|---:|
| 4 × 512 | 272 MiB | 0.53 MiB |
| **7 × 512** | **476 MiB** | 0.93 MiB |
| 8 × 512 | 544 MiB | 1.06 MiB |

A decode row carried in a 512-wide tick pays **512×** its own footprint here, and `y2` exists
*only* when `len(chunks) > 1` — i.e. only because the tick is wide.

For scale against the buffer just fixed: PO at 8×512 with the per-width split count is
768 MiB, and `y2` is 544 MiB in f32. They are the same order, which is why halving one moved
the wall rather than removing it.

## Arm 2 measured the harness, not the ceiling

The ctx=1024 probe never reached 1024. It printed its ctx=32 row (61.2 tok/s) and then
**stalled at its own ctx=512 step**: `ctx=512: prefill stalled 300 s with 8/8 admitted` — the
same context that reads **88.5 tok/s** when 512 tops the sweep. Two rows both labelled
"ctx=512", opposite outcomes, and per the standing rule that is a contradiction to suspect
rather than report.

The cause is in the harness, at `bench_ctx_decode.py:220`: `num_blocks` comes from
`max(ctxs)`, so **raising `--max-ctx` grows the KV pool and then re-runs every shorter
context inside a smaller free-memory budget**:

| invocation | blocks | pool | sweeps |
|---|---:|---:|---|
| `--max-ctx 512` | 312 | 287 MB | 32, 128, **512** |
| `--max-ctx 1024` | 568 | **523 MB** | 32, 128, **512**, 1024 |

**236 MB apart**, against a free figure of 1.11 GiB — 21% of the headroom the ctx=512 result
depends on. So arm 2's ctx=512 step was never the configuration arm 1 measured, and its stall
says nothing about 1024.

Fixed with `--min-ctx` (`85f76d7`), so a ceiling probe can be pinned to one context:
`--min-ctx 512 --max-ctx 512` reproduces 312 blocks exactly, verified locally by asserting
that a pinned single context sizes the pool identically to the sweep that ends there. The run
now also **prints the pool it sized and the sweep it will do** — that was the variable
differing silently between two runs.

### And the isolated probe hit a second instrument fault

Pinned to ctx=1024 alone, it stalled again — `prefill stalled 300 s with 8/8 admitted`. The
message was a **timeout, not a hang**, and against a recorded number the budget was simply too
small. Prefill costs **~31 ms per prompt token** on this card, so:

| config | prompt tokens | legitimate prefill | flat budget | used |
|---|---:|---:|---:|---:|
| B=8 ctx=512 | 4096 | 127 s | 300 s | **42%** |
| **B=8 ctx=1024** | 8192 | **254 s** | 300 s | **85%** |

`8/8 admitted` says every request was accepted and progressing. At 85% of the budget any
variance fires the guard, which is exactly why ctx=512 passed and 1024 did not: **the deadline
was sized for short contexts and became a measurement of itself.** It also compounds fault #1
— arm 2's ctx=512 step paid its 127 s *and* ran with 236 MB less free memory.

Budget is now `max(300, 4·batch·ctx·0.031)` (`3c9c704`): the floor keeps every short config
byte-identical (B=8 ctx=32 stays at 300 s, 37.8× its estimate) and long ones get a uniform 4×
headroom — 1016 s for ctx=1024. Worst case becomes budget + 600 window + 600 drain = **2216 s**,
above the 1700 s outer `timeout` I had been passing, so the outer kill would have preempted the
guard and hidden which arm failed. Raised to 2400 s. Third attempt running.

**Two instrument faults, zero measurements of the thing I set out to measure.** Both were in
guards I wrote earlier to catch exactly this class of problem, and both fired on healthy runs
rather than broken ones.

### Third attempt: prefill completed, and ctx=1024 OOMs in a third buffer

With the budget scaled, prefill finished — so the fix was correct and the earlier stall was
never a memory event. ctx=1024 then failed in `silu_mul`:

```
model.py:341 _mlp -> backend.py:1022 silu_mul
    up = self._c(self._f32(up).reshape(-1))
torch.OutOfMemoryError: Tried to allocate 238.00 MiB, 172.38 MiB free
```

`238 MiB / 4 B = 62,390,272 = 3584 × 17408` — **3584 = 7 × 512 for the third time.**
`max_num_batched_tokens` caps the chunk at 512 even at ctx=1024, so the padded shape is
identical; only the buffer that overflows first keeps changing.

**The cause is a layout, not a dtype.** `_mlp` reads the fused projection and slices it:

```python
gu = self._linear(backend, h, gu_key)          # [M, 2*I] contiguous
gate = autograd.slice(gu, ..., slice(0, I))    # row stride 2*I -> NOT contiguous
up   = autograd.slice(gu, ..., slice(I, None)) # NOT contiguous
```

Verified locally: `gu.is_contiguous()` is True, both slices are False. So `_c()` in `silu_mul`
copies **each half**, and `_f32` is a no-op because `y2` is already f32 — the allocation is
purely the strided layout. One MLP layer's transient peak:

| at M = 7×512 | f32 |
|---|---:|
| `y2` (fused output) | 476 MiB |
| `gate.contiguous()` | 238 MiB |
| `up.contiguous()` | 238 MiB |
| **peak, one layer** | **952 MiB** |

For scale, PO at 8×512 after the per-width split count is 768 MiB. **The MLP's padding
transient is now larger than the attention buffer two entries were spent halving.**

**Verdict: ctx=1024 does not run at B=8 depth 3. The ceiling sits between 512 and 1024**, and
the next lever is not a fourth dtype cast — it is either the layout (have the fused GEMV write
`[2, M, I]` so each half is contiguous, a load-time weight permutation with no runtime cost) or
the padding itself. Neither is measured yet; both are named with their costs rather than
guessed at.

**Does not decide:** whether to ship the flag. `expandable_segments` changes allocator
behaviour globally and its cost on this workload is unmeasured here — the earlier arm that
introduced it measured only free memory, not throughput. One env default is a small diff and a
large blast radius; it needs its own A/B.

**Does decide** the next lever's shape, and it is not another dtype cast. Three buffers have now
OOMed at **exactly 7 × 512 rows**: the attention partials, the fused `gate_up` output, and
`silu_mul`'s contiguous copy. All three are the same mechanism — **`engine.py:729` gives every
row in a tick one bucketed width**, so a decode row in a mixed tick is inflated to the prefill
chunk's width and every per-row buffer downstream scales with `rows × width`. Halving one
buffer's dtype moved the wall to the next buffer, twice.

Two levers, both named with costs rather than guessed at, neither measured:

1. **The layout.** `gate`/`up` are strided slices of a `[M, 2I]` fused output, so `silu_mul`
   copies both halves — 476 MiB of copies at 7×512 on top of the 476 MiB output. Having the
   fused GEMV write `[2, M, I]` makes each half contiguous and removes both copies; the weight
   permutation is load-time, so the runtime cost is zero. Confined to the MLP.
2. **The padding.** Stop inflating decode rows — either don't mix decode and prefill in one
   tick, or bound `rows × width` the way `max_num_batched_tokens` bounds summed chunk length
   (verified at `engine.py:563` and `:580`: it does **not** bound the product). This removes the
   class rather than one instance, and it costs TTFT.

## Rule

**An env var that changes a result is part of the result.** The 88.5 was published with the
flag in the pod script and no mention of it in the entry, which would have read as a property
of the build. One `grep` showed the tree never sets it; one arm showed the number does not
survive without it. When a measurement's harness sets an environment variable, either the tree
sets it too or the entry states it as a precondition.

Second: **a sweep's own range is a variable.** `--max-ctx` looked like a filter — "skip
contexts above this" — and it also sizes the KV pool, so widening the range silently changed
the conditions for every point already in it. A knob that both selects what is measured and
alters how it is measured will produce two different numbers under the same label, and the
label is what gets compared later. The parameter that sizes the run should be printed by the
run.

Third: **a guard's threshold has to scale with whatever the guard is watching.** Both instrument
faults here were in bounds I wrote earlier to catch stalls, and both fired on healthy runs: a
pool sized from the sweep's maximum, and a flat 300 s deadline over work that grows with
`batch × ctx`. A constant timeout is a statement that the work is constant. When it isn't, the
guard eventually measures itself — and reports it in the vocabulary of a real failure
("stalled", "8/8 admitted"), which is what makes it expensive: two runs, ~40 minutes of pod
time, and a conclusion about a memory ceiling that was never tested.

Fourth: **when the same number keeps appearing, stop fixing buffers.** 7 × 512 has now been the
failing shape three times in three different files. I halved a dtype, then chose a split count
per width, and each time the wall moved to the next allocation of the same shape — because none
of it touched the thing producing the shape. Three data points at an identical value is the
signal to go up a level, and I took two of them as separate bugs.

## Gate

Every failing allocation read from its traceback rather than inferred — which mattered three
times: my prediction for arm 1 named PO and the traceback named the MLP; my prediction for
ctx=1024 named PO or `y2` and the traceback named `silu_mul`. The non-contiguity of the fused
halves was verified locally with real shapes, not assumed from the slice syntax. GPU verified
idle before each of the three launches. No runtime code changed.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 51bea44 | V100 | cuda sm70 | qwen38-27b | **ctx=512 without the allocator flag** | **OOM — the flag is load-bearing** |
| 2026-09-03 | 51bea44 | V100 | cuda sm70 | qwen38-27b | the failing allocation | **476 MiB, free 214 MiB, short 262** |
| 2026-09-03 | 51bea44 | V100 | cuda sm70 | qwen38-27b | where | **`backend.py:500` fused gate_up `y2`, f32 — not attention** |
| 2026-09-03 | 51bea44 | V100 | cuda sm70 | qwen38-27b | 476 MiB decoded | **M=3584 = 7×512, N=34816 — the same padded shape** |
| 2026-09-03 | 51bea44 | V100 | cuda sm70 | qwen38-27b | reserved but unallocated | 1.22 GiB — what the flag would free |
| 2026-09-03 | 51bea44 | V100 | cuda sm70 | qwen38-27b | `y2` at 8×512 vs its useful rows | **544 MiB vs 1.06 MiB — 512× inflation** |
| 2026-09-03 | 51bea44 | V100 | cuda sm70 | qwen38-27b | PO at 8×512 (just fixed) for scale | 768 MiB — same order as `y2` |
| 2026-09-03 | 85f76d7 | V100 | cuda sm70 | qwen38-27b | **`--max-ctx 1024` vs 512: KV pool** | **568 blocks (523 MB) vs 312 (287) — 236 MB apart** |
| 2026-09-03 | 85f76d7 | V100 | cuda sm70 | qwen38-27b | its ctx=512 step | **stalled — same context reads 88.5 when 512 tops the sweep** |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=1024 legitimate prefill** | **254 s (8192 tok × 31 ms) against a flat 300 s budget — 85%** |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | B=8 ctx=512 for comparison | 127 s of 300 s — 42%, which is why it passed |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | scaled budget at ctx=1024 | 1016 s; short configs keep the 300 s floor |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | **ctx=1024 prefill with the scaled budget** | **completed — the stall was never a memory event** |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=1024** | **OOM — 238 MiB at `silu_mul`, 172 MiB free** |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | 238 MiB decoded | **3584 × 17408 f32; 3584 = 7×512 — the SAME shape, third buffer** |
| 2026-09-03 | 3c9c704 | Mac | — | — | are the fused-projection halves contiguous? | **no — `gu` yes, both slices no; `_c()` copies each** |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | **one MLP layer's transient peak at 7×512** | **952 MiB (476 y2 + 238 + 238)** |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | vs PO at 8×512 after the split-count fix | 768 MiB — **the MLP transient is now the larger one** |
| 2026-09-03 | 3c9c704 | V100 | cuda sm70 | qwen38-27b | **verdict: B=8 depth 3 context ceiling** | **between 512 and 1024** |
