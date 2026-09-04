# B=8's ceiling is ctx=32 — the draft head's f32 readout OOMs at prefill width, V100 sm70, 2026-09-03

> Status: **measured, and the traceback names the allocation.** `B=8` depth 3 runs at
> **ctx=32** and **OOMs at ctx=512**, asking for **1.41 GiB with 12 MiB free of 31.74**. The
> allocation is `y2` in `linear_fp4` (`backend.py:496`), reached from
> `_draft_step → DraftHead.forward → trunk._linear(x, head)` — the **draft head's lm_head
> readout at prefill width, in f32**. So the throughput result stands but its scope does not:
> **B=8's 74.85 tok/s is a ctx=32 number**, and raising `serve`'s `--max-batch` to 8 is not
> available at any real context.

## Context

[B=8 fills the 32 rung and is 1.79× `serve`'s default](../wins/2026-09-03-b8-fills-the-32-rung-1.79x-over-serve-default.md),
measured at ctx=32 with a peak of 31110 of 32768 MiB — 1.6 GB spare. B=4 was already known to
OOM from ctx≥512. The flip was held back on exactly this question.

**Three previous guesses at the B=4 ceiling were wrong** — the KV pool (0.81 GB), the
split-kernel partials (13 MB), and "32 GB simply has no room" — and my own arithmetic
refuted the first two against the 1.50 GiB request. This time the run was left to fail with
its traceback intact instead of being reasoned about.

## Results

`bench_ctx_decode.py --depth 3 --batch 8 --tokens 32`, one process per `--max-ctx` so a
failure at 512 keeps the lower rows.

| `--max-ctx` | result |
|---:|---|
| 32 | **60.1 tok/s**, 16.7 ms/token, tok/forward 14.00 |
| 512 | **OOM** — 1.41 GiB wanted, 12.38 MiB free of 31.74 GiB |
| 1024 | not run — needs strictly more memory than 512 |

The ctx=32 row is **60.1**, not the A/B's 74.85, and the two are the same configuration. The
difference is not the kernel:

| | A/B (`--tokens 64`) | ceiling sweep (`--tokens 32`) |
|---|---:|---:|
| tok/s | 74.85 | 60.1 |
| ms/token | 13.3 | 16.7 |
| tok/forward | 17.52 | 14.00 |
| **tick (ms/token × tok/fwd)** | **233.0** | **233.8** |

**The ticks agree to 0.3%.** The entire 0.80× gap is `tok/forward` — 14.00 vs 17.52, or 1.75
vs 2.19 per request — so the kernel does identical work and fewer drafts are accepted.
Acceptance depends on the text, and a 32-token window sits earlier in each sequence than a
64-token one. Two different workloads, not a contradiction: **the sweep's tok/s must not be
compared with the A/B's.**

## What the traceback establishes

```
_run_forward (engine.py:744)  -> self._draft_step(rows)
_draft_step  (engine.py:905)  -> self._draft.forward(torch.cat(hs, dim=0), ...)
DraftHead.forward (spec.py:117) -> self.trunk._linear(backend, x, head)
linear_fp4 (backend.py:496)   -> y2 = torch.empty(M, N, dtype=torch.float32)
                                 OutOfMemoryError: 1.41 GiB
```

Three facts, none of which needs arithmetic:

1. **It is the draft, not the trunk.** `_draft_step` runs on **every** tick, including
   chunked-prefill ticks — `engine.py:744` says why: "or a chunked prefill leaves the draft KV
   empty".
2. **It is a prefill-width tick.** `y2` is allocated only when `len(chunks) > 1`, i.e. M > 32.
   A decode tick is 32 rows in one chunk and never allocates it.
3. **The readout is f32 and vocab-wide.** `head` is the trunk's lm_head, and the checkpoint's
   real vocab is **248320** (under `text_config`, not the top level).

Not established: the exact M. 1.41 GiB / 4 B / 248320 is **1524 rows**, which is not a round
multiple of the 512-token chunk cap, so the shape has two unknowns and one equation. I do not
need it for the verdict, and I twice tried to invert it anyway — first against an assumed
vocab of 151936 (wrong constant, wrong answer), then printing M=2491 and writing "M=2611
gives exactly 1.41" in the next line. Neither number was right.

## Verdict

**`serve --max-batch 8` is rejected for now** — not on throughput, which is 1.79×, but
because B=8 does not survive a real prompt. Three independent reasons now point the same way:

1. **Memory**: OOM at ctx=512, and `serve`'s own default context is far above 32.
2. **Startup**: graphs = `buckets ≤ max_batch × widths` = **16 at B=8 against 8 at B=2**
   (formula reproduces both measured precapture lines, 12 at B=4 and 16 at B=8), which is
   122-155 s a single-user endpoint pays for concurrency it will not use.
3. **Per-request latency**: 9.4 tok/s at B=8 against 20.9 at B=2.

What the number is good for: it prices the **rung**, and that is a kernel fact independent of
the batch that reaches it. A future B=8 needs the draft readout fixed first. **Not by casting
the output to f16** — I wrote that first and the code refutes it: `kernels_linear.py` declares
`Y = T.empty((M, N), "float32")` at every GEMV, and `backend.py:499` already carries the note
"kernel's Y is f32", so an f16 readout is a kernel change, not a call-site one.

**The real lever is that 191× of that buffer is never read.** `_draft_step` takes the vocab-wide
result and immediately reduces it to one row per request:

```python
logits = self._draft.forward(torch.cat(hs, dim=0), ids, pos, kv, ...)   # [M, 248320] f32
last   = torch.tensor([q - 1 for q in sq], device=dev)
tok, prob = self._backend.greedy(logits[rng, last].unsqueeze(1))         # 8 rows used
```

At B=8 that is **8 rows of 248320 = 7.6 MiB needed against 1444 MiB allocated**. Reducing the
hidden state to those rows *before* the lm_head makes the readout `[n, vocab]` and the buffer
disappears — `dh[-1][rng, last]` already applies exactly that index one step later.

Not settled here: `DraftHead.forward` produces logits and `hidden_out` in one pass, so the
reduction has to move inside it, and whether `last` is reachable there is a code question I
have not read. Recorded as the lever with its size, not as a plan.

## Rule

**Let the failing run keep its traceback.** Three guesses at the B=4 ceiling produced three
wrong causes and cost more than this one run, which named the file, the line, the call chain
and the size. A stack trace is a measurement; a plausible cause is not.

Second: **two harness settings make two workloads.** 60.1 and 74.85 are the same code at the
same context, and the 0.80× between them is entirely acceptance from a different `--tokens`.
Before calling a gap a regression, multiply out to the tick — if the ticks match, the kernel
is not what changed.

Third, still costing me: **check a model constant against the checkpoint, never from
memory.** The vocab is 248320; I used 151936 and derived a shape from it.

## Gate

One process per context so a failure preserves the lower rows. Batch spy asserts B requests
decoded together (`6fbe738`). GPU verified idle before launch and after each kill; the 1024
arm was killed by fd-verified pid (`/proc/<pid>/cwd` and `cmdline` read first), wrapper before
child, never by pattern — it needs strictly more memory than the arm that just failed.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **B=8 depth 3, ctx=512** | **OOM — 1.41 GiB wanted, 12 MiB free** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **the allocation** | **`y2` in `linear_fp4`, from `_draft_step`** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | B=8 depth 3, ctx=32, `--tokens 32` | 60.1 tok/s, tok/fwd 14.00 |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | same vs the A/B's `--tokens 64` | **ticks 233.8 vs 233.0 — agree to 0.3%** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | the 0.80× tok/s gap | **entirely acceptance: 14.00 vs 17.52** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | checkpoint vocab | **248320** (under `text_config`) |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | **`serve --max-batch 8`** | **rejected — memory, startup, per-request** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | draft readout rows used vs allocated | **8 rows (7.6 MiB) of 1444 MiB — 191×** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | next lever | reduce hidden to `last` BEFORE the lm_head |
