# Incremental SSE was a starved poll loop, not the engine lock, 2026-09-03

> Status: **SHIPPED.** The previous attempt (`errors/2026-09-03-sse-streaming-is-blocked-by-the-engine-lock.md`)
> concluded that incremental streaming needs `step()`'s lock narrowed — a correctness change to
> the engine core. That conclusion was wrong. The lock was never touched. A **lock-free**
> `engine.peek()` plus a poll loop that does not call `take()` gives **5-6 content deltas** on the
> V100 at **32.4 tok/s streamed vs 32.4 non-streamed** — no throughput cost, Chinese never split.

## What the last entry got wrong

It measured `peek()` returning `None` on 100% of calls under the daemon loop and blamed
`step()` holding `_lock` across the forward. The lock measurement was real (one `take()` blocked
325 ms of a 335 ms generation). The **attribution** was wrong on two counts:

1. **`peek` does not need the lock at all.** Under the GIL, the writer's `req.output.append(tok)`
   (`engine.py:1082`) and a reader's `list(req.output)` are each a single bytecode. The copy is
   always a consistent prefix — a stale one is fine, and `take()` stays the authority on the
   final sequence. The reverted version took `_lock`, which is what made it unobservable.
2. **The poll loop called `take()` every iteration.** `take()` *does* take the lock. So even a
   lock-free `peek` was queued behind a call that blocks for a whole forward. The loop got
   **2 polls** for a 24-token generation.

Instrumented rather than reasoned about — `scripts/probe_sse_deltas.py` counts `peek` calls:

| version | deltas | peek calls | peek lens seen |
|---|---:|---:|---|
| `peek` lock-free, `take()` each poll | 1 | **2** | `[0]` |
| `peek` lock-free, `take()` only after `peek` → None | **3** | **15** | `[0]×9, [1]×4, [13]` |

Same `peek`. The only change is where `take()` sits. **The obstacle was in my own poll loop,
six lines from where I was looking, for the second time in this task.**

## The shipped shape

`engine.peek(request_id)` scans `(*self._waiting, *self._running)` and returns `list(req.output)`,
or `None` once the request has left both queues — which by construction means `_finish` has
already filed it under `_finished`/`_failed`, so `take()` will answer. That is what lets the
stream poll without touching the lock until the run is over: **poll `peek`, and call `take()`
exactly once, after `peek` reports None.**

Two UTF-8 rules, both load-bearing:

- Decode the **whole prefix** each poll, never the new ids. One token can be a partial UTF-8
  sequence, so a per-token decode splits multi-byte characters.
- A prefix can still **end** mid-character, which a decoder renders as U+FFFD. Cut at the first
  replacement char and hold the rest for the next poll.

## Measured on the V100

Live server, `Qwen3.8-27B` NVFP4, sm70, `--depth 3 --max-batch 2`, 128 tokens, temperature 0,
prompt `用三句话讲讲张量并行` (ctx ≈ 15). Three runs each, after warmup:

| path | run 1 | run 2 | run 3 | deltas |
|---|---:|---:|---:|---:|
| non-stream | 2.9 (warmup) | **32.3** | **32.5** | — |
| stream | **32.2** | **32.4** | **32.4** | **6** |

**Streaming costs nothing measurable** — 32.4 vs 32.4, well inside the 1.16% noise floor. The
rate agrees with the recorded ctx=32 single-stream number (32.7 tok/s,
`wins/2026-09-03-single-stream-b1-baseline.md`) to 0.9%, so the instrument is not lying about
either the throughput or the streaming.

Chinese survives intact — 5 deltas, zero U+FFFD mid-stream:

```
0 '<think>'
1 '\n用户要求用三句话解释'
2 '张量并行（Tensor Parallelism）。我需要简洁、准确地'
3 '用三句话概括张量并行的核心概念。\n\n张量并行的关键点：\n1. 核心思想：将单个权重矩阵（如线性层的'
```

## The ceiling, stated plainly

**6 deltas for 128 tokens is one delta per ~21 tokens.** A 20 ms poll over a 3.95 s generation
has ~197 chances and takes 6. The remaining serialization is the lock on the short paths
(`_finish`, `submit`, `stats`) plus the GIL against a thread doing GPU work — narrowing
`step()`'s lock is still the lever if per-token streaming is ever wanted. It is not needed for a
viewer to see text arrive, which is what this bought.

`# ponytail: one delta per ~21 tokens, narrow step()'s lock for per-token`

## Rule

**When an accessor cannot see shared state, suspect the reader before the writer.** The last
attempt correctly measured that a reader was blocked and incorrectly concluded the writer's lock
had to change — a correctness change to the engine core, with the tape and KV pools downstream.
The actual fix touched no locking semantics: it removed a lock acquisition from the reader and
moved one call out of a loop. **The cheap side of a contention problem is the side you control.**

Second: **the test that was xfailed as "needs the lock narrowed" was passing one edit later.**
`strict=True` did its job — it forced the marker to be re-examined instead of aging into
scenery. But the *reason* text asserted a cause I had not exhausted, and a wrong cause in an
xfail reason tells the next reader the problem is bigger than it is.

## What the CPU gate cannot prove

`tests/test_server.py` asserts >1 delta, joined == non-streamed, and no U+FFFD **mid-stream**.
The final delta is exempt: `tiny()` has random weights, so its bytes are not valid UTF-8 at all
and the non-stream path returns replacement chars too — asserting no U+FFFD anywhere would
contradict `joined == expected`. So the CPU gate proves the split points are chosen correctly;
only the V100 run above proves real multi-byte text survives.

## Gate

193 passed, 4 skipped, ruff clean. The V100 numbers are from the live server after a
by-pid restart (fd-verified, pid 1342942, 25 nvidia fds, cmdline checked) with the pod tree
verified byte-identical to `HEAD` before the swap.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | Mac | cpu | tiny | deltas, `take()` inside the poll loop | **1** (2 polls) |
| 2026-09-03 | (this) | Mac | cpu | tiny | deltas, `take()` after `peek` → None | **3** (15 polls) |
| 2026-09-03 | (this) | V100 | cuda | qwen38-27b | non-stream, 128 tok, ctx 15 | **32.3 / 32.5 tok/s** |
| 2026-09-03 | (this) | V100 | cuda | qwen38-27b | **stream, same request** | **32.2 / 32.4 / 32.4 tok/s** |
| 2026-09-03 | (this) | V100 | cuda | qwen38-27b | streaming overhead | **none measurable (0.0-0.3%)** |
| 2026-09-03 | (this) | V100 | cuda | qwen38-27b | content deltas per 128 tokens | **6** (one per ~21 tok) |
| 2026-09-03 | (this) | V100 | cuda | qwen38-27b | U+FFFD mid-stream, Chinese prompt | **none** |
| 2026-09-03 | (this) | V100 | cuda | qwen38-27b | agreement with recorded ctx=32 B=1 | **0.9%** (32.4 vs 32.7) |
