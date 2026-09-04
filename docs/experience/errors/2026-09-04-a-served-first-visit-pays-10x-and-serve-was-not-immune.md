# A served request pays 10.1x on its first visit to a prompt length, and I had documented serve as immune

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + 456M draft, `serve --depth 1 --max-batch 1 --max-ctx 4096`
**Task:** #73
**Instrument:** `scripts/probe_served_rate.py`

## The two-arm test

Same live server, request window read from the clock rather than inferred from the
order I ran commands:

| prompt | compiles in window | wall | end-to-end |
|---|---:|---:|---:|
| 13 tok | 0 | 1650 ms | 46.1 tok/s |
| 15 tok | 0 | 1811 ms | 42.0 tok/s |
| **37 tok, first visit** | **14** | **17320 ms** | **4.4 tok/s** |
| same 37 tok, repeat | 0 | 1714 ms | 44.3 tok/s |
| same 37 tok, again | 0 | 1687 ms | 45.0 tok/s |

**10.1x on the identical prompt**, purely first-visit JIT. The arithmetic closes:
`17320 − 1811 = 15509 ms` over 14 compiles is **1108 ms each**, and the log's own
begin/complete timestamps show 1-4 s per compile. 81 compiles across 40 minutes of
server uptime.

## What this refutes, both of them mine and both published today

**1. "Serve is immune."**
`errors/2026-09-04-the-recompiles-reproduce-and-are-not-a-shape-set.md` states:
"Bench-only, eager-path. The graph path forces the JIT to finish before capture
(`engine.py:223`) … so a served tick replays a graph whose compiles were paid once at
capture."

Measured: the live server compiles 81 times and a first-visit request pays 15.5 s of it
inline. The captured graph covers the **decode** shapes it captured; a request that
introduces a new shape still compiles on the eager path first. The claim was reasoning
from `engine.py:223` without measuring the path it describes.

**2. The per-call cache-key candidate.**
That entry left two mechanisms open — (a) the disk cache is not consulted for these two
kernels, (b) their cache key includes something that changes per call. **(b) is
refuted:** an identical repeat compiles zero times, so whatever varies is stable across
identical requests. (a) is also not it, since the second visit clearly hits a cache.

It is first-visit cost, which is the ordinary shape-specialization story — the same one I
declared dead this morning off a single B=8 log and then un-declared dead off a 6-group
B=1 run. Third data point, same conclusion as the second.

> **RESOLVED**: it was `spec.py` skipping two shape buckets the trunk has — the prefill
> width and the block-table width. Fixed in `029b27c` + `6c6f6df`; a first visit at a new
> prompt length now compiles **0** where it compiled 14, wall **17320 → 1104 ms (15.7x)**.
> See [`../wins/2026-09-04-the-draft-skipped-two-shape-buckets.md`](../wins/2026-09-04-the-draft-skipped-two-shape-buckets.md).
> The section below records the state before it was located, including the two candidates
> I eliminated and the two I had not thought of.

## What keys on 37 is NOT identified

Two candidates checked and eliminated:

- **Not the prefill bucket.** `_PREFILL_BUCKET = 64`, and 13/15/37 all round to 64. The
  prefill shape is identical in all three arms, so it cannot separate 0 compiles from 14.
- **Not block count.** `BLOCK_TOKENS = 16`; `13+76 = 89` grows 1 → 6 blocks and
  `37+76 = 113` grows 3 → 8. Both add 5. Identical growth, opposite outcomes.

The pair is always `write_tokens_f32` + `paged_attention_split` — the only two kernels
taking `seq_q_lens` — with `rope`/`rmsnorm_*`/`embedding_f16` joining on some visits. Which
argument varies is unmeasured, and I am not naming one this time.

## Cost, and the shape of the fix

A user's first request at any new prompt length pays 1-17 s. Every subsequent request at
that length is clean. So the cost is not a throughput ceiling; it is a latency spike on
cold shapes, and it is worst exactly when a person is trying the server for the first
time.

A warmup that pre-visits the reachable shape set would hide it entirely — which is the
same sizing question the 269-compile B=8 run left open, so the two are one task rather
than two.

## Three instruments failed before one worked

Recorded because each looked correct while being wrong, and two produced numbers I
briefly believed:

1. **wall_ms / tokens** charged prefill to decode. #24 measured prefill at 31 ms/prompt
   token — ~310 ms of a 1650 ms request — against a bench figure (35.56 ms/tick) that is
   a decode-only window by construction. That comparison read as a **15% serve
   regression that does not exist**, and I stated it before catching it.
2. **`curl -N` buffers the SSE body**, so time-to-first-chunk came back **0 ms**.
3. **Counting SSE frames measures the poll period, not ticks.** `server.py`'s stream loop
   sleeps 0.02 s between peeks; 76 tokens arriving in 39 frames is 1.95 tok/frame, which
   is close enough to the real 1.88 tok/forward to look like a tick count. It gave
   **22.6 tok/s** for a server doing 46.

The working instrument brackets `/health`'s cumulative `decode_forwards` around one HTTP
request, so tok/forward and acceptance come from the engine's own counters rather than
the clock, and `prefill_forwards == 1` **proves** the prefill sits outside rather than
being assumed to.

I also asserted the 13:20-13:23 compile block "was not in my request windows" from
command ordering, and separately that "nobody else is hitting the server" from a request
count that came to 11 against `finished 12`. Neither was read from the log. The fix was
to stamp `date +%H:%M:%S` on both sides of the request and `awk` the window — after which
the result reproduced on demand.

## Rule

A claim that one code path is immune needs that path measured, not the mechanism that
should make it immune read off the source. `engine.py:223` does force the JIT before
capture, and the conclusion drawn from it was still wrong, because it covers the shapes
capture visited and a request can introduce one it did not.

And a request's window is a timestamp, not the order you typed the commands in. Both
wrong attributions here came from reconstructing when something ran instead of reading
when it ran.
