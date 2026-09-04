# The ~600s that justified an 1800s cap was never a 4K prefill — 2026-09-05

## Context

`5cdbf7e` (2026-08-31) raised the per-completion wall-clock cap in `server.py` from
600 s to 1800 s. Its whole justification, and the only record of the number
anywhere, is one line of commit body:

> 4K prefill on sm70 takes ~600s; the 600s server-side timeout fired before the
> decode request finished. Match the client-side 1800s.

I cited that figure twice today — in `936396c`'s commit body and in
`errors/2026-09-05-the-timeout-fix-landed-on-one-of-two-api-paths.md` — as the
reason `/v1/messages` needed the same 1800 s. The structural defect that commit
fixed is real and independent of the figure. The figure is not.

## What I measured

The live V100 service runs `550740a`, which still carries `messages.py`'s two 600 s
constants, so one request measures the cost and tests the cap at the same time.
`--max-batch 1`, `--max-ctx 4096`, 27B NVFP4 + draft head at depth 1. Client budget
set to 900 s deliberately, above the server's 600 s, so that whichever side gives up
first is identifiable — a client timeout would have said nothing about the server.

```
POST 15687 bytes, client budget 900s (server cap is 600s in the deployed 550740a)
COMPLETED in 39.1s  input_tokens=3478 output_tokens=8  (11.25 ms/input token)
```

**39.1 s.** The 600 s cap did not fire and could not have: adding a full 2048-token
output at the measured 50.3 tok/s decode rate gives 39.1 + 40.7 = **80 s**, still
7.5x under the cap. At this service's `--max-ctx 4096` the 600 s limit is
unreachable.

So the cited ~600 s is **15x** the real single-request cost.

## Root cause of the wrong figure

Two things, and the ordering is the interesting one.

**The number was measured at B=8, not B=1, and for the whole prefill tick rather
than one request.** `wins/2026-08-31-sm70-gdn-chunk-fused.md:60-65` is the only
prefill measurement from that day: `B=8`, 27B NVFP4, warm prefill **~65 s before,
15.1 s after**, with the breakdown labelled "tick 1, 8x64 tokens". A B=8 tick at
512 tokens each, extrapolated to 4096 tokens each, lands in the hundreds of seconds
— which is where ~600 s plausibly comes from. It is a *batch* figure being quoted
as a *request* figure.

**And the 4.3x fix landed one commit BEFORE the cap was raised.** `git log` order:
`1ecf8ee` (the 4.3x prefill win) → `2e1bf72` → `50076b4` → `bd17d55` → `5cdbf7e`
(the cap raise). So the cap was raised to accommodate a cost that the tree had
already made 4.3x cheaper four commits earlier. 4.3x of the 15x gap is exactly this.

Neither the commit nor any `docs/experience/` entry records a 4K single-request
timing; `grep` for "600 s" across `docs/` and `CHANGELOG.md` returns nothing. The
figure entered the tree as a commit-body assertion and was never checked.

## The instrument I nearly reported instead

My first attempt fitted a line through four shorter prompts and extrapolated:

```
input_tokens=  124  wall= 13.20s
input_tokens=  465  wall=  8.75s
input_tokens=  921  wall= 20.00s
input_tokens= 1603  wall= 17.78s
fit: wall = 11.04s + 5.007 ms/token   ->  4096 tokens: 31.5s
```

31.5 s is close to the 39.1 s that direct measurement later gave, so the conclusion
would have survived. The fit still had to be thrown away: **465 tokens came back
faster than 124**. A longer prompt finishing sooner means the dominant term is not
prefill — the 124-token point almost certainly absorbed a scheduling or
first-request cost — so the slope describes noise, and any conclusion resting on it
would have been right by luck. Direct measurement was one request away.

## What stands and what is withdrawn

**Stands:** `936396c` is correct and independent of this. The defect it fixed is
that one policy lived in two constants and only one was updated, so the same
request completed through `/v1/chat/completions` and raised `TimeoutError` through
`/v1/messages`. Whether the shared value should be 600 or 1800 is a separate
question from whether the two paths must agree.

**Withdrawn:** "a 4K prefill on sm70 takes ~600 s on its own" — as a fact about a
single request on the current tree. It is 39.1 s measured, 80 s worst-case with a
full output. The cap could be 300 s and 4K would still fit.

**Not claimed:** that ~600 s was wrong for the configuration it was taken in. A B=8
tick at 4096 tokens per row was never measured, and this entry does not measure it.
The error is quoting a batch-tick figure as a per-request one, not the figure itself.

**The `pending-remote` stub in
`errors/2026-09-05-the-timeout-fix-landed-on-one-of-two-api-paths.md` is now
resolved** — by a measurement that removes the reason the stub existed.

## Rule

A number that survives only in a commit body has never been checked. Before citing
one as the reason for a change, find the `docs/experience/` entry it came from; if
there is none, either measure it or label it as an unverified assertion from that
commit. And check what the figure's *units* were: ~600 s was a B=8 whole-tick cost
quoted as a single-request cost, which is the same class of error as summing a
ceiling with a trace row ([`bound-is-not-a-measurement`]).
