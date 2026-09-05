# A cache benchmark measured its own earlier self — sm70, 2026-09-05

> Status: withdrawn

## Context

Measuring the prefix cache's win on the V100 after the ragged-publish fix. The
first script served one target prompt cold, warmed the store with the
conversation head, then re-served the target — all in one server process — and
reported:

```
cold           prompt=  3265  wall=   20.63s  hits=0  published=7
warm-up(head)  prompt=  1092  wall=    1.14s  hits=1  published=0
warm           prompt=  3265  wall=    2.00s  hits=1  published=0
speedup 10.312
```

## Root Cause

The cold arm publishes entries covering its **whole** prompt — `published=7` says
so in the same output. The warm arm then matched one of those, not the head, so
it measured re-sending an identical prompt: a full-length cache hit.

The workload's shareable span is 1092 of 3265 tokens, **33%**. Reuse cannot save
time on work the arms do not share, so the ceiling was ~1.5x. A 10.3x reading was
7x above what the experiment could produce.

The tell was in the output the whole time and went unread: `warm-up(head)` at
1.14 s for 1092 tokens is 1.04 ms/token against a cold 6.32, so the head itself
was already being served from cache — the store was contaminated before the warm
arm started.

## Fix

`--arm cold|warm`, one arm per server restart, and an assert at entry:

```python
assert st["prefix_published"] == 0, (
    f"the store already holds {st['prefix_published']} publishes; restart the server "
    f"before this arm or it measures a contaminated store"
)
```

Rerun: cold 20.63 s (6.32 ms/tok), warm 14.19 s (4.35 ms/tok), **1.45x**. The
consistency check confirms it: 6.44 s saved is 31% of 20.63 against a 33%
shareable fraction, so the implied cost on the shared span is 5.90 ms/token
against 6.32 overall.

## Rule

Compute the shareable fraction before running a cache benchmark and treat it as a
hard ceiling on the ratio — a speedup above it is not a better result, it is a
different experiment. Assert the cache is empty at arm entry so contamination
fails loudly, and give each arm its own process when the cache lives in the
server. 10.3x also felt like success, which is why it survived: an error that
inflates the number you want produces no surprise.
