# A prompt's own publishes evict the prefix it shares — 2026-09-07

> Status: open. The mechanism is measured on the CPU target; the fix is not
> written. Listed in [`OPEN.md`](../OPEN.md).

## Context

`/health` on the V100 read **115 published, 11 resident, 105 evictions**, with the
block pool 91% free — so every eviction was state bytes, and the store turned over
its whole contents roughly ten times while serving a handful of requests.

That is not eviction pressure from *many* conversations. One prompt does it alone.
One conversation does it alone, from **two** publish sites. `_finish_prefills`
publishes at each *chunk* end (`engine.py:998` — a guard on `prefill_from %
BLOCK_TOKENS`, not a per-block loop), and `max_num_batched_tokens` defaults to 512,
so a 2729-token prompt publishes **6** times: ⌈2729/512⌉, which matches #241's
measured 6 exactly. The decode path publishes every 16 *generated* tokens
(`engine.py:1308`), so **64** more at gen 1024. Each of the 70 clones a constant
156,893,184 B snapshot, against a budget that holds **11**.

The volume is on the decode axis, not the prompt axis: at gen 8 there are **zero**
decode publishes, which is why a short-generation probe sees none of this. The live
server's 115 published against 3210 tokens_generated fits 3210/16 = 200 decode
boundaries and not its 4 prefill forwards.

Memcpy time is **not** the problem, and this is a bound rather than a measurement:
64 clones × 156.9 MB at the V100's ~900 GB/s is ≤ 11 ms spread over ~1024 decode
steps. The churn is the problem — 70 publishes into 11 slots turns the store over
six times per conversation.

## Root Cause

The entries a prompt publishes are ordered by length, and **LRU keeps the longest
few**. The shared prefix — a system prompt, a common instruction header — is at the
*start*, so it is the first thing evicted by the prompt's own tail.

Measured, budget for 3 entries, one conversation publishing 8 boundaries:

```
resident lengths after 8 publishes:  [96, 112, 128]
a new session sharing the 32-token system prompt:  MISS
```

**Control, same probe with budget for 99:** the new session hits at length 32, with
0 evictions. So the miss is caused by eviction, not by the entry never existing —
the shared-prefix entry is published correctly and then thrown away.

The intermediate entries are **not** useless, which is why this is not simply "stop
publishing them". For the same conversation they never serve: turn 2 is
prompt + reply + followup, strictly longer, so the longest entry answers (measured:
a turn-2 query hits at 96 of 96). But for a *different* session sharing only the
header, the boundary entry is the only thing that can serve — and that is the case
LRU discards first.

## Fix

Not written. The shape the measurement licenses: eviction must not treat a prefix
another session could share the same as one only this conversation can use. Either
score entries by shareability rather than recency, or stop publishing boundaries
inside the divergent tail while keeping the ones at a shared-prefix edge — which
needs a way to know an edge is shared, and nothing currently tracks that.

**Not quantified, and this is the operand the fix turns on:** what fraction of the
70 publishes sit on a prefix another session shares. Arithmetic over assumed
fractions gives 1.4% useful at 0% sharing and 50.0% at 50% — a range so wide it
decides nothing, and the sharing rate has never been measured here. It comes from
traffic, not from a card. A fix priced against an assumed rate would be priced
against nothing.

## Rule

An LRU over entries of *increasing* length evicts the shared prefix first, because
the shared part is the oldest part. Recency and shareability point in opposite
directions when one producer emits a nested family of keys.
