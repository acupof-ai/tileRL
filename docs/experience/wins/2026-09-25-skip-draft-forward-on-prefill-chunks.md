# Skip the draft forward on prefill chunks the decode window never reads — V100 sparse d1 W2048, 2026-09-25

> Status: Device-verified, CPU-gated; pending review/merge (awb impl#D3).

## Context

Every tick containing a chunked-prefill row ran one draft-head forward for
that row (`Engine._run_forward → _draft_step(rows)`, rows unfiltered). A
prefill-row draft forward is wide (one query per chunk token, up to the 512
prefill bucket) and perf1 measured 6.5-10 s stalls where a prefill row shared
a tick with decoding rows. The forward's only product for an interior chunk is
a back-fill of the draft's OWN dense KV pool: position q consumes the trunk
hidden at q-1, the engine keeps only the last forward's hidden (plus one
previous position), so `DraftHead.step` clamps its span to the newest hidden
and fills that chunk's slice of draft K/V. By the time the row decodes, the
draft attention reads only its trailing W=2048 window; K/V written well before
that window is never read.

## What Worked

Skip the draft forward for an interior prefill chunk whose whole write span
precedes the decode read window. In `Engine._draft_prefill_skip_ids`:

- a chunk ending at `s = prefill_from + c` is skippable when
  `s <= floor((n-W)/BLOCK_TOKENS)*BLOCK_TOKENS`, the first page the windowed
  attention actually reads;
- the finishing chunk (`s == n`) is never skipped — it leaves the first chain
  in `r.drafts` for the first decode tick;
- W=0 (full prefix, the head default when the window flag is off) skips
  nothing, so non-window serving is byte-for-byte unchanged;
- the skip set is snapshotted before `_finish_prefills`, which flips the
  finishing rows to DECODE.

At n=5200 / chunk 512 / W 2048 the window first page is 3152, so the chunks
ending 512..3072 (six forwards) are skipped and the four ending
3584..5120 are kept, leaving the window's draft K/V fully populated before
decode.

Device gate, sparse-k128 d1 W2048 ThinkingCap-orig on V100, same fixed 5200
prompt, two process arms differing only by the skip (OLD patches the helper to
return the empty set), run in BOTH orders to control for first-use TileLang
compilation:

| arm (warm) | prefill draft forwards | prefill draft ms | TTFT ms | accepted/drafted | output |
|---|---|---|---|---|---|
| OLD | 11 | 1099 | 18417 | 63/64 | identical |
| NEW | 5 | 736 | 18226 | 63/64 | identical |

Warm steady state: accept rate and every emitted token identical; six prefill
draft forwards removed per cold 5200-token request; ~0.36 s of draft kernel
and ~0.2 s solo TTFT saved. A mixed prefill+decode round (three 300-token
requests decoding alongside the 5200 prompt), fully warm, gave mixed-tick
draft time 1567 ms (NEW) vs 2101 ms (OLD), 110/128 accepted on both, same
long-request tokens.

The cold runs exposed the mechanism behind perf1's 6.5-10 s stalls without
giving a clean cold delta: the very first OLD-first TTFT was 135.2 s while the
NEW-second arm was 18.3 s, but that pair is order-confounded — first-use
TileLang compilation of a wide prefill-draft kernel lands on whichever arm
first touches that (batch, width) shape, which is why the two-order run was
required. Warm, the same-shape forwards are 100-270 ms, not seconds. The
defensible cold claim is the mechanism only: six chunks are skipped, so the
wide draft kernel shapes those chunks uniquely need are never compiled or
launched on a cold server. The steady production benefit is six fewer serial
wide-kernel launches per cold ~5200-token request (~0.36 s of warm draft
kernel, ~0.53 s in a prefill+decode mix), with identical output and accept
rate, not a decode-throughput change.

## Rule

A per-chunk auxiliary forward that only back-fills its own KV is skippable on
every chunk whose product precedes the window later reads; key the cutoff on
the window's first PAGE (paged attention rounds the read down), keep the
finishing chunk (it carries the next-tick chain), and make W=0 a no-op so the
full-prefix default is unchanged. Always gate with the arms in BOTH orders: a
first-use compile lands on whichever arm first touches a kernel shape and a
one-order comparison attributes minutes of compile to the wrong arm.

## Gates

- `tests/test_draft_skip_prefill.py` (CPU, ~2.6 s): W=0 skips nothing; the
  page-aligned cutoff picks exactly the pre-window chunks; the finishing chunk
  and sub-window prompts never skip; an end-to-end NEW-vs-OLD drive emits
  identical greedy tokens with exactly two empty prefill draft ticks NEW and
  none OLD. Negative control: OLD code (helper returns the empty set) fails the
  empty-tick-count assertion, verified red.
- Full CPU suite green (the one failure class was a test `_OracleDraft` double
  without `attn_window_tokens`; read it via `getattr(..., 0)`, real heads carry
  the attribute).
- Device numbers above from `scripts/probe_draft_prefill_skip.py` (solo,
  per-arm forwards/wall/TTFT/accept) and `scripts/probe_draft_prefill_mixed.py`
  (prefill+decode tick breakdown).
