# A ragged prompt length published nothing — sm70, 2026-09-05

> Status: Shipped

## Context

The live V100 server had served two 6-turn chats and reported `prefix_hits: 0`,
`prefix_published: 4` — every one of the four from decode, none from prefill.
Two theories were wrong before the measurement: that the store matched only
whole entries (it does not), and that the `<think>` tail broke sharing (104
usable aligned cuts exist in every thinking mode).

The mechanism is one line. `_finish_prefills` published only when the WHOLE
prompt length was block-aligned:

```python
if pf.phase != _PHASE_DONE and prompt_len % BLOCK_TOKENS == 0:
```

15 of every 16 prompt lengths are not. The gate is not arbitrary — the state
pool is exact only at a chunk end, and publishing a truncated entry pairs KV
for N tokens with a state that absorbed more.

## What Worked

Cut the first prefill chunk at a `_PREFILL_BUCKET` (64) boundary, so the chunk
end IS a publish point where the state pool is exact. The tail becomes a second
forward — a launch, not extra tokens. `_PREFILL_BUCKET == _WY_CHUNK == 64`, so
`-(-chunk // 64) * 64` is the identity on the shortened chunk and the width
bucket does not move.

Two arms of the same 4-turn chat on one V100 process, `--grow 40`:

| turn | prompt | wall_s | ms/tok | hits |
|---:|---:|---:|---:|---:|
| 0 | 1092 | 6.29 | 5.76 | 0 |
| 1 | 3280 | 14.26 | 4.35 | 1 |
| 2 | 6548 | 28.11 | 4.29 | 1 |
| 3 | 10896 | 55.51 | 5.09 | 1 |

Cumulative over two runs: **13 hits / 3 misses** where the same server
previously reported 0 / 2. One hit per prompt is correct — a prompt is looked up
once; the several publishes per turn are the chunk boundaries the fix created.

`published` per turn rises with length (2, 5, 6, 10) because a long prompt
crosses several chunk boundaries and each is now a publish point.

### The controlled arm

The turn-over-turn table changes the prompt and the store together. One prompt,
one arm per server restart, `prefix_published == 0` asserted at entry:

| arm | prompt | wall_s | ms/tok | hits |
|---|---:|---:|---:|---:|
| cold (empty store) | 3265 | 20.63 | 6.32 | 0 |
| warm (head served first) | 3265 | 14.19 | 4.35 | 1 |

**1.45x on a prompt that is 33% shareable.** The consistency check is what makes
it credible: 6.44 s saved of 20.63 is 31%, against a 33% shareable fraction, so
the implied cost on the shared span is 5.90 ms/token against 6.32 overall — the
reuse removes almost exactly the shared span and nothing more. A ratio above the
shareable fraction would mean the arms were contaminated.

## What was NOT the answer, and cost the most time

**Blockwise storage at 64 granularity.** `lookup` already returns the longest
prefix at any length: probed with a 100-token query against entries stored at
16..128, it returns `length=96`. Matching never required the whole entry to be a
prefix, and the granularity was already 16 — four times finer than the 64 that
was specified. No code was needed.

**Per-boundary state snapshots.** A GDN snapshot is 144 MiB at 27B (48 layers ×
48 heads × 128² × f32). An 11K prompt has 172 boundaries at stride 64 = 24.2 GiB,
against 1.43 GiB of KV for the same span — **16.9x, independent of length** — on
a card with 5.7 GiB free after weights. Keeping one boundary per prompt is what
fits; keeping all of them is not a tuning question.

**Exporting boundary states from the kernel.** Written and verified
(`reference.gdn_chunk_core` overwrites `s` at every chunk end, so a snapshot is
one clone; four boundaries checked against fresh prefix runs, `max|delta|`
0.000e+00) — then **reverted**. Splitting the chunk reaches the same 64-aligned
publish point with no kernel, backend, or 48-layer shadow-pool change.

## The gate

`test_a_ragged_prompt_publishes_and_its_state_matches_no_store` asserts the
restored GDN state is `allclose` to a `NoPrefixStore` engine's, not just that
the output matches. Output equality is not evidence: an earlier attempt at this
was byte-identical while the snapshot's norm was 74.0 against a correct 39.75 —
argmax over a vocabulary absorbs a 1.86x state error.

Both arms must be read at the SAME `prefill_from`. Reading after one tick puts
the hit arm at 164 and the control at 128 and reports 66.5 vs 43.75; that is a
measurement error, not a defect. At the same position: `max|delta|` **0.000e+00**.

Negative control on `origin/main` in a detached worktree with `__pycache__`
cleared: `published 0`, the test fails at the first assert. `hits >= 1` is
asserted before the state comparison — without a hit, the comparison is two
identical fresh computations and proves nothing.

## The measurement that was thrown away

The first cold/warm script served the target prompt cold, then warmed the store
with the conversation head, then re-served the target — all in one process, and
reported **10.312x** (20.63 s → 2.00 s at 3265 tokens). That number is invalid.
The cold arm published 7 entries covering its *whole* prompt, so the warm arm
matched its own earlier self: it measures re-sending an identical prompt, not
multi-turn reuse. 10.3x on a workload whose shareable fraction is 33% is the
tell — a speedup above what the shared span can explain is a self-hit.

The script now takes `--arm cold|warm`, one arm per server restart, and asserts
`prefix_published == 0` at entry so a contaminated store fails loudly instead of
producing a plausible number.

## Rule

A publish gate that requires an aligned length publishes nothing for 15 of 16
real prompts, and the counter that would reveal it (`prefix_published`) looks
healthy because a different code path keeps it non-zero. Before adding
machinery to a cache that does not hit, measure the store layer alone: the same
6-turn chat scored 5/6 against `PrefixStore` directly and 0/6 through the
engine, which located the defect in one run.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-05 | b197b67 | V100 32GB | sm70 | Qwen3.8-27B NVFP4 | 4.35 (3265 tok, warm store) | — | 230.1 |
| 2026-09-05 | b197b67 | V100 32GB | sm70 | Qwen3.8-27B NVFP4 | 6.32 (3265 tok, cold store) | — | 158.2 |

Raw artifacts: `scripts/bench_chat_reuse.py`, `scripts/bench_chat_cold_warm.py`.
