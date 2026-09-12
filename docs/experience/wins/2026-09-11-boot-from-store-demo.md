# Boot-from-store demo: a fully-prefilled prompt is computed once — CPU tiny, 2026-09-11

> Status: **pending-remote** — the save → stop → cold-boot equivalence and the
> prefill/boot timing run on the CPU tiny model; the 256k 27B card run is cc's.

## Context

Unit C (`--kv-store`, #514) saves a fully prefilled context to disk and boots
a later request from it, so a 256k prompt's 2.25 h V100 prefill is paid once.
ckl asked for a runnable "只算一次" demonstration: send one long prompt,
`save_boot`, stop the engine, restart against the same store, send a
continuation request, prove the booted continuation equals the in-process one,
and print prefill seconds vs boot seconds. `scripts/boot_from_store.py` drives
the engine through the same `build_engine` path `tilerl serve --kv-store` uses.

## What worked

Two fresh engines in one process (writer shuts down, releasing HBM and the
prefix store; booter starts against the same store dir), greedy
(temperature=0) continuations compared token for token. The boot arm asserts
`boot_hits == 1`, `prefill_forwards == 1` (only the `<16`-token tail forwards)
and identical continuation. On tiny, 83-token prompt / 6 generated:

    prompt 83 tok, 13573 bytes stored
    prefill 0.226 s   boot 0.020 s   ~11x

The speedup is a mechanics number, not the 27B claim — 0.2 s includes fixed
per-engine build cost; the card run at 256k is the real measurement.

Building the demo found a real #514 gap: a **page-aligned** prompt (T%16==0,
which includes 256k = 262144) booted its whole context, left zero residual
tokens, no prefill chunk ever forwarded, and the row stuck in PREFILL with no
first-token logits. The fix and its red test landed on #514 as 998b0349 (5f):
the admit re-forwards the last loaded OWN page once — its K/V are overwritten
with the identical values and the tail position's logits appear — costing the
same one page as any `<16`-token tail
(`tests/test_kv_boot_store.py::test_a_page_aligned_prompt_boots_and_continues_identically`).

A length one token short of a page (T%16==15) stays refused by the demo:
`save_boot` at the first decode tick saves `floor((T+1)/16)` pages, which for
that length stores the first *generated* token in the last page, so the later
prompt-only prefix misses. The script says this instead of silently measuring
a miss.

`--sparse-k PAGES` is parsed but raises: page selection is not wired into
serving attention yet (the selector lives in #497), so a nonzero value must
not boot a dense engine silently.

## Rule

A bulk prefix load that covers the whole request must still produce one
forward's worth of logits: when the loaded length equals the prompt length,
re-forward the last page rather than scheduling a zero-token chunk. A
cold-start demo proves equivalence in one process with two fresh engines —
the boot arm's continuation must equal the writer's, token for token.

## Results

| date | commit | machine | target | model | result |
|---|---|---|---|---|---|
| 2026-09-11 | (PR head) | Mac CPU | cpu f32 | tiny | 4/4 boot-store gates; 83-tok prefill 0.226 s vs boot 0.020 s, identical greedy continuation; aligned prompt boots with 1 hit / 1 forward |

Raw artifacts: `scripts/boot_from_store.py`; `tests/test_kv_boot_store.py`.
The 256k 27B card run (`--model qwen38-27b --kv-store DIR --prompt-tokens
262144`) is cc's; no card number is claimed here.
