# "Chunkwise-WY is incompatible with decay-first" — wrong, and it cost a 1.36x prefill win

## Context

`docs/experience/wins/2026-08-24-gdn-prefill-chunk.md` shipped
`make_gdn_chunk_fused`: one block per (value head, batch), a SERIAL scan over
all T tokens carrying the state in HBM. It justified not porting the tilelang
chunkwise-WY prefill path with:

> Not fla's chunk delta rule (that freezes chunk-start state — incompatible
> with decay-first).

That closed the question. Four days later the kernel is 28% of prefill GPU time
(24 launches x 1773.6 us on an 8-layer profile) at roughly 1% of both rooflines,
because a scan over T=2048 is 2048 serial steps no matter how wide the block.

## Root Cause

The claim reads the `chunk_delta_h` kernel in isolation. It does freeze the
state at chunk start — but the intra-chunk delta interactions are not dropped,
they are carried by its `W`/`U` inputs, which `wy_fast` builds from the
inverted triangular UT matrix `A`
(`tilelang/examples/gdn/example_wy_fast.py:113-127`):

    U = A @ (V * beta)
    W = A @ (K * beta * exp(g))

and the per-token decay is folded in as `V_new *= exp(g_last - g_i)` with
`S *= exp(g_last)` (`example_chunk_delta_h.py:198-211`) — decay-first, at chunk
granularity. The chunked form is the same recurrence reassociated, not a
different rule.

## Fix

Port the five-kernel pipeline (cumsum -> scaled_dot_kkt -> wy_fast ->
chunk_delta_h -> chunk_o, ~1179 lines upstream) and gate it on parity against
`reference.gdn_forward`, which is the only thing that could ever have settled
this.

## Rule

An algorithm ruled out on an argument, with no parity run behind it, is not
ruled out — it is untested. Write the objection down as a question ("does the
chunked form drop the intra-chunk term?") and the answer is one file away;
write it down as a conclusion and it survives unexamined for as long as nobody
re-reads it.
