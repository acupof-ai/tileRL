# The draft wrote a position the trunk had no block for — sm70/CPU, 2026-09-04

> Status: fixed. `xfail(strict)` on the `manychunks` e2e case is removed and a
> smaller case reproduces the real mechanism.

## Context

Extending `ab_draft_depth.py` along the batch axis (task #40 needs B=1/2/4 to
price the depth default on the kernel that actually ships) crashed on the first
B=2 submit:

    IndexError: index 1 is out of bounds for dimension 0 with size 1
      kv_cache.py:149  blk = kv.block_table[bi, pos // BLOCK_TOKENS]

This looked like task #58, filed as "draft KV write indexes past its block table
on multi-row chunked prefill" and already `xfail(strict)`. It is the same crash
and **not the same mechanism**.

## Root Cause

The engine grows `r.blocks` for `decodes` only (`engine.py:707`), inside the loop
that sizes the chain. `_draft.step(rows)` then runs on **every** row, including
one that just finished prefill this tick — and the draft writes position
`seq_len - 1`.

A 15-token prompt prefills to `seq_len = 16`, owns exactly one block (16 tokens),
and the draft immediately asks for position 15's successor at 16 → block index 1
of a 1-column table. Traced shapes on the tick that fails:

    write_tokens k=(2, 64, 2, 16)  bt=(2, 64)  seq_len=[16, 16]   <- trunk, fine
    draft.step rows=2 planned=[(17, 0, ...), (17, 0, ...)]
    write_tokens k=(2, 16, 2, 16)  bt=(2, 1)   seq_len=[17, 17]   <- raises

**Chunked prefill is incidental, and so is multi-row.** It reproduces with a
16-token prompt in a single block, no chunking, on the first tick after prefill.
The `manychunks` case hit it because 32-token prompts at `batched_tokens=8` land
a row on a block edge, not because they chunked.

## The Fix, and the Wrong Fix I Tried First

**Wrong:** clamp the draft's span to the blocks the row owns, mirroring the
hidden-span clamp two lines above. It stops the crash and the suite goes green
except for one case, which is the point — deferring a position leaves a **hole in
the draft's KV**, and the next position attends over it. Measured: the engine
drafted token 79 at position 22 where full context drafts 61. A crash traded for
silent divergence.

**Right:** allocate what the draft will write, at the layer that owns allocation.
`engine.py` grows blocks to cover `seq_len - 1` for every row in `rows` before
`_draft.step`, using the same `while` the decode path uses.

The `r.blocks and` guard keeps a row that has not prefilled yet (empty block
list) out of it; those rows are filtered by `step`'s own `hidden is None` check.

## Gate

`tests/test_e2e.py::...[blockedge]` — `(rows=2, plen=15, batched_tokens=512,
depth=1)`: a prompt ending one token short of a block boundary, batched, no
chunking. The existing test already compares the engine's draft against a
full-context draft, so it catches both the crash and the divergence the clamp
would have introduced.

Negative control **run**: with the growth loop disabled (`for r in []`), both
`blockedge` and `manychunks` fail. With it, 241 passed / 7 skipped, and the
`xfail` is gone rather than flipped to a pass.

## Rule

A crash and its filed description can disagree. #58 said "multi-row chunked
prefill" because that is the case where it was first seen; the mechanism needed
neither. Reduce to the smallest reproducer before believing the label — the
16-token single-block case named the cause in one trace, and the 32-token
chunked case had hidden it behind two irrelevant variables for a day.

Second: when a fix makes a crash go away, check what the code now *computes*, not
only that it runs. The clamp passed 47 of 48 cases; the one it failed was the
only one that compared output against a reference.
