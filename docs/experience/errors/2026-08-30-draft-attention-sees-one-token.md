# The draft head's attention sees one token in the loop, the whole prefix in the probe — 2026-08-30

> Status: found by reading, NOT yet fixed or measured. It reopens the
> speculation verdict in
> [wins/2026-08-29-spec-decode-net-win.md](../wins/2026-08-29-spec-decode-net-win.md),
> whose break-even is `p >= 66%` against a measured 55.8%.

## Context

Two acceptance numbers have sat side by side in the speculation entry since it
was written:

- **teacher-forced top-1 agreement 84.4%** (`scripts/draft_check.py`)
- **in-loop acceptance 55.8%** at depth 1 (`bench_harness --suite spec`)

At depth 1 there is no drift to explain a gap: the draft reads the trunk's own
last hidden, which is exactly the teacher-forced condition. The two should
agree. They differ by 28.6 points.

## Root Cause

The two runs give the draft head's attention layer different amounts of
context.

`draft_check.py` runs the head ONCE over every position:

```python
dk = kv(pools(draft.cfg.num_layers, ...), t - 1, t - 1)   # seq_len = seq_q = t-1
dl = draft.forward(hid[-1][:, : t - 1], arr([ids[1:]]), arr(range(1, t)), dk, backend)
```

Position i attends causally over positions 1..i. **Full context.**

`Engine._draft_chains` (`engine.py:906-912`) runs it one position at a time
against a chain-local pool:

```python
kv = BatchKv(block_table=bt, seq_len=ones * (j + 1), ...,
             kv_pool=self._draft_kv, seq_q_lens=ones)
```

At `j = 0` that is `seq_len = 1`. The draft attends over **exactly one token,
itself**. A softmax over a single logit is 1.0, so the attention output is
`v(self)` — the layer's attention parameters contribute nothing. The head runs
as an MLP with a residual.

`_draft_kv` appears twice in the whole file — allocated at `engine.py:386`,
used at `engine.py:910`. It is sized one block per row ("one block per row
holds a chain"), is never filled with the prompt, and is never filled with
accepted tokens. There is no path by which the draft could see context.

## Why it was not caught

Every gate the feature has is a *correctness* gate, and chain-local KV is
correct — the verify accepts a draft only when it equals what the trunk
sampled, so a context-starved draft produces a worse guess, never a wrong
output. `test_speculation_reproduces_greedy_decode` passes either way. The
quality signal lived only in the two numbers above, in different files, and the
entry that held both treated 55.8% as the real one.

## What it is worth, and what is not yet known

The break-even in the speculation entry is `1 + p >= 17.9 / 10.76`, i.e.
**`p >= 66%`**. The probe's 84.4% clears it; the loop's 55.8% does not.

That is NOT a claim that fixing this wins. Unknown, and only a GPU run settles
each one:

- Whether in-loop acceptance with full context actually reaches the probe's
  84.4%. The probe is teacher-forced on a real text; the loop drafts from the
  model's own output.
- What the fill pass costs. Giving the draft real context needs its KV
  populated over the prompt (one 1-layer forward per prefill chunk) and over
  accepted tokens each tick (a width-`n_ok+1` draft forward instead of
  width-1).
- Whether the wider draft forward eats the gain. The draft step measured
  2.06 ms against a 10.76 ms plain tick, so there is room, but not unlimited.

## The change it implies

The invariant is: **the draft must hold KV at every position it will later
attend over, and no gaps.** Everything below follows from it.

1. `_draft_kv` sized by the trunk's block count, indexed by the request's own
   `r.blocks`, instead of one block per row. Costs one full-attn layer's worth
   of the trunk's KV — 1/16 of it on the 27B.
2. `_draft_chains` passing the request's absolute position and its block table,
   not `j + 1` and a row-indexed one.
3. **Carry the whole forward's hidden, not its last position.** `r.hidden` is
   `hid[-1][i, seq_q-1:seq_q]` in all four places it is set. Keeping the full
   `[1, seq_q, H]` slice makes the fill and the draft ONE forward: the draft
   runs at positions `[draft_pos+1 .. seq_len-1]`, and the last of those IS the
   draft step. No separate fill pass — that is what the first version of this
   entry got wrong.

Three details found while scoping, each a place to be silently wrong:

- **Chunked prefill breaks the no-gap invariant.** Drafting happens only on
  pure-decode ticks (`chains` is None whenever `prefills` is non-empty), so a
  prompt materialized over N chunks leaves N-1 chunks' worth of positions with
  no draft KV, and only the last chunk's hiddens survive in `r.hidden`. The fill
  therefore has to run on EVERY tick that materializes tokens, including mixed
  and prefill ticks — a 1-layer forward against the trunk's 64, ~1.6% there.
- **`draft_pos` must track committed positions, not drafted ones.** A rejected
  chain leaves draft KV at `p+1..p+depth` for tokens that were never committed;
  advancing the watermark only to `p` makes the next tick overwrite them.
- **Position 0 is never drafted** (the draft at position q consumes hidden
  `H[q-1]`), so its KV page is whatever the previous owner of that block left.
  `draft_check.py` allocates a fresh zeroed pool and therefore reads zeros —
  the engine must zero it too, or the two will not agree even when everything
  else is right.

## Why it is not implemented yet

`test_engine_draft_matches_full_context_draft` covers **one row, unchunked
prompt, depth 1**. The cases it does not cover — multi-row ragged widths,
chunked prefill, depth > 1, a rejected chain's stale KV — are exactly where
the details above bite, and their failure mode is the one this whole entry is
about: **a wrong draft is still a correct output**, so nothing goes red. The
gate has to be extended to those four before the implementation can be trusted,
or the fix ships with the same blind spot that hid the bug.

## Rule

When one feature has two quality numbers from two harnesses, reconcile them
before trusting either. The gap here was not noise and not bookkeeping: the two
scripts were running different models of the same feature, and the cheaper one
to write was the one that disabled a whole layer.

Corollary: a correctness gate cannot see a quality regression. Speculation is
built so that a bad draft costs throughput and never output, which is good
design and is exactly why nothing failed for a full day.
