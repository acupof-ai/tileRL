# A trunk-output parity gate cannot see a contaminated draft — V100/CPU d1 W2048, 2026-09-25

> Status: rejected (PR #835 closed, not merged). The optimization was built,
> CPU-gated, and device-measured; review disproved its correctness claim and
> the cost/benefit did not survive. No code change remains.

## Context

Each chunked-prefill tick ran one draft-head forward for the prefilling row.
On V100 perf1 measured 6.5-10 s stalls on ticks that mixed a prefilling row
with decoding rows, and the forward's only product for an interior chunk is a
back-fill of the draft's OWN dense KV pool. The idea (#835): skip an interior
chunk when its draft K/V precedes the trailing W=2048 decode read window. The
cutoff was page-aligned (`s <= floor((n-W)/16)*16`), the finishing chunk was
always drafted, and W=0/full-prefix skipped nothing.

A device A/B on a single fixed 5200 prompt, run in both orders, showed
identical greedy output and identical accept (63/64 solo, 110/128 in a
prefill+decode mix). It looked structurally safe.

## Root Cause

Two separate errors, one in the gate and one in the attribution.

1. **The CPU parity gate compared the wrong output.** It asserted OLD-vs-NEW
   trunk tokens under temp=0. The trunk verifies every token itself, so a
   draft forward can only change WHICH candidates get accepted, never the
   sampled sequence — that gate passes whether or not the draft KV is garbage.
   It was vacuous for the thing being changed. The discriminator that bites is
   the draft's own first proposal, `r.drafts`, on the tick prefill ends.

   With that probe, CPU tiny d1 shows the first DECODE-tick proposal
   `r.drafts` DIFFERING OLD vs NEW on three of the four geometries tried (my
   independent gate, OLD first draft token → NEW):

   ```
   n=160 W=64  ch16: 206 -> 206  SAME
   n=320 W=64  ch16:  12 -> 37   DIFFER
   n=320 W=128 ch32:  12 -> 90   DIFFER
   n=640 W=64  ch16: 251 -> 13   DIFFER
   ```

   rev-ec's independent gate likewise found most geometries differing with no
   fixed direction (its exact NEW values differ — the token is probe/build
   sensitive; whether it differs is the stable signal). The single-prompt
   device identity was an empirical coincidence of one prompt, not a
   structural zero.

   Why the kept chunks are not clean: a KEPT interior chunk is still in the
   PREFILL phase, and `DraftHead._windowed_read_kv` bails to the FULL prefix
   for any prefilling row. Its self-attention therefore reads the pages of the
   earlier, SKIPPED chunks — which the draft pool never back-filled and which
   in sparse mode hold another pool user's residual K/V. The contaminated read
   changes the K/V that the kept chunk writes INTO the decode window, so the
   window the cutoff thought it left complete is itself poisoned. A
   structurally correct skip would have to still write back the skipped pages,
   removing most of the saving.

2. **The 6.5-10 s was first-use TileLang compilation, not steady cost.** Warm,
   the same-shape prefill draft forwards are 100-270 ms. The seconds only
   appear on a cold server the first time a (batch, width) kernel shape is
   touched. A one-order cold comparison read 135.2 s vs 18.3 s, but that pair
   is order-confounded — the compile lands on whichever arm first compiles the
   shape — and cannot be quoted as the fix's benefit. The defensible warm
   saving was only ~0.36-0.53 s per cold 5200-token request, with no
   decode-throughput change.

## Fix

None — the PR was closed. Making the skip structurally correct requires
back-filling the skipped pages anyway, so the residual saving does not justify
a prompt-dependent acceptance-rate risk. Reject verdict recorded in CHANGELOG.

## Rule

- A gate on an optimization that changes a side computation must assert THAT
  computation's output, not the downstream result that recomputes it. Draft
  KV/proposals are checked via `r.drafts` (or acceptance), never via trunk
  tokens — verify makes the latter invariant by construction.
- "Identical on one prompt" is one sample, not structural equivalence. A
  change that perturbs attention inputs needs either a proof the read cannot
  reach the perturbed data or a population of prompts; list the population if
  you only measured some.
- A seconds-level cost attributed to a hot-path op needs a warm rerun before
  it is believed. First-use JIT/compile lands on the arm that first touches a
  shape, so cold comparisons must be run in both orders and the warm numbers
  reported.
