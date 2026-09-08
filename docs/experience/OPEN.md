# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [the training rollout tick is 2.6x serving](errors/2026-09-08-the-training-rollout-tick-is-2.6x-serving.md) | `src/tilerl/train.py:442-445` rollout vs `build_engine` config | two full-27B B=8 decode measurements disagree **2.64x** with the slower one on the *shorter* context — 55.21 and 61.68 ms/tick training (two shas, two cards, agreeing to 1.12x) against 23.34 serving, all graph-on, all `wall / decode_forwards`. Worth **45.9% of a GRPO step** if it closed and **93% unattributed**: LoRA is ≤4% of the gap (0.537 GFLOP/tick is 0.054 ms even at 10 TFLOP/s; 768 launches inside a captured graph is 1.54 ms) and `fuse_projections` 2.9%. The reason on record was false — `_training_kv` is at `train.py:160` in the forward/backward, not the rollout, and the serving arm was itself W=1 no-draft. Arm: one process, one card, two engines, config the only variable — a reproduction there **localizes to the bundle, not a mechanism** |
| [a miss self-reinforces](errors/2026-09-07-a-miss-self-reinforces.md) | `src/tilerl/engine.py:1031` `_finish_prefills` | one 31k-token miss published 62 chunk entries into a 6-snapshot budget and evicted every other session's shared head, so 11 of 12 sessions missed in sequence at 14.1 s each. **Fixed at the publisher** — the first interior boundary plus the last, a constant 2 publishes at any prompt length ([wins/2026-09-08](wins/2026-09-08-cut-the-prefill-publish-flood.md)). The remaining open half is the **V100 alternation** the cell was run to check: turn-0 hits on every other conversation, which did not reproduce on the H20 and needs the V100 grid with the per-row instrument |
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/eval.py` `MATCHERS`, GRPO advantage | a length term in the reward or a length-aware advantage — `boxed_match` is correctness-only, so nothing prefers the shorter of two correct answers and the policy lengthens until it truncates |
| [entries-per-row against the snapshot budget](errors/2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md) | `src/tilerl/engine.py:1044` `_finish_prefills` publish gate | #271 traded 62 nested entries for 2 and costs the DRAM-tier cell **1.81–2.12x** on the parent-child pair `45acd87`→`a43a379` (16 lines). Measured: TTFT is **linear in the tokens a hit did not match**, R² 0.998, predicting the unfitted miss row to −6.8%; and depth is a **512×k ratchet in submission order**, so each session's match is set by its queue position, earliest worst at 1.7%. Sign flips with turn depth (+259 s at turns 0–1, −54 s at turn 2), so **both endpoints of the count axis lose**. The two sizing constraints **conflict**: `K × snapshot_bytes ≤ budget` gives K ≈ 1 here while the depth threshold admits K ≈ 4–11. Arm: same pair at a budget where K ≥ 2 fits, in TTFT — after the fixed term is pinned |
| [a hit carries 1.2–2.5 s of depth-independent cost](errors/2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md) | unattributed; `src/tilerl/engine.py` `_admit` hit path | the fixed term of the hit-cost model, paid at any match length. **A range, not a number**: the data's unmatched fraction bottoms out at 0.437, so the constant is extrapolated 44% past the nearest point — the three per-turn lines agree to 6.1% mid-range and to 75.9% at zero. Extra forwards explain 0.15–0.39 s (+22 to +58 forwards over the arithmetic minimum); the state restore has the right shape (a snapshot is constant size at any prefix length, `kv_cache.py:1434`) and is 0.17 ms at HBM bandwidth against ~1200–2500 ms. Fix: one run with turns held constant (one turn, more sessions) pins it without the covariate. It caps how closely a publisher ladder can space rungs, so the ladder cannot be priced until it is |
| [`_last_prefill_boundary` takes one argument where the boundary takes two](errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md) | `src/tilerl/engine.py:57` `_last_prefill_boundary`, `:1056` the `spill` decision, `:1398` `PrefixStore.insert` | the helper predicts where `_build_plan` ends the last prefill chunk, and `_finish_prefills` compares against it to pick the **one publish that reaches disk**. The walk depends on `budget = max_num_batched_tokens - len(decodes)`, which the helper never sees: n=961 gives 944 at budget 512 and **448** at 511, so one decode row sharing the tick makes `last` unreachable and the prompt spills nothing — silently, every counter normal, the request correct. This PR closes `budget == 512`; **2448 of 35756 prompts in the default 504–512 window still miss**. Four fixes measured and rejected: a budget-free predicate misses twice as often (12.8% vs 6.8%) or double-spills 41962 times; a budget-carrying replay is exact but **unsound** — the walk depends on the budget *history*, and the deepest boundary moves on 352–414 lengths across schedules; a deferred publish at DONE scores exact but **pairs the entry's tokens with the whole prompt's GDN state** (`:1394` clones the slot as-is, `:722` copies it back), which is silent wrong inference. **The fix that remains**: capture the snapshot at the boundary and spill it later, which needs `PrefixStore` to accept a spill for an entry it already holds — 15 unspillable prompts against today's 2448, schedule-independent. 30k prompts are unaffected, which is why no bench saw it |
| [the tier converts byte pressure into block pressure](errors/2026-09-08-a-missing-branch-that-cannot-satisfy-its-loop.md) | `src/tilerl/kv_cache.py:1330` `evict_until_free`, `_drop` → `_dram.forget` | a demote keeps the entry and therefore keeps its **blocks**, so enabling the tier took peak occupancy **45.6% → 99.1%** on the same pool and workload. The block path then evicts already-demoted entries and `_drop` calls `_dram.forget`, orphaning the host copy: **103 demotions, 1864 ms, 0 promotions**, 5.9 GiB of tier budget unused. Not a missing demote branch — a demote frees zero blocks, which is why that reading was struck below. Fix: retain fewer blocks on demote, or size the pool against the tier's retention |

## Struck, not fixed — claims that were measured and turned out not to be defects

A row deleted with no record invites the same reasoning back. These were on the list above
and came off because driving the code refuted them, not because a fix landed.

**`_demote_one` is unreachable under count pressure** — `src/tilerl/kv_cache.py:1225`,
struck 2026-09-08. The reading was that the `len(self._by_id) <= self.capacity` guard sits
inside a `while len > capacity or bytes > budget` loop, so the guard can never be true and
the store evicts where it could demote. That holds only on the branch where the loop is
entered on the **count** term; under byte-only pressure the count term is not over, the
guard is true, and demote runs. Driving the real `PrefixStore` with a `DramSnapshots` tier,
512-byte snapshots, three pressures:

| pressure | entries | demoted | evictions |
|---|---:|---:|---:|
| byte only — `capacity=64`, `state_bytes=1200`, 6 published | 6 | **4** | **0** |
| count only — `capacity=3`, `state_bytes=1 GiB`, 6 published | 3 | 0 | 3 |
| both — `capacity=4`, `state_bytes=1200`, 8 published | 4 | **2** | 4 |
| boundary — `capacity=9`, `state_bytes=2500`, 9 published | 9 | **6** | **0** |

The first row is the refutation: if the guard made the path unreachable, 4 demotions and 0
evictions is impossible. The guard is also not conservative — `_demote_one` leaves the entry
in `_by_id`, so a demote cannot shrink the count term and only eviction can satisfy it.
Removing the guard would demote every entry to the host tier under count pressure, find the
count unchanged, and evict them anyway: one wasted tier write plus a `forget` per entry.

The last row is `len == capacity` exactly, at different constants (800-byte snapshots,
16-token prefixes, a 4096-block pool so the block term cannot participate) — the boundary
where "the guard is always false" should hold if it holds anywhere, and it does not.

**The block axis is a third term, struck 2026-09-08 on the same argument.**
`evict_until_free` (`kv_cache.py:1330`) has no demote branch, which reads as the same
omission on another axis. It is not: `_demote_one` touches neither the block pool nor
`_by_id`, so a demote frees **zero** blocks and there is nothing the branch could do.
Confirmed: a forced `_demote_one` moved `dram.demotions` 3→4 and `free_blocks` 58→58. Why,
at length, in
[a missing branch that cannot satisfy its own loop](errors/2026-09-08-a-missing-branch-that-cannot-satisfy-its-loop.md).

The reasoning was self-consistent and no step in it was wrong. It described one branch while
claiming something about all of them, and nobody ran the store. Two sessions read the shape;
three rows answered it in minutes.
