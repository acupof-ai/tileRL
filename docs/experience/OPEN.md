# Open defects

Documented in `errors/` with a fix named and not landed. A PR that lands the fix removes
its line. Writing the entry and adding the line are one act — an entry with `Status: open`
and no line here is the same defect this file exists to stop. Reviewers check a change on the same path against this list.

| entry | path | fix |
|---|---|---|
| [the training rollout tick is 2.6x serving](errors/2026-09-08-the-training-rollout-tick-is-2.6x-serving.md) | `src/tilerl/train.py` `grpo_loop` rollout (`rollout_secs`) vs `build_engine` config | two full-27B B=8 decode measurements disagree **2.64x** with the slower one on the *shorter* context — 55.21 and 61.68 ms/tick training (two shas, two cards, agreeing to 1.12x) against 23.34 serving, all graph-on, all `wall / decode_forwards`. Worth **45.9% of a GRPO step** if it closed and **93% unattributed**: LoRA is ≤4% of the gap (0.537 GFLOP/tick is 0.054 ms even at 10 TFLOP/s; 768 launches inside a captured graph is 1.54 ms) and `fuse_projections` 2.9%. The reason on record was false — `_training_kv` is at `train.py:160` in the forward/backward, not the rollout, and the serving arm was itself W=1 no-draft. Arm: one process, one card, two engines, config the only variable — a reproduction there **localizes to the bundle, not a mechanism** |
| [a miss self-reinforces](errors/2026-09-07-a-miss-self-reinforces.md) | `src/tilerl/engine.py` `_finish_prefills` | **Publisher half fixed** — the first interior boundary plus the last, a constant 2 publishes at any prompt length ([wins/2026-09-08](wins/2026-09-08-cut-the-prefill-publish-flood.md)). **Open half: V100 alternation** the cell was run to check: turn-0 hits on every other conversation, which did not reproduce on the H20 and needs the V100 grid with the per-row instrument |
| [the rollouts grew into the cap](errors/2026-09-06-the-rollouts-grew-into-the-cap.md) | `src/tilerl/cli.py` `_length_aware`; **not** `eval.py` `MATCHERS` | the length term **landed on CPU** ([wins/2026-09-08](wins/2026-09-08-a-length-term-in-the-grpo-reward.md)): an all-right group now orders by length instead of tying at zero advantage, λ default 0.1. It is in the RL reward closure and NOT in `MATCHERS`, because that matcher's count is `manifest["metrics"]["gsm8k_*"]` — P1's exit criterion. **Open half:** that the policy stops lengthening is unproven and CPU cannot prove it. Needs one real run with `per_rollout`'s 8 pairs per step and `_within_group_r` showing a negative within-group correlation that shrinks over training; two group means per step cannot express it. Ceiling is **17 of run 2's 19 no-gradient steps** — 41 and 44 sat exactly at the cap, where any function of length is constant within the group, so only a bigger cap (fix 4) reaches them |
| [entries-per-row against the snapshot budget](errors/2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md) | `src/tilerl/engine.py` `_finish_prefills` publish gate | #271 traded 62 nested entries for 2 and costs the DRAM-tier cell **1.81–2.12x** on the parent-child pair `45acd87`→`a43a379` (16 lines). Measured: TTFT is **linear in the tokens a hit did not match**, R² 0.998, predicting the unfitted miss row to −6.8%; and depth is a **512×k ratchet in submission order**, so each session's match is set by its queue position, earliest worst at 1.7%. Sign flips with turn depth (+259 s at turns 0–1, −54 s at turn 2), so **both endpoints of the count axis lose**. The two sizing constraints **conflict**: `K × snapshot_bytes ≤ budget` gives K ≈ 1 here while the depth threshold admits K ≈ 4–11. Arm: same pair at a budget where K ≥ 2 fits, in TTFT — after the fixed term is pinned |
| [a hit carries 1.2–2.5 s of depth-independent cost](errors/2026-09-08-four-mechanisms-one-regression-and-a-copied-flag.md) | unattributed; `src/tilerl/engine.py` `_admit` hit path | the fixed term of the hit-cost model, paid at any match length. **A range, not a number**: the data's unmatched fraction bottoms out at 0.437, so the constant is extrapolated 44% past the nearest point — the three per-turn lines agree to 6.1% mid-range and to 75.9% at zero. Extra forwards explain 0.15–0.39 s (+22 to +58 forwards over the arithmetic minimum); the state restore has the right shape (a snapshot is constant size at any prefix length, `kv_cache.py:1434`) and is 0.17 ms at HBM bandwidth against ~1200–2500 ms. Fix: one run with turns held constant (one turn, more sessions) pins it without the covariate. It caps how closely a publisher ladder can space rungs, so the ladder cannot be priced until it is |
| [`_last_prefill_boundary` takes one argument where the boundary takes two](errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md) | `src/tilerl/engine.py` `_last_prefill_boundary`, `_publish_prefix` `spill=`; `kv_cache.py` `PrefixStore.insert` | the helper predicts where `_build_plan` ends the last prefill chunk, and `_finish_prefills` compares against it to pick the **one publish that reaches disk**. The walk depends on `budget = max_num_batched_tokens - len(decodes)`, which the helper never sees: n=961 gives 944 at budget 512 and **448** at 511, so one decode row sharing the tick makes `last` unreachable and the prompt spills nothing — silently, every counter normal, the request correct. This PR closes `budget == 512`; **2448 of 35756 prompts in the default 504–512 window still miss**. Four fixes measured and rejected: a budget-free predicate misses twice as often (12.8% vs 6.8%) or double-spills 41962 times; a budget-carrying replay is exact but **unsound** — the walk depends on the budget *history*, and the deepest boundary moves on 352–414 lengths across schedules; a deferred publish at DONE scores exact but **pairs the entry's tokens with the whole prompt's GDN state** (`:1394` clones the slot as-is, `:722` copies it back), which is silent wrong inference. **The fix that remains**: capture the snapshot at the boundary and spill it later, which needs `PrefixStore` to accept a spill for an entry it already holds — 15 unspillable prompts against today's 2448, schedule-independent. **Priced 2026-09-08 and it is half of the fix, not all of it**: 2448 is not 2448 recoverable, because the boundary entry has to survive to DONE. What fraction does is **`clamp(s − b, 0, b) / b`** for batch `b` and snapshot slots `s = state_bytes / snapshot_bytes` — linear, not a threshold, verified over batches 2..32 × every slot count 0..2b+2. It falls out of LRU order: `b` boundary snapshots are published, then `b` prompt-complete ones that are all newer, so the `s` survivors are the newest and only `s − b` of them are boundary entries. At the shipped V100 default (`b=8`, and `s=9` from `state_bytes = mem_get_info()[0] // 4`, `engine.py:1676`, priced at 9 snapshots in `kv_cache.py:377`) that is **0.125** — about 300 of the 2448. The other half is capacity: `s` is a config value, not a constant, and raising the quarter to a half gives `s=18` and a yield of 1.0. **That flip is unmeasured and card-bound** — `engine.py:1670` prices 8 GiB as most of the V100's post-weights headroom, so doubling the snapshot budget takes HBM from the block pool or prefill transients, and the snapshot is 144 MiB at 27B so `s` differs on every card. The DRAM tier moves a different quantity: it leaves this yield untouched but takes the fraction still *matchable* from 0% to 100%, since `_entries_capacity` (`kv_cache.py:1444`) adds the host budget to the same denominator and `lookup` promotes a demoted entry in place (`:1256`, 12.7 ms against 163 s to re-prefill). **Two yields, and the tier already delivers the one that serves turns.** What `spill_held` adds is only the disk half — and the disk is `KvTier`, under a recorded **REJECT on the serve path** (1.65x worse wall clock per turn and 0 hits at 12 sessions, [errors/2026-09-06](errors/2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md)), off by default (`--ssd-path` empty). So this row is **not 2448 prompts of open opportunity**: the boundary publish's second-turn purpose is met by the DRAM tier today, and its persistence purpose feeds a tier that was measured to lose. It reopens if the block-granular store that entry names as the upgrade lands, which does not inherit that verdict. **Same root as the row below**: neither the tier nor the pool is sized against what it retains. `scripts/probe_boundary_survival.py`. 30k prompts are unaffected, which is why no bench saw it |
| [the tier converts byte pressure into block pressure](errors/2026-09-08-a-missing-branch-that-cannot-satisfy-its-loop.md) | `src/tilerl/kv_cache.py` `evict_until_free`, `_drop` → `_dram.forget` | a demote keeps the entry and therefore keeps its **blocks**, so enabling the tier took peak occupancy **45.6% → 99.1%** on the same pool and workload. The block path then evicts already-demoted entries and `_drop` calls `_dram.forget`, orphaning the host copy: **103 demotions, 1864 ms, 0 promotions**, 5.9 GiB of tier budget unused. Not a missing demote branch — a demote frees zero blocks, which is why that reading was struck below. Fix: retain fewer blocks on demote, or size the pool against the tier's retention |
| [`--eval-curve-n` 20 cannot resolve the effect the curve looks for](errors/2026-09-08-a-default-that-cannot-resolve-its-own-effect.md) | `src/tilerl/cli.py` `--eval-curve-n` default; `ledger.py` `curve_point_se` computes the SE that condemns it | the curve exists to locate the step a target score is first reached. At n=20 the binomial SE is **11.18 pt against the +5.6 pt** effect on record — 2.0x the signal, so the crossing step it reports is chosen by which 20 rows fell where. `ledger.py:122-128` already computes this and `_se_note` warns above 5 pt, so the default sits below its own warning threshold. Genuinely two-sided: the flag's help asks for "under 5% of a step" and n=500 costs 1733% of one on this model, so 20 was picked against the cost constraint with the resolution constraint never computed. Fix: refuse at run time rather than warn at read time, which needs the target as a run-time input (`--time-to-score` already takes it at read time, `cli.py` `time_to_score`) |
| [prefetch deadline too thin on CPU: fetch drops under load](errors/2026-09-10-prefetch-deadline-gil-contention.md) | `src/tilerl/engine.py` `step` spin; `kv_cache.py` `KvTier.any_fetching` | **Fixed 2026-09-10** by the spin-until-ready loop (72d83303, [wins entry](wins/2026-09-10-spin-until-ready-fixes-cpu-prefetch-flake.md)): the e2e flake went 3/16 → 0/20. **Open remainder**: the spin's cost on 27B is unmeasured — the slice's 18 KB spill exits in one window, but the 27B snapshot is 144 MiB and needs many windows, so the spin actually runs and approaches the 50 ms bound per tick; and `any_fetching()` is global, not per-request, so under concurrency an unrelated fetch makes every tick spin. Both need a 27B SSD-on measurement |
| GDN state differs when prefix-hit and fresh-prefill use different chunk sizes | `src/tilerl/engine.py` `_build_plan` first-chunk cut; GDN parallel scan | When a prefix hit restores at a boundary the warm-up reached with chunk size A and the fresh prefill reaches with chunk size B ≠ A, the GDN states differ by ~3e-4 (2.5% relative on near-zero elements). Source: the parallel scan rounds per chunk length. **CPU/metal reference path: bounded.** `test_gdn_chunk_rounding_bound` (PR #445) measures the worst per-element violation ratio against the consumer's `allclose(rtol=1e-2, atol=1e-5)` over a 4×5 chunk×seq grid, three seeds: ≤1.20e-2 on CPU (bound 2.4e-2). **CUDA: still unbounded** — the recorded defect is H20 card 1 ratio ~2.0, 118× above the CPU bound, and CUDA has no regular runner (row below). This row truly closes only when the CUDA runner lands. Measured 2026-09-10, H20 card 1, see [errors/2026-09-10-gdn-state-chunk-size-rounding.md](errors/2026-09-10-gdn-state-chunk-size-rounding.md). The gate bounds the torch reference, not the sm90 kernel — same structure, different arithmetic |
| CUDA tests have no regular runner | `tests/` — GPU tests auto-skip in CI | CUDA tests only execute when someone manually runs the pod; no scheduled runner. **10 red found 2026-09-10, red for how long unknown.** CI auto-skips GPU tests, so the pod is the only place they run, and nobody watches it. This is the largest instance of "a gate that never runs" — bigger than any single defect above |

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

**`test_submit_rollback_and_terminal_failure` DID NOT RAISE on CUDA** —
`tests/test_e2e.py`, struck 2026-09-10. The reading was that a path which should fail did
not: the test patches `engine._model.forward` to raise, and the pod run saw
`DID NOT RAISE RuntimeError` where CPU raised — an engine error-path defect, CUDA-specific.
Driving the test with entry counters on card 1, tree `73e095cb`:

| | `_model.forward` entered | `_run_decode_graph` entered | `step()` | `take(1)` | blocks/slots |
|---|---:|---:|---|---|---:|
| CUDA | 0 | 1 | no raise | None | 1/1 |
| CPU | 1 | 0 | raise | RequestFailed | 0/0 |

The patched seam is entered 0 times on the card: a CUDA pure-decode tick replays a captured
graph through `_run_decode_graph` and never calls `_model.forward`, so the failure was never
injected. The engine's handler was never reached — no engine line changed; the fix landed in
the test, which now patches both seams. The red was loud by construction: `pytest.raises`
fails when the seam it patches stops being called, unlike an "output changed" assertion,
which goes green in the same situation — 9b's fp8 seam, the same root, was silent. Why, at
length, in
[a test patched a seam the CUDA tick never crosses](errors/2026-09-10-a-test-patched-a-seam-the-cuda-tick-never-crosses.md).

**`test_a_ragged_prompt_publishes_and_its_state_matches_no_store` failed on CUDA with
`max|delta| 3.586e-04, norms 43.6672 vs 43.6672`** — struck 2026-09-10. The reading was a
silent correctness issue on the GDN recursive-state restore path. It is not: the restore is
an exact `clone()` → `copy_()`, and both arms are individually bit-identical across reruns.
The difference is between the arms, not within them. The warm-up prompt (100 tokens)
produces a first prefill chunk of 64; the reference (164 tokens) produces 128. The GDN
kernel's parallel scan rounds differently per chunk length, so the state at the restore
point (token 64) is the **end** of a 64-token chunk in one arm and an **intermediate point**
in a 128-token chunk in the other. The deterministic ~3e-04 delta propagates to token 164.
Two near-zero elements fail `allclose(rtol=1e-2, atol=1e-5)`:

| element | ref | hit | delta | tolerance |
|---|---|---|---|---|
| (0,0,2,5) | -1.184e-02 | -1.154e-02 | 2.99e-04 | 1.28e-04 |
| (0,0,3,5) | -9.560e-03 | -9.773e-03 | 2.13e-04 | 1.06e-04 |

The max-delta element (ref=5.05, delta=3.59e-04) passes with tolerance 0.0505 — the failure
is entirely at near-zero elements where `rtol` gives a tight bound. Fix: match chunk sizes
(`short, long = 66, 100`), making the states bit-identical. The chunk-size-dependent
rounding itself is real and uncovered — see the new OPEN row above. Full analysis in
[errors/2026-09-10-gdn-state-chunk-size-rounding.md](errors/2026-09-10-gdn-state-chunk-size-rounding.md).
