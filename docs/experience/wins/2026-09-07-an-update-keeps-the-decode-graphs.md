# An update keeps the decode graphs and refills the casts — H20 sm90, 2026-09-07

> Status: Shipped. **3.87 s off a 133.65 s GRPO step (2.9%)** under the shipped
> LoRA recipe, H20 card 6, and the refill verified by a full-parameter arm.

## Context

The RL loop recaptured every decode graph after every optimizer step.
`grpo_loop(recapture_graph=True)` calls `engine.invalidate_weights()` after each
update, and that call cleared `_decode_graphs`; the next rollout then paid a
capture on the first tick of every bucket it touched. Measured here at **~0.6 s
per warm recapture**, four per step — see below for why the 14 s in
`wins/2026-09-02-precapture-the-decode-graphs.md` is not the applicable figure.

This is a live cost, not a latent one. The shipped `--rl` path builds its engine
with `decode_graph=True` (`cli.py:538`) and passes `recapture_graph=True`
(`cli.py:639`), so every step of a real run pays it. Stated for GRPO only — the
OPD path at `cli.py:684` also passes the waiver, but its engine was not read.

Almost all of that work was already valid. An in-place optimizer step leaves
every address a capture baked exactly where it was: `AdamW.step_one` and
`Adafactor.step_one` both end `p.copy_()`, and `materialize` rebuilds the dict
but not the tensors.

## What Worked

**The premise was incomplete, and the gap is the whole entry.** `_const_f32`
caches a parameter's f32 cast and refills that buffer in place when called with
new values (#190) — but only when **called**, and a graph replay calls nothing.
Measured on cpu before any of this was written: the address survives `p.copy_()`
as designed and the values do not, so a kept graph replays the cast taken before
the step. #190 makes the refill possible; it does not make it happen. Keeping
the graphs on that basis alone would have been silently off-policy — the one
failure mode this path cannot afford, because "faster" and "wrong" are the same
measurement without a control.

So `invalidate_weights()` drives the refill itself.
`Backend.refill_const_f32()` walks the cache, re-casts every live entry into its
existing buffer, and returns the count. Entries whose parameter is gone are
dropped; an entry whose cast would change shape is dropped rather than refilled,
since the buffer a graph baked cannot be resized. The prefix store is cleared as
before — it holds KV, not addresses.

Only the ~27 call sites that reach a kernel *through* `_const_f32` were ever at
risk (oscale, wscale, the rmsnorm weight, the four gated-delta constants). A raw
bf16 weight a kernel reads directly was always fine: `p.copy_()` writes into the
tensor the graph baked.

`keep_graphs` was deleted rather than defaulted. No caller needs the old
behaviour, so there is no configuration in which dropping the graphs is right.

## Three quantities, not one

The saving and the cost of its replacement are separate measurements, and the
LoRA recipe can only produce the first:

1. **Captures removed** — N per step, from arm A vs arm B. This is the perf claim.
2. **The refill walk** — `invalidate_secs`, the cost of iterating the cache.
   Under LoRA it is 0.7 ms **finding nothing to do**: the refill count is 0.
3. **The first-call copy** — what a refill actually costs when a cached
   parameter has moved. Unmeasured, and **structurally unmeasurable under
   LoRA**: `rl_step` passes `trainable`, so the optimizer touches only
   `lora_a`/`lora_b`, and no adapter reaches a kernel through `_const_f32`.
   Every cached parameter's `_version` is unchanged, so the walk correctly
   refills nothing however long the run goes.

Quoting (2) as the price of removing (1) would compare a real cost against a
number measured on a path where the work does not happen. Only a
full-parameter update can produce (3), which is what the existence arm is for.

## The existence arm

`scripts/existence_arm.py`, H20 card 6, **tiny with a full-parameter AdamW step**
— not the 27B, whose full-parameter moments are 200.4 GiB. The question is a
mechanism, so the smallest model that captures a graph answers it.

```
cache_entries 11   casts_refilled 11   refilled_all_cached true
graphs_held 1      graphs_after 1      graph_matches_eager true
rollout_changed true   params_reallocated 0
```

All 11 cached casts refilled, the kept graph agrees with an eager engine on the
same post-update weights, and the rollout changed — so the graph replays the new
values, not the ones baked at capture.

Three things this arm had to get right to mean anything:

- **The count is derived, not hardcoded.** `refilled == cached_before` catches a
  partial refill that `> 0` would pass, and `cached_before == 0` fails as vacuous
  rather than passing as success.
- **The population had to be cacheable.** `_const_f32` returns early and caches
  *nothing* for a tensor already f32, on-device and unpadded
  (`backend.py:1416`), so an f32 model would report 0 refills for a reason
  unrelated to the refill path — the same null the LoRA arms produce. tiny's norm
  weights are bf16 (`model.py:622`), so `_rmsnorm` casts each through
  `_const_f32` and the update bumps `_version`. Caught by tilerl-48 before the
  run.
- **`params_reallocated: 0` is the result, not a failure.** It counts parameters
  whose *address* changed; zero is the in-place premise holding, which is what
  makes a baked address still valid. `rollout_changed` is what shows the values
  moved. Named `params_moved` at first, which read as "nothing updated".


## Rule

A cache that refills lazily does not refill for a consumer that never calls it.
Before keeping anything across an invalidation, ask which of its inputs are
refreshed by *being read* — a replay reads nothing.

## Gates

Four controls on the refill walk, each run separately:

| control | fails with |
|---|---|
| walk does not copy | `the cached cast still holds the pre-step values` |
| walk reallocates instead of copying | `the refill rebound the cache to a new buffer` |
| walk skips every entry | `the walk refilled nothing; every arm below is vacuous` |
| dead entries kept | `the entry outlived its parameter` |

Rows 1 and 2 first failed on the **same** assertion: the test held a local handle
to the buffer, so "never refilled" and "refilled into a new buffer" were
indistinguishable. It now asks the cache what it holds before checking values —
the arm that catches a reallocating refill, the case that breaks a captured
graph while every value assertion still passes.

`test_a_recapturing_engine_drops_what_the_update_invalidated` asserted the
contract this reverses (`_decode_graphs == {}` after a step). Renamed to
`..._clears_...`; it now asserts the graphs survive while the prefix is cleared.
Both per-cache refusals and the re-dirty-between-steps structure are unchanged.

The staleness question itself cannot be gated on cpu: capture calls
`torch.cuda.graph_pool_handle()`, which raises there, and the handler flips
`_decode_graph_on` to False — so both arms would compare eager to eager. That
half is the existence arm above.

## Results

Recipe: `qwen38-27b`, group 8, gen 1024, LoRA-16, micro 1, `--blocks 2304`,
4 steps, warm means of 3 excluding step 0. Both arms on H20 card 6 in one
session, `scripts/prof_grpo_step.py --invalidate`.

Arm A is `a7967cd` with `invalidate_weights` locally reverted to
`self._decode_graphs.clear()` — a one-line revert, not a checkout of the old
branch, so nothing else differs between arms.

**Why not compared to the 34.09 s in `AGENTS.md`:** that figure is
`wins/2026-09-05-recapture-after-update.md`, and it is well sourced — same card,
n=10 pooled over both arm orders, 73.12±0.90 / 74.12±2.07 → 34.09. It is not
comparable here because it is **`max_new_tokens=256`** against this recipe's
1024. The bench-baseline JSON row was removed in #193, but the measurement was
never in doubt; both arms are re-measured in one session because the recipe
differs, not because the old number is suspect.

| | step s | rollout | decode | prefill | mixed | backward | invalidate s | graphs dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — recapture | 133.65 | 65.24 | 61.76 | 0.26 | 3.20 | 68.30 | 0.0242 | 4 |
| B — graphs kept | 129.78 | 62.78 | 59.28 | 0.28 | 3.20 | 66.92 | 0.0006 | 0 |
| **A − B** | **3.87** | 2.46 | 2.48 | −0.02 | 0.00 | 1.38 | 0.024 | |

**3.87 s of a 133.65 s step: 2.9%.** The decomposition closes to 0.0008 s in
both arms, and `unexplained_ticks` is 0.

**2.48 s of the 3.87 is decode**, which is where a recapture lands: it happens
inside the first tick of its bucket after the invalidate, so it never appears as
its own bucket. **1.38 s is backward**, which recapture cannot touch — against
the 4.61 s span of six dense `backward_secs` readings from six trees
(71.53 / 68.30 / 68.26 / 67.77 / 67.31 / 66.92) that is noise, not a term.

<!-- SLOW-TICK ROW PENDING: arm A re-run with per-tick instrumentation -->

### Where the 2.48 s is, and what was not measured

A recapture happens inside one tick — the first of its bucket after the
invalidate — so it never appears as its own bucket. Arm A was re-run with a
per-tick instrument recording every tick over 1 s, and **it failed to resolve
the captures**:

| step | graphs dropped | slow ticks | slow tick s | decode s | step s |
|---|---:|---:|---:|---:|---:|
| 1 | 4 | 2 | 2.12 | 62.55 | 137.19 |
| 2 | 4 | **0** | **0** | 61.70 | 139.44 |

Step 2 drops and recaptures four graphs, carries the same decode excess, and has
no tick over 1 s at all — while being the longer step. So the threshold is set
above the quantity it was built to find. 2.48 s of decode delta across 4
recaptures implies **~0.6 s per warm recapture**, which is consistent with the
09-02 entry's 19 s / 8 graphs on another arch, and which this instrument cannot
confirm. Step 1's two ticks at 1.03 / 1.10 s are a capture plus whatever else
that tick was doing, not a clean per-capture reading.

**The 2 → 0 is not clean evidence either.** The re-run's steps climb
monotonically (137.19 → 139.44 → 140.43), the signature of a load still ramping
rather than of scatter, so its three steps were taken under three machine
states. The neighbours' job arriving between step 1 and step 2 is a second
candidate for the slow-tick count dropping to zero, and it happens to point at
the same conclusion — which is a reason to trust the conclusion less, not more.

**Not separately verified.** What would settle it is printing the `(B, W)` graph
key beside each slow tick and lowering the floor below 0.6 s — the buckets are
`8 → 4 → 2 → 1` as the batch drains, so the four captures are not equally
costly. The 1 s floor was chosen before the per-capture cost was known, which is
the same error as the 14 s projection below.

### The re-run's step times are excluded

Partway through the re-run the other team's job took cards 1-5 and 7 to 100%
util. Card 6 stayed at 59% with no throttle flags (`clocks_throttle_reasons`
`0x0`, 1980 MHz, 56 °C), so this is host or bandwidth contention, not thermal.
The re-run's mean step is **139.95 s against arm A's 133.65 s on identical
code — 4.7%**, and almost all of it lands in train (74.48 vs 68.38, +6.1 s)
while decode barely moves (61.94 vs 61.76, +0.18 s).

Arms A and B ran back to back before this, each internally consistent with no
drift (A: 137.14 / 133.97 / 133.37 / 133.60; B: 135.24 / 129.50 / 129.84 /
130.00), so the 3.87 s delta stands. The re-run's step times are not comparable
to either and are quoted only for the slow-tick counts.


### The 14 s figure does not apply here

`wins/2026-09-02-precapture-the-decode-graphs.md` measures 14.0 s and 11.7 s per
capture, and projecting that onto this arm predicted 4 × 14 = 56 s/step, i.e.
**43% of a step**. That prediction was relayed to two peers as a planning number
before anything was measured. The measurement is **3.87 s, 2.9%** — a 14x
overestimate, and the difference is not a refinement: "recapture costs 43% of an
RL step" and "recapture costs 3% of an RL step" support different decisions
about whether this work was worth doing.

That entry is **V100 sm70**, and its own ponytail line says `first token pays
JIT + capture`: it is first-capture-per-bucket with the TileLang JIT for that
shape, on another arch. The comparable number is in the same entry's header —
**208 s → 19 s for all 8 graphs**, i.e. ~2.4 s per capture once the JIT is not
being paid per bucket. Quoting 14 s took the cold figure and dropped the word
cold.

### Two labels that lied

`invalidate_secs` is 24 ms in arm A against 0.6 ms in arm B: freeing four
captures costs something, keeping them nearly nothing. Neither is the story.

The probe filed `invalidate_weights()`'s return value under `casts_refilled`.
Arm A's revert makes that method return **graphs dropped**, so arm A's summary
read `casts_refilled: 4.0` while the true refill count was 0 — the field is now
`invalidate_returned`, with a comment saying why. Read literally, the old label
said the LoRA path exercises the refill, which is the opposite of what this
recipe does.

Raw artifacts: `/work/kg_armA.json`, `/work/kg_armB.json`, `/work/kg_armA2.json`.

